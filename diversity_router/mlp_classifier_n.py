#!/usr/bin/env python3
"""
Trains one MLP classifier per candidate model (n classifiers total) to predict
which model produces the best results for each query. Each classifier outputs
a score for whether its corresponding model is the best. 
Usage:

# 18 models, unv
    python diversity_router/mlp_classifier_n.py \
        --data wildchat \
        --strategy list_all \
        --n_epochs 10 \
        --hidden_dim 512 \
        --soft_labels \
        --data_dir outputs_18_models \
        --output_dir outputs_18_models


"""
import datetime
import json
import argparse
import numpy as np
from pathlib import Path
import sys
from typing import Tuple, Dict, List
from tqdm import tqdm

## MLP
import torch
import torch.nn.functional as F
import torch.nn as nn

# train_test_split for splitting data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

#overall best baseline
from best_overall_baseline import calc_best_overall_baseline_split

from utils import *

#use logging for documenting cumsum and accuracy for different hyperparameters
import logging

#calc distribution divergence
from scipy.stats import entropy

#fix torch random seed for reproducibility
torch.manual_seed(2025)


class MLPClassifier(nn.Module):
    """Single-output MLP classifier for binary probability prediction."""

    def __init__(self, input_dim: int = 3584, hidden_dim: int = 1024, use_logits: bool = False):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Single score output
        self.relu = nn.ReLU()
        self.use_logits = use_logits

        # Initialize weights
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='sigmoid')

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if not self.use_logits:
            x = torch.sigmoid(x)
        return x


def run_mlp_classification_split(embeddings: np.ndarray, label_indices: np.ndarray, label_encoder: LabelEncoder,
                           lr: float = 0.001, n_epochs: int = 50, batch_size: int = 32, hidden_dim: int = 1024,
                           use_soft_labels: bool = False, soft_labels: np.ndarray = None,
                           per_model_features: Dict[str, np.ndarray] = None,
                           output_dir: Path = None, model_cumsum_dict: Dict[str, np.ndarray] = None, weight_decay: float = 0,
                           use_class_weight: bool = False,
                           test_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, float, List[int], List[int], Dict, np.ndarray, Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Run train/val/test split with multi-classifier (n classifiers) approach using 80/10/10 split.

    For the training phase, trains one MLP classifier per candidate model. Each classifier
    predicts a probability (0-1) indicating whether its model is the best for the query.

    If test_mask is provided, use it to determine test indices instead of random split.
    This is used for wild_inf_chat to ensure the test set matches wildchat's test split.
    """

    indices = np.arange(len(embeddings))

    if test_mask is not None:
        # Use provided test mask (for wild_inf_chat to match wildchat test split)
        test_idx = np.where(test_mask)[0]
        train_val_idx = np.where(~test_mask)[0]
        print(f"Using provided test mask: {len(test_idx)} test samples, {len(train_val_idx)} train+val samples")
    else:
        # First split: separate test set (20%)
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=label_indices,
            random_state=2025
        )

    # Second split: separate validation set (10% of original, ~12.5% of train_val)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=1/8,
        stratify=label_indices[train_val_idx],
        random_state=2025
    )

    # Split base embeddings (shared across all models)
    X_train_base, X_val_base, X_test_base = embeddings[train_idx], embeddings[val_idx], embeddings[test_idx]
    y_train, y_val, y_test = label_indices[train_idx], label_indices[val_idx], label_indices[test_idx]

    # Split soft labels if using them
    if use_soft_labels:
        assert soft_labels is not None, "Soft labels must be provided when use_soft_labels is True."
        soft_labels_train = soft_labels[train_idx]
        soft_labels_val = soft_labels[val_idx]
        soft_labels_test = soft_labels[test_idx]
        print(f"Using soft labels for training.")

    print(f"Train size: {len(X_train_base)}, Val size: {len(X_val_base)}, Test size: {len(X_test_base)}")

    # Prepare per-model training data
    # Each model gets base embeddings + its own features
    per_model_train_data = {}
    per_model_val_data = {}
    per_model_test_data = {}

    for model_name in label_encoder.classes_:
        # Start with base embeddings
        train_features = X_train_base
        val_features = X_val_base
        test_features = X_test_base

        # Concatenate per-model features if provided
        if per_model_features is not None and model_name in per_model_features:
            model_features = per_model_features[model_name]
            
            # Split model-specific features
            model_features_train = model_features[train_idx]
            model_features_val = model_features[val_idx]
            model_features_test = model_features[test_idx]

            # Concatenate to base embeddings
            train_features = np.concatenate([X_train_base, model_features_train], axis=1)
            val_features = np.concatenate([X_val_base, model_features_val], axis=1)
            test_features = np.concatenate([X_test_base, model_features_test], axis=1)

            print(f"  {model_name}: input_dim = {train_features.shape[1]} "
                  f"(base={X_train_base.shape[1]} + features={model_features_train.shape[1]})")

        # Convert to PyTorch tensors
        per_model_train_data[model_name] = torch.tensor(train_features, dtype=torch.float32)
        per_model_val_data[model_name] = torch.tensor(val_features, dtype=torch.float32)
        per_model_test_data[model_name] = torch.tensor(test_features, dtype=torch.float32)

    # Train one model per candidate
    models = {}
    optimizers = {}
    criteria = {}  # Per-model criteria (may have different class weights)

    # Initialize models, optimizers, and criteria for each candidate
    # All models have the same input dimension
    for model_idx, model_name in enumerate(label_encoder.classes_):
        input_dim = per_model_train_data[model_name].shape[1]

        models[model_name] = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            use_logits=use_class_weight
        )
        optimizers[model_name] = torch.optim.Adam(
            models[model_name].parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Create loss function with optional class weights
        if use_class_weight:
            n_pos = (y_train == model_idx).sum()
            n_neg = len(y_train) - n_pos
            pos_weight = torch.tensor([n_neg / max(n_pos, 1)])  # avoid div by 0
            criteria[model_name] = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"  {model_name}: pos_weight={pos_weight.item():.2f} (n_pos={n_pos}, n_neg={n_neg})")
        else:
            criteria[model_name] = nn.BCELoss()

    # Log model architecture, criterion, optimizer
    logging.info(f"Training {len(label_encoder.classes_)} separate classifiers (one per model)...")
    logging.info(f"MLP Classifier Architecture: {models[label_encoder.classes_[0]]}")
    logging.info(f"Loss Function: {criteria[label_encoder.classes_[0]]} (use_class_weight={use_class_weight})")
    logging.info(f"Optimizer: {optimizers[label_encoder.classes_[0]]}")

    # Initialize loss tracking
    train_losses = {model_name: [] for model_name in label_encoder.classes_}
    val_losses = {model_name: [] for model_name in label_encoder.classes_}

    # Training loop with validation monitoring
    best_val_acc = 0.0

    for epoch in tqdm(range(n_epochs), desc="Training"):
        print(f"Epoch {epoch+1}/{n_epochs}")

        # Train each model separately
        for model_idx, model_name in enumerate(label_encoder.classes_):
            model = models[model_name]
            optimizer = optimizers[model_name]

            # Get model-specific training data
            X_train_model = per_model_train_data[model_name]

            # Create target labels for this model
            if use_soft_labels:
                y_train_target = soft_labels_train[:, model_idx]  # Extract soft probability for this model
            else:
                y_train_target = (y_train == model_idx).astype(float)  # Hard binary {0, 1}

            y_train_target_tensor = torch.tensor(y_train_target, dtype=torch.float32).unsqueeze(1)

            model.train()
            permutation = torch.randperm(X_train_model.size()[0])

            epoch_losses = []
            for i in range(0, X_train_model.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_x = X_train_model[indices]
                batch_y = y_train_target_tensor[indices]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criteria[model_name](outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            train_losses[model_name].append(avg_loss)
            if epoch % 10 == 0 or epoch == n_epochs - 1:  # Print every 10 epochs to reduce clutter
                print(f"  Model {model_name}: Loss = {avg_loss:.4f}")

        # Validation: Get scores from all models and compute validation loss
        scores_per_sample_val = np.zeros((len(X_val_base), len(label_encoder.classes_)))

        for model_idx, model_name in enumerate(label_encoder.classes_):
            model = models[model_name]
            model.eval()

            # Get model-specific validation data
            X_val_model = per_model_val_data[model_name]

            # Prepare validation target labels for this model
            if use_soft_labels:
                y_val_target = soft_labels_val[:, model_idx]
            else:
                y_val_target = (y_val == model_idx).astype(float)
            y_val_target_tensor = torch.tensor(y_val_target, dtype=torch.float32).unsqueeze(1)

            with torch.no_grad():
                outputs = model(X_val_model)

                # Compute validation loss for this model (before sigmoid for class_weight mode)
                val_loss_model = criteria[model_name](outputs, y_val_target_tensor).item()
                val_losses[model_name].append(val_loss_model)

                # Apply sigmoid for scoring when using class weights (model outputs logits)
                if use_class_weight:
                    outputs = torch.sigmoid(outputs)
                scores = outputs.squeeze().numpy()
                scores_per_sample_val[:, model_idx] = scores

        # For each sample, predict the model with highest score
        val_pred = np.argmax(scores_per_sample_val, axis=1)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"Validation Accuracy = {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.3f}")

    # Plot loss curves
    if output_dir is not None:
        plot_loss_curves(train_losses, val_losses, output_dir)

    # Final evaluation on test set
    scores_per_sample_test = np.zeros((len(X_test_base), len(label_encoder.classes_)))

    for model_idx, model_name in enumerate(label_encoder.classes_):
        model = models[model_name]
        model.eval()

        # Get model-specific test data
        X_test_model = per_model_test_data[model_name]

        with torch.no_grad():
            outputs = model(X_test_model)
            # Apply sigmoid for scoring when using class weights (model outputs logits)
            if use_class_weight:
                outputs = torch.sigmoid(outputs)
            scores = outputs.squeeze().numpy()
            scores_per_sample_test[:, model_idx] = scores

    # For each sample, predict the model with highest score
    y_pred = np.argmax(scores_per_sample_test, axis=1)
    test_acc = accuracy_score(y_test, y_pred)

    if use_soft_labels:
        # Calculate calibration metrics if soft labels are used
        # Predicted scores already go through sigmoid, so they're in [0,1]
        # Normalize to probability distribution
        pred_probs = scores_per_sample_test / (scores_per_sample_test.sum(axis=1, keepdims=True) + 1e-10)

        # Calculate KL divergence: KL(true_soft || pred_probs)
        kl_divergence = entropy(soft_labels_test.T, pred_probs.T)
        print(f"Test KL Divergence = {np.average(kl_divergence):.4f}")

    print(f"Test Accuracy = {test_acc:.3f}")

    return np.array(y_test), y_pred, test_acc, train_idx.tolist(), test_idx.tolist(), models, scores_per_sample_test, train_losses, val_losses


def load_per_model_features(
    base_dir: Path,
    dataset: str,
    strategy: str,
    model_name: str,
    use_quality: bool,
    use_perplexity: bool,
    use_enc_ans: bool,
    n_samples: int
) -> Tuple[np.ndarray, int]:
    """
    Load per-model features based on enabled flags.

    Args:
        base_dir: Base directory for the project
        dataset: Dataset name (e.g., 'longform_qa', 'wildchat')
        strategy: Strategy name (e.g., 'list_all')
        model_name: Model name (e.g., 'llama-3.2-1b')
        use_quality: Whether to load quality features
        use_perplexity: Whether to load perplexity features
        use_enc_ans: Whether to load encoded answer features
        n_samples: Expected number of samples for validation

    Returns:
        features: Combined features array of shape (n_samples, feature_dim)
        feature_dim: Total dimension of loaded features

    Raises:
        FileNotFoundError: If feature directory or files don't exist
        ValueError: If feature array shapes don't match n_samples
    """
    features_dir = base_dir / 'outputs' / dataset / strategy / 'per_model_features' / model_name

    if not features_dir.exists():
        raise FileNotFoundError(
            f"Per-model features directory not found: {features_dir}\n"
            f"Please generate features first using the feature construction scripts."
        )

    feature_arrays = []
    feature_names = []

    if use_quality:
        quality_path = features_dir / 'quality.npy'
        if not quality_path.exists():
            raise FileNotFoundError(f"Quality features not found: {quality_path}")
        quality = np.load(quality_path, allow_pickle=True).astype(np.float32)
        if quality.ndim == 1:
            quality = quality.reshape(-1, 1)
        if quality.shape[0] != n_samples:
            raise ValueError(
                f"Quality feature size mismatch for {model_name}: "
                f"expected {n_samples}, got {quality.shape[0]}"
            )
        feature_arrays.append(quality)
        feature_names.append('quality')

    if use_perplexity:
        perplexity_path = features_dir / 'avg_logprobs.npy'
        if not perplexity_path.exists():
            raise FileNotFoundError(f"Perplexity features not found: {perplexity_path}")
        perplexity = np.load(perplexity_path, allow_pickle=True).astype(np.float32)
        if perplexity.ndim == 1:
            perplexity = perplexity.reshape(-1, 1)
        if perplexity.shape[0] != n_samples:
            raise ValueError(
                f"Perplexity feature size mismatch for {model_name}: "
                f"expected {n_samples}, got {perplexity.shape[0]}"
            )
        feature_arrays.append(perplexity)
        feature_names.append('perplexity')

    if use_enc_ans:
        enc_ans_path = features_dir / 'enc_ans.npy'
        if not enc_ans_path.exists():
            raise FileNotFoundError(f"Encoded answer features not found: {enc_ans_path}")
        enc_ans = np.load(enc_ans_path, allow_pickle=True).astype(np.float32)
        if enc_ans.shape[0] != n_samples:
            raise ValueError(
                f"Encoded answer feature size mismatch for {model_name}: "
                f"expected {n_samples}, got {enc_ans.shape[0]}"
            )
        feature_arrays.append(enc_ans)
        feature_names.append('enc_ans')

    if not feature_arrays:
        return np.zeros((n_samples, 0)), 0

    combined_features = np.concatenate(feature_arrays, axis=1)
    feature_dim = combined_features.shape[1]

    print(f"  Loaded {model_name} features: {', '.join(feature_names)} (dim={feature_dim})")

    return combined_features, feature_dim


def main():
    parser = argparse.ArgumentParser(description="MLP Multi-Classifier with Train/Val/Test Split")
    parser.add_argument('--data_dir', default='outputs', choices=['outputs', 'outputs_18_models'], 
                        help='strategy dir where train/test data is stored')
    parser.add_argument('--output_dir', default='outputs_split', choices=['outputs_split', 'outputs_18_models'],
                        help='output directory for results')
    parser.add_argument('--data', default='longform_qa', help='Dataset to analyze')
    parser.add_argument('--strategy', default='list_all', choices=['list_all'],
                       help='Strategy to analyze')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for MLP')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden layer dimension')
    parser.add_argument('--no_cache', action='store_true',
                       help='Regenerate embeddings instead of using cache')
    parser.add_argument('--soft_labels', action='store_true',
                       help='Use soft labels instead of hard labels')
    parser.add_argument('--quality', action='store_true',
                       help='Include per-model quality features (1-dim)')
    parser.add_argument('--perplexity', action='store_true',
                       help='Include per-model perplexity features (1-dim)')
    parser.add_argument('--enc_ans', action='store_true',
                       help='Include per-model encoded answer features (3584-dim)')
    parser.add_argument('--subset', action='store_true',
                       help='Use a subset of data for wildchat')
    parser.add_argument('--normalize_func', type=str, default='default',
                       help='Normalization function for soft labels (if any)')
    parser.add_argument('--exp', type=str, default='',
                       help='Experiment name for logging')
    parser.add_argument('--save_model', action='store_true',
                       help='Whether to save the trained models')
    parser.add_argument('--weight_decay', type=float, default=0,
                       help='Weight decay (L2 regularization) for optimizer')
    parser.add_argument('--class_weight', action='store_true',
                       help='Use class weights to handle imbalanced classes')
    args = parser.parse_args()

    #get current timestamp
    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup paths
    base_dir = Path(__file__).parent.parent
    strategy_dir = base_dir / args.data_dir / args.data / args.strategy # path w/ train/test data
    output_parent_dir = base_dir / args.output_dir / args.data / args.strategy
    data_path = strategy_dir / 'best_model_per_query.csv'

    output_dir = output_parent_dir / 'mlp_n'
    cache_path = strategy_dir / 'query_embeddings.npy'  # Cache at strategy level

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create logs directory
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)

    # Logging file name
    if not args.soft_labels:
        log_file_name = f'{args.exp}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}_hd_{args.hidden_dim}'
    else:
        log_file_name = f'{args.exp}_soft_labels_{args.normalize_func}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}_hd_{args.hidden_dim}'

    # Add feature flags to filename
    feature_flags = []
    if args.quality:
        feature_flags.append('q')
    if args.perplexity:
        feature_flags.append('p')
    if args.enc_ans:
        feature_flags.append('e')

    if feature_flags:
        log_file_name += '_features_' + '_'.join(feature_flags)

    # Add weight decay to log file name if non-zero
    if args.weight_decay > 0:
        log_file_name += f'_wd_{args.weight_decay}'

    # Add class weight flag to log file name if enabled
    if args.class_weight:
        log_file_name += '_cw'

    # Add current time to log file name
    log_file_name += f"_{cur_time}.log"
    log_file = output_dir / 'logs' / log_file_name

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(sys.stdout)
                    ])

    # Verify data file exists
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    if args.subset:
        print("Using 500 subset of data for wildchat...")
        log_file = log_file.stem + '_subset_500.log'
        cache_path = Path(cache_path.stem + '_subset_500.npy')

    # Load data and dataframe (needed for baseline calculation)
    import pandas as pd
    df = pd.read_csv(data_path)
    if args.subset:
        df = df.sample(n=500, random_state=2025).reset_index(drop=True)

    if not args.soft_labels:
        queries, labels, query_ids, label_indices, label_encoder, model_cumsum_dict, model_unique_dict, model_quality_dict, model_unique_quality_dict = load_data(data_path, subset=args.subset)
    else:
        logging.info("Using soft labels for training.")
        normalize_func = NORMALIZE_NAME_TO_FUNC.get(args.normalize_func, normalize_default)
        queries, labels, query_ids, label_indices, label_encoder, model_cumsum_dict, soft_labels, model_unique_dict, model_quality_dict, model_unique_quality_dict = load_data_with_soft_labels(data_path, subset=args.subset, normalize_func=normalize_func)

    # Generate base embeddings (shared across all models)
    embeddings = generate_embeddings(queries, cache_path, use_cache=not args.no_cache)
    print(f"Base embeddings shape: {embeddings.shape}")

    # Load test mask for wild_inf_chat (to match wildchat test split)
    test_mask = None
    if args.data == 'wild_inf_chat':
        test_mask_path = strategy_dir / 'wildchat_test_mask.npy'
        if test_mask_path.exists():
            test_mask = np.load(test_mask_path)
            print(f"Loaded test mask from {test_mask_path}")
            print(f"  Test mask shape: {test_mask.shape}, test samples: {test_mask.sum()}")
        else:
            print(f"WARNING: Test mask not found at {test_mask_path}")
            print("  Run merge_data/generate_wildchat_test_mask.py to generate it.")
            print("  Falling back to random split.")

    # Load per-model features if any flags are enabled
    per_model_features = None
    use_any_features = args.quality or args.perplexity or args.enc_ans

    if use_any_features:
        print("\nLoading per-model features...")
        print(f"  Enabled flags: quality={args.quality}, perplexity={args.perplexity}, enc_ans={args.enc_ans}")

        per_model_features = {}
        n_samples = len(queries)

        try:
            for model_name in label_encoder.classes_:
                features, feature_dim = load_per_model_features(
                    base_dir=base_dir,
                    dataset=args.data,
                    strategy=args.strategy,
                    model_name=model_name,
                    use_quality=args.quality,
                    use_perplexity=args.perplexity,
                    use_enc_ans=args.enc_ans,
                    n_samples=n_samples
                )
                per_model_features[model_name] = features

            print(f"Successfully loaded per-model features for {len(per_model_features)} models")

        except (FileNotFoundError, ValueError) as e:
            print(f"\nERROR: Failed to load per-model features")
            print(f"  {str(e)}")
            print(f"\nPlease ensure feature files exist at:")
            print(f"  {base_dir / 'outputs' / args.data / args.strategy / 'per_model_features' / '<model_name>/'}")
            sys.exit(1)
    else:
        print("\nNo per-model features enabled (using base embeddings only)")

    # Run train/val/test split MLP multi-classification
    print(f"\nRunning train/val/test split MLP multi-classification...")
    print(f"Training {len(label_encoder.classes_)} separate classifiers (one per model)...")

    if args.soft_labels:
        y_true, y_pred, test_accuracy, train_val_indices, test_indices, models, predictions_per_model, train_losses, val_losses = run_mlp_classification_split(
            embeddings, label_indices, label_encoder,
            lr=args.lr, n_epochs=args.n_epochs, batch_size=args.batch_size, hidden_dim=args.hidden_dim,
            use_soft_labels=True, soft_labels=soft_labels,
            per_model_features=per_model_features,
            output_dir=output_dir, model_cumsum_dict=model_cumsum_dict,
            weight_decay=args.weight_decay,
            use_class_weight=args.class_weight,
            test_mask=test_mask
        )
    else:
        y_true, y_pred, test_accuracy, train_val_indices, test_indices, models, predictions_per_model, train_losses, val_losses = run_mlp_classification_split(
            embeddings, label_indices, label_encoder,
            lr=args.lr, n_epochs=args.n_epochs, batch_size=args.batch_size, hidden_dim=args.hidden_dim,
            per_model_features=per_model_features,
            output_dir=output_dir, model_cumsum_dict=model_cumsum_dict,
            weight_decay=args.weight_decay,
            use_class_weight=args.class_weight,
            test_mask=test_mask
        )

    # Save trained models
    if args.save_model:
        # Build model directory name (same logic as log file)
        if not args.soft_labels:
            model_dir_name = f'{args.exp}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}_hd_{args.hidden_dim}'
        else:
            model_dir_name = f'{args.exp}_soft_labels_{args.normalize_func}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}_hd_{args.hidden_dim}'

        # Add feature flags to directory name
        if feature_flags:
            model_dir_name += '_features_' + '_'.join(feature_flags)

        # Add weight decay to directory name if non-zero
        if args.weight_decay > 0:
            model_dir_name += f'_wd_{args.weight_decay}'

        # Add class weight flag to directory name if enabled
        if args.class_weight:
            model_dir_name += '_cw'

        # Add current time
        model_dir_name += f"_{cur_time}"

        model_dir = output_dir / 'models' / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model in models.items():
            model_path = model_dir / f'{model_name}.pt'
            torch.save(model, model_path)

        logging.info(f"Trained models saved to {model_dir}")
        # Save label classes as well
        label_classes_path = model_dir / 'label_classes.npy'
        np.save(label_classes_path, label_encoder.classes_)
        # also save test indices
        test_indices_path = model_dir / 'test_indices.npy'
        np.save(test_indices_path, np.array(test_indices))

    # Calculate metrics (adapted for single evaluation)
    fold_accuracies = [test_accuracy]  # Single accuracy value
    results = calculate_metrics(y_true, y_pred, test_indices, fold_accuracies, label_encoder, -1, 1, model_cumsum_dict, model_unique_dict, model_quality_dict, model_unique_quality_dict)

    # add overall best model baseline (using train/test split, no data leakage)
    best_overall_acc, best_overall_cumsum, best_overall_unique, best_overall_quality, best_overall_unique_quality = calc_best_overall_baseline_split(
        df,
        np.array(train_val_indices),
        np.array(test_indices),
        model_unique_dict,
        model_quality_dict,
        model_unique_quality_dict
    )
    results['cumsum_metrics']['best_overall_baseline_cumsum'] = best_overall_cumsum
    results['best_overall_baseline_accuracy'] = best_overall_acc
    if best_overall_unique is not None and results.get('unique_metrics'):
        results['unique_metrics']['best_overall_baseline_unique'] = best_overall_unique
    if best_overall_quality is not None and results.get('quality_metrics'):
        results['quality_metrics']['best_overall_baseline_quality'] = best_overall_quality
    if best_overall_unique_quality is not None and results.get('unique_quality_metrics'):
        results['unique_quality_metrics']['best_overall_baseline_unique_quality'] = best_overall_unique_quality

    # Create predictions DataFrame
    predictions_df = create_predictions_df(query_ids, queries, y_true, y_pred, label_encoder, test_indices, model_cumsum_dict, model_unique_dict, model_quality_dict, model_unique_quality_dict)

    # Save results
    print("\nSaving results...")
    save_results(results, predictions_df, output_dir)
    save_confusion_matrix_csv(results['confusion_matrix'], label_encoder, output_dir)
    create_confusion_matrix_plot(results['confusion_matrix'], label_encoder, output_dir)
    save_per_model_metrics_csv(results, label_encoder, output_dir)
    # also save predictions_per_model, numpy array
    predictions_per_model_path = output_dir / 'predictions_per_model.npy'
    np.save(predictions_per_model_path, predictions_per_model)
    # and label classes
    label_classes_path = output_dir / 'label_classes.npy'
    np.save(label_classes_path, label_encoder.classes_)
    

    # Print summary
    print(f"\n{'='*70}")
    print("Train/Val/Test Split MLP Multi-Classification Results")
    print(f"{'='*70}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"\nPer-Model Performance:")
    print(f"{'-'*70}")
    print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"{'-'*70}")
    for model in label_encoder.classes_:
        metrics = results['per_model_metrics'][model]
        print(f"{model:<25} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
              f"{metrics['f1_score']:>10.3f} {metrics['support']:>10}")
    print(f"{'='*70}")

    # Print cumsum summary
    logging.info(f"Cumulative Sum Metrics:")
    logging.info(f"{'-'*70}")
    logging.info(f"Ground Truth Mean Cumsum: {results['cumsum_metrics']['ground_truth_cumsum_mean']:.2f}")
    logging.info(f"Predicted Mean Cumsum:    {results['cumsum_metrics']['predicted_cumsum_mean']:.2f}")
    logging.info(f"Best Overall Cumsum: {results['cumsum_metrics']['best_overall_baseline_cumsum']:.2f}")
    logging.info(f"Random Mean Cumsum:       {results['cumsum_metrics']['random_cumsum_mean']:.2f}")
    logging.info('-'*70)

    # Print unique answer metrics (if available)
    if results.get('unique_metrics'):
        logging.info(f"Unique Answer Metrics:")
        logging.info(f"Ground Truth Mean Unique: {results['unique_metrics']['ground_truth_unique_mean']:.2f}")
        logging.info(f"Predicted Mean Unique:    {results['unique_metrics']['predicted_unique_mean']:.2f}")
        logging.info(f"Random Mean Unique:       {results['unique_metrics']['random_unique_mean']:.2f}")
        logging.info('-'*70)

    # Print quality metrics (if available)
    if results.get('quality_metrics'):
        logging.info(f"Quality Metrics:")
        logging.info(f"Ground Truth Mean Quality: {results['quality_metrics']['ground_truth_quality_mean']:.2f}")
        logging.info(f"Predicted Mean Quality:    {results['quality_metrics']['predicted_quality_mean']:.2f}")
        logging.info(f"Random Mean Quality:       {results['quality_metrics']['random_quality_mean']:.2f}")
        logging.info('-'*70)

    # Print unique_quality metrics (if available)
    if results.get('unique_quality_metrics'):
        logging.info(f"Unique Quality Metrics:")
        logging.info(f"Ground Truth Mean Unique Quality: {results['unique_quality_metrics']['ground_truth_unique_quality_mean']:.2f}")
        logging.info(f"Predicted Mean Unique Quality:    {results['unique_quality_metrics']['predicted_unique_quality_mean']:.2f}")
        logging.info(f"Random Mean Unique Quality:       {results['unique_quality_metrics']['random_unique_quality_mean']:.2f}")
        logging.info('-'*70)

    # Print acc summary
    logging.info(f"Random Baseline Accuracy: {1/len(label_encoder.classes_)*100:.2f}%")
    logging.info(f"Best Overall Baseline Accuracy: {best_overall_acc*100:.2f}%")
    logging.info(f"MLP Multi-Classifier Test Accuracy: {test_accuracy*100:.2f}%")
    logging.info('-'*70)

    print("\nAll outputs saved successfully!")

if __name__ == "__main__":
    main()
