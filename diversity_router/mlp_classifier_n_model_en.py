#!/usr/bin/env python3
"""
Usage (run from diversity_router_code/ root):
#18 models, spec
    python diversity_router/mlp_classifier_n_model_en.py --data wildchat --strategy list_all --n_epochs 10 --hidden_dim 512 --data_dir outputs_18_models --output_dir outputs_18_models
    


"""
import datetime
import time
import json
import argparse
import numpy as np
from pathlib import Path
import sys
from typing import Tuple, Dict, List

import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from best_overall_baseline import calc_best_overall_baseline_split
from utils import *
from utils import _load_auxiliary_metrics

import logging
from scipy.stats import entropy

#torch.manual_seed(2025)


class MLPClassifier(nn.Module):
    """Single-output MLP classifier for binary probability prediction."""

    def __init__(self, input_dim: int = 3584, hidden_dim: int = 1024):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='sigmoid')

        

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


def load_per_model_query_encodings(
    base_dir: Path,
    data_dir: str,
    dataset: str,
    strategy: str,
    model_names: List[str],
    n_samples: int,
    max_dim: int = None,
) -> Dict[str, np.ndarray]:
    """
    Load per-model query encodings from disk.

    Args:
        base_dir: Base directory for the project
        dataset: Dataset name (e.g., 'longform_qa', 'wildchat')
        strategy: Strategy name (e.g., 'list_all')
        model_names: List of model names to load encodings for
        n_samples: Expected number of samples for validation

    Returns:
        Dict mapping model_name -> query_encodings (shape: n_samples x embedding_dim)

    Raises:
        FileNotFoundError: If encoding directory or files don't exist
        ValueError: If encoding array shapes don't match n_samples
    """
    if dataset == 'wild_inf_chat' and strategy == 'list_all':
        print("Loading from query_encode_per_model_ directory for wild_inf_chat/list_all")
        encodings_base_dir = base_dir / data_dir / dataset / strategy / 'query_encode_per_model_'
    else:
        encodings_base_dir = base_dir / data_dir / dataset / strategy / 'query_encode_per_model'

    if not encodings_base_dir.exists():
        raise FileNotFoundError(
            f"Per-model encodings directory not found: {encodings_base_dir}\n"
            f"Please generate encodings first using the encoding construction scripts."
        )

    per_model_encodings = {}

    for model_name in model_names:
        model_dir = encodings_base_dir / model_name
        encoding_path = model_dir / 'query_encode.npy'

        if not encoding_path.exists():
            raise FileNotFoundError(
                f"Query encodings not found for {model_name}: {encoding_path}\n"
                f"Please generate encodings for this model first."
            )

        encodings = np.load(encoding_path).astype(np.float32)

        if max_dim is not None and encodings.shape[1] > max_dim:
            original_dim = encodings.shape[1]
            encodings = encodings[:, :max_dim]
            print(f"  Truncated {model_name} encodings from dim={original_dim} to dim={max_dim}")

        if encodings.shape[0] != n_samples:
            raise ValueError(
                f"Encoding size mismatch for {model_name}: "
                f"expected {n_samples} samples, got {encodings.shape[0]}"
            )

        per_model_encodings[model_name] = encodings
        print(f"  Loaded {model_name}: shape={encodings.shape}, dtype={encodings.dtype}")

    return per_model_encodings


def load_per_model_features(
    base_dir: Path,
    data_dir: str,
    dataset: str,
    strategy: str,
    model_name: str,
    use_quality: bool,
    use_perplexity: bool,
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
        n_samples: Expected number of samples for validation

    Returns:
        features: Combined features array of shape (n_samples, feature_dim)
        feature_dim: Total dimension of loaded features

    Raises:
        FileNotFoundError: If feature directory or files don't exist
        ValueError: If feature array shapes don't match n_samples
    """
    features_dir = base_dir / data_dir / dataset / strategy / 'per_model_features' / model_name

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

    if not feature_arrays:
        return np.zeros((n_samples, 0)), 0

    combined_features = np.concatenate(feature_arrays, axis=1)
    feature_dim = combined_features.shape[1]

    print(f"  Loaded {model_name} features: {', '.join(feature_names)} (dim={feature_dim})")

    return combined_features, feature_dim


def run_mlp_classification_split_per_model_encoding(
    per_model_encodings: Dict[str, np.ndarray],
    label_indices: np.ndarray,
    label_encoder: LabelEncoder,
    lr: float = 0.001,
    n_epochs: int = 50,
    batch_size: int = 32,
    hidden_dim: int = 1024,
    use_soft_labels: bool = False,
    soft_labels: np.ndarray = None,
    per_model_features: Dict[str, np.ndarray] = None,
    output_dir: Path = None,
    model_cumsum_dict: Dict[str, np.ndarray] = None,
    weight_decay: float = 0,
    test_mask: np.ndarray = None,
    train_mask: np.ndarray = None,
    train_val_indices: np.ndarray = None,
    test_indices_precomputed: np.ndarray = None,
    subset : int = None,
) -> Tuple[np.ndarray, np.ndarray, float, List[int], List[int], Dict, Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Run train/val/test split with multi-classifier (n classifiers) approach using 80/10/10 split.
    Uses per-model query encodings instead of shared embeddings.

    For the training phase, trains one MLP classifier per candidate model. Each classifier
    predicts a probability (0-1) indicating whether its model is the best for the query.

    If test_mask is provided, use it to determine test indices instead of random split.
    This is used for wild_inf_chat to ensure the test set matches wildchat's test split.
    """

    n_samples = len(label_indices)

    indices = np.arange(n_samples)
    
    if train_val_indices is not None and test_indices_precomputed is not None:
        train_val_idx = train_val_indices
        test_idx = test_indices_precomputed
        print(f"Using pre-computed indices: {len(train_val_idx)} train+val, {len(test_idx)} test")
    elif train_mask is not None and test_mask is not None:

        train_val_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        print(f"Using provided train and test masks: {len(train_val_idx)} train+val samples, {len(test_idx)} test samples")
    elif test_mask is not None:
        # Use provided test mask (for wild_inf_chat to match wildchat test split)
        raise ValueError("Test mask provided without train mask. Please provide both train and test masks to ensure proper splitting.")
        # test_idx = np.where(test_mask)[0]
        # train_val_idx = np.where(~test_mask)[0]
        # print(f"Using provided test mask: {len(test_idx)} test samples, {len(train_val_idx)} train+val samples")
    else:
        # First split: separate test set (20%)
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=label_indices,
            random_state=2025
        )
        
    # if subset is specified, take  a random subset of the train_val_idx for faster experimentation
    if subset is not None:
        print(f"Taking subset of data for faster experimentation: {subset} samples")
        print(f"Before subsetting: {len(train_val_idx)} train+val samples")
        train_val_idx = np.random.choice(train_val_idx, size=subset, replace=False)
        print(f"Using subset of data: {len(train_val_idx)} train+val samples")

    # Second split: separate validation set (10% of original, ~12.5% of train_val)
    if not subset:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=1/8,
            stratify=label_indices[train_val_idx],
            random_state=2025
        )
    else:
        # If using subset, just split into train/val without stratification to avoid issues with small sample size
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=1/5,
            #stratify=label_indices[train_val_idx],
            random_state=2025
        )

    y_train, y_val, y_test = label_indices[train_idx], label_indices[val_idx], label_indices[test_idx]

    if use_soft_labels:
        assert soft_labels is not None, "Soft labels must be provided when use_soft_labels is True."
        soft_labels_train = soft_labels[train_idx]
        soft_labels_val = soft_labels[val_idx]
        soft_labels_test = soft_labels[test_idx]
        print(f"Using soft labels for training.")

    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")

    per_model_train_data = {}
    per_model_val_data = {}
    per_model_test_data = {}

    for model_name in label_encoder.classes_:
        if model_name not in per_model_encodings:
            raise ValueError(f"Missing encodings for model: {model_name}")

        encodings = per_model_encodings[model_name]
        # L2 normalization
        encodings = F.normalize(torch.tensor(encodings), p=2, dim=1).numpy()

        # Concatenate per-model features if provided
        if per_model_features is not None and model_name in per_model_features:
            model_features = per_model_features[model_name]
            encodings = np.concatenate([encodings, model_features], axis=1)

        encodings_train = encodings[train_idx]
        encodings_val = encodings[val_idx]
        encodings_test = encodings[test_idx]

        per_model_train_data[model_name] = torch.tensor(encodings_train, dtype=torch.float32)
        per_model_val_data[model_name] = torch.tensor(encodings_val, dtype=torch.float32)
        per_model_test_data[model_name] = torch.tensor(encodings_test, dtype=torch.float32)

        print(f"  {model_name}: input_dim = {encodings_train.shape[1]}")

    models = {}
    optimizers = {}
    criterion = nn.BCELoss()

    for model_idx, model_name in enumerate(label_encoder.classes_):
        input_dim = per_model_train_data[model_name].shape[1]

        models[model_name] = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        optimizers[model_name] = torch.optim.Adam(
            models[model_name].parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    logging.info(f"Training {len(label_encoder.classes_)} separate classifiers (one per model)...")
    logging.info(f"MLP Classifier Architecture: {models[label_encoder.classes_[0]]}")
    logging.info(f"Loss Function: {criterion}")
    logging.info(f"Optimizer: {optimizers[label_encoder.classes_[0]]}")

    # Initialize loss tracking
    train_losses = {model_name: [] for model_name in label_encoder.classes_}
    val_losses = {model_name: [] for model_name in label_encoder.classes_}

    best_val_acc = 0.0

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")

        for model_idx, model_name in enumerate(label_encoder.classes_):
            model = models[model_name]
            optimizer = optimizers[model_name]

            X_train_model = per_model_train_data[model_name]

            if use_soft_labels:
                y_train_target = soft_labels_train[:, model_idx]
            else:
                y_train_target = (y_train == model_idx).astype(float)

            y_train_target_tensor = torch.tensor(y_train_target, dtype=torch.float32).unsqueeze(1)

            model.train()
            permutation = torch.randperm(X_train_model.size()[0])

            epoch_losses = []
            for i in range(0, X_train_model.size()[0], batch_size):
                indices_batch = permutation[i:i + batch_size]
                batch_x = X_train_model[indices_batch]
                batch_y = y_train_target_tensor[indices_batch]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            train_losses[model_name].append(avg_loss)
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"  Model {model_name}: Loss = {avg_loss:.4f}")

        scores_per_sample_val = np.zeros((len(val_idx), len(label_encoder.classes_)))

        for model_idx, model_name in enumerate(label_encoder.classes_):
            model = models[model_name]
            model.eval()

            X_val_model = per_model_val_data[model_name]

            # Prepare validation target labels for this model
            if use_soft_labels:
                y_val_target = soft_labels_val[:, model_idx]
            else:
                y_val_target = (y_val == model_idx).astype(float)
            y_val_target_tensor = torch.tensor(y_val_target, dtype=torch.float32).unsqueeze(1)

            with torch.no_grad():
                outputs = model(X_val_model)
                scores = outputs.squeeze().numpy()
                scores_per_sample_val[:, model_idx] = scores

                # Compute validation loss for this model
                val_loss_model = criterion(outputs, y_val_target_tensor).item()
                val_losses[model_name].append(val_loss_model)

        val_pred = np.argmax(scores_per_sample_val, axis=1)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"Validation Accuracy = {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.3f}")


    # Plot loss curves
    if output_dir is not None:
        plot_loss_curves(train_losses, val_losses, output_dir)

    # Calculate final training accuracy
    scores_per_sample_train = np.zeros((len(train_idx), len(label_encoder.classes_)))

    for model_idx, model_name in enumerate(label_encoder.classes_):
        model = models[model_name]
        model.eval()

        X_train_model = per_model_train_data[model_name]

        with torch.no_grad():
            outputs = model(X_train_model)
            scores = outputs.squeeze().numpy()
            scores_per_sample_train[:, model_idx] = scores

    y_pred_train = np.argmax(scores_per_sample_train, axis=1)
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"Train Accuracy = {train_acc:.3f}")

    scores_per_sample_test = np.zeros((len(test_idx), len(label_encoder.classes_)))

    inference_start = time.time()
    for model_idx, model_name in enumerate(label_encoder.classes_):
        model = models[model_name]
        model.eval()

        X_test_model = per_model_test_data[model_name]

        with torch.no_grad():
            outputs = model(X_test_model)
            scores = outputs.squeeze().numpy()
            scores_per_sample_test[:, model_idx] = scores
    inference_elapsed = time.time() - inference_start
    avg_inference_time_s = inference_elapsed / len(test_idx)
    print(f"Avg inference time per test sample = {avg_inference_time_s:.6f} s")

    y_pred = np.argmax(scores_per_sample_test, axis=1)
    test_acc = accuracy_score(y_test, y_pred)

    if use_soft_labels:
        pred_probs = scores_per_sample_test / (scores_per_sample_test.sum(axis=1, keepdims=True) + 1e-10)
        kl_divergence = entropy(soft_labels_test.T, pred_probs.T)
        print(f"Test KL Divergence = {np.average(kl_divergence):.4f}")

    print(f"Test Accuracy = {test_acc:.3f}")

    return np.array(y_test), y_pred, test_acc, train_acc, train_idx.tolist(), test_idx.tolist(), models, train_losses, val_losses,scores_per_sample_test


def create_predictions_df_with_scores(
    query_ids, queries, y_true, y_pred, label_encoder, test_indices,
    model_cumsum_dict, scores_per_sample,
    model_unique_dict=None, model_quality_dict=None, model_unique_quality_dict=None
):
    """Call create_predictions_df and add per-model MLP scores per sample.

    scores_per_sample: np.ndarray of shape (n_test, n_models) — raw MLP
    probabilities aligned with label_encoder.classes_.
    Adds one column per model: 'score_<model_name>'.
    """
    df = create_predictions_df(
        query_ids, queries, y_true, y_pred, label_encoder, test_indices,
        model_cumsum_dict, model_unique_dict, model_quality_dict,
        model_unique_quality_dict
    )
    for model_idx, model_name in enumerate(label_encoder.classes_):
        df[f'score_{model_name}'] = scores_per_sample[:, model_idx]
    return df


def main():
    parser = argparse.ArgumentParser(description="MLP Multi-Classifier with Train/Val/Test Split (Per-Model Encodings)")
    parser.add_argument('--data_dir', default='outputs', choices=['outputs', 'outputs_18_models'], 
                        help='strategy dir where train/test data is stored')
    parser.add_argument('--output_dir', default='outputs_split', choices=['outputs_split','outputs_18_models'],
                        help='output directory for results')
    parser.add_argument('--data', default='wildchat', choices=['wildchat', 'wild_inf_chat', 'inf_chat', 'inf_chat_2k', 'wild_inf_chat_2k'],
                       help='Dataset to analyze (only wildchat supported)')
    parser.add_argument('--strategy', default='list_all', choices=['list_all', 'sample_1', 'sample_2'],
                       help='Strategy to analyze')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for MLP')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Hidden layer dimension')
    parser.add_argument('--soft_labels', action='store_true',
                       help='Use soft labels instead of hard labels')
    parser.add_argument('--normalize_func', type=str, default='default',
                       help='Normalization function for soft labels (if any)')
    parser.add_argument('--exp', type=str, default='',
                       help='Experiment name for logging')
    parser.add_argument('--save_model', action='store_true',
                       help='Whether to save the trained models')
    parser.add_argument('--weight_decay', type=float, default=0,
                       help='Weight decay (L2 regularization) for optimizer')
    parser.add_argument('--quality', action='store_true',
                       help='Include per-model quality features (1-dim)')
    parser.add_argument('--perplexity', action='store_true',
                       help='Include per-model perplexity features (1-dim)')
    parser.add_argument('--max_dim', type=int, default=None,
                        help='Truncate per-model query encodings to this max dimension (default: no truncation)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Custom log directory (default: output_dir/logs)')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--subset', type=int, default=None, help='Use only a subset of the data for faster experimentation (for debugging)')
    parser.add_argument('--save_data', action='store_true', help='Save query_ids, train_val_indices, test_indices, per_model_encodings, and model_cumsum_dict for debugging')
    args = parser.parse_args()
    
    # set seed
    torch.manual_seed(args.seed)

    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = Path(__file__).parent.parent
    strategy_dir = base_dir / args.data_dir / args.data / args.strategy
    output_parent_dir = base_dir / args.output_dir / args.data / args.strategy
    data_path = strategy_dir / 'best_model_per_query.csv'

    output_dir = output_parent_dir / 'mlp_n_model_en'

    output_dir.mkdir(parents=True, exist_ok=True)
    log_parent = Path(args.log_dir) if args.log_dir else output_dir / 'logs'
    log_parent.mkdir(parents=True, exist_ok=True)

    if not args.soft_labels:
        log_file_name = f'{args.exp}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}_hd_{args.hidden_dim}'
    else:
        log_file_name = f'{args.exp}_soft_labels_{args.normalize_func}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}_hd_{args.hidden_dim}'
        
    # Add weight decay to log file name if non-zero
    if args.weight_decay > 0:
        log_file_name += f'_wd_{args.weight_decay}'

    # Add feature flags to filename
    feature_flags = []
    if args.quality:
        feature_flags.append('q')
    if args.perplexity:
        feature_flags.append('p')

    if feature_flags:
        log_file_name += '_features_' + '_'.join(feature_flags)

    if args.max_dim is not None:
        log_file_name += f'_maxdim_{args.max_dim}'
    log_file_name += f'_seed_{args.seed}'

    log_file_name += f"_{cur_time}.log"
    log_file = log_parent / log_file_name

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(sys.stdout)
                    ])

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    import pandas as pd

    # Initialize split control variables
    precomputed_train_val_idx = None
    precomputed_test_idx = None
    soft_labels = None
    per_model_encodings = None
    load_dataset = args.data  # effective dataset for loading encodings/features

    # === Standard single-source loading ===
    df = pd.read_csv(data_path)
    if not args.soft_labels:
        queries, labels, query_ids, label_indices, label_encoder, model_cumsum_dict, model_unique_dict, model_quality_dict, model_unique_quality_dict = load_data(data_path)
    else:
        logging.info("Using soft labels for training.")
        normalize_func = NORMALIZE_NAME_TO_FUNC.get(args.normalize_func, normalize_default)
        queries, labels, query_ids, label_indices, label_encoder, model_cumsum_dict, soft_labels, model_unique_dict, model_quality_dict, model_unique_quality_dict = load_data_with_soft_labels(data_path, normalize_func=normalize_func)

    n_samples = len(queries)
    print(f"\nTotal samples: {n_samples}")

    print("Loading per-model query encodings...")
    try:
        per_model_encodings = load_per_model_query_encodings(
            base_dir=base_dir, data_dir=args.data_dir, dataset=load_dataset,
            strategy=args.strategy, model_names=label_encoder.classes_.tolist(),
            n_samples=n_samples, max_dim=args.max_dim)
        print(f"Successfully loaded per-model encodings for {len(per_model_encodings)} models")
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: Failed to load per-model encodings")
        print(f"  {str(e)}")
        sys.exit(1)

    # Load per-model features if any flags are enabled
    per_model_features = None
    use_any_features = args.quality or args.perplexity


    if use_any_features:
        print("\nLoading per-model features...")
        print(f"  Enabled flags: quality={args.quality}, perplexity={args.perplexity}")

        per_model_features = {}

        try:
            for model_name in label_encoder.classes_:
                features, feature_dim = load_per_model_features(
                    base_dir=base_dir,
                    data_dir=args.data_dir,
                    dataset=load_dataset,
                    strategy=args.strategy,
                    model_name=model_name,
                    use_quality=args.quality,
                    use_perplexity=args.perplexity,
                    n_samples=n_samples
                )
                per_model_features[model_name] = features

            print(f"Successfully loaded per-model features for {len(per_model_features)} models")

        except (FileNotFoundError, ValueError) as e:
            print(f"\nERROR: Failed to load per-model features")
            print(f"  {str(e)}")
            print(f"\nPlease ensure feature files exist at:")
            print(f"  {base_dir / args.data_dir / args.data / args.strategy / 'per_model_features' / '<model_name>/'}")
            sys.exit(1)
    else:
        print("\nNo per-model features enabled (using encodings only)")

    print(f"\nRunning train/val/test split MLP multi-classification...")
    print(f"Training {len(label_encoder.classes_)} separate classifiers (one per model)...")

    if args.soft_labels:
        y_true, y_pred, test_accuracy, train_accuracy, train_val_indices, test_indices, models, train_losses, val_losses, scores_per_sample_test = run_mlp_classification_split_per_model_encoding(
            per_model_encodings, label_indices, label_encoder,
            lr=args.lr, n_epochs=args.n_epochs, batch_size=args.batch_size, hidden_dim=args.hidden_dim,
            use_soft_labels=True, soft_labels=soft_labels,
            per_model_features=per_model_features,
            output_dir=output_dir, model_cumsum_dict=model_cumsum_dict,
            weight_decay=args.weight_decay,
            train_val_indices=precomputed_train_val_idx,
            test_indices_precomputed=precomputed_test_idx,
            subset=args.subset
        )
    else:
        y_true, y_pred, test_accuracy, train_accuracy, train_val_indices, test_indices, models, train_losses, val_losses, scores_per_sample_test = run_mlp_classification_split_per_model_encoding(
            per_model_encodings, label_indices, label_encoder,
            lr=args.lr, n_epochs=args.n_epochs, batch_size=args.batch_size, hidden_dim=args.hidden_dim,
            per_model_features=per_model_features,
            output_dir=output_dir, model_cumsum_dict=model_cumsum_dict,
            weight_decay=args.weight_decay,
            train_val_indices=precomputed_train_val_idx,
            test_indices_precomputed=precomputed_test_idx,
            subset=args.subset
        )

    if args.save_model:
        if not args.soft_labels:
            model_dir_name = f'{args.exp}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}_hd_{args.hidden_dim}'
        else:
            model_dir_name = f'{args.exp}_soft_labels_{args.normalize_func}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}_hd_{args.hidden_dim}'
            

        # Add weight decay to directory name if non-zero
        if args.weight_decay > 0:
            model_dir_name += f'_wd_{args.weight_decay}'

        # Add feature flags to directory name
        if feature_flags:
            model_dir_name += '_features_' + '_'.join(feature_flags)

        if args.max_dim is not None:
            model_dir_name += f'_maxdim_{args.max_dim}'
        model_dir_name += f'_seed_{args.seed}'

        model_dir_name += f"_{cur_time}"

        model_dir = output_dir / 'models' / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model in models.items():
            model_path = model_dir / f'{model_name}.pt'
            torch.save(model, model_path)

        logging.info(f"Trained models saved to {model_dir}")
        label_classes_path = model_dir / 'label_classes.npy'
        np.save(label_classes_path, label_encoder.classes_)


    if args.save_data:
        # save query_ids, train_val_indices, test_indices,per_model_encodings, model_cumsum_dict for debugging
        np.save(output_dir / "query_ids.npy", query_ids)
        np.save(output_dir / "train_val_indices.npy", np.array(train_val_indices))
        np.save(output_dir / "test_indices.npy", np.array(test_indices))
        np.save(output_dir / "per_model_encodings.npy", per_model_encodings)
        np.save(output_dir / "model_cumsum_dict.npy", model_cumsum_dict)

    fold_accuracies = [test_accuracy]
    results = calculate_metrics(y_true, y_pred, test_indices, fold_accuracies, label_encoder, -1, 1, model_cumsum_dict, model_unique_dict, model_quality_dict, model_unique_quality_dict)
    
    # double check the y_true, y_pred, test_indices, label_encoder and model_cumsum
    # print example query_id, y_true, y_pred, test_index, model_cumsum for first 5 test samples
    print("\nExample test samples:")
    for i in range(min(5, len(test_indices))):
        idx = test_indices[i]
        print(f"Query ID: {query_ids[idx]}, True Label: {label_encoder.inverse_transform([y_true[i]])[0]}, Predicted Label: {label_encoder.inverse_transform([y_pred[i]])[0]}, Cumsum: {model_cumsum_dict[label_encoder.inverse_transform([y_true[i]])[0]][idx]:.2f}")  
        

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

    predictions_df = create_predictions_df_with_scores(query_ids, queries, y_true, y_pred, label_encoder, test_indices, model_cumsum_dict, scores_per_sample_test, model_unique_dict, model_quality_dict, model_unique_quality_dict)

    print("\nSaving results...")
    save_results(results, predictions_df, output_dir)
    save_confusion_matrix_csv(results['confusion_matrix'], label_encoder, output_dir)
    create_confusion_matrix_plot(results['confusion_matrix'], label_encoder, output_dir)
    save_per_model_metrics_csv(results, label_encoder, output_dir)

    print(f"\n{'='*70}")
    print("Train/Val/Test Split MLP Multi-Classification Results (Per-Model Encodings)")
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

    logging.info(f"Random Baseline Accuracy: {1/len(label_encoder.classes_)*100:.2f}%")
    logging.info(f"Best Overall Baseline Accuracy: {best_overall_acc*100:.2f}%")
    logging.info(f"MLP Multi-Classifier Train Accuracy: {train_accuracy*100:.2f}%")
    logging.info(f"MLP Multi-Classifier Test Accuracy: {test_accuracy*100:.2f}%")
    logging.info('-'*70)
    
    # Acc	# unique answers	answer quality	avg unique answer qualiy	Cumsum	Cumsum %
    logging.info(f"ready to csv:")
    random_baseline_metrics = [
        f"{1/len(label_encoder.classes_)*100:.2f}%",
        f"{results['unique_metrics']['random_unique_mean']:.2f}" if results.get('unique_metrics') else 'N/A',
        f"{results['quality_metrics']['random_quality_mean']:.2f}" if results.get('quality_metrics') else 'N/A',
        f"{results['unique_quality_metrics']['random_unique_quality_mean']:.2f}" if results.get('unique_quality_metrics') else 'N/A',
        f"{results['cumsum_metrics']['random_cumsum_mean']:.2f}",
        f"{results['cumsum_metrics']['random_cumsum_mean']/500*100:.2f}%"
    ]
    logging.info(f" random baseline:"+ ', '.join(random_baseline_metrics))
    best_overall_baseline_metrics = [
        f"{best_overall_acc*100:.2f}%",
        f"{results['unique_metrics']['best_overall_baseline_unique']:.2f}" if results.get('unique_metrics') else 'N/A',
        f"{results['quality_metrics']['best_overall_baseline_quality']:.2f}" if results.get('quality_metrics') else 'N/A',
        f"{results['unique_quality_metrics']['best_overall_baseline_unique_quality']:.2f}" if results.get('unique_quality_metrics') else 'N/A',
        f"{results['cumsum_metrics']['best_overall_baseline_cumsum']:.2f}",
        f"{results['cumsum_metrics']['best_overall_baseline_cumsum']/500*100:.2f}%"
    ]
    logging.info(f" Majority:"+ ', '.join(best_overall_baseline_metrics))
    
    oracle_metrics = [
        f"100%",
        f"{results['unique_metrics']['ground_truth_unique_mean']:.2f}" if results.get('unique_metrics') else 'N/A',
        f"{results['quality_metrics']['ground_truth_quality_mean']:.2f}" if results.get('quality_metrics') else 'N/A',
        f"{results['unique_quality_metrics']['ground_truth_unique_quality_mean']:.2f}" if results.get('unique_quality_metrics') else 'N/A',
        f"{results['cumsum_metrics']['ground_truth_cumsum_mean']:.2f}",
        f"{results['cumsum_metrics']['ground_truth_cumsum_mean']/500*100:.2f}%"
    ]
    logging.info(f" Oracle:"+ ', '.join(oracle_metrics))
    
    mlp_classifier_metrics = [
        f"{test_accuracy*100:.2f}%",
        f"{results['unique_metrics']['predicted_unique_mean']:.2f}" if results.get('unique_metrics') else 'N/A',
        f"{results['quality_metrics']['predicted_quality_mean']:.2f}" if results.get('quality_metrics') else 'N/A',
        f"{results['unique_quality_metrics']['predicted_unique_quality_mean']:.2f}" if results.get('unique_quality_metrics') else 'N/A',
        f"{results['cumsum_metrics']['predicted_cumsum_mean']:.2f}",
        f"{results['cumsum_metrics']['predicted_cumsum_mean']/500*100:.2f}%"
    ]
    logging.info(f" MLP multi-classifier:"+ ', '.join(mlp_classifier_metrics))
    
    logging.info('-'*70)

    # Router model selection frequency
    model_order = [
        'llama-3.2-1b', 'llama-3.2-3b', 'llama-3.1-8b', 'llama-3.3-70b',
        'qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b', 'qwen3-8b', 'qwen3-14b', 'qwen2.5-72b-instruct',
        'olmo-2-0425-1b', 'olmo-2-1124-7b', 'olmo-2-1124-13b', 'olmo-2-0325-32b',
        'gemma-3-1b-it', 'gemma-3-4b-it', 'gemma-3-12b-it', 'gemma-3-27b-it'
    ]
    pred_counts = predictions_df['predicted_model'].value_counts()
    total_preds = len(predictions_df)
    pcts = [pred_counts.get(model, 0) / total_preds for model in model_order]
    logging.info('Router model selection frequency:')
    logging.info(','.join(f"{p:.4f}" for p in pcts))
    
    
    print("\nAll outputs saved successfully!")

if __name__ == "__main__":
    main()
