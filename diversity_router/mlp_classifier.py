#!/usr/bin/env python3
"""
Usage:

    # 18 models, unv
    python diversity_router/mlp_classifier.py \
        --data wildchat \
        --strategy list_all \
        --n_epochs 10 \
        --hidden_dim 512 \
        --soft_labels \
        --data_dir outputs_18_models \
        --output_dir outputs_18_models
        
    # 18 models, spec
    python diversity_router/mlp_classifier.py \
        --data wildchat \
        --strategy list_all \
        --n_epochs 10 \
        --hidden_dim 512 \
        --soft_labels \
        --data_dir outputs_18_models \
        --output_dir outputs_18_models \
        --per_model_enc_embed concat_truncated \
        --truncate_dim 200
        
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
from best_overall_baseline import  calc_best_overall_baseline_split

from utils import *

#use logging for documenting cumsum and accuracy for different hyperparameters
import logging

#calc distribution divergence
from scipy.stats import entropy

#fix torch random seed for reproducibility
torch.manual_seed(2025)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int=3584, hidden_dim: int=1024, output_dim: int=10):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        #initialize weights
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MLPClassifier3layers(nn.Module):
    def __init__(self, input_dim: int=3584, hidden_dim1: int=1024, hidden_dim2: int=512, output_dim: int=10):
        super(MLPClassifier3layers, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()

        #initialize weights
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='linear')

        #dropout layer
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def run_mlp_classification_split(embeddings: np.ndarray, label_indices: np.ndarray, label_encoder: LabelEncoder,
                           lr: float = 0.001, n_epochs: int = 50, batch_size: int = 32, hidden_dim: int = 1024,
                           use_soft_labels: bool = False, soft_labels: np.ndarray = None,
                           output_dir: Path = None, model_cumsum_dict: Dict[str, np.ndarray] = None, weight_decay: float = 0,
                           test_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, float, List[int], List[int], nn.Module, Dict[str, List[float]], Dict[str, List[float]]]:
    """Run train/val/test split MLP classification with 80/10/10 split.

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

    X_train, X_val, X_test = embeddings[train_idx], embeddings[val_idx], embeddings[test_idx]
    y_train, y_val, y_test = label_indices[train_idx], label_indices[val_idx], label_indices[test_idx]

    if use_soft_labels:
        assert soft_labels is not None, "Soft labels must be provided when use_soft_labels is True."
        y_train_soft, y_val_soft, y_test_soft = soft_labels[train_idx], soft_labels[val_idx], soft_labels[test_idx]
        print(f"Using soft labels for training.")

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Initialize model, loss function, and optimizer
    model = MLPClassifier(input_dim=X_train.shape[1], hidden_dim=hidden_dim, output_dim=len(label_encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
        )
    
    # Log model architecture, criterion, optimizer
    logging.info(f"MLP Classifier Architecture: {model}")
    logging.info(f"Loss Function: {criterion}")
    logging.info(f"Optimizer: {optimizer}")

    # Initialize loss tracking
    train_losses = {'model': []}
    val_losses = {'model': []}

    # Training loop with validation monitoring
    model.train()
    best_val_acc = 0.0

    for epoch in tqdm(range(n_epochs), desc="Training"):
        print(f"Epoch {epoch+1}/{n_epochs}")

        # Training
        permutation = torch.randperm(X_train_tensor.size()[0])
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        print(f"Average Training Loss = {avg_train_loss:.4f}")
        train_losses['model'].append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, val_pred = torch.max(val_outputs, 1)
            val_acc = accuracy_score(y_val, val_pred.numpy())

            val_loss = criterion(val_outputs, y_val_tensor).item()
            print(f"Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_acc:.3f}")
            val_losses['model'].append(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best validation accuracy: {best_val_acc:.3f}")

        model.train()

    # Plot loss curves
    if output_dir is not None:
        plot_loss_curves(train_losses, val_losses, output_dir)

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, y_pred = torch.max(test_outputs, 1)
        test_acc = accuracy_score(y_test, y_pred.numpy())

        if use_soft_labels:
            # Calculate calibration metrics if soft labels are used
            y_pred_probs = F.softmax(test_outputs, dim=1)
            kl_divergence = entropy(np.array(y_test_soft).T, np.array(y_pred_probs).T)
            print(f"Test KL Divergence = {np.average(kl_divergence):.4f}")

    print(f"Test Accuracy = {test_acc:.3f}")

    return np.array(y_test), y_pred.numpy(), test_acc, train_val_idx.tolist(), test_idx.tolist(), model, train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(description="MLP Classifier with Train/Val/Test Split")
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
    parser.add_argument('--additional_features', action='store_true',
                       help='Use additional features along with embeddings')
    parser.add_argument('--additional_feature_path', type=str, default=None,
                       help='Path to additional features file (if any)')
    parser.add_argument('--additional_feature_dim', type=int, default=20,
                       help='Dimension of additional features (if any)')
    parser.add_argument('--subset', action='store_true',
                       help='Use a subset of data for wildchat')
    parser.add_argument('--normalize_func', type=str, default='default',
                       help='Normalization function for soft labels (if any)')
    parser.add_argument('--exp', type=str, default='',
                       help='Experiment name for logging')
    parser.add_argument('--save_model', action='store_true',
                       help='Whether to save the trained model')
    parser.add_argument('--weight_decay', type=float, default=0,
                       help='Weight decay (L2 regularization) for optimizer')
    parser.add_argument('--per_model_enc_embed', type=str, default=None,
                       choices=['concat_all', 'concat_truncated'],
                       help='Use per-model encodings concatenated as embedding instead of SentenceTransformer')
    parser.add_argument('--truncate_dim', type=int, default=200,
                       help='Number of dimensions to keep per model (only used with concat_truncated)')
    args = parser.parse_args()
    
    #get current timestamp
    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup paths
    base_dir = Path(__file__).parent.parent
    strategy_dir = base_dir / args.data_dir / args.data / args.strategy # path w/ train/test data
    output_parent_dir =  base_dir / args.output_dir / args.data / args.strategy
    data_path = strategy_dir / 'best_model_per_query.csv'

    output_dir = output_parent_dir / 'mlp'
    cache_path = strategy_dir / 'query_embeddings.npy'  # Cache at strategy level

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create logs directory
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)

    # Logging file name
    if not args.soft_labels:
        log_file = output_dir / 'logs' / f'{args.exp}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}_hd_{args.hidden_dim}.log'
    else:
        log_file = output_dir / 'logs' / f'{args.exp}_soft_labels_{args.normalize_func}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}_hd_{args.hidden_dim}.log'

    if args.additional_features:
        log_file = log_file.stem + f"_add_dim_{args.additional_feature_dim}" + '_with_additional_features.log'
        log_file = output_dir / 'logs' / log_file

    # Add weight decay to log file name if non-zero
    if args.weight_decay > 0:
        log_file = log_file.stem + f'_wd_{args.weight_decay}.log'
        log_file = output_dir / 'logs' / log_file

    # Add per_model_enc_embed mode to log file name
    if args.per_model_enc_embed:
        suffix = args.per_model_enc_embed
        if args.per_model_enc_embed == 'concat_truncated':
            suffix += f'_dim{args.truncate_dim}'
        log_file = log_file.stem + f'_enc_{suffix}.log'
        log_file = output_dir / 'logs' / log_file

    #add current time to log file name
    log_file = log_file.stem + f"_{cur_time}.log"
    log_file = output_dir / 'logs' / log_file
    
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

    # mask df
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
    if args.data == 'inf_chat_2k':
        # Use infchat_test_mask to identify inf_chat's 20% test queries in wild_inf_chat,
        # then locate them in inf_chat_2k by query_id
        wic_dir = base_dir / args.data_dir / 'wild_inf_chat' / args.strategy
        infchat_test_mask_path = wic_dir / 'infchat_test_mask.npy'
        wic_csv_path = wic_dir / 'best_model_per_query.csv'
        if infchat_test_mask_path.exists() and wic_csv_path.exists():
            infchat_test_mask_arr = np.load(infchat_test_mask_path)
            wic_df = pd.read_csv(wic_csv_path)
            query_ids_in_wic = wic_df['query_id'].tolist()
            infchat_test_qids = [query_ids_in_wic[i] for i in range(len(infchat_test_mask_arr)) if infchat_test_mask_arr[i]]
            
            query_ids = df['query_id'].tolist()
            test_mask = np.array([qid in infchat_test_qids for qid in query_ids])
            print(f"inf_chat_2k test mask: {test_mask.sum()} test (inf_chat 20% split) / {(~test_mask).sum()} train+val")
            
        else:
            print(f"WARNING: infchat_test_mask or wild_inf_chat CSV not found at {wic_dir}, falling back to random split.")

    if not args.soft_labels:
        queries, labels, query_ids, label_indices, label_encoder, model_cumsum_dict, model_unique_dict, model_quality_dict, model_unique_quality_dict = load_data(data_path, subset=args.subset)
    else:
        logging.info("Using soft labels for training.")
        normalize_func = NORMALIZE_NAME_TO_FUNC.get(args.normalize_func, normalize_default)
        queries, labels, query_ids, label_indices, label_encoder, model_cumsum_dict, soft_labels, model_unique_dict, model_quality_dict, model_unique_quality_dict = load_data_with_soft_labels(data_path, subset=args.subset, normalize_func=normalize_func)

    # Generate embeddings
    if args.per_model_enc_embed:
        embeddings = load_per_model_encodings_as_embeddings(
            base_dir=base_dir,
            data_dir=args.data_dir,
            dataset=args.data,
            strategy=args.strategy,
            model_names=list(label_encoder.classes_),
            n_samples=len(queries),
            mode=args.per_model_enc_embed,
            truncate_dim=args.truncate_dim,
        )
    else:
        embeddings = generate_embeddings(queries, cache_path, use_cache=not args.no_cache)

    

    

    if args.additional_features and args.additional_feature_path is not None:
        print(f"Loading additional features from {args.additional_feature_path}...")
        additional_features = np.load(args.additional_feature_path)[:,:args.additional_feature_dim]
        print(f"Additional features shape: {additional_features.shape}")
        # Concatenate embeddings with additional features
        embeddings = np.concatenate((embeddings, additional_features), axis=1)
        print(f"Combined embeddings shape: {embeddings.shape}")

    # Run train/val/test split MLP classification
    print(f"\nRunning train/val/test split MLP classification...")
    if args.soft_labels:
        y_true, y_pred, test_accuracy, train_val_indices, test_indices, model, train_losses, val_losses = run_mlp_classification_split(
            embeddings, label_indices, label_encoder,
            lr=args.lr, n_epochs=args.n_epochs, batch_size=args.batch_size, hidden_dim=args.hidden_dim,
            use_soft_labels=True, soft_labels=soft_labels,
            output_dir=output_dir, model_cumsum_dict=model_cumsum_dict,
            weight_decay=args.weight_decay,
            test_mask=test_mask
        )
    else:
        y_true, y_pred, test_accuracy, train_val_indices, test_indices, model, train_losses, val_losses = run_mlp_classification_split(
            embeddings, label_indices, label_encoder,
            lr=args.lr, n_epochs=args.n_epochs, batch_size=args.batch_size, hidden_dim=args.hidden_dim,
            output_dir=output_dir, model_cumsum_dict=model_cumsum_dict,
            weight_decay=args.weight_decay,
            test_mask=test_mask
        )
        
    # debug save test query ids
    test_query_ids = [query_ids[i] for i in test_indices]
    test_query_ids_path = output_dir / 'test_query_ids.npy'
    np.save(test_query_ids_path, test_query_ids)
        
    # Save trained model
    if args.save_model:
        model_dir_name = f'{args.exp}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}_hd_{args.hidden_dim}'

        # Add weight decay to directory name if non-zero
        if args.weight_decay > 0:
            model_dir_name += f'_wd_{args.weight_decay}'

        # Add per_model_enc_embed mode to directory name
        if args.per_model_enc_embed:
            suffix = args.per_model_enc_embed
            print(f"Using per-model encoding mode: {suffix}")
            if args.per_model_enc_embed == 'concat_truncated':
                suffix += f'_dim{args.truncate_dim}'
            model_dir_name += f'_enc_{suffix}'

        model_dir_name += f'_{cur_time}'
        model_path = output_dir / 'models' / model_dir_name / 'model.pt'
        if not model_path.parent.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, model_path)
        logging.info(f"Trained model saved to {model_path}")
        #save label classes as well
        label_classes_path = output_dir / 'models' / model_dir_name / 'label_classes.npy'
        np.save(label_classes_path, label_encoder.classes_)

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

    # Print summary
    print(f"\n{'='*70}")
    print("In domain test results")
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
    logging.info(f"MLP Classifier Test Accuracy: {test_accuracy*100:.2f}%")
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

    print("\nAll outputs saved successfully!")

if __name__ == "__main__":
    main()
