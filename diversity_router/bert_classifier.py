#!/usr/bin/env python3
"""

Usage:
    # 18 models
    python diversity_router/bert_classifier.py --data wildchat --strategy list_all --n_epochs 3 --data_dir outputs_18_models --output_dir outputs_18_models --model_name google-bert/bert-base-uncased --exp "18_models_test" --soft_labels
    
"""
import datetime
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Tuple, Dict, List
from tqdm import tqdm
import logging
import random

# PyTorch and Transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Local imports
from best_overall_baseline import calc_best_overall_baseline_split
from utils import *

# Calc distribution divergence
from scipy.stats import entropy

# Set random seeds for reproducibility
def set_seed(seed=2025):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(2025)


def create_model(model_name: str, num_labels: int, hidden_dropout_prob: float = 0.1):
    """
    Initialize BERT model for sequence classification.

    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of candidate models (classes)
        hidden_dropout_prob: Dropout for regularization

    Returns:
        model: AutoModelForSequenceClassification
        tokenizer: AutoTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=hidden_dropout_prob
    )

    return model, tokenizer


def run_bert_classification_split(
    queries: List[str],
    label_indices: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str = "google-bert/bert-base-uncased",
    lr: float = 2e-5,
    n_epochs: int = 3,
    batch_size: int = 8,
    hidden_dropout_prob: float = 0.1,
    max_length: int = 512,
    device: str = 'cuda',
    use_soft_labels: bool = False,
    soft_labels: np.ndarray = None,
    test_mask: np.ndarray = None,
    model_cumsum_dict: Dict[str, np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, float, List[int], List[int], nn.Module]:
    """
    Run train/val/test split BERT classification with 80/10/10 split.

    Args:
        queries: List of query strings
        label_indices: Array of label indices (class labels)
        label_encoder: LabelEncoder for converting between labels and indices
        model_name: HuggingFace model identifier
        lr: Learning rate
        n_epochs: Number of training epochs
        batch_size: Batch size
        hidden_dropout_prob: Dropout probability
        max_length: Maximum sequence length
        device: Device to train on (cuda or cpu)
        use_soft_labels: Whether to use soft labels
        soft_labels: Soft labels array (if use_soft_labels=True)
        test_mask: Boolean mask for test indices (if provided, use instead of random split)
        model_cumsum_dict: Dict mapping model_name -> cumsum values 

    Returns:
        y_test: True labels for test samples
        y_pred: Predicted labels for test samples
        test_acc: Test accuracy
        train_val_indices: Combined train+val indices
        test_indices: Test indices
        model: Trained model
    """
    indices = np.arange(len(queries))

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

    # Split queries and labels
    train_queries = [queries[i] for i in train_idx]
    val_queries = [queries[i] for i in val_idx]
    test_queries = [queries[i] for i in test_idx]

    y_train = label_indices[train_idx]
    y_val = label_indices[val_idx]
    y_test = label_indices[test_idx]

    if use_soft_labels:
        assert soft_labels is not None, "Soft labels must be provided when use_soft_labels is True."
        y_train_soft = soft_labels[train_idx]
        y_val_soft = soft_labels[val_idx]
        y_test_soft = soft_labels[test_idx]
        print(f"Using soft labels for training.")

    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")

    # Initialize tokenizer and tokenize all queries
    model, tokenizer = create_model(model_name, len(label_encoder.classes_), hidden_dropout_prob)

    print(f"Tokenizing {len(queries)} queries...")
    train_inputs = tokenizer(
        train_queries,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    val_inputs = tokenizer(
        val_queries,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    test_inputs = tokenizer(
        test_queries,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    # Create dataloaders
    train_dataset = TensorDataset(
        train_inputs['input_ids'],
        train_inputs['attention_mask'],
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        eps=1e-8
    )

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * n_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Move model to device
    model.to(device)

    # Log model architecture, optimizer
    logging.info(f"BERT Classifier Architecture: {model_name}")
    logging.info(f"Optimizer: {optimizer}")

    # Training loop with validation monitoring
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 2  # Early stopping patience

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(n_epochs), desc="Training"):
        print(f"Epoch {epoch+1}/{n_epochs}")

        # Training phase
        model.train()
        epoch_train_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(
                input_ids=val_inputs['input_ids'].to(device),
                attention_mask=val_inputs['attention_mask'].to(device),
                labels=torch.tensor(y_val, dtype=torch.long).to(device)
            )
            val_loss = val_outputs.loss.item()
            val_losses.append(val_loss)

            # Calculate validation accuracy
            val_logits = val_outputs.logits
            val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_pred)

        print(f"  Epoch {epoch+1}/{n_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break


    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(
            input_ids=test_inputs['input_ids'].to(device),
            attention_mask=test_inputs['attention_mask'].to(device)
        )
        test_logits = test_outputs.logits
        y_pred = torch.argmax(test_logits, dim=1).cpu().numpy()
        test_acc = accuracy_score(y_test, y_pred)

        if use_soft_labels:
            # Calculate calibration metrics if soft labels are used
            y_pred_probs = F.softmax(test_logits, dim=1).cpu().numpy()
            kl_divergence = entropy(np.array(y_test_soft).T, np.array(y_pred_probs).T)
            print(f"Test KL Divergence = {np.average(kl_divergence):.4f}")

    print(f"Test Accuracy = {test_acc:.3f}")

    return y_test, y_pred, test_acc, train_idx.tolist(), test_idx.tolist(), model


def main():
    parser = argparse.ArgumentParser(description="BERT Classifier with Train/Val/Test Split")
    parser.add_argument('--data_dir', default='outputs', choices=['outputs', 'outputs_18_models'],
                        help='strategy dir where train/test data is stored')
    parser.add_argument('--output_dir', default='outputs_split', choices=['outputs_split', 'outputs_18_models'],
                        help='output directory for results')
    parser.add_argument('--data', default='longform_qa', help='Dataset to analyze')
    parser.add_argument('--strategy', default='list_all', choices=['list_all'],
                       help='Strategy to analyze')
    parser.add_argument('--model_name', default='google-bert/bert-base-uncased',
                       help='BERT model name from HuggingFace')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--no_cache', action='store_true',
                       help='Regenerate embeddings instead of using cache')
    parser.add_argument('--soft_labels', action='store_true',
                       help='Use soft labels instead of hard labels')
    parser.add_argument('--normalize_func', type=str, default='default',
                       help='Normalization function for soft labels (if any)')
    parser.add_argument('--subset', action='store_true',
                       help='Use a subset of data for wildchat')
    parser.add_argument('--exp', type=str, default='',
                       help='Experiment name for logging')
    parser.add_argument('--save_model', action='store_true',
                       help='Whether to save the trained model')
    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Get current timestamp
    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup paths
    base_dir = Path(__file__).parent.parent
    strategy_dir = base_dir / args.data_dir / args.data / args.strategy  # path w/ train/test data
    output_parent_dir = base_dir / args.output_dir / args.data / args.strategy
    data_path = strategy_dir / 'best_model_per_query.csv'

    output_dir = output_parent_dir / 'bert'

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create logs directory
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)

    # Logging file name
    model_name_safe = args.model_name.replace('/', '_')
    if not args.soft_labels:
        log_file = output_dir / 'logs' / f'{args.exp}_{model_name_safe}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}.log'
    else:
        log_file = output_dir / 'logs' / f'{args.exp}_{model_name_safe}_soft_labels_{args.normalize_func}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}.log'

    # Add current time to log file name
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

    # Load data and dataframe (needed for baseline calculation)
    df = pd.read_csv(data_path)
    if args.subset:
        df = df.sample(n=500, random_state=2025).reset_index(drop=True)

    if not args.soft_labels:
        queries, labels, query_ids, label_indices, label_encoder, model_cumsum_dict, model_unique_dict, model_quality_dict, model_unique_quality_dict = load_data(data_path, subset=args.subset)
    else:
        logging.info("Using soft labels for training.")
        normalize_func = NORMALIZE_NAME_TO_FUNC.get(args.normalize_func, normalize_default)
        queries, labels, query_ids, label_indices, label_encoder, model_cumsum_dict, soft_labels, model_unique_dict, model_quality_dict, model_unique_quality_dict = load_data_with_soft_labels(data_path, subset=args.subset, normalize_func=normalize_func)

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

    # Run train/val/test split BERT classification
    print(f"\nRunning train/val/test split BERT classification...")
    print(f"Model: {args.model_name}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")

    if args.soft_labels:
        y_true, y_pred, test_accuracy, train_val_indices, test_indices, model = run_bert_classification_split(
            queries, label_indices, label_encoder,
            model_name=args.model_name, lr=args.lr, n_epochs=args.n_epochs,
            batch_size=args.batch_size, hidden_dropout_prob=args.hidden_dropout_prob,
            max_length=args.max_length, device=args.device,
            use_soft_labels=True, soft_labels=soft_labels,
            test_mask=test_mask,
            model_cumsum_dict=model_cumsum_dict
        )
    else:
        y_true, y_pred, test_accuracy, train_val_indices, test_indices, model = run_bert_classification_split(
            queries, label_indices, label_encoder,
            model_name=args.model_name, lr=args.lr, n_epochs=args.n_epochs,
            batch_size=args.batch_size, hidden_dropout_prob=args.hidden_dropout_prob,
            max_length=args.max_length, device=args.device,
            test_mask=test_mask,
            model_cumsum_dict=model_cumsum_dict
        )

    # Save trained model
    if args.save_model:
        model_path = output_dir / 'models' / f'{args.exp}_{model_name_safe}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}_{cur_time}/model.pt'
        if not model_path.parent.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, model_path)
        logging.info(f"Trained model saved to {model_path}")
        # Save label classes as well
        label_classes_path = output_dir / 'models' / f'{args.exp}_{model_name_safe}_lr_{args.lr}_ep_{args.n_epochs}_bs_{args.batch_size}_{cur_time}/label_classes.npy'
        np.save(label_classes_path, label_encoder.classes_)
        # Save tokenizer alongside model
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer_path = model_path.parent / 'tokenizer'
        tokenizer.save_pretrained(tokenizer_path)
        logging.info(f"Tokenizer saved to {tokenizer_path}")

    # Calculate metrics (adapted for single evaluation)
    fold_accuracies = [test_accuracy]  # Single accuracy value
    results = calculate_metrics(y_true, y_pred, test_indices, fold_accuracies, label_encoder, -1, 1, model_cumsum_dict, model_unique_dict, model_quality_dict, model_unique_quality_dict)

    # Add overall best model baseline (using train/test split, no data leakage)
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

    # Add model configuration to results
    results['model_name'] = args.model_name
    results['learning_rate'] = args.lr
    results['n_epochs'] = args.n_epochs
    results['batch_size'] = args.batch_size
    results['hidden_dropout_prob'] = args.hidden_dropout_prob

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
    print(f"Model: {args.model_name}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"\nPer-Model Performance:")
    print(f"{'-'*70}")
    print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"{'-'*70}")
    for model_name in label_encoder.classes_:
        metrics = results['per_model_metrics'][model_name]
        print(f"{model_name:<25} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
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
    logging.info(f"BERT Classifier Test Accuracy: {test_accuracy*100:.2f}%")
    logging.info('-'*70)

    print("\nAll outputs saved successfully!")


if __name__ == "__main__":
    main()
