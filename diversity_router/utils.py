
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
import umap
import torch.nn as nn
import torch

# Embeddings
from sentence_transformers import SentenceTransformer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

_MASK_BASE = Path(__file__).parent.parent / 'outputs_18_models' / 'wild_inf_chat' / 'list_all'
MASK_MAP = {
    'train':
        {
            'wildchat': str(_MASK_BASE / 'wildchat_train_mask_hard.npy'),
            'inf_chat': str(_MASK_BASE / 'infchat_train_mask.npy'),
            'wild_inf_chat': str(_MASK_BASE / 'wild_inf_chat_train_mask.npy'),
        },
    'test':
        {
            'wildchat': str(_MASK_BASE / 'wildchat_test_mask_hard.npy'),
            'inf_chat': str(_MASK_BASE / 'infchat_test_mask.npy'),
            'wild_inf_chat': str(_MASK_BASE / 'wild_inf_chat_test_mask.npy'),
        }
}


def _load_auxiliary_metrics(csv_path: Path, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load auxiliary metrics (unique_answer, answer_quality, and unique_quality) from CSV files in the same directory.

    Args:
        csv_path: Path to the main CSV file (best_model_per_query.csv)
        df: Already loaded DataFrame (to get model names from cumsum columns)

    Returns:
        model_unique_dict: Dict mapping model_name -> unique answer counts (distinctness)
        model_quality_dict: Dict mapping model_name -> quality scores
        model_unique_quality_dict: Dict mapping model_name -> unique quality scores
    """
    parent_dir = csv_path.parent

    # Get model names from cumsum columns
    cumsum_cols = [col for col in df.columns if col.endswith('_cumsum')]
    model_names = [col.replace('_cumsum', '') for col in cumsum_cols]

    model_unique_dict = {}
    model_quality_dict = {}
    model_unique_quality_dict = {}

    # Load distinctness_per_query.csv (unique answers)
    distinctness_path = parent_dir / 'distinctness_per_query.csv'
    if distinctness_path.exists():
        distinctness_df = pd.read_csv(distinctness_path)
        # Columns are model names directly (e.g., 'llama-3.2-1b')
        for model_name in model_names:
            if model_name in distinctness_df.columns:
                model_unique_dict[model_name] = distinctness_df[model_name].values
        print(f"Loaded distinctness metrics for {len(model_unique_dict)} models from {distinctness_path}")
    else:
        print(f"Warning: distinctness_per_query.csv not found at {distinctness_path}")

    # Load quality_per_query.csv
    quality_path = parent_dir / 'quality_per_query.csv'
    if quality_path.exists():
        quality_df = pd.read_csv(quality_path)
        # Columns are model names directly (e.g., 'llama-3.2-1b')
        for model_name in model_names:
            if model_name in quality_df.columns:
                model_quality_dict[model_name] = quality_df[model_name].values
        print(f"Loaded quality metrics for {len(model_quality_dict)} models from {quality_path}")
    else:
        print(f"Warning: quality_per_query.csv not found at {quality_path}")

    # Load avg_unique_quality_per_query.csv
    unique_quality_path = parent_dir / 'avg_unique_quality_per_query.csv'
    if unique_quality_path.exists():
        unique_quality_df = pd.read_csv(unique_quality_path)
        for model_name in model_names:
            if model_name in unique_quality_df.columns:
                model_unique_quality_dict[model_name] = unique_quality_df[model_name].values
        print(f"Loaded unique_quality metrics for {len(model_unique_quality_dict)} models from {unique_quality_path}")
    else:
        print(f"Warning: avg_unique_quality_per_query.csv not found at {unique_quality_path}")

    return model_unique_dict, model_quality_dict, model_unique_quality_dict


def load_data(csv_path: Path, subset = False) -> Tuple[List[str], List[str], List[str], np.ndarray, LabelEncoder, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load query data from CSV file.
    The labels here are hard labels
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    #sample if subset is True
    if subset:
        print("Using subset of data for wildchat...")
        df = df.sample(n=500, random_state=2025).reset_index(drop=True)

    #assert len(df) == 92, f"Expected 92 queries, got {len(df)}"
    assert df['query_content'].notna().all(), "Found missing query content"
    assert df['best_model_name'].notna().all(), "Found missing model labels"

    queries = df['query_content'].tolist()
    labels = df['best_model_name'].tolist()
    query_ids = df['query_id'].tolist()

    # Extract cumsum scores for all models
    cumsum_cols = [col for col in df.columns if col.endswith('_cumsum')]
    model_names = sorted([col.replace('_cumsum', '') for col in cumsum_cols])
    model_cumsum_dict = {}
    for col in cumsum_cols:
        model_name = col.replace('_cumsum', '')
        model_cumsum_dict[model_name] = df[col].values

    # Encode labels to integers using all model names (not just best models)
    label_encoder = LabelEncoder()
    label_encoder.fit(model_names)  # Fit on all models
    label_indices = label_encoder.transform(labels)  # Transform best model labels

    # Load auxiliary metrics (unique answers, quality, and unique_quality)
    model_unique_dict, model_quality_dict, model_unique_quality_dict = _load_auxiliary_metrics(csv_path, df)

    print(f"Loaded {len(queries)} queries with {len(label_encoder.classes_)} unique models")
    return queries, labels, query_ids, label_indices, label_encoder, model_cumsum_dict, model_unique_dict, model_quality_dict, model_unique_quality_dict

def normalize_default(scores: np.ndarray) -> np.ndarray:
    """Normalize scores using default softmax normalization."""
    
    soft_labels = scores / scores.sum(axis=1, keepdims=True)
    return soft_labels

def normalize_regret_based(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Normalize scores using regret-based normalization."""
    adv = scores - scores.max(axis=1, keepdims=True)

    # Apply temperature-scaled softmax
    T = temperature
    exp = np.exp(adv / T)
    soft_labels = exp / exp.sum(axis=1, keepdims=True)
    return soft_labels

def normalize_max(scores: np.ndarray) -> np.ndarray:
    """Normalize scores using min-max normalization."""
    
    max_vals = scores.max(axis=1, keepdims=True)
    normalized = scores / (max_vals + 1e-10)
    return normalized
def normalize_within_model(scores: np.ndarray) -> np.ndarray:
    # z-score normalization within each model
    mean = scores.mean(axis=0, keepdims=True)
    std = scores.std(axis=0, keepdims=True) + 1e-10
    normalized = (scores - mean) / std
    # then apply softmax
    exp = np.exp(normalized)
    soft_labels = exp / exp.sum(axis=1, keepdims=True)
    return soft_labels

def normalize_centered_softmax(scores: np.ndarray, temperature: float = 10.0) -> np.ndarray:
    """
    Normalize scores using centered softmax with large temperature.

    Args:
        scores: Raw cumulative utility scores, shape (n_queries, n_models)
        temperature: Softmax temperature parameter (larger = smoother distribution)

    Returns:
        Soft labels as probability distribution, shape (n_queries, n_models)
    """
    # Center scores by subtracting mean across models for each query
    centered = scores - scores.mean(axis=1, keepdims=True)

    # Apply temperature-scaled softmax
    exp = np.exp(centered / temperature)
    soft_labels = exp / exp.sum(axis=1, keepdims=True)

    return soft_labels

NORMALIZE_NAME_TO_FUNC = {
    'default': normalize_default,
    'regret_based': normalize_regret_based,
    'max': normalize_max,
    'within_model': normalize_within_model,
    'centered_softmax': normalize_centered_softmax
}

def load_data_with_soft_labels(csv_path: Path, subset = False, normalize_func=normalize_default) -> Tuple[List[str], List[np.ndarray], List[str], np.ndarray, LabelEncoder, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load query data from CSV file with soft labels.
    The labels here are soft labels (weighted by cumsum)
    """
    print(f"Loading data from {csv_path} with soft labels...")
    df = pd.read_csv(csv_path)

    # sample if subset is True
    if subset:
        print("Using subset of data for wildchat...")
        df = df.sample(n=500, random_state=2025).reset_index(drop=True)

    #assert len(df) == 92, f"Expected 92 queries, got {len(df)}"
    assert df['query_content'].notna().all(), "Found missing query content"

    queries = df['query_content'].tolist()
    query_ids = df['query_id'].tolist()

    # Extract cumsum columns and sort model names alphabetically to match LabelEncoder
    cumsum_cols = [col for col in df.columns if col.endswith('_cumsum')]
    model_names = sorted([col.replace('_cumsum', '') for col in cumsum_cols])

    # Reorder soft labels to match alphabetically sorted model names
    labels = df['best_model_name'].tolist()
    sorted_cumsum_cols = [f'{name}_cumsum' for name in model_names]
    soft_labels = df[sorted_cumsum_cols].values # shape (n_samples, n_models) in alphabetical order
    print(soft_labels.shape)
    # normalize soft labels to sum to 1
    # soft_labels = soft_labels / soft_labels.sum(axis=1, keepdims=True)
    soft_labels = normalize_func(soft_labels)
    print(soft_labels.shape)
    # the first row
    print(f"First row soft labels: {soft_labels[0]}")
    print(f"sum of first row soft labels: {soft_labels[0].sum()}")

    # Load hard labels for label encoding using all model names
    label_encoder = LabelEncoder()
    label_encoder.fit(model_names)  # Fit on all models from cumsum columns
    label_indices = label_encoder.transform(labels)  # Transform best model labels

    # Verify model_names match label_encoder.classes_ order
    print(f"Model names (sorted): {model_names}")
    print(f"Label encoder classes: {label_encoder.classes_.tolist()}")
    assert model_names == label_encoder.classes_.tolist(), \
        f"Model names order mismatch! model_names={model_names}, label_encoder.classes_={label_encoder.classes_.tolist()}"

    # Extract cumsum scores for all models
    cumsum_cols = [col for col in df.columns if col.endswith('_cumsum')]
    model_cumsum_dict = {}
    for col in cumsum_cols:
        model_name = col.replace('_cumsum', '')
        model_cumsum_dict[model_name] = df[col].values

    # Load auxiliary metrics (unique answers, quality, and unique_quality)
    model_unique_dict, model_quality_dict, model_unique_quality_dict = _load_auxiliary_metrics(csv_path, df)

    print(f"Loaded {len(queries)} queries with soft labels")
    return queries, labels, query_ids, label_indices, label_encoder, model_cumsum_dict, soft_labels, model_unique_dict, model_quality_dict, model_unique_quality_dict


def generate_embeddings(queries: List[str], cache_path: Path, use_cache: bool = True, embedding_model: str = "infly/inf-retriever-v1") -> np.ndarray:
    """Generate or load cached query embeddings."""
    if use_cache and cache_path.exists() and embedding_model == "infly/inf-retriever-v1":
        print(f"Loading cached embeddings from {cache_path}...")
        embeddings = np.load(cache_path)
        print(f"Loaded embeddings with shape {embeddings.shape}")
        return embeddings

    print(f"Generating embeddings using {embedding_model}...")
    encoder = SentenceTransformer(embedding_model, trust_remote_code=True)
    embeddings = encoder.encode(queries, show_progress_bar=True, convert_to_numpy=True)

    # Verify no NaN values
    assert not np.isnan(embeddings).any(), "NaN values found in embeddings"

    # Save cache
    if embedding_model != "infly/inf-retriever-v1":
        cache_path = cache_path.parent / f"{cache_path.stem}_{embedding_model.replace('/', '_')}.npy"
    print(f"Saving embeddings to {cache_path}...")
    np.save(cache_path, embeddings)
    print(f"Generated embeddings with shape {embeddings.shape}")

    return embeddings


def load_per_model_encodings_as_embeddings(
    base_dir: Path,
    data_dir: str,
    dataset: str,
    strategy: str,
    model_names: List[str],
    n_samples: int,
    mode: str = 'concat_all',
    truncate_dim: int = 200,
) -> np.ndarray:
    """Load per-model query encodings and concatenate them into a single embedding per query.

    Args:
        base_dir: Base directory for data files.
        data_dir: Data directory name (e.g., 'outputs_18_models').
        dataset: Dataset name (e.g., 'wildchat').
        strategy: Strategy name (e.g., 'list_all').
        model_names: List of model names to load encodings for.
        n_samples: Expected number of samples.
        mode: 'concat_all' to concatenate full encodings, or 'concat_truncated'
              to take first truncate_dim dimensions from each model.
        truncate_dim: Number of dimensions to keep per model (only used with 'concat_truncated').

    Returns:
        np.ndarray of shape (n_samples, total_dim) where total_dim depends on mode.
    """
    import torch.nn.functional as F

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

    all_encodings = []
    for model_name in model_names:
        model_dir = encodings_base_dir / model_name
        enc_path = model_dir / 'query_encode.npy'
        if not enc_path.exists():
            raise FileNotFoundError(f"Encoding file not found: {enc_path}")

        encodings = np.load(enc_path)
        if encodings.shape[0] != n_samples:
            raise ValueError(
                f"Encoding shape mismatch for {model_name}: "
                f"expected {n_samples} samples, got {encodings.shape[0]}"
            )

        # L2 normalization
        encodings = F.normalize(torch.tensor(encodings), p=2, dim=1).numpy()

        if mode == 'concat_truncated':
            encodings = encodings[:, :truncate_dim]

        all_encodings.append(encodings)

    result = np.concatenate(all_encodings, axis=1)
    print(f"Per-model encoding embeddings: mode={mode}, shape={result.shape}")
    return result


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, test_idx:np.ndarray,fold_accuracies: List[float],
                     label_encoder: LabelEncoder, k: int, n_folds: int, model_cumsum_dict: Dict[str, np.ndarray],
                     model_unique_dict: Optional[Dict[str, np.ndarray]] = None,
                     model_quality_dict: Optional[Dict[str, np.ndarray]] = None,
                     model_unique_quality_dict: Optional[Dict[str, np.ndarray]] = None) -> Dict:
    """Calculate all evaluation metrics."""
    # Overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)

    # Cumsum metrics
    gt_cumsum_values = []
    pred_cumsum_values = []
    print(f"y_true:{len(y_true)}, y_pred:{len(y_pred)}, test_idx:{len(test_idx)}")
    for true_idx, pred_idx, orig_idx in zip(y_true, y_pred, test_idx):
        true_model = label_encoder.inverse_transform([true_idx])[0]
        pred_model = label_encoder.inverse_transform([pred_idx])[0]

        gt_cumsum_values.append(float(model_cumsum_dict[true_model][orig_idx]))
        pred_cumsum_values.append(float(model_cumsum_dict[pred_model][orig_idx]))
    random_cumsum_values = []
    for orig_idx in test_idx:
        # Get cumsum scores for all models for this query
        all_model_cumsums = [model_cumsum_dict[model][orig_idx] for model in model_cumsum_dict.keys()]
        random_cumsum_values.append(float(np.mean(all_model_cumsums)))
    cumsum_metrics = {
        'ground_truth_cumsum_mean': float(np.mean(gt_cumsum_values)),
        'predicted_cumsum_mean': float(np.mean(pred_cumsum_values)),
        'random_cumsum_mean': float(np.mean(random_cumsum_values))
    }

    # Unique answer metrics (if available)
    unique_metrics = None
    if model_unique_dict:
        gt_unique_values = []
        pred_unique_values = []
        random_unique_values = []
        for true_idx, pred_idx, orig_idx in zip(y_true, y_pred, test_idx):
            true_model = label_encoder.inverse_transform([true_idx])[0]
            pred_model = label_encoder.inverse_transform([pred_idx])[0]
            if true_model in model_unique_dict and pred_model in model_unique_dict:
                gt_unique_values.append(float(model_unique_dict[true_model][orig_idx]))
                pred_unique_values.append(float(model_unique_dict[pred_model][orig_idx]))
        for orig_idx in test_idx:
            all_model_uniques = [model_unique_dict[model][orig_idx] for model in model_unique_dict.keys()]
            random_unique_values.append(float(np.mean(all_model_uniques)))
        if gt_unique_values:
            unique_metrics = {
                'ground_truth_unique_mean': float(np.mean(gt_unique_values)),
                'predicted_unique_mean': float(np.mean(pred_unique_values)),
                'random_unique_mean': float(np.mean(random_unique_values))
            }

    # Quality metrics (if available)
    quality_metrics = None
    if model_quality_dict:
        gt_quality_values = []
        pred_quality_values = []
        random_quality_values = []
        for true_idx, pred_idx, orig_idx in zip(y_true, y_pred, test_idx):
            true_model = label_encoder.inverse_transform([true_idx])[0]
            pred_model = label_encoder.inverse_transform([pred_idx])[0]
            if true_model in model_quality_dict and pred_model in model_quality_dict:
                gt_quality_values.append(float(model_quality_dict[true_model][orig_idx]))
                pred_quality_values.append(float(model_quality_dict[pred_model][orig_idx]))
        for orig_idx in test_idx:
            all_model_qualities = [model_quality_dict[model][orig_idx] for model in model_quality_dict.keys()]
            random_quality_values.append(float(np.mean(all_model_qualities)))
        if gt_quality_values:
            quality_metrics = {
                'ground_truth_quality_mean': float(np.mean(gt_quality_values)),
                'predicted_quality_mean': float(np.mean(pred_quality_values)),
                'random_quality_mean': float(np.mean(random_quality_values))
            }

    # Unique quality metrics (if available)
    unique_quality_metrics = None
    if model_unique_quality_dict:
        gt_unique_quality_values = []
        pred_unique_quality_values = []
        random_unique_quality_values = []
        for true_idx, pred_idx, orig_idx in zip(y_true, y_pred, test_idx):
            true_model = label_encoder.inverse_transform([true_idx])[0]
            pred_model = label_encoder.inverse_transform([pred_idx])[0]
            if true_model in model_unique_quality_dict and pred_model in model_unique_quality_dict:
                gt_unique_quality_values.append(float(model_unique_quality_dict[true_model][orig_idx]))
                pred_unique_quality_values.append(float(model_unique_quality_dict[pred_model][orig_idx]))
        for orig_idx in test_idx:
            all_model_unique_qualities = [model_unique_quality_dict[model][orig_idx] for model in model_unique_quality_dict.keys()]
            random_unique_quality_values.append(float(np.mean(all_model_unique_qualities)))
        if gt_unique_quality_values:
            unique_quality_metrics = {
                'ground_truth_unique_quality_mean': float(np.mean(gt_unique_quality_values)),
                'predicted_unique_quality_mean': float(np.mean(pred_unique_quality_values)),
                'random_unique_quality_mean': float(np.mean(random_unique_quality_values))
            }

    # Per-model metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(label_encoder.classes_)),
        zero_division=0
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(label_encoder.classes_)))

    # Build results dictionary
    results = {
        'overall_accuracy': float(overall_accuracy),
        'mean_fold_accuracy': float(np.mean(fold_accuracies)),
        'std_fold_accuracy': float(np.std(fold_accuracies)),
        'fold_accuracies': [float(acc) for acc in fold_accuracies],
        'n_samples': int(len(y_true)),
        'n_folds': int(n_folds),
        'n_neighbors': int(k),
        'distance_metric': 'cosine',
        'model_names': label_encoder.classes_.tolist(),
        'confusion_matrix': conf_matrix.tolist(),
        'per_model_metrics': {},
        'cumsum_metrics': cumsum_metrics,
        'unique_metrics': unique_metrics,
        'quality_metrics': quality_metrics,
        'unique_quality_metrics': unique_quality_metrics
    }

    for idx, model in enumerate(label_encoder.classes_):
        results['per_model_metrics'][model] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1_score': float(f1[idx]),
            'support': int(support[idx])
        }

    return results


def create_predictions_df(query_ids: List[str], queries: List[str], y_true: np.ndarray,
                         y_pred: np.ndarray, label_encoder: LabelEncoder,
                         test_indices: List[int], model_cumsum_dict: Dict[str, np.ndarray],
                         model_unique_dict: Optional[Dict[str, np.ndarray]] = None,
                         model_quality_dict: Optional[Dict[str, np.ndarray]] = None,
                         model_unique_quality_dict: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """Create DataFrame with predictions and cumsum values."""
    true_labels_test = label_encoder.inverse_transform(y_true)
    pred_labels = label_encoder.inverse_transform(y_pred)
    correct = (y_true == y_pred).tolist()

    # Get cumsum values
    gt_cumsum = [model_cumsum_dict[model][idx] for model, idx in zip(true_labels_test, test_indices)]
    pred_cumsum = [model_cumsum_dict[model][idx] for model, idx in zip(pred_labels, test_indices)]

    query_ids_test = [query_ids[idx] for idx in test_indices]
    queries_test = [queries[idx] for idx in test_indices]

    df = pd.DataFrame({
        'query_id': query_ids_test,
        'query_content': queries_test,
        'true_model': true_labels_test,
        'predicted_model': pred_labels,
        'correct': correct,
        'true_model_cumsum': gt_cumsum,
        'predicted_model_cumsum': pred_cumsum
    })

    # Add unique answer columns if available
    if model_unique_dict:
        gt_unique = [model_unique_dict[model][idx] if model in model_unique_dict else None
                     for model, idx in zip(true_labels_test, test_indices)]
        pred_unique = [model_unique_dict[model][idx] if model in model_unique_dict else None
                       for model, idx in zip(pred_labels, test_indices)]
        df['true_model_unique'] = gt_unique
        df['predicted_model_unique'] = pred_unique

    # Add quality columns if available
    if model_quality_dict:
        gt_quality = [model_quality_dict[model][idx] if model in model_quality_dict else None
                      for model, idx in zip(true_labels_test, test_indices)]
        pred_quality = [model_quality_dict[model][idx] if model in model_quality_dict else None
                        for model, idx in zip(pred_labels, test_indices)]
        df['true_model_quality'] = gt_quality
        df['predicted_model_quality'] = pred_quality

    # Add unique_quality columns if available
    if model_unique_quality_dict:
        gt_unique_quality = [model_unique_quality_dict[model][idx] if model in model_unique_quality_dict else None
                             for model, idx in zip(true_labels_test, test_indices)]
        pred_unique_quality = [model_unique_quality_dict[model][idx] if model in model_unique_quality_dict else None
                               for model, idx in zip(pred_labels, test_indices)]
        df['true_model_unique_quality'] = gt_unique_quality
        df['predicted_model_unique_quality'] = pred_unique_quality

    return df


def create_tsne_plot(embeddings: np.ndarray, label_indices: np.ndarray,
                     label_encoder: LabelEncoder, output_dir: Path):
    """Generate t-SNE visualization."""
    print("Generating t-SNE visualization...")

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=2025, perplexity=30,metric='cosine',max_iter=2000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Set style matching create_topk_plots.py
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
    plt.rcParams.update({'font.size': 15})

    # Create plot
    plt.figure(figsize=(12, 10))
    colors = sns.color_palette("husl", len(label_encoder.classes_))

    for idx, model in enumerate(label_encoder.classes_):
        mask = label_indices == idx
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors[idx]], label=model, alpha=0.7, s=100,
                   edgecolors='white', linewidth=1)

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('Query Embeddings Colored by Best Model (t-SNE)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save both PNG and PDF
    plt.savefig(output_dir / 'tsne_embeddings.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'tsne_embeddings.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved t-SNE plot to {output_dir / 'tsne_embeddings.png'}")


def create_umap_plot(embeddings: np.ndarray, label_indices: np.ndarray,
                     label_encoder: LabelEncoder, output_dir: Path):
    """Generate UMAP visualization."""
    print("Generating UMAP visualization...")

    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=2025)
    embeddings_2d = reducer.fit_transform(embeddings)

    # Set style
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
    plt.rcParams.update({'font.size': 15})

    # Create plot
    plt.figure(figsize=(12, 10))
    colors = sns.color_palette("husl", len(label_encoder.classes_))

    for idx, model in enumerate(label_encoder.classes_):
        mask = label_indices == idx
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors[idx]], label=model, alpha=0.7, s=100,
                   edgecolors='white', linewidth=1)

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('Query Embeddings Colored by Best Model (UMAP)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save both PNG and PDF
    plt.savefig(output_dir / 'umap_embeddings.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'umap_embeddings.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved UMAP plot to {output_dir / 'umap_embeddings.png'}")


def save_confusion_matrix_csv(conf_matrix: np.ndarray, label_encoder: LabelEncoder,
                              output_dir: Path):
    """Save confusion matrix as CSV."""
    print("Saving confusion matrix to CSV...")

    conf_matrix_array = np.array(conf_matrix)

    # Create DataFrame with model names as index and columns
    df = pd.DataFrame(
        conf_matrix_array,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )

    # Add row/column labels
    df.index.name = 'True Model'
    df.columns.name = 'Predicted Model'

    # Save to CSV
    csv_path = output_dir / 'confusion_matrix.csv'
    df.to_csv(csv_path)
    print(f"Saved confusion matrix to {csv_path}")


def create_confusion_matrix_plot(conf_matrix: np.ndarray, label_encoder: LabelEncoder,
                                 output_dir: Path):
    """Generate confusion matrix heatmap."""
    print("Generating confusion matrix plot...")

    conf_matrix_array = np.array(conf_matrix)

    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
    plt.rcParams.update({'font.size': 15})

    plt.figure(figsize=(14, 12))

    # Normalize for better visualization
    conf_matrix_normalized = conf_matrix_array.astype('float') / (conf_matrix_array.sum(axis=1)[:, np.newaxis] + 1e-10)

    sns.heatmap(conf_matrix_normalized, annot=conf_matrix_normalized, fmt='.2f',
                cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cbar_kws={'label': 'Normalized Frequency'},
                linewidths=0.5, linecolor='gray')

    plt.xlabel('Predicted Model')
    plt.ylabel('True Model')
    plt.title(f'Confusion Matrix: 10-Fold CV Classifier')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'confusion_matrix.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved confusion matrix plot to {output_dir / 'confusion_matrix.png'}")


def save_per_model_metrics_csv(results: Dict, label_encoder: LabelEncoder, output_dir: Path):
    """Save per-model metrics as CSV."""
    print("Saving per-model metrics to CSV...")

    # Extract metrics for each model
    models = label_encoder.classes_
    data = {
        'model': models,
        'precision': [results['per_model_metrics'][m]['precision'] for m in models],
        'recall': [results['per_model_metrics'][m]['recall'] for m in models],
        'f1_score': [results['per_model_metrics'][m]['f1_score'] for m in models],
        'support': [results['per_model_metrics'][m]['support'] for m in models]
    }

    df = pd.DataFrame(data)

    # Save to CSV
    csv_path = output_dir / 'per_model_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved per-model metrics to {csv_path}")


def save_results(results: Dict, predictions_df: pd.DataFrame, output_dir: Path):
    """Save all results to output directory."""
    # Save metrics JSON
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")

    # Save predictions CSV
    predictions_path = output_dir / 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")


def plot_loss_curves(
    train_losses: Dict[str, List[float]],
    val_losses: Dict[str, List[float]],
    output_dir: Path,
    title_suffix: str = ""
):
    """
    Plot and save training/validation loss curves in one figure with subplots.

    Args:
        train_losses: Dict mapping model_name -> list of training losses per epoch
        val_losses: Dict mapping model_name -> list of validation losses per epoch
        output_dir: Directory to save the plots
        title_suffix: Optional suffix for plot title
    """
    print("Generating combined loss curves with subplots...")

    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'serif']
    plt.rcParams.update({'font.size': 15})

    model_names = sorted(train_losses.keys())
    n_models = len(model_names)
    if n_models == 0:
        print("No training losses found; skipping loss plot generation.")
        return

    # Use a compact grid up to 3 columns
    n_cols = min(3, n_models)
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.5 * n_cols, 4.5 * n_rows),
        squeeze=False
    )

    colors = sns.color_palette("husl", 2)

    for idx, model_name in enumerate(model_names):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]

        epochs = range(1, len(train_losses[model_name]) + 1)
        ax.plot(
            epochs,
            train_losses[model_name],
            label='Training Loss',
            color=colors[0],
            linewidth=2,
            linestyle='-'
        )

        model_val_losses = val_losses.get(model_name, [])
        if len(model_val_losses) > 0:
            val_epochs = range(1, len(model_val_losses) + 1)
            ax.plot(
                val_epochs,
                model_val_losses,
                label='Validation Loss',
                color=colors[1],
                linewidth=2,
                linestyle='--',
                alpha=0.9
            )

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(model_name)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

    # Hide any unused subplot axes
    for idx in range(n_models, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    figure_title = 'Training and Validation Loss Curves'
    if title_suffix:
        figure_title += f' - {title_suffix}'
    fig.suptitle(figure_title, fontsize=18)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    png_path = output_dir / 'loss_curves_subplots.png'
    pdf_path = output_dir / 'loss_curves_subplots.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved combined loss curves to {png_path}")

