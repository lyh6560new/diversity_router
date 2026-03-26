'''
Frequency-based baseline for model selection.

Unlike best_overall_baseline.py which always predicts the single best model,
this baseline uses the label frequency distribution from training data to make
weighted random predictions on the test set.

Usage:
    # In-distribution evaluation
    python study_longform_nb_questions/study_router/frequency_baseline.py \
        --csv_path study_longform_nb_questions/study_router/outputs/wildchat/list_all/best_model_per_query.csv

    # Out-of-distribution evaluation (wildchat freq -> longform_qa)
    python study_longform_nb_questions/study_router/frequency_baseline.py \
        --csv_path study_longform_nb_questions/study_router/outputs/wildchat/list_all/best_model_per_query.csv \
        --OOD
    
    # 18 models
    python study_longform_nb_questions/study_router/frequency_baseline.py \
        --csv_path study_longform_nb_questions/study_router/outputs_18_models/wildchat/list_all/best_model_per_query.csv \
        --OOD
        
    # 18 models wild_inf_chat
    python study_longform_nb_questions/study_router/frequency_baseline.py \
        --csv_path study_longform_nb_questions/study_router/outputs_18_models/wild_inf_chat/list_all/best_model_per_query.csv \
        --OOD   
        
    # 18 models sample_1
    python study_longform_nb_questions/study_router/frequency_baseline.py \
        --csv_path study_longform_nb_questions/study_router/outputs_18_models/wildchat/sample_1/best_model_per_query.csv
    
    # 18 models sample_2
    python study_longform_nb_questions/study_router/frequency_baseline.py \
        --csv_path study_longform_nb_questions/study_router/outputs_18_models/wildchat/sample_2/best_model_per_query.csv
'''


import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Dict


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


def get_frequency_distribution(df: pd.DataFrame, train_idx: np.ndarray) -> Tuple[List[str], List[float]]:
    '''
    Get frequency distribution of best model labels from training data.

    Args:
        df: Dataframe with query data
        train_idx: Indices for training data

    Returns:
        models: List of model names
        probs: List of probabilities corresponding to each model
    '''
    train_df = df.iloc[train_idx]
    label_counts = train_df['best_model_name'].value_counts()
    label_probs = (label_counts / len(train_df)).to_dict()

    # Print frequency distribution
    print("\nTraining Label Frequency Distribution:")
    for model, prob in sorted(label_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {prob:.4f} ({int(prob * len(train_df))} samples)")

    # Verify probabilities sum to 1.0
    total_prob = sum(label_probs.values())
    assert abs(total_prob - 1.0) < 1e-6, f"Probabilities should sum to 1.0, got {total_prob}"

    models = list(label_probs.keys())
    probs = list(label_probs.values())

    return models, probs


def eval_ood_with_frequency(ood_df: pd.DataFrame, models: List[str], probs: List[float],
                            model_unique_dict: dict = None, model_quality_dict: dict = None,
                            model_unique_quality_dict: dict = None) -> Tuple[float, float, float, float, float]:
    '''
    Evaluate on OOD dataset using frequency distribution from training data.

    Args:
        ood_df: Out-of-distribution dataset (longform_qa)
        models: Model names from training distribution
        probs: Probabilities for each model
        model_unique_dict: Optional dict mapping model_name -> unique answer counts (distinctness)
        model_quality_dict: Optional dict mapping model_name -> quality scores
        model_unique_quality_dict: Optional dict mapping model_name -> unique quality scores

    Returns:
        ood_accuracy, ood_cumsum_mean, ood_unique_mean, ood_quality_mean, ood_unique_quality_mean
    '''
    np.random.seed(2025)

    # Sample predictions for entire OOD dataset
    ood_labels = ood_df['best_model_name'].values
    predictions = np.random.choice(models, size=len(ood_labels), p=probs)

    # Calculate metrics
    ood_acc = accuracy_score(ood_labels, predictions)

    # Calculate cumsum, unique, and quality for each prediction
    cumsum_values = []
    unique_values = []
    quality_values = []
    unique_quality_values = []
    for i, pred_model in enumerate(predictions):
        cumsum_values.append(ood_df.iloc[i][f"{pred_model}_cumsum"])
        if model_unique_dict and pred_model in model_unique_dict:
            unique_values.append(model_unique_dict[pred_model][i])
        if model_quality_dict and pred_model in model_quality_dict:
            quality_values.append(model_quality_dict[pred_model][i])
        if model_unique_quality_dict and pred_model in model_unique_quality_dict:
            unique_quality_values.append(model_unique_quality_dict[pred_model][i])

    ood_cumsum = np.mean(cumsum_values)
    ood_unique = float(np.mean(unique_values)) if unique_values else None
    ood_quality = float(np.mean(quality_values)) if quality_values else None
    ood_unique_quality = float(np.mean(unique_quality_values)) if unique_quality_values else None

    return ood_acc, ood_cumsum, ood_unique, ood_quality, ood_unique_quality


def calc_frequency_baseline_split(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                                   model_unique_dict: dict = None, model_quality_dict: dict = None,
                                   model_unique_quality_dict: dict = None) -> Tuple[float, float, float, float, float]:
    '''
    Calculate frequency-based baseline using train/test split.

    Steps:
    1. Extract training labels from train_idx (80% of data) and compute frequency distribution
    2. Print frequency distribution for visibility
    3. For each test sample, randomly sample a model weighted by training frequencies
    4. Calculate test accuracy, cumsum, unique, and quality metrics

    Args:
        df: Full dataframe with query data and model cumsum values
        train_idx: Indices for training data (80% from first split, used for frequency calculation)
        test_idx: Indices for test set (20% from first split)
        model_unique_dict: Optional dict mapping model_name -> unique answer counts (distinctness)
        model_quality_dict: Optional dict mapping model_name -> quality scores
        model_unique_quality_dict: Optional dict mapping model_name -> unique quality scores

    Returns:
        test_accuracy, test_cumsum_mean, test_unique_mean, test_quality_mean, test_unique_quality_mean
    '''
    # Get frequency distribution from training data
    models, probs = get_frequency_distribution(df, train_idx)

    # Set random seed for reproducibility
    np.random.seed(2025)

    # Sample predictions for test set using frequency distribution
    test_df = df.iloc[test_idx]
    test_labels = test_df['best_model_name'].values
    predictions = np.random.choice(models, size=len(test_labels), p=probs)

    # Calculate metrics
    test_acc = accuracy_score(test_labels, predictions)

    # Calculate cumsum, unique, and quality for each prediction
    cumsum_values = []
    unique_values = []
    quality_values = []
    unique_quality_values = []
    for pred_model, orig_idx in zip(predictions, test_idx):
        cumsum_values.append(df.iloc[orig_idx][f"{pred_model}_cumsum"])
        if model_unique_dict and pred_model in model_unique_dict:
            unique_values.append(model_unique_dict[pred_model][orig_idx])
        if model_quality_dict and pred_model in model_quality_dict:
            quality_values.append(model_quality_dict[pred_model][orig_idx])
        if model_unique_quality_dict and pred_model in model_unique_quality_dict:
            unique_quality_values.append(model_unique_quality_dict[pred_model][orig_idx])

    test_cumsum = np.mean(cumsum_values)
    test_unique = float(np.mean(unique_values)) if unique_values else None
    test_quality = float(np.mean(quality_values)) if quality_values else None
    test_unique_quality = float(np.mean(unique_quality_values)) if unique_quality_values else None

    print(f"\nFrequency-based predictions use {len(models)} unique models from training data")

    return test_acc, test_cumsum, test_unique, test_quality, test_unique_quality


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate frequency-based baseline accuracy and cumsum.")
    parser.add_argument("--csv_path", type=str, required=True,
                       help="Path to the CSV file containing query data and model cumsum values.")
    parser.add_argument('--OOD', action='store_true',
                       help='Perform out-of-distribution evaluation on longform_qa using wildchat frequency distribution')
    args = parser.parse_args()

    # Load data
    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} queries from {args.csv_path}")

    # Load auxiliary metrics (unique answers and quality)
    model_unique_dict, model_quality_dict, model_unique_quality_dict = _load_auxiliary_metrics(csv_path, df)

    # Create train/test split matching mlp_classifier_n.py split strategy
    label_encoder = LabelEncoder()
    label_indices = label_encoder.fit_transform(df['best_model_name'].values)

    # Check if using wild_inf_chat and load test mask if available
    test_mask = None
    if 'wild_inf_chat' in str(csv_path):
        test_mask_path = csv_path.parent / 'wildchat_test_mask.npy'
        if test_mask_path.exists():
            test_mask = np.load(test_mask_path)
            print(f"Loaded test mask from {test_mask_path}")
            print(f"  Test mask shape: {test_mask.shape}, test samples: {test_mask.sum()}")
        else:
            print(f"WARNING: Test mask not found at {test_mask_path}")
            print("  Run merge_data/generate_wildchat_test_mask.py to generate it.")
            print("  Falling back to random split.")

    indices = np.arange(len(df))

    if test_mask is not None:
        # Use provided test mask (for wild_inf_chat to match wildchat test split)
        test_idx = np.where(test_mask)[0]
        train_val_idx = np.where(~test_mask)[0]
        print(f"Using provided test mask: {len(test_idx)} test samples, {len(train_val_idx)} train+val samples")
    else:
        # First split: 80/20 for train_val/test
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=label_indices,
            random_state=2025
        )

    # Second split: split train_val into train/val (only needed for consistency with MLP)
    # For frequency baseline, we use train_val_idx directly
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=1/8,
        stratify=label_indices[train_val_idx],
        random_state=2025
    )

    print(f"\nSplit sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Using train_val (Train+Val={len(train_val_idx)} samples) for frequency calculation")

    # Get frequency distribution from training split
    models, probs = get_frequency_distribution(df, train_idx)

    if args.OOD:
        # OOD evaluation on longform_qa
        print("\n" + "="*70)
        print("OUT-OF-DISTRIBUTION EVALUATION")
        print("="*70)

        # Determine OOD dataset path
        ood_path = Path(args.csv_path).parent.parent.parent / 'longform_qa' / 'list_all' / 'best_model_per_query.csv'
        print(f"Loading OOD dataset from: {ood_path}")

        if not ood_path.exists():
            print(f"Error: OOD dataset not found at {ood_path}")
            sys.exit(1)

        # Load entire longform_qa dataset
        ood_df = pd.read_csv(ood_path)
        print(f"\nLoaded {len(ood_df)} OOD queries from {ood_path}")

        # Load OOD auxiliary metrics
        ood_unique_dict, ood_quality_dict, ood_unique_quality_dict = _load_auxiliary_metrics(ood_path, ood_df)

        # Evaluate on OOD dataset
        ood_acc, ood_cumsum, ood_unique, ood_quality, ood_unique_quality = eval_ood_with_frequency(
            ood_df, models, probs, ood_unique_dict, ood_quality_dict, ood_unique_quality_dict
        )

        print(f"\nOOD Evaluation Results (Wildchat freq -> Longform QA):")
        print(f"OOD Accuracy: {ood_acc:.4f} ({ood_acc*100:.2f}%)")
        print(f"OOD Cumsum Mean: {ood_cumsum:.2f}")
        if ood_unique is not None:
            print(f"OOD Unique Mean: {ood_unique:.4f}")
        if ood_quality is not None:
            print(f"OOD Quality Mean: {ood_quality:.4f}")
        if ood_unique_quality is not None:
            print(f"OOD Unique Quality Mean: {ood_unique_quality:.4f}")
    else:
        # In-distribution evaluation on test split
        print("\nRunning frequency-based baseline on test split...")
        test_acc, test_cumsum, test_unique, test_quality, test_unique_quality = calc_frequency_baseline_split(
            df, train_idx, test_idx, model_unique_dict, model_quality_dict, model_unique_quality_dict
        )

        print(f"\nFrequency Baseline Results:")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Test Cumsum Mean: {test_cumsum:.2f}")
        if test_unique is not None:
            print(f"Test Unique Mean: {test_unique:.4f}")
        if test_quality is not None:
            print(f"Test Quality Mean: {test_quality:.4f}")
        if test_unique_quality is not None:
            print(f"Test Unique Quality Mean: {test_unique_quality:.4f}")

        # Print comparison baselines
        n_models = len(label_encoder.classes_)
        random_baseline_acc = 1.0 / n_models
        print(f"\nBaseline Comparisons:")
        print(f"Random Baseline Accuracy: {random_baseline_acc:.4f} ({random_baseline_acc*100:.2f}%) (1/{n_models})")
        print(f"Frequency Baseline Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
    print(f"ready to csv:")

    
    mlp_classifier_metrics = [
        f"{test_acc*100:.2f}%",
        f"{test_unique:.2f}" if test_unique is not None else 'N/A',
        f"{test_quality:.2f}" if test_quality is not None else 'N/A',
        f"{test_unique_quality:.2f}" if test_unique_quality is not None else 'N/A',
        f"{test_cumsum:.2f}",
        f"{test_cumsum/500*100:.2f}%"
    ]
    print(f" Frequency baseline:"+ ', '.join(mlp_classifier_metrics))
    
    print('-'*70)
