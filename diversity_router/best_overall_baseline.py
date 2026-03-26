'''
This is code is to implement same strategy as  study_longform_nb_questions/study_ensemble/ensemble_strategy_analysis.py
'''
'''Example usage:
Usage:
    python diversity_router/best_overall_baseline.py --csv_path outputs_18_models/wildchat/list_all/best_model_per_query.csv
'''
import sys
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np


def calculate_best_models(model_scores: Dict[str, Dict[str, float]]) -> Tuple[str, Tuple[str, str]]:
    """
    Calculate best overall model and two best models from single model scores.

    Returns:
        (best_model, (best_model, second_best_model))
    """
    # Calculate total sum for each model
    model_total_scores = {}
    for model, question_scores in model_scores.items():
        total_sum = sum(question_scores.values())
        model_total_scores[model] = total_sum

    # Sort by score
    sorted_models = sorted(model_total_scores.items(), key=lambda x: x[1], reverse=True)

    best_model = sorted_models[0][0]
    two_best = (sorted_models[0][0], sorted_models[1][0])

    return best_model, two_best


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


# get overall best model
def get_best_overall_model(df: pd.DataFrame) -> str:
    '''
    Get best overall model from DataFrame.

    csv format:
    query_id,query_content,best_model_name,llama-3.2-1b_cumsum,llama-3.2-3b_cumsum,...
    return: best model name
    '''
    # Construct dict[model_name] = query_cumsum_dict
    # query_cumsum_dict = {[query_id]: cumsum_value}

    # First get all model names by selecting columns that end with '_cumsum'
    model_names = [col.split('_cumsum')[0] for col in df.columns if col.endswith('_cumsum')]
    model_cumsum_dict = {model: dict(zip(df['query_id'], df[f"{model}_cumsum"])) for model in model_names}

    best_model, _ = calculate_best_models(model_cumsum_dict)
    return best_model

def get_best_overall_model_from_indices(df: pd.DataFrame, train_idx: np.ndarray) -> str:
    '''
    Get best overall model using only training indices.
    Prevents data leakage by excluding test set.
    '''
    train_df = df.iloc[train_idx]
    return get_best_overall_model(train_df)

def run_baseline_cv(df: pd.DataFrame, n_folds: int = 10) -> Tuple[float, float]:
    '''
    Run K-fold cross-validation for best overall baseline.

    For each fold:
    1. Determine best overall model using TRAINING data only
    2. Predict that model for all TEST samples
    3. Track accuracy and cumsum

    Returns:
        mean_accuracy, mean_cumsum
    '''
    # Extract labels
    labels = df['best_model_name'].values
    label_encoder = LabelEncoder()
    label_indices = label_encoder.fit_transform(labels)

    # Setup CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2025)

    fold_accuracies = []
    fold_cumsums = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(label_indices)), label_indices)):
        # Find best model using training data only
        best_model = get_best_overall_model_from_indices(df, train_idx)

        # Evaluate on test set
        test_labels = df.iloc[test_idx]['best_model_name'].values
        predictions = [best_model] * len(test_labels)

        # Calculate fold metrics
        fold_acc = accuracy_score(test_labels, predictions)
        fold_cumsum = df.iloc[test_idx][f"{best_model}_cumsum"].mean()

        fold_accuracies.append(fold_acc)
        fold_cumsums.append(fold_cumsum)

        print(f"Fold {fold+1}: best_model={best_model}, acc={fold_acc:.4f}, cumsum={fold_cumsum:.4f}")

    return np.mean(fold_accuracies), np.mean(fold_cumsums)

def calc_best_overall_baseline_split(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                                      model_unique_dict: dict = None, model_quality_dict: dict = None,
                                      model_unique_quality_dict: dict = None) -> Tuple[float, float, float, float, float]:
    '''
    Calculate best overall baseline using train/test split (no data leakage).

    Args:
        df: Full dataframe with query data and model cumsum values
        train_idx: Indices for training set
        test_idx: Indices for test set
        model_unique_dict: Optional dict mapping model_name -> unique answer counts (distinctness)
        model_quality_dict: Optional dict mapping model_name -> quality scores
        model_unique_quality_dict: Optional dict mapping model_name -> unique quality scores

    Returns:
        test_accuracy, test_cumsum_mean, test_unique_mean, test_quality_mean, test_unique_quality_mean
    '''
    # Find best model using training data only
    best_model = get_best_overall_model_from_indices(df, train_idx)
    print(f"Best overall model (from training data): {best_model}")

    # Evaluate on test set only
    test_df = df.iloc[test_idx]
    test_labels = test_df['best_model_name'].values
    predictions = [best_model] * len(test_labels)

    # Calculate metrics on test set
    test_acc = accuracy_score(test_labels, predictions)
    test_cumsum = test_df[f"{best_model}_cumsum"].mean()

    # Calculate unique answer metric if available
    test_unique = None
    if model_unique_dict and best_model in model_unique_dict:
        test_unique = float(np.mean([model_unique_dict[best_model][idx] for idx in test_idx]))

    # Calculate quality metric if available
    test_quality = None
    if model_quality_dict and best_model in model_quality_dict:
        test_quality = float(np.mean([model_quality_dict[best_model][idx] for idx in test_idx]))

    # Calculate unique_quality metric if available
    test_unique_quality = None
    if model_unique_quality_dict and best_model in model_unique_quality_dict:
        test_unique_quality = float(np.mean([model_unique_quality_dict[best_model][idx] for idx in test_idx]))

    print(f"Best overall baseline test accuracy: {test_acc:.4f}")
    print(f"Best overall baseline test cumsum: {test_cumsum:.4f}")
    if test_unique is not None:
        print(f"Best overall baseline test unique: {test_unique:.4f}")
    if test_quality is not None:
        print(f"Best overall baseline test quality: {test_quality:.4f}")
    if test_unique_quality is not None:
        print(f"Best overall baseline test unique_quality: {test_unique_quality:.4f}")

    return test_acc, test_cumsum, test_unique, test_quality, test_unique_quality

def calc_best_overall_baseline_split_with_model(df: pd.DataFrame, model_name: str, test_idx: np.ndarray,
                                                 model_unique_dict: dict = None, model_quality_dict: dict = None,
                                                 model_unique_quality_dict: dict = None) -> Tuple[float, float, float, float, float]:
    '''
    Calculate best overall baseline using specified model on test set.

    Args:
        df: Full dataframe with query data and model cumsum values
        model_name: Name of the model to evaluate
        test_idx: Indices for test set
        model_unique_dict: Optional dict mapping model_name -> unique answer counts (distinctness)
        model_quality_dict: Optional dict mapping model_name -> quality scores
        model_unique_quality_dict: Optional dict mapping model_name -> unique quality scores

    Returns:
        test_accuracy, test_cumsum_mean, test_unique_mean, test_quality_mean, test_unique_quality_mean
    '''
    # Evaluate on test set only
    test_df = df.iloc[test_idx]
    test_labels = test_df['best_model_name'].values
    predictions = [model_name] * len(test_labels)

    # Calculate metrics on test set
    test_acc = accuracy_score(test_labels, predictions)
    test_cumsum = test_df[f"{model_name}_cumsum"].mean()

    # Calculate unique answer metric if available
    test_unique = None
    if model_unique_dict and model_name in model_unique_dict:
        test_unique = float(np.mean([model_unique_dict[model_name][idx] for idx in test_idx]))

    # Calculate quality metric if available
    test_quality = None
    if model_quality_dict and model_name in model_quality_dict:
        test_quality = float(np.mean([model_quality_dict[model_name][idx] for idx in test_idx]))

    # Calculate unique_quality metric if available
    test_unique_quality = None
    if model_unique_quality_dict and model_name in model_unique_quality_dict:
        test_unique_quality = float(np.mean([model_unique_quality_dict[model_name][idx] for idx in test_idx]))

    print(f"Specified model '{model_name}' test accuracy: {test_acc:.4f}")
    print(f"Specified model '{model_name}' test cumsum: {test_cumsum:.4f}")
    if test_unique is not None:
        print(f"Specified model '{model_name}' test unique: {test_unique:.4f}")
    if test_quality is not None:
        print(f"Specified model '{model_name}' test quality: {test_quality:.4f}")
    if test_unique_quality is not None:
        print(f"Specified model '{model_name}' test unique_quality: {test_unique_quality:.4f}")

    return test_acc, test_cumsum, test_unique, test_quality, test_unique_quality

def calc_best_overall_baseline(csv_path: str, subset = False) -> pd.DataFrame:
    '''
    query_id,query_content,best_model_name,llama-3.2-1b_cumsum,llama-3.2-3b_cumsum,...
    return: cumsum, acc if always choose best overall model
    '''
    df = pd.read_csv(csv_path)

    # Sample
    if subset:
        print("Using subset of data for wildchat...")
        df = df.sample(n=500, random_state=2025).reset_index(drop=True)

    print(f"Calculating best overall baseline from {csv_path}...")
    best_model = get_best_overall_model(df)
    print(f"Best overall model is {best_model}")
    gt_labels = df['best_model_name'].tolist()
    pred_labels = [best_model] * len(gt_labels)
    acc = accuracy_score(gt_labels, pred_labels)
    cumsum_avg = df[f"{best_model}_cumsum"].mean()
    return acc, cumsum_avg

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate best overall baseline accuracy and cumsum.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing query data and model cumsum values.")
    parser.add_argument("--subset", action='store_true', help="Use 500-sample subset for wildchat")
    parser.add_argument("--n_folds", type=int, default=0, help="Number of CV folds (use 0 for no CV)")
    args = parser.parse_args()

    # Load data
    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)
    if args.subset:
        print("Using subset of data for wildchat...")
        df = df.sample(n=500, random_state=2025).reset_index(drop=True)

    # Load auxiliary metrics (unique answers, quality, and unique_quality)
    model_unique_dict, model_quality_dict, model_unique_quality_dict = _load_auxiliary_metrics(csv_path, df)

    # Run with or without CV
    if args.n_folds > 0:
        print(f"Running {args.n_folds}-fold cross-validation...")
        mean_acc, mean_cumsum = run_baseline_cv(df, n_folds=args.n_folds)
        print(f"\nBest Overall Baseline (CV) Mean Accuracy: {mean_acc:.4f}")
        print(f"Best Overall Baseline (CV) Mean Cumsum: {mean_cumsum:.4f}")
    else:
        # Original non-CV version - now using train/test split to avoid data leakage
        label_encoder = LabelEncoder()
        label_indices = label_encoder.fit_transform(df['best_model_name'].values)

        indices = np.arange(len(df))
        # print label distribution
        unique, counts = np.unique(label_indices, return_counts=True)
        label_dist = dict(zip(label_encoder.inverse_transform(unique), counts))
        print(f"Label distribution: {label_dist}")
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=label_indices,
            random_state=2025
        )

        acc, cumsum_avg, unique_avg, quality_avg, unique_quality_avg = calc_best_overall_baseline_split(
            df, train_idx, test_idx, model_unique_dict, model_quality_dict, model_unique_quality_dict
        )

        print(f"\nBest Overall Baseline Results:")
        print(f"Test Accuracy: {acc*100:.2f}%")
        print(f"Test Cumsum Mean: {cumsum_avg:.2f}")
        if unique_avg is not None:
            print(f"Test Unique Mean: {unique_avg:.2f}")
        if quality_avg is not None:
            print(f"Test Quality Mean: {quality_avg:.2f}")
        if unique_quality_avg is not None:
            print(f"Test Unique Quality Mean: {unique_quality_avg:.2f}")
