#!/usr/bin/env python3
"""
Usage:
    python diversity_router/knn_classifier.py --data wildchat --strategy list_all --k 5 --data_dir outputs_18_models --output_dir outputs_18_models

"""
import datetime
import json
import argparse
import numpy as np
from pathlib import Path
import sys
from typing import Tuple, Dict, List

# Machine learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Overall best baseline
from best_overall_baseline import calc_best_overall_baseline_split

from utils import *

# Use logging for documenting
import logging

# load data
import pandas as pd

# Fix random seed for reproducibility
np.random.seed(2025)

def load_mixed_data(mix_type: str) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    """Load data for mixed dataset, train on one dataset and test on another. 
       Return:
         - df 
         - queries
         - query ids
         - labels
         - embeddings
         - label indices
         - model_unique_dict, model_quality_dict, model_unique_quality_dict
         - test_mask (boolean array indicating which examples are in test set, used for wild_inf_chat to match wildchat test split)
    """
    if mix_type == 'train_wildchat_test_longform_qa':
        train_data_source_path = Path(__file__).parent / 'outputs_18_models' / 'wildchat' / 'list_all' / 'best_model_per_query.csv'
        test_data_source_path = Path(__file__).parent / 'outputs_18_models' / 'longform_qa' / 'list_all' / 'best_model_per_query.csv'
        
        df_train = pd.read_csv(train_data_source_path)
        df_test = pd.read_csv(test_data_source_path)
        
        # keys = queries, labels, query_ids, label_indices, label_encoder, model_cumsum_dict, model_unique_dict, model_quality_dict, model_unique_quality_dict
        #train data
        train_data = {
            'queries': None,
            'labels': None,
            'query_ids': None,
            'label_indices': None,
            'label_encoder': None,
            'model_cumsum_dict': None,
            'model_unique_dict': None,
            'model_quality_dict': None,
            'model_unique_quality_dict': None
        }
        train_data['queries'], train_data['labels'], train_data['query_ids'], train_data['label_indices'], train_data['label_encoder'], train_data['model_cumsum_dict'], train_data['model_unique_dict'], train_data['model_quality_dict'], train_data['model_unique_quality_dict'] = load_data(train_data_source_path)
        
        #test data
        test_data = {
            'queries': None,
            'labels': None,
            'query_ids': None,
            'label_indices': None,
            'label_encoder': None,
            'model_cumsum_dict': None, 
            'model_unique_dict': None,
            'model_quality_dict': None,
            'model_unique_quality_dict': None
        }
        test_data['queries'], test_data['labels'], test_data['query_ids'], test_data['label_indices'], test_data['label_encoder'], test_data['model_cumsum_dict'], test_data['model_unique_dict'], test_data['model_quality_dict'], test_data['model_unique_quality_dict'] = load_data(test_data_source_path)

        # make sure label encoders are the same (use train data's label encoder)
        assert set(train_data['label_encoder'].classes_) == set(test_data['label_encoder'].classes_), "Label encoders for train and test data must have the same classes"
        # also the idx of each class must be the same
        for cls in train_data['label_encoder'].classes_:
            assert train_data['label_encoder'].transform([cls])[0] == test_data['label_encoder'].transform([cls])[0], f"Class {cls} has different indices in train and test label encoders"

        # Generate train embeddings
        train_embeddings = generate_embeddings(train_data['queries'], cache_path=Path(__file__).parent.parent / 'outputs_18_models' / 'wildchat' / 'list_all' / 'query_embeddings.npy', use_cache=True)
        # Generate test embeddings
        test_embeddings = generate_embeddings(test_data['queries'], cache_path=Path(__file__).parent.parent / 'outputs_18_models' / 'longform_qa' / 'list_all' / 'query_embeddings.npy', use_cache=True)
        
        # split train source data
        train_source_indices = np.arange(len(train_data['queries']))
        train_train_idx, train_test_idx = train_test_split(
                train_source_indices,
                test_size=0.2,
                stratify=train_data['label_indices'],
                random_state=2025
            )
        print(f"Train source split: {len(train_train_idx)} train samples, {len(train_test_idx)} test samples")
        
        # use full test source data as test set (no split, all test)
        test_test_idx = np.arange(len(test_data['queries']))
        print(f"Test source: {len(test_test_idx)} test samples")
        

        # take the train_train_idx from train source as train, and test_test_idx from test source as test
        final_df_train = df_train.iloc[train_train_idx]
        final_train_queries = [train_data['queries'][i] for i in train_train_idx]
        final_train_query_ids = [train_data['query_ids'][i] for i in train_train_idx]
        final_train_embeddings = [train_embeddings[i] for i in train_train_idx]
        final_train_labels = [train_data['labels'][i] for i in train_train_idx]
        final_train_label_indices = [train_data['label_indices'][i] for i in train_train_idx]
        
        final_df_test = df_test
        final_test_queries = [test_data['queries'][i] for i in test_test_idx]
        final_test_query_ids = [test_data['query_ids'][i] for i in test_test_idx]
        final_test_embeddings = test_embeddings[test_test_idx]
        final_test_labels = [test_data['labels'][i] for i in test_test_idx]
        final_test_label_indices = [test_data['label_indices'][i] for i in test_test_idx]
        
        final_df = pd.concat([final_df_train, final_df_test], ignore_index=True)
        final_queries = np.concatenate([final_train_queries, final_test_queries], axis=0)
        final_query_ids = np.concatenate([final_train_query_ids, final_test_query_ids], axis=0)
        final_label_encoder = train_data['label_encoder']  # Use label encoder from train data
        final_embeddings = np.concatenate([final_train_embeddings, final_test_embeddings], axis=0)
        final_labels = np.concatenate([final_train_labels, final_test_labels], axis=0)
        final_label_indices = np.concatenate([final_train_label_indices, final_test_label_indices], axis=0)
        
        #derive test_mask for final data 
        final_test_mask = np.array([False]*len(final_embeddings))
        final_test_mask[len(final_train_embeddings):] = True  # Mark test examples (from test source) as True
        print(f"Final dataset: {len(final_embeddings)} samples, with {final_test_mask.sum()} test samples and {len(final_embeddings) - final_test_mask.sum()} train samples")
        
        # combine model dicts (for cumsum, unique, quality, unique_quality)
        def merge_model_dicts(train_dict, test_dict,train_idx, test_idx):
            merged_dict = {}
            for model in train_dict.keys():
                merged_dict[model] = [train_dict[model][i] for i in train_idx] + [test_dict[model][i] for i in test_idx]
            return merged_dict
        final_model_cumsum_dict = merge_model_dicts(train_data['model_cumsum_dict'], test_data['model_cumsum_dict'], train_train_idx, test_test_idx)
        final_model_unique_dict = merge_model_dicts(train_data['model_unique_dict'], test_data['model_unique_dict'], train_train_idx, test_test_idx)
        final_model_quality_dict = merge_model_dicts(train_data['model_quality_dict'], test_data['model_quality_dict'], train_train_idx, test_test_idx)
        final_model_unique_quality_dict = merge_model_dicts(train_data['model_unique_quality_dict'], test_data['model_unique_quality_dict'], train_train_idx, test_test_idx)
        
        
        # print the first two examples of final queries, labels, label indices, and embeddings to verify
        print("Final Queries (first 2):", final_queries[:2])
        print("Final Query IDs (first 2):", final_query_ids[:2])
        print("Final Labels (first 2):", final_labels[:2])
        print("Final Label Indices (first 2):", final_label_indices[:2])
        print("Final Embeddings (first 2):", final_embeddings[:2]) 
        print("Final Model Cumsum Dict (first 2 examples for each model):")
        for model, cumsums in final_model_cumsum_dict.items():
            print(f"  {model}: {cumsums[:2]}")
        print("Final Model Unique Dict (first 2 examples for each model):")
        for model, uniques in final_model_unique_dict.items():
            print(f"  {model}: {uniques[:2]}")
        print("Final Model Quality Dict (first 2 examples for each model):")
        for model, qualities in final_model_quality_dict.items():
            print(f"  {model}: {qualities[:2]}")
        print("Final Model Unique Quality Dict (first 2 examples for each model):")
        for model, unique_qualities in final_model_unique_quality_dict.items():
            print(f"  {model}: {unique_qualities[:2]}")
        
        return final_df, final_queries, final_query_ids, final_labels, final_label_indices, final_label_encoder, final_embeddings, final_model_cumsum_dict, final_model_unique_dict, final_model_quality_dict, final_model_unique_quality_dict, final_test_mask
    else:
        raise ValueError(f"Invalid mix_type: {mix_type}. Must be 'train_wildchat_test_longform_qa'.")



def run_knn_classification_split(embeddings: np.ndarray, label_indices: np.ndarray, label_encoder: LabelEncoder,
                                  k: int = 5, filter_distant: bool = False, threshold: float = 0.20,
                                  test_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, float, List[int], List[int]]:
    """Run train/test split KNN classification with 80/20 split.

    If test_mask is provided, use it to determine test indices instead of random split.
    This is used for wild_inf_chat to ensure the test set matches wildchat's test split.
    """

    indices = np.arange(len(embeddings))

    if test_mask is not None:
        # Use provided test mask (for wild_inf_chat to match wildchat test split)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        print(f"Using provided test mask: {len(test_idx)} test samples, {len(train_idx)} train samples")
    else:
        # Split: 80% train, 20% test
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=label_indices,
            random_state=2025
        )

    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = label_indices[train_idx], label_indices[test_idx]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train KNN with distance weighting
    # The embeddings are normalized, so cosine distance is equivalent to dot product
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance')
    knn.fit(X_train, y_train)

    # Predict on test set
    y_pred = knn.predict(X_test)

    # Filter out test examples with no close neighbors
    if filter_distant:
        distances, _ = knn.kneighbors(X_test, n_neighbors=k)
        filtered_mask = np.array([distances[i].min() < threshold for i in range(len(distances))])

        # Apply filter to y_test, y_pred, and test_idx
        y_test = y_test[filtered_mask]
        y_pred = y_pred[filtered_mask]
        test_idx = test_idx[filtered_mask]

        print(f"Filtered: {sum(filtered_mask)}/{len(filtered_mask)} samples kept (threshold={threshold})")

    # Calculate test accuracy
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy = {test_acc:.3f}")

    return np.array(y_test), y_pred, test_acc, train_idx.tolist(), test_idx.tolist()



def main():
    parser = argparse.ArgumentParser(description="KNN Classifier with Train/Test Split")
    parser.add_argument('--data_dir', default='outputs', choices=['outputs', 'outputs_18_models'],
                        help='strategy dir where train/test data is stored')
    parser.add_argument('--output_dir', default='outputs_split', choices=['outputs_split', 'outputs_18_models'],
                        help='output directory for results')
    parser.add_argument('--data', default='longform_qa', help='Dataset to analyze')
    parser.add_argument('--strategy', default='list_all', choices=['list_all'],
                       help='Strategy to analyze')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors for KNN')
    parser.add_argument('--no_cache', action='store_true',
                       help='Regenerate embeddings instead of using cache')
    parser.add_argument('--exp', type=str, default='',
                       help='Experiment name for logging')
    parser.add_argument('--filter', action='store_true',
                       help='Filter out test examples with no close neighbors')
    parser.add_argument('--threshold', type=float, default=0.20,
                       help='Distance threshold for filtering (default: 0.20)')
    args = parser.parse_args()

    # Get current timestamp
    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup paths
    base_dir = Path(__file__).parent.parent
    strategy_dir = base_dir / args.data_dir / args.data / args.strategy  # path w/ train/test data
    output_parent_dir = base_dir / args.output_dir / args.data / args.strategy
    

    output_dir = output_parent_dir / 'knn'
    

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create logs directory
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)

    # Logging file name
    if args.exp:
        log_file = output_dir / 'logs' / f'{args.exp}_k_{args.k}_{cur_time}.log'
    else:
        log_file = output_dir / 'logs' / f'k_{args.k}_{cur_time}.log'

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(sys.stdout)
                    ])
        
    if 'train_' in args.data:
        df, queries, query_ids, labels, label_indices, label_encoder, embeddings, model_cumsum_dict, model_unique_dict, model_quality_dict, model_unique_quality_dict, test_mask = load_mixed_data(args.data)
    else:
        #raise ValueError(f"Invalid data option for train/test split: {args.data}. Must be one of 'train_wildchat_test_longform_qa'.")
        data_path = strategy_dir / 'best_model_per_query.csv'
        print("Reading data from:", data_path)
        
        # Verify data file exists
        if not data_path.exists():
            print(f"Error: Data file not found at {data_path}")
            sys.exit(1)
            
        cache_path = strategy_dir / 'query_embeddings.npy'  # Cache at strategy level
        
        # Load data and dataframe (needed for baseline calculation)
        
        df = pd.read_csv(data_path)

        queries, labels, query_ids, label_indices, label_encoder, model_cumsum_dict, model_unique_dict, model_quality_dict, model_unique_quality_dict = load_data(data_path)

        # Generate embeddings
        embeddings = generate_embeddings(queries, cache_path, use_cache=not args.no_cache)

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

    # Run train/test split KNN classification
    print(f"\nRunning train/test split KNN classification with k={args.k}...")
    y_true, y_pred, test_accuracy, train_indices, test_indices = run_knn_classification_split(
        embeddings, label_indices, label_encoder, k=args.k,
        filter_distant=args.filter, threshold=args.threshold,
        test_mask=test_mask
    )

    # Calculate metrics (adapted for single evaluation)
    fold_accuracies = [test_accuracy]  # Single accuracy value
    results = calculate_metrics(y_true, y_pred, test_indices, fold_accuracies, label_encoder, args.k, 1, model_cumsum_dict, model_unique_dict, model_quality_dict, model_unique_quality_dict)

    # Add overall best model baseline (using train/test split, no data leakage)
    
    best_overall_acc, best_overall_cumsum, best_overall_unique, best_overall_quality, best_overall_unique_quality = calc_best_overall_baseline_split(
        df,
        np.array(train_indices),
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
    logging.info(f"KNN(K={args.k}) Classifier Test Accuracy: {test_accuracy*100:.2f}%")
    logging.info('-'*70)
    
    # ready to csv: accuracy, unique answer, answer quality, avg unique answer quality, cumsum, cumsum %
    logging.info(f"\nFinal ready-to-csv line:")
    final_metrics = [
        f"{test_accuracy*100:.1f}%",
        f"{results['unique_metrics']['predicted_unique_mean']:.1f}" if results.get('unique_metrics') else 'N/A',
        f"{results['quality_metrics']['predicted_quality_mean']:.1f}" if results.get('quality_metrics') else 'N/A',
        f"{results['unique_quality_metrics']['predicted_unique_quality_mean']:.2f}" if results.get('unique_quality_metrics') else 'N/A',
        f"{results['cumsum_metrics']['predicted_cumsum_mean']:.1f}",
        f"{results['cumsum_metrics']['predicted_cumsum_mean']/500*100:.1f}%"
    ]
    logging.info(', '.join(final_metrics))


    print("\nAll outputs saved successfully!")


if __name__ == "__main__":
    main()
