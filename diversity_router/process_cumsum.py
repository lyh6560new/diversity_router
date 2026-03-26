#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import tiktoken
import argparse

repo_root = Path(__file__).parent.parent

MODELS = [
    "llama-3.2-1b",
    "llama-3.2-3b",
    "llama-3.1-8b",
    "llama-3.3-70b",
    "qwen2.5-72b-instruct",
    "qwen3-0.6b",
    "qwen3-1.7b",
    "qwen3-4b",
    "qwen3-8b",
    "qwen3-14b",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "olmo-2-0425-1b",
    "olmo-2-1124-7b",
    "olmo-2-1124-13b",
    "olmo-2-0325-32b"
]

# Hardcoded size lookup for the 18 supported models
MODEL_SIZE_MAP = {
    "llama-3.2-1b": "1B",
    "llama-3.2-3b": "3B",
    "llama-3.1-8b": "8B",
    "llama-3.3-70b": "70B",
    "qwen2.5-72b-instruct": "72B",
    "qwen3-0.6b": "0.6B",
    "qwen3-1.7b": "1.7B",
    "qwen3-4b": "4B",
    "qwen3-8b": "8B",
    "qwen3-14b": "14B",
    "gemma-3-1b-it": "1B",
    "gemma-3-4b-it": "4B",
    "gemma-3-12b-it": "12B",
    "gemma-3-27b-it": "27B",
    "olmo-2-0425-1b": "1B",
    "olmo-2-1124-7b": "7B",
    "olmo-2-1124-13b": "13B",
    "olmo-2-0325-32b": "32B",
}


# Inlined from study_longform_nb_questions/study_top10_average/create_topk_table.py
def calculate_total_sum(generation_scores):
    """
    Calculate the total sum of all generation scores, excluding duplicates (0s).

    Args:
        generation_scores: List of generation scores (0 for duplicates, 1-10 for unique)

    Returns:
        Sum of all non-zero scores
    """
    if not generation_scores:
        return 0.0
    non_zero_scores = [score for score in generation_scores if score > 0]
    return sum(non_zero_scores)


def load_query_content_wildchat() -> Dict[str, str]:
    """Load query content from wildchat dataset"""
    content_map = {}
    curated_file = repo_root / "data" / "wildchat-1k.jsonl"
    with open(curated_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            content_map[data['id']] = data['prompt']
    return content_map


def load_query_content(data: str) -> Dict[str, str]:
    """Load query content based on dataset type"""
    if data == "wildchat":
        return load_query_content_wildchat()
    else:
        raise ValueError(f"Unsupported data type: {data}")


def get_model_size(model_name: str) -> str:
    """Get the size string for a model (e.g., '1B', '8B', '72B')"""
    return MODEL_SIZE_MAP.get(model_name, "Unknown")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken with cl100k_base encoding."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback to approximate token count if tiktoken fails
        return int(len(text.split()) * 1.3)


def _load_model_data_from_dir(
    strategy: str,
    query_ids: List[str],
    base_dir: Path,
    n_value: int
) -> Dict[str, Dict[str, Dict]]:
    """Load model data for the given query_ids from base_dir/strategy/model/n_{n_value}/."""
    all_data = {qid: {} for qid in query_ids}
    query_ids_set = set(query_ids)

    for model in MODELS:
        model_dir = base_dir / strategy / model / f"n_{n_value}"
        generations_file = model_dir / "generations.jsonl"
        scores_file = model_dir / "scores.jsonl"

        if not generations_file.exists():
            print(f"    Warning: Missing generations file for {model}: {generations_file}")
            continue

        if not scores_file.exists():
            print(f"    Warning: Missing scores file for {model}: {scores_file}")
            continue

        # Load generations
        generations_map = {}
        with open(generations_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    qid = entry['id']
                    if qid in query_ids_set:
                        generations_map[qid] = entry.get('generations', [])
                except json.JSONDecodeError as e:
                    print(f"    Warning: Skipping malformed JSON in {model} generations: {e}")
                    continue

        # Load scores and calculate cumulative sum and other metrics on first 50
        with open(scores_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    qid = entry['id']
                    if qid not in query_ids_set:
                        continue

                    gen_scores = entry.get('generation_scores', [])
                    raw_generation_scores = entry.get('raw_generation_scores', gen_scores)
                    partition = entry.get('partition', [])
                    partition_scores = entry.get('partition_scores', [])

                    gen_scores_first_50 = gen_scores[:50]
                    raw_scores_first_50 = raw_generation_scores[:50]
                    partition_first_50 = partition[:50]

                    cumsum = calculate_total_sum(gen_scores_first_50)
                    distinctness = len(set(partition_first_50))
                    quality = (sum(raw_scores_first_50) / len(raw_scores_first_50)) if raw_scores_first_50 else 0.0

                    unique_partitions = set(partition_first_50)
                    selected_scores = [
                        partition_scores[p]
                        for p in unique_partitions
                        if isinstance(p, int) and 0 <= p < len(partition_scores)
                    ]
                    avg_unique_quality = (sum(selected_scores) / len(selected_scores)) if selected_scores else 0.0

                    all_data[qid][model] = {
                        'output': generations_map.get(qid, []),
                        'cumulative_sum_score': cumsum,
                        'distinctness': distinctness,
                        'quality': quality,
                        'avg_unique_quality': avg_unique_quality,
                    }

                except json.JSONDecodeError as e:
                    print(f"    Warning: Skipping malformed JSON in {model} scores: {e}")
                    continue

    return all_data


def load_single_model_data(strategy: str, query_ids: List[str], data: str) -> Dict[str, Dict[str, Dict]]:
    """
    Load generations and cumulative sum scores from single model results.

    NOTE: The raw results directories referenced below are NOT included in this release.
    Download the pre-computed CSV outputs from HuggingFace:
        https://huggingface.co/datasets/yuhan-nlp/diversity-router-data
    The downloaded files should be placed under outputs_18_models/<data>/<strategy>/.
    Only run this script if you have access to the raw results directories.

    Returns: {query_id: {model_name: {'output': [generations], 'cumulative_sum_score': float}}}
    """
    data_base_dirs = {
        "wildchat": {
            "sample_1": repo_root / "results" / "longform_qa_wildchat",
            "sample_2": repo_root / "results" / "longform_qa_wildchat",
            "list_all": repo_root / "results" / "longform_qa_wildchat"
        },
    }

    data_n_values = {
        "wildchat": {
            "sample_1": 50,
            "sample_2": 25,
            "list_all": 25
        },
    }

    base_dir = data_base_dirs[data][strategy]
    n_value = data_n_values[data][strategy]

    return _load_model_data_from_dir(strategy, query_ids, base_dir, n_value)


def get_best_model_from_scores(model_data: Dict[str, Dict]) -> str:
    """
    Determine best model based on highest cumulative sum score.

    Args:
        model_data: {model_name: {'output': [...], 'cumulative_sum_score': float}}

    Returns:
        Name of model with highest cumsum score
    """
    if not model_data:
        return ""

    best_model = max(model_data.items(),
                     key=lambda x: x[1].get('cumulative_sum_score', 0.0))
    return best_model[0]


def process_strategy(strategy: str, output_dir: Path, data: str = "wildchat"):
    """Main processing function for one strategy"""

    print(f"  Loading single model data (generations and scores)...")
    query_content = load_query_content(data)
    query_ids = sorted(query_content.keys())
    print(f"    Found {len(query_ids)} queries from dataset")

    all_model_data = load_single_model_data(strategy, query_ids, data)
    print(f"    Loaded data for {len(MODELS)} models")

    print(f"  Building output data structures...")
    csv_rows = []
    jsonl_rows = []
    simple_csv_rows = []
    distinctness_rows = []
    quality_rows = []
    avg_unique_quality_rows = []

    for qid in query_ids:
        model_data = all_model_data.get(qid, {})
        best_model_name = get_best_model_from_scores(model_data)

        if not best_model_name:
            print(f"    Warning: No model data found for query {qid}, skipping")
            continue

        content = query_content.get(qid, "")

        csv_row = {
            'query_id': qid,
            'query_content': content,
            'best_model_name': best_model_name
        }
        for model in MODELS:
            col_name = f"{model}_cumsum"
            model_info = model_data.get(model, {})
            csv_row[col_name] = model_info.get('cumulative_sum_score', 0.0)

        csv_rows.append(csv_row)

        jsonl_rows.append({
            'query_id': qid,
            'query_content': content,
            'best_model_name': best_model_name,
            'all_model_outputs': model_data
        })

        best_model_size = get_model_size(best_model_name)
        best_model_data = model_data.get(best_model_name, {})
        generations = best_model_data.get('output', [])
        first_answer = generations[0] if generations else ""
        ans_len = count_tokens(first_answer)

        simple_csv_rows.append({
            'query_id': qid,
            'query_content': content,
            'best_model_name': best_model_name,
            'best_model_size': best_model_size,
            'ans_len': ans_len
        })

        distinctness_row = {'query_id': qid, 'query_content': content}
        quality_row = {'query_id': qid, 'query_content': content}
        avg_unique_quality_row = {'query_id': qid, 'query_content': content}
        for model in MODELS:
            mdata = model_data.get(model, {})
            distinctness_row[model] = mdata.get('distinctness', 0)
            quality_row[model] = mdata.get('quality', 0.0)
            avg_unique_quality_row[model] = mdata.get('avg_unique_quality', 0.0)

        distinctness_rows.append(distinctness_row)
        quality_rows.append(quality_row)
        avg_unique_quality_rows.append(avg_unique_quality_row)

    strategy_output_dir = output_dir / data / strategy
    strategy_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Writing CSV output...")
    columns = ['query_id', 'query_content', 'best_model_name'] + \
              [f"{model}_cumsum" for model in MODELS]

    df = pd.DataFrame(csv_rows)
    df = df[columns]

    label_counts = df['best_model_name'].value_counts()
    print(f"    Label distribution (df):\n{label_counts}")
    labels_to_keep = label_counts[label_counts > 1].index
    original_len = len(df)
    df = df[df['best_model_name'].isin(labels_to_keep)]
    dropped_count = original_len - len(df)
    print(f"    Dropped {dropped_count} rows with singleton labels (kept {len(df)} rows)")

    kept_query_ids = set(df['query_id'].tolist())

    csv_path = strategy_output_dir / "best_model_per_query.csv"
    df.to_csv(csv_path, index=False)
    print(f"    Wrote {csv_path}")

    print(f"  Writing JSONL output...")
    jsonl_path = strategy_output_dir / "model_outputs_per_query.jsonl"
    with open(jsonl_path, 'w') as f:
        for row in jsonl_rows:
            if row['query_id'] in kept_query_ids:
                f.write(json.dumps(row) + '\n')
    print(f"    Wrote {jsonl_path}")

    print(f"  Writing query model length and size CSV output...")
    simple_df = pd.DataFrame(simple_csv_rows)
    simple_df = simple_df[simple_df['query_id'].isin(kept_query_ids)]
    simple_csv_path = strategy_output_dir / "query_model_len_size.csv"
    simple_df.to_csv(simple_csv_path, index=False)
    print(f"    Wrote {simple_csv_path}")

    print(f"  Writing distinctness CSV output...")
    other_columns = ['query_id', 'query_content'] + MODELS
    distinctness_df = pd.DataFrame(distinctness_rows)
    distinctness_df = distinctness_df[distinctness_df['query_id'].isin(kept_query_ids)]
    distinctness_df = distinctness_df[other_columns]
    distinctness_path = strategy_output_dir / "distinctness_per_query.csv"
    distinctness_df.to_csv(distinctness_path, index=False)
    print(f"    Wrote {distinctness_path}")

    print(f"  Writing quality CSV output...")
    quality_df = pd.DataFrame(quality_rows)
    quality_df = quality_df[quality_df['query_id'].isin(kept_query_ids)]
    quality_df = quality_df[other_columns]
    quality_path = strategy_output_dir / "quality_per_query.csv"
    quality_df.to_csv(quality_path, index=False)
    print(f"    Wrote {quality_path}")

    print(f"  Writing avg unique quality CSV output...")
    avg_unique_quality_df = pd.DataFrame(avg_unique_quality_rows)
    avg_unique_quality_df = avg_unique_quality_df[avg_unique_quality_df['query_id'].isin(kept_query_ids)]
    avg_unique_quality_df = avg_unique_quality_df[other_columns]
    avg_unique_quality_path = strategy_output_dir / "avg_unique_quality_per_query.csv"
    avg_unique_quality_df.to_csv(avg_unique_quality_path, index=False)
    print(f"    Wrote {avg_unique_quality_path}")

    ref_query_ids = df['query_id'].tolist()
    assert simple_df['query_id'].tolist() == ref_query_ids, "simple_df query_id order mismatch"
    assert distinctness_df['query_id'].tolist() == ref_query_ids, "distinctness_df query_id order mismatch"
    assert quality_df['query_id'].tolist() == ref_query_ids, "quality_df query_id order mismatch"
    assert avg_unique_quality_df['query_id'].tolist() == ref_query_ids, "avg_unique_quality_df query_id order mismatch"
    print(f"    Verified all outputs have consistent query_id order")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Process strategies for best model per query analysis.")
    parser.add_argument('--data', type=str, choices=['wildchat'], default='wildchat',
                        help='Dataset type to process (default: wildchat)')
    args = parser.parse_args()

    strategies_config = ['list_all']

    output_dir = repo_root / "outputs_18_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    for strategy in strategies_config:
        print(f"\nProcessing {strategy}...")
        process_strategy(strategy, output_dir, data=args.data)
        print(f"Completed {strategy}")

    print("\nAll strategies processed successfully!")


if __name__ == "__main__":
    main()
