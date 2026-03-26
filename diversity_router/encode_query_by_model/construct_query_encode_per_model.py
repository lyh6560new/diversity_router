"""
This script encodes queries using each model's last hidden state.
For each query, we extract the last non-padding token's hidden state.

Usage (run from diversity_router_code/ root):
python diversity_router/encode_query_by_model/construct_query_encode_per_model.py \
    --data wildchat --strategy list_all --model llama-3.2-1b --batch_size 32
"""

from pathlib import Path

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import pandas as pd
import gc
from typing import List

# Map from model names in csv to model names inference
MODEL_NAME_MAP = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "olmo-2-0425-1b":"allenai/OLMo-2-0425-1B-Instruct",
    "olmo-2-1124-7b":"allenai/OLMo-2-1124-7B-Instruct",
    "olmo-2-1124-13b":"allenai/OLMo-2-1124-13B-Instruct",
    "olmo-2-0325-32b":"allenai/OLMo-2-0325-32B-Instruct",
    "gemma-3-1b-it": "google/gemma-3-1b-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "gemma-3-27b-it": "google/gemma-3-27b-it", 
}

_BASE = Path(__file__).parent.parent.parent / 'outputs_18_models'

input_data_map = {
    'wildchat': {
        'list_all': _BASE / 'wildchat' / 'list_all' / 'best_model_per_query.csv',
        'sample_1': _BASE / 'wildchat' / 'sample_1' / 'best_model_per_query.csv',
        'sample_2': _BASE / 'wildchat' / 'sample_2' / 'best_model_per_query.csv',
    },
}


def encode_queries_with_model(queries: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """
    Encode queries using model's last hidden state

    Args:
        queries: List of query strings
        model_name: HuggingFace model name
        batch_size: Batch size for processing

    Returns:
        numpy array of shape (n_queries, hidden_dim)
    """
    print(f"Loading model: {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        device_map="auto",
        torch_dtype=torch.float32 # otherwise will be NaN for gemma models
    )

    print(f"Model loaded. Device: {model.device}")

    # Add "/no_think" suffix for Qwen models to prevent chain-of-thought
    if "qwen" in model_name.lower():
        queries = [q if '/no_think' in q else q + " /no_think" for q in queries]
        print("Added /no_think suffix for Qwen model")

    # Create conversation format for each query
    conversations = [[{"role": "user", "content": query}] for query in queries]

    # Apply chat template
    print("Applying chat template...")
    formatted_texts = []
    for conv in conversations:
        formatted_text = tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_texts.append(formatted_text)

    # Process in batches
    all_embeddings = []
    n_batches = (len(formatted_texts) - 1) // batch_size + 1
    print(f"Processing {len(formatted_texts)} queries in {n_batches} batches of {batch_size}...")

    for i in range(0, len(formatted_texts), batch_size):
        batch_texts = formatted_texts[i:i+batch_size]
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{n_batches} ({len(batch_texts)} queries)")

        # Tokenize batch with padding
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        # Move inputs to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Get hidden states
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract last layer hidden states
        last_layer = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"]

        # Get last non-padding token indices
        last_token_idx = attention_mask.sum(dim=1) - 1

        # Extract last token hidden states
        current_batch_size = last_layer.size(0)
        batch_indices = torch.arange(current_batch_size, device=last_layer.device)
        last_token_hidden = last_layer[batch_indices, last_token_idx]

        # Convert to numpy and store
        batch_embeddings = last_token_hidden.cpu().numpy()
        all_embeddings.append(batch_embeddings)

        print(f"  Batch {batch_num} shape: {batch_embeddings.shape}")

        # Cleanup batch
        del inputs, outputs, last_layer, attention_mask
        torch.cuda.empty_cache()

    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)
    print(f"Final embeddings shape: {embeddings.shape}")

    # Replace NaN values with 0.0
    if np.isnan(embeddings).any():
        nan_count = np.isnan(embeddings).sum()
        print(f"Warning: {nan_count} NaN values found, replacing with 0.0")
        embeddings = np.nan_to_num(embeddings, nan=0.0)

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory freed. Current CUDA memory allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

    return embeddings


def construct_query_encode_per_model(dataset: str, strategy: str, model_name: str, batch_size: int = 32):
    """
    Main function to encode queries for a single model

    Args:
        dataset: Dataset name (longform_qa, wildchat, wild_inf_chat, or inf_chat)
        strategy: Strategy name (list_all, sample_1, sample_2)
        model_name: Model name from MODEL_NAME_MAP
        batch_size: Batch size for processing (default=32)
    """
    print("=" * 80)
    print(f"Constructing query encodings for model: {model_name}")
    print(f"Dataset: {dataset}, Strategy: {strategy}, Batch size: {batch_size}")
    print("=" * 80)

    # Setup directories
    base_dir = Path(__file__).parent.parent.parent
    strategy_dir = base_dir / 'outputs_18_models_' / dataset / strategy
    output_dir = strategy_dir / 'query_encode_per_model' / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    output_file = output_dir / 'query_encode.npy'
    if output_file.exists():
        print(f"Model {model_name} already processed, skipping...")
        print(f"Output file exists: {output_file}")
        return

    # Load queries
    data_path = input_data_map[dataset][strategy]
    print(f"Loading queries from: {data_path}")
    df = pd.read_csv(data_path)
    queries = df['query_content'].tolist()
    print(f"Loaded {len(queries)} queries")

    # Encode queries
    embeddings = encode_queries_with_model(queries, MODEL_NAME_MAP[model_name], batch_size=batch_size)

    # Save
    np.save(output_file, embeddings)
    print(f"Saved query encodings to {output_file}")
    print(f"Shape: {embeddings.shape}")
    print("=" * 80)
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Construct Per-Model Query Encodings")
    parser.add_argument('--data', default='wildchat', choices=['wildchat'],
                       help='Dataset to analyze')
    parser.add_argument('--strategy', default='list_all', choices=['list_all', 'sample_1', 'sample_2'],
                       help='Strategy to analyze')
    parser.add_argument('--model', required=True,
                       help='Model name from MODEL_NAME_MAP to process')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for processing queries (default=32)')

    args = parser.parse_args()

    # Validate model name
    if args.model not in MODEL_NAME_MAP:
        raise ValueError(f"Model {args.model} not found in MODEL_NAME_MAP. "
                        f"Available models: {list(MODEL_NAME_MAP.keys())}")

    construct_query_encode_per_model(args.data, args.strategy, args.model, args.batch_size)