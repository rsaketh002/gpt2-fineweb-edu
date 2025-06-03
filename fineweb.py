"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Downloads and tokenizes the data and saves data shards to disk.

Run simply as:
    python fineweb.py

Will save shards to local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Directory to store token shards
LOCAL_DIR = "edu_fineweb10B"
REMOTE_NAME = "sample-10BT"
SHARD_SIZE = int(1e8)  # 100M tokens per shard, ~100 shards total

# Create local cache for shards if not present
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), LOCAL_DIR)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Load dataset split named REMOTE_NAME from HuggingFace
fw = load_dataset(
    "HuggingFaceFW/fineweb-edu", name=REMOTE_NAME, split="train"
)

# Initialize GPT-2 tokenizer and get end-of-text token ID
enc = tiktoken.get_encoding("gpt2")
EOT_TOKEN = enc._special_tokens["<|endoftext|>"]


def tokenize(doc):
    """
    Tokenize a single document and return a numpy array of
    uint16 tokens. Insert EOT_TOKEN before each document.

    Args:
        doc (dict): A dataset example with key "text".

    Returns:
        numpy.ndarray: Array of token IDs with dtype uint16.
    """
    # Prepend end-of-text marker to separate documents
    tokens = [EOT_TOKEN]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.int32)

    # Ensure token IDs fit in uint16
    if not (tokens_np.min() >= 0 and tokens_np.max() < 2**16):
        raise ValueError("Token IDs exceed uint16 range.")
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    """
    Save a numpy array of tokens to disk as .npy.

    Args:
        filename (str): Path to save file (no extension).
        tokens_np (numpy.ndarray): Array of token IDs.
    """
    np.save(filename, tokens_np)


def main():
    """
    Tokenize dataset and write shards of size SHARD_SIZE to disk.
    The first shard is used for validation, others for training.
    """
    nprocs = max(1, os.cpu_count() // 2)

    # Pool for parallel tokenization of documents
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # Preallocate buffer for the current shard
        buffer = np.empty((SHARD_SIZE,), dtype=np.uint16)
        token_count = 0
        progress = None

        # Iterate over tokenized documents
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            # If tokens fit in current shard buffer
            if token_count + len(tokens) < SHARD_SIZE:
                buffer[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)

                # Initialize or update progress bar
                if progress is None:
                    desc = f"Shard {shard_index:06d}"
                    progress = tqdm(
                        total=SHARD_SIZE, unit="tokens", desc=desc
                    )
                progress.update(len(tokens))

            else:
                # Write the current shard to disk
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    DATA_CACHE_DIR,
                    f"edufineweb_{split}_{shard_index:06d}"
                )

                # Fill the remainder of the shard
                remainder = SHARD_SIZE - token_count
                progress.update(remainder)
                buffer[token_count : token_count + remainder] = tokens[:remainder]
                write_datafile(filename, buffer)

                shard_index += 1
                progress.close()
                progress = None

                # Start new shard with leftover tokens
                leftover = tokens[remainder:]
                buffer[: len(leftover)] = leftover
                token_count = len(leftover)

        # Write any remaining tokens as the last shard
        if token_count > 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR,
                f"edufineweb_{split}_{shard_index:06d}"
            )
            write_datafile(filename, buffer[:token_count])


if __name__ == "__main__":
    main()
