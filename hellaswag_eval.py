"""
Downloads and evaluates HellaSwag examples using a GPT-2 language model.

HellaSwag dataset: https://github.com/rowanz/hellaswag

Usage:
    python hellaswag_eval.py -m gpt2 -d cuda
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

# Directory to cache HellaSwag data files
DATA_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "hellaswag"
)

# URLs for HellaSwag splits
HELLASWAG_URLS = {
    "train": (
        "https://raw.githubusercontent.com/rowanz/"
        "hellaswag/master/data/hellaswag_train.jsonl"
    ),
    "val": (
        "https://raw.githubusercontent.com/rowanz/"
        "hellaswag/master/data/hellaswag_val.jsonl"
    ),
    "test": (
        "https://raw.githubusercontent.com/rowanz/"
        "hellaswag/master/data/hellaswag_test.jsonl"
    ),
}

# Initialize GPT-2 tokenizer
ENC = tiktoken.get_encoding("gpt2")


def download_file(url: str, fname: str, chunk_size: int = 1024):
    """
    Download a file from the given URL and save it to fname.

    Args:
        url: URL to download from.
        fname: Local filename to write.
        chunk_size: Chunk size for streaming download.
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download(split: str):
    """
    Ensure that the HellaSwag JSONL file for the split is downloaded.

    Args:
        split: One of "train", "val", or "test".
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = HELLASWAG_URLS[split]
    data_filename = os.path.join(
        DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"
    )
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)


def render_example(example: dict):
    """
    Convert a HellaSwag example to token and mask tensors.

    Args:
        example: Dictionary containing keys "ctx", "label", "endings".

    Returns:
        data: Dict with 'label', 'ctx_tokens', 'ending_tokens'.
        tokens: Tensor of shape (4, max_len) with token IDs.
        mask: Tensor of shape (4, max_len) where 1 marks completion part.
        label: Integer index of the correct ending.
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # Encode context tokens once
    ctx_tokens = ENC.encode(ctx)
    data = {
        "label": label,
        "ctx_tokens": ctx_tokens,
        "ending_tokens": [],
    }

    tok_rows = []
    mask_rows = []
    for end in endings:
        # Prepend space so tokenizer separates properly
        end_tokens = ENC.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # Determine maximum sequence length among 4 candidates
    max_len = max(len(r) for r in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    # Populate token and mask tensors
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        length = len(tok_row)
        tokens[i, :length] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split: str):
    """
    Generator that yields each JSON-decoded HellaSwag example.

    Args:
        split: One of "train", "val", or "test".

    Yields:
        example: Parsed JSON object for each line.
    """
    download(split)
    file_path = os.path.join(
        DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                yield example
            except json.JSONDecodeError as e:
                print(f"[!] Skipping malformed line {i}: {e}")


@torch.no_grad()
def evaluate(model_type: str, device: str):
    """
    Evaluate a GPT-2 model on the HellaSwag validation set.

    Args:
        model_type: HuggingFace model identifier, e.g., "gpt2".
        device: Device string ("cpu" or "cuda").
    """
    # Use TF32 precision for faster matmuls on supported GPUs
    torch.set_float32_matmul_precision("high")
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)

    num_correct = 0
    num_correct_norm = 0
    num_total = 0

    for example in iterate_examples("val"):
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Compute logits for all token positions
        logits = model(tokens).logits

        # Shift logits and tokens for next-token loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_targets = shift_tokens.view(-1)

        # Compute cross-entropy loss per token
        shift_losses = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        )
        shift_losses = shift_losses.view(tokens.size(0), -1)

        # Align mask to shifted positions
        shift_mask = mask[..., 1:].contiguous()
        masked_losses = shift_losses * shift_mask

        # Sum and average loss over completion tokens
        sum_loss = masked_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        # Pick candidate with minimum loss
        pred = int(sum_loss.argmin().item())
        pred_norm = int(avg_loss.argmin().item())

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        print(
            f"{num_total}  acc_norm: "
            f"{num_correct_norm}/{num_total}="
            f"{num_correct_norm/num_total:.4f}"
        )

        # Print first few examples for debugging
        if num_total < 10:
            print("---")
            print(f"Context:\n{example['ctx']}")
            print("Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate GPT-2 on HellaSwag validation set."
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default="gpt2",
        help="HuggingFace model type (e.g., 'gpt2')",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on ('cpu' or 'cuda')",
    )
    args = parser.parse_args()
    evaluate(args.model_type, args.device)
