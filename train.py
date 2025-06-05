import inspect
import tiktoken
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import numpy as np
import os
from hellaswag import render_example, iterate_examples


class CausalSelfAttention(nn.Module):
    """
    Implements a causal self-attention layer using PyTorch's
    scaled_dot_product_attention for autoregressive modeling.
    """

    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                "Embedding dimension must be divisible by num heads"
            )
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Buffer for causal mask: triangular lower matrix for positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        """
        Forward pass for causal self-attention.
        Args:
            x: Tensor of shape (B, T, C) where B=batch size,
               T=sequence length, C=embedding dim.
        Returns:
            Output tensor of same shape as input.
        """
        B, T, C = x.size()
        # Compute combined projections for q, k, v
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Reshape and transpose for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Compute scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Combine heads and apply output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """
    Implements a feed-forward layer with GELU activation
    and projection, commonly used in transformer blocks.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        """
        Forward pass for feed-forward network.
        Args:
            x: Tensor of shape (B, T, C).
        Returns:
            Tensor of same shape after transformation.
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


class Block(nn.Module):
    """
    A single transformer block consisting of layer norms,
    causal self-attention, and MLP layers with residuals.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Forward pass for transformer block.
        Args:
            x: Tensor of shape (B, T, C).
        Returns:
            Transformed tensor of same shape.
        """
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


@dataclass
class GPTConfig:
    """
    Configuration dataclass for GPT model hyperparameters.
    Attributes:
        block_size: Maximum sequence length.
        vocab_size: Size of tokenizer vocabulary.
        n_layer: Number of transformer blocks.
        n_head: Number of attention heads.
        n_embd: Dimensionality of embeddings.
    """

    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    """
    GPT language model implementation with token and position
    embeddings, transformer blocks, and output head.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # Token and position embeddings
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(
                    config.vocab_size, config.n_embd
                ),
                "wpe": nn.Embedding(
                    config.block_size, config.n_embd
                ),
                "h": nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        # Language modeling head: projects to vocab size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size,
                                 bias=False)
        # Share weights between token embeddings and head
        self.transformer.wte.weight = self.lm_head.weight
        # Initialize all parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for linear and embedding layers using normal
        distribution. Adjust std based on layer count for stability.
        """
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std += (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        """
        Forward pass for GPT model.
        Args:
            idx: Tensor of token indices of shape (B, T).
            targets: Tensor of target indices for computing loss,
                     same shape as idx.
        Returns:
            logits: Tensor of shape (B, T, vocab_size).
            loss: Cross-entropy loss if targets provided, else None.
        """
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block size "
                f"{self.config.block_size}"
            )
        # Compute position indices and embeddings
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        # Compute token embeddings
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Load pretrained GPT-2 weights from HuggingFace and map them to
        this model's architecture.
        Args:
            model_type: One of "gpt2", "gpt2-medium", "gpt2-large",
                        or "gpt2-xl".
        Returns:
            GPT model instance with loaded weights.
        """
        if model_type not in {
            "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
        }:
            raise ValueError(f"Unsupported model type: {model_type}")
        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained GPT: {model_type}")
        # Map model_type to configuration parameters
        config_map = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }
        config_args = config_map[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        # Initialize model from scratch
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        # Remove attention bias buffers from state dict keys
        sd_keys = [k for k in sd if not k.endswith(".attn.bias")]
        # Load HuggingFace model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # Filter out buffer weights
        hf_keys = [
            k for k in sd_hf
            if not k.endswith(".attn.masked_bias")
            and not k.endswith(".attn.bias")
        ]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        if len(hf_keys) != len(sd_keys):
            raise RuntimeError(
                f"Mismatch in parameter keys: {len(hf_keys)} vs "
                f"{len(sd_keys)}"
            )
        for k in hf_keys:
            target = k
            if any(k.endswith(w) for w in transposed):
                # Transpose Conv1D weights to Linear
                if sd_hf[k].shape[::-1] != sd[target].shape:
                    raise RuntimeError(
                        f"Shape mismatch for {k}: "
                        f"{sd_hf[k].shape} vs {sd[target].shape}"
                    )
                with torch.no_grad():
                    sd[target].copy_(sd_hf[k].t())
            else:
                if sd_hf[k].shape != sd[target].shape:
                    raise RuntimeError(
                        f"Shape mismatch for {k}: "
                        f"{sd_hf[k].shape} vs {sd[target].shape}"
                    )
                with torch.no_grad():
                    sd[target].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """
        Prepare AdamW optimizer with weight decay applied to
        parameters with dim >=2.
        Args:
            weight_decay: Float for weight decay factor.
            learning_rate: Initial learning rate.
            device_type: "cpu" or "cuda" for optimizer options.
        Returns:
            Configured AdamW optimizer.
        """
        # Separate parameters by dimension for weight decay
        param_dict = {
            name: p for name, p in self.named_parameters() if p.requires_grad
        }
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        print(
            f"Num decayed param tensors: {len(decay_params)}, "
            f"num params: {sum(p.numel() for p in decay_params):,}"
        )
        print(
            f"Num non-decayed param tensors: {len(nodecay_params)}, "
            f"num params: {sum(p.numel() for p in nodecay_params):,}"
        )
        # Use fused AdamW if available and on CUDA
        fused_fn = inspect.signature(torch.optim.AdamW).parameters
        use_fused = "fused" in fused_fn and device_type == "cuda"
        print(f"Using fused AdamW: {use_fused}")
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )


def load_tokens(filename):
    """
    Load tokens from a .npy file and convert to torch tensor.
    Args:
        filename: Path to numpy file containing integer tokens.
    Returns:
        Tensor of type torch.long.
    """
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    """
    Lightweight data loader that reads token shards from disk
    and provides batches for training and validation.
    """

    def __init__(self, B, T, split):
        """
        Args:
            B: Batch size (number of sequences per batch).
            T: Sequence length (tokens per sequence).
            split: "train" or "val" to select shard files.
        """
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.B = B
        self.T = T
        data_root = "edu_fineweb10B"
        shards = [
            os.path.join(data_root, s)
            for s in sorted(os.listdir(data_root))
            if split in s
        ]
        if not shards:
            raise RuntimeError(f"No shards found for split {split}")
        self.shards = shards
        print(f"Found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        """
        Reset shard index and load first shard into memory.
        """
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[0])
        self.current_position = 0

    def next_batch(self):
        """
        Retrieve next batch of tokens and wrap around shards.
        Returns:
            x: Input tensor of shape (B, T).
            y: Target tensor of shape (B, T).
        """
        B, T = self.B, self.T
        # If we don't have enough tokens left for a full batch, move to the
        # next shard before constructing the view. This prevents view errors
        # when the slice is smaller than B*T+1.
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0

        end_pos = self.current_position + B * T + 1
        buf = self.tokens[self.current_position:end_pos]

        if buf.numel() < B * T + 1:
            raise RuntimeError(
                "Shard does not contain enough tokens for a full batch"
            )

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        return x, y


def get_most_likely_row(tokens, mask, logits):
    """
    Select the completion with lowest average cross-entropy loss
    across masked positions.
    Args:
        tokens: Tensor of token indices (B, T).
        mask: Tensor mask indicating prompt vs completion (B, T).
        logits: Model output logits (B, T, vocab_size).
    Returns:
        Index of row with minimal loss among completions.
    """
    # Shift logits and tokens for autoregressive loss computation
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_targets = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_logits, flat_targets, reduction="none"
    ).view(tokens.size(0), -1)
    # Shift mask to align with completion tokens
    shift_mask = mask[..., 1:].contiguous()
    masked_losses = shift_losses * shift_mask
    sum_loss = masked_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    return int(avg_loss.argmin().item())


# Determine device: use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Training hyperparameters
total_batch_size = 524288  # desired total tokens per step
B = 32  # micro batch size
T = 1024  # sequence length
if total_batch_size % (B * T) != 0:
    raise ValueError("total_batch_size must be divisible by B * T")
grad_accum_steps = total_batch_size // (B * T)

print(f"Total batch size: {total_batch_size}")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Initialize data loaders
train_loader = DataLoaderLite(B=B, T=T, split="train")
val_loader = DataLoaderLite(B=B, T=T, split="val")

# Set high precision for matmul
torch.set_float32_matmul_precision("high")

# Initialize GPT model and move to device
model = GPT(GPTConfig(vocab_size=50304))
model = model.to(device)

# Optionally compile model; disabled due to eval issues
use_compile = False
if use_compile:
    model = torch.compile(model)

# Learning rate schedule parameters
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073


def get_lr(it):
    """
    Compute learning rate with linear warmup and cosine decay.
    Args:
        it: Current training step (int).
    Returns:
        Learning rate for given step.
    """
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coef * (max_lr - min_lr)


# Set random seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Tokenizer encoding
enc = tiktoken.get_encoding("gpt2")

# Configure optimizer with weight decay settings
optimizer = model.configure_optimizers(
    weight_decay=0.1, learning_rate=max_lr, device_type=device
)

# Setup logging and checkpoint paths
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
checkpoint_path = os.path.join(log_dir, "latest.pt")
start_step = 0

# Resume from checkpoint if available
if os.path.exists(checkpoint_path):
    print(f"Resuming training from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        rng_state = checkpoint["cuda_rng_state"]
        if isinstance(rng_state, list):
            for i, state in enumerate(rng_state):
                if isinstance(state, torch.ByteTensor):
                    torch.cuda.set_rng_state(state, i)
        else:
            print("Warning: Unexpected cuda_rng_state format. Skipping restore.")
    start_step = checkpoint["step"]
    print(f"Resumed from step {start_step}")

# Main training loop
for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1

    # Validation evaluation every 100 steps or at last step
    if step % 100 == 0 or last_step:
        model.eval()
        val_loader.reset()
        val_loss_accum = 0.0
        val_loss_steps = 20
        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                val_loss_accum += (loss / val_loss_steps).detach()
        val_loss = val_loss_accum.item()
        print(f"validation loss: {val_loss:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss:.4f}\n")
        # Save checkpoint
        if step > 0 and (step % 100 == 0 or last_step):
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": model.config,
                "step": step,
                "val_loss": val_loss_accum,
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(
                checkpoint,
                os.path.join(log_dir, f"model_{step:05d}.pt")
            )

    # HellaSwag evaluation every 100 steps (skip if compiling)
    if (step % 100 == 0 or last_step) and not use_compile:
        num_correct = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            _, tokens, mask, label = render_example(example)
            tokens, mask = tokens.to(device), mask.to(device)
            with torch.no_grad():
                with torch.autocast(
                    device_type=device, dtype=torch.bfloat16
                ):
                    logits, _ = model(tokens)
                pred = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct += int(pred == label)
        acc = num_correct / num_total
        print(f"HellaSwag accuracy: {num_correct}/{num_total}={acc:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} hella {acc:.4f}\n")

    # Generation samples every 100 steps (skip if compiling)
    if ((step > 0 and step % 100 == 0) or last_step) and not use_compile:
        model.eval()
        num_return = 4
        max_length = 32
        prompt = "Hello, I'm a language model,"
        tokens = torch.tensor(
            enc.encode(prompt), dtype=torch.long
        ).unsqueeze(0).repeat(num_return, 1)
        xgen = tokens.to(device)
        rng = torch.Generator(device=device)
        rng.manual_seed(42)
        # Autoregressive generation loop
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(
                    device_type=device, dtype=torch.bfloat16
                ):
                    logits, _ = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        # Print generated samples
        for i in range(num_return):
            decoded = enc.decode(xgen[i].tolist())
            print(f"sample {i}: {decoded}")

    # Training step
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for _ in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    # Gradient clipping and optimizer step
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000  # ms
    tokens_per_sec = B * T * grad_accum_steps / (dt / 1000)
    print(
        f"step {step:5d} | loss: {loss_accum:.6f} | "
        f"lr {lr:.4e} | norm: {norm:.4f} | "
        f"dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
    )
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum:.6f}\n")
