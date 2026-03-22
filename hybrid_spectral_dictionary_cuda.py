
"""
hybrid_spectral_dictionary_cuda.py

Larger CUDA-capable prototype for comparing:
1. SGD
2. Pure SECD (spectral projection only)
3. Hybrid SECD (mixture of full gradient + spectral projection)

Dataset format expected:
    WORD|definition text...

Example:
    A|A is the first letter of the alphabet...
    AAA|AAA refers to the highest credit rating...

This script builds a next-character prediction task over the WORD field only by default.
Optionally, it can also train on the definition text.

Main idea:
    For a gradient matrix G, construct a projector P_k from the top-k eigenvectors
    of an activation covariance matrix C, then use

        G_spec = P_k G

    or the hybrid form

        G_hybrid = alpha * G + (1 - alpha) * G_spec

Symbols:
    X         = activation/input matrix used to form the covariance
    C         = covariance matrix = (X^T X) / N
    v_i       = eigenvector i
    lambda_i  = eigenvalue i
    V_k       = matrix of top-k eigenvectors
    P_k       = V_k V_k^T
    G         = ordinary gradient
    G_spec    = projected spectral gradient
    alpha     = hybrid mixing coefficient
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------------------------------
# Reproducibility
# -------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------
# Dataset parsing
# -------------------------------------------------

def parse_word_definitions(
    path: Path,
    use_definitions: bool = False,
    lowercase: bool = True,
    min_word_len: int = 1,
    max_word_len: int = 32,
    max_entries: int | None = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Parse lines of the form:
        WORD|definition text

    Returns:
        texts   = training strings
        entries = list of (word, definition)
    """
    texts: List[str] = []
    entries: List[Tuple[str, str]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            word, definition = line.split("|", 1)
            word = word.strip()
            definition = definition.strip()

            if lowercase:
                word = word.lower()
                definition = definition.lower()

            if not (min_word_len <= len(word) <= max_word_len):
                continue

            if use_definitions:
                # Include a separator to preserve word/definition boundary.
                text = word + "||" + definition
            else:
                text = word

            texts.append(text)
            entries.append((word, definition))

            if max_entries is not None and len(texts) >= max_entries:
                break

    if not texts:
        raise ValueError(f"No valid entries found in {path}")

    return texts, entries


def build_vocab(texts: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = set("^$")
    for t in texts:
        chars.update(t)
    vocab = sorted(chars)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


class CharNextDataset(Dataset):
    """
    Next-character dataset with fixed context length.

    Each sample:
        input  = previous context_len characters
        target = next character
    """
    def __init__(self, texts: List[str], stoi: Dict[str, int], context_len: int = 8):
        self.stoi = stoi
        self.context_len = context_len
        self.samples: List[Tuple[List[int], int]] = []

        for t in texts:
            seq = "^" * context_len + t + "$"
            ids = [stoi[ch] for ch in seq]
            for pos in range(context_len, len(ids)):
                x = ids[pos - context_len:pos]
                y = ids[pos]
                self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def split_texts(texts: List[str], valid_fraction: float = 0.2, seed: int = 0) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    shuffled = list(texts)
    rng.shuffle(shuffled)
    n_valid = max(1, int(len(shuffled) * valid_fraction))
    valid = shuffled[:n_valid]
    train = shuffled[n_valid:]
    return train, valid


# -------------------------------------------------
# Model
# -------------------------------------------------

class CharMLP(nn.Module):
    """
    Character-level MLP:
        embedding -> flatten -> Linear -> GELU -> Linear

    The spectral filter is applied to the gradient of fc1.weight,
    using the current batch's flattened embedding activations.
    """
    def __init__(self, vocab_size: int, context_len: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(context_len * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

        self.last_flat_input = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)                     # [B, T, E]
        flat = emb.reshape(emb.shape[0], -1)       # [B, T*E]
        self.last_flat_input = flat.detach()
        h = F.gelu(self.fc1(flat))
        logits = self.fc2(h)
        return logits


# -------------------------------------------------
# Spectral projection utilities
# -------------------------------------------------

@torch.no_grad()
def topk_projector_from_batch(X: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Compute P_k = V_k V_k^T from batch activations X.

    X shape:
        [B, D]

    C = (X^T X) / B

    Returns:
        P_k of shape [D, D]
    """
    B, D = X.shape
    k = min(top_k, D)
    C = (X.T @ X) / max(B, 1)

    # Symmetric eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(C)
    idx = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, idx]
    V_k = eigvecs[:, :k]
    P_k = V_k @ V_k.T
    return P_k


@torch.no_grad()
def apply_spectral_or_hybrid_update_to_fc1(
    model: CharMLP,
    top_k: int,
    alpha: float,
    mode: str,
) -> None:
    """
    Modify model.fc1.weight.grad in-place.

    fc1 has shape:
        [H, D]
    PyTorch gradient shape is also [H, D]

    The projector acts on the input dimension D, so we use:
        G_spec = G @ P_k
    because each row of G corresponds to one output neuron over D inputs.

    Modes:
        - sgd     : leave gradient untouched
        - secd    : use only spectral projection
        - hybrid  : alpha * G + (1 - alpha) * G_spec
    """
    if mode == "sgd":
        return

    G = model.fc1.weight.grad
    if G is None:
        return

    X = model.last_flat_input
    if X is None:
        return

    P_k = topk_projector_from_batch(X, top_k=top_k)   # [D, D]
    G_spec = G @ P_k

    if mode == "secd":
        model.fc1.weight.grad.copy_(G_spec)
    elif mode == "hybrid":
        G_hybrid = alpha * G + (1.0 - alpha) * G_spec
        model.fc1.weight.grad.copy_(G_hybrid)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# -------------------------------------------------
# Training / evaluation
# -------------------------------------------------

@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    valid_loss: float
    valid_acc: float


@dataclass
class RunSummary:
    mode: str
    top_k: int
    alpha: float
    epochs: int
    batch_size: int
    embed_dim: int
    hidden_dim: int
    final_train_loss: float
    final_train_acc: float
    final_valid_loss: float
    final_valid_acc: float
    device: str


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        pred = logits.argmax(dim=-1)
        total_loss += loss.item()
        total_correct += (pred == y).sum().item()
        total_count += y.numel()

    return total_loss / total_count, total_correct / total_count


def train_one_run(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    vocab_size: int,
    context_len: int,
    embed_dim: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    top_k: int,
    alpha: float,
    mode: str,
    device: torch.device,
    seed: int,
) -> Tuple[CharMLP, List[EpochRecord], RunSummary]:
    set_seed(seed)

    model = CharMLP(
        vocab_size=vocab_size,
        context_len=context_len,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history: List[EpochRecord] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            apply_spectral_or_hybrid_update_to_fc1(
                model=model,
                top_k=top_k,
                alpha=alpha,
                mode=mode,
            )

            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                running_loss += loss.item() * y.numel()
                running_correct += (pred == y).sum().item()
                running_count += y.numel()

        train_loss = running_loss / running_count
        train_acc = running_correct / running_count
        valid_loss, valid_acc = evaluate(model, valid_loader, device)

        history.append(EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            valid_loss=valid_loss,
            valid_acc=valid_acc,
        ))

        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(
                f"[{mode.upper():>6}] epoch {epoch+1:4d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
            )

    final = history[-1]
    summary = RunSummary(
        mode=mode,
        top_k=top_k,
        alpha=alpha,
        epochs=epochs,
        batch_size=train_loader.batch_size or 0,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        final_train_loss=final.train_loss,
        final_train_acc=final.train_acc,
        final_valid_loss=final.valid_loss,
        final_valid_acc=final.valid_acc,
        device=str(device),
    )
    return model, history, summary


# -------------------------------------------------
# Generation
# -------------------------------------------------

@torch.no_grad()
def sample_text(
    model: CharMLP,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    context_len: int,
    device: torch.device,
    temperature: float = 1.0,
    max_steps: int = 24,
) -> str:
    model.eval()
    ctx = ["^"] * context_len
    out = []

    for _ in range(max_steps):
        x = torch.tensor([[stoi[ch] for ch in ctx]], dtype=torch.long, device=device)
        logits = model(x)[0] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ch = itos[next_id]
        if ch == "$":
            break
        out.append(ch)
        ctx = ctx[1:] + [ch]

    return "".join(out)


def save_history_csv(path: Path, history_by_mode: Dict[str, List[EpochRecord]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "epoch", "train_loss", "train_acc", "valid_loss", "valid_acc"])
        for mode, records in history_by_mode.items():
            for r in records:
                writer.writerow([mode, r.epoch, r.train_loss, r.train_acc, r.valid_loss, r.valid_acc])


def save_summary_csv(path: Path, summaries: List[RunSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="word_definitions.txt")
    parser.add_argument("--use-definitions", action="store_true")
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--context-len", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    data_path = Path(args.data)
    texts, entries = parse_word_definitions(
        data_path,
        use_definitions=args.use_definitions,
        max_entries=args.max_entries,
    )

    train_texts, valid_texts = split_texts(texts, valid_fraction=args.valid_fraction, seed=args.seed)
    stoi, itos = build_vocab(texts)

    train_ds = CharNextDataset(train_texts, stoi, context_len=args.context_len)
    valid_ds = CharNextDataset(valid_texts, stoi, context_len=args.context_len)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    print("Dataset summary")
    print("---------------")
    print(f"data path             = {data_path}")
    print(f"use definitions       = {args.use_definitions}")
    print(f"entries               = {len(entries)}")
    print(f"train texts           = {len(train_texts)}")
    print(f"valid texts           = {len(valid_texts)}")
    print(f"train samples         = {len(train_ds)}")
    print(f"valid samples         = {len(valid_ds)}")
    print(f"vocab size            = {len(stoi)}")
    print(f"context length        = {args.context_len}")
    print(f"embed dim             = {args.embed_dim}")
    print(f"hidden dim            = {args.hidden_dim}")
    print(f"epochs                = {args.epochs}")
    print(f"batch size            = {args.batch_size}")
    print(f"top-k                 = {args.top_k}")
    print(f"alpha                 = {args.alpha}")
    print(f"device                = {device}")

    history_by_mode: Dict[str, List[EpochRecord]] = {}
    summaries: List[RunSummary] = []
    trained_models: Dict[str, CharMLP] = {}

    for mode in ["sgd", "secd", "hybrid"]:
        model, history, summary = train_one_run(
            train_loader=train_loader,
            valid_loader=valid_loader,
            vocab_size=len(stoi),
            context_len=args.context_len,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            top_k=args.top_k,
            alpha=args.alpha,
            mode=mode,
            device=device,
            seed=args.seed,
        )
        history_by_mode[mode] = history
        summaries.append(summary)
        trained_models[mode] = model

    print("\nFinal summary")
    print("-------------")
    for s in summaries:
        print(
            f"{s.mode:>6} | "
            f"train_acc={s.final_train_acc:.4f} | valid_acc={s.final_valid_acc:.4f} | "
            f"train_loss={s.final_train_loss:.4f} | valid_loss={s.final_valid_loss:.4f}"
        )

    print("\nSampled generations")
    print("-------------------")
    for mode, model in trained_models.items():
        print(f"\n{mode.upper()}")
        for _ in range(8):
            print(" ", sample_text(
                model, stoi, itos,
                context_len=args.context_len,
                device=device,
                temperature=0.9,
                max_steps=24,
            ))

    out_dir = Path.cwd()
    save_history_csv(out_dir / "hybrid_spectral_history.csv", history_by_mode)
    save_summary_csv(out_dir / "hybrid_spectral_summary.csv", summaries)

    print(f"\nSaved history to: {out_dir / 'hybrid_spectral_history.csv'}")
    print(f"Saved summary to: {out_dir / 'hybrid_spectral_summary.csv'}")


if __name__ == "__main__":
    main()
