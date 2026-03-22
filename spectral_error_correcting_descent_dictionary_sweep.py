
"""
spectral_error_correcting_descent_dictionary_sweep.py

Extension of the earlier SECD prototype.

This script:
1. Loads a real dictionary-style word list if available.
2. Falls back to a synthetic word list if no dictionary file is found.
3. Builds a tiny next-character prediction dataset.
4. Compares:
      - SGD baseline
      - SECD (Spectral Error-Correcting Descent)
5. Sweeps over multiple top-k spectral subspace sizes.

Core equations:

Input covariance:
    C = (X^T X) / N

Eigen-decomposition:
    C v_i = lambda_i v_i

Ordinary gradient:
    G = (1/N) X^T (P - Y)

Spectral-filtered gradient:
    G_spec = V_k V_k^T G

Weight update:
    W <- W - eta G_spec

Symbols:
    X         = input matrix
    Y         = target matrix
    N         = number of samples
    C         = input covariance matrix
    v_i       = eigenvector i
    lambda_i  = eigenvalue i
    V_k       = matrix of the top-k eigenvectors
    G         = ordinary gradient
    G_spec    = spectral-filtered gradient
    W         = weight matrix
    eta       = learning rate
    P         = predicted class probabilities
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import csv
import math
import os
import re

import numpy as np


# -------------------------------------------------
# Dataset helpers
# -------------------------------------------------

def build_synthetic_words() -> List[str]:
    stems = ["bio", "neuro", "astro", "quant", "logic", "graph", "phase", "field", "wave", "code"]
    suffixes = ["form", "scope", "loop", "path", "state", "shift", "craft", "mind", "stack", "trace"]
    words = [a + b for a in stems for b in suffixes]
    words += [
        "alphaalpha", "betabeta", "gammagamma",
        "looploop", "statecraft", "mindfield", "fieldmind",
        "wavepath", "logicloop", "graphstate",
    ]
    return words


def normalize_word(word: str) -> str:
    """
    Keep only lowercase alphabetic words.
    """
    word = word.strip().lower()
    if not word:
        return ""
    if not re.fullmatch(r"[a-z]+", word):
        return ""
    return word


def load_dictionary_words(
    max_words: int = 5000,
    min_len: int = 3,
    max_len: int = 12,
) -> Tuple[List[str], str]:
    """
    Try a few common local dictionary locations.
    If none are available, fall back to a synthetic set.
    """
    candidates = [
        Path("/usr/share/dict/words"),
        Path("/usr/dict/words"),
        Path("/usr/share/dict/web2"),
    ]

    collected: List[str] = []
    source = ""

    for path in candidates:
        if path.exists() and path.is_file():
            seen = set()
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    w = normalize_word(line)
                    if not w:
                        continue
                    if len(w) < min_len or len(w) > max_len:
                        continue
                    if w in seen:
                        continue
                    seen.add(w)
                    collected.append(w)
                    if len(collected) >= max_words:
                        break
            if collected:
                source = str(path)
                return collected, source

    collected = build_synthetic_words()
    source = "synthetic_fallback"
    return collected, source


def build_word_dataset(
    words: List[str],
    context_len: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Build a next-character prediction dataset with concatenated one-hot inputs.

    For each example:
        input  = previous context_len characters
        target = next character
    """
    chars = set("^$")
    sequences = []

    for w in words:
        seq = "^" * context_len + w + "$"
        sequences.append(seq)
        chars.update(seq)

    vocab = sorted(chars)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}

    V = len(vocab)
    D_in = context_len * V

    X_rows = []
    Y_rows = []

    for seq in sequences:
        for t in range(context_len, len(seq)):
            ctx = seq[t - context_len:t]
            nxt = seq[t]

            x = np.zeros(D_in, dtype=np.float64)
            for j, ch in enumerate(ctx):
                x[j * V + stoi[ch]] = 1.0

            y = np.zeros(V, dtype=np.float64)
            y[stoi[nxt]] = 1.0

            X_rows.append(x)
            Y_rows.append(y)

    X = np.stack(X_rows, axis=0)
    Y = np.stack(Y_rows, axis=0)
    return X, Y, stoi, itos


def train_valid_split(
    X: np.ndarray,
    Y: np.ndarray,
    valid_fraction: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    N = len(X)
    idx = np.arange(N)
    rng.shuffle(idx)

    n_valid = max(1, int(valid_fraction * N))
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]

    return X[train_idx], Y[train_idx], X[valid_idx], Y[valid_idx]


# -------------------------------------------------
# Math helpers
# -------------------------------------------------

def softmax(z: np.ndarray) -> np.ndarray:
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-np.mean(np.sum(y * np.log(p), axis=1)))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(p, axis=1)
    true = np.argmax(y, axis=1)
    return float(np.mean(pred == true))


def eig_sorted_symmetric(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


@dataclass
class TrainHistory:
    name: str
    train_loss: List[float]
    train_acc: List[float]
    valid_loss: List[float]
    valid_acc: List[float]
    W: np.ndarray


def evaluate_model(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> Tuple[float, float]:
    p = softmax(X @ W)
    return cross_entropy(p, Y), accuracy(p, Y)


# -------------------------------------------------
# Optimizers
# -------------------------------------------------

def train_sgd(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_valid: np.ndarray,
    Y_valid: np.ndarray,
    epochs: int = 200,
    lr: float = 1.0,
    seed: int = 0,
) -> TrainHistory:
    rng = np.random.default_rng(seed)
    N, D_in = X_train.shape
    V = Y_train.shape[1]
    W = 0.01 * rng.standard_normal((D_in, V))

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []

    for _ in range(epochs):
        p = softmax(X_train @ W)
        G = (X_train.T @ (p - Y_train)) / N
        W = W - lr * G

        tl, ta = evaluate_model(X_train, Y_train, W)
        vl, va = evaluate_model(X_valid, Y_valid, W)

        train_loss.append(tl)
        train_acc.append(ta)
        valid_loss.append(vl)
        valid_acc.append(va)

    return TrainHistory("SGD", train_loss, train_acc, valid_loss, valid_acc, W)


def train_secd(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_valid: np.ndarray,
    Y_valid: np.ndarray,
    epochs: int = 200,
    lr: float = 1.0,
    top_k: int = 8,
    seed: int = 0,
) -> Tuple[TrainHistory, np.ndarray]:
    rng = np.random.default_rng(seed)
    N, D_in = X_train.shape
    V = Y_train.shape[1]
    W = 0.01 * rng.standard_normal((D_in, V))

    C = (X_train.T @ X_train) / N
    eigvals, eigvecs = eig_sorted_symmetric(C)
    k = min(top_k, D_in)
    V_k = eigvecs[:, :k]
    P_k = V_k @ V_k.T

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []

    for _ in range(epochs):
        p = softmax(X_train @ W)
        G = (X_train.T @ (p - Y_train)) / N
        G_spec = P_k @ G
        W = W - lr * G_spec

        tl, ta = evaluate_model(X_train, Y_train, W)
        vl, va = evaluate_model(X_valid, Y_valid, W)

        train_loss.append(tl)
        train_acc.append(ta)
        valid_loss.append(vl)
        valid_acc.append(va)

    history = TrainHistory(f"SECD_topk={k}", train_loss, train_acc, valid_loss, valid_acc, W)
    return history, eigvals


# -------------------------------------------------
# Inspection helpers
# -------------------------------------------------

def decode_next_char_distribution(
    W: np.ndarray,
    context: str,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    context_len: int = 3,
    top_n: int = 8,
) -> List[Tuple[str, float]]:
    V = len(stoi)
    if len(context) < context_len:
        context = "^" * (context_len - len(context)) + context
    context = context[-context_len:]

    x = np.zeros((1, context_len * V), dtype=np.float64)
    for j, ch in enumerate(context):
        if ch in stoi:
            x[0, j * V + stoi[ch]] = 1.0

    p = softmax(x @ W)[0]
    order = np.argsort(p)[::-1][:top_n]
    return [(itos[i], float(p[i])) for i in order]


def generate_word(
    W: np.ndarray,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    context_len: int = 3,
    max_steps: int = 16,
) -> str:
    ctx = "^" * context_len
    out = []
    for _ in range(max_steps):
        items = decode_next_char_distribution(W, ctx, stoi, itos, context_len=context_len, top_n=1)
        ch = items[0][0]
        if ch == "$":
            break
        out.append(ch)
        ctx = ctx[1:] + ch
    return "".join(out)


def save_histories_csv(out_path: Path, histories: List[TrainHistory]) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epoch", "train_loss", "train_acc", "valid_loss", "valid_acc"])
        for hist in histories:
            for epoch in range(len(hist.train_loss)):
                writer.writerow([
                    hist.name,
                    epoch,
                    hist.train_loss[epoch],
                    hist.train_acc[epoch],
                    hist.valid_loss[epoch],
                    hist.valid_acc[epoch],
                ])


def print_summary(history: TrainHistory) -> None:
    print(f"\n=== {history.name} ===")
    print(f"final train loss : {history.train_loss[-1]:.6f}")
    print(f"final train acc  : {history.train_acc[-1]:.4f}")
    print(f"final valid loss : {history.valid_loss[-1]:.6f}")
    print(f"final valid acc  : {history.valid_acc[-1]:.4f}")


# -------------------------------------------------
# Main experiment
# -------------------------------------------------

def main():
    max_words = 4000
    context_len = 3
    epochs = 200
    learning_rate = 1.0
    valid_fraction = 0.2
    seed = 0

    top_k_values = [2, 4, 8, 16, 32]

    words, source = load_dictionary_words(max_words=max_words, min_len=3, max_len=12)
    X, Y, stoi, itos = build_word_dataset(words, context_len=context_len)
    X_train, Y_train, X_valid, Y_valid = train_valid_split(X, Y, valid_fraction=valid_fraction, seed=seed)

    print("Dataset summary")
    print("---------------")
    print(f"word source              = {source}")
    print(f"number of words          = {len(words)}")
    print(f"number of samples        = {len(X)}")
    print(f"training samples         = {len(X_train)}")
    print(f"validation samples       = {len(X_valid)}")
    print(f"vocabulary size          = {len(stoi)}")
    print(f"context length           = {context_len}")
    print(f"input dimension D_in     = {X.shape[1]}")
    print(f"output dimension V       = {Y.shape[1]}")
    print(f"epochs                   = {epochs}")
    print(f"learning rate            = {learning_rate}")
    print(f"top-k sweep              = {top_k_values}")

    histories: List[TrainHistory] = []

    baseline = train_sgd(
        X_train, Y_train, X_valid, Y_valid,
        epochs=epochs, lr=learning_rate, seed=seed,
    )
    histories.append(baseline)
    print_summary(baseline)

    eigvals_cache = None
    for k in top_k_values:
        hist, eigvals = train_secd(
            X_train, Y_train, X_valid, Y_valid,
            epochs=epochs, lr=learning_rate, top_k=k, seed=seed,
        )
        histories.append(hist)
        print_summary(hist)
        if eigvals_cache is None:
            eigvals_cache = eigvals

    if eigvals_cache is not None:
        print("\nTop eigenvalues of the input covariance matrix")
        print("----------------------------------------------")
        print(eigvals_cache[: min(12, len(eigvals_cache))])

    print("\nRepresentative next-character distributions")
    print("------------------------------------------")
    test_contexts = ["^^^", "^^a", "^^s", "com", "pro", "ing", "sta", "ion"]
    for ctx in test_contexts:
        print(f"\nContext = {ctx!r}")
        for hist in histories[:3]:
            print(f"{hist.name:>12}: {decode_next_char_distribution(hist.W, ctx, stoi, itos, context_len=context_len)}")

    print("\nGreedy generations")
    print("------------------")
    for hist in histories[:3]:
        print(f"\n{hist.name}")
        for i in range(6):
            print(f"  sample {i+1}: {generate_word(hist.W, stoi, itos, context_len=context_len)}")

    out_dir = Path.cwd()
    csv_path = out_dir / "spectral_dictionary_sweep_results.csv"
    save_histories_csv(csv_path, histories)
    print(f"\nSaved epoch-by-epoch results to: {csv_path}")


if __name__ == "__main__":
    main()
