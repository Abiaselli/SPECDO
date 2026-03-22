
"""
spectral_error_correcting_descent_demo.py

Toy prototype for a "spectral error-correcting" optimizer on a tiny next-character task.

Core idea:
    Instead of applying the full gradient G directly,
    project G onto the top-k eigen-directions of the input covariance matrix.

Mathematically:
    C = (X^T X) / N
    C v_i = lambda_i v_i

Let V_k be the matrix whose columns are the top-k eigenvectors.
Then the projected gradient is:
    G_spec = V_k V_k^T G

Where:
    X       = input design matrix
    N       = number of samples
    C       = input covariance matrix
    v_i     = eigenvector i
    lambda_i= eigenvalue for eigenvector i
    G       = full gradient of the loss with respect to the weight matrix
    G_spec  = spectral-filtered gradient
    V_k     = top-k eigenvector matrix

This is not a replacement for modern optimizers yet.
It is a small, testable prototype for your broader idea.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np


# ----------------------------
# Dataset builders
# ----------------------------

def build_synthetic_words() -> List[str]:
    """
    Create a tiny structured dataset with repeated motifs.
    This gives the optimizer some recurring statistical structure to exploit.
    """
    stems = ["bio", "neuro", "astro", "quant", "logic", "graph", "phase", "field", "wave", "code"]
    suffixes = ["form", "scope", "loop", "path", "state", "shift", "craft", "mind", "stack", "trace"]
    words = []
    for a in stems:
        for b in suffixes:
            words.append(a + b)
    # add some repeated patterns for stronger recurrence structure
    words += [
        "alphaalpha", "betabeta", "gammagamma",
        "looploop", "statecraft", "mindfield", "fieldmind",
        "wavepath", "logicloop", "graphstate"
    ]
    return words


def build_word_dataset(words: List[str], context_len: int = 2) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Build a next-character prediction dataset.

    Each training example:
        input  = previous `context_len` characters
        target = next character

    We pad with '^' as start token and '$' as end token.
    Input representation is concatenated one-hot vectors.

    Returns:
        X      = array of shape [N, D_in]
        Y      = array of shape [N, V]
        stoi   = char -> index
        itos   = index -> char
    """
    seqs = []
    chars = set("^$ ")
    for w in words:
        seq = "^" * context_len + w + "$"
        seqs.append(seq)
        chars.update(seq)

    vocab = sorted(chars)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    V = len(vocab)
    D_in = context_len * V

    X_rows = []
    Y_rows = []

    for seq in seqs:
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


# ----------------------------
# Model + math helpers
# ----------------------------

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Row-wise softmax.

    z shape: [N, V]
    """
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Mean cross-entropy:
        L = -(1/N) sum_n sum_c y[n,c] log p[n,c]
    """
    p = np.clip(p, eps, 1.0)
    return float(-np.mean(np.sum(y * np.log(p), axis=1)))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    """
    Classification accuracy.
    """
    pred = np.argmax(p, axis=1)
    true = np.argmax(y, axis=1)
    return float(np.mean(pred == true))


@dataclass
class TrainResult:
    name: str
    losses: List[float]
    accs: List[float]
    W: np.ndarray


def eig_sorted_symmetric(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a symmetric matrix C:
        C v_i = lambda_i v_i

    Returns eigenvalues and eigenvectors sorted by descending eigenvalue.
    """
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs


# ----------------------------
# Training procedures
# ----------------------------

def train_sgd(
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 250,
    lr: float = 1.0,
    seed: int = 0
) -> TrainResult:
    """
    Baseline full-gradient descent on a linear softmax model.

    Model:
        logits = X W
        p = softmax(logits)

    Gradient:
        G = (1/N) X^T (p - Y)

    Update:
        W <- W - lr * G
    """
    rng = np.random.default_rng(seed)
    N, D_in = X.shape
    V = Y.shape[1]
    W = 0.01 * rng.standard_normal((D_in, V))

    losses = []
    accs = []

    for _ in range(epochs):
        logits = X @ W
        p = softmax(logits)
        G = (X.T @ (p - Y)) / N

        W = W - lr * G

        losses.append(cross_entropy(p, Y))
        accs.append(accuracy(p, Y))

    return TrainResult(name="SGD", losses=losses, accs=accs, W=W)


def train_secd(
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 250,
    lr: float = 1.0,
    top_k: int = 16,
    seed: int = 0
) -> TrainResult:
    """
    Spectral Error-Correcting Descent (toy version).

    Step 1:
        Build input covariance:
            C = (1/N) X^T X

    Step 2:
        Eigendecompose C:
            C v_i = lambda_i v_i

    Step 3:
        Keep top-k eigenvectors in V_k

    Step 4:
        Compute gradient:
            G = (1/N) X^T (p - Y)

    Step 5:
        Project gradient onto dominant input directions:
            G_spec = V_k V_k^T G

    Update:
        W <- W - lr * G_spec

    Interpretation:
        Only update along statistically dominant input modes.
    """
    rng = np.random.default_rng(seed)
    N, D_in = X.shape
    V = Y.shape[1]
    W = 0.01 * rng.standard_normal((D_in, V))

    C = (X.T @ X) / N
    eigvals, eigvecs = eig_sorted_symmetric(C)
    k = min(top_k, D_in)
    V_k = eigvecs[:, :k]
    P_k = V_k @ V_k.T  # projection matrix

    losses = []
    accs = []

    for _ in range(epochs):
        logits = X @ W
        p = softmax(logits)
        G = (X.T @ (p - Y)) / N
        G_spec = P_k @ G

        W = W - lr * G_spec

        losses.append(cross_entropy(p, Y))
        accs.append(accuracy(p, Y))

    print("\nTop eigenvalues of input covariance:")
    print(eigvals[:min(10, len(eigvals))])

    return TrainResult(name=f"SECD_topk={k}", losses=losses, accs=accs, W=W)


# ----------------------------
# Evaluation utilities
# ----------------------------

def decode_next_char_distribution(
    W: np.ndarray,
    context: str,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    context_len: int = 2,
    top_n: int = 8
) -> List[Tuple[str, float]]:
    """
    Inspect the model's next-character distribution for a given context.
    """
    V = len(stoi)
    if len(context) < context_len:
        context = "^" * (context_len - len(context)) + context
    context = context[-context_len:]

    x = np.zeros((1, context_len * V), dtype=np.float64)
    for j, ch in enumerate(context):
        if ch not in stoi:
            continue
        x[0, j * V + stoi[ch]] = 1.0

    p = softmax(x @ W)[0]
    order = np.argsort(p)[::-1][:top_n]
    return [(itos[i], float(p[i])) for i in order]


def generate_word(
    W: np.ndarray,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    context_len: int = 2,
    max_steps: int = 20
) -> str:
    """
    Greedy generation from start context.
    """
    ctx = "^" * context_len
    out = []
    for _ in range(max_steps):
        next_items = decode_next_char_distribution(W, ctx, stoi, itos, context_len=context_len, top_n=1)
        ch = next_items[0][0]
        if ch == "$":
            break
        out.append(ch)
        ctx = ctx[1:] + ch
    return "".join(out)


def print_training_summary(result: TrainResult, every: int = 25) -> None:
    print(f"\n=== {result.name} ===")
    print("epoch | loss       | accuracy")
    print("-----------------------------")
    for i in range(0, len(result.losses), every):
        print(f"{i:5d} | {result.losses[i]:.6f} | {result.accs[i]:.4f}")
    print(f"{len(result.losses)-1:5d} | {result.losses[-1]:.6f} | {result.accs[-1]:.4f}")


# ----------------------------
# Main demo
# ----------------------------

def main():
    random.seed(0)
    np.random.seed(0)

    context_len = 2
    words = build_synthetic_words()
    X, Y, stoi, itos = build_word_dataset(words, context_len=context_len)

    print("Dataset summary:")
    print(f"  number of words           = {len(words)}")
    print(f"  number of training pairs  = {len(X)}")
    print(f"  vocabulary size           = {len(stoi)}")
    print(f"  input dimension D_in      = {X.shape[1]}")
    print(f"  output dimension V        = {Y.shape[1]}")

    sgd = train_sgd(X, Y, epochs=250, lr=1.0, seed=0)
    secd = train_secd(X, Y, epochs=250, lr=1.0, top_k=16, seed=0)

    print_training_summary(sgd)
    print_training_summary(secd)

    test_contexts = ["^^", "^n", "lo", "ph", "st", "wa"]
    print("\nNext-character distributions:")
    for ctx in test_contexts:
        print(f"\nContext = {ctx!r}")
        print("  SGD :", decode_next_char_distribution(sgd.W, ctx, stoi, itos, context_len=context_len))
        print("  SECD:", decode_next_char_distribution(secd.W, ctx, stoi, itos, context_len=context_len))

    print("\nGreedy generations:")
    for i in range(8):
        print(f"  SGD  sample {i+1}: {generate_word(sgd.W, stoi, itos, context_len=context_len)}")
    for i in range(8):
        print(f"  SECD sample {i+1}: {generate_word(secd.W, stoi, itos, context_len=context_len)}")


if __name__ == "__main__":
    main()
