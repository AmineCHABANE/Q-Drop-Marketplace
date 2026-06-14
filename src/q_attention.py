"""
Q-ATTENTION — The Transformer attention mechanism, from scratch
Vaswani et al., "Attention Is All You Need", NeurIPS 2017 (arXiv:1706.03762)

The single most important algorithm of the current AI era — the mechanism
inside GPT, Claude, Gemini, Llama, BERT, Stable Diffusion, AlphaFold.

Implemented here in pure Python (no numpy) for total transparency:
  - Scaled dot-product attention   softmax(QKᵀ/√dₖ)V
  - Causal (autoregressive) masking — the GPT decoder constraint
  - Multi-head attention            — parallel attention subspaces
  - Sinusoidal positional encoding  — order without recurrence
  - Layer normalization             — Ba et al. 2016
  - Position-wise feed-forward + GELU
  - Full pre-norm Transformer block (the GPT-2/LLaMA arrangement)

Everything is exact and testable: softmax sums to 1, causal masks block the
future, attention is a convex combination of values, and a hand-built example
proves attention is a *programmable* content-addressable lookup — not a black box.

Zero dependencies.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

Matrix = List[List[float]]
Vector = List[float]
NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Minimal linear-algebra layer (pure Python, no numpy)
# ---------------------------------------------------------------------------

def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def shape(m: Matrix) -> Tuple[int, int]:
    return (len(m), len(m[0]) if m else 0)

def transpose(m: Matrix) -> Matrix:
    return [list(col) for col in zip(*m)]

def matmul(a: Matrix, b: Matrix) -> Matrix:
    """Standard (rows_a × inner) · (inner × cols_b) matrix product."""
    inner = len(b)
    if len(a[0]) != inner:
        raise ValueError(f"matmul shape mismatch: {shape(a)} · {shape(b)}")
    bt = transpose(b)
    return [[sum(a[i][k] * bt[j][k] for k in range(inner))
             for j in range(len(bt))]
            for i in range(len(a))]

def mat_add(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def scalar_mul(a: Matrix, s: float) -> Matrix:
    return [[v * s for v in row] for row in a]


def softmax(vec: Sequence[float]) -> Vector:
    """Numerically stable softmax: subtract the max before exponentiating.

    Positions equal to -inf (masked) receive exactly zero probability.
    """
    finite = [v for v in vec if v != NEG_INF]
    if not finite:
        # Degenerate: everything masked → uniform (shouldn't happen in practice)
        n = len(vec)
        return [1.0 / n for _ in vec]
    m = max(finite)
    exps = [0.0 if v == NEG_INF else math.exp(v - m) for v in vec]
    total = sum(exps)
    return [e / total for e in exps]


def gelu(x: float) -> float:
    """Gaussian Error Linear Unit (tanh approximation, as used by GPT-2)."""
    return 0.5 * x * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def relu(x: float) -> float:
    return x if x > 0 else 0.0


# ---------------------------------------------------------------------------
# Scaled dot-product attention — the heart of the Transformer
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: Matrix, K: Matrix, V: Matrix,
    causal: bool = False,
    mask: Optional[List[List[bool]]] = None,
) -> Tuple[Matrix, Matrix]:
    """
    Attention(Q, K, V) = softmax(Q·Kᵀ / √dₖ) · V

    :param Q: queries  (seq_q × d_k)
    :param K: keys     (seq_k × d_k)
    :param V: values   (seq_k × d_v)
    :param causal: if True, position i may only attend to positions ≤ i
    :param mask: optional explicit boolean mask; True means "blocked"
    :returns: (output (seq_q × d_v), attention_weights (seq_q × seq_k))

    Each row of the returned attention_weights sums to 1 — the output is a
    convex combination of the value vectors, i.e. a soft content-addressed read.
    """
    seq_q, d_k = shape(Q)
    seq_k, _ = shape(K)
    scale = 1.0 / math.sqrt(d_k)

    scores = scalar_mul(matmul(Q, transpose(K)), scale)  # (seq_q × seq_k)

    # Apply masking
    for i in range(seq_q):
        for j in range(seq_k):
            blocked = False
            if causal and j > i:
                blocked = True
            if mask is not None and mask[i][j]:
                blocked = True
            if blocked:
                scores[i][j] = NEG_INF

    weights = [softmax(row) for row in scores]   # (seq_q × seq_k)
    output = matmul(weights, V)                  # (seq_q × d_v)
    return output, weights


# ---------------------------------------------------------------------------
# Dense / linear layer
# ---------------------------------------------------------------------------

@dataclass
class Linear:
    """Affine layer y = x·Wᵀ + b  with Kaiming-ish random init."""
    in_features: int
    out_features: int
    W: Matrix = field(default_factory=list)   # (out × in)
    b: Vector = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.W:
            self.W = zeros(self.out_features, self.in_features)
        if not self.b:
            self.b = [0.0] * self.out_features

    @classmethod
    def random(cls, in_features: int, out_features: int,
               rng: Optional[random.Random] = None) -> "Linear":
        rng = rng or random.Random(0)
        scale = math.sqrt(2.0 / in_features)
        W = [[rng.gauss(0, scale) for _ in range(in_features)]
             for _ in range(out_features)]
        b = [0.0] * out_features
        return cls(in_features, out_features, W, b)

    def forward(self, x: Matrix) -> Matrix:
        """x: (seq × in) → (seq × out)."""
        wt = transpose(self.W)   # (in × out)
        out = matmul(x, wt)
        return [[out[i][j] + self.b[j] for j in range(self.out_features)]
                for i in range(len(x))]


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------

class MultiHeadAttention:
    """
    Multi-head attention: run `n_heads` attention operations in parallel
    subspaces of dimension d_model / n_heads, then concatenate and project.

    Splitting the model dimension lets different heads specialize — some track
    syntax, some track long-range coreference, some copy the previous token.
    """

    def __init__(self, d_model: int, n_heads: int,
                 rng: Optional[random.Random] = None):
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        rng = rng or random.Random(0)
        self.W_q = Linear.random(d_model, d_model, rng)
        self.W_k = Linear.random(d_model, d_model, rng)
        self.W_v = Linear.random(d_model, d_model, rng)
        self.W_o = Linear.random(d_model, d_model, rng)

    def _split_heads(self, x: Matrix) -> List[Matrix]:
        """(seq × d_model) → n_heads × (seq × d_head)."""
        seq = len(x)
        heads = []
        for h in range(self.n_heads):
            lo = h * self.d_head
            hi = lo + self.d_head
            heads.append([row[lo:hi] for row in x])
        return heads

    def forward(self, x: Matrix, causal: bool = False
                ) -> Tuple[Matrix, List[Matrix]]:
        """
        :param x: (seq × d_model)
        :returns: (output (seq × d_model), per-head attention weights)
        """
        Q = self.W_q.forward(x)
        K = self.W_k.forward(x)
        V = self.W_v.forward(x)

        q_heads = self._split_heads(Q)
        k_heads = self._split_heads(K)
        v_heads = self._split_heads(V)

        head_outputs: List[Matrix] = []
        all_weights: List[Matrix] = []
        for h in range(self.n_heads):
            out, w = scaled_dot_product_attention(
                q_heads[h], k_heads[h], v_heads[h], causal=causal)
            head_outputs.append(out)
            all_weights.append(w)

        # Concatenate heads back to (seq × d_model)
        seq = len(x)
        concat = [[] for _ in range(seq)]
        for h in range(self.n_heads):
            for i in range(seq):
                concat[i].extend(head_outputs[h][i])

        output = self.W_o.forward(concat)
        return output, all_weights


# ---------------------------------------------------------------------------
# Layer normalization
# ---------------------------------------------------------------------------

class LayerNorm:
    """Normalize each row to zero mean / unit variance, then scale+shift."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.gamma = [1.0] * dim
        self.beta = [0.0] * dim

    def forward(self, x: Matrix) -> Matrix:
        out = []
        for row in x:
            mean = sum(row) / self.dim
            var = sum((v - mean) ** 2 for v in row) / self.dim
            inv = 1.0 / math.sqrt(var + self.eps)
            out.append([(row[j] - mean) * inv * self.gamma[j] + self.beta[j]
                        for j in range(self.dim)])
        return out


# ---------------------------------------------------------------------------
# Position-wise feed-forward network
# ---------------------------------------------------------------------------

class FeedForward:
    """Two linear layers with a GELU non-linearity (the Transformer MLP)."""

    def __init__(self, d_model: int, d_ff: int,
                 activation=gelu, rng: Optional[random.Random] = None):
        rng = rng or random.Random(0)
        self.fc1 = Linear.random(d_model, d_ff, rng)
        self.fc2 = Linear.random(d_ff, d_model, rng)
        self.activation = activation

    def forward(self, x: Matrix) -> Matrix:
        h = self.fc1.forward(x)
        h = [[self.activation(v) for v in row] for row in h]
        return self.fc2.forward(h)


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------

def positional_encoding(seq_len: int, d_model: int) -> Matrix:
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Injects absolute position into token embeddings without recurrence.
    Bounded in [-1, 1], deterministic, and extrapolates to unseen lengths.
    """
    pe = zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            div = math.exp(-(math.log(10000.0) * i / d_model))
            pe[pos][i] = math.sin(pos * div)
            if i + 1 < d_model:
                pe[pos][i + 1] = math.cos(pos * div)
    return pe


# ---------------------------------------------------------------------------
# Full pre-norm Transformer block (GPT-2 / LLaMA style)
# ---------------------------------------------------------------------------

class TransformerBlock:
    """
    A single decoder block with the modern pre-normalization arrangement:

        x = x + MHA(LN(x))
        x = x + FFN(LN(x))

    Residual connections carry gradients; pre-norm stabilizes deep stacks.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: Optional[int] = None,
                 causal: bool = True, rng: Optional[random.Random] = None):
        rng = rng or random.Random(0)
        d_ff = d_ff or 4 * d_model
        self.causal = causal
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, rng)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, rng=rng)

    def forward(self, x: Matrix) -> Tuple[Matrix, List[Matrix]]:
        attn_out, weights = self.attn.forward(self.ln1.forward(x),
                                              causal=self.causal)
        x = mat_add(x, attn_out)                       # residual
        ffn_out = self.ffn.forward(self.ln2.forward(x))
        x = mat_add(x, ffn_out)                        # residual
        return x, weights


class TransformerEncoder:
    """A stack of N Transformer blocks with positional encoding at the input."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 d_ff: Optional[int] = None, causal: bool = True,
                 seed: int = 0):
        rng = random.Random(seed)
        self.d_model = d_model
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, causal, rng)
                       for _ in range(n_layers)]

    def forward(self, embeddings: Matrix) -> Tuple[Matrix, List[List[Matrix]]]:
        """
        :param embeddings: (seq × d_model) token embeddings
        :returns: (final hidden states, per-layer per-head attention weights)
        """
        seq = len(embeddings)
        pe = positional_encoding(seq, self.d_model)
        x = mat_add(embeddings, pe)

        all_attn = []
        for block in self.blocks:
            x, weights = block.forward(x)
            all_attn.append(weights)
        return x, all_attn


# ---------------------------------------------------------------------------
# Demonstrations — proof the mechanism actually routes information
# ---------------------------------------------------------------------------

def demonstrate_attention_as_lookup() -> dict:
    """
    Attention is a soft, differentiable hash-map.

    We hand-build Q/K/V so that query 0 matches key #3 almost exactly.
    The output should be (approximately) value #3 — a content-addressed read.
    No training: this is the raw mechanism.
    """
    d = 4
    # Four orthogonal keys (a tiny "memory")
    K = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    # Distinct values stored at each slot
    V = [
        [10.0, 0.0],
        [20.0, 0.0],
        [30.0, 0.0],
        [40.0, 0.0],
    ]
    # A single query that points hard at slot #3 (scaled up to sharpen softmax).
    # Score for slot 3 = (20·1)/√4 = 10, so softmax puts ~99.99% mass there.
    Q = [[0.0, 0.0, 0.0, 20.0]]

    out, weights = scaled_dot_product_attention(Q, K, V)
    return {
        "query_points_at": 3,
        "attention_weights": [round(w, 4) for w in weights[0]],
        "retrieved_value": [round(v, 3) for v in out[0]],
        "expected_value": V[3],
        "match": abs(out[0][0] - 40.0) < 1.0,
    }


def demonstrate_causal_averaging() -> dict:
    """
    With uniform queries/keys, causal attention computes a *running prefix mean* —
    each position sees the average of itself and everything before it.

    This is exactly the "attention as information aggregation" intuition, and
    it is verifiable to the last decimal.
    """
    seq = 4
    d = 2
    # Identical Q and K everywhere → all scores equal → uniform softmax (causal)
    Q = [[0.0, 0.0] for _ in range(seq)]
    K = [[0.0, 0.0] for _ in range(seq)]
    V = [[float(i + 1), float((i + 1) * 10)] for i in range(seq)]  # 1,2,3,4

    out, weights = scaled_dot_product_attention(Q, K, V, causal=True)

    # Expected: out[i] = mean(V[0..i])
    expected = []
    for i in range(seq):
        prefix = V[: i + 1]
        m0 = sum(r[0] for r in prefix) / (i + 1)
        m1 = sum(r[1] for r in prefix) / (i + 1)
        expected.append([m0, m1])

    ok = all(abs(out[i][0] - expected[i][0]) < 1e-9 for i in range(seq))
    return {
        "values": V,
        "causal_prefix_means": [[round(v, 3) for v in r] for r in out],
        "expected": expected,
        "exact_match": ok,
    }


def count_parameters(d_model: int, n_heads: int, n_layers: int,
                     d_ff: Optional[int] = None) -> int:
    """Total learnable parameters in a TransformerEncoder of this size."""
    d_ff = d_ff or 4 * d_model
    per_block = (
        4 * (d_model * d_model + d_model)   # W_q,W_k,W_v,W_o (+ biases)
        + 2 * d_model                       # two LayerNorms (gamma+beta) ≈
        + (d_model * d_ff + d_ff)           # fc1
        + (d_ff * d_model + d_model)        # fc2
    )
    return per_block * n_layers
