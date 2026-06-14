"""
Q-BPE — Byte-Pair Encoding tokenizer (the input pipeline of every LLM)
Sennrich, Haddow & Birch, "Neural Machine Translation of Rare Words with
Subword Units", ACL 2016 — adapted to the byte-level scheme of GPT-2
(Radford et al. 2019) used by GPT, Claude, Llama and friends.

Before a single token reaches the attention mechanism (see q_attention.py),
raw text is segmented into subword tokens by BPE. This module implements the
full lifecycle:
    • train()   — learn a merge table from a corpus by greedily merging the
                  most frequent adjacent symbol pair, vocab_size times
    • encode()  — apply the learned merges to turn text into token ids
    • decode()  — invert token ids back to the exact original bytes

Byte-level: the base alphabet is the 256 bytes, so ANY input (emoji, binary,
any language) round-trips losslessly — never an out-of-vocabulary failure.

Zero dependencies.
"""

from __future__ import annotations
import json
from collections import Counter
from typing import Dict, List, Optional, Tuple


Pair = Tuple[int, int]


class BPETokenizer:
    """
    A byte-level Byte-Pair Encoding tokenizer.

    Token ids 0..255 are the raw bytes. Each learned merge introduces a new
    token id (256, 257, ...) standing for the concatenation of two existing
    tokens. Encoding repeatedly applies the highest-priority applicable merge.
    """

    def __init__(self) -> None:
        # merges: ordered list of (pair -> new_id); rank = priority (lower first)
        self.merges: Dict[Pair, int] = {}
        self._ranks: Dict[Pair, int] = {}
        # vocab: token_id -> bytes it expands to
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, corpus: str, vocab_size: int = 512,
              verbose: bool = False) -> "BPETokenizer":
        """
        Learn merges until the vocabulary reaches `vocab_size` tokens
        (or no mergeable pair remains). `vocab_size` must be ≥ 256.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the byte alphabet)")

        # Start from the raw byte sequence
        tokens: List[int] = list(corpus.encode("utf-8"))
        n_merges = vocab_size - 256

        for step in range(n_merges):
            stats = self._pair_counts(tokens)
            if not stats:
                break
            # Most frequent adjacent pair (ties broken by smaller pair → stable)
            best_pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
            if stats[best_pair] < 2:
                break   # no repetition left to exploit

            new_id = 256 + step
            self.merges[best_pair] = new_id
            self._ranks[best_pair] = step
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            tokens = self._merge(tokens, best_pair, new_id)

            if verbose:
                a = self.vocab[best_pair[0]]
                b = self.vocab[best_pair[1]]
                print(f"  merge {step}: {a!r}+{b!r} -> id {new_id} "
                      f"({stats[best_pair]}×)")

        return self

    @staticmethod
    def _pair_counts(tokens: List[int]) -> Counter:
        counts: Counter = Counter()
        for a, b in zip(tokens, tokens[1:]):
            counts[(a, b)] += 1
        return counts

    @staticmethod
    def _merge(tokens: List[int], pair: Pair, new_id: int) -> List[int]:
        """Replace every non-overlapping occurrence of `pair` with `new_id`."""
        out: List[int] = []
        i = 0
        n = len(tokens)
        while i < n:
            if i < n - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Turn text into a list of token ids using the learned merge order."""
        tokens: List[int] = list(text.encode("utf-8"))

        # Greedily apply the available merge with the lowest rank until none apply
        while len(tokens) >= 2:
            # Find the mergeable pair with the smallest rank
            best_rank: Optional[int] = None
            best_pair: Optional[Pair] = None
            for pair in zip(tokens, tokens[1:]):
                rank = self._ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            tokens = self._merge(tokens, best_pair, self.merges[best_pair])

        return tokens

    def decode(self, ids: List[int]) -> str:
        """Invert token ids back to text (lossless, byte-exact)."""
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def decode_bytes(self, ids: List[int]) -> bytes:
        """Decode to raw bytes (no utf-8 assumption)."""
        return b"".join(self.vocab[i] for i in ids)

    # ------------------------------------------------------------------
    # Stats & persistence
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def compression_ratio(self, text: str) -> float:
        """bytes-per-token: how many input bytes each token packs on average."""
        n_bytes = len(text.encode("utf-8"))
        n_tokens = len(self.encode(text))
        return n_bytes / n_tokens if n_tokens else 0.0

    def to_json(self) -> str:
        """Serialize the merge table (vocab is derivable from it)."""
        return json.dumps({
            "merges": [[a, b, nid] for (a, b), nid in self.merges.items()],
        })

    @classmethod
    def from_json(cls, blob: str) -> "BPETokenizer":
        data = json.loads(blob)
        tok = cls()
        for rank, (a, b, nid) in enumerate(data["merges"]):
            pair = (a, b)
            tok.merges[pair] = nid
            tok._ranks[pair] = rank
            tok.vocab[nid] = tok.vocab[a] + tok.vocab[b]
        return tok

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

_SAMPLE_CORPUS = (
    "the quantum future is near. the quantum computer factors numbers. "
    "the quantum algorithm searches faster. quantum cryptography protects data. "
    "the future of computing is quantum and the future is now. "
) * 20


def demonstrate_bpe() -> dict:
    """
    Train a small tokenizer on a repetitive corpus and show that frequent
    words ('quantum', 'the future') collapse into single tokens, achieving a
    bytes-per-token ratio well above 1 — exactly what a real LLM tokenizer does.
    """
    tok = BPETokenizer().train(_SAMPLE_CORPUS, vocab_size=320)

    sample = "the quantum future"
    ids = tok.encode(sample)
    restored = tok.decode(ids)

    # How many tokens does the word "quantum" cost after training?
    quantum_tokens = tok.encode("quantum")

    return {
        "vocab_size": tok.vocab_size,
        "n_merges": len(tok.merges),
        "sample": sample,
        "sample_bytes": len(sample.encode("utf-8")),
        "sample_token_ids": ids,
        "sample_token_count": len(ids),
        "roundtrip_ok": restored == sample,
        "quantum_token_count": len(quantum_tokens),
        "bytes_per_token": round(tok.compression_ratio(_SAMPLE_CORPUS), 2),
    }
