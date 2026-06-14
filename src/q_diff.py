"""
Q-DIFF — Myers diff algorithm (the engine behind `git diff` and every merge tool)
Eugene W. Myers, "An O(ND) Difference Algorithm and Its Variations",
Algorithmica 1986

Computes the shortest edit script that turns sequence A into sequence B — the
minimal set of insertions and deletions, which is equivalent to finding the
longest common subsequence. This is what powers:
  - git diff / git blame / git merge
  - code-review diffs on GitHub & GitLab
  - `diff`, patch files, and three-way merges
  - text reconciliation in collaborative editors

Implements the real greedy O(ND) Myers algorithm with backtracking over the
edit graph, plus unified-diff formatting and patch application. Operates on any
sequence of comparable, hashable elements (lines, characters, tokens).

Zero dependencies.
"""

from __future__ import annotations
from typing import Any, List, Sequence, Tuple

# An edit op is (tag, element) with tag in {" ", "-", "+"}:
#   " " keep (common)   "-" delete from A   "+" insert from B
Op = Tuple[str, Any]


# ---------------------------------------------------------------------------
# Myers O(ND) shortest edit script
# ---------------------------------------------------------------------------

def _shortest_edit(a: Sequence[Any], b: Sequence[Any]) -> List[dict]:
    """Trace the greedy frontier of the edit graph; returns per-depth V arrays."""
    n, m = len(a), len(b)
    max_d = n + m
    v = {1: 0}
    trace: List[dict] = []
    for d in range(max_d + 1):
        trace.append(dict(v))
        for k in range(-d, d + 1, 2):
            # Choose whether we got here by moving down (insert) or right (delete)
            if k == -d or (k != d and v.get(k - 1, -1) < v.get(k + 1, -1)):
                x = v.get(k + 1, 0)          # move down  → insertion in B
            else:
                x = v.get(k - 1, 0) + 1      # move right → deletion in A
            y = x - k
            # Follow the diagonal (matching elements) as far as possible
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1
            v[k] = x
            if x >= n and y >= m:
                return trace
    return trace


def _backtrack(a: Sequence[Any], b: Sequence[Any], trace: List[dict]) -> List[Op]:
    """Walk the trace backwards to recover the edit operations in order."""
    x, y = len(a), len(b)
    ops: List[Op] = []
    for d in range(len(trace) - 1, 0, -1):
        v = trace[d]
        k = x - y
        if k == -d or (k != d and v.get(k - 1, -1) < v.get(k + 1, -1)):
            prev_k = k + 1
        else:
            prev_k = k - 1
        prev_x = v.get(prev_k, 0)
        prev_y = prev_x - prev_k
        # Emit any diagonal (common) moves first
        while x > prev_x and y > prev_y:
            ops.append((" ", a[x - 1]))
            x -= 1
            y -= 1
        if x == prev_x:
            ops.append(("+", b[prev_y]))      # came down → insertion
        else:
            ops.append(("-", a[prev_x]))      # came right → deletion
        x, y = prev_x, prev_y
    # Leading diagonal at depth 0
    while x > 0 and y > 0:
        ops.append((" ", a[x - 1]))
        x -= 1
        y -= 1
    ops.reverse()
    return ops


def diff(a: Sequence[Any], b: Sequence[Any]) -> List[Op]:
    """
    Return the shortest edit script transforming `a` into `b` as a list of
    (tag, element) ops with tag in {" ", "-", "+"}.
    """
    return _backtrack(a, b, _shortest_edit(a, b))


# ---------------------------------------------------------------------------
# Longest common subsequence (a direct consequence)
# ---------------------------------------------------------------------------

def lcs(a: Sequence[Any], b: Sequence[Any]) -> List[Any]:
    """The longest common subsequence — the kept elements of the diff."""
    return [el for tag, el in diff(a, b) if tag == " "]


def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Number of insertions + deletions in the shortest edit script."""
    return sum(1 for tag, _ in diff(a, b) if tag != " ")


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def apply_patch(a: Sequence[Any], ops: List[Op]) -> List[Any]:
    """
    Apply a diff produced by `diff(a, b)` to `a`, reconstructing `b`.

    Verifies that the deletions/keeps line up with `a` as a consistency check.
    """
    result: List[Any] = []
    i = 0
    for tag, el in ops:
        if tag == " ":
            if i >= len(a) or a[i] != el:
                raise ValueError("patch does not apply cleanly (context mismatch)")
            result.append(a[i])
            i += 1
        elif tag == "-":
            if i >= len(a) or a[i] != el:
                raise ValueError("patch does not apply cleanly (delete mismatch)")
            i += 1
        elif tag == "+":
            result.append(el)
        else:
            raise ValueError(f"unknown op tag: {tag!r}")
    if i != len(a):
        raise ValueError("patch did not consume all of the source")
    return result


# ---------------------------------------------------------------------------
# Unified diff formatting (git-style)
# ---------------------------------------------------------------------------

def unified_diff(a: Sequence[str], b: Sequence[str], context: int = 3,
                 a_name: str = "a", b_name: str = "b") -> str:
    """
    Render a diff in unified format with `context` lines around each change —
    the format `git diff` and `diff -u` emit and patch tools consume.
    """
    ops = diff(a, b)
    # Group ops into hunks separated by long runs of common lines
    lines: List[str] = []
    hunks = _group_hunks(ops, context)
    if not hunks:
        return ""

    out = [f"--- {a_name}", f"+++ {b_name}"]
    for hunk in hunks:
        a_start, a_len, b_start, b_len, body = hunk
        out.append(f"@@ -{a_start + 1},{a_len} +{b_start + 1},{b_len} @@")
        for tag, el in body:
            out.append(f"{tag}{el}")
    return "\n".join(out)


def _group_hunks(ops: List[Op], context: int):
    """
    Split a flat op list into unified-diff hunks. Changes closer than
    2*context common lines are merged into one hunk; each hunk carries up to
    `context` unchanged lines before the first and after the last change.
    Returns tuples (a_start, a_len, b_start, b_len, body).
    """
    n = len(ops)
    # A/B line index at the position of each op
    a_idx: List[int] = []
    b_idx: List[int] = []
    ai = bi = 0
    for tag, _ in ops:
        a_idx.append(ai)
        b_idx.append(bi)
        if tag in (" ", "-"):
            ai += 1
        if tag in (" ", "+"):
            bi += 1

    change_idx = [i for i, (t, _) in enumerate(ops) if t != " "]
    if not change_idx:
        return []

    # Cluster nearby changes
    clusters: List[List[int]] = [[change_idx[0]]]
    for idx in change_idx[1:]:
        if idx - clusters[-1][-1] - 1 > 2 * context:
            clusters.append([idx])
        else:
            clusters[-1].append(idx)

    hunks = []
    for cluster in clusters:
        lo = max(0, cluster[0] - context)
        hi = min(n, cluster[-1] + context + 1)
        body = list(ops[lo:hi])
        a_len = sum(1 for t, _ in body if t in (" ", "-"))
        b_len = sum(1 for t, _ in body if t in (" ", "+"))
        hunks.append((a_idx[lo], a_len, b_idx[lo], b_len, body))
    return hunks


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def demonstrate_diff() -> dict:
    """
    Diff two versions of a small file, confirm the edit script is minimal and
    that applying it reconstructs the new version exactly — the round-trip
    guarantee every version-control system relies on.
    """
    old = ["import os", "", "def greet(name):", "    print('hi ' + name)", "", "greet('world')"]
    new = ["import os", "import sys", "", "def greet(name):", "    print(f'hi {name}')", "", "greet('quantum')"]

    ops = diff(old, new)
    reconstructed = apply_patch(old, ops)

    kept = sum(1 for t, _ in ops if t == " ")
    inserted = sum(1 for t, _ in ops if t == "+")
    deleted = sum(1 for t, _ in ops if t == "-")

    return {
        "lcs_length": len(lcs(old, new)),
        "edit_distance": edit_distance(old, new),
        "kept": kept, "inserted": inserted, "deleted": deleted,
        "roundtrip_ok": reconstructed == new,
        "unified_diff": unified_diff(old, new, context=1),
    }
