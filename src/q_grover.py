"""
q_grover.py — Grover's Quantum Search Algorithm (statevector simulation)

Reference: L. Grover, "A fast quantum mechanical algorithm for database search",
           STOC 1996. https://arxiv.org/abs/quant-ph/9605043

Grover's gives a quadratic speedup over classical search:
  - Classical: O(N) queries to find 1 item in N
  - Quantum:   O(√N) queries

This module simulates the full statevector, so it runs on classical hardware.
It is exact — no shots, no noise — and shows precisely how amplitude amplification
works. The exponential memory cost (2^n complex numbers) is the trade-off; real
quantum hardware stores this in superposition for free.

Key concepts implemented:
  1. Hadamard transform      — uniform superposition over all 2^n basis states
  2. Phase oracle            — marks the target state by flipping its amplitude sign
  3. Diffusion operator      — reflects around the mean (Grover diffuser)
  4. Optimal iteration count — π/4 · √(N/M) for M solutions in N items
"""

import math
import cmath
import random
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Linear algebra primitives (no numpy)
# ---------------------------------------------------------------------------

def _hadamard_single(state: List[complex], qubit: int, n_qubits: int) -> List[complex]:
    """Apply single-qubit Hadamard gate to `qubit` (0 = most significant)."""
    out = [0j] * len(state)
    half = 1.0 / math.sqrt(2)
    step = 1 << (n_qubits - 1 - qubit)
    for i in range(len(state)):
        bit = (i >> (n_qubits - 1 - qubit)) & 1
        partner = i ^ step
        if bit == 0:
            out[i]       += half * state[i]
            out[i]       += half * state[partner]
        else:
            out[i]       += -half * state[partner]
            out[i]       +=  half * state[i]
    # Above double-counts; fix: only iterate over pairs
    return out


def _hadamard_all(n_qubits: int) -> List[complex]:
    """
    Return the statevector after applying H⊗n to |0...0⟩.
    All 2^n amplitudes equal 1/√(2^n).
    """
    N = 1 << n_qubits
    amp = 1.0 / math.sqrt(N)
    return [complex(amp, 0)] * N


def _apply_hadamard_transform(state: List[complex], n_qubits: int) -> List[complex]:
    """
    Apply H⊗n to an arbitrary statevector using the fast Walsh-Hadamard transform.
    In-place Cooley-Tukey style: O(N log N).
    """
    s = list(state)
    N = len(s)
    h = 1
    while h < N:
        for i in range(0, N, h * 2):
            for j in range(i, i + h):
                x, y = s[j], s[j + h]
                s[j]     = (x + y) / math.sqrt(2)
                s[j + h] = (x - y) / math.sqrt(2)
        h *= 2
    return s


# ---------------------------------------------------------------------------
# Grover's core operators
# ---------------------------------------------------------------------------

class PhaseOracle:
    """
    Marks one or more target indices by negating their amplitude.
    In a real quantum computer this would be a unitary circuit; here
    we directly flip the complex sign of the solution amplitude(s).
    """

    def __init__(self, targets: List[int], n_qubits: int):
        N = 1 << n_qubits
        for t in targets:
            if not (0 <= t < N):
                raise ValueError(f"Target {t} out of range [0, {N})")
        self.targets = set(targets)

    def apply(self, state: List[complex]) -> List[complex]:
        """O(N) scan — flip sign of each solution state."""
        out = list(state)
        for t in self.targets:
            out[t] = -out[t]
        return out


class DiffusionOperator:
    """
    Grover diffuser: D = 2|ψ⟩⟨ψ| − I
    where |ψ⟩ = H⊗n|0⟩ is the uniform superposition.

    Effect: reflects every amplitude around the current mean.
    Amplitudes above the mean get pushed down; amplitudes below
    (the marked one, freshly negated by the oracle) get pushed up.
    Each round amplifies the solution by ≈ 2/√N.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def apply(self, state: List[complex]) -> List[complex]:
        N = len(state)
        # mean amplitude
        mean = sum(state) / N
        # reflect: new_amp = 2*mean - amp
        return [2 * mean - a for a in state]


# ---------------------------------------------------------------------------
# High-level GroverSearch class
# ---------------------------------------------------------------------------

class GroverSearch:
    """
    Grover's quantum search over a list of N items.

    Usage
    -----
    # Search for a specific value in a database of 64 items
    db = list(range(64))
    target_value = 42
    gs = GroverSearch(n_qubits=6)                      # 2^6 = 64 items
    result_idx = gs.search(oracle_indices=[42])
    assert db[result_idx] == target_value

    Parameters
    ----------
    n_qubits : int
        Number of qubits. Database size N = 2^n_qubits.
    n_solutions : int
        Number of marked items M (default 1). Affects optimal iterations.
    seed : int | None
        Optional random seed for measurement sampling.
    """

    def __init__(self, n_qubits: int, n_solutions: int = 1, seed: Optional[int] = None):
        if n_qubits < 1:
            raise ValueError("Need at least 1 qubit")
        self.n_qubits = n_qubits
        self.N = 1 << n_qubits
        self.n_solutions = n_solutions
        self._rng = random.Random(seed)

    def optimal_iterations(self) -> int:
        """
        The exact formula: floor(π/4 · √(N/M)).
        More iterations than this start UN-amplifying the solution.
        """
        ratio = self.N / self.n_solutions
        return max(1, math.floor(math.pi / 4 * math.sqrt(ratio)))

    def search(
        self,
        oracle_indices: List[int],
        n_iterations: Optional[int] = None,
    ) -> int:
        """
        Run the full Grover circuit and return the measured index.

        Steps:
          1. H⊗n — create uniform superposition
          2. Repeat k times:
             a. Oracle — negate solution amplitude(s)
             b. Diffuser — reflect around mean
          3. Measure — sample from |amplitude|² distribution
        """
        if not oracle_indices:
            raise ValueError("Provide at least one oracle target index")

        k = n_iterations if n_iterations is not None else self.optimal_iterations()

        oracle = PhaseOracle(oracle_indices, self.n_qubits)
        diffuser = DiffusionOperator(self.n_qubits)

        # Step 1: uniform superposition
        state = _hadamard_all(self.n_qubits)

        # Step 2: amplitude amplification
        for _ in range(k):
            state = oracle.apply(state)
            state = diffuser.apply(state)

        # Step 3: measure (sample from probability distribution)
        return self._measure(state)

    def statevector(
        self,
        oracle_indices: List[int],
        n_iterations: Optional[int] = None,
    ) -> List[complex]:
        """Return the full complex statevector (for inspection / testing)."""
        k = n_iterations if n_iterations is not None else self.optimal_iterations()
        oracle = PhaseOracle(oracle_indices, self.n_qubits)
        diffuser = DiffusionOperator(self.n_qubits)
        state = _hadamard_all(self.n_qubits)
        for _ in range(k):
            state = oracle.apply(state)
            state = diffuser.apply(state)
        return state

    def probability_of_success(
        self,
        oracle_indices: List[int],
        n_iterations: Optional[int] = None,
    ) -> float:
        """
        Exact success probability = sum of |amplitude|² for solution states.
        For optimal k this approaches 1 - O(M/N).
        """
        state = self.statevector(oracle_indices, n_iterations)
        return sum(abs(state[i]) ** 2 for i in oracle_indices)

    def _measure(self, state: List[complex]) -> int:
        """
        Sample one basis state proportional to |amplitude|².
        Uses alias-free linear scan (fine for N ≤ 2^20).
        """
        probs = [abs(a) ** 2 for a in state]
        r = self._rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return i
        return len(state) - 1  # floating-point edge case


# ---------------------------------------------------------------------------
# Quantum Fourier Transform (bonus — used by Shor's algorithm)
# ---------------------------------------------------------------------------

class QuantumFourierTransform:
    """
    Simulated QFT over n qubits.

    The QFT maps |j⟩ → (1/√N) Σ_k exp(2πijk/N)|k⟩
    It is the quantum analogue of the DFT and is used in:
      - Shor's factoring algorithm (period finding)
      - Phase estimation
      - Quantum simulation

    Runtime: O(N log N) — same as classical FFT, but on a quantum computer
    only O(n²) gates are needed (exponential saving over classical).
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.N = 1 << n_qubits

    def apply(self, state: List[complex]) -> List[complex]:
        """Apply QFT to statevector using iterative Cooley-Tukey."""
        s = list(state)
        N = self.N
        # Bit-reversal permutation
        s = self._bit_reverse(s)
        # Butterfly stages
        length = 2
        while length <= N:
            half = length // 2
            w = cmath.exp(2j * cmath.pi / length)
            for i in range(0, N, length):
                wn = 1 + 0j
                for j in range(half):
                    u = s[i + j]
                    v = s[i + j + half] * wn
                    s[i + j]        = (u + v) / math.sqrt(2)
                    s[i + j + half] = (u - v) / math.sqrt(2)
                    wn *= w
            length *= 2
        return s

    def inverse(self, state: List[complex]) -> List[complex]:
        """Inverse QFT = conjugate of QFT."""
        conjugated = [a.conjugate() for a in state]
        result = self.apply(conjugated)
        return [a.conjugate() for a in result]

    def _bit_reverse(self, s: List[complex]) -> List[complex]:
        n = self.n_qubits
        N = self.N
        out = list(s)
        for i in range(N):
            j = int(bin(i)[2:].zfill(n)[::-1], 2)
            if j > i:
                out[i], out[j] = out[j], out[i]
        return out
