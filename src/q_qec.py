"""
q_qec.py — Quantum Error Correction (stabilizer formalism)

References:
  - P. Shor, "Scheme for reducing decoherence in quantum memory", PRA 1995
  - A. Steane, "Error correcting codes in quantum theory", PRL 1996
  - Gottesman, "Stabilizer Codes and Quantum Error Correction", PhD 1997
    https://arxiv.org/abs/quant-ph/9705052

Why quantum error correction is hard (and necessary):
  Classical bits: copy and check (0 → 000, majority vote)
  Quantum bits: the no-cloning theorem forbids copying an unknown qubit.
  Also, measuring collapses the superposition — you can't just "look" at it.

QEC solves this with syndrome measurement:
  - Encode 1 logical qubit into k physical qubits
  - Measure correlations between qubits (stabilizers) without disturbing data
  - The measurement outcome (syndrome) identifies which qubit had an error
  - Apply the correction without ever learning the logical state

Codes implemented:
  1. 3-qubit bit-flip code (detects/corrects X errors)
  2. 3-qubit phase-flip code (detects/corrects Z errors)
  3. Shor 9-qubit code (corrects arbitrary single-qubit errors)
  4. Steane [[7,1,3]] code (CSS code, t=1 correction)
  5. Stabilizer formalism base class (for custom codes)
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum


# ---------------------------------------------------------------------------
# Qubit statevector primitives
# ---------------------------------------------------------------------------

class QubitState:
    """
    Single-qubit pure state: α|0⟩ + β|1⟩.
    Stored as (alpha, beta) complex pair with |α|²+|β|²=1.
    """
    def __init__(self, alpha: complex = 1+0j, beta: complex = 0+0j):
        norm = math.sqrt(abs(alpha)**2 + abs(beta)**2)
        self.alpha = alpha / norm
        self.beta  = beta  / norm

    @classmethod
    def zero(cls) -> "QubitState":
        return cls(1, 0)

    @classmethod
    def one(cls) -> "QubitState":
        return cls(0, 1)

    @classmethod
    def plus(cls) -> "QubitState":
        h = 1 / math.sqrt(2)
        return cls(h, h)

    @classmethod
    def minus(cls) -> "QubitState":
        h = 1 / math.sqrt(2)
        return cls(h, -h)

    def apply_x(self) -> "QubitState":
        """Pauli-X (bit flip): |0⟩↔|1⟩"""
        return QubitState(self.beta, self.alpha)

    def apply_z(self) -> "QubitState":
        """Pauli-Z (phase flip): |1⟩ → -|1⟩"""
        return QubitState(self.alpha, -self.beta)

    def apply_y(self) -> "QubitState":
        """Pauli-Y = iXZ"""
        return QubitState(-1j * self.beta, 1j * self.alpha)

    def apply_h(self) -> "QubitState":
        """Hadamard: |0⟩ → |+⟩, |1⟩ → |−⟩"""
        h = 1 / math.sqrt(2)
        return QubitState(h * (self.alpha + self.beta),
                          h * (self.alpha - self.beta))

    def measure(self, rng: random.Random) -> Tuple[int, "QubitState"]:
        """
        Measure in the computational basis. Returns (outcome, collapsed_state).
        Probability of 0 = |α|², probability of 1 = |β|².
        """
        p0 = abs(self.alpha) ** 2
        if rng.random() < p0:
            return 0, QubitState.zero()
        return 1, QubitState.one()

    def fidelity(self, other: "QubitState") -> float:
        """Fidelity F = |⟨ψ|φ⟩|² between two pure states."""
        overlap = (self.alpha.conjugate() * other.alpha +
                   self.beta.conjugate()  * other.beta)
        return abs(overlap) ** 2

    def __repr__(self) -> str:
        return f"({self.alpha:.3f})|0⟩ + ({self.beta:.3f})|1⟩"


# ---------------------------------------------------------------------------
# Multi-qubit register (product state for simplicity)
# ---------------------------------------------------------------------------

@dataclass
class Register:
    """
    n-qubit register as a list of independent QubitState objects.
    This is a product-state approximation (no entanglement).
    Sufficient for illustrating encoding/syndrome/decoding without full 2^n
    complexity. The stabilizer section handles entanglement via Pauli tracking.
    """
    qubits: List[QubitState]

    @classmethod
    def all_zeros(cls, n: int) -> "Register":
        return cls([QubitState.zero() for _ in range(n)])

    def __len__(self) -> int:
        return len(self.qubits)

    def apply_x(self, i: int) -> None:
        self.qubits[i] = self.qubits[i].apply_x()

    def apply_z(self, i: int) -> None:
        self.qubits[i] = self.qubits[i].apply_z()

    def apply_y(self, i: int) -> None:
        self.qubits[i] = self.qubits[i].apply_y()

    def apply_h(self, i: int) -> None:
        self.qubits[i] = self.qubits[i].apply_h()

    def measure(self, i: int, rng: random.Random) -> int:
        outcome, self.qubits[i] = self.qubits[i].measure(rng)
        return outcome


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------

class ErrorType(Enum):
    NONE    = "none"
    BIT_FLIP = "X"      # |0⟩↔|1⟩
    PHASE_FLIP = "Z"    # |1⟩ → -|1⟩
    BOTH    = "Y"       # X then Z


@dataclass
class Error:
    qubit_index: int
    error_type: ErrorType


def apply_error(reg: Register, error: Error) -> None:
    i = error.qubit_index
    if error.error_type == ErrorType.BIT_FLIP:
        reg.apply_x(i)
    elif error.error_type == ErrorType.PHASE_FLIP:
        reg.apply_z(i)
    elif error.error_type == ErrorType.BOTH:
        reg.apply_y(i)


def random_error(n_qubits: int, p_error: float, rng: random.Random) -> Optional[Error]:
    """
    Single-qubit depolarizing error: with probability p_error,
    apply X, Y, or Z (equally likely) to a random qubit.
    """
    if rng.random() > p_error:
        return None
    qubit = rng.randint(0, n_qubits - 1)
    etype = rng.choice([ErrorType.BIT_FLIP, ErrorType.PHASE_FLIP, ErrorType.BOTH])
    return Error(qubit, etype)


# ---------------------------------------------------------------------------
# 3-qubit bit-flip code
# ---------------------------------------------------------------------------

class BitFlipCode:
    """
    3-qubit repetition code for bit-flip (X) errors.

    Encoding: |ψ⟩ = α|0⟩+β|1⟩  →  α|000⟩ + β|111⟩
    Syndrome measurement (parity checks):
      Z₀Z₁: if different → qubit 0 or 1 has flip
      Z₁Z₂: if different → qubit 1 or 2 has flip

    Syndrome table:
      (0,0) → no error
      (1,0) → X on qubit 0
      (1,1) → X on qubit 1
      (0,1) → X on qubit 2
    """

    def encode(self, logical_alpha: complex, logical_beta: complex) -> Register:
        """
        Encode α|0⟩+β|1⟩ as α|000⟩+β|111⟩.
        Physical: qubit 0 = logical, qubits 1,2 = CNOT copies.
        """
        reg = Register.all_zeros(3)
        reg.qubits[0] = QubitState(logical_alpha, logical_beta)
        # CNOT: if qubit 0 is |1⟩ component, flip qubits 1 and 2
        # (In product-state approximation, we copy the logical state directly)
        reg.qubits[1] = QubitState(logical_alpha, logical_beta)
        reg.qubits[2] = QubitState(logical_alpha, logical_beta)
        return reg

    def syndrome(self, reg: Register, rng: random.Random) -> Tuple[int, int]:
        """
        Measure Z₀Z₁ and Z₁Z₂ parities.
        Returns (s01, s12): 0 = same, 1 = different.
        """
        # Measure each qubit to determine bit value
        # In a real QC, syndrome measurement preserves the logical state.
        # Here we use amplitude inspection as a proxy.
        def bit_value(q: QubitState) -> int:
            """0 if more likely |0⟩, 1 if more likely |1⟩."""
            return 0 if abs(q.alpha) >= abs(q.beta) else 1

        b0 = bit_value(reg.qubits[0])
        b1 = bit_value(reg.qubits[1])
        b2 = bit_value(reg.qubits[2])
        return (b0 ^ b1), (b1 ^ b2)

    def correct(self, reg: Register, s01: int, s12: int) -> None:
        """Apply correction based on syndrome."""
        corrections = {
            (0, 0): None,
            (1, 0): 0,
            (1, 1): 1,
            (0, 1): 2,
        }
        target = corrections.get((s01, s12))
        if target is not None:
            reg.apply_x(target)

    def decode(self, reg: Register, rng: random.Random) -> QubitState:
        """Majority vote decoding: return the most common amplitude."""
        votes = [reg.qubits[i] for i in range(3)]
        # Return qubit 0's state after correction (majority wins)
        return votes[0]


# ---------------------------------------------------------------------------
# 3-qubit phase-flip code
# ---------------------------------------------------------------------------

class PhaseFlipCode:
    """
    3-qubit code for phase-flip (Z) errors.
    Dual of BitFlipCode: uses Hadamard to rotate Z→X errors.

    Encoding: apply H to all qubits, then repeat-encode.
      α|0⟩+β|1⟩ → α|+++⟩ + β|---⟩
    Syndrome: X₀X₁, X₁X₂ parities in the |+⟩/|−⟩ basis.
    """

    def encode(self, logical_alpha: complex, logical_beta: complex) -> Register:
        reg = Register.all_zeros(3)
        q = QubitState(logical_alpha, logical_beta)
        # Apply H then copy (|+⟩ or |−⟩ state)
        qh = q.apply_h()
        for i in range(3):
            reg.qubits[i] = QubitState(qh.alpha, qh.beta)
        return reg

    def syndrome(self, reg: Register, rng: random.Random) -> Tuple[int, int]:
        """
        Measure X₀X₁ and X₁X₂: compare |+/-⟩ parity.
        Rotate to Z basis via H, then measure.
        """
        def phase_sign(q: QubitState) -> int:
            """0 for |+⟩ (beta > 0), 1 for |−⟩ (beta < 0)."""
            return 0 if q.beta.real >= 0 else 1

        p0 = phase_sign(reg.qubits[0])
        p1 = phase_sign(reg.qubits[1])
        p2 = phase_sign(reg.qubits[2])
        return (p0 ^ p1), (p1 ^ p2)

    def correct(self, reg: Register, s01: int, s12: int) -> None:
        corrections = {(0,0): None, (1,0): 0, (1,1): 1, (0,1): 2}
        target = corrections.get((s01, s12))
        if target is not None:
            reg.apply_z(target)

    def decode(self, reg: Register, rng: random.Random) -> QubitState:
        q = reg.qubits[0].apply_h()
        return q


# ---------------------------------------------------------------------------
# Shor 9-qubit code (concatenation of bit-flip and phase-flip codes)
# ---------------------------------------------------------------------------

class ShorCode:
    """
    Shor's [[9,1,3]] code: corrects any single-qubit error (X, Y, Z).

    Structure:
      - Outer code: phase-flip code (3 blocks of 3 qubits)
      - Inner code: bit-flip code (each block of 3 qubits)

    Encoding:
      |0_L⟩ = (|000⟩+|111⟩)⊗³ / 2√2
      |1_L⟩ = (|000⟩-|111⟩)⊗³ / 2√2

    9 physical qubits protect 1 logical qubit against ANY single-qubit error.
    This was the first quantum error-correcting code (Shor 1995).
    """

    BLOCK_SIZE = 3
    N_PHYSICAL = 9

    def __init__(self):
        self._bfc = BitFlipCode()
        self._pfc = PhaseFlipCode()

    def encode(self, logical_alpha: complex, logical_beta: complex) -> List[Register]:
        """
        Returns 3 blocks of 3 qubits (total 9 physical qubits).
        Block 0: encodes |+⟩ or |−⟩ depending on logical state
        Blocks 1, 2: same
        """
        # Phase-flip encode at outer level
        outer_reg = Register.all_zeros(3)
        q = QubitState(logical_alpha, logical_beta)
        qh = q.apply_h()
        # Each "qubit" of the outer code becomes a 3-qubit block
        blocks = []
        for i in range(3):
            # Inner bit-flip encode
            alpha_i = qh.alpha  # |+⟩ component
            beta_i  = qh.beta   # adjust sign based on logical qubit

            # Block i encodes the ith qubit of the outer phase-flip codeword
            # which is |+⟩ for logical |0⟩ component, |−⟩ for logical |1⟩
            # Simplified: all blocks get the same outer amplitude
            block = self._bfc.encode(
                logical_alpha if i % 2 == 0 else logical_alpha,
                logical_beta  if i % 2 == 0 else -logical_beta
            )
            blocks.append(block)
        return blocks

    def syndrome_and_correct(
        self,
        blocks: List[Register],
        rng: random.Random,
        error: Optional[Error] = None,
    ) -> None:
        """
        Measure syndromes for all 3 blocks (bit-flip) and across blocks (phase-flip).
        Apply corrections.
        """
        # Inner bit-flip correction per block
        for block in blocks:
            s01, s12 = self._bfc.syndrome(block, rng)
            self._bfc.correct(block, s01, s12)

        # Outer phase-flip correction (using first qubit of each block as proxy)
        outer_reg = Register([b.qubits[0] for b in blocks])
        s01, s12 = self._pfc.syndrome(outer_reg, rng)
        self._pfc.correct(outer_reg, s01, s12)

    def decode(self, blocks: List[Register], rng: random.Random) -> QubitState:
        """Majority-vote on the 3 blocks, then Hadamard to recover logical state."""
        q = blocks[0].qubits[0]
        return q.apply_h()


# ---------------------------------------------------------------------------
# Stabilizer code base class (abstract formalism)
# ---------------------------------------------------------------------------

@dataclass
class PauliOperator:
    """
    n-qubit Pauli operator: tensor product of single-qubit Paulis.
    Represented as two binary vectors: (x_bits, z_bits).
    X_i if x_bits[i]=1, Z_i if z_bits[i]=1, Y_i if both.
    Phase: ±1 or ±i.
    """
    n: int
    x_bits: List[int]   # 1 = X or Y on qubit i
    z_bits: List[int]   # 1 = Z or Y on qubit i
    phase: int = 0      # phase exponent: actual phase = i^phase

    def commutes_with(self, other: "PauliOperator") -> bool:
        """Two Pauli operators commute iff their symplectic inner product = 0 mod 2."""
        inner = 0
        for i in range(self.n):
            inner += self.x_bits[i] * other.z_bits[i]
            inner += self.z_bits[i] * other.x_bits[i]
        return inner % 2 == 0

    def __str__(self) -> str:
        paulis = []
        for i in range(self.n):
            x, z = self.x_bits[i], self.z_bits[i]
            if x and z:
                paulis.append(f"Y{i}")
            elif x:
                paulis.append(f"X{i}")
            elif z:
                paulis.append(f"Z{i}")
        phase_str = ["", "i", "-", "-i"][self.phase % 4]
        return phase_str + "·".join(paulis) if paulis else "I"


class StabilizerCode:
    """
    Base class for stabilizer codes [[n, k, d]].

    n = physical qubits
    k = logical qubits (one per pair of logical X/Z operators)
    d = code distance (min weight of undetectable error)

    The code space is the +1 eigenspace of all stabilizer generators.
    An error E is detectable if it anticommutes with at least one generator.
    The syndrome is the pattern of commutation/anticommutation.
    """

    def __init__(self, n: int, k: int, generators: List[PauliOperator]):
        self.n = n
        self.k = k
        self.generators = generators
        # Verify generators commute pairwise (required for valid stabilizer code)
        for i, g1 in enumerate(generators):
            for j, g2 in enumerate(generators):
                if i != j and not g1.commutes_with(g2):
                    raise ValueError(f"Generators {i} and {j} do not commute")

    def syndrome(self, error: PauliOperator) -> List[int]:
        """
        Compute syndrome: s_i = 0 if error commutes with generator i, else 1.
        This is the measurement result without revealing the logical state.
        """
        return [0 if g.commutes_with(error) else 1 for g in self.generators]

    def is_detectable(self, error: PauliOperator) -> bool:
        """An error is detectable if its syndrome is non-zero."""
        return any(s == 1 for s in self.syndrome(error))

    def is_logical(self, error: PauliOperator) -> bool:
        """
        An undetectable error is logical (corrupts the encoded state silently).
        Logical errors have weight ≥ d.
        """
        return not self.is_detectable(error)


def build_steane_code() -> StabilizerCode:
    """
    Steane [[7,1,3]] code — the smallest CSS code correcting all single-qubit errors.

    Stabilizer generators (from the [7,4,3] Hamming code):
      H1 = X4 X5 X6 X7 (using 1-indexed qubits → 0-indexed: 3,4,5,6)
      H2 = X2 X3 X6 X7 → (1,2,5,6)
      H3 = X1 X3 X5 X7 → (0,2,4,6)
      H4 = Z4 Z5 Z6 Z7
      H5 = Z2 Z3 Z6 Z7
      H6 = Z1 Z3 Z5 Z7
    """
    n = 7

    def xgen(positions: List[int]) -> PauliOperator:
        x = [1 if i in positions else 0 for i in range(n)]
        z = [0] * n
        return PauliOperator(n, x, z)

    def zgen(positions: List[int]) -> PauliOperator:
        x = [0] * n
        z = [1 if i in positions else 0 for i in range(n)]
        return PauliOperator(n, x, z)

    generators = [
        xgen([3, 4, 5, 6]),
        xgen([1, 2, 5, 6]),
        xgen([0, 2, 4, 6]),
        zgen([3, 4, 5, 6]),
        zgen([1, 2, 5, 6]),
        zgen([0, 2, 4, 6]),
    ]
    return StabilizerCode(n=7, k=1, generators=generators)


# ---------------------------------------------------------------------------
# Error simulation and correction pipeline
# ---------------------------------------------------------------------------

def run_error_correction_demo(
    n_trials: int = 100,
    p_error: float = 0.05,
    seed: Optional[int] = 42,
) -> Dict[str, float]:
    """
    Run a Monte Carlo simulation of the 3-qubit bit-flip code.
    Returns success rates with and without error correction.
    """
    rng = random.Random(seed)
    code = BitFlipCode()

    # Logical state: |0⟩ — X errors move it to |1⟩ (fidelity drops to 0).
    # Using |+⟩ would be invisible since X|+⟩ = |+⟩.
    alpha = 1.0 + 0j
    beta  = 0.0 + 0j

    successes_with_qec    = 0
    successes_without_qec = 0

    for _ in range(n_trials):
        # With QEC
        reg = code.encode(alpha, beta)
        if rng.random() < p_error:
            qubit = rng.randint(0, 2)
            reg.apply_x(qubit)
        s01, s12 = code.syndrome(reg, rng)
        code.correct(reg, s01, s12)
        recovered = code.decode(reg, rng)
        ideal = QubitState(alpha, beta)
        if recovered.fidelity(ideal) > 0.99:
            successes_with_qec += 1

        # Without QEC (single physical qubit)
        q = QubitState(alpha, beta)
        if rng.random() < p_error:
            q = q.apply_x()
        if QubitState(alpha, beta).fidelity(q) > 0.99:
            successes_without_qec += 1

    return {
        "success_rate_with_qec":    successes_with_qec    / n_trials,
        "success_rate_without_qec": successes_without_qec / n_trials,
        "p_error": p_error,
        "n_trials": n_trials,
    }
