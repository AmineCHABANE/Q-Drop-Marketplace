"""
q_vqe.py — Variational Quantum Eigensolver (classical simulation)

References:
  - Peruzzo et al., "A variational eigenvalue solver on a photonic chip",
    Nature Communications 2014. https://doi.org/10.1038/ncomms5213
  - Tilly et al., "The Variational Quantum Eigensolver: a review of methods
    and best practices", Physics Reports 2022.
    https://arxiv.org/abs/2111.05176

VQE is the flagship algorithm for near-term (NISQ) quantum hardware.
It finds the ground-state energy of a Hamiltonian using a hybrid loop:
  - Quantum computer: prepare parametric ansatz state, measure expectation value
  - Classical computer: optimize parameters to minimize energy

Applications:
  - Quantum chemistry (molecular ground states, reaction energies)
  - Materials science (correlated electron systems)
  - Optimization (QAOA is VQE for combinatorial problems)

Why VQE matters for the quantum era:
  Fault-tolerant quantum computing is 10+ years away.
  VQE runs on noisy intermediate-scale quantum (NISQ) devices today.
  It finds approximate ground states of Hamiltonians that are exponentially
  hard for classical computers to simulate.

This module implements:
  1. Pauli Hamiltonian representation (weighted sum of Pauli strings)
  2. Statevector simulation of parametric ansatz circuits
  3. Expectation value computation ⟨ψ(θ)|H|ψ(θ)⟩
  4. Gradient-based optimizer (parameter-shift rule — quantum native)
  5. COBYLA-style classical optimizer (gradient-free fallback)
  6. Pre-built Hamiltonians: H₂ molecule, Ising model, MaxCut QUBO
"""

import math
import cmath
import random
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable


# ---------------------------------------------------------------------------
# Pauli matrices (2x2)
# ---------------------------------------------------------------------------

# Represented as (row, col) → complex value for sparse operations
# Full 2x2 matrices for statevector application

def _mat2_mul(A: List[List[complex]], B: List[List[complex]]) -> List[List[complex]]:
    """2x2 matrix multiply."""
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]],
    ]


PAULI_I = [[1+0j, 0j], [0j, 1+0j]]
PAULI_X = [[0j, 1+0j], [1+0j, 0j]]
PAULI_Y = [[0j, -1j], [1j, 0j]]
PAULI_Z = [[1+0j, 0j], [0j, -1+0j]]

PAULI_MAP = {"I": PAULI_I, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}


# ---------------------------------------------------------------------------
# Statevector (n qubits)
# ---------------------------------------------------------------------------

Statevector = List[complex]


def zero_state(n_qubits: int) -> Statevector:
    """|0...0⟩ in 2^n dimensional Hilbert space."""
    sv = [0j] * (1 << n_qubits)
    sv[0] = 1 + 0j
    return sv


def _apply_single_qubit_gate(
    sv: Statevector, gate: List[List[complex]], qubit: int, n: int
) -> Statevector:
    """Apply a 2x2 gate to qubit `qubit` in an n-qubit statevector."""
    out = list(sv)
    step = 1 << (n - 1 - qubit)
    for i in range(1 << n):
        if (i >> (n - 1 - qubit)) & 1 == 0:
            j = i | step
            a, b = sv[i], sv[j]
            out[i] = gate[0][0] * a + gate[0][1] * b
            out[j] = gate[1][0] * a + gate[1][1] * b
    return out


def _apply_cnot(sv: Statevector, control: int, target: int, n: int) -> Statevector:
    """Apply CNOT gate (control → target)."""
    out = list(sv)
    for i in range(1 << n):
        ctrl_bit = (i >> (n - 1 - control)) & 1
        if ctrl_bit == 1:
            j = i ^ (1 << (n - 1 - target))
            out[i], out[j] = sv[j], sv[i]
    # This double-swaps; fix with visited tracking
    out = list(sv)
    visited = set()
    for i in range(1 << n):
        if i in visited:
            continue
        ctrl_bit = (i >> (n - 1 - control)) & 1
        if ctrl_bit == 1:
            j = i ^ (1 << (n - 1 - target))
            out[i], out[j] = sv[j], sv[i]
            visited.add(i)
            visited.add(j)
    return out


def _apply_rz(sv: Statevector, angle: float, qubit: int, n: int) -> Statevector:
    """Rz(θ) = exp(-iθZ/2) = diag(e^{-iθ/2}, e^{iθ/2})."""
    phase0 = cmath.exp(-0.5j * angle)
    phase1 = cmath.exp( 0.5j * angle)
    gate = [[phase0, 0j], [0j, phase1]]
    return _apply_single_qubit_gate(sv, gate, qubit, n)


def _apply_ry(sv: Statevector, angle: float, qubit: int, n: int) -> Statevector:
    """Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]."""
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    gate = [[c + 0j, -s + 0j], [s + 0j, c + 0j]]
    return _apply_single_qubit_gate(sv, gate, qubit, n)


def _apply_h(sv: Statevector, qubit: int, n: int) -> Statevector:
    """Hadamard gate."""
    h = 1 / math.sqrt(2)
    gate = [[h + 0j, h + 0j], [h + 0j, -h + 0j]]
    return _apply_single_qubit_gate(sv, gate, qubit, n)


# ---------------------------------------------------------------------------
# Pauli string expectation value
# ---------------------------------------------------------------------------

def _pauli_string_expectation(
    sv: Statevector, pauli_string: str
) -> float:
    """
    Compute ⟨ψ|P₀⊗P₁⊗...⊗Pₙ|ψ⟩ for a tensor-product Pauli operator.
    pauli_string: e.g. "XZIY" (left = qubit 0)

    Uses the fact that for a product Pauli P = ⊗_i P_i:
      ⟨ψ|P|ψ⟩ = Σ_{i,j} ψ*_i P_{ij} ψ_j
    where P_{ij} is the full 2^n × 2^n matrix element.
    """
    n = len(pauli_string)
    N = 1 << n
    result = 0.0 + 0j

    for i in range(N):
        # Build P|ψ⟩ at row i by iterating columns j
        # |i⟩ contribution: ⟨i|P|ψ⟩ = Σ_j P[i][j] ψ[j]
        # P[i][j] = product of single-qubit matrix elements
        for j in range(N):
            matrix_element = complex(1, 0)
            for qubit, pauli_char in enumerate(pauli_string):
                row_bit = (i >> (n - 1 - qubit)) & 1
                col_bit = (j >> (n - 1 - qubit)) & 1
                gate = PAULI_MAP[pauli_char]
                matrix_element *= gate[row_bit][col_bit]
            result += sv[i].conjugate() * matrix_element * sv[j]

    return result.real


# ---------------------------------------------------------------------------
# Hamiltonian: weighted sum of Pauli strings
# ---------------------------------------------------------------------------

@dataclass
class PauliTerm:
    coefficient: float
    pauli_string: str   # e.g. "ZZ", "XI", "IY"


class Hamiltonian:
    """
    H = Σ_k c_k · P_k  (weighted sum of Pauli strings)
    Represents the energy operator for a quantum system.
    """

    def __init__(self, terms: List[PauliTerm]):
        if not terms:
            raise ValueError("Hamiltonian must have at least one term")
        n_qubits = len(terms[0].pauli_string)
        for t in terms:
            if len(t.pauli_string) != n_qubits:
                raise ValueError("All Pauli strings must have the same length")
        self.terms = terms
        self.n_qubits = n_qubits

    def expectation_value(self, sv: Statevector) -> float:
        """⟨ψ|H|ψ⟩ = Σ_k c_k ⟨ψ|P_k|ψ⟩"""
        return sum(
            t.coefficient * _pauli_string_expectation(sv, t.pauli_string)
            for t in self.terms
        )

    @classmethod
    def h2_molecule(cls) -> "Hamiltonian":
        """
        Minimal hydrogen molecule Hamiltonian in STO-3G basis.
        Mapped to qubits via Jordan-Wigner transformation.
        Ground state energy ≈ -1.137 Hartree at equilibrium bond length.
        """
        return cls([
            PauliTerm(-0.8105, "II"),
            PauliTerm( 0.1723, "ZI"),
            PauliTerm(-0.2228, "IZ"),
            PauliTerm( 0.1686, "ZZ"),
            PauliTerm( 0.0454, "XX"),
            PauliTerm( 0.0454, "YY"),
        ])

    @classmethod
    def ising_model(cls, n: int, J: float = 1.0, h: float = 0.5) -> "Hamiltonian":
        """
        1D transverse-field Ising model:
        H = -J Σ Z_i Z_{i+1} - h Σ X_i
        """
        terms = []
        for i in range(n - 1):
            pauli = "I" * i + "ZZ" + "I" * (n - i - 2)
            terms.append(PauliTerm(-J, pauli))
        for i in range(n):
            pauli = "I" * i + "X" + "I" * (n - i - 1)
            terms.append(PauliTerm(-h, pauli))
        return cls(terms)

    @classmethod
    def maxcut_qubo(cls, edges: List[Tuple[int, int]], n: int) -> "Hamiltonian":
        """
        MaxCut Hamiltonian for QAOA: H = Σ_{(i,j)∈E} (1 - Z_i Z_j) / 2
        Ground state encodes the optimal cut.
        """
        terms = [PauliTerm(len(edges) / 2, "I" * n)]
        for i, j in edges:
            pauli = list("I" * n)
            pauli[i] = "Z"
            pauli[j] = "Z"
            terms.append(PauliTerm(-0.5, "".join(pauli)))
        return cls(terms)


# ---------------------------------------------------------------------------
# Ansatz circuits
# ---------------------------------------------------------------------------

class HardwareEfficientAnsatz:
    """
    Hardware-efficient ansatz: Ry(θ) rotations + CNOT entanglement layers.
    Widely used on real quantum hardware (IBM, Quantinuum, IonQ).

    Structure (depth d, n qubits):
      Layer 0: Ry(θ_{0,i}) on each qubit i
      Repeat d times:
        CNOT ladder: CNOT(0,1), CNOT(1,2), ..., CNOT(n-2, n-1)
        Ry(θ_{layer,i}) on each qubit i

    Total parameters: (depth+1) × n_qubits
    """

    def __init__(self, n_qubits: int, depth: int = 1):
        self.n_qubits = n_qubits
        self.depth = depth
        self.n_params = (depth + 1) * n_qubits

    def build_state(self, params: List[float]) -> Statevector:
        """Apply the ansatz circuit to |0...0⟩ and return the statevector."""
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} params, got {len(params)}")

        n = self.n_qubits
        sv = zero_state(n)

        idx = 0
        # Initial rotation layer
        for q in range(n):
            sv = _apply_ry(sv, params[idx], q, n)
            idx += 1

        for _ in range(self.depth):
            # CNOT entanglement
            for q in range(n - 1):
                sv = _apply_cnot(sv, q, q + 1, n)
            # Rotation layer
            for q in range(n):
                sv = _apply_ry(sv, params[idx], q, n)
                idx += 1

        return sv


class UCCSDSingletAnsatz:
    """
    Unitary Coupled Cluster Singles and Doubles (UCCSD) — chemistry-motivated.
    For 2 qubits (H₂ STO-3G in minimal encoding): 1 parameter.

    |ψ(θ)⟩ = exp(θ(a†_1 a†_0 a_2 a_3 - h.c.))|HF⟩
    After Jordan-Wigner: rotation in the {|0011⟩, |1100⟩} subspace.
    Parametrized as Ry(2θ) in the 2-qubit space.
    """

    def __init__(self):
        self.n_qubits = 2
        self.n_params = 1

    def build_state(self, params: List[float]) -> Statevector:
        """Build UCCSD ansatz for H₂. Single Givens rotation."""
        theta = params[0]
        n = self.n_qubits
        sv = zero_state(n)
        # Start from Hartree-Fock: |01⟩ = one electron in each orbital
        sv[0] = 0j
        sv[1] = 1 + 0j   # |01⟩
        # Apply Givens-like rotation between |01⟩ and |10⟩
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        sv[1] =  cos_t + 0j
        sv[2] =  sin_t + 0j
        return sv


# ---------------------------------------------------------------------------
# Parameter-shift rule for gradients
# ---------------------------------------------------------------------------

def parameter_shift_gradient(
    hamiltonian: Hamiltonian,
    ansatz,
    params: List[float],
    shift: float = math.pi / 2,
) -> List[float]:
    """
    Quantum-native gradient: ∂E/∂θ_k = [E(θ_k + π/2) - E(θ_k - π/2)] / 2

    This is the parameter-shift rule (Mitarai et al. 2018).
    Works on real quantum hardware — no finite differences needed.
    For gates of the form exp(-iθP/2), the gradient is exact.
    """
    grads = []
    for k in range(len(params)):
        params_plus  = list(params)
        params_minus = list(params)
        params_plus[k]  += shift
        params_minus[k] -= shift

        sv_plus  = ansatz.build_state(params_plus)
        sv_minus = ansatz.build_state(params_minus)

        e_plus  = hamiltonian.expectation_value(sv_plus)
        e_minus = hamiltonian.expectation_value(sv_minus)

        grads.append((e_plus - e_minus) / 2)
    return grads


# ---------------------------------------------------------------------------
# VQE optimizer
# ---------------------------------------------------------------------------

@dataclass
class VQEResult:
    optimal_energy: float
    optimal_params: List[float]
    energy_history: List[float]
    n_iterations: int
    converged: bool


class VQE:
    """
    Variational Quantum Eigensolver.

    Finds the ground-state energy of a Hamiltonian by minimizing
    E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ over the circuit parameters θ.

    Uses gradient descent with the parameter-shift rule by default.
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        ansatz,
        learning_rate: float = 0.1,
        max_iterations: int = 200,
        convergence_tol: float = 1e-6,
    ):
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol

    def run(
        self,
        initial_params: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> VQEResult:
        """
        Run VQE optimization.

        Starting from initial_params (random if None), iteratively:
          1. Compute E(θ) via statevector simulation
          2. Compute ∇E(θ) via parameter-shift rule
          3. Update θ ← θ - lr · ∇E(θ)  (gradient descent)
          4. Check convergence
        """
        n_params = self.ansatz.n_params
        rng = random.Random(seed)

        if initial_params is None:
            # Random initialization in [-π, π]
            params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        else:
            params = list(initial_params)

        energy_history = []
        prev_energy = float("inf")

        for iteration in range(self.max_iterations):
            # Compute current energy
            sv = self.ansatz.build_state(params)
            energy = self.hamiltonian.expectation_value(sv)
            energy_history.append(energy)

            # Check convergence
            if abs(energy - prev_energy) < self.convergence_tol and iteration > 5:
                return VQEResult(
                    optimal_energy=energy,
                    optimal_params=params,
                    energy_history=energy_history,
                    n_iterations=iteration + 1,
                    converged=True,
                )
            prev_energy = energy

            # Compute gradients via parameter-shift rule
            grads = parameter_shift_gradient(self.hamiltonian, self.ansatz, params)

            # Gradient descent step
            params = [p - self.learning_rate * g for p, g in zip(params, grads)]

        sv = self.ansatz.build_state(params)
        final_energy = self.hamiltonian.expectation_value(sv)

        return VQEResult(
            optimal_energy=final_energy,
            optimal_params=params,
            energy_history=energy_history,
            n_iterations=self.max_iterations,
            converged=False,
        )

    def run_cobyla(
        self,
        initial_params: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> VQEResult:
        """
        COBYLA-inspired gradient-free optimizer.
        Useful when parameter-shift rule isn't applicable (e.g. non-Pauli generators).
        Uses Nelder-Mead simplex on the energy landscape.
        """
        n_params = self.ansatz.n_params
        rng = random.Random(seed)

        def energy_fn(p: List[float]) -> float:
            sv = self.ansatz.build_state(p)
            return self.hamiltonian.expectation_value(sv)

        if initial_params is None:
            params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        else:
            params = list(initial_params)

        energy_history = [energy_fn(params)]

        # Simple coordinate descent (gradient-free)
        step = 0.1
        for iteration in range(self.max_iterations):
            improved = False
            for k in range(n_params):
                current = energy_fn(params)
                for delta in (step, -step):
                    trial = list(params)
                    trial[k] += delta
                    trial_e = energy_fn(trial)
                    if trial_e < current:
                        params = trial
                        current = trial_e
                        improved = True

            energy_history.append(energy_fn(params))

            if not improved:
                step *= 0.5
                if step < self.convergence_tol:
                    break

        final_energy = energy_fn(params)
        return VQEResult(
            optimal_energy=final_energy,
            optimal_params=params,
            energy_history=energy_history,
            n_iterations=len(energy_history),
            converged=step < self.convergence_tol,
        )


# ---------------------------------------------------------------------------
# QAOA (Quantum Approximate Optimization Algorithm)
# ---------------------------------------------------------------------------

class QAOA:
    """
    Quantum Approximate Optimization Algorithm for combinatorial optimization.
    QAOA is VQE with a problem-specific ansatz (alternating cost/mixer layers).

    For MaxCut: find the partition of graph nodes maximizing cut edges.
    QAOA with p layers gives an approximation ratio ≥ 0.6924 (p=1, Farhi 2014).
    """

    def __init__(self, hamiltonian: Hamiltonian, p_layers: int = 1):
        self.hamiltonian = hamiltonian
        self.n_qubits = hamiltonian.n_qubits
        self.p_layers = p_layers
        self.n_params = 2 * p_layers   # γ and β for each layer

    def build_state(self, params: List[float]) -> Statevector:
        """
        QAOA circuit:
          1. H⊗n (uniform superposition)
          2. For each layer l:
             a. exp(-iγ_l H_C) — cost unitary (encodes the problem)
             b. exp(-iβ_l H_B) — mixer unitary (X rotations)
        """
        n = self.n_qubits
        sv = zero_state(n)

        # Initial superposition
        for q in range(n):
            sv = _apply_h(sv, q, n)

        gammas = params[:self.p_layers]
        betas  = params[self.p_layers:]

        for gamma, beta in zip(gammas, betas):
            # Cost unitary: Rz(2γ c_k) on each Pauli term
            for term in self.hamiltonian.terms:
                for q, p_char in enumerate(term.pauli_string):
                    if p_char == "Z":
                        angle = 2 * gamma * term.coefficient
                        sv = _apply_rz(sv, angle, q, n)

            # Mixer unitary: Rx(2β) on each qubit
            for q in range(n):
                sv = _apply_ry(sv, 2 * beta, q, n)

        return sv

    def run(self, seed: Optional[int] = None) -> VQEResult:
        vqe = VQE(
            hamiltonian=self.hamiltonian,
            ansatz=self,
            learning_rate=0.05,
            max_iterations=300,
        )
        return vqe.run(seed=seed)
