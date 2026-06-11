"""
q_shor.py — Shor's Factoring Algorithm (classical simulation)

Reference: P. Shor, "Polynomial-Time Algorithms for Prime Factorization and
           Discrete Logarithms on a Quantum Computer", SIAM J. Comput. 1997.
           https://arxiv.org/abs/quant-ph/9508027

Why Shor's algorithm matters:
  RSA security relies on the hardness of factoring large numbers.
  The best classical algorithm (GNFS) runs in exp(O(n^(1/3))).
  Shor's quantum algorithm runs in O(n³) — polynomial time.
  A 4000-qubit fault-tolerant quantum computer would break RSA-2048.

Structure of the algorithm:
  1. Classical pre-processing: reduce factoring to period-finding
  2. Quantum period-finding: quantum Fourier transform finds period r
     of f(x) = a^x mod N
  3. Classical post-processing: extract factors from the period r

This simulation replaces the quantum period-finding step with a classical
implementation of QFT on a statevector. It is exponential in memory (we
simulate 2*n qubits), but algorithmically faithful: the same steps run
on a real quantum computer.

Modules:
  - ModExp: modular exponentiation and period-finding (classical oracle)
  - QuantumPeriodFinder: simulate the quantum circuit for period finding
  - ShorFactorer: full algorithm tying everything together
"""

import math
import random
import fractions
from typing import Optional, List, Tuple
import cmath


# ---------------------------------------------------------------------------
# Classical number theory helpers
# ---------------------------------------------------------------------------

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def is_prime(n: int) -> bool:
    """Miller-Rabin primality test (deterministic for n < 3,317,044,064,679,887,385,961,981)."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    # Witnesses sufficient for n < 3.3e24
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def modular_sqrt(a: int, p: int) -> Optional[int]:
    """Tonelli-Shanks: find x such that x² ≡ a (mod p)."""
    if a == 0:
        return 0
    if pow(a, (p - 1) // 2, p) != 1:
        return None
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    # General Tonelli-Shanks
    s, q = 0, p - 1
    while q % 2 == 0:
        s += 1
        q //= 2
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while True:
        if t == 0:
            return 0
        if t == 1:
            return r
        i, temp = 0, t
        while temp != 1:
            temp = temp * temp % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, b * b % p, t * b * b % p, r * b % p


def find_period_classical(a: int, N: int) -> Optional[int]:
    """
    Classical period-finding: smallest r > 0 such that a^r ≡ 1 (mod N).
    O(N) in the worst case — the quantum version does this in O(poly(log N)).
    Used as a reference / fallback in this simulation.
    """
    if gcd(a, N) != 1:
        return None
    x = a
    for r in range(1, N + 1):
        if x == 1:
            return r
        x = x * a % N
    return None


# ---------------------------------------------------------------------------
# Quantum Fourier Transform on statevector (simulation)
# ---------------------------------------------------------------------------

def _qft_statevector(state: List[complex]) -> List[complex]:
    """
    Apply QFT to a statevector using the Cooley-Tukey FFT.
    The QFT maps |j⟩ → (1/√N) Σ_k ω^(jk) |k⟩ where ω = e^(2πi/N).
    """
    N = len(state)
    s = list(state)

    # Bit-reversal
    n_bits = int(math.log2(N))
    for i in range(N):
        j = int(bin(i)[2:].zfill(n_bits)[::-1], 2)
        if j > i:
            s[i], s[j] = s[j], s[i]

    # Cooley-Tukey butterfly
    length = 2
    while length <= N:
        half = length // 2
        w_base = cmath.exp(2j * cmath.pi / length)
        for start in range(0, N, length):
            w = complex(1, 0)
            for k in range(half):
                u = s[start + k]
                v = s[start + k + half] * w
                s[start + k]        = (u + v) / math.sqrt(2)
                s[start + k + half] = (u - v) / math.sqrt(2)
                w *= w_base
        length *= 2
    return s


# ---------------------------------------------------------------------------
# Quantum period finder (simulated)
# ---------------------------------------------------------------------------

class QuantumPeriodFinder:
    """
    Simulate the quantum circuit for period-finding.

    The quantum circuit:
      1. Initialize |0⟩|1⟩ — control register + work register
      2. Apply H⊗n to control register → uniform superposition
      3. Apply controlled-U^k gates: |x⟩|y⟩ → |x⟩|y · a^x mod N⟩
         This entangles the control with the modular exponentiation
      4. Measure work register (collapses control to period superposition)
      5. Apply inverse QFT to control register
      6. Measure control register — gives k/r for small k

    Parameters
    ----------
    N : int
        Number to factor (or whose period we want)
    n_control_bits : int
        Size of the control register (2·ceil(log2(N)) in production)
    """

    def __init__(self, N: int, n_control_bits: Optional[int] = None):
        self.N = N
        if n_control_bits is None:
            n_control_bits = 2 * math.ceil(math.log2(max(N, 2)))
        self.n_control_bits = n_control_bits
        self.dim = 1 << n_control_bits

    def find_period(self, a: int, seed: Optional[int] = None) -> Optional[int]:
        """
        Simulate one run of the quantum period-finding circuit for f(x) = a^x mod N.
        Returns a period candidate (may need classical post-processing).
        """
        rng = random.Random(seed)
        dim = self.dim

        # Step 1-2: create uniform superposition over control register
        amp = 1.0 / math.sqrt(dim)
        control = [complex(amp, 0)] * dim

        # Step 3: entangle control with modular exponentiation
        # After controlled-U, the state is:
        #   (1/√dim) Σ_x |x⟩|a^x mod N⟩
        # Measuring the work register collapses to some value m = a^x0 mod N
        # and the control becomes: (1/√(dim/r)) Σ_j |x0 + j·r⟩

        # Simulate the measurement of the work register
        x0 = rng.randint(0, self.N - 1)
        # Build the post-measurement control state
        # We need to enumerate x such that a^x ≡ m (mod N) within range [0, dim)
        # This is: x0, x0+r, x0+2r, ... (periodic with period r)
        # We don't know r yet, so we construct the state using the actual values

        # Compute all a^x mod N for x in [0, dim) and find which equal a^x0 mod N
        m = pow(a, x0, self.N)
        matching_x = [x for x in range(dim) if pow(a, x, self.N) == m]

        if not matching_x:
            return None

        # Normalize the post-measurement state
        n_matching = len(matching_x)
        collapsed_amp = 1.0 / math.sqrt(n_matching)
        collapsed = [0j] * dim
        for x in matching_x:
            collapsed[x] = complex(collapsed_amp, 0)

        # Step 4-5: apply inverse QFT to collapsed control register
        # (inverse QFT reveals frequency peaks at multiples of dim/r)
        transformed = _qft_statevector(collapsed)

        # Step 6: measure — sample from probability distribution
        probs = [abs(a_) ** 2 for a_ in transformed]
        total = sum(probs)
        if total == 0:
            return None

        # Normalize (floating point drift)
        probs = [p / total for p in probs]

        r_val = rng.random()
        cumulative = 0.0
        measured = dim - 1
        for i, p in enumerate(probs):
            cumulative += p
            if r_val < cumulative:
                measured = i
                break

        # Step 6 post-processing: measured ≈ k·(dim/r) for some k
        # Use continued fractions to find r from the fraction k/r ≈ measured/dim
        if measured == 0:
            return None

        frac = fractions.Fraction(measured, dim).limit_denominator(self.N)
        return frac.denominator


# ---------------------------------------------------------------------------
# Shor's factoring algorithm
# ---------------------------------------------------------------------------

class ShorFactorer:
    """
    Shor's algorithm for integer factorization.

    Input:  composite integer N
    Output: non-trivial factor of N, or None

    Algorithm:
      Repeat:
        1. Pick random a ∈ (2, N-1) with gcd(a, N) = 1
        2. Find period r of f(x) = a^x mod N (quantum step)
        3. If r is odd or a^(r/2) ≡ -1 (mod N): retry
        4. gcd(a^(r/2) ± 1, N) gives a non-trivial factor

    Classical success probability ≥ 1/2 per random a (proven).
    """

    def __init__(self, use_quantum_simulation: bool = True, max_attempts: int = 30):
        self.use_quantum_simulation = use_quantum_simulation
        self.max_attempts = max_attempts

    def factor(self, N: int, seed: Optional[int] = None) -> Optional[int]:
        """
        Factor N. Returns a non-trivial factor p (where 1 < p < N),
        or None if factoring failed (increase max_attempts or N is prime).
        """
        if N < 4:
            raise ValueError("N must be ≥ 4")
        if is_prime(N):
            raise ValueError(f"{N} is prime — nothing to factor")
        if N % 2 == 0:
            return 2

        # Check for perfect powers: N = a^k → return a
        perfect = self._try_perfect_power(N)
        if perfect:
            return perfect

        rng = random.Random(seed)
        qpf = QuantumPeriodFinder(N) if self.use_quantum_simulation else None

        for attempt in range(self.max_attempts):
            # Step 1: random a coprime to N
            a = rng.randint(2, N - 1)
            g = gcd(a, N)
            if g > 1:
                return g  # lucky: gcd already gives a factor

            # Step 2: find period
            if self.use_quantum_simulation and qpf is not None:
                r = qpf.find_period(a, seed=rng.randint(0, 2**31))
                if r is None or r == 0:
                    continue
                # Validate quantum result against classical check for small N
                if pow(a, r, N) != 1:
                    r = find_period_classical(a, N)
            else:
                r = find_period_classical(a, N)

            if r is None or r % 2 != 0:
                continue

            # Step 3: check a^(r/2) ≢ -1 (mod N)
            half = pow(a, r // 2, N)
            if half == N - 1:
                continue

            # Step 4: extract factors
            f1 = gcd(half + 1, N)
            f2 = gcd(half - 1, N)

            for f in (f1, f2):
                if 1 < f < N:
                    return f

        return None

    def full_factorization(self, N: int, seed: Optional[int] = None) -> List[int]:
        """
        Recursively factor N into all prime factors.
        Returns sorted list of primes (with repetition).
        """
        if N <= 1:
            return []
        if is_prime(N):
            return [N]

        f = self.factor(N, seed=seed)
        if f is None:
            return [N]  # gave up — treat as prime

        factors_left  = self.full_factorization(f, seed=seed)
        factors_right = self.full_factorization(N // f, seed=seed)
        return sorted(factors_left + factors_right)

    def _try_perfect_power(self, N: int) -> Optional[int]:
        """Check if N = a^k for k ≥ 2, return a if so."""
        for k in range(2, math.floor(math.log2(N)) + 1):
            a = round(N ** (1 / k))
            for candidate in (a - 1, a, a + 1):
                if candidate > 1 and candidate ** k == N:
                    return candidate
        return None


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def demonstrate_why_rsa_breaks() -> str:
    """
    Textual explanation of why Shor's breaks RSA.
    RSA: public key (N=p·q, e), private key d = e⁻¹ mod λ(N).
    Breaking RSA requires factoring N to find p, q, then compute λ(N).
    """
    lines = [
        "Why Shor's algorithm breaks RSA:",
        "",
        "  RSA security: given N = p·q (product of two large primes),",
        "  it is classically hard to find p and q.",
        "",
        "  Classical best: General Number Field Sieve",
        "    Runtime: exp(O((log N)^(1/3) · (log log N)^(2/3)))",
        "    RSA-2048 would take ~10^25 CPU-years.",
        "",
        "  Shor's quantum algorithm:",
        "    Runtime: O((log N)^3) — polynomial!",
        "    A 4096-qubit fault-tolerant QC breaks RSA-2048 in hours.",
        "",
        "  Post-quantum solution: replace RSA/ECC with lattice-based",
        "  algorithms (CRYSTALS-Kyber, CRYSTALS-Dilithium) which are",
        "  resistant to both classical and quantum attacks.",
    ]
    return "\n".join(lines)
