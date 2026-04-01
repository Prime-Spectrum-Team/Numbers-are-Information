"""
spectral_matrix.py — Core module for the PrimeSpec framework.

Implements:
  - Seven-class topological classification of natural numbers
  - 7×7 Cayley multiplication table
  - SpectralAddress (K-dimensional p-adic valuation vector)
  - Resonance relation

Based on:
  "Part 1: Algebraic Foundations of Prime Factorization —
   Topological Classification and Multiplicative Semigroup Structure"
  Prime-Spectrum-Team, March 2026

Classes:
  S0 = Unit       (n = 1)
  S1 = Solar      (prime p ≡ 1 mod 6)
  S2 = Lunar      (prime p ≡ 5 mod 6)
  S3 = Dyadic     (n = 2)
  S4 = Triadic    (n = 3)
  S5 = Semiprime  (Ω(n) = 2)
  S6 = Multiprime (Ω(n) ≥ 3)
"""

from __future__ import annotations
from typing import List, Tuple
import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S0, S1, S2, S3, S4, S5, S6 = 0, 1, 2, 3, 4, 5, 6
N_CLASSES = 7

CLASS_NAME = {
    S0: "Unit",
    S1: "Solar",
    S2: "Lunar",
    S3: "Dyadic",
    S4: "Triadic",
    S5: "Semiprime",
    S6: "Multiprime",
}

CLASS_SYMBOL = {
    S0: "S₀", S1: "S₁", S2: "S₂", S3: "S₃",
    S4: "S₄", S5: "S₅", S6: "S₆",
}

# Default prime basis for SpectralAddress (K=7)
BASIS: Tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17)

# ---------------------------------------------------------------------------
# Cayley multiplication table  (MUL_TABLE[i][j] = cl(a*b) for a∈Sᵢ, b∈Sⱼ)
# Derived from Ω-additivity: Ω(a·b) = Ω(a) + Ω(b)
#
#       S0  S1  S2  S3  S4  S5  S6
# S0  [ S0  S1  S2  S3  S4  S5  S6 ]  identity row
# S1  [ S1  S5  S5  S5  S5  S6  S6 ]  prime × anything
# S2  [ S2  S5  S5  S5  S5  S6  S6 ]
# S3  [ S3  S5  S5  S5  S5  S6  S6 ]
# S4  [ S4  S5  S5  S5  S5  S6  S6 ]
# S5  [ S5  S6  S6  S6  S6  S6  S6 ]  semiprime × anything ≥ prime → Ω≥3
# S6  [ S6  S6  S6  S6  S6  S6  S6 ]  absorbing element
# ---------------------------------------------------------------------------

MUL_TABLE: List[List[int]] = [
    # S0   S1   S2   S3   S4   S5   S6
    [S0,  S1,  S2,  S3,  S4,  S5,  S6],   # S0 (identity)
    [S1,  S5,  S5,  S5,  S5,  S6,  S6],   # S1 Solar
    [S2,  S5,  S5,  S5,  S5,  S6,  S6],   # S2 Lunar
    [S3,  S5,  S5,  S5,  S5,  S6,  S6],   # S3 Dyadic
    [S4,  S5,  S5,  S5,  S5,  S6,  S6],   # S4 Triadic
    [S5,  S6,  S6,  S6,  S6,  S6,  S6],   # S5 Semiprime
    [S6,  S6,  S6,  S6,  S6,  S6,  S6],   # S6 Multiprime (absorbing)
]

# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------

def omega(n: int) -> int:
    """Ω(n): number of prime factors with multiplicity. Ω(1) = 0."""
    if n <= 1:
        return 0
    count = 0
    d = 2
    while d * d <= n:
        while n % d == 0:
            count += 1
            n //= d
        d += 1
    if n > 1:
        count += 1
    return count


def is_prime(n: int) -> bool:
    """Miller-style primality for small n."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(n: int) -> int:
    """
    Assign n ∈ ℕ to one of {S0..S6}.

    Definition (Definition 3 in the paper):
      S0: n = 1
      S3: n = 2
      S4: n = 3
      S1: prime p > 3, p ≡ 1 (mod 6)
      S2: prime p > 3, p ≡ 5 (mod 6)
      S5: Ω(n) = 2
      S6: Ω(n) ≥ 3
    """
    if n <= 0:
        raise ValueError(f"classify expects n ≥ 1, got {n}")
    if n == 1:
        return S0
    if n == 2:
        return S3
    if n == 3:
        return S4
    if is_prime(n):
        r = n % 6
        if r == 1:
            return S1
        if r == 5:
            return S2
        # p=2,3 already handled; should not reach here
        raise RuntimeError(f"Prime {n} has unexpected residue mod 6: {r}")
    w = omega(n)
    if w == 2:
        return S5
    # w >= 3
    return S6


def mul_class(a: int, b: int) -> int:
    """Cayley table lookup: cl(a) ⊗ cl(b)."""
    return MUL_TABLE[classify(a)][classify(b)]


# ---------------------------------------------------------------------------
# SpectralAddress
# ---------------------------------------------------------------------------

class SpectralAddress:
    """
    K-dimensional p-adic valuation vector for a natural number.

    SA(n)[i] = ν_{p_i}(n)  where p_i ∈ BASIS.

    For B-smooth integers (no prime factor outside BASIS), SA is lossless.
    For non-smooth integers, the remainder r = n / ∏ p_i^v_i is tracked.
    """

    __slots__ = ("v", "remainder", "n_orig")

    def __init__(self, v: List[int], remainder: int = 1, n_orig: int = 0):
        self.v = list(v)
        self.remainder = remainder   # prime factors outside BASIS
        self.n_orig = n_orig

    # ---- constructors -------------------------------------------------------

    @classmethod
    def from_int(cls, n: int, basis: Tuple[int, ...] = BASIS) -> "SpectralAddress":
        """Compute SA(n) with respect to basis."""
        if n <= 0:
            raise ValueError(f"SpectralAddress.from_int: n must be ≥ 1, got {n}")
        v = []
        remainder = n
        for p in basis:
            exp = 0
            while remainder % p == 0:
                exp += 1
                remainder //= p
            v.append(exp)
        return cls(v, remainder, n)

    @classmethod
    def zero(cls, k: int = len(BASIS)) -> "SpectralAddress":
        return cls([0] * k, 1, 1)

    # ---- algebraic operations -----------------------------------------------

    def __add__(self, other: "SpectralAddress") -> "SpectralAddress":
        """SA(a·b) = SA(a) + SA(b)  (component-wise, ignoring remainder)."""
        v = [x + y for x, y in zip(self.v, other.v)]
        return SpectralAddress(v, self.remainder * other.remainder)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpectralAddress):
            return NotImplemented
        return self.v == other.v and self.remainder == other.remainder

    def to_int(self) -> int:
        """
        Reconstruct integer from SA.
        Exact iff remainder == 1 (B-smooth).
        """
        result = self.remainder
        for exp, p in zip(self.v, BASIS):
            result *= p ** exp
        return result

    def l1_norm(self) -> int:
        return sum(self.v)

    def gcd_sa(self, other: "SpectralAddress") -> "SpectralAddress":
        """SA(gcd(a,b)) = component-wise min."""
        v = [min(x, y) for x, y in zip(self.v, other.v)]
        return SpectralAddress(v)

    # Alias: test code calls .gcd() and .lcm()
    def gcd(self, other: "SpectralAddress") -> "SpectralAddress":
        return self.gcd_sa(other)

    def lcm_sa(self, other: "SpectralAddress") -> "SpectralAddress":
        """SA(lcm(a,b)) = component-wise max."""
        v = [max(x, y) for x, y in zip(self.v, other.v)]
        return SpectralAddress(v)

    def lcm(self, other: "SpectralAddress") -> "SpectralAddress":
        return self.lcm_sa(other)

    def __mul__(self, other: "SpectralAddress") -> "SpectralAddress":
        """SA(a·b) = SA(a) + SA(b)."""
        return self.__add__(other)

    def dot(self, other: "SpectralAddress") -> int:
        return sum(x * y for x, y in zip(self.v, other.v))

    def is_coprime_to(self, other: "SpectralAddress") -> bool:
        """gcd(a,b)=1 iff dot product of SA vectors = 0."""
        return self.dot(other) == 0

    def is_b_smooth(self) -> bool:
        return self.remainder == 1

    # ---- resonance ----------------------------------------------------------

    def resonance(self, other: "SpectralAddress") -> int:
        """R(i,j) = ‖min(SA(i), SA(j))‖₁ = Ω(gcd(i,j))."""
        return sum(min(x, y) for x, y in zip(self.v, other.v))

    # ---- display ------------------------------------------------------------

    def __repr__(self) -> str:
        r = f"SA({self.n_orig})" if self.n_orig else "SA"
        r += f" = {self.v}"
        if self.remainder > 1:
            r += f" × {self.remainder}*"
        return r

    def spectral_class(self) -> int:
        """Alias for classify() — recover spectral class from SA."""
        return self.classify()

    def classify(self) -> int:
        """Recover spectral class from SA."""
        omega_val = self.l1_norm()
        if self.remainder > 1:
            # Non-smooth: remainder contributes at least 1 to Ω
            omega_val += omega(self.remainder)
        if omega_val == 0:
            return S0
        if omega_val == 1:
            # Reconstruct n to check mod 6
            n = self.to_int()
            if n == 2:
                return S3
            if n == 3:
                return S4
            r = n % 6
            return S1 if r == 1 else S2
        if omega_val == 2:
            return S5
        return S6


# ---------------------------------------------------------------------------
# Resonance (module-level)
# ---------------------------------------------------------------------------

def resonance(i: int, j: int) -> int:
    """R(i,j) = Ω(gcd(i,j)) — structural affinity between two integers."""
    return omega(math.gcd(i, j))


def is_resonant(i: int, j: int) -> bool:
    return math.gcd(i, j) > 1


def class_pow(cls: int, k: int) -> int:
    """
    Compute the class of n^k given the class of n.

    Power dynamics (Theorem 5):
      Ω(n^k) = k · Ω(n)

      S0 (Ω=0): stays S0 for all k
      S3/S4 (Ω=1 singleton): k=1→same, k=2→S5, k≥3→S6
      S1/S2 (Ω=1 prime):     k=1→same, k=2→S5, k≥3→S6
      S5 (Ω=2):              k=1→S5, k≥2→S6
      S6 (Ω≥3):              always S6
    """
    if k < 1:
        raise ValueError(f"class_pow: k must be ≥ 1, got {k}")

    if cls == S0:
        return S0

    # Ω of the class
    omega_map = {S0: 0, S1: 1, S2: 1, S3: 1, S4: 1, S5: 2, S6: 3}
    base_omega = omega_map[cls]
    result_omega = base_omega * k

    if result_omega == 0:
        return S0
    if result_omega == 1:
        # Only possible when k=1 and base is a prime class
        return cls
    if result_omega == 2:
        return S5
    # result_omega >= 3
    return S6


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Minimal smoke test — runs on import if __name__ == '__main__'."""
    # Classification
    assert classify(1) == S0
    assert classify(2) == S3
    assert classify(3) == S4
    assert classify(7) == S1    # 7 ≡ 1 mod 6
    assert classify(5) == S2    # 5 ≡ 5 mod 6
    assert classify(4) == S5    # 4 = 2²; Ω = 2
    assert classify(8) == S6    # 8 = 2³; Ω = 3

    # Cayley table: prime × prime → semiprime
    assert MUL_TABLE[S1][S1] == S5
    assert MUL_TABLE[S5][S1] == S6
    assert MUL_TABLE[S6][S6] == S6

    # SpectralAddress
    sa60 = SpectralAddress.from_int(60)    # 60 = 2²·3·5
    assert sa60.v[:3] == [2, 1, 1], sa60
    assert sa60.to_int() == 60
    assert sa60.l1_norm() == 4             # Ω(60) = 4

    sa7 = SpectralAddress.from_int(7)
    sa11 = SpectralAddress.from_int(11)
    assert sa7.is_coprime_to(sa11)         # gcd(7,11) = 1
    assert sa7.resonance(sa11) == 0

    sa6 = SpectralAddress.from_int(6)      # 6 = 2·3
    sa4 = SpectralAddress.from_int(4)      # 4 = 2²
    assert sa6.resonance(sa4) == 1         # gcd(6,4) = 2, Ω(2) = 1

    print("spectral_matrix self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
