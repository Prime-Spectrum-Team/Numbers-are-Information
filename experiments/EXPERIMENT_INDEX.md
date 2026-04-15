# Experiment Index

## Overview

This document maps every experiment in the repository to its source file,
result artifact, and shared-module dependency. All Python experiments are
pytest-runnable:

```
python -m pytest experiments/tests/ -v
```

---

## Dependencies

### Python (stdlib)
- `os`, `sys`, `json`, `math`, `time`, `random`, `datetime`
- `collections` (Counter, defaultdict)
- `typing` (List, Tuple, Dict)

### Python (third-party)
| Package   | Version | Required by                                           |
|-----------|---------|-------------------------------------------------------|
| sympy     | >=1.12  | test_exp_04_1                                         |
| torch     | >=2.0   | test_exp_00_0, hw_verification/exp_hw_exact_spectral.py |
| pytest    | >=7.0   | Test runner (all test_exp_*.py)                       |
| scipy     | optional | CI computation in test_exp_01_1, 01_2, 02_3b, 03_1, 05_1, 05_2, 10_0 (fallback if absent) |

### Internal
| Module                    | Provides                                                             | Used by        |
|---------------------------|----------------------------------------------------------------------|----------------|
| shared/spectral_matrix.py | classify, CLASS_NAME, MUL_TABLE, class_pow, SpectralAddress, BASIS   | 15 of 19 tests |

### C++ (for exp_hw_bench_cpp20)
| Requirement | Version |
|-------------|---------|
| g++/clang++ | C++20   |
| Flags       | -O3 -march=native -std=c++20 |

---

## Experiment-to-File Mapping

| ID     | Name                                    | Test file                                       | Result file                            | Uses spectral_matrix |
|--------|-----------------------------------------|-------------------------------------------------|----------------------------------------|:--------------------:|
| 00.0   | SpectralMemory benchmark proof          | tests/test_exp_00_0_benchmark_proof.py          | exp_00_0_benchmark_proof.json          | no                   |
| 01.0   | Maximality of 7-class partition (illustrative) | tests/test_exp_01_0_maximality.py        | exp_01_0_maximality.json               | yes                  |
| 01.0b  | Partition-lattice enumeration (Bell(7)=877)    | tests/test_exp_01_0b_partition_lattice.py | exp_01_0b_partition_lattice.json      | yes                  |
| 01.1   | Omega-function complete additivity      | tests/test_exp_01_1_omega_additivity.py         | exp_01_1_omega_additivity.json         | yes                  |
| 01.2   | Prime residue classes mod 6             | tests/test_exp_01_2_prime_residues.py           | exp_01_2_prime_residues.json           | yes                  |
| 02.1   | Partition completeness                  | tests/test_exp_02_1_partition_completeness.py   | exp_02_1_partition_completeness.json   | yes                  |
| 02.3a  | Coprime sparsity density 6/pi^2        | tests/test_exp_02_3a_sparsity.py                | exp_02_3a_sparsity.json               | no                   |
| 02.3b  | Semigroup identity and absorption       | tests/test_exp_02_3b_semigroup_elements.py      | exp_02_3b_semigroup_elements.json      | yes                  |
| 03.1   | Power dynamics and universality         | tests/test_exp_03_1_power_dynamics.py           | exp_03_1_power_dynamics.json           | yes                  |
| 04.1   | Solar + Lunar deterministic addition    | tests/test_exp_04_1_solar_lunar_addition.py     | exp_04_1_solar_lunar_addition.json     | no                   |
| 05.1   | SpectralAddress bijection               | tests/test_exp_05_1_spectral_bijection.py       | exp_05_1_spectral_bijection.json       | yes                  |
| 05.2   | Spectral algebraic isomorphisms (40K)   | tests/test_exp_05_2_spectral_isomorphisms.py    | exp_05_2_spectral_isomorphisms.json    | yes                  |
| 05.3a  | Classification recovery from SA         | tests/test_exp_05_3a_classification_recovery.py | exp_05_3a_classification_recovery.json | yes                  |
| 05.3b  | B-smooth boundary limitation            | tests/test_exp_05_3b_b_smooth_boundary.py       | exp_05_3b_b_smooth_boundary.json       | yes                  |
| 06.2   | Multi-scale resonance hierarchy         | tests/test_exp_06_2_resonance_hierarchy.py      | exp_06_2_resonance_hierarchy.json      | yes                  |
| 07.1   | Cayley multiplication table             | tests/test_exp_07_1_cayley_table.py             | exp_07_1_cayley_table.json             | yes                  |
| 10.0   | Addition laws with 95% CI (28 pairs)   | tests/test_exp_10_0_addition_ci.py              | exp_10_0_addition_ci.json              | yes                  |
| S0.S2  | S0+S2 non-determinism proof             | tests/test_exp_S0_S2_counterexample.py          | exp_S0_S2_counterexample.json          | yes                  |
| hw.vlog| Verilog behavioral emulation            | tests/test_exp_hw_verilog_behavioral.py         | exp_hw_verilog_behavioral.json         | no                   |

### Non-pytest experiments

| ID      | Name                          | Source file                              | Result file                  | Uses spectral_matrix |
|---------|-------------------------------|------------------------------------------|------------------------------|:--------------------:|
| hw.exact| HW exact spectral emulation   | hw_verification/exp_hw_exact_nontorch.py | exp_hw_exact.json            | no                   |
| hw.bench| C++20 performance benchmark   | scripts/exp_hw_bench_cpp20.cpp           | exp_hw_bench_cpp20.json      | no                   |

All result files are stored in `experiments/results/`.

---

## Experiment Descriptions

### E00.0 -- SpectralMemory benchmark proof
Benchmarks the SpectralMemory architecture against standard attention across
five tests: context scaling (T=128-1024), latency scaling, memory scaling,
infinite retrieval (10K-500K distance), and sliding window comparison.
Requires PyTorch; imports from `hw_verification/exp_hw_exact_spectral.py`.

### E01.0 -- Maximality of 7-class partition (illustrative)
Three-step argument for the maximality of the 7-class partition
{S0..S6}: multiplicative equivalence of S1-S4, additive distinction of
S1 vs S2, and vacuity of S5/S6 splits. Serves as didactic proof; the
rigorous exhaustive version is E01.0b. References paper Theorem
thm:maximality.

### E01.0b -- Exhaustive partition-lattice enumeration
Enumerates all Bell(7)=877 set partitions of the ground set {S0..S6}
via restricted-growth-string recursion and verifies two quotient
conditions per partition: (i) otimes-closure (well-definedness of the
induced multiplication) and (ii) preservation of all 7 deterministic
addition laws from paper Table tab:7laws. Result: 68 partitions are
otimes-closed, 7 preserve all laws, and these 7 form a chain with the
identity 7-partition as the unique maximal (finest) element.
Empirical output-class sets per 28-pair are loaded from
exp_10_0_addition_ci.json. Runtime approx 0.06s.
References paper Theorem thm:maximality.

### E01.1 -- Omega-function complete additivity
Verifies Omega(a*b) = Omega(a) + Omega(b) for 100,000 random pairs from
[1, 1,000,000] with 95% confidence intervals.
References P1 Section 1.1.

### E01.2 -- Prime residue classes mod 6
Confirms that all 664,579 primes below 10^7 (beyond 2 and 3) belong to
residue classes {1, 5} mod 6 with approximately equal Dirichlet density.
References P1 Section 1.2.

### E02.1 -- Partition completeness
Verifies Proposition 3.2: every n in [1, 100,000] belongs to exactly one
of the seven PrimeSpec classes (S0..S6). Exhaustive and mutually exclusive.

### E02.3a -- Coprime sparsity density 6/pi^2
Numerically verifies that the fraction of coprime pairs converges to
6/pi^2 (approx 0.6079) as T grows, confirming sparsity of the resonance
mask. Does not use spectral_matrix (only stdlib math).
References P1 Section VI-C.

### E02.3b -- Semigroup identity and absorption
Tests 1,400 multiplication pairs verifying S0 (Unit) as the identity element
and S6 (Multiprime) as the absorption element of the semigroup (S, x).
References P1 Section 2.3.

### E03.1 -- Power dynamics and universality
5,600 power trajectory tests plus 99,999 convergence tests validate universal
reachability of S6 via exponentiation. Verifies that for n >= 2,
cl(n^k) = S6 is reached within k <= ceil(3/Omega(n)) steps.
References P1 Sections 3.1 and 3.2.

### E04.1 -- Solar + Lunar deterministic addition
Proves that for any Solar prime p (p = 1 mod 6) and Lunar prime q (q = 5 mod 6),
cl(p+q) = S6 (Multiprime). Tests 90,000 random pairs from primes up to 100,000.
Uses sympy for factorization; does not use spectral_matrix.

### E05.1 -- SpectralAddress bijection
Verifies perfect round-trip bijection SA(n) -> n for 10,000 B-smooth integers
(composed only of basis primes {2,3,5,7,11,13,17}).
References P1 Section 5.1.

### E05.2 -- Spectral algebraic isomorphisms
40,000 tests verify four algebraic isomorphisms: multiplication maps to
vector addition, GCD maps to component-wise min, LCM maps to component-wise
max, and coprimality maps to zero inner product (4 properties x 10K each).
References P1 Section 5.2.

### E05.3a -- Classification recovery from SpectralAddress
Verifies that for 100,000 random B-smooth integers (basis {2,3,5,7,11,13,17}),
direct classification cl(n) agrees perfectly with classification inferred
from SA(n). Unique count after deduplication.
References P1 Section 5.3.

### E05.3b -- B-smooth boundary limitation
Documents that SpectralAddress bijection is lossless only for B-smooth
integers. For K=7 basis, the first breakpoint is n=19 (prime > 17).
This defines the domain of validity, not a failure.

### E06.2 -- Multi-scale resonance hierarchy
Analyzes a 2,100-element sequence for multi-scale resonance structure.
Empirical pair densities of shared prime factors match theoretical predictions
((1/2)^2, (1/6)^2, (1/30)^2, (1/210)^2) within 2% relative tolerance
(20% for type-2357 due to small sample size).
References P1 Section 6.2.

### E07.1 -- Cayley multiplication table
9,800 class multiplication pairs verify the 7x7 Cayley table with zero
deviations (200 representative pairs x 49 cells).
References P1 Section 2.2.

### E10.0 -- Addition laws with 95% CI
Reports all 28 symmetric class-addition pair distributions with 95%
binomial confidence intervals at N=100K sample size. Each pair carries
an `is_trivial` flag (set when at least one operand is a singleton
class S0={1}, S3={2}, S4={3}; paper Definition def:trivial-det).
Summary split: 1 non-trivial + 6 trivial (= 7 deterministic) + 21
probabilistic.
References paper Section 4 (Theorem thm:7laws, Table tab:7laws).

### ES0.S2 -- S0+S2 non-determinism proof
Demonstrates that S0+S2 is not mathematically deterministic by providing
a counterexample: cl(1+5) = cl(6) = S5, not S6. Distinguishes statistical
near-certainty from mathematical determinism.

### EHW.VLOG -- Verilog behavioral emulation
Behavioral emulation of spectral_cell.v: builds a 1024-entry LUT, simulates
a 1-cycle registered pipeline, and runs 14 test vectors against expected
outputs. Standalone; does not use spectral_matrix.

### EHW.EXACT -- HW exact spectral emulation (non-pytest)
Torch-free verification of the hardware spectral pipeline. Ports
SpectralAddressExtractor and SpectralAndGate without PyTorch dependency,
suitable for CI and paper verification.

### EHW.BENCH -- C++20 performance benchmark (non-pytest)
Measures performance of Cayley lookup, SpectralAddress computation, and
SA GCD (component-wise min) in C++20 with -O3 optimization.

---

## Shared Module

### shared/spectral_matrix.py
Core module for the PrimeSpec framework. Implements:
- Seven-class topological classification (`classify`, `CLASS_NAME`)
- 7x7 Cayley multiplication table (`MUL_TABLE`)
- Class exponentiation (`class_pow`)
- SpectralAddress: K-dimensional p-adic valuation vector (`SpectralAddress`, `BASIS`)
- Resonance relation

### hw_verification/exp_hw_exact_spectral.py
PyTorch reference implementation of three Verilog hardware modules:
SpectralAddressExtractor, SpectralAndGate, SpectralMatrixTop pipeline.
Uses 10 basis primes (2..29) with full p-adic valuations.

---

## Notes

1. **Unified interface**: 19 of 21 experiments are pytest tests. Each test generates its result JSON via `if __name__ == "__main__"`.
2. **Exceptions**: `hw.exact` runs as a standalone HW verification (torch-free). `hw.bench` is C++20.
3. **hw_verification**: `exp_hw_exact_spectral.py` (PyTorch) is the reference implementation. `exp_hw_exact_nontorch.py` is the CI verifier.
4. **sympy**: Only `test_exp_04_1` requires sympy. All other tests run with stdlib + spectral_matrix.
5. **scipy**: Several tests use scipy for confidence interval computation but include fallback code when scipy is absent.
6. **Result files**: All 21 result JSON files are present in `experiments/results/`.
