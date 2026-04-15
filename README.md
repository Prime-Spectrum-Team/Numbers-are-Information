# Publication Package -- 2026-04-10

## Title
**Arithmetic Classification and Multiplicative Monoid Structure of Natural Numbers:
A Seven-Class Partition, Prime Coordinate Representation, and Deterministic Addition Laws**

**Authors:** Prime-Spectrum-Team
**Version:** 1.2 (April 16, 2026)
**Venue:** TechRxiv (IEEE preprint)
**Archive (Zenodo, Concept DOI — always latest version):** [10.5281/zenodo.19511136](https://doi.org/10.5281/zenodo.19511136)

---

## Abstract

Natural numbers are conventionally treated as ordered magnitudes.
This paper develops a complementary framework: every natural number n
encodes intrinsic structural information through its prime factorization --
independent of magnitude and computable in deterministic, finite time.

Five contributions (C1--C5): three systematize classical results into a
unified framework, two are novel.

- **C1** Seven-class partition of N with a finite commutative monoid (S, x)
  and complete 49-entry Cayley table
- **C2** Power Dynamics: universality theorem -- every n > 1 reaches the
  absorbing class S6 in at most 3 steps
- **C3** Seven deterministic addition laws, one non-trivial:
  Solar + Lunar -> S6 (verified on 90,000 pairs)
- **C4** SpectralAddress: K-dimensional p-adic valuation vector with four
  O(K) isomorphisms (multiplication, GCD, LCM, coprimality)
- **C5** Resonance relation R(i,j) = Omega(gcd(i,j)) with approximately 61%
  non-resonant pairs (6/pi^2)

All results are verified in Python (100,000 pairs), C++20 (>=475 M Cayley lookups/s),
and a Verilog RTL module (14/14 test vectors, behavioral emulation).

---

## Archive Contents

```
paper/
  20260401_PrimeSpectrum.tex       <- Main document (LaTeX, self-contained)
  20260401_PrimeSpectrum.pdf       <- Compiled PDF
  20260401_PrimeSpectrum.pdf.ots   <- OpenTimestamps proof

demo/
  numbers-are-information.html     <- Interactive single-file demo (open in browser)

experiments/
  EXPERIMENT_INDEX.md              <- Complete experiment index
  conftest.py                      <- pytest configuration
  shared/spectral_matrix.py        <- Core library (classify, MUL_TABLE, SpectralAddress)
  scripts/exp_hw_bench_cpp20.cpp   <- C++20 benchmark
  results/                         <- All result JSONs (21 files)
  tests/                           <- pytest-compatible test files (19 tests)
  hw_verification/                 <- HW emulation (PyTorch + torch-free)

README.md                          <- This document
```

---

## Interactive Demo

Open `demo/numbers-are-information.html` in any browser — no server, no dependencies.
Classifies any natural number, shows its SpectralAddress, Cayley table, and resonance structure.

---

## Paper Structure

| Sec | Title                                               |
|-----|-----------------------------------------------------|
| 1   | Introduction                                        |
| 2   | Preliminaries (Omega, prime residues mod 6, p-adic) |
| 3   | The Seven-Class Partition and Multiplicative Monoid  |
| 4   | Addition Laws                                       |
| 5   | SpectralAddress: Numbers as Coordinates             |
| 6   | Resonance Relation and Sparsity                     |
| 7   | Computational Verification (Python, C++, Verilog)   |
| 8   | Computational Framework                             |
| 9   | Related Work and Open Problems                      |
| 10  | Conclusion                                          |
| App | Python Reference Classifier                         |

---

## Experiment Status

### pytest Tests (19 experiments)

| ID    | Experiment                        | Test file                                 | Status | Paper Ref |
|-------|-----------------------------------|-------------------------------------------|--------|-----------|
| 00.0  | SpectralMemory benchmark proof    | test_exp_00_0_benchmark_proof.py          | PASS   | (P2)      |
| 01.0  | Maximality of 7-class partition (illustrative) | test_exp_01_0_maximality.py    | PASS   | Sec 3     |
| 01.0b | Partition-lattice enumeration (Bell(7)=877) | test_exp_01_0b_partition_lattice.py | PASS | Sec 3     |
| 01.1  | Omega-function complete additivity| test_exp_01_1_omega_additivity.py         | PASS   | Sec 2     |
| 01.2  | Prime residue classes mod 6       | test_exp_01_2_prime_residues.py           | PASS   | Sec 2     |
| 02.1  | Partition completeness            | test_exp_02_1_partition_completeness.py   | PASS   | Sec 3 C1  |
| 02.3a | Coprime sparsity density 6/pi^2  | test_exp_02_3a_sparsity.py                | PASS   | Sec 6 C5  |
| 02.3b | Semigroup identity and absorption | test_exp_02_3b_semigroup_elements.py      | PASS   | Sec 3 C1  |
| 03.1  | Power dynamics and universality   | test_exp_03_1_power_dynamics.py           | PASS   | Sec 3 C2  |
| 04.1  | Solar + Lunar deterministic addition | test_exp_04_1_solar_lunar_addition.py  | PASS   | Sec 4 C3  |
| 05.1  | SpectralAddress bijection         | test_exp_05_1_spectral_bijection.py       | PASS   | Sec 5 C4  |
| 05.2  | Spectral algebraic isomorphisms   | test_exp_05_2_spectral_isomorphisms.py    | PASS   | Sec 5 C4  |
| 05.3a | Classification recovery from SA   | test_exp_05_3a_classification_recovery.py | PASS   | Sec 5     |
| 05.3b | B-smooth boundary limitation      | test_exp_05_3b_b_smooth_boundary.py       | PASS   | Sec 5 C4  |
| 06.2  | Multi-scale resonance hierarchy   | test_exp_06_2_resonance_hierarchy.py      | PASS   | Sec 6 C5  |
| 07.1  | Cayley multiplication table       | test_exp_07_1_cayley_table.py             | PASS   | Sec 3 C1  |
| 10.0  | Addition laws with 95% CI         | test_exp_10_0_addition_ci.py              | PASS   | Sec 4 C3  |
| S0.S2 | S0+S2 non-determinism proof       | test_exp_S0_S2_counterexample.py          | PASS   | Sec 4 C3  |
| hw    | Verilog behavioral emulation      | test_exp_hw_verilog_behavioral.py         | PASS   | Sec 7     |

### Non-pytest Experiments

| ID      | Experiment                      | File                                     | Status | Paper Ref |
|---------|---------------------------------|------------------------------------------|--------|-----------|
| hw.exact| HW exact spectral emulation     | hw_verification/exp_hw_exact_nontorch.py | PASS   | Sec 7     |
| hw.bench| C++20 performance benchmark     | scripts/exp_hw_bench_cpp20.cpp           | PASS   | Sec 7     |

All result JSONs are stored in `experiments/results/`.

---

## Reproduction

```bash
# Run all pytest tests
python -m pytest experiments/tests/ -v

# Run individual experiments
python experiments/tests/test_exp_01_0_maximality.py
python experiments/hw_verification/exp_hw_exact_nontorch.py

# C++20 benchmark
g++ -O3 -march=native -std=c++20 experiments/scripts/exp_hw_bench_cpp20.cpp -o bench && ./bench

# Compile LaTeX
cd paper && pdflatex 20260401_PrimeSpectrum.tex
```

---

## Dependencies

| Package | Version  | Used by                                      |
|---------|----------|----------------------------------------------|
| pytest  | >=7.0    | Test runner (all test_exp_*.py)              |
| torch   | >=2.0    | test_exp_00_0, hw_verification (optional)    |
| sympy   | >=1.12   | test_exp_04_1 (Solar+Lunar)                  |
| scipy   | optional | CI computation in several tests (fallback if absent) |

---

## Third-Party Datasets

Experiment E0.0 (SpectralMemory benchmark) downloads and uses:

- **Tiny Shakespeare** — concatenated works of William Shakespeare (public domain
  text) packaged and distributed by Andrej Karpathy under the MIT License as part
  of the [char-rnn](https://github.com/karpathy/char-rnn) repository.
  The file is fetched at benchmark runtime from
  `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
  and is **not redistributed** as part of this repository. See [`NOTICE`](NOTICE)
  for the full attribution and MIT permission notice.

---

## Known Limitations

1. **Verilog RTL synthesis** pending (behavioral emulation confirmed, timing simulation not yet performed)
2. **exp_hw_exact_spectral.py** requires PyTorch >= 2.0; for CI use `exp_hw_exact_nontorch.py` (torch-free)
3. **ML applications** (SpectralGPT T=512 crossover) are a weak signal -- reserved for Paper 2

---

*Package prepared by Prime-Spectrum-Team, 2026-04-10*
