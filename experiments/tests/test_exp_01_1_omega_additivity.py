#!/usr/bin/env python3
"""
=============================================================================
EXP-1.1: Ω-Function Complete Additivity (Foundation Axiom)
=============================================================================
Thesis: "100,000 pairs (a,b) confirm Omega(a*b) = Omega(a) + Omega(b) with 95% CI."

For each of 100K random pairs (a,b) from [1, 1,000,000], compute prime
factorizations via sympy, calculate Ω values, verify additivity, and
generate distribution histogram with 95% confidence intervals.

→ P1 §1.1 [Foundation Axiom: Prime Factorization Completeness]
© Prime-Spectrum-Team, March 2026
=============================================================================
"""
import os, sys, time, json, random
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'shared'))

from spectral_matrix import classify, CLASS_NAME, MUL_TABLE

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Using normal approximation for CI.")

try:
    from sympy import factorint
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    print("ERROR: sympy required for E1.1 (Ω-function computation)")
    sys.exit(1)

N_SAMPLES = 100_000
MAX_VAL = 1_000_000
RANDOM_SEED = 42

# =============================================================================
# Utility Functions
# =============================================================================

def omega(n):
    """Compute Ω(n) = total number of prime factors (with multiplicity)."""
    if n <= 1:
        return 0
    factors = factorint(n)
    return sum(factors.values())

def binomial_ci(successes, total, confidence=0.95):
    """Compute binomial confidence interval (Clopper-Pearson or Wilson)."""
    p_hat = successes / total if total > 0 else 0.0
    if HAS_SCIPY:
        # Clopper-Pearson exact interval
        alpha = 1 - confidence
        lo = scipy_stats.beta.ppf(alpha/2, successes, total - successes + 1) if successes > 0 else 0.0
        hi = scipy_stats.beta.ppf(1 - alpha/2, successes + 1, total - successes) if successes < total else 1.0
    else:
        # Normal approximation (Wilson interval)
        import math
        z = 1.96  # 95%
        denom = 1 + z**2 / total if total > 0 else 1
        center = (p_hat + z**2 / (2*total)) / denom if total > 0 else 0
        margin = (z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4*total)) / total) / denom) if total > 0 else 0
        lo = max(0, center - margin)
        hi = min(1, center + margin)
    return p_hat, lo, hi

# =============================================================================
# Main Experiment
# =============================================================================

def run_omega_additivity_experiment():
    print("\n" + "="*70)
    print("  EXP-1.1: Ω-Function Complete Additivity")
    print("  N = {:,} random pairs from [1, {:,}]".format(N_SAMPLES, MAX_VAL))
    print("="*70)

    t0 = time.time()

    # Generate sample pairs and verify additivity
    print(f"\n  Computing Ω(a), Ω(b), Ω(a·b) for {N_SAMPLES:,} pairs...")

    random.seed(RANDOM_SEED)
    omega_distribution = Counter()

    failures = 0
    success_count = 0

    for i in range(N_SAMPLES):
        a = random.randint(2, MAX_VAL)
        b = random.randint(2, MAX_VAL)

        omega_a = omega(a)
        omega_b = omega(b)

        # Compute Ω(a·b) carefully to avoid overflow for very large products
        # Instead, we'll use: Ω(a·b) = Ω(a) + Ω(b) - Ω(gcd(a,b)) + Ω(gcd(a,b))
        # which simplifies to Ω(a) + Ω(b) always
        omega_ab_expected = omega_a + omega_b

        # Verify by computing actual factorization of a*b
        # For large products, we may need to be careful
        try:
            product = a * b
            omega_ab_actual = omega(product)

            if omega_ab_actual == omega_ab_expected:
                success_count += 1
                # Track distribution of Ω values
                omega_distribution[omega_a] += 1
                omega_distribution[omega_b] += 1
            else:
                failures += 1
        except Exception as e:
            failures += 1

    elapsed = time.time() - t0

    # Compute success rate with CI
    success_rate, ci_lo, ci_hi = binomial_ci(success_count, N_SAMPLES)

    # =================================================================
    # Print Results
    # =================================================================
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"\n  {'='*70}")
    print(f"  ADDITIVITY VERIFICATION")
    print(f"  {'='*70}")
    print(f"  Pairs tested: {N_SAMPLES:,}")
    print(f"  Successes: {success_count:,} / {N_SAMPLES:,}")
    print(f"  Success rate: {success_rate:.6f}")
    print(f"  95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"  Failures: {failures:,}")

    # Print Ω distribution histogram
    print(f"\n  {'='*70}")
    print(f"  Ω VALUE DISTRIBUTION")
    print(f"  {'='*70}")
    print(f"  {'Ω Value':>8} | {'Count':>10} | {'Probability':>12} | {'Percent':>8}")
    print(f"  {'-'*8} | {'-'*10} | {'-'*12} | {'-'*8}")

    total_omega_samples = sum(omega_distribution.values())
    for omega_val in sorted(omega_distribution.keys()):
        count = omega_distribution[omega_val]
        prob = count / total_omega_samples if total_omega_samples > 0 else 0
        pct = prob * 100
        print(f"  {omega_val:>8} | {count:>10,} | {prob:>12.6f} | {pct:>7.2f}%")

    # =================================================================
    # Success Criteria
    # =================================================================
    print(f"\n  {'='*70}")
    print(f"  SUCCESS CRITERIA")
    print(f"  {'='*70}")

    all_pass = success_rate >= 0.999
    print(f"  Success rate ≥ 99.9%: {success_rate:.4f}")
    print(f"    → {'✅ PASS' if all_pass else '❌ FAIL'}")

    return {
        'omega_values': dict(omega_distribution),
    }, all_pass, {
        'total_pairs': N_SAMPLES,
        'successes': success_count,
        'failures': failures,
        'success_rate': success_rate,
        'ci_95_lo': ci_lo,
        'ci_95_hi': ci_hi,
        'time_seconds': elapsed,
    }

# =============================================================================
# PYTEST ENTRY POINT
# =============================================================================
def test_omega_additivity():
    _, passed, _ = run_omega_additivity_experiment()
    assert passed, "EXP-1.1: Omega additivity verification failed"


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║  EXP-1.1: Ω-FUNCTION COMPLETE ADDITIVITY                          ║")
    print("║  N=100K pairs, Ω(a·b) = Ω(a) + Ω(b) verification                  ║")
    print("╚" + "═"*68 + "╝")

    t0 = time.time()

    results, passed, summary = run_omega_additivity_experiment()

    total_time = time.time() - t0

    print(f"\n  {'='*60}")
    print(f"  EXPERIMENT {'PASSED ✅' if passed else 'FAILED ❌'}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  {'='*60}")

    # Save results
    output = {
        'experiment': 'EXP-1.1: Ω-Function Additivity',
        'n_samples': N_SAMPLES,
        'max_val': MAX_VAL,
        'summary': summary,
        'passed': passed,
        'results': results,
        'target': 'P1 §1.1 Foundation Axiom',
    }

    results_file = os.path.join(PROJECT_ROOT, 'results', 'exp_01_1_omega_additivity.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")
