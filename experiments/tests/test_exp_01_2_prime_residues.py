#!/usr/bin/env python3
"""
=============================================================================
EXP-1.2: Prime Residue Classes Modulo 6 (Foundation Axiom)
=============================================================================
Thesis: "All 664,579 primes < 10^7 belong to residue classes {1, 5} mod 6."

Generate all primes < 10,000,000 via sieve of Eratosthenes, classify
each prime p > 3 by p mod 6, and verify 100% distribution in {1, 5}
with Dirichlet balance ratio ≈ 1:1 expected.

→ P1 §1.2 [Foundation Axiom: Prime Distribution]
© Prime-Spectrum-Team, March 2026
=============================================================================
"""
import os, sys, time, json

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

SIEVE_LIMIT = 10_000_000
RANDOM_SEED = 42

# =============================================================================
# Utility Functions
# =============================================================================

def sieve_of_eratosthenes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []

    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False

    return [i for i in range(2, limit + 1) if is_prime[i]]

def binomial_ci(successes, total, confidence=0.95):
    """Compute binomial confidence interval."""
    p_hat = successes / total if total > 0 else 0.0
    if HAS_SCIPY:
        alpha = 1 - confidence
        lo = scipy_stats.beta.ppf(alpha/2, successes, total - successes + 1) if successes > 0 else 0.0
        hi = scipy_stats.beta.ppf(1 - alpha/2, successes + 1, total - successes) if successes < total else 1.0
    else:
        import math
        z = 1.96
        denom = 1 + z**2 / total if total > 0 else 1
        center = (p_hat + z**2 / (2*total)) / denom if total > 0 else 0
        margin = (z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4*total)) / total) / denom) if total > 0 else 0
        lo = max(0, center - margin)
        hi = min(1, center + margin)
    return p_hat, lo, hi

# =============================================================================
# Main Experiment
# =============================================================================

def run_prime_residues_experiment():
    print("\n" + "="*70)
    print("  EXP-1.2: Prime Residue Classes Modulo 6")
    print("  Sieve limit: {:,}".format(SIEVE_LIMIT))
    print("="*70)

    t0_sieve = time.time()

    # Generate all primes < 10M
    print(f"\n  Generating all primes < {SIEVE_LIMIT:,}...")
    primes = sieve_of_eratosthenes(SIEVE_LIMIT)
    elapsed_sieve = time.time() - t0_sieve

    print(f"  Generated {len(primes):,} primes in {elapsed_sieve:.1f}s")

    # Classify primes by residue class modulo 6
    # Special handling: 2 and 3 are exceptions, all others must be 1 or 5 mod 6
    residue_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    t0_classify = time.time()

    special_primes = [p for p in primes if p <= 3]
    remaining_primes = [p for p in primes if p > 3]

    for p in remaining_primes:
        residue_counts[p % 6] += 1

    elapsed_classify = time.time() - t0_classify

    # =================================================================
    # Print Results
    # =================================================================
    print(f"\n  Classification completed in {elapsed_classify:.1f}s")

    print(f"\n  {'='*70}")
    print(f"  RESIDUE CLASS DISTRIBUTION (mod 6) - Primes > 3 only")
    print(f"  {'='*70}")
    print(f"  {'Class':>8} | {'Count':>10} | {'Percentage':>12} | {'Expected':>12}")
    print(f"  {'-'*8} | {'-'*10} | {'-'*12} | {'-'*12}")

    for residue in range(6):
        count = residue_counts[residue]
        pct = 100 * count / len(remaining_primes) if len(remaining_primes) > 0 else 0
        expected = "{1, 5}" if residue in [1, 5] else "None"
        print(f"  {residue:>8} | {count:>10,} | {pct:>11.4f}% | {expected:>12}")

    # Verify main criterion: all primes > 3 in {1, 5}
    primes_in_valid_classes = residue_counts[1] + residue_counts[5]
    primes_in_invalid_classes = residue_counts[0] + residue_counts[2] + residue_counts[3] + residue_counts[4]

    print(f"\n  {'='*70}")
    print(f"  VERIFICATION")
    print(f"  {'='*70}")
    print(f"  Special primes (2, 3): {len(special_primes)}")
    print(f"  Primes > 3: {len(remaining_primes):,}")
    print(f"  Primes in {{1, 5}} mod 6: {primes_in_valid_classes:,}")
    print(f"  Primes NOT in {{1, 5}} mod 6: {primes_in_invalid_classes:,}")

    # Dirichlet balance (for primes > 3 only)
    count_1 = len([p for p in remaining_primes if p % 6 == 1])
    count_5 = len([p for p in remaining_primes if p % 6 == 5])
    ratio = count_1 / count_5 if count_5 > 0 else 0
    ratio_diff = abs(ratio - 1.0)

    print(f"\n  Dirichlet Balance (primes > 3):")
    print(f"    Class 1 (mod 6): {count_1:,}")
    print(f"    Class 5 (mod 6): {count_5:,}")
    print(f"    Ratio (1/5): {ratio:.6f}")
    print(f"    Expected: 1.0 ± 0.05")
    print(f"    Deviation: {ratio_diff:.6f}")

    # Confidence intervals
    p_hat_valid, ci_lo_valid, ci_hi_valid = binomial_ci(
        primes_in_valid_classes, len(remaining_primes)
    )

    print(f"\n  {'='*70}")
    print(f"  SUCCESS CRITERIA")
    print(f"  {'='*70}")

    all_valid = primes_in_invalid_classes == 0
    dirichlet_pass = ratio_diff < 0.05
    all_pass = all_valid and dirichlet_pass

    print(f"  100% in {{1, 5}} (primes > 3): {primes_in_invalid_classes == 0}")
    print(f"    → {'✅ PASS' if all_valid else '❌ FAIL'}")
    print(f"  Dirichlet balance |ratio - 1.0| < 0.05: {ratio_diff < 0.05}")
    print(f"    → {'✅ PASS' if dirichlet_pass else '❌ FAIL'}")

    return {
        'residue_distribution': residue_counts,
        'total_primes': len(primes),
        'primes_in_valid_classes': primes_in_valid_classes,
        'primes_in_invalid_classes': primes_in_invalid_classes,
        'dirichlet_ratio': ratio,
    }, all_pass, {
        'total_primes': len(primes),
        'primes_gt_3': len(remaining_primes),
        'class_1_count': count_1,
        'class_5_count': count_5,
        'dirichlet_ratio': ratio,
        'sieve_time': elapsed_sieve,
        'classify_time': elapsed_classify,
    }

# =============================================================================
# PYTEST ENTRY POINT
# =============================================================================
def test_prime_residues():
    _, passed, _ = run_prime_residues_experiment()
    assert passed, "EXP-1.2: Prime residue classes verification failed"


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║  EXP-1.2: PRIME RESIDUE CLASSES MODULO 6                          ║")
    print("║  All primes < 10^7 in {1, 5} mod 6, Dirichlet balance             ║")
    print("╚" + "═"*68 + "╝")

    t0 = time.time()

    results, passed, summary = run_prime_residues_experiment()

    total_time = time.time() - t0

    print(f"\n  {'='*60}")
    print(f"  EXPERIMENT {'PASSED ✅' if passed else 'FAILED ❌'}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  {'='*60}")

    # Save results
    output = {
        'experiment': 'EXP-1.2: Prime Residue Classes Mod 6',
        'sieve_limit': SIEVE_LIMIT,
        'summary': summary,
        'passed': passed,
        'results': results,
        'target': 'P1 §1.2 Foundation Axiom',
    }

    results_file = os.path.join(PROJECT_ROOT, 'results', 'exp_01_2_prime_residues.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")
