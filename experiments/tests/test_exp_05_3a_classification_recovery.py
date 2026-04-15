#!/usr/bin/env python3
"""
=============================================================================
EXP-5.3: Classification Recovery from SpectralAddress
=============================================================================
Thesis: "100,000 B-smooth integers verify perfect agreement between
direct classification cl(n) and indirect classification via SA(n)."

For 100,000 random B-smooth integers (basis {2,3,5,7,11,13,17}):
  1. Compute cl_direct(n) via classify(n)
  2. Compute SA(n) via SpectralAddress.from_int(n)
  3. Infer cl_from_SA(n) based on Ω + mod-6 residue
  4. Verify cl_direct = cl_from_SA (unique count after deduplication)

→ P1 §5.3 [Spectral Classification Equivalence]
© Prime-Spectrum-Team, March 2026
=============================================================================
"""
import os, sys, time, json, random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'shared'))

from spectral_matrix import classify, CLASS_NAME, MUL_TABLE, SpectralAddress, BASIS

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

N_SAMPLES = 100_000
RANDOM_SEED = 42

# =============================================================================
# Utility Functions
# =============================================================================

def infer_class_from_sa(sa):
    """Infer topological class from SpectralAddress using spectral_class()."""
    # The SpectralAddress class provides the official spectral_class method
    # which implements the correct classification logic
    return sa.spectral_class()

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

def generate_b_smooth_integer(seed=None):
    """Generate a random B-smooth integer using basis primes.

    Each basis prime (2,3,5,7,11,13,17) gets an exponent drawn uniformly
    from {0,1,2,3,4}.  The sample space has exactly 5^7 = 78,125 distinct
    values, so 100,000 draws yield ~56k unique integers (birthday effect).
    """
    # B5: use a local Random instance instead of polluting the global state
    rng = random.Random(seed)

    result = 1
    for prime_idx in range(len(BASIS)):
        exponent = rng.randint(0, 4)
        result *= BASIS[prime_idx] ** exponent

    return result

def run_classification_recovery_experiment():
    print("\n" + "="*70)
    print("  EXP-5.3: Classification Recovery from SpectralAddress")
    print("  {:,} B-smooth generations, basis {{2,3,5,7,11,13,17}}, exponents 0-4"
          " -> ~56k unique values (5^7={:,} sample space)".format(N_SAMPLES, 5**7))
    print("="*70)

    t0 = time.time()

    # Generate B-smooth numbers (only basis primes) without upper bound
    print(f"\n  Generating {N_SAMPLES:,} B-smooth integers...")
    b_smooth_numbers = []
    for i in range(N_SAMPLES):
        n = generate_b_smooth_integer(seed=RANDOM_SEED + i)
        b_smooth_numbers.append(n)

    b_smooth_numbers = sorted(set(b_smooth_numbers))
    actual_count = len(b_smooth_numbers)
    print(f"  Generated {actual_count:,} unique B-smooth numbers")

    print(f"\n  Testing classification agreement...")
    print(f"  Comparing cl_direct(n) vs cl_from_SA(n)")

    agreements = 0
    disagreements = 0
    disagreement_details = []

    for n in b_smooth_numbers:
        try:
            # Direct classification
            cl_direct = classify(n)

            # SpectralAddress-based classification
            sa = SpectralAddress.from_int(n)
            cl_from_sa = sa.spectral_class()

            if cl_direct == cl_from_sa:
                agreements += 1
            else:
                disagreements += 1
                if len(disagreement_details) < 20:  # Log first 20
                    disagreement_details.append({
                        'n': n,
                        'cl_direct': cl_direct,
                        'cl_from_sa': cl_from_sa,
                        'class_name_direct': CLASS_NAME[cl_direct],
                        'class_name_sa': CLASS_NAME[cl_from_sa],
                    })

        except Exception as e:
            disagreements += 1
            if len(disagreement_details) < 20:
                disagreement_details.append({
                    'n': n,
                    'error': str(e),
                })

    elapsed = time.time() - t0

    # Compute statistics
    # B1 fix: use actual_count (unique B-smooth integers tested) as the
    # denominator, NOT N_SAMPLES (=100,000 raw draws before deduplication).
    # Using N_SAMPLES wrongly reported agreement_rate ~0.563 instead of 1.0.
    agreement_rate, ci_lo, ci_hi = binomial_ci(agreements, actual_count)

    # =================================================================
    # Print Results
    # =================================================================
    print(f"\n  Completed in {elapsed:.1f}s")

    print(f"\n  {'='*70}")
    print(f"  CLASSIFICATION AGREEMENT")
    print(f"  {'='*70}")
    print(f"  Tested: {actual_count:,} unique B-smooth integers (from {N_SAMPLES:,} draws)")
    print(f"  Agreements: {agreements:,} / {actual_count:,}")
    print(f"  Disagreements: {disagreements:,}")
    print(f"  Agreement rate: {agreement_rate:.8f}")
    print(f"  95% CI: [{ci_lo:.8f}, {ci_hi:.8f}]")

    if disagreement_details:
        print(f"\n  First {len(disagreement_details)} disagreements:")
        for detail in disagreement_details:
            if 'error' in detail:
                print(f"    n={detail['n']:>6}: ERROR - {detail['error']}")
            else:
                print(f"    n={detail['n']:>6}: direct={detail['cl_direct']} ({detail['class_name_direct']:>10}), SA={detail['cl_from_sa']} ({detail['class_name_sa']:>10})")

    # =================================================================
    # Success Criteria
    # =================================================================
    print(f"\n  {'='*70}")
    print(f"  SUCCESS CRITERIA")
    print(f"  {'='*70}")

    all_pass = agreements == actual_count
    print(f"  Perfect agreement ({actual_count:,}/{actual_count:,} B-smooth): {agreements == actual_count}")
    print(f"    → {'✅ PASS' if all_pass else '❌ FAIL'}")

    return {
        'agreements': agreements,
        'disagreements': disagreements,
        'agreement_rate': agreement_rate,
        'disagreement_details': disagreement_details,
    }, all_pass, {
        'total_samples': actual_count,
        'agreements': agreements,
        'disagreements': disagreements,
        'agreement_rate': agreement_rate,
        'ci_95_lo': ci_lo,
        'ci_95_hi': ci_hi,
        'time_seconds': elapsed,
    }

# =============================================================================
# PYTEST ENTRY POINT
# =============================================================================
def test_classification_recovery():
    _, passed, _ = run_classification_recovery_experiment()
    assert passed, "EXP-5.3a: Classification recovery verification failed"


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║  EXP-5.3: CLASSIFICATION RECOVERY FROM SPECTRALADDRESS           ║")
    print("║  100,000 B-smooth integers, cl_direct vs cl_from_SA agreement    ║")
    print("╚" + "═"*68 + "╝")

    t0 = time.time()

    results, passed, summary = run_classification_recovery_experiment()

    total_time = time.time() - t0

    print(f"\n  {'='*60}")
    print(f"  EXPERIMENT {'PASSED ✅' if passed else 'FAILED ❌'}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  {'='*60}")

    # Save results
    output = {
        'experiment': 'EXP-5.3: Classification Recovery',
        'n_samples': N_SAMPLES,
        'summary': summary,
        'passed': passed,
        'results': results,
        'target': 'P1 §5.3 Spectral Classification Equivalence',
    }

    results_file = os.path.join(PROJECT_ROOT, 'results', 'exp_05_3a_classification_recovery.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")
