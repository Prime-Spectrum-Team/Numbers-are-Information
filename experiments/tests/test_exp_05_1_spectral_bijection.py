#!/usr/bin/env python3
"""
=============================================================================
EXP-5.1: SpectralAddress Bijection (Lossless Encoding)
=============================================================================
Thesis: "10,000 B-smooth integers verify perfect round-trip bijection
SA(n) → n via SpectralAddress.to_int() reconstruction."

For B-smooth integers (composed only of {2,3,5,7,11,13,17}):
  1. Generate 10,000 random B-smooth integers
  2. Compute SA(n) via SpectralAddress.from_int(n)
  3. Reconstruct n' = ∏ᵢ pᵢ^SA(n).v[i]
  4. Verify n' = n for all 10,000 samples

→ P1 §5.1 [Spectral Representation]
© Prime-Spectrum-Team, March 2026
=============================================================================
"""
import os, sys, time, json, random, math

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

N_SAMPLES = 10_000
RANDOM_SEED = 42

# =============================================================================
# Utility Functions
# =============================================================================

def generate_b_smooth_integer(seed=None):
    """Generate a random B-smooth integer using basis primes."""
    if seed is not None:
        random.seed(seed)

    result = 1
    for prime_idx in range(len(BASIS)):
        # Randomly choose exponent (0-5 typically)
        exponent = random.randint(0, 4)
        result *= BASIS[prime_idx] ** exponent

    return result

def binomial_ci(successes, total, confidence=0.95):
    """Compute binomial confidence interval."""
    p_hat = successes / total if total > 0 else 0.0
    if HAS_SCIPY:
        alpha = 1 - confidence
        lo = scipy_stats.beta.ppf(alpha/2, successes, total - successes + 1) if successes > 0 else 0.0
        hi = scipy_stats.beta.ppf(1 - alpha/2, successes + 1, total - successes) if successes < total else 1.0
    else:
        import math as mlib
        z = 1.96
        denom = 1 + z**2 / total if total > 0 else 1
        center = (p_hat + z**2 / (2*total)) / denom if total > 0 else 0
        margin = (z * mlib.sqrt((p_hat * (1 - p_hat) + z**2 / (4*total)) / total) / denom) if total > 0 else 0
        lo = max(0, center - margin)
        hi = min(1, center + margin)
    return p_hat, lo, hi

# =============================================================================
# Main Experiment
# =============================================================================

def run_spectral_bijection_experiment():
    print("\n" + "="*70)
    print("  EXP-5.1: SpectralAddress Bijection")
    print("  N = {:,} B-smooth integers (basis: {})".format(N_SAMPLES, BASIS))
    print("="*70)

    t0 = time.time()

    print(f"\n  Generating {N_SAMPLES:,} B-smooth integers...")
    b_smooth_numbers = []
    random.seed(RANDOM_SEED)

    for i in range(N_SAMPLES):
        n = generate_b_smooth_integer(seed=RANDOM_SEED + i)
        b_smooth_numbers.append(n)

    print(f"  Generated {len(b_smooth_numbers):,} numbers")
    print(f"  Min: {min(b_smooth_numbers):,}")
    print(f"  Max: {max(b_smooth_numbers):,}")
    print(f"  Avg: {sum(b_smooth_numbers) // len(b_smooth_numbers):,}")

    # Test bijection: SA(n) → n
    print(f"\n  Testing bijection: SA(n) → n'...")
    t0_test = time.time()

    passes = 0
    failures = 0
    failure_details = []

    for i, n in enumerate(b_smooth_numbers):
        try:
            # Convert to SpectralAddress
            sa = SpectralAddress.from_int(n)

            # Reconstruct
            n_reconstructed = sa.to_int()

            if n == n_reconstructed:
                passes += 1
            else:
                failures += 1
                if len(failure_details) < 10:  # Log first 10 failures
                    failure_details.append({
                        'index': i,
                        'original': n,
                        'reconstructed': n_reconstructed,
                        'sa': str(sa),
                    })

        except Exception as e:
            failures += 1
            if len(failure_details) < 10:
                failure_details.append({
                    'index': i,
                    'original': n,
                    'error': str(e),
                })

    elapsed_test = time.time() - t0_test

    # Compute statistics
    success_rate, ci_lo, ci_hi = binomial_ci(passes, N_SAMPLES)

    print(f"  Completed in {elapsed_test:.1f}s")

    # =================================================================
    # Print Results
    # =================================================================
    print(f"\n  {'='*70}")
    print(f"  BIJECTION VERIFICATION")
    print(f"  {'='*70}")
    print(f"  Tested: {N_SAMPLES:,} B-smooth integers")
    print(f"  Passes (n = n'): {passes:,} / {N_SAMPLES:,}")
    print(f"  Failures: {failures:,}")
    print(f"  Success rate: {success_rate:.8f}")
    print(f"  95% CI: [{ci_lo:.8f}, {ci_hi:.8f}]")

    if failure_details:
        print(f"\n  First {len(failure_details)} failures:")
        for detail in failure_details:
            if 'error' in detail:
                print(f"    [#{detail['index']}] {detail['original']:>12,} → ERROR: {detail['error']}")
            else:
                print(f"    [#{detail['index']}] {detail['original']:>12,} → {detail['reconstructed']:>12,} (diff: {detail['reconstructed'] - detail['original']})")

    elapsed = time.time() - t0

    # =================================================================
    # Success Criteria
    # =================================================================
    print(f"\n  {'='*70}")
    print(f"  SUCCESS CRITERIA")
    print(f"  {'='*70}")

    all_pass = passes == N_SAMPLES
    print(f"  Perfect bijection (10,000/10,000): {passes == N_SAMPLES}")
    print(f"    → {'✅ PASS' if all_pass else '❌ FAIL'}")

    return {
        'passes': passes,
        'failures': failures,
        'success_rate': success_rate,
        'failure_details': failure_details,
    }, all_pass, {
        'total_samples': N_SAMPLES,
        'passes': passes,
        'failures': failures,
        'success_rate': success_rate,
        'ci_95_lo': ci_lo,
        'ci_95_hi': ci_hi,
        'time_seconds': elapsed,
    }

# =============================================================================
# PYTEST ENTRY POINT
# =============================================================================
def test_spectral_bijection():
    _, passed, _ = run_spectral_bijection_experiment()
    assert passed, "EXP-5.1: SpectralAddress bijection verification failed"


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║  EXP-5.1: SPECTRALADDRESS BIJECTION                              ║")
    print("║  10,000 B-smooth integers, perfect round-trip verification       ║")
    print("╚" + "═"*68 + "╝")

    t0 = time.time()

    results, passed, summary = run_spectral_bijection_experiment()

    total_time = time.time() - t0

    print(f"\n  {'='*60}")
    print(f"  EXPERIMENT {'PASSED ✅' if passed else 'FAILED ❌'}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  {'='*60}")

    # Save results
    output = {
        'experiment': 'EXP-5.1: SpectralAddress Bijection',
        'n_samples': N_SAMPLES,
        'basis': BASIS,
        'summary': summary,
        'passed': passed,
        'results': results,
        'target': 'P1 §5.1 Spectral Representation',
    }

    results_file = os.path.join(PROJECT_ROOT, 'results', 'exp_05_1_spectral_bijection.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")
