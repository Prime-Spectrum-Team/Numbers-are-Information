#!/usr/bin/env python3
"""
=============================================================================
EXP-5.2: Spectral Algebraic Isomorphisms (4 Properties)
=============================================================================
Thesis: "40,000 tests verify 4 fundamental algebraic isomorphisms:
(1) Multiplication: SA(a·b) = SA(a) + SA(b)
(2) GCD: SA(gcd(a,b)) = min(SA(a), SA(b))
(3) LCM: SA(lcm(a,b)) = max(SA(a), SA(b))
(4) Coprimality: gcd(a,b)=1 ⟺ SA(a)·SA(b)=0 (inner product)"

Sample 10,000 B-smooth pairs for each of properties 1-3, and
10,000 total for property 4 (5,000 coprime + 5,000 non-coprime).

→ P1 §5.2 [Spectral Structure Preservation]
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

N_TESTS_PER_PROPERTY = 10_000
N_COPRIME_TESTS = 5_000
N_NON_COPRIME_TESTS = 5_000
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
        exponent = random.randint(0, 4)
        result *= BASIS[prime_idx] ** exponent

    return result

def gcd(a, b):
    """Compute GCD of two integers."""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Compute LCM of two integers."""
    return (a * b) // gcd(a, b)

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

def run_spectral_isomorphisms_experiment():
    print("\n" + "="*70)
    print("  EXP-5.2: Spectral Algebraic Isomorphisms")
    print("  35,000 tests: 4 properties × 10K pairs (coprimality: 5K+5K)")
    print("="*70)

    t0 = time.time()

    results = {
        'multiplication': {'passes': 0, 'total': N_TESTS_PER_PROPERTY},
        'gcd': {'passes': 0, 'total': N_TESTS_PER_PROPERTY},
        'lcm': {'passes': 0, 'total': N_TESTS_PER_PROPERTY},
        'coprimality': {'passes': 0, 'total': N_COPRIME_TESTS + N_NON_COPRIME_TESTS},
    }

    # =================================================================
    # Property 1: Multiplication SA(a·b) = SA(a) + SA(b)
    # =================================================================
    print(f"\n  Property 1: Multiplication (10,000 pairs)")
    print(f"  {'-'*70}")

    random.seed(RANDOM_SEED + 1000)
    for test_num in range(N_TESTS_PER_PROPERTY):
        a = generate_b_smooth_integer(seed=RANDOM_SEED + 1000 + test_num * 2)
        b = generate_b_smooth_integer(seed=RANDOM_SEED + 1001 + test_num * 2)

        try:
            sa_a = SpectralAddress.from_int(a)
            sa_b = SpectralAddress.from_int(b)
            sa_product_direct = SpectralAddress.from_int(a * b)

            # Compute SA(a·b) via isomorphism
            sa_product_computed = sa_a * sa_b

            if sa_product_direct.v == sa_product_computed.v:
                results['multiplication']['passes'] += 1
        except:
            pass

    mult_rate = results['multiplication']['passes'] / N_TESTS_PER_PROPERTY
    print(f"    Passes: {results['multiplication']['passes']:,} / {N_TESTS_PER_PROPERTY:,}")
    print(f"    Success rate: {mult_rate:.6f}")

    # =================================================================
    # Property 2: GCD SA(gcd(a,b)) = min(SA(a), SA(b))
    # =================================================================
    print(f"\n  Property 2: GCD (10,000 pairs)")
    print(f"  {'-'*70}")

    for test_num in range(N_TESTS_PER_PROPERTY):
        a = generate_b_smooth_integer(seed=RANDOM_SEED + 2000 + test_num * 2)
        b = generate_b_smooth_integer(seed=RANDOM_SEED + 2001 + test_num * 2)

        try:
            g = gcd(a, b)
            sa_a = SpectralAddress.from_int(a)
            sa_b = SpectralAddress.from_int(b)
            sa_g_direct = SpectralAddress.from_int(g)

            # Compute SA(gcd(a,b)) via isomorphism
            sa_g_computed = sa_a.gcd(sa_b)

            if sa_g_direct.v == sa_g_computed.v:
                results['gcd']['passes'] += 1
        except:
            pass

    gcd_rate = results['gcd']['passes'] / N_TESTS_PER_PROPERTY
    print(f"    Passes: {results['gcd']['passes']:,} / {N_TESTS_PER_PROPERTY:,}")
    print(f"    Success rate: {gcd_rate:.6f}")

    # =================================================================
    # Property 3: LCM SA(lcm(a,b)) = max(SA(a), SA(b))
    # =================================================================
    print(f"\n  Property 3: LCM (10,000 pairs)")
    print(f"  {'-'*70}")

    for test_num in range(N_TESTS_PER_PROPERTY):
        a = generate_b_smooth_integer(seed=RANDOM_SEED + 3000 + test_num * 2)
        b = generate_b_smooth_integer(seed=RANDOM_SEED + 3001 + test_num * 2)

        try:
            l = lcm(a, b)
            sa_a = SpectralAddress.from_int(a)
            sa_b = SpectralAddress.from_int(b)
            sa_l_direct = SpectralAddress.from_int(l)

            # Compute SA(lcm(a,b)) via isomorphism
            sa_l_computed = sa_a.lcm(sa_b)

            if sa_l_direct.v == sa_l_computed.v:
                results['lcm']['passes'] += 1
        except:
            pass

    lcm_rate = results['lcm']['passes'] / N_TESTS_PER_PROPERTY
    print(f"    Passes: {results['lcm']['passes']:,} / {N_TESTS_PER_PROPERTY:,}")
    print(f"    Success rate: {lcm_rate:.6f}")

    # =================================================================
    # Property 4: Coprimality gcd(a,b)=1 ⟺ SA(a)·SA(b)=0
    # =================================================================
    print(f"\n  Property 4: Coprimality (5,000 coprime + 5,000 non-coprime)")
    print(f"  {'-'*70}")

    # Generate coprime pairs
    for test_num in range(N_COPRIME_TESTS):
        # Generate two B-smooth numbers from disjoint prime subsets
        a_primes = random.sample(range(len(BASIS)), len(BASIS) // 2)
        b_primes = [i for i in range(len(BASIS)) if i not in a_primes]

        a = 1
        for i in a_primes:
            a *= BASIS[i] ** random.randint(1, 3)

        b = 1
        for i in b_primes:
            b *= BASIS[i] ** random.randint(1, 3)

        if a > 1 and b > 1:
            try:
                sa_a = SpectralAddress.from_int(a)
                sa_b = SpectralAddress.from_int(b)

                # For coprime: inner product should be 0
                inner_product = sum(sa_a.v[i] * sa_b.v[i] for i in range(len(sa_a.v)))

                if gcd(a, b) == 1 and inner_product == 0:
                    results['coprimality']['passes'] += 1
                elif gcd(a, b) > 1 and inner_product > 0:
                    results['coprimality']['passes'] += 1
            except:
                pass

    # Generate non-coprime pairs
    for test_num in range(N_NON_COPRIME_TESTS):
        a = generate_b_smooth_integer(seed=RANDOM_SEED + 4000 + test_num * 2)
        b = generate_b_smooth_integer(seed=RANDOM_SEED + 4001 + test_num * 2)

        # Multiply by common factor to ensure non-coprime
        common = random.choice(BASIS)
        a *= common
        b *= common

        try:
            sa_a = SpectralAddress.from_int(a)
            sa_b = SpectralAddress.from_int(b)

            inner_product = sum(sa_a.v[i] * sa_b.v[i] for i in range(len(sa_a.v)))

            if gcd(a, b) > 1 and inner_product > 0:
                results['coprimality']['passes'] += 1
        except:
            pass

    coprimality_rate = results['coprimality']['passes'] / results['coprimality']['total']
    print(f"    Passes: {results['coprimality']['passes']:,} / {results['coprimality']['total']:,}")
    print(f"    Success rate: {coprimality_rate:.6f}")

    elapsed = time.time() - t0

    # =================================================================
    # Print Summary
    # =================================================================
    print(f"\n  Completed in {elapsed:.1f}s")

    print(f"\n  {'='*70}")
    print(f"  SUMMARY")
    print(f"  {'='*70}")
    total_passes = sum(results[p]['passes'] for p in results)
    total_tests = sum(results[p]['total'] for p in results)

    print(f"  Total tests: {total_tests:,}")
    print(f"  Total passes: {total_passes:,}")
    print(f"  Overall rate: {total_passes / total_tests:.6f}")

    # =================================================================
    # Success Criteria
    # =================================================================
    print(f"\n  {'='*70}")
    print(f"  SUCCESS CRITERIA")
    print(f"  {'='*70}")

    all_pass = all(results[p]['passes'] == results[p]['total'] for p in results)

    for prop in ['multiplication', 'gcd', 'lcm', 'coprimality']:
        rate = results[prop]['passes'] / results[prop]['total']
        passes_count = results[prop]['passes']
        total_count = results[prop]['total']
        print(f"  {prop.capitalize()}: {passes_count:,} / {total_count:,} (100% required)")
        print(f"    → {'✅ PASS' if passes_count == total_count else '❌ FAIL'}")

    return results, all_pass, {
        'multiplication_rate': mult_rate,
        'gcd_rate': gcd_rate,
        'lcm_rate': lcm_rate,
        'coprimality_rate': coprimality_rate,
        'total_tests': total_tests,
        'total_passes': total_passes,
        'time_seconds': elapsed,
    }

# =============================================================================
# PYTEST ENTRY POINT
# =============================================================================
def test_spectral_isomorphisms():
    _, passed, _ = run_spectral_isomorphisms_experiment()
    assert passed, "EXP-5.2: Spectral algebraic isomorphisms verification failed"


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║  EXP-5.2: SPECTRAL ALGEBRAIC ISOMORPHISMS                        ║")
    print("║  40,000 tests: multiplication, GCD, LCM, coprimality             ║")
    print("╚" + "═"*68 + "╝")

    t0 = time.time()

    results, passed, summary = run_spectral_isomorphisms_experiment()

    total_time = time.time() - t0

    print(f"\n  {'='*60}")
    print(f"  EXPERIMENT {'PASSED ✅' if passed else 'FAILED ❌'}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  {'='*60}")

    # Save results
    output = {
        'experiment': 'EXP-5.2: Spectral Isomorphisms',
        'summary': summary,
        'passed': passed,
        'results': results,
        'target': 'P1 §5.2 Spectral Structure Preservation',
    }

    results_file = os.path.join(PROJECT_ROOT, 'results', 'exp_05_2_spectral_isomorphisms.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")
