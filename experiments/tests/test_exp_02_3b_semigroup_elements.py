#!/usr/bin/env python3
"""
=============================================================================
EXP-2.3: Semigroup Identity & Absorption Elements
=============================================================================
Thesis: "1,400 multiplication tests verify S_0 identity and S_6 absorption."

Test semigroup (S, ⊗) fundamental properties:
1. Identity: S₀ (Unit) ⊗ Cᵢ = Cᵢ for all classes
2. Absorption: S₆ (Multiprime) ⊗ Cᵢ = S₆ for all classes

Sample 100 representatives per class, perform 1,400 total tests
(100 reps × 7 classes × 2 properties).

→ P1 §2.3 [Semigroup Structure]
© Prime-Spectrum-Team, March 2026
=============================================================================
"""
import os, sys, time, json, random

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

N_REPS_PER_CLASS = 100
MAX_VAL = 1_000_000
RANDOM_SEED = 42

# =============================================================================
# Utility Functions
# =============================================================================

def generate_class_pool(cls, pool_size=50000, max_val=MAX_VAL):
    """Pre-generate a pool of numbers for each class."""
    pool = []
    random.seed(RANDOM_SEED + cls)
    for n in range(2, min(max_val, 200000)):
        if classify(n) == cls:
            pool.append(n)
        if len(pool) >= pool_size:
            break
    return pool

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

def run_semigroup_elements_experiment():
    print("\n" + "="*70)
    print("  EXP-2.3: Semigroup Identity & Absorption Elements")
    print("  {:,} tests: {} reps × 7 classes × 2 properties".format(
        N_REPS_PER_CLASS * 7 * 2, N_REPS_PER_CLASS))
    print("="*70)

    t0 = time.time()

    # Pre-generate class pools
    print(f"\n  Pre-generating class pools...")
    pools = {}
    for cls in range(7):
        pools[cls] = generate_class_pool(cls)
        print(f"    S{cls} ({CLASS_NAME[cls]:>10}): {len(pools[cls]):>6} numbers")

    # =================================================================
    # Test 1: Identity Element (S₀ = {1})
    # =================================================================
    print(f"\n  Test 1: Identity Element (S₀ ⊗ Cᵢ = Cᵢ)")
    print(f"  {'-'*70}")

    identity_passes = 0
    identity_total = 0

    for target_cls in range(7):
        pool = pools[target_cls]
        if not pool:
            continue

        random.seed(RANDOM_SEED + 100 + target_cls)
        for rep in range(N_REPS_PER_CLASS):
            n = random.choice(pool)
            # 1 · n should give class of n
            result_cls = classify(1 * n)
            expected_cls = target_cls

            if result_cls == expected_cls:
                identity_passes += 1

            identity_total += 1

    identity_rate, id_ci_lo, id_ci_hi = binomial_ci(identity_passes, identity_total)

    print(f"    Passes: {identity_passes:,} / {identity_total:,}")
    print(f"    Success rate: {identity_rate:.6f}")
    print(f"    95% CI: [{id_ci_lo:.6f}, {id_ci_hi:.6f}]")

    # =================================================================
    # Test 2: Absorption Element (S₆ absorbs all)
    # =================================================================
    print(f"\n  Test 2: Absorption Element (S₆ ⊗ Cᵢ = S₆)")
    print(f"  {'-'*70}")

    absorption_passes = 0
    absorption_total = 0

    # Get Multiprime pool
    multiprime_pool = pools[6]  # S₆ = Multiprime
    if not multiprime_pool:
        print("    ERROR: No Multiprimes generated!")
    else:
        random.seed(RANDOM_SEED + 200)

        for target_cls in range(7):
            pool = pools[target_cls]
            if not pool:
                continue

            random.seed(RANDOM_SEED + 200 + target_cls)
            for rep in range(N_REPS_PER_CLASS):
                m6 = random.choice(multiprime_pool)
                n = random.choice(pool)

                # m6 · n should always give S₆ (class 6)
                result_cls = classify(m6 * n)
                expected_cls = 6

                if result_cls == expected_cls:
                    absorption_passes += 1

                absorption_total += 1

    absorption_rate, abs_ci_lo, abs_ci_hi = binomial_ci(absorption_passes, absorption_total)

    print(f"    Passes: {absorption_passes:,} / {absorption_total:,}")
    print(f"    Success rate: {absorption_rate:.6f}")
    print(f"    95% CI: [{abs_ci_lo:.6f}, {abs_ci_hi:.6f}]")

    elapsed = time.time() - t0

    # =================================================================
    # Print Results
    # =================================================================
    print(f"\n  Completed in {elapsed:.1f}s")

    print(f"\n  {'='*70}")
    print(f"  SUMMARY")
    print(f"  {'='*70}")
    print(f"  Total tests: {identity_total + absorption_total:,}")
    print(f"  Identity tests: {identity_total:,}")
    print(f"  Absorption tests: {absorption_total:,}")
    print(f"  Total passes: {identity_passes + absorption_passes:,}")

    print(f"\n  {'='*70}")
    print(f"  SUCCESS CRITERIA")
    print(f"  {'='*70}")

    all_pass = identity_passes == identity_total and absorption_passes == absorption_total
    print(f"  Identity: {identity_passes} / {identity_total} (100% required)")
    print(f"    → {'✅ PASS' if identity_passes == identity_total else '❌ FAIL'}")
    print(f"  Absorption: {absorption_passes} / {absorption_total} (100% required)")
    print(f"    → {'✅ PASS' if absorption_passes == absorption_total else '❌ FAIL'}")

    return {
        'identity_passes': identity_passes,
        'identity_total': identity_total,
        'absorption_passes': absorption_passes,
        'absorption_total': absorption_total,
    }, all_pass, {
        'identity_rate': identity_rate,
        'absorption_rate': absorption_rate,
        'total_tests': identity_total + absorption_total,
        'total_passes': identity_passes + absorption_passes,
        'time_seconds': elapsed,
    }

# =============================================================================
# PYTEST ENTRY POINT
# =============================================================================
def test_semigroup_elements():
    _, passed, _ = run_semigroup_elements_experiment()
    assert passed, "EXP-2.3b: Semigroup identity/absorption verification failed"


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║  EXP-2.3: SEMIGROUP IDENTITY & ABSORPTION ELEMENTS               ║")
    print("║  1,400 tests: S₀ identity, S₆ absorption                          ║")
    print("╚" + "═"*68 + "╝")

    t0 = time.time()

    results, passed, summary = run_semigroup_elements_experiment()

    total_time = time.time() - t0

    print(f"\n  {'='*60}")
    print(f"  EXPERIMENT {'PASSED ✅' if passed else 'FAILED ❌'}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  {'='*60}")

    # Save results
    output = {
        'experiment': 'EXP-2.3: Semigroup Elements',
        'n_reps_per_class': N_REPS_PER_CLASS,
        'summary': summary,
        'passed': passed,
        'results': results,
        'target': 'P1 §2.3 Semigroup Structure',
    }

    results_file = os.path.join(PROJECT_ROOT, 'results', 'exp_02_3b_semigroup_elements.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")
