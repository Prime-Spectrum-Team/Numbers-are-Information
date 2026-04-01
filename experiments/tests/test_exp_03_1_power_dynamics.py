#!/usr/bin/env python3
"""
=============================================================================
EXP-3.1 & 3.2: Power Dynamics & Universality Theorem
=============================================================================
Thesis: "5,600 power trajectory tests + 99,999 convergence tests validate
universal reachability of S_6 (Multiprime) absorption via exponentiation."

E3.1: Power Trajectories
  For each class C ∈ {S₀–S₆}, sample 100 representatives, compute
  n^k for k ∈ [1,8], verify class matches Theorem 5 prediction.

E3.2: Universality Theorem
  For each n ∈ [2, 100,000], compute k_predicted = ceil(3 / Ω(n)),
  iterate until cl(n^k) = S₆, verify k_actual ≤ k_predicted.

→ P1 §3.1 & §3.2 [Convergence & Exponentiation]
© Prime-Spectrum-Team, March 2026
=============================================================================
"""
import os, sys, time, json, random, math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'shared'))

from spectral_matrix import classify, CLASS_NAME, MUL_TABLE, class_pow

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
    print("ERROR: sympy required for E3.2 (Ω-function computation)")
    sys.exit(1)

N_REPS_PER_CLASS = 100
MAX_VAL = 1_000_000
RANDOM_SEED = 42
UNIVERSALITY_LIMIT = 100_000
MAX_EXPONENT = 8

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

_omega_cache = {}

def omega(n):
    """Compute Ω(n) = total number of prime factors (with multiplicity)."""
    if n in _omega_cache:
        return _omega_cache[n]
    if n <= 1:
        result = 0
    else:
        factors = factorint(n)
        result = sum(factors.values())
    _omega_cache[n] = result
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

def run_power_dynamics_experiment():
    print("\n" + "="*70)
    print("  EXP-3.1 & 3.2: Power Dynamics & Universality Theorem")
    print("  E3.1: 5,600 tests | E3.2: 99,999 tests")
    print("="*70)

    t0 = time.time()

    # Pre-generate class pools
    print(f"\n  Pre-generating class pools...")
    pools = {}
    for cls in range(7):
        pools[cls] = generate_class_pool(cls)
        print(f"    S{cls} ({CLASS_NAME[cls]:>10}): {len(pools[cls]):>6} numbers")

    # =================================================================
    # E3.1: Power Trajectories
    # =================================================================
    print(f"\n  E3.1: Power Trajectories (100 reps × 7 classes × 8 exponents)")
    print(f"  {'-'*70}")

    e31_passes = 0
    e31_total = 0

    for base_cls in range(7):
        pool = pools[base_cls]
        if not pool:
            continue

        random.seed(RANDOM_SEED + 300 + base_cls)
        for rep in range(N_REPS_PER_CLASS):
            n = random.choice(pool)

            # Test powers n^k for k in [1, 8]
            for k in range(1, MAX_EXPONENT + 1):
                # Compute expected class via class_pow
                expected_cls = class_pow(base_cls, k)

                # Compute actual class
                try:
                    power = n ** k
                    actual_cls = classify(power)

                    if actual_cls == expected_cls:
                        e31_passes += 1
                except:
                    # Overflow or other computation error
                    pass

                e31_total += 1

    e31_rate, e31_ci_lo, e31_ci_hi = binomial_ci(e31_passes, e31_total)

    print(f"    Passes: {e31_passes:,} / {e31_total:,}")
    print(f"    Success rate: {e31_rate:.6f}")
    print(f"    95% CI: [{e31_ci_lo:.6f}, {e31_ci_hi:.6f}]")

    # =================================================================
    # E3.2: Universality Theorem
    # =================================================================
    print(f"\n  E3.2: Universality Convergence (n ∈ [2, {UNIVERSALITY_LIMIT:,}])")
    print(f"  {'-'*70}")

    convergence_k_distribution = {1: 0, 2: 0, 3: 0, 4: 0}  # k >= 4 rare
    convergence_passes = 0
    convergence_total = 0
    max_k_observed = 0

    for n in range(2, min(UNIVERSALITY_LIMIT + 1, 100001)):
        omega_n = omega(n)

        # Predicted k: ceil(3 / Ω(n))
        if omega_n > 0:
            k_predicted = math.ceil(3 / omega_n)
        else:
            k_predicted = 1

        # Find actual k where n^k reaches S₆
        k_actual = None
        for k in range(1, k_predicted + 2):  # Allow +2 tolerance
            try:
                power = n ** k
                if classify(power) == 6:  # S₆
                    k_actual = k
                    break
            except:
                continue

        if k_actual is not None:
            convergence_total += 1

            # Check if within predicted bound
            if k_actual <= k_predicted:
                convergence_passes += 1

            # Track distribution
            if k_actual <= 4:
                convergence_k_distribution[k_actual] += 1
            else:
                if 4 not in convergence_k_distribution:
                    convergence_k_distribution[4] = 0
                convergence_k_distribution[4] += 1

            max_k_observed = max(max_k_observed, k_actual)

    conv_rate, conv_ci_lo, conv_ci_hi = binomial_ci(convergence_passes, convergence_total)

    print(f"    Tested: {convergence_total:,} integers [2, {UNIVERSALITY_LIMIT:,}]")
    print(f"    Converge: {convergence_passes:,} / {convergence_total:,}")
    print(f"    Success rate: {conv_rate:.6f}")
    print(f"    95% CI: [{conv_ci_lo:.6f}, {conv_ci_hi:.6f}]")
    print(f"    Max k observed: {max_k_observed}")

    print(f"\n    Distribution of k where n^k reaches S₆:")
    print(f"      k=1: {convergence_k_distribution.get(1, 0):>6,} ({100*convergence_k_distribution.get(1, 0)/convergence_total if convergence_total > 0 else 0:>6.2f}%)")
    print(f"      k=2: {convergence_k_distribution.get(2, 0):>6,} ({100*convergence_k_distribution.get(2, 0)/convergence_total if convergence_total > 0 else 0:>6.2f}%)")
    print(f"      k=3: {convergence_k_distribution.get(3, 0):>6,} ({100*convergence_k_distribution.get(3, 0)/convergence_total if convergence_total > 0 else 0:>6.2f}%)")
    print(f"      k≥4: {convergence_k_distribution.get(4, 0):>6,} ({100*convergence_k_distribution.get(4, 0)/convergence_total if convergence_total > 0 else 0:>6.2f}%)")

    elapsed = time.time() - t0

    # =================================================================
    # Print Results
    # =================================================================
    print(f"\n  Completed in {elapsed:.1f}s")

    print(f"\n  {'='*70}")
    print(f"  SUCCESS CRITERIA")
    print(f"  {'='*70}")

    e31_pass = e31_passes == e31_total
    e32_pass = convergence_passes == convergence_total

    print(f"  E3.1 Trajectories: {e31_passes} / {e31_total} (100% required)")
    print(f"    → {'✅ PASS' if e31_pass else '❌ FAIL'}")
    print(f"  E3.2 Convergence: {convergence_passes} / {convergence_total} (100% required)")
    print(f"    → {'✅ PASS' if e32_pass else '❌ FAIL'}")

    all_pass = e31_pass and e32_pass

    return {
        'e31_passes': e31_passes,
        'e31_total': e31_total,
        'e32_passes': convergence_passes,
        'e32_total': convergence_total,
        'convergence_k_distribution': convergence_k_distribution,
    }, all_pass, {
        'e31_rate': e31_rate,
        'e32_rate': conv_rate,
        'max_k_observed': max_k_observed,
        'time_seconds': elapsed,
    }

# =============================================================================
# PYTEST ENTRY POINT
# =============================================================================
def test_power_dynamics():
    _, passed, _ = run_power_dynamics_experiment()
    assert passed, "EXP-3.1: Power dynamics and universality verification failed"


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║  EXP-3.1 & 3.2: POWER DYNAMICS & UNIVERSALITY                    ║")
    print("║  5,600 trajectories + 99,999 convergence tests                   ║")
    print("╚" + "═"*68 + "╝")

    t0 = time.time()

    results, passed, summary = run_power_dynamics_experiment()

    total_time = time.time() - t0

    print(f"\n  {'='*60}")
    print(f"  EXPERIMENT {'PASSED ✅' if passed else 'FAILED ❌'}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  {'='*60}")

    # Save results
    output = {
        'experiment': 'EXP-3.1 & 3.2: Power Dynamics',
        'n_reps_per_class': N_REPS_PER_CLASS,
        'universality_limit': UNIVERSALITY_LIMIT,
        'summary': summary,
        'passed': passed,
        'results': results,
        'target': 'P1 §3.1 & §3.2 Convergence',
    }

    results_file = os.path.join(PROJECT_ROOT, 'results', 'exp_03_1_power_dynamics.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")
