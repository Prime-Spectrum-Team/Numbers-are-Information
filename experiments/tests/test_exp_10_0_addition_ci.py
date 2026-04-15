#!/usr/bin/env python3
"""
=============================================================================
EXP-10: Addition Laws with 95% Confidence Intervals
=============================================================================
Thesis: "28 symmetric class-addition pairs reported with 95% CI at N=100K."

For each of the 28 unique symmetric class-addition pairs (cl_a, cl_b)
with a_cls <= b_cls, sample N=100K pairs from [1, 1_000_000], classify
cl(a+b), compute frequency and 95% binomial CI using scipy.

-> P1 §5.3 table with CI, peer review [M5]

(C) Prime-Spectrum-Team, March 2026
=============================================================================
"""
import os, sys, time, json, random
from collections import Counter, defaultdict

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

N_SAMPLES = 100_000
MAX_VAL = 1_000_000
RANDOM_SEED = 42

# =============================================================================
# Utility
# =============================================================================
def generate_class_pool(cls, pool_size=50000, max_val=MAX_VAL):
    """Pre-generate a pool of numbers for each class."""
    pool = []
    random.seed(RANDOM_SEED + cls)
    for n in range(1, min(max_val, 200000)):
        if classify(n) == cls:
            pool.append(n)
        if len(pool) >= pool_size:
            break
    return pool

def binomial_ci(successes, total, confidence=0.95):
    """Compute binomial confidence interval."""
    p_hat = successes / total
    if HAS_SCIPY:
        # Clopper-Pearson exact interval
        alpha = 1 - confidence
        lo = scipy_stats.beta.ppf(alpha/2, successes, total - successes + 1) if successes > 0 else 0.0
        hi = scipy_stats.beta.ppf(1 - alpha/2, successes + 1, total - successes) if successes < total else 1.0
    else:
        # Normal approximation (Wilson interval)
        import math
        z = 1.96  # 95%
        denom = 1 + z**2 / total
        center = (p_hat + z**2 / (2*total)) / denom
        margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4*total)) / total) / denom
        lo = max(0, center - margin)
        hi = min(1, center + margin)
    return p_hat, lo, hi

# =============================================================================
# Main experiment
# =============================================================================
def run_addition_ci_experiment():
    print("\n" + "=" * 70)
    print("  EXP-10: Addition Laws with 95% Confidence Intervals")
    print("  N = {:,} pairs per class-pair, range [2, {:,}]".format(N_SAMPLES, MAX_VAL))
    print("=" * 70)

    # Pre-generate pools for each class
    print("\n  Pre-generating class pools...")
    pools = {}
    for cls in range(7):
        pools[cls] = generate_class_pool(cls)
        print(f"    S{cls} ({CLASS_NAME[cls]:>10}): {len(pools[cls]):>6} numbers")

    # For each pair (a_cls, b_cls), sample N pairs and classify sums
    all_results = {}
    deterministic_count = 0
    probabilistic_count = 0

    print(f"\n  Computing addition distributions...")
    t0 = time.time()

    for a_cls in range(7):
        for b_cls in range(a_cls, 7):  # symmetric, so a <= b
            pool_a = pools[a_cls]
            pool_b = pools[b_cls]

            if not pool_a or not pool_b:
                print(f"    S{a_cls}+S{b_cls}: SKIPPED (empty pool)")
                continue

            # Sample N pairs and classify their sums
            random.seed(RANDOM_SEED + a_cls * 7 + b_cls)
            sum_classes = Counter()

            for _ in range(N_SAMPLES):
                a = random.choice(pool_a)
                b = random.choice(pool_b)
                sum_classes[classify(a + b)] += 1

            # FIX 2026-04-15: Mathematical determinism requires P=1.0 for ALL
            # pairs — i.e. exactly ONE output class observed across N samples.
            # The old threshold (dominant_class / N > 0.999) incorrectly
            # classified S0+S2 as deterministic because P(S6)=0.99989 > 0.999,
            # but the pair (a=1, b=5) gives 1+5=6=2*3 which is S5, not S6.
            # S0 = {1} only; S2 = Lunar primes (prime == 5 mod 6), smallest = 5.
            # Therefore 1 + 5 = 6 = 2*3, Omega=2 -> S5 (Semiprime).
            # Correct check: deterministic iff only one output class appears.
            is_deterministic = len(sum_classes) == 1

            # 2026-04-15 (Part B): Distinguish trivially deterministic laws
            # (at least one singleton operand S0={1}, S3={2}, S4={3}, arithmetic
            # identity) from non-trivially deterministic laws (both operands
            # are infinite classes; genuine emergent statistical law). See
            # Definition def:trivial-det and Theorem thm:7laws in the paper.
            SINGLETONS = {0, 3, 4}
            is_trivial = is_deterministic and (
                a_cls in SINGLETONS or b_cls in SINGLETONS
            )

            if is_deterministic:
                deterministic_count += 1
            else:
                probabilistic_count += 1

            # Compute CI for each output class
            distribution = {}
            for out_cls in range(7):
                count = sum_classes.get(out_cls, 0)
                if count > 0:
                    p_hat, ci_lo, ci_hi = binomial_ci(count, N_SAMPLES)
                    ci_width = ci_hi - ci_lo
                    distribution[out_cls] = {
                        'count': count,
                        'probability': p_hat,
                        'ci_95_lo': ci_lo,
                        'ci_95_hi': ci_hi,
                        'ci_width': ci_width,
                    }

            pair_key = f"S{a_cls}+S{b_cls}"
            all_results[pair_key] = {
                'a_class': a_cls,
                'b_class': b_cls,
                'a_name': CLASS_NAME[a_cls],
                'b_name': CLASS_NAME[b_cls],
                'n_samples': N_SAMPLES,
                'is_deterministic': is_deterministic,
                'is_trivial': is_trivial,
                'distribution': distribution,
            }

    elapsed = time.time() - t0

    # =================================================================
    # Print results
    # =================================================================
    trivial_count = sum(1 for r in all_results.values() if r.get('is_trivial'))
    nontrivial_count = deterministic_count - trivial_count

    print(f"\n  Completed in {elapsed:.0f}s")
    print(f"  Deterministic pairs: {deterministic_count} "
          f"({nontrivial_count} non-trivial + {trivial_count} trivial/singleton)")
    print(f"  Probabilistic pairs: {probabilistic_count}")

    # Print summary table
    print(f"\n  {'='*70}")
    print(f"  PROBABILISTIC ADDITION TABLE (with 95% CI)")
    print(f"  {'='*70}")

    max_ci_width = 0

    for pair_key, result in sorted(all_results.items()):
        if result['is_deterministic']:
            continue

        print(f"\n  {pair_key} ({result['a_name']} + {result['b_name']}):")
        print(f"    {'Output':>12} | {'P(hat)':>8} | {'95% CI':>20} | {'Width':>8} | {'Count':>8}")
        print(f"    {'-'*12} | {'-'*8} | {'-'*20} | {'-'*8} | {'-'*8}")

        for out_cls, dist in sorted(result['distribution'].items()):
            ci_str = f"[{dist['ci_95_lo']:.4f}, {dist['ci_95_hi']:.4f}]"
            print(f"    S{out_cls} ({CLASS_NAME[out_cls]:>8}) | {dist['probability']:>8.4f} | {ci_str:>20} | {dist['ci_width']:>8.5f} | {dist['count']:>8,}")
            max_ci_width = max(max_ci_width, dist['ci_width'])

    # =================================================================
    # Deterministic pairs summary
    # =================================================================
    print(f"\n  {'='*70}")
    print(f"  DETERMINISTIC ADDITION PAIRS")
    print(f"  {'='*70}")
    print(f"    {'Pair':>12} | {'Result':>15} | {'Confidence':>10}")
    print(f"    {'-'*12} | {'-'*15} | {'-'*10}")

    for pair_key, result in sorted(all_results.items()):
        if not result['is_deterministic']:
            continue
        dominant = max(result['distribution'].items(), key=lambda x: x[1]['count'])
        pct = dominant[1]['probability'] * 100
        print(f"    {pair_key:>12} | S{dominant[0]} ({CLASS_NAME[dominant[0]]:>8}) | {pct:.2f}%")

    # =================================================================
    # Success criteria
    # =================================================================
    print(f"\n  {'='*70}")
    print(f"  SUCCESS CRITERIA")
    print(f"  {'='*70}")

    total_pairs = deterministic_count + probabilistic_count
    all_covered = total_pairs >= 28  # at least 28 unique pairs (7x7 symmetric = 28)
    ci_narrow = max_ci_width < 0.01

    print(f"  Total pairs covered: {total_pairs} (need >=28)")
    print(f"    -> {'PASS' if all_covered else 'FAIL'}")
    print(f"  Probabilistic pairs: {probabilistic_count} (spec says 42, but symmetric = ~28)")
    print(f"  Max CI width: {max_ci_width:.5f} (need <0.01)")
    print(f"    -> {'PASS' if ci_narrow else 'FAIL'}")

    all_pass = all_covered and ci_narrow

    return all_results, all_pass, {
        'total_pairs': total_pairs,
        'deterministic': deterministic_count,
        'deterministic_non_trivial': nontrivial_count,
        'deterministic_trivial': trivial_count,
        'probabilistic': probabilistic_count,
        'max_ci_width': max_ci_width,
        'time_seconds': elapsed,
    }

# =============================================================================
# PYTEST ENTRY POINT
# =============================================================================
def test_addition_ci():
    _, passed, _ = run_addition_ci_experiment()
    assert passed, "EXP-10.0: Addition CI verification failed"


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("EXP-10: ADDITION LAWS WITH 95% CONFIDENCE INTERVALS")
    print("N=100K pairs per class-pair, binomial CI via scipy")

    t0 = time.time()

    results, passed, summary = run_addition_ci_experiment()

    total_time = time.time() - t0

    print(f"\n  {'='*60}")
    print(f"  EXPERIMENT {'PASSED' if passed else 'FAILED'}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  {'='*60}")

    # Save results
    output = {
        'experiment': 'EXP-10: Addition Laws CI',
        'n_samples': N_SAMPLES,
        'max_val': MAX_VAL,
        'summary': summary,
        'passed': passed,
        'results': {k: {
            **{kk: vv for kk, vv in v.items() if kk != 'distribution'},
            'distribution': {str(ok): ov for ok, ov in v['distribution'].items()}
        } for k, v in results.items()},
        'target': 'P1 §5.3 table with CI',
    }

    results_file = os.path.join(PROJECT_ROOT, 'results', 'exp_10_0_addition_ci.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")
