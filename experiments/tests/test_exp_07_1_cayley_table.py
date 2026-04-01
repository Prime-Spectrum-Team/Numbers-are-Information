#!/usr/bin/env python3
"""
=============================================================================
EXP-7.1 (Bonus): Pure Cayley Table Unit Test
=============================================================================
Thesis: "9,800 class multiplication pairs verify 7x7 Cayley table accuracy
with zero deviations across all cells."

Complement to test_cayley_exhaustive.py (which uses transformer attention).
This test directly validates MUL_TABLE via empirical multiplication:

  For each cell (Cᵢ, Cⱼ) in 7×7 table:
    1. Sample 200 representative pairs (nᵢ ∈ Cᵢ, nⱼ ∈ Cⱼ)
    2. Compute nᵢ · nⱼ
    3. Classify result → Cₖ
    4. Verify Cₖ = MUL_TABLE[i][j]

  Generate 7×7 deviation heatmap.

→ P1 §2.2 [Semigroup Multiplication]
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

N_REPS_PER_CELL = 200
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

def run_cayley_table_experiment():
    print("\n" + "="*70)
    print("  EXP-7.1 (Bonus): Pure Cayley Table Unit Test")
    print("  {:,} pairs: 200 reps × 7×7 cells = 9,800 pairs".format(N_REPS_PER_CELL * 49))
    print("="*70)

    t0 = time.time()

    # Pre-generate class pools
    print(f"\n  Pre-generating class pools...")
    pools = {}
    for cls in range(7):
        pools[cls] = generate_class_pool(cls)
        print(f"    S{cls} ({CLASS_NAME[cls]:>10}): {len(pools[cls]):>6} numbers")

    # Build deviation heatmap
    deviation_heatmap = [[0 for _ in range(7)] for _ in range(7)]
    total_passes = 0
    total_tests = 0

    print(f"\n  Testing all 49 cells...")

    for i in range(7):
        for j in range(7):
            pool_i = pools[i]
            pool_j = pools[j]

            if not pool_i or not pool_j:
                print(f"    S{i} × S{j}: SKIPPED (empty pool)")
                continue

            expected_class = MUL_TABLE[i][j]

            # Sample N_REPS_PER_CELL pairs
            random.seed(RANDOM_SEED + i * 7 + j)
            matches = 0

            for rep in range(N_REPS_PER_CELL):
                n_i = random.choice(pool_i)
                n_j = random.choice(pool_j)

                product = n_i * n_j
                actual_class = classify(product)

                if actual_class == expected_class:
                    matches += 1

                total_tests += 1

            total_passes += matches
            deviation_heatmap[i][j] = N_REPS_PER_CELL - matches

    elapsed = time.time() - t0

    # =================================================================
    # Print Results
    # =================================================================
    print(f"\n  Completed in {elapsed:.1f}s")

    print(f"\n  {'='*70}")
    print(f"  CAYLEY TABLE VERIFICATION")
    print(f"  {'='*70}")
    print(f"  Total tests: {total_tests:,}")
    print(f"  Total passes: {total_passes:,}")
    print(f"  Total deviations: {total_tests - total_passes:,}")

    # Print Cayley table with results
    print(f"\n  Cayley Table (MUL_TABLE[i][j] = expected class):")
    print(f"  {'':>3} | " + " | ".join(f"S{i}" for i in range(7)))
    print(f"  {'-'*3} | " + " | ".join(["-"] * 7))

    for i in range(7):
        row_str = f"S{i} | "
        for j in range(7):
            row_str += f"{MUL_TABLE[i][j]} | "
        print(row_str)

    # Print deviation heatmap
    print(f"\n  Deviation Heatmap (number of failures per cell):")
    print(f"  {'':>3} | " + " | ".join(f"S{i}" for i in range(7)))
    print(f"  {'-'*3} | " + " | ".join(["-"] * 7))

    for i in range(7):
        row_str = f"S{i} | "
        for j in range(7):
            deviations = deviation_heatmap[i][j]
            if deviations == 0:
                row_str += "✅ | "
            else:
                row_str += f"{deviations:>2} | "
        print(row_str)

    # =================================================================
    # Success Criteria
    # =================================================================
    print(f"\n  {'='*70}")
    print(f"  SUCCESS CRITERIA")
    print(f"  {'='*70}")

    all_pass = total_passes == total_tests
    max_deviation = max(max(row) for row in deviation_heatmap)

    print(f"  Zero deviations in all cells: {total_passes == total_tests}")
    print(f"    → {'✅ PASS' if all_pass else '❌ FAIL'}")
    print(f"  Max deviations in any cell: {max_deviation}")
    print(f"    → {'✅ PASS' if max_deviation == 0 else '❌ FAIL'}")

    return {
        'total_tests': total_tests,
        'total_passes': total_passes,
        'deviation_heatmap': deviation_heatmap,
    }, all_pass, {
        'total_tests': total_tests,
        'total_passes': total_passes,
        'total_deviations': total_tests - total_passes,
        'max_deviation': max_deviation,
        'time_seconds': elapsed,
    }

# =============================================================================
# PYTEST ENTRY POINT
# =============================================================================
def test_cayley_table():
    _, passed, _ = run_cayley_table_experiment()
    assert passed, "EXP-7.1: Cayley table verification failed"


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("╔" + "═"*68 + "╗")
    print("║  EXP-7.1 (Bonus): PURE CAYLEY TABLE UNIT TEST                   ║")
    print("║  9,800 pairs from 7×7 cells, 200 reps per cell                   ║")
    print("╚" + "═"*68 + "╝")

    t0 = time.time()

    results, passed, summary = run_cayley_table_experiment()

    total_time = time.time() - t0

    print(f"\n  {'='*60}")
    print(f"  EXPERIMENT {'PASSED ✅' if passed else 'FAILED ❌'}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  {'='*60}")

    # Save results
    output = {
        'experiment': 'EXP-7.1 (Bonus): Cayley Table',
        'n_reps_per_cell': N_REPS_PER_CELL,
        'summary': summary,
        'passed': passed,
        'results': results,
        'target': 'P1 §2.2 Semigroup Multiplication (Bonus)',
    }

    results_file = os.path.join(PROJECT_ROOT, 'results', 'exp_07_1_cayley_table.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")
