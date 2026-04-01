#!/usr/bin/env python3
"""
=============================================================================
EXP-2.1: Partition Completeness Verification
=============================================================================
Verifies Proposition 3.2: every n in [1, 100_000] belongs to exactly one
PrimeSpec class (S0..S6). The 7-class partition is exhaustive and mutually
exclusive by construction (deterministic classify function).

=============================================================================
"""
import os, sys, time, json, datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'shared'))

from spectral_matrix import classify, CLASS_NAME

# =============================================================================

N = 100_000
VALID_CLASSES = set(range(7))  # 0..6


def run_partition_completeness_experiment():
    t0 = time.time()
    counts = {i: 0 for i in range(7)}
    exceptions = []
    exhaustive = True

    for n in range(1, N + 1):
        try:
            cl = classify(n)
            if cl not in VALID_CLASSES:
                exceptions.append({"n": n, "error": f"Unknown class: {cl}"})
                exhaustive = False
            else:
                counts[cl] += 1
        except Exception as e:
            exceptions.append({"n": n, "error": str(e)})
            exhaustive = False

    total_time = time.time() - t0
    total_assigned = sum(counts.values())
    all_pass = exhaustive and len(exceptions) == 0 and total_assigned == N

    distribution = {}
    for cls, count in counts.items():
        distribution[f"S{cls} ({CLASS_NAME[cls]})"] = {
            "count": count,
            "pct": round(count / N * 100, 3)
        }

    print(f"\n  EXP-2.1: Partition Completeness [1, {N}]")
    print(f"  {'='*50}")
    print(f"  Exhaustive: {exhaustive}")
    print(f"  Exceptions: {len(exceptions)}")
    print(f"  Total assigned: {total_assigned}/{N}")
    for label, d in distribution.items():
        print(f"    {label:>20}: {d['count']:>6} ({d['pct']:.3f}%)")
    print(f"  Time: {total_time:.2f}s")
    print(f"  PASS: {all_pass}")

    summary = {
        "n_range": [1, N],
        "total": N,
        "total_assigned": total_assigned,
        "exhaustive": exhaustive,
        "mutually_exclusive": True,
        "exceptions_count": len(exceptions),
    }

    return distribution, all_pass, summary, total_time, exceptions


# =============================================================================
# PYTEST ENTRY POINT
# =============================================================================
def test_partition_completeness():
    _, passed, _, _, _ = run_partition_completeness_experiment()
    assert passed, "EXP-2.1: Partition completeness verification failed"


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  EXP-2.1: Partition Completeness Verification")
    print("="*60)

    distribution, passed, summary, elapsed, exceptions = \
        run_partition_completeness_experiment()

    output = {
        "experiment": "EXP-2.1: Partition Completeness",
        "description": "Every n in [1, N] belongs to exactly one PrimeSpec class",
        **summary,
        "class_distribution": distribution,
        "exceptions": exceptions[:10],
        "success": passed,
        "time_seconds": round(elapsed, 3),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version
    }

    results_file = os.path.join(PROJECT_ROOT, 'results',
                                'exp_02_1_partition_completeness.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")

    sys.exit(0 if passed else 1)
