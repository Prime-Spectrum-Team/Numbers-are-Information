#!/usr/bin/env python3
"""
=============================================================================
EXP-S0.S2: S0+S2 Non-Determinism Proof
=============================================================================
Shows that S0+S2 is NOT mathematically deterministic.
Counterexample: cl(1+5) = cl(6) = S5, not S6.

EXP-10 used statistical criterion (P=0.99989 ~ 1.0) instead of
mathematical determinism (P=1.0 for ALL pairs without exception).

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

LIMIT = 100_000


def omega(n):
    count, d = 0, 2
    while d * d <= n:
        while n % d == 0:
            count += 1
            n //= d
        d += 1
    return count + (1 if n > 1 else 0)


def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0: return False
    return True


def is_lunar_prime(n):
    return is_prime(n) and (n == 5 or n % 6 == 5)


def run_s0_s2_counterexample_experiment():
    t0 = time.time()

    lunar_primes = [q for q in range(2, LIMIT + 1) if is_lunar_prime(q)]

    giving_S5 = []
    giving_S6 = []

    for q in lunar_primes:
        s = 1 + q
        cl = classify(s)
        if cl == 5:  # S5
            giving_S5.append({"q": q, "sum": s, "omega": omega(s)})
        elif cl == 6:  # S6
            giving_S6.append(q)

    # Core assertion: cl(1+5) = cl(6) = S5, not S6
    cl_6 = classify(6)
    om_6 = omega(6)
    counterexample_valid = (cl_6 == 5)  # S5

    # Analytical proof for q>=11
    analytical_cases = []
    for q in [q for q in lunar_primes if q >= 11][:10]:
        s = 1 + q
        analytical_cases.append({
            "q": q, "sum": s,
            "divisible_by_6": s % 6 == 0,
            "total_omega": omega(s),
            "cl": classify(s)
        })

    elapsed = time.time() - t0
    all_pass = counterexample_valid and len(giving_S5) > 0

    print(f"\n  EXP-S0.S2: S0+S2 Non-Determinism Proof")
    print(f"  {'='*50}")
    print(f"  Counterexample: cl(1+5) = cl(6) = S{cl_6} "
          f"({CLASS_NAME[cl_6]}), Omega={om_6}")
    print(f"  Valid counterexample: {counterexample_valid}")
    print(f"  Lunar primes checked: {len(lunar_primes)}")
    print(f"    Giving S5: {[x['q'] for x in giving_S5]}")
    print(f"    Giving S6: {len(giving_S6)} primes")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  PASS: {all_pass}")

    assert counterexample_valid, \
        f"cl(6) must be S5 (Semiprime), got S{cl_6}"
    assert len(giving_S5) > 0, \
        "Must find at least one S0+S2 pair giving S5"

    summary = {
        "counterexample": {
            "a": 1, "b": 5, "sum": 6,
            "factorization": "2 x 3",
            "omega_sum": om_6,
            "cl_sum": cl_6,
            "cl_name": CLASS_NAME[cl_6],
            "expected_if_deterministic": "S6",
            "is_counterexample": counterexample_valid
        },
        "enumeration": {
            "lunar_primes_checked": len(lunar_primes),
            "limit": LIMIT,
            "giving_S5": giving_S5,
            "giving_S6_count": len(giving_S6),
        },
        "analytical_spot_check": analytical_cases,
    }

    return summary, all_pass, elapsed


# =============================================================================
# PYTEST ENTRY POINT
# =============================================================================
def test_s0_s2_counterexample():
    _, passed, _ = run_s0_s2_counterexample_experiment()
    assert passed, "EXP-S0.S2: S0+S2 counterexample verification failed"


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  EXP-S0.S2: S0+S2 Non-Determinism Counterexample")
    print("="*60)

    summary, passed, elapsed = run_s0_s2_counterexample_experiment()

    output = {
        "experiment": "EXP-S0.S2: S0+S2 Non-Determinism Proof",
        "claim": "S0+S2 is NOT mathematically deterministic",
        **summary,
        "exp10_correction": {
            "original_flag": True,
            "corrected": False,
            "root_cause": (
                "exp10 used P=0.99989 ~ 1.0 as deterministic. "
                "Mathematical determinism requires P=1.0 for ALL pairs. "
                "The exceptions correspond to the pair (1, 5)."
            ),
        },
        "success": passed,
        "time_seconds": round(elapsed, 3),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version
    }

    results_file = os.path.join(PROJECT_ROOT, 'results',
                                'exp_S0_S2_counterexample.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")

    sys.exit(0 if passed else 1)
