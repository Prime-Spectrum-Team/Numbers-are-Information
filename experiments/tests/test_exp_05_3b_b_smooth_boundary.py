#!/usr/bin/env python3
"""
=============================================================================
EXP-5.3b: B-Smooth Boundary Limitation
=============================================================================
Documents that SpectralAddress bijection is lossless ONLY for B-smooth
integers. For K=7 basis (2,3,5,7,11,13,17), the first breakpoint is n=19
(prime > max(basis)=17), where SA(19) = SA(1) = [0]*K.

This is NOT a failure — it defines the domain of validity.

=============================================================================
"""
import os, sys, time, json, datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'shared'))

from spectral_matrix import SpectralAddress, BASIS

# =============================================================================

K = len(BASIS)


def spectral_address(n):
    sa = []
    rem = n
    for p in BASIS:
        v = 0
        while rem % p == 0:
            v += 1
            rem //= p
        sa.append(v)
    return tuple(sa)


def reconstruct(sa):
    n = 1
    for v, p in zip(sa, BASIS):
        n *= p ** v
    return n


def is_b_smooth(n):
    for p in BASIS:
        while n % p == 0:
            n //= p
    return n == 1


def run_b_smooth_boundary_experiment():
    t0 = time.time()

    # Part 1: Bijection break at n=19
    sa_19 = spectral_address(19)
    sa_1 = spectral_address(1)
    recon_19 = reconstruct(sa_19)
    collision = sa_19 == sa_1

    # Part 2: Smoothness fractions
    smoothness_fractions = {}
    for N in [1000, 10000, 100000]:
        count = sum(1 for n in range(1, N + 1) if is_b_smooth(n))
        smoothness_fractions[str(N)] = {
            "smooth_count": count,
            "total": N,
            "fraction": round(count / N, 6)
        }

    # Part 3: Remainder examples
    example_ns = [19, 23, 29, 31, 37, 38, 57, 361, 529]
    remainder_examples = []
    for n in example_ns:
        sa = spectral_address(n)
        recon = reconstruct(sa)
        remainder_examples.append({
            "n": n,
            "sa": list(sa),
            "reconstructed_from_sa": recon,
            "lossless": recon == n
        })

    elapsed = time.time() - t0

    # Assertions
    assert collision, "SA(19) must collide with SA(1)"
    assert recon_19 == 1, f"reconstruct(SA(19)) must be 1, got {recon_19}"
    assert not is_b_smooth(19), "19 must not be B-smooth"
    assert is_b_smooth(17), "17 must be B-smooth (in basis)"

    all_pass = collision and recon_19 == 1

    print(f"\n  EXP-5.3b: B-Smooth Boundary")
    print(f"  {'='*50}")
    print(f"  Bijection break at n=19: collision={collision}")
    print(f"  SA(19)={list(sa_19)}, SA(1)={list(sa_1)}")
    print(f"  reconstruct(SA(19)) = {recon_19}")
    print(f"\n  Smoothness fractions (basis up to {max(BASIS)}):")
    for N_str, d in smoothness_fractions.items():
        print(f"    N={N_str:>8}: {d['smooth_count']:>7} smooth "
              f"({d['fraction']*100:.2f}%)")
    print(f"\n  Time: {elapsed:.2f}s")
    print(f"  PASS: {all_pass}")

    summary = {
        "basis": list(BASIS),
        "K": K,
        "max_basis_prime": max(BASIS),
        "bijection_break": {
            "n": 19,
            "sa_19": list(sa_19),
            "sa_1": list(sa_1),
            "collision": collision,
            "reconstructed_as": recon_19,
            "first_non_smooth_prime": 19
        },
        "smoothness_fractions": smoothness_fractions,
        "remainder_examples": remainder_examples,
    }

    return summary, all_pass, elapsed


# =============================================================================
# PYTEST ENTRY POINT
# =============================================================================
def test_b_smooth_boundary():
    _, passed, _ = run_b_smooth_boundary_experiment()
    assert passed, "EXP-5.3b: B-smooth boundary limitation verification failed"


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  EXP-5.3b: B-Smooth Boundary Limitation")
    print("="*60)

    summary, passed, elapsed = run_b_smooth_boundary_experiment()

    output = {
        "experiment": "EXP-5.3b: B-Smooth Boundary",
        **summary,
        "success": passed,
        "time_seconds": round(elapsed, 3),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version
    }

    results_file = os.path.join(PROJECT_ROOT, 'results',
                                'exp_05_3b_b_smooth_boundary.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")

    sys.exit(0 if passed else 1)
