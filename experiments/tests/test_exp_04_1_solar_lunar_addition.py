"""
E4.1: Deterministic Addition Law — Solar + Lunar → Multiprime

Experiment Specification (from manifesto_experiments.md):
- Claim: For any p ∈ S₁ (Solar) and q ∈ S₂ (Lunar): cl(p+q) = S₆ (Multiprime)
- Test Range: Solar primes p ≡ 1 (mod 6), p ∈ [7, 100,000]
              Lunar primes q ≡ 5 (mod 6), q ∈ [5, 100,000]
- Sample Size: 90,000 random pairs from cross-product
- Success Criterion: 100% of pairs yield cl(p+q) = S₆

Proof (from manifesto):
  p ≡ 1 (mod 6), q ≡ 5 (mod 6) → p + q ≡ 0 (mod 6)
  → 6 | (p+q) and p ≥ 7, q ≥ 5 → p+q ≥ 12
  → Ω(p+q) ≥ Ω(2·3·2) ≥ 3 → cl(p+q) = S₆
"""

import json
import os
import random
from sympy import primefactors, factorint
from typing import List, Tuple, Dict

# ============================================================================
# Classification Function (from P1 framework)
# ============================================================================

def omega(n: int) -> int:
    """Number of prime factors with multiplicity (big Omega function)."""
    if n <= 1:
        return 0
    factors = factorint(n)
    return sum(factors.values())

def classify(n: int) -> str:
    """
    Classify integer n into one of seven topological classes.

    Returns:
        str: One of {'S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6'}
    """
    if n == 1:
        return 'S0'  # Unit
    elif n == 2:
        return 'S3'  # Dyadic
    elif n == 3:
        return 'S4'  # Triadic
    elif omega(n) == 1:  # Prime
        if n % 6 == 1:
            return 'S1'  # Solar
        elif n % 6 == 5:
            return 'S2'  # Lunar
        else:
            raise ValueError(f"Prime {n} has invalid residue mod 6")
    elif omega(n) == 2:
        return 'S5'  # Semiprime
    elif omega(n) >= 3:
        return 'S6'  # Multiprime
    else:
        raise ValueError(f"Cannot classify {n}")

def is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# ============================================================================
# Test Harness
# ============================================================================

def generate_solar_primes(max_val: int = 100000) -> List[int]:
    """Generate all primes p ≡ 1 (mod 6) up to max_val."""
    primes = []
    for p in range(7, max_val + 1, 6):
        if is_prime(p):
            primes.append(p)
    return primes

def generate_lunar_primes(max_val: int = 100000) -> List[int]:
    """Generate all primes q ≡ 5 (mod 6) up to max_val."""
    primes = []
    for q in range(5, max_val + 1, 6):
        if is_prime(q):
            primes.append(q)
    return primes

def test_solar_lunar_addition(sample_size: int = 90000, max_val: int = 100000) -> Dict:
    """
    Test E4.1: Solar + Lunar → Multiprime

    Args:
        sample_size: Number of random pairs to test (default 90,000 from manifesto)
        max_val: Maximum value for prime range (default 100,000)

    Returns:
        dict: Test results with statistics
    """
    print("\n" + "="*80)
    print("E4.1: DETERMINISTIC ADDITION LAW -- Solar + Lunar = Multiprime")
    print("="*80)

    # Generate solar and lunar primes
    print("\nGenerating Solar primes (p ≡ 1 mod 6)...")
    solar_primes = generate_solar_primes(max_val)
    print(f"  Found {len(solar_primes)} Solar primes in [{7}, {max_val}]")

    print("Generating Lunar primes (q ≡ 5 mod 6)...")
    lunar_primes = generate_lunar_primes(max_val)
    print(f"  Found {len(lunar_primes)} Lunar primes in [{5}, {max_val}]")

    # Sample random pairs
    print(f"\nSampling {sample_size} random pairs...")
    results = {
        'experiment': 'E4.1: Solar + Lunar Addition Law',
        'max_val': max_val,
        'n_solar': len(solar_primes),
        'n_lunar': len(lunar_primes),
        'n_samples': sample_size,
        'results': [],
        'summary': {}
    }

    successes = 0
    failures = []
    outcome_distribution = {}

    random.seed(42)  # Reproducible results

    for i in range(sample_size):
        p = random.choice(solar_primes)
        q = random.choice(lunar_primes)
        s = p + q

        # Classify result
        try:
            result_class = classify(s)
            omega_s = omega(s)

            outcome_distribution[result_class] = outcome_distribution.get(result_class, 0) + 1

            is_success = (result_class == 'S6')
            if is_success:
                successes += 1
            else:
                failures.append({
                    'p': p,
                    'q': q,
                    's': s,
                    'omega_s': omega_s,
                    'class': result_class
                })

            results['results'].append({
                'pair_index': i + 1,
                'p': p,
                'q': q,
                'sum': s,
                'omega': omega_s,
                'class': result_class,
                'success': is_success
            })

        except Exception as e:
            failures.append({
                'p': p,
                'q': q,
                's': s,
                'error': str(e)
            })
            results['results'].append({
                'pair_index': i + 1,
                'p': p,
                'q': q,
                'sum': s,
                'error': str(e)
            })

        # Progress
        if (i + 1) % 10000 == 0:
                    print(f"  Processed {i + 1} / {sample_size} pairs...")

    # Summary statistics
    success_rate = (successes / sample_size) * 100

    results['summary'] = {
        'total_pairs': sample_size,
        'successes': successes,
        'failures': len(failures),
        'success_rate': success_rate,
        'outcome_distribution': outcome_distribution,
        'passed': (len(failures) == 0)
    }

    # Print summary
    print("\n" + "-"*80)
    print("TEST RESULTS")
    print("-"*80)
    print(f"Total pairs tested:        {sample_size}")
    print(f"Successes (cl(p+q)=S₆):    {successes} ({success_rate:.2f}%)")
    print(f"Failures:                  {len(failures)}")
    print(f"\nOutcome distribution:")
    for cls in sorted(outcome_distribution.keys()):
        count = outcome_distribution[cls]
        pct = (count / sample_size) * 100
        print(f"  {cls}: {count:5d} ({pct:6.2f}%)")

    if len(failures) > 0:
        print(f"\nFAILURES (first 5):")
        for failure in failures[:5]:
            print(f"  p={failure.get('p')}, q={failure.get('q')}, p+q={failure.get('s')}, "
                  f"Ω={failure.get('omega_s')}, class={failure.get('class')}")

    # Test verdict
    print("\n" + "-"*80)
    verdict = "✓ PASS" if results['summary']['passed'] else "✗ FAIL"
    print(f"Verdict: {verdict}")

    if results['summary']['passed']:
        print("\nTHEOREM 4 EMPIRICALLY VERIFIED:")
        print("  For all Solar primes p ≡ 1 (mod 6) and Lunar primes q ≡ 5 (mod 6),")
        print("  the sum p + q is classified as S₆ (Multiprime, Ω(p+q) ≥ 3).")
        print(f"\n  Verified on {sample_size:,} random pairs with 100% success rate.")
    else:
        print(f"\nWARNING: {len(failures)} failures detected. Theory may be incorrect.")

    print("="*80 + "\n")

    return results

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Run test with manifesto-specified parameters
    results = test_solar_lunar_addition(sample_size=90000, max_val=100000)

    # Save results to JSON
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    output_file = os.path.join(PROJECT_ROOT, 'results', 'exp_04_1_solar_lunar_addition.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

    # Exit with appropriate code
    exit(0 if results['summary']['passed'] else 1)
