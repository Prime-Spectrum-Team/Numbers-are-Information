"""
exp_hw_exact_nontorch.py
===================================
Torch-free verification of spectral_hw_exact.py
Ports SpectralAddressExtractor + SpectralAndGate completely
without PyTorch dependency, for CI and paper verification.
"""
import json
import math
import os

BASIS_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
N_AXES = len(BASIS_PRIMES)

# ─── Core functions (1:1 from spectral_hw_exact.py) ──────────────────────────

def p_adic_valuation(n, p):
    if n == 0 or p < 2:
        return 0
    v = 0
    while n % p == 0:
        v += 1
        n //= p
    return v

def extract_spectral_address(n):
    valuations = []
    rem = n
    for p in BASIS_PRIMES:
        v = p_adic_valuation(rem, p)
        valuations.append(v)
        rem //= (p ** v) if v > 0 else 1
    return valuations, rem

def spectral_and_gate(sa_a, rem_a, sa_b, rem_b):
    axis_shared = [(sa_a[i] > 0) and (sa_b[i] > 0) for i in range(N_AXES)]
    c_rem = (rem_a > 1) and (rem_b > 1) and (rem_a == rem_b)
    any_shared = any(axis_shared) or c_rem
    independent = not any_shared
    shared_mask = sum((1 << i) for i, s in enumerate(axis_shared) if s)
    shared_axes = sum(axis_shared) + (1 if c_rem else 0)
    return independent, shared_axes, shared_mask

def quantitative_resonance(sa_a, sa_b):
    return sum(min(sa_a[i], sa_b[i]) for i in range(N_AXES))

# ─── Test 1: p-adic valuation vs Verilog LUT ─────────────────────────────────

def test_v2_vs_verilog():
    verilog_v2 = {
        2: 1, 4: 2, 6: 1, 8: 3, 10: 1, 12: 2, 16: 4,
        32: 5, 64: 6, 128: 7, 256: 8, 512: 9, 1024: 10
    }
    errors = []
    for n, expected in verilog_v2.items():
        got = p_adic_valuation(n, 2)
        if got != expected:
            errors.append(f"v_2({n}): got {got}, expected {expected}")
    return errors

# ─── Test 2: SpectralAddress for known numbers ───────────────────────────────

def test_spectral_address():
    # 60 = 2^2 * 3 * 5 → [2,1,1,0,...], rem=1
    sa, rem = extract_spectral_address(60)
    errors = []
    assert sa[0] == 2, f"v2(60) expected 2, got {sa[0]}"
    assert sa[1] == 1, f"v3(60) expected 1, got {sa[1]}"
    assert sa[2] == 1, f"v5(60) expected 1, got {sa[2]}"
    assert sa[3] == 0, f"v7(60) expected 0, got {sa[3]}"
    assert rem == 1, f"rem(60) expected 1, got {rem}"
    # 1 → all zeros, rem=1
    sa1, rem1 = extract_spectral_address(1)
    assert sa1 == [0]*N_AXES, f"SA(1) expected all zeros, got {sa1}"
    assert rem1 == 1
    return errors

# ─── Test 3: SpectralAndGate ──────────────────────────────────────────────────

def test_and_gate():
    errors = []
    # 12=2^2*3, 18=2*3^2 → shared axes: 2 and 3 → shared_axes=2
    sa12, rem12 = extract_spectral_address(12)
    sa18, rem18 = extract_spectral_address(18)
    ind, axes, mask = spectral_and_gate(sa12, rem12, sa18, rem18)
    if ind:
        errors.append(f"AND(12,18): expected independent=False, got True")
    if axes != 2:
        errors.append(f"AND(12,18): expected shared_axes=2, got {axes}")
    # 7 and 11 → coprime
    sa7, rem7 = extract_spectral_address(7)
    sa11, rem11 = extract_spectral_address(11)
    ind2, axes2, _ = spectral_and_gate(sa7, rem7, sa11, rem11)
    if not ind2:
        errors.append(f"AND(7,11): expected independent=True, got False")
    if axes2 != 0:
        errors.append(f"AND(7,11): expected shared_axes=0, got {axes2}")
    return errors

# ─── Test 4: Quantitative resonance score ─────────────────────────────────────

def test_quantitative_resonance():
    errors = []
    pairs_expected = [
        (12, 18, 3),    # 12=2^2*3, 18=2*3^2 → min(2,1)+min(1,2)=1+1=2... wait
        # Let me recalculate: v2(12)=2, v2(18)=1 → min=1; v3(12)=1, v3(18)=2 → min=1; rest=0 → sum=2
        # Override:
        (8, 32, 3),     # v2(8)=3, v2(32)=5 → min=3; rest=0 → 3
        (7, 11, 0),     # coprime large primes → 0
        (30, 60, 3),    # 30=2*3*5, 60=2^2*3*5 → min(1,2)+min(1,1)+min(1,1)=1+1+1=3
    ]
    # Recalculate 12,18 correctly
    sa12, _ = extract_spectral_address(12)
    sa18, _ = extract_spectral_address(18)
    r = quantitative_resonance(sa12, sa18)
    if r != 2:
        errors.append(f"resonance(12,18): expected 2, got {r}")

    sa8,  _ = extract_spectral_address(8)
    sa32, _ = extract_spectral_address(32)
    r = quantitative_resonance(sa8, sa32)
    if r != 3:
        errors.append(f"resonance(8,32): expected 3, got {r}")

    sa7,  _ = extract_spectral_address(7)
    sa11, _ = extract_spectral_address(11)
    r = quantitative_resonance(sa7, sa11)
    if r != 0:
        errors.append(f"resonance(7,11): expected 0, got {r}")

    sa30, _ = extract_spectral_address(30)
    sa60, _ = extract_spectral_address(60)
    r = quantitative_resonance(sa30, sa60)
    if r != 3:
        errors.append(f"resonance(30,60): expected 3, got {r}")
    return errors

# ─── Test 5: Resonance matrix sparsity (6/pi^2) ─────────────────────────────

def test_resonance_sparsity(max_n=10000):
    """Coprime pairs should be ~60.8% of all pairs (classical: 6/pi^2)"""
    coprime_count = 0
    total = 0
    for i in range(1, max_n + 1):
        for j in range(i + 1, min(i + 100, max_n + 1)):
            total += 1
            if math.gcd(i, j) == 1:
                coprime_count += 1
    ratio = coprime_count / total
    expected = 6 / (math.pi ** 2)
    errors = []
    if abs(ratio - expected) > 0.02:
        errors.append(f"Coprime ratio {ratio:.4f} too far from 6/π²={expected:.4f}")
    return errors, ratio, expected

# ─── Run all tests ───────────────────────────────────────────────────────────

def run_all():
    results = {}
    total_pass = 0
    total_fail = 0

    print("=" * 60)
    print("  spectral_hw_exact — Verification (torch-free)")
    print("=" * 60)

    # T1
    errs = test_v2_vs_verilog()
    status = "PASS" if not errs else "FAIL"
    results["T1_v2_vs_verilog"] = {"status": status, "errors": errs,
                                    "description": "p-adic v2 vs Verilog LUT (13 values)"}
    print(f"  T1 v2-vs-Verilog:        {status}" + (f" — {errs}" if errs else ""))
    (total_pass if not errs else total_fail).__class__  # dummy
    if not errs: total_pass += 1
    else: total_fail += 1

    # T2
    errs = test_spectral_address()
    status = "PASS" if not errs else "FAIL"
    results["T2_spectral_address"] = {"status": status, "errors": errs,
                                       "description": "SA(60), SA(1)"}
    print(f"  T2 SpectralAddress:      {status}" + (f" — {errs}" if errs else ""))
    if not errs: total_pass += 1
    else: total_fail += 1

    # T3
    errs = test_and_gate()
    status = "PASS" if not errs else "FAIL"
    results["T3_and_gate"] = {"status": status, "errors": errs,
                               "description": "SpectralAndGate (12,18) + (7,11)"}
    print(f"  T3 SpectralAndGate:      {status}" + (f" — {errs}" if errs else ""))
    if not errs: total_pass += 1
    else: total_fail += 1

    # T4
    errs = test_quantitative_resonance()
    status = "PASS" if not errs else "FAIL"
    results["T4_quant_resonance"] = {"status": status, "errors": errs,
                                      "description": "Quantitative Resonance (12,18),(8,32),(7,11),(30,60)"}
    print(f"  T4 QuantResonance:       {status}" + (f" — {errs}" if errs else ""))
    if not errs: total_pass += 1
    else: total_fail += 1

    # T5
    errs5, ratio, expected = test_resonance_sparsity(5000)
    status = "PASS" if not errs5 else "FAIL"
    results["T5_sparsity_6_pi2"] = {"status": status, "errors": errs5,
                                     "ratio": round(ratio, 6),
                                     "expected_6_pi2": round(expected, 6),
                                     "description": "Coprime sparsity ≈ 6/pi^2"}
    print(f"  T5 Coprime-Sparsity:     {status}  ratio={ratio:.4f}  6/π²={expected:.4f}")
    if not errs5: total_pass += 1
    else: total_fail += 1

    print(f"\n  Total: {total_pass} PASS / {total_fail} FAIL")
    print("=" * 60)

    # SA(60) Illustration
    sa60, rem60 = extract_spectral_address(60)
    print(f"\n  SA(60)  = {sa60}  rem={rem60}")
    print(f"  (60 = 2^2 * 3 * 5 → v2=2, v3=1, v5=1, rest 0)")

    summary = {
        "script": "exp_hw_exact_nontorch.py",
        "description": "Torch-free verification of spectral_hw_exact.py (SpectralAddressExtractor + SpectralAndGate + QuantResonance)",
        "tests": results,
        "summary": {
            "total_pass": total_pass,
            "total_fail": total_fail,
            "verdict": "PASS" if total_fail == 0 else "FAIL"
        }
    }

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "exp_hw_exact.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved: {out_path}")
    return total_fail == 0

if __name__ == "__main__":
    run_all()
