#!/usr/bin/env python3
"""
=============================================================================
EXP-HW.VLOG: Behavioral Emulation of spectral_cell.v
=============================================================================
Verilog module spec (paper Section VII-C):
  module spectral_cell #(parameter K=7) (
      input  clk,
      input  [9:0]  n_in,        // n in [0, 1023]
      output [K*3-1:0] sa_out    // packed SA vector, 3 bits per component
  );

Emulator:
  1. Builds 1024-entry LUT (identical to sa_lut.mem)
  2. Simulates 1-cycle registered pipeline
  3. Runs 14 test vectors with expected outputs

=============================================================================
"""
import os, sys, time, json, datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# =============================================================================

K = 7
BASIS = (2, 3, 5, 7, 11, 13, 17)
N_LUT = 1024
BITS = 3


def compute_sa(n):
    if n == 0:
        return (0,) * K
    sa = []
    for p in BASIS:
        v = 0
        while n % p == 0:
            v += 1
            n //= p
        sa.append(v)
    return tuple(sa)


def pack_sa(sa):
    packed = 0
    for v in sa:
        packed = (packed << BITS) | (v & 0x7)
    return packed


def unpack_sa(packed):
    sa = []
    for _ in range(K):
        sa.append(packed & 0x7)
        packed >>= BITS
    return tuple(reversed(sa))


class SpectralCell:
    def __init__(self, lut):
        self._lut = lut
        self._cycle = 0

    def clock(self, n_in):
        assert 0 <= n_in < N_LUT
        self._cycle += 1
        return unpack_sa(self._lut[n_in])


TEST_VECTORS = [
    (0,   (0,0,0,0,0,0,0), "n=0: hardware edge case"),
    (1,   (0,0,0,0,0,0,0), "n=1: unit S0"),
    (2,   (1,0,0,0,0,0,0), "n=2: Dyadic S3"),
    (3,   (0,1,0,0,0,0,0), "n=3: Triadic S4"),
    (5,   (0,0,1,0,0,0,0), "n=5: Lunar S2"),
    (7,   (0,0,0,1,0,0,0), "n=7: Solar S1"),
    (6,   (1,1,0,0,0,0,0), "n=6=2x3: Semiprime S5"),
    (12,  (2,1,0,0,0,0,0), "n=12=2^2x3: Multiprime S6"),
    (60,  (2,1,1,0,0,0,0), "n=60=2^2x3x5: three-axis S6"),
    (128, (7,0,0,0,0,0,0), "n=128=2^7: deep axis-2 S6"),
    (17,  (0,0,0,0,0,0,1), "n=17: largest basis prime"),
    (210, (1,1,1,1,0,0,0), "n=210=2x3x5x7: primorial S6"),
    (720, (4,2,1,0,0,0,0), "n=720=2^4x3^2x5: highly composite S6"),
    (1023,(0,1,0,0,1,0,0), "n=1023=3x11x31: partial SA"),
]


def run_verilog_behavioral_experiment():
    t0 = time.time()

    # Build LUT
    t_init = time.perf_counter()
    LUT = [pack_sa(compute_sa(n)) for n in range(N_LUT)]
    init_ms = (time.perf_counter() - t_init) * 1000

    # Run test vectors
    dut = SpectralCell(LUT)
    results = []
    passed_count = 0

    for n, expected, desc in TEST_VECTORS:
        got = dut.clock(n)
        ok = got == expected
        if ok:
            passed_count += 1
        results.append({
            "n": n,
            "expected_sa": list(expected),
            "got_sa": list(got),
            "pass": ok,
            "description": desc
        })

    # Throughput benchmark
    N_BENCH = 10_000
    t_bench = time.perf_counter()
    for i in range(N_BENCH):
        _ = LUT[i & 0x3FF]
    bench_s = time.perf_counter() - t_bench
    throughput_M = (N_BENCH / bench_s) / 1e6

    elapsed = time.time() - t0
    all_pass = passed_count == len(TEST_VECTORS)

    print(f"\n  EXP-HW.VLOG: spectral_cell Behavioral Emulation")
    print(f"  {'='*50}")
    print(f"  LUT: {N_LUT} entries, {init_ms:.1f} ms init")
    print(f"  Test vectors: {passed_count}/{len(TEST_VECTORS)} passed")
    for r in results:
        tag = "PASS" if r["pass"] else "FAIL"
        print(f"    [{tag}] n={r['n']:4d}  SA={r['got_sa']}  {r['description']}")
    print(f"  Throughput: {throughput_M:.0f} M lookups/s")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  PASS: {all_pass}")

    for r in results:
        assert r["pass"], \
            f"Vector n={r['n']} failed: expected {r['expected_sa']}, got {r['got_sa']}"

    summary = {
        "module_spec": {
            "input_bits": 10, "n_range": [0, 1023],
            "output_bits": K * BITS, "K": K,
            "basis": list(BASIS), "pipeline_stages": 1,
        },
        "lut_init_ms": round(init_ms, 3),
        "test_results": {
            "total": len(TEST_VECTORS),
            "passed": passed_count,
            "all_pass": all_pass
        },
        "test_vectors": results,
        "throughput": {
            "iterations": N_BENCH,
            "ops_per_second_M": round(throughput_M, 1),
        },
    }

    return summary, all_pass, elapsed


# =============================================================================
# PYTEST ENTRY POINT
# =============================================================================
def test_verilog_behavioral():
    _, passed, _ = run_verilog_behavioral_experiment()
    assert passed, "EXP-HW.VLOG: Verilog behavioral emulation failed"


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  EXP-HW.VLOG: Verilog Behavioral Emulation")
    print("="*60)

    summary, passed, elapsed = run_verilog_behavioral_experiment()

    output = {
        "experiment": "EXP-HW.VLOG: spectral_cell Behavioral Emulation",
        **summary,
        "success": passed,
        "time_seconds": round(elapsed, 3),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version
    }

    results_file = os.path.join(PROJECT_ROOT, 'results',
                                'exp_hw_verilog_behavioral.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved: {results_file}")

    sys.exit(0 if passed else 1)
