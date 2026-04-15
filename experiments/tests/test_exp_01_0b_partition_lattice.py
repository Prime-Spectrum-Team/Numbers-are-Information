#!/usr/bin/env python3
"""
=============================================================================
EXP-01.0b: Exhaustive partition-lattice enumeration (Maximality proof)
=============================================================================
Strengthens Theorem thm:maximality (paper/20260401_PrimeSpectrum.tex) by
enumerating *every* partition P of the ground set {S0, S1, ..., S6}
(Bell(7) = 877 total) and verifying two quotient conditions per partition:

  (i)  otimes-closure:
       For every ordered pair of blocks (B_i, B_j), the image set
       { MUL_TABLE[a][b] : a in B_i, b in B_j } is contained in a single
       block of P. Otherwise the induced multiplication C_i (x) C_j is
       not well-defined.

  (ii) Preservation of the 7 deterministic addition laws
       (Table tab:7laws in the paper):
         (S0,S0)->S3, (S0,S3)->S4, (S0,S4)->S5,
         (S3,S3)->S5, (S3,S4)->S2, (S4,S4)->S5, (S1,S2)->S6.
       A law (i,j)->k is preserved iff every pair (i',j') whose blocks
       coincide with those of (i,j) has all observed output classes in
       block [k]. Empirical output class sets are read from
       results/exp_10_0_addition_ci.json.

Main claim (verified here): the identity 7-block partition is the unique
MAXIMAL (finest) element in the refinement lattice of partitions satisfying
both (i) and (ii). Any coarsening of it that still satisfies both conditions
is strictly coarser and therefore loses distinguishing power; no strictly
finer partition exists.

-> paper Theorem 3.3 (thm:maximality)

(C) Prime-Spectrum-Team, April 2026
=============================================================================
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Iterator, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'shared'))

from spectral_matrix import MUL_TABLE, CLASS_NAME  # noqa: E402

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
N_CLASSES = 7

# Seven deterministic addition laws (unordered pairs) from Table tab:7laws.
DET_LAWS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 3),   # 1+1=2
    (0, 3, 4),   # 1+2=3
    (0, 4, 5),   # 1+3=4
    (3, 3, 5),   # 2+2=4
    (3, 4, 2),   # 2+3=5
    (4, 4, 5),   # 3+3=6
    (1, 2, 6),   # Solar+Lunar (Theorem thm:solar-lunar)
)

# -----------------------------------------------------------------------------
# Restricted-growth-string (RGS) enumeration of set partitions
# -----------------------------------------------------------------------------
def restricted_growth_strings(n: int) -> Iterator[Tuple[int, ...]]:
    """Yield all RGS of length n; each encodes one set partition of {0..n-1}.

    RGS convention: a[0] = 0 and a[i] <= 1 + max(a[0..i-1]).
    Number of RGS of length n equals Bell(n).
    """
    a = [0] * n

    def rec(i: int, max_so_far: int) -> Iterator[Tuple[int, ...]]:
        if i == n:
            yield tuple(a)
            return
        for v in range(max_so_far + 2):
            a[i] = v
            yield from rec(i + 1, max(max_so_far, v))

    if n == 0:
        yield ()
        return
    yield from rec(1, 0)


def rgs_blocks(rgs: Tuple[int, ...]) -> List[List[int]]:
    """Convert RGS to list of blocks (each a sorted list of class indices)."""
    groups: dict[int, List[int]] = {}
    for i, v in enumerate(rgs):
        groups.setdefault(v, []).append(i)
    return [sorted(b) for b in sorted(groups.values(), key=lambda b: b[0])]


# -----------------------------------------------------------------------------
# Load 28-pair addition output-class sets from EXP-10.0 results
# -----------------------------------------------------------------------------
def load_addition_output_sets() -> dict[Tuple[int, int], frozenset]:
    """Return {(a,b) with a<=b: frozenset of output classes} from EXP-10.0 JSON."""
    path = os.path.join(PROJECT_ROOT, 'results', 'exp_10_0_addition_ci.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required artifact not found: {path}\n"
            "Run test_exp_10_0_addition_ci.py first to generate 28-pair distributions."
        )
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result: dict[Tuple[int, int], frozenset] = {}
    for key, entry in data['results'].items():
        a = entry['a_class']
        b = entry['b_class']
        out = frozenset(int(k) for k in entry['distribution'].keys())
        result[(min(a, b), max(a, b))] = out
    return result


# -----------------------------------------------------------------------------
# Quotient checks under a partition
# -----------------------------------------------------------------------------
def check_otimes_closure(block_of: Tuple[int, ...]) -> bool:
    """
    True iff for every unordered pair of blocks (B_i, B_j) the image
    { MUL_TABLE[a][b] : a in B_i, b in B_j } lies in a single block.
    """
    # block_of[c] is the block index of class c.
    # Collect per block-pair the set of output-block-indices.
    pair_outputs: dict[Tuple[int, int], set[int]] = {}
    for a in range(N_CLASSES):
        for b in range(N_CLASSES):
            bi, bj = block_of[a], block_of[b]
            key = (min(bi, bj), max(bi, bj))
            pair_outputs.setdefault(key, set()).add(block_of[MUL_TABLE[a][b]])
    return all(len(s) == 1 for s in pair_outputs.values())


def check_addition_laws_preserved(
    block_of: Tuple[int, ...],
    add_output_sets: dict[Tuple[int, int], frozenset],
) -> Tuple[bool, int]:
    """
    For each of the 7 deterministic laws (i,j)->k, verify that under the
    quotient induced by block_of, the block-pair ([i],[j]) still has a
    single output block (and that block is [k]).

    Returns (all_preserved, num_preserved).
    """
    preserved = 0
    for (i, j, k) in DET_LAWS:
        target_block = block_of[k]
        # Collect every class pair (i',j') with same input-block pair.
        out_blocks: set[int] = set()
        for (a, b), out_set in add_output_sets.items():
            if {block_of[a], block_of[b]} != {block_of[i], block_of[j]}:
                continue
            for c in out_set:
                out_blocks.add(block_of[c])
        # Law preserved iff block-level pair is deterministic AND maps to [k].
        if out_blocks == {target_block}:
            preserved += 1
    return preserved == len(DET_LAWS), preserved


# -----------------------------------------------------------------------------
# Main enumeration
# -----------------------------------------------------------------------------
def enumerate_partition_lattice() -> dict:
    print("\n" + "=" * 70)
    print("  EXP-01.0b: exhaustive partition-lattice enumeration (Bell(7)=877)")
    print("=" * 70)

    add_output_sets = load_addition_output_sets()
    print(f"  Loaded 28-pair output-class sets from EXP-10.0 results.")

    t0 = time.time()

    partitions: List[dict] = []
    closure_count = 0
    laws_all_count = 0
    both_count = 0
    identity_rgs = tuple(range(N_CLASSES))

    for rgs in restricted_growth_strings(N_CLASSES):
        k = max(rgs) + 1
        blocks = rgs_blocks(rgs)
        closure_ok = check_otimes_closure(rgs)
        laws_ok, laws_preserved = check_addition_laws_preserved(rgs, add_output_sets)

        if closure_ok:
            closure_count += 1
        if laws_ok:
            laws_all_count += 1
        if closure_ok and laws_ok:
            both_count += 1

        partitions.append({
            'rgs': list(rgs),
            'num_blocks': k,
            'blocks': blocks,
            'otimes_closed': closure_ok,
            'det_laws_preserved_count': laws_preserved,
            'det_laws_all_preserved': laws_ok,
            'is_identity': rgs == identity_rgs,
        })

    elapsed = time.time() - t0

    total = len(partitions)
    identity_entry = next(p for p in partitions if p['is_identity'])
    both_entries = [p for p in partitions if p['otimes_closed'] and p['det_laws_all_preserved']]

    print(f"\n  Total partitions enumerated : {total} (expected Bell(7)=877)")
    print(f"  otimes-closed partitions    : {closure_count}")
    print(f"  All 7 det.laws preserved    : {laws_all_count}")
    print(f"  Both (i) and (ii)           : {both_count}")
    print(f"  Elapsed                     : {elapsed:.2f}s")

    print("\n  Partitions satisfying BOTH conditions:")
    for p in both_entries:
        blk_str = " | ".join("{" + ",".join(f"S{c}" for c in b) + "}" for b in p['blocks'])
        marker = " <-- identity" if p['is_identity'] else ""
        print(f"    k={p['num_blocks']}  {blk_str}{marker}")

    # --- Verify chain structure of the both-condition partitions ----------
    # Each valid partition (other than the identity) must be a strict
    # coarsening of the identity; moreover, they must form a chain, i.e.
    # every pair is refinement-comparable. The maximum (finest) is unique.
    def is_refinement_of(fine: Tuple[int, ...], coarse: Tuple[int, ...]) -> bool:
        """True iff every block of `fine` is contained in some block of `coarse`."""
        for a in range(N_CLASSES):
            for b in range(a + 1, N_CLASSES):
                if fine[a] == fine[b] and coarse[a] != coarse[b]:
                    return False
        return True

    both_rgs = [tuple(p['rgs']) for p in both_entries]
    # Sort by num_blocks descending: identity first, trivial last.
    both_rgs_sorted = sorted(both_rgs, key=lambda r: -(max(r) + 1))
    chain_ok = all(
        is_refinement_of(both_rgs_sorted[i], both_rgs_sorted[i + 1])
        for i in range(len(both_rgs_sorted) - 1)
    )
    max_blocks = max((max(r) + 1) for r in both_rgs)
    finest = [r for r in both_rgs if max(r) + 1 == max_blocks]

    # --- Assertions (main claim) -------------------------------------------
    assert total == 877, f"Bell(7) should be 877, got {total}"
    assert identity_entry['otimes_closed'], "Identity partition must be otimes-closed"
    assert identity_entry['det_laws_all_preserved'], "Identity must preserve all 7 laws"
    assert len(finest) == 1 and finest[0] == identity_rgs, (
        f"Uniqueness of finest refinement violated: {finest}"
    )
    assert chain_ok, (
        "The partitions satisfying (i) and (ii) are expected to form a chain "
        "in the refinement order; enumeration found an incomparable pair."
    )

    print(f"\n  Chain verification: all {len(both_rgs)} valid partitions are")
    print(f"  pairwise refinement-comparable -> chain structure confirmed.")
    print(f"  Unique maximal (finest) element: identity 7-class partition.")

    return {
        'experiment': 'EXP-01.0b: Exhaustive partition-lattice enumeration',
        'ground_set_size': N_CLASSES,
        'bell_number': total,
        'counts': {
            'otimes_closed': closure_count,
            'det_laws_all_preserved': laws_all_count,
            'both_conditions': both_count,
        },
        'unique_maximal_is_identity': True,
        'forms_chain': chain_ok,
        'max_block_count': max_blocks,
        'elapsed_seconds': elapsed,
        'partitions': partitions,
    }


# -----------------------------------------------------------------------------
# Pytest entry point
# -----------------------------------------------------------------------------
def test_partition_lattice_maximality():
    result = enumerate_partition_lattice()
    assert result['bell_number'] == 877
    assert result['max_block_count'] == 7
    assert result['unique_maximal_is_identity'] is True
    assert result['forms_chain'] is True


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    output = enumerate_partition_lattice()
    out_path = os.path.join(PROJECT_ROOT, 'results', 'exp_01_0b_partition_lattice.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")
    print("  Maximality of the 7-class partition: VERIFIED exhaustively.")
