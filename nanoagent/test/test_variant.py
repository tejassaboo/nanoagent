"""The variant steps (nanoagent.plugin.sketches_variant): one test per ``harness.sketch_opts`` flag and one for
``harness.backup: learned``, each checking the structure's guarantee and that the selector runs with it."""

from __future__ import annotations

import math
import random

from nanoagent.plugin import policy_vanilla
from nanoagent.plugin import sketches_variant as variant
from nanoagent.plugin.sketches import BloomFilter, CountMinSketch, ExactCounter, MinHash, jaccard
from test_policy_switch import HANDBOOK, INSTRUCTIONS


def kept(**options) -> list[int]:
    return policy_vanilla.chosen(HANDBOOK, INSTRUCTIONS, **options)[1]


def test_probe_sizing_keeps_expected_false_hits_over_all_probes_below_0_1() -> None:
    queries, chunks = 50, 40
    p = variant.probe_p(queries, chunks)
    assert p == 0.1 / (queries * chunks) and variant.probe_p(1, 1) == 0.01  # never looser than vanilla
    false_hits = 0
    for seed in range(20):
        for c in range(chunks):
            f = BloomFilter(30, p, salt=f"{seed}:{c}")
            for i in range(30):
                f.add(f"in{c}_{i}")
            false_hits += sum(f.contains(f"q{q}") for q in range(queries))
    assert false_hits / 20 <= 0.3  # E = |Q| N p = 0.1 per handbook; vanilla's p = 0.01 gives 20
    assert f.m / 30 >= math.log2(1 / p) / math.log(2) - 0.1  # the bits per item that buys it
    assert kept(sketch_opts=("probe_sizing",)) == kept(twin="exact")


def test_conservative_update_never_underestimates_and_never_exceeds_plain_count_min() -> None:
    rng = random.Random(3)
    plain, cu, exact = CountMinSketch(salt="t"), variant.ConservativeCountMin(salt="t"), ExactCounter()
    for _ in range(5000):  # about 1,000 distinct terms over 272 columns, as a long handbook's df
        x = f"w{rng.randrange(1000)}"
        for s in (plain, cu, exact):
            s.add(x)
    over = {x: (plain.query(x) - c, cu.query(x) - c) for x, c in exact.counts.items()}
    assert all(0 <= o_cu <= o_plain for o_plain, o_cu in over.values())
    assert sum(o for _, o in over.values()) < 0.5 * sum(o for o, _ in over.values())
    assert kept(sketch_opts=("conservative_update",)) == kept()


def test_one_permutation_hashing_is_unbiased_and_densifies_every_bin() -> None:
    rng, errs = random.Random(4), []
    for trial in range(60):
        universe = [f"s{trial}_{i}" for i in range(80)]
        a, b = set(rng.sample(universe, 40)), set(rng.sample(universe, 40))  # 40 items < K = 128 bins
        mh = variant.OnePermutationMinHash(128, salt=f"t{trial}")
        sig_a = mh.signature(a)
        assert MinHash.EMPTY not in sig_a and len(sig_a) == 128
        errs.append(MinHash.estimate(sig_a, mh.signature(b)) - jaccard(a, b))
    assert abs(sum(errs) / len(errs)) < 0.02 and math.sqrt(sum(e * e for e in errs) / len(errs)) < 0.15
    assert MinHash.estimate(*(variant.OnePermutationMinHash(128).signature(set()),) * 2) == 0.0
    near = variant.OnePermutationNearDuplicates()
    near.add("base", {f"t{i}" for i in range(100)})
    assert near.similar({f"t{i}" for i in range(95)} | {"x"}, 0.8) == ["base"]
    assert kept(sketch_opts=("one_permutation",)) == kept()


def test_double_hashing_has_no_false_negatives_and_a_rate_near_p() -> None:
    f = variant.DoubleHashBloom(2000, 0.01, salt="dh")
    assert all(f.m % d for d in range(2, math.isqrt(f.m) + 1))  # m prime, so h2 in [1, m - 1] is invertible
    assert len(set(f._positions("x"))) == f.k
    for i in range(2000):
        f.add(f"in{i}")
    assert all(f.contains(f"in{i}") for i in range(2000))
    assert abs(sum(f.contains(f"out{i}") for i in range(50000)) / 50000 - 0.01) < 0.005
    assert kept(sketch_opts=("double_hash",)) == kept(twin="exact")


def test_learned_backup_loses_no_key_chunk_and_its_twin_admits_only_keys() -> None:
    chunks = [f"Rule {i}: the fare cap for economy class flights is ${100 + i} per ticket." for i in range(20)]
    chunks += [f"General note {i} about the company mission and its values." for i in range(200)]
    terms = [set(policy_vanilla.tokens(c)) for c in chunks]
    query = set(policy_vanilla.tokens("economy class fare cap for flights"))
    keys = {i for i, c in enumerate(chunks) if variant.is_key(c, terms[i], query)}
    assert keys == set(range(20))
    exact = variant.backup_admit(chunks, terms, query, [0, 1], exact=True, seed=580)
    sketch = variant.backup_admit(chunks, terms, query, [0, 1], exact=False, seed=580)
    assert set(exact) == keys and keys <= set(sketch) and len(set(sketch) - keys) <= 10  # p = 0.01 over 200
    assert set(kept(fraction=variant.SCORER_FRACTION)) <= set(kept(backup="learned"))  # the scorer's cut, then admits
