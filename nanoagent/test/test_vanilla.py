"""The vanilla sketches, the shared chunker and the ``bloom_chunk`` selector (nanoagent.plugin.sketches,
chunking, policy_vanilla): each structure's guarantee, its exact twin, and the selector's determinism."""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from nanoagent.plugin import chunking, policy_select, policy_vanilla
from nanoagent.plugin.sketches import LSH, BloomFilter, CountMinSketch, ExactCounter, ExactNearDuplicates, MinHash, NearDuplicates, jaccard
from test_policy_switch import HANDBOOK, INSTRUCTIONS

# A folder of real handbooks (*.txt) for the chunk identity check, e.g. the travel-agent project's policies/.
HANDBOOK_DIR = os.environ.get("NANOAGENT_HANDBOOK_DIR")
LONG = HANDBOOK + "\n7. Expenses\n" + "\n".join(f"{c}. Rule {c}\nThe {c} rule covers {' '.join(['receipts'] * 30)}." for c in "abcdef")


# -- the 11 reference tests: Bloom (3), count-min (2), MinHash and LSH (3), selector (3) -------------- #


@pytest.mark.parametrize("n", [1, 10, 1000])
def test_bloom_has_no_false_negatives(n: int) -> None:
    f = BloomFilter(n, 0.01, salt=f"t{n}")
    items = [f"item{i}" for i in range(n)]
    for x in items:
        f.add(x)
    assert all(f.contains(x) for x in items)


@pytest.mark.parametrize("p", [0.1, 0.01])
def test_bloom_false_positive_rate_is_near_p(p: float) -> None:
    f = BloomFilter(2000, p, salt="fpr")
    for i in range(2000):
        f.add(f"in{i}")
    rate = sum(f.contains(f"out{i}") for i in range(50000)) / 50000
    assert abs(rate - p) < 0.5 * p, (p, rate)


def test_bloom_sizing_formula() -> None:
    f = BloomFilter(100, 0.01)
    assert f.m == math.ceil(-100 * math.log(0.01) / math.log(2) ** 2) == 959
    assert f.k == 7


def test_count_min_never_underestimates_and_holds_the_eps_bound() -> None:
    rng = random.Random(1)
    cms, exact = CountMinSketch(eps=0.01, delta=0.01, salt="t"), ExactCounter()
    for _ in range(20000):
        x = f"w{int(rng.paretovariate(1.2))}"
        cms.add(x)
        exact.add(x)
    over = [cms.query(x) - c for x, c in exact.counts.items()]
    assert all(o >= 0 for o in over)
    assert sum(o > 0.01 * cms.total for o in over) / len(over) <= 0.01 + 0.02  # P[error > eps T] <= delta, plus slack


def test_count_min_shape() -> None:
    assert (CountMinSketch(eps=0.01, delta=0.01).width, CountMinSketch(eps=0.01, delta=0.01).depth) == (272, 5)
    assert (CountMinSketch(eps=0.001, delta=0.01).width, CountMinSketch(eps=0.001, delta=0.01).depth) == (2719, 5)


@pytest.mark.parametrize("K", [64, 256])
def test_minhash_is_unbiased_with_error_about_one_over_sqrt_k(K: int) -> None:
    rng, errs = random.Random(2), []
    for trial in range(60):
        universe = [f"s{trial}_{i}" for i in range(400)]
        a, b = set(rng.sample(universe, 200)), set(rng.sample(universe, 200))
        mh = MinHash(K, salt=f"t{trial}")
        errs.append(MinHash.estimate(mh.signature(a), mh.signature(b)) - jaccard(a, b))
    assert math.sqrt(sum(e * e for e in errs) / len(errs)) < 1.5 / math.sqrt(K)
    assert abs(sum(errs) / len(errs)) < 3 / math.sqrt(K * len(errs)) + 0.01


def test_minhash_identical_and_empty_sets() -> None:
    mh = MinHash(128)
    assert MinHash.estimate(mh.signature({"x", "y", "z"}), mh.signature({"x", "y", "z"})) == 1.0
    assert MinHash.estimate(mh.signature(set()), mh.signature(set())) == 0.0


def test_lsh_finds_a_near_duplicate_and_its_twin_agrees() -> None:
    mh, lsh = MinHash(128), LSH(32, 4)
    base = {f"t{i}" for i in range(100)}
    near = set(sorted(base)[:95]) | {"new1", "new2"}  # J = 95/102, about 0.93
    lsh.add("base", mh.signature(base))
    assert "base" in lsh.candidates(mh.signature(near))
    sketch, exact = NearDuplicates(), ExactNearDuplicates()
    for d in (sketch, exact):
        d.add("base", base)
    assert sketch.similar(near, 0.8) == exact.similar(near, 0.8) == ["base"]


def test_the_selector_is_deterministic() -> None:
    assert policy_vanilla.select(HANDBOOK, INSTRUCTIONS) == policy_vanilla.select(HANDBOOK, INSTRUCTIONS)


@pytest.mark.parametrize("twin", ["sketch", "exact"])
def test_the_sketch_and_its_twin_run_and_the_budget_holds(twin: str) -> None:
    out = policy_vanilla.select(LONG, INSTRUCTIONS, fraction=0.35, twin=twin)
    title = len(policy_select.first_line(LONG).split())
    assert out.startswith(policy_select.first_line(LONG)) and len(out.split()) <= 1.05 * 0.35 * len(LONG.split()) + title
    assert "lowest logical airfare" in out and "casino" not in out


def test_the_selection_is_independent_of_pythonhashseed() -> None:
    code = f"from nanoagent.plugin import policy_vanilla as v; import hashlib; print(hashlib.sha256(v.select({LONG!r}, {INSTRUCTIONS!r}).encode()).hexdigest())"
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": str(s)}).stdout
        for s in range(4)
    }
    assert len(outs) == 1, outs


# -- the shared chunker and tokenizer ---------------------------------------------------------------- #


def bm25_pieces(handbook: str) -> list[str]:
    """The pieces policy_select.bm25 scores (chunks_of + split_chunk, a later piece under its section title)."""
    out = []
    for chunk in policy_select.chunks_of(handbook):
        title = chunk.strip().splitlines()[0]
        out += [piece if j == 0 else f"{title}\n{piece}" for j, piece in enumerate(policy_select.split_chunk(chunk))]
    return out


def handbooks() -> list[Path]:
    return sorted(Path(HANDBOOK_DIR).glob("*.txt")) if HANDBOOK_DIR else []


def test_structured_chunks_and_tokens_equal_policy_selects() -> None:
    for text in (HANDBOOK, LONG):
        assert chunking.structured_chunks(text) == bm25_pieces(text)
        assert chunking.tokens(text) == policy_select.tokens(text)
    assert len(chunking.structured_chunks(LONG)) > len(policy_select.chunks_of(LONG))  # the long section was cut


@pytest.mark.skipif(not handbooks(), reason="set NANOAGENT_HANDBOOK_DIR to a folder of handbooks")
@pytest.mark.parametrize("path", handbooks(), ids=lambda p: p.stem)
def test_structured_chunks_are_byte_identical_on_real_handbooks(path: Path) -> None:
    text = path.read_text()
    assert chunking.structured_chunks(text) == bm25_pieces(text)
    assert chunking.tokens(text) == policy_select.tokens(text)


def test_cms_idf_is_zero_for_a_term_in_no_chunk_and_none_counts_hits() -> None:
    chunk_terms = [{"air", "fare"}, {"air", "hotel"}]
    exact = policy_vanilla.weights({"air", "fare", "absent"}, chunk_terms, "cms", True, 580)
    assert exact == {"air": 0.0, "fare": math.log(2), "absent": 0.0}
    assert policy_vanilla.weights({"air", "fare", "absent"}, chunk_terms, "cms", False, 580) == exact  # 272 x 5 is exact here
    assert policy_vanilla.weights({"air"}, chunk_terms, "none", False, 580) == {"air": 1.0}


def test_a_word_cap_lowers_the_budget() -> None:
    _, capped = policy_vanilla.chosen(LONG, INSTRUCTIONS, fraction=1.0, max_words=40)
    _, free = policy_vanilla.chosen(LONG, INSTRUCTIONS, fraction=1.0)
    assert set(capped) < set(free)
