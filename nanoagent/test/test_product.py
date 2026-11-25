"""The product bot (nanoagent.plugin.policy_product): sketch-BM25's structures and guarantees, its exact twin
``bm25x``, the budget rule, the light stopword strip, the ladder rungs L9-L14 and the reasoning passthrough (VS1-VS3 are
test_variant's)."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from nanoagent.plugin import bots, policy_product, policy_select
from nanoagent.plugin.bots import PRESETS, resolve
from nanoagent.plugin.policy import POLICY_FILE, policy_text
from nanoagent.plugin.policy_product import count_min, strip_light
from nanoagent.plugin.sketches import CountMinSketch, ExactCounter
from nanoagent.plugin.sketches_variant import ConservativeCountMin
from test_bots import harness
from test_policy_switch import HANDBOOK, INSTRUCTIONS, SYSTEM
from test_vanilla import LONG, handbooks

PRODUCT = {"idf": "cms", "dedup": "minhash", "twin": "sketch", "chunker": "structured", "sketch_opts": policy_product.OPTIONS, "seed": 580}


# -- the structures ---------------------------------------------------------------------------------- #


@pytest.mark.parametrize("conservative", [False, True])
def test_count_min_by_size_never_underestimates_and_conservative_is_tighter(conservative: bool) -> None:
    rng = random.Random(2)
    sketch, plain, exact = count_min(64, 4, "t", conservative), count_min(64, 4, "t"), ExactCounter()
    for _ in range(3000):
        x, c = f"w{int(rng.paretovariate(1.1))}", rng.randint(1, 3)
        for s in (sketch, plain, exact):
            s.add(x, c)
    assert (len(sketch.rows), len(sketch.rows[0]), sketch.total) == (4, 64, exact.total)
    assert all(exact.query(x) <= sketch.query(x) <= plain.query(x) for x in exact.counts)
    assert all(a <= b for r, q in zip(sketch.rows, plain.rows) for a, b in zip(r, q))
    assert isinstance(sketch, ConservativeCountMin if conservative else CountMinSketch)


def test_the_tf_sketch_at_8_counters_per_token_is_almost_always_exact() -> None:
    rng, wrong, queries = random.Random(3), 0, 0
    for i in range(200):
        doc = [f"t{int(rng.paretovariate(1.0))}" for _ in range(rng.randint(5, 120))]
        tf = count_min(policy_product.TF_PER_TOKEN * len(doc), policy_product.TF_DEPTH, f"tf{i}", True)
        for t in doc:
            tf.add(t)
        wrong += sum(tf.query(t) != doc.count(t) for t in set(doc))
        queries += len(set(doc))
    assert wrong / queries <= 10 * 8**-4  # P[tf^ != tf] <= 8^-4 per query, with slack for 200 small samples


# -- the selector and its exact twin ------------------------------------------------------------------ #


@pytest.mark.parametrize("dedup", ["none", "minhash"])
@pytest.mark.parametrize("text", [HANDBOOK, LONG])
def test_bm25x_is_policy_selects_bm25_at_the_same_budget(text: str, dedup: str) -> None:
    for fraction in (0.2, 0.35, 0.5):
        twin = policy_product.select(text, INSTRUCTIONS, **{**PRODUCT, "twin": "exact", "dedup": dedup}, fraction=fraction)
        assert twin == policy_select.bm25_frac(text, INSTRUCTIONS, fraction)


@pytest.mark.parametrize("path", handbooks(), ids=lambda p: p.stem)
def test_bm25x_is_byte_identical_to_bm25_frac_on_real_handbooks(path: Path) -> None:
    text = path.read_text()
    assert policy_product.select(text, INSTRUCTIONS, **{**PRODUCT, "twin": "exact", "dedup": "none"}) == policy_select.bm25_frac(text, INSTRUCTIONS, 0.35)


def test_the_twin_scores_are_bm25_and_the_sketch_scores_bound_present_terms() -> None:
    parts = policy_product.units(LONG, "structured")
    docs = [policy_product.tokens(t) for t, _, _ in parts]
    query = set(policy_product.tokens(policy_select.in_scope(INSTRUCTIONS)))
    exact = policy_product.scores(query, docs, "cms", True, (), 580)
    assert exact == pytest.approx(policy_select.bm25_scores(query, docs), abs=1e-12)
    sketch = policy_product.scores(query, docs, "cms", False, policy_product.OPTIONS, 580)
    assert all(s > 0 for s, e in zip(sketch, exact) if e > 0)  # no false negatives: a scored chunk stays scored


def test_the_word_cap_and_the_fraction_both_bound_the_budget() -> None:
    title = len(policy_select.first_line(LONG).split())
    for fraction, cap in ((0.35, None), (0.35, 40), (0.6, 25)):
        out = policy_product.select(LONG, INSTRUCTIONS, **PRODUCT, fraction=fraction, max_words=cap)
        budget = min(int(fraction * len(LONG.split())), cap or 10**9)
        assert len(out.split()) <= 1.05 * budget + title and out.startswith(policy_select.first_line(LONG))


def test_the_selection_is_deterministic_and_independent_of_pythonhashseed() -> None:
    code = (
        "from nanoagent.plugin import policy_product as p; import hashlib; "
        f"print(hashlib.sha256(p.select({LONG!r}, {INSTRUCTIONS!r}, sketch_opts=p.OPTIONS).encode()).hexdigest())"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env={**env, "PYTHONHASHSEED": str(s)}).stdout for s in range(4)}
    assert len(outs) == 1, outs


# -- the light stopword strip ------------------------------------------------------------------------- #


def test_the_light_strip_removes_only_listed_words_outside_guarded_lines() -> None:
    text = "1. Air Travel\nThe traveller must book the lowest fare that is available, unless the trip is not urgent.\nFares of $500 or more need the approval of a VP."
    out = strip_light(text).split("\n")
    assert out[0] == "1. Air Travel" and out[2] == text.split("\n")[2]  # a heading and a line with a digit are kept whole
    assert out[1] == "traveller must book lowest fare available, unless trip not urgent."
    assert len(policy_product.LIGHT) == 27 and not policy_product.LIGHT & {"not", "no", "must", "only", "unless", "if", "or", "and", "never"}


# -- the bots ------------------------------------------------------------------------------------------ #


@pytest.fixture
def plugin(tmp_path: Path) -> Path:
    (tmp_path / POLICY_FILE).write_text(LONG)
    return tmp_path


def test_product_and_optimal_run_and_differ_only_in_the_twin(plugin: Path) -> None:
    for name, twin in (("product", "sketch"), ("optimal", "exact")):
        c = resolve(harness(bot=name))
        bots.check_runnable(c, "agent")
        text = policy_text(plugin, bots.method(c), SYSTEM)
        assert text == strip_light(policy_product.select(LONG, SYSTEM, **{**PRODUCT, "twin": twin}, fraction=0.35, max_words=1500))
        assert c.stop_on_submit and c.reasoning_effort is None  # PA4 is added only if probe G4 admits it


# Ladder rungs L9-L14 (each a preset plus one override from the rung before) and S-c's leave-one-out from product.
LADDER = {
    "L9": {"bot": "variant", "selector": "sketch_bm25"},
    "L10": {"bot": "variant", "selector": "sketch_bm25", "stop_on_submit": True},
    "L11": {"bot": "product", "stopwords": "none"},
    "L13": {"bot": "product"},
    "L14": {"bot": "optimal"},
    "-PA1": {"bot": "product", "stop_on_submit": False},
    "-PA2": {"bot": "product", "budget": "frac:0.35"},
    "-PA3": {"bot": "product", "stopwords": "none"},
    "-dedup": {"bot": "product", "dedup": "none"},
    "+PA4": {"bot": "product", "reasoning_effort": "low"},
}


@pytest.mark.parametrize("rung", sorted(LADDER))
def test_each_product_rung_runs(rung: str, plugin: Path) -> None:
    c = resolve(harness(**LADDER[rung]))
    bots.check_runnable(c, "agent")
    assert policy_text(plugin, bots.method(c), SYSTEM).startswith(policy_select.first_line(LONG))


def test_product_is_variant_with_the_selector_and_pa1_to_pa3() -> None:
    changes = {"selector": "sketch_bm25", "stop_on_submit": True, "budget": "frac:0.35,words:1500", "stopwords": "light"}
    assert PRESETS["product"] == dataclasses.replace(PRESETS["variant"], **changes)
    assert resolve(harness(**LADDER["L10"], budget="frac:0.35,words:1500")) == resolve(harness(**LADDER["L11"]))


@pytest.mark.parametrize(
    ("fields", "error", "message"),
    [
        ({"selector": "sketch_bm25"}, ValueError, "'sketch_bm25' needs idf, dedup, twin, chunker, budget, seed, stopwords"),
        ({"bot": "product", "idf": "none"}, ValueError, "needs a document frequency"),
        ({"bot": "product", "sketch_opts": ["double_hash"]}, NotImplementedError, "run on bloom_chunk"),
        ({"bot": "product", "backup": "learned"}, NotImplementedError, "run on bloom_chunk"),
    ],
)
def test_a_sketch_bm25_it_cannot_run_is_refused(fields: dict, error: type, message: str) -> None:
    with pytest.raises(error, match=message):
        bots.check_runnable(resolve(harness(**fields)), "agent")


def test_bm25_with_a_word_cap_is_bm25_at_the_smaller_budget(plugin: Path) -> None:
    c = resolve(harness(selector="bm25", budget="frac:0.5,words:30"))
    bots.check_runnable(c, "agent")
    assert policy_text(plugin, bots.method(c), SYSTEM) == policy_select.bm25(LONG, SYSTEM, budget=min(int(0.5 * len(LONG.split())), 30))


def test_vanilla_takes_the_light_strip_too(plugin: Path) -> None:
    c = resolve(harness(bot="vanilla", stopwords="light"))
    bots.check_runnable(c, "agent")
    plain = policy_text(plugin, bots.method(resolve(harness(bot="vanilla"))), SYSTEM)
    assert policy_text(plugin, bots.method(c), SYSTEM) == strip_light(plain) != plain


# -- reasoning effort reaches the OpenRouter request only when set ----------------------------------------- #


def test_reasoning_effort_reaches_the_openrouter_request_only_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from nanoagent.cli.plugin import _with_reasoning
    from nanoagent.runtime.model import Model
    from test_billing import PIN, StreamingCompletions, _openrouter_config

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    model = _openrouter_config(extra_body=[PIN]).runtime()
    assert _with_reasoning(model, PRESETS["product"].reasoning_effort) is model
    sent = {}
    for effort in (None, "low"):
        backend = Model.from_config(_with_reasoning(model, effort))._backend
        completions = StreamingCompletions()
        backend._client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})})()
        asyncio.run(Model(backend).query([{"role": "user", "content": "hi"}], [], on_delta=lambda kind, text: None))
        [sent[effort]] = completions.requests
    assert "reasoning" not in sent[None]["extra_body"] and sent["low"]["extra_body"]["reasoning"] == {"effort": "low"}
    assert sent["low"]["extra_body"]["provider"] == sent[None]["extra_body"]["provider"]
