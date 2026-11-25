"""``harness.bot``: one name that expands into every setting of how a plugin agent sees its handbook.

A bot is a preset (:data:`PRESETS`) of :class:`Components`. Each component field on
:class:`~nanoagent.harness.nanoagent.NanoAgentConfig` that is set (not null) replaces the preset's
value, so every ablation rung is a preset plus one override, such as
``-- harness.bot=vanilla harness.twin=exact``. ``harness.policy`` is the older name for
``harness.selector`` and still works; setting both is an error. A config that sets none of these
resolves to ``off``, which is the upstream agent: no handbook text and no other change.

:func:`resolve` checks every name and value, and :func:`check_runnable` refuses, before any model
call, a component this NanoAgent defines but does not run yet. Nothing falls back to another setting.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanoagent.harness.nanoagent import NanoAgentConfig
    from nanoagent.profiler import RunProfile

# The selectors of the first switch (nanoagent.plugin.policy), which run as they always have.
LEGACY_SELECTORS = ("off", "full", "bloom", "scoped", "rule_quota", "bm25", "bm25_frac")
# Selectors built from randomized sketches; each structure has an exact twin (`twin: exact`).
SKETCH_SELECTORS = ("bloom_chunk", "sketch_bm25")
CHOICES: dict[str, tuple[str, ...]] = {
    "idf": ("none", "exact", "cms"),
    "dedup": ("none", "minhash"),
    "twin": ("sketch", "exact"),
    # probe_sizing = VS1, conservative_update = VS2, one_permutation = VS3, double_hash = N1
    "sketch_opts": ("probe_sizing", "conservative_update", "one_permutation", "double_hash"),
    "chunker": ("plain", "structured"),
    "backup": ("none", "learned"),
    "stopwords": ("none", "light"),
    "reasoning_effort": ("low", "medium", "high"),
    "query": ("instructions", "instructions+input"),
}
# Components that only a sketch selector reads.
SKETCH_ONLY = ("idf", "dedup", "twin", "sketch_opts", "chunker", "backup", "stopwords", "seed")
BUDGET = re.compile(r"\Afrac:(?P<frac>\d+(?:\.\d+)?)(?:,words:(?P<words>\d+))?\Z")
_OFF = ("off", "false", "none", "")
# Selectors defined here but not run yet, with the package that adds them.
NOT_YET = {"sketch_bm25": "WP6"}
# The components bloom_chunk reads (each must be set), and values it does not run yet, with their package.
BLOOM_CHUNK_READS = ("idf", "dedup", "twin", "chunker", "budget", "seed")
NOT_YET_IN_BLOOM_CHUNK = {"sketch_opts": "WP5", "backup": "WP5", "stopwords": "WP6"}


@dataclass(frozen=True)
class Components:
    """Everything a bot decides. ``None`` means the component does not apply (or, for
    ``reasoning_effort``, that nothing is sent)."""

    selector: str = "off"
    idf: str | None = None
    dedup: str | None = None
    twin: str | None = None
    sketch_opts: tuple[str, ...] | None = None
    chunker: str | None = None
    budget: str | None = None  # frac:<f> or frac:<f>,words:<n> (the smaller of the two)
    backup: str | None = None
    stopwords: str | None = None
    stop_on_submit: bool = False
    reasoning_effort: str | None = None
    seed: int | None = None
    query: str = "instructions"


_VANILLA = Components(
    selector="bloom_chunk",
    idf="cms",
    dedup="minhash",
    twin="sketch",
    sketch_opts=(),
    chunker="structured",
    budget="frac:0.35",
    backup="none",
    stopwords="none",
    seed=580,
)
# Ladder rung V-a: per-chunk Bloom filters scored by hit count. V-b is V-a + idf cms (or vanilla + dedup none),
# V-c is vanilla, T is vanilla + twin exact and C0 is vanilla + chunker plain: each a preset plus one override.
_VANILLA_A = dataclasses.replace(_VANILLA, idf="none", dedup="none")
_VARIANT = dataclasses.replace(_VANILLA, sketch_opts=("probe_sizing", "conservative_update", "one_permutation"))
_PRODUCT = dataclasses.replace(_VARIANT, selector="sketch_bm25", budget="frac:0.35,words:1500", stopwords="light", stop_on_submit=True)
PRESETS: dict[str, Components] = {
    "off": Components(),
    "full": Components(selector="full"),
    "full_plus": Components(selector="full", stop_on_submit=True),
    "vanilla": _VANILLA,
    "vanilla_a": _VANILLA_A,
    "variant": _VARIANT,
    "product": _PRODUCT,
    # The pre-registered pick, initially bm25x: product with every sketch swapped for its exact twin.
    "optimal": dataclasses.replace(_PRODUCT, twin="exact"),
}


def normalise(name: Any) -> str:
    """A bot or selector name as written in YAML: a bare ``off`` arrives as False or 'False'."""
    if name is None or name is False or str(name).lower() in _OFF:
        return "off"
    return "full_plus" if name == "full+" else str(name)


def _check_selector(selector: str) -> None:
    if selector.startswith("bm25_frac:"):
        try:
            fraction = float(selector.split(":", 1)[1])
        except ValueError:
            fraction = -1.0
        if not 0 < fraction <= 1:
            raise ValueError(f"selector {selector!r}: the fraction after bm25_frac: must be in (0, 1]")
    elif selector not in (*LEGACY_SELECTORS, *SKETCH_SELECTORS):
        raise ValueError(f"unknown selector {selector!r}: use one of {[*LEGACY_SELECTORS, *SKETCH_SELECTORS]} or bm25_frac:<fraction>")


def _check(c: Components) -> None:
    _check_selector(c.selector)
    for name, allowed in CHOICES.items():
        value = getattr(c, name)
        for item in value if isinstance(value, tuple) else () if value is None else (value,):
            if item not in allowed:
                raise ValueError(f"harness.{name}={item!r}: use one of {list(allowed)}")
    if c.budget is not None:
        match = BUDGET.match(c.budget)
        if match is None or not 0 < float(match["frac"]) <= 1 or (match["words"] is not None and int(match["words"]) < 1):
            raise ValueError(f"harness.budget={c.budget!r}: use frac:<fraction in (0, 1]> or frac:<fraction>,words:<n>")
    if c.selector not in SKETCH_SELECTORS:
        stray = [name for name in SKETCH_ONLY if getattr(c, name) is not None]
        if stray:
            raise ValueError(f"selector {c.selector!r} reads no {', '.join(stray)}: those apply to {list(SKETCH_SELECTORS)}")
        if c.budget is not None and c.selector != "bm25":
            raise ValueError(f"selector {c.selector!r} takes no budget: budget applies to bm25 and {list(SKETCH_SELECTORS)}")


def resolve(h: NanoAgentConfig) -> Components:
    """The bot ``h`` describes: its preset, then each component ``h`` sets explicitly."""
    bot = normalise(h.bot)
    if bot not in PRESETS:
        raise ValueError(f"unknown harness.bot {h.bot!r}: use one of {sorted(PRESETS)} (or full+)")
    explicit: dict[str, Any] = {f.name: getattr(h, f.name) for f in dataclasses.fields(Components) if getattr(h, f.name) is not None}
    if normalise(h.policy) != "off":
        if "selector" in explicit:
            raise ValueError("harness.policy is the older name of harness.selector: set one of them, not both")
        explicit["selector"] = h.policy
    if "selector" in explicit:
        explicit["selector"] = normalise(explicit["selector"])
    if explicit.get("sketch_opts") is not None:
        explicit["sketch_opts"] = tuple(explicit["sketch_opts"])
    components = dataclasses.replace(PRESETS[bot], **explicit)
    _check(components)
    return components


def legacy_policy(c: Components) -> str:
    """The ``nanoagent.plugin.policy`` method name for a legacy selector (``bm25`` with a budget is ``bm25_frac``)."""
    if c.selector == "bm25" and c.budget is not None:
        match = BUDGET.match(c.budget)
        assert match is not None  # _check
        if match["words"] is not None:
            raise NotImplementedError("harness.budget words:<n> is not implemented for selector bm25 yet (WP6)")
        return f"bm25_frac:{match['frac']}"
    return c.selector


def method(c: Components) -> str | Callable[[str, str], str]:
    """What :func:`nanoagent.plugin.policy.with_policy` runs: a legacy method name, or for ``bloom_chunk``
    the vanilla selector (:func:`nanoagent.plugin.policy_vanilla.select`) with ``c``'s components."""
    if c.selector != "bloom_chunk":
        return legacy_policy(c)
    from nanoagent.plugin import policy_vanilla

    match = BUDGET.match(c.budget or "")
    assert match is not None and c.seed is not None  # check_runnable
    words = None if match["words"] is None else int(match["words"])
    options = {"idf": c.idf, "dedup": c.dedup, "twin": c.twin, "chunker": c.chunker, "fraction": float(match["frac"]), "max_words": words}
    return functools.partial(policy_vanilla.select, **options, seed=c.seed)


def check_runnable(c: Components, runner: str) -> None:
    """Refuse, before any model call, a bot this NanoAgent cannot run as configured."""
    if c.selector in NOT_YET:
        raise NotImplementedError(f"harness.selector={c.selector!r} is defined but not implemented yet ({NOT_YET[c.selector]})")
    if c.selector == "bloom_chunk":
        unset = [name for name in BLOOM_CHUNK_READS if getattr(c, name) is None]
        if unset:
            raise ValueError(f"selector 'bloom_chunk' needs {', '.join(unset)}: set harness.bot=vanilla (or vanilla_a), or set each")
        for name, package in NOT_YET_IN_BLOOM_CHUNK.items():
            if getattr(c, name) not in (None, (), "none"):
                raise NotImplementedError(f"harness.{name}={getattr(c, name)!r} is defined but not implemented yet for bloom_chunk ({package})")
    if c.query == "instructions+input" and runner != "agent":
        raise ValueError("harness.query instructions+input needs an agent task: a decide task has no single task input message")
    legacy_policy(c)


def is_upstream(h: NanoAgentConfig) -> bool:
    """True when ``h`` sets none of the switch's fields, so the run must equal the upstream agent."""
    names = [f.name for f in dataclasses.fields(Components)] + ["hosted_token_limit"]
    return normalise(h.bot) == "off" and normalise(h.policy) == "off" and all(getattr(h, name) is None for name in names)


def hosted_cap_stop(profile: RunProfile | None, cap: int | None) -> int | None:
    """The step a run capped at ``cap`` total tokens would have been stopped before (the loop checks
    the cap before each step), or None when it would have finished."""
    if profile is None or cap is None:
        return None
    total = 0
    for index, step in enumerate(profile.steps):
        if total >= cap:
            return index + 1
        total += step.usage.get("total_tokens", 0)
    return None


def provenance(c: Components, h: NanoAgentConfig, llm: Any, handbook_sha256: str | None) -> dict[str, Any]:
    """What a run records about its bot: the preset, the resolved components, the shared limits and
    a hash of all of them with the model settings (equal hashes = the same arm)."""
    arm = {
        "components": dataclasses.asdict(c),
        "limits": {name: getattr(h, name) for name in ("max_steps", "cost_limit", "token_limit", "hosted_token_limit", "context_window")},
        "llm": dataclasses.asdict(llm) if dataclasses.is_dataclass(llm) else dict(llm),
    }
    digest = hashlib.sha256(json.dumps(arm, sort_keys=True, default=str).encode()).hexdigest()
    return {"bot": normalise(h.bot), **arm, "config_sha256": digest, "handbook_sha256": handbook_sha256}
