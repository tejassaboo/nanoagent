"""The ``harness.policy`` switch: which company policy text a plugin agent sees.

``off`` adds nothing (the baseline), ``full`` adds the plugin's whole ``policy.txt``, and any other
value names a method in :data:`METHODS` that compresses ``policy.txt`` before it is added. A method
also gets the system prompt, so it can keep what the task needs (``bloom`` and ``scoped`` ignore it).
The text goes at the end of the system prompt, so a run differs from its baseline in this one setting.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from nanoagent.plugin import bloom_filter, policy_select

POLICY_FILE = "policy.txt"
BUNDLED_POLICY_FILE = "src/plugin/data/policy.txt"
TITLE = "Company travel policy"


def _bloom(level: str) -> Callable[[str, str], str]:
    return lambda handbook, task_text: bloom_filter.compress(handbook, level)[0]


# Method name -> function from (the full handbook, the task text) to the text shown. A new method is one entry.
METHODS: dict[str, Callable[[str, str], str]] = {
    "bloom": _bloom("flight_paragraphs+curated"),
    "scoped": _bloom("travel_paragraphs+curated"),
    "rule_quota": policy_select.rule_quota,
    "bm25": policy_select.bm25,
    "bm25_frac": policy_select.bm25_frac,  # ablation: budget = 35% of the handbook, no flight vocabulary
}


def policy_text(root: Path, policy: str | Callable[[str, str], str], task_text: str) -> str:
    """The policy text for ``policy`` ("" for ``off``), read from the plugin folder ``root``, for the task ``task_text``.
    ``policy`` may also be a configured selector, a function of (handbook, task text) (``nanoagent.plugin.bots.method``)."""
    if callable(policy):
        return policy(handbook_path(root, "harness.selector").read_text(), task_text)
    # YAML 1.1 reads a bare `off` as false, so `policy: off` in a config arrives as False.
    if policy is False or policy is None or str(policy).lower() in ("off", "false", "none", ""):
        return ""
    fraction = None
    if isinstance(policy, str) and policy.startswith("bm25_frac:"):  # e.g. bm25_frac:0.25 = budget 25% of the handbook
        try:
            fraction = float(policy.split(":", 1)[1])
        except ValueError:
            fraction = -1.0
        if not 0 < fraction <= 1:
            raise ValueError(f"policy {policy!r}: the fraction after bm25_frac: must be in (0, 1]")
    elif policy != "full" and policy not in METHODS:
        raise ValueError(f"unknown policy {policy!r}: use off, full, bm25_frac:<fraction> or one of {sorted(METHODS)}")
    handbook = handbook_path(root, policy).read_text()
    if fraction is not None:
        return policy_select.bm25_frac(handbook, task_text, fraction)
    return handbook if policy == "full" else METHODS[policy](handbook, task_text)


def handbook_path(root: Path, policy: str) -> Path:
    """The plugin's handbook: ``policy.txt`` at its root, else bundled as ``src/plugin/data/policy.txt``
    (a layout stock ``nanoagent plugin inspect`` accepts)."""
    for path in (root / POLICY_FILE, root / BUNDLED_POLICY_FILE):
        if path.is_file():
            return path
    raise FileNotFoundError(f"policy={policy!r} needs {root / POLICY_FILE} or {root / BUNDLED_POLICY_FILE}")


def selection_query(system_prompt: str, task_input: str, query: str) -> str:
    """What a method selects for: the system prompt (``instructions``), or the task's input message
    and its instructions (``instructions+input``). The input goes first, so the instructions' last
    section, which a method may cut to the end of the text ("Rules not in scope"), cannot take it."""
    if query == "instructions":
        return system_prompt
    if query == "instructions+input":
        return f"{task_input}\n\n{policy_select.task_portion(system_prompt)}"
    raise ValueError(f"unknown query {query!r}: use instructions or instructions+input")


def with_policy(system_prompt: str, root: Path, policy: str | Callable[[str, str], str], query_text: str | None = None) -> str:
    """``system_prompt`` with the policy section added, unchanged when ``policy`` is ``off``.
    A method selects for ``query_text`` (default: the system prompt itself)."""
    text = policy_text(root, policy, system_prompt if query_text is None else query_text)
    return f"{system_prompt}\n\n## {TITLE}\n\n{text.strip()}" if text.strip() else system_prompt
