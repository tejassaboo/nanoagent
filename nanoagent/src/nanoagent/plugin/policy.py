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


def policy_text(root: Path, policy: str, task_text: str) -> str:
    """The policy text for ``policy`` ("" for ``off``), read from the plugin folder ``root``, for the task ``task_text``."""
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
    handbook_path = root / POLICY_FILE
    if not handbook_path.is_file():
        raise FileNotFoundError(f"policy={policy!r} needs {handbook_path}")
    handbook = handbook_path.read_text()
    if fraction is not None:
        return policy_select.bm25_frac(handbook, task_text, fraction)
    return handbook if policy == "full" else METHODS[policy](handbook, task_text)


def with_policy(system_prompt: str, root: Path, policy: str) -> str:
    """``system_prompt`` with the policy section added, unchanged when ``policy`` is ``off``."""
    text = policy_text(root, policy, system_prompt)
    return f"{system_prompt}\n\n## {TITLE}\n\n{text.strip()}" if text.strip() else system_prompt
