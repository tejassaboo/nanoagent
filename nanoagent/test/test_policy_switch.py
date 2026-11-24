from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nanoagent.harness.nanoagent import NanoAgentConfig
from nanoagent.plugin import bloom_filter, policy_select
from nanoagent.plugin.policy import METHODS, POLICY_FILE, TITLE, policy_text, with_policy
from nanoagent.plugin.tasks import decide_prompt, task_prompt

HANDBOOK = """Acme Corporation Travel Policy

1. Purpose

This policy sets out how employees book and pay for business travel on behalf of the company.

2. Air Travel

Employees must book the lowest logical airfare in economy class for every flight under six hours.

Business class is allowed only for a flight of more than six hours, with approval from a vice president.

Book air travel at least fourteen days before departure; a later booking needs a written justification.

3. Lodging

The nightly lodging cap is $250 in major cities and $180 elsewhere, taxes included.

Employees may not stay at casino resorts or properties on the non-reimbursable list.

4. Meals

Meals are reimbursed at the daily per diem of the destination; alcohol is never reimbursed.

5. Information Security

Laptops must be encrypted and never left unattended in a vehicle or a hotel lobby.

6. Gifts and Entertainment

Gifts to government officials are prohibited regardless of their value.
"""

INSTRUCTIONS = """# Book a trip under the Acme handbook

## Rules

- C1 Pick the lowest logical airfare in economy class when the flight is under six hours.
- C2 Business class only when the flight is over six hours, with vice president approval.
- C3 Advance booking: book at least fourteen days before departure.

## Rules not in scope

Lodging caps, meals and per diem are handled by another task.
"""

TASK = SimpleNamespace(description="Book a flight under the Acme travel handbook", instructions=INSTRUCTIONS)


def agent_prompt(plugin_name: str, task: SimpleNamespace) -> str:
    """The system prompt NanoAgent gives an agent task (task_prompt reads only these fields)."""
    return task_prompt(SimpleNamespace(name=plugin_name), task, {})[0]  # type: ignore[arg-type]


SYSTEM = agent_prompt("Acme Travel", TASK)


@pytest.fixture
def plugin(tmp_path: Path) -> Path:
    (tmp_path / POLICY_FILE).write_text(HANDBOOK)
    return tmp_path


def section(text: str) -> str:
    return f"{SYSTEM}\n\n## {TITLE}\n\n{text.strip()}"


def test_the_switch_is_off_by_default() -> None:
    assert NanoAgentConfig.__dataclass_fields__["policy"].default == "off"


def test_off_leaves_the_prompt_byte_identical(plugin: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    assert with_policy(SYSTEM, plugin, "off") is SYSTEM
    # off needs no policy.txt
    assert with_policy(SYSTEM, tmp_path_factory.mktemp("bare"), "off") == SYSTEM


def test_full_appends_the_whole_handbook(plugin: Path) -> None:
    assert with_policy(SYSTEM, plugin, "full") == section(HANDBOOK)


@pytest.mark.parametrize(("policy", "level"), [("bloom", "flight_paragraphs+curated"), ("scoped", "travel_paragraphs+curated")])
def test_bloom_methods_append_the_compressed_handbook(plugin: Path, policy: str, level: str) -> None:
    compressed, _ = bloom_filter.compress(HANDBOOK, level)
    assert compressed.strip()
    assert with_policy(SYSTEM, plugin, policy) == section(compressed)
    assert policy_text(plugin, policy, "any task") == compressed  # the task text does not change them


def paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n\n") if p.strip()]


def test_rule_quota_keeps_the_chunks_each_rule_needs(plugin: Path) -> None:
    text = policy_text(plugin, "rule_quota", SYSTEM)
    assert paragraphs(text)[0] == "Acme Corporation Travel Policy"
    assert "lowest logical airfare in economy class" in text  # C1
    assert "Business class is allowed only" in text  # C2
    assert "at least fourteen days before departure" in text  # C3
    for unrelated in ("nightly lodging cap", "alcohol", "encrypted", "government officials"):
        assert unrelated not in text
    assert all(p in HANDBOOK for p in paragraphs(text))  # verbatim chunks
    assert with_policy(SYSTEM, plugin, "rule_quota") == section(text)


def test_bm25_keeps_the_best_ranked_chunks_within_its_budget(plugin: Path) -> None:
    text = policy_text(plugin, "bm25", SYSTEM)
    assert paragraphs(text)[0] == "Acme Corporation Travel Policy"
    assert "lowest logical airfare in economy class" in text
    for unrelated in ("alcohol", "encrypted", "government officials"):
        assert unrelated not in text
    assert all(p in HANDBOOK for p in paragraphs(text))
    budget = policy_select.word_budget(HANDBOOK)
    assert len(text.split()) - len(paragraphs(text)[0].split()) <= policy_select.SLACK * budget
    assert with_policy(SYSTEM, plugin, "bm25") == section(text)


@pytest.mark.parametrize("policy", ["rule_quota", "bm25"])
def test_task_methods_read_only_the_task_portion_of_the_prompt(plugin: Path, policy: str) -> None:
    # The same instructions behind another plugin's name and description select the same text.
    other = agent_prompt("Hotel Lodging Meals", SimpleNamespace(description="Lodging caps per diem", instructions=INSTRUCTIONS))
    assert policy_text(plugin, policy, SYSTEM) == policy_text(plugin, policy, INSTRUCTIONS) == policy_text(plugin, policy, other)


@pytest.mark.parametrize("policy", ["rule_quota", "bm25"])
def test_task_methods_are_deterministic(plugin: Path, policy: str) -> None:
    assert len({policy_text(plugin, policy, SYSTEM) for _ in range(3)}) == 1


def test_task_portion_finds_the_instructions_in_every_plugin_prompt() -> None:
    assert policy_select.task_portion(SYSTEM) == INSTRUCTIONS
    decide = decide_prompt(SimpleNamespace(name="Acme Travel"), TASK)  # type: ignore[arg-type]
    assert policy_select.task_portion(decide) == INSTRUCTIONS
    assert policy_select.task_portion(INSTRUCTIONS) == INSTRUCTIONS  # a bare task is its own task portion


def test_rules_are_the_labelled_items_in_scope() -> None:
    rules = policy_select.rules_of(SYSTEM)
    assert [r.split()[1] for r in rules] == ["C1", "C2", "C3"]


def test_every_method_takes_the_handbook_and_the_task() -> None:
    assert set(METHODS) == {"bloom", "scoped", "rule_quota", "bm25", "bm25_frac"}


@pytest.mark.parametrize("folder", ["with_handbook", "without_handbook"])
def test_an_unknown_policy_is_an_error(plugin: Path, tmp_path_factory: pytest.TempPathFactory, folder: str) -> None:
    root = plugin if folder == "with_handbook" else tmp_path_factory.mktemp("bare")
    with pytest.raises(ValueError, match="unknown policy 'bloomy'"):
        with_policy(SYSTEM, root, "bloomy")


@pytest.mark.parametrize("policy", ["full", "bloom", "scoped", "rule_quota", "bm25"])
def test_a_missing_policy_file_is_an_error(tmp_path: Path, policy: str) -> None:
    with pytest.raises(FileNotFoundError, match=POLICY_FILE):
        with_policy(SYSTEM, tmp_path, policy)


def test_yaml_off_parses_as_false_and_still_means_off(tmp_path):
    from nanoagent.plugin.policy import with_policy

    (tmp_path / "policy.txt").write_text("Fly economy.")
    assert with_policy("PROMPT", tmp_path, False) == "PROMPT"


def test_bm25_frac_takes_a_fraction(tmp_path):
    import pytest

    from nanoagent.plugin.policy import policy_text

    (tmp_path / "policy.txt").write_text("\n\n".join(f"Section {i}. Airfare rule {i} about economy fares." for i in range(40)))
    small = policy_text(tmp_path, "bm25_frac:0.1", "Book an economy airfare.")
    large = policy_text(tmp_path, "bm25_frac:0.5", "Book an economy airfare.")
    assert len(small.split()) < len(large.split())
    for bad in ("bm25_frac:0", "bm25_frac:1.5", "bm25_frac:x"):
        with pytest.raises(ValueError):
            policy_text(tmp_path, bad, "task")
