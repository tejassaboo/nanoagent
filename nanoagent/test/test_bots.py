"""``harness.bot`` presets, their resolution and the run seams they use (nanoagent.plugin.bots)."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from nanoagent.harness.nanoagent import NanoAgentConfig
from nanoagent.plugin import bots
from nanoagent.plugin.bots import PRESETS, Components, resolve
from nanoagent.plugin.policy import BUNDLED_POLICY_FILE, POLICY_FILE, TITLE, handbook_path, policy_text, selection_query, with_policy
from test_policy_switch import HANDBOOK, SYSTEM


def harness(**fields) -> NanoAgentConfig:
    """A NanoAgentConfig as a run config loads it: the loop limits set, plus ``fields``."""
    base = dict(max_steps=6, cost_limit=None, token_limit=None, context_window=None, hooks=[], skills=None, context_files=[], events=None)
    return NanoAgentConfig(**{**base, **fields})


def test_a_config_with_no_switch_field_is_the_upstream_agent() -> None:
    h = harness()
    assert resolve(h) == PRESETS["off"] == Components()
    assert bots.is_upstream(h)
    assert not bots.is_upstream(harness(hosted_token_limit=65536))
    assert not bots.is_upstream(harness(bot="full"))


@pytest.mark.parametrize("written", ["off", False, "False", "false", None, ""])
def test_yaml_off_in_any_form_is_off(written) -> None:
    assert bots.normalise(written) == "off"
    assert resolve(harness(bot=written)) == Components()
    assert resolve(harness(selector=written)) == Components()


def test_every_preset_resolves_and_full_plus_has_two_spellings() -> None:
    for name in ("off", "full", "full_plus", "vanilla", "variant", "product", "optimal"):
        assert resolve(harness(bot=name)) == PRESETS[name]
    assert resolve(harness(bot="full+")) == PRESETS["full_plus"]
    assert PRESETS["full_plus"] == dataclasses.replace(PRESETS["full"], stop_on_submit=True)
    assert PRESETS["optimal"] == dataclasses.replace(PRESETS["product"], twin="exact")
    assert PRESETS["vanilla"].chunker == PRESETS["variant"].chunker == "structured"


@pytest.mark.parametrize(
    ("bot", "override", "value"),
    [("vanilla", "twin", "exact"), ("vanilla", "sketch_opts", ["probe_sizing"]), ("product", "stop_on_submit", False), ("full", "query", "instructions+input")],
)
def test_a_rung_is_a_preset_plus_one_override(bot: str, override: str, value) -> None:
    resolved = resolve(harness(bot=bot, **{override: value}))
    expected = tuple(value) if isinstance(value, list) else value
    assert resolved == dataclasses.replace(PRESETS[bot], **{override: expected})


def test_policy_is_the_older_name_of_selector() -> None:
    assert resolve(harness(policy="bm25")) == Components(selector="bm25")
    assert resolve(harness(policy="bm25")) == resolve(harness(selector="bm25"))
    assert resolve(harness(bot="full", policy="rule_quota")).selector == "rule_quota"
    with pytest.raises(ValueError, match="set one of them, not both"):
        resolve(harness(policy="bm25", selector="full"))


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"bot": "vanila"}, "unknown harness.bot 'vanila'"),
        ({"selector": "bm26"}, "unknown selector 'bm26'"),
        ({"selector": "bm25_frac:2"}, "must be in"),
        ({"bot": "vanilla", "idf": "cmss"}, "harness.idf='cmss'"),
        ({"bot": "vanilla", "sketch_opts": ["conservative"]}, "harness.sketch_opts='conservative'"),
        ({"bot": "vanilla", "budget": "35%"}, "harness.budget=.35%."),
        ({"bot": "vanilla", "budget": "frac:0.3.5"}, "harness.budget=.frac:0.3.5."),
        ({"query": "input"}, "harness.query='input'"),
        ({"bot": "full", "idf": "cms"}, "selector 'full' reads no idf"),
        ({"selector": "bm25", "seed": 581}, "reads no seed"),
        ({"bot": "full", "budget": "frac:0.3"}, "takes no budget"),
    ],
)
def test_a_wrong_name_or_value_is_refused(fields: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve(harness(**fields))


@pytest.mark.parametrize(("bot", "package"), [("vanilla", "WP4"), ("variant", "WP4"), ("product", "WP6"), ("optimal", "WP6"), ("full_plus", "WP2")])
def test_a_defined_but_unbuilt_component_fails_loudly(bot: str, package: str) -> None:
    with pytest.raises(NotImplementedError, match=f"not implemented yet .*{package}"):
        bots.check_runnable(resolve(harness(bot=bot)), "agent")


def test_bm25_with_a_budget_is_bm25_frac(plugin: Path) -> None:
    c = resolve(harness(selector="bm25", budget="frac:0.25"))
    assert bots.legacy_policy(c) == "bm25_frac:0.25"
    assert policy_text(plugin, bots.legacy_policy(c), SYSTEM) == policy_text(plugin, "bm25_frac:0.25", SYSTEM)
    with pytest.raises(NotImplementedError, match="words"):
        bots.check_runnable(resolve(harness(selector="bm25", budget="frac:0.35,words:1500")), "agent")


def test_input_queries_need_an_agent_task() -> None:
    c = resolve(harness(bot="full", query="instructions+input"))
    bots.check_runnable(c, "agent")
    with pytest.raises(ValueError, match="needs an agent task"):
        bots.check_runnable(c, "decide")


@pytest.fixture
def plugin(tmp_path: Path) -> Path:
    (tmp_path / POLICY_FILE).write_text(HANDBOOK)
    return tmp_path


def test_the_instructions_query_is_the_system_prompt_itself() -> None:
    assert selection_query(SYSTEM, "Task input:\n{}", "instructions") is SYSTEM


def test_an_input_query_changes_the_selection_with_the_task_input(plugin: Path) -> None:
    lodging = 'Task input:\n{\n  "request": "a casino resort hotel, lodging cap in a major city"\n}'
    security = 'Task input:\n{\n  "request": "my laptop is encrypted; may I leave it in the vehicle or hotel lobby"\n}'
    by_instructions = {with_policy(SYSTEM, plugin, "bm25", selection_query(SYSTEM, text, "instructions")) for text in (lodging, security)}
    by_input = [with_policy(SYSTEM, plugin, "bm25", selection_query(SYSTEM, text, "instructions+input")) for text in (lodging, security)]
    assert len(by_instructions) == 1  # today's behaviour: one selection for every task input
    assert by_input[0] != by_input[1]
    assert "Laptops must be encrypted" in by_input[1] and "Laptops must be encrypted" not in by_input[0]
    # The input query still reads the instructions: their rules are selected for either input.
    assert all("lowest logical airfare in economy class" in text for text in by_input)


def test_the_handbook_is_read_at_the_root_first_then_bundled(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="policy.txt or .*src/plugin/data/policy.txt"):
        handbook_path(tmp_path, "full")
    bundled = tmp_path / BUNDLED_POLICY_FILE
    bundled.parent.mkdir(parents=True)
    bundled.write_text("Bundled handbook.")
    assert handbook_path(tmp_path, "full") == bundled
    assert with_policy(SYSTEM, tmp_path, "full") == f"{SYSTEM}\n\n## {TITLE}\n\nBundled handbook."
    (tmp_path / POLICY_FILE).write_text("Root handbook.")
    assert with_policy(SYSTEM, tmp_path, "full") == f"{SYSTEM}\n\n## {TITLE}\n\nRoot handbook."


def test_a_root_policy_file_passes_the_plugin_check(tmp_path: Path) -> None:
    from nanoagent.plugin.authoring import check_plugin
    from test_plugin import write_plugin

    write_plugin(tmp_path)
    (tmp_path / POLICY_FILE).write_text(HANDBOOK)
    check_plugin(tmp_path)


def _profile(*totals: int):
    from nanoagent.profiler import RunProfile, StepProfile

    steps = tuple(StepProfile(step=i + 1, duration_s=0, model_s=0, tools_s=0, overhead_s=0, usage={"total_tokens": t}, cost=0) for i, t in enumerate(totals))
    return RunProfile(duration_s=0, usage={}, cost=0, steps=steps)


def test_the_hosted_cap_names_the_step_a_capped_run_would_not_have_reached() -> None:
    assert bots.hosted_cap_stop(_profile(30_000, 30_000), 65_536) is None
    assert bots.hosted_cap_stop(_profile(30_000, 30_000, 10_000), 65_536) is None
    assert bots.hosted_cap_stop(_profile(30_000, 36_000, 10_000), 65_536) == 3
    assert bots.hosted_cap_stop(_profile(70_000), 65_536) is None  # the cap is checked before a step
    assert bots.hosted_cap_stop(_profile(70_000, 1), 65_536) == 2
    assert bots.hosted_cap_stop(_profile(70_000, 1), None) is None


def test_the_config_hash_is_the_arm() -> None:
    record = bots.provenance(resolve(harness(bot="full")), harness(bot="full"), {"model": "m"}, "abc")
    again = bots.provenance(resolve(harness(bot="full")), harness(bot="full"), {"model": "m"}, "abc")
    assert record == again and record["bot"] == "full" and record["components"]["selector"] == "full"
    other = bots.provenance(resolve(harness(bot="full_plus")), harness(bot="full_plus"), {"model": "m"}, "abc")
    capped = bots.provenance(resolve(harness(bot="full")), harness(bot="full", token_limit=200_000), {"model": "m"}, "abc")
    assert len({record["config_sha256"], other["config_sha256"], capped["config_sha256"]}) == 3


# --- the plugin CLI run ---------------------------------------------------------------------------

_LLM = """llm > nanoagent.llm.openrouter.OpenRouterConfig:
  model: a-model
  max_tokens: null
  temperature: 0.0
  max_attempts: 1
  read_timeout: 60.0
  input_price: 0.0
  output_price: 0.0
"""
_HARNESS = """harness > nanoagent.harness.nanoagent.NanoAgentConfig:
  max_steps: 6
  cost_limit: null
  token_limit: null
  context_window: null
  hooks: []
  skills: null
  context_files: []
  events: null
"""


class _Submitting:
    """A chat model that submits the task input's value, then answers; it records each request."""

    def __init__(self) -> None:
        self.requests: list[tuple[list[dict], list[dict]]] = []

    async def query(self, messages, tools, on_delta=None, **_):
        from nanoagent.core.agent import Reply, ToolCall

        self.requests.append(([dict(m) for m in messages], tools))
        if len(self.requests) == 1:
            return Reply(content=None, tool_calls=[ToolCall(id="c1", name="submit", arguments='{"value": "hello"}')], usage={"total_tokens": 70_000})
        return Reply(content="Submitted it.", usage={"total_tokens": 100})


def _agent_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harness_lines: str = "") -> tuple[_Submitting, Path]:
    from nanoagent.cli import plugin as cli
    from nanoagent.runtime.model import Model
    from test_plugin import _echo_plugin

    model = _Submitting()
    monkeypatch.setattr(Model, "from_config", classmethod(lambda _cls, cfg: model))
    monkeypatch.setattr(cli, "_arguments", lambda path, task: {"value": "hello"})
    root = _echo_plugin(tmp_path)
    manifest = root / "nanoagent.yaml"
    manifest.write_text(manifest.read_text().replace("runner: tool", "runner: agent"))
    (root / POLICY_FILE).write_text(HANDBOOK)
    config = tmp_path / "run.yaml"
    config.write_text(
        f"_ > nanoagent.cli.plugin.PluginOpenRouterAgentConfig:\nplugin: {root}\ntask: echo\narguments: unused.json\n"
        f"inputs: []\ntimeout: 60.0\n{_LLM}{_HARNESS}{harness_lines}"
    )
    home = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["nanoagent", "plugin", "agent"])
    with pytest.raises(SystemExit) as exit_:
        cli.main(["agent", f"config={config}", f"home={home}"])
    assert exit_.value.code == 0, (home / "run.log").read_text()
    return model, home


def test_an_upstream_run_config_sends_the_upstream_prompt_and_records_nothing_new(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nanoagent.plugin import load_manifest
    from nanoagent.plugin.tasks import task_prompt

    model, home = _agent_run(tmp_path, monkeypatch)
    manifest = load_manifest(tmp_path / "plugin")
    system, user = task_prompt(manifest, manifest.task("echo"), {"value": "hello"})
    assert model.requests[0][0] == [{"role": "system", "content": system}, {"role": "user", "content": user}]
    messages = json.loads((home / "messages.json").read_text())
    assert messages["final_metrics"]["extra"] == {"cost": 0.0, "stop_reason": "answer"}
    assert "harness:" not in (home / "run.log").read_text()


def test_a_bot_run_records_its_components_hash_and_the_hosted_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model, home = _agent_run(tmp_path, monkeypatch, '  bot: full\n  query: instructions+input\n  hosted_token_limit: 65536\n')
    system = model.requests[0][0][0]["content"]
    assert system.endswith(f"\n\n## {TITLE}\n\n{HANDBOOK.strip()}")
    record = json.loads((home / "messages.json").read_text())["final_metrics"]["extra"]["harness"]
    assert record["bot"] == "full"
    assert record["components"] == {**dataclasses.asdict(PRESETS["full"]), "query": "instructions+input"}
    assert record["limits"]["hosted_token_limit"] == 65536 and record["llm"]["model"] == "a-model"
    assert len(record["config_sha256"]) == 64 and len(record["handbook_sha256"]) == 64
    assert record["hosted_cap"] == {"limit": 65536, "exceeded": True, "stops_before_step": 2}
    log = (home / "run.log").read_text()
    assert f'"config_sha256": "{record["config_sha256"]}"' in log


def test_a_misspelt_harness_key_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(Exception, match="selectr"):
        _agent_run(tmp_path, monkeypatch, "  selectr: bm25\n")


def test_yaml_off_for_the_bot_runs_the_upstream_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nanoagent.plugin import load_manifest
    from nanoagent.plugin.tasks import task_prompt

    model, _home = _agent_run(tmp_path, monkeypatch, "  bot: off\n  policy: off\n")
    manifest = load_manifest(tmp_path / "plugin")
    system, _user = task_prompt(manifest, manifest.task("echo"), {"value": "hello"})
    assert model.requests[0][0][0]["content"] == system


def test_the_graph_records_the_prompt_the_agent_was_built_with(tmp_path: Path) -> None:
    import asyncio

    from nanoagent.core.agent import Agent
    from nanoagent.plugin import load_manifest
    from nanoagent.plugin.tasks import run_task
    from test_plugin import _echo_plugin

    root = _echo_plugin(tmp_path)
    manifest_file = root / "nanoagent.yaml"
    manifest_file.write_text(manifest_file.read_text().replace("runner: tool", "runner: agent"))
    manifest = load_manifest(root)
    events: list[tuple[str, dict]] = []
    seen: list[str] = []

    def make_agent(system: str, task_input: str) -> Agent:
        seen.append(task_input)
        return Agent(_Submitting(), [], system_prompt=f"{system}\n\n## {TITLE}\n\nFly economy.")

    asyncio.run(
        run_task(
            manifest,
            manifest.task("echo"),
            {"value": "hello"},
            tmp_path / "work",
            make_agent=make_agent,
            make_decider=None,
            timeout=60,
            on_event=lambda kind, data: events.append((kind, data)),
            stage=lambda folder: folder.mkdir(parents=True, exist_ok=True),
        )
    )
    assert seen == ['Task input:\n{\n  "value": "hello"\n}']
    nodes = [data["node"] for kind, data in events if kind == "task.graph" and data["op"] == "node"]
    [system] = [node for node in nodes if node["id"] == "system"]
    assert system["value"].endswith("Fly economy.")
