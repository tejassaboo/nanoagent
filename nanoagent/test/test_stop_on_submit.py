"""Stop-on-submit: an agent task's run ends at its first accepted submit, with no further model call."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nanoagent.core.agent import Agent, Reply, StopReason, ToolCall
from nanoagent.core.tool import Tool
from nanoagent.plugin import load_manifest
from nanoagent.plugin.tasks import run_task
from test_plugin import _echo_plugin

SUBMIT_HELLO = ToolCall(id="s1", name="submit", arguments='{"value": "hello"}')
ECHO = ToolCall(id="e1", name="echo", arguments='{"value": "hello"}')


class Scripted:
    """A chat model that gives one scripted reply per request and fails past the script."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.requests = 0

    async def query(self, messages, tools, on_delta=None, **_):
        self.requests += 1
        if not self.replies:
            raise AssertionError("the model was asked again after the run should have stopped")
        return self.replies.pop(0)


def reply(*calls: ToolCall) -> Reply:
    return Reply(content=None, tool_calls=list(calls), usage={"total_tokens": 10})


def answer() -> Reply:
    return Reply(content="I submitted hello.", usage={"total_tokens": 10})


def _run(tmp_path: Path, model: Scripted, *, stop_on_submit: bool, max_steps: int = 6, events=None):
    root = _echo_plugin(tmp_path)
    manifest_file = root / "nanoagent.yaml"
    manifest_file.write_text(manifest_file.read_text().replace("runner: tool", "runner: agent"))
    manifest = load_manifest(root)
    seen: list[tuple[str, dict]] = []
    outcome = asyncio.run(
        run_task(
            manifest,
            manifest.task("echo"),
            {"value": "hello"},
            tmp_path / "work",
            make_agent=lambda system, _input: Agent(model, [], system_prompt=system, max_steps=max_steps, events=events),
            make_decider=None,
            timeout=60,
            on_event=lambda kind, data: seen.append((kind, data)),
            stage=lambda folder: folder.mkdir(parents=True, exist_ok=True),
            stop_on_submit=stop_on_submit,
        )
    )
    return outcome, seen


def test_stop_on_submit_ends_after_the_first_accepted_submit(tmp_path: Path) -> None:
    model = Scripted(reply(ECHO), reply(SUBMIT_HELLO))  # a third request would fail
    outcome, _ = _run(tmp_path, model, stop_on_submit=True)
    assert model.requests == 2
    assert outcome.agent.stop_reason is StopReason.SUBMITTED and outcome.agent.steps == 2
    assert outcome.output == {"value": "hello"} and outcome.verdict.valid
    assert outcome.agent.messages[-1] == {"role": "tool", "tool_call_id": "s1", "content": "Submitted."}


def test_stop_on_submit_off_is_unchanged(tmp_path: Path) -> None:
    model = Scripted(reply(ECHO), reply(SUBMIT_HELLO), answer())
    outcome, _ = _run(tmp_path, model, stop_on_submit=False)
    assert model.requests == 3
    assert outcome.agent.stop_reason is StopReason.ANSWER and outcome.agent.answer == "I submitted hello."
    assert outcome.output == {"value": "hello"}


def test_a_rejected_submit_does_not_stop(tmp_path: Path) -> None:
    rejected = ToolCall(id="s0", name="submit", arguments="{}")  # `value` missing: build_output refuses it
    model = Scripted(reply(rejected), reply(SUBMIT_HELLO))
    outcome, seen = _run(tmp_path, model, stop_on_submit=True)
    assert model.requests == 2
    assert [(row["name"], row["is_error"]) for row in outcome.agent.tool_calls] == [("submit", True), ("submit", False)]
    assert outcome.agent.stop_reason is StopReason.SUBMITTED and outcome.output == {"value": "hello"}
    assert [data["isError"] for kind, data in seen if kind == "task.tool"] == [True, False]


def test_stop_keeps_every_call_of_the_submitting_step(tmp_path: Path) -> None:
    model = Scripted(reply(ECHO, SUBMIT_HELLO))
    outcome, seen = _run(tmp_path, model, stop_on_submit=True)
    assert model.requests == 1
    assert [row["name"] for row in outcome.agent.tool_calls] == ["echo", "submit"]
    assert not any(row["is_error"] for row in outcome.agent.tool_calls)
    assert [data["name"] for kind, data in seen if kind == "task.tool"] == ["echo", "submit"]
    assert [m["tool_call_id"] for m in outcome.agent.messages if m["role"] == "tool"] == ["e1", "s1"]


def test_submit_on_the_last_step_reports_submitted(tmp_path: Path) -> None:
    model = Scripted(reply(SUBMIT_HELLO))
    outcome, _ = _run(tmp_path, model, stop_on_submit=True, max_steps=1)
    assert model.requests == 1
    assert outcome.agent.stop_reason is StopReason.SUBMITTED and outcome.agent.steps == 1
    assert outcome.output == {"value": "hello"}


def test_the_done_event_carries_submitted(tmp_path: Path) -> None:
    from nanoagent.execution.events import EventWriter

    writer = EventWriter(tmp_path / "events.ndjson")
    _run(tmp_path, Scripted(reply(ECHO), reply(SUBMIT_HELLO)), stop_on_submit=True, events=writer)
    writer.close()
    lines = [json.loads(line) for line in (tmp_path / "events.ndjson").read_text().splitlines()]
    assert [line["type"] for line in lines] == ["tool", "step", "tool", "done"]
    done = lines[-1]
    assert done["stop_reason"] == "submitted" and done["steps"] == 2
    assert len(done["metrics"]["steps"]) == 2 and done["usage"]["total_tokens"] == 20


class Flaky(Tool):
    """Fails on its first call and succeeds on the next."""

    NAME = "t"
    PARAMETERS = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.calls = 0

    def run(self) -> str:
        self.calls += 1
        if self.calls == 1:
            raise ValueError("not yet")
        return "done"


def test_the_agent_stops_after_the_named_tool_succeeds_and_not_before() -> None:
    call = ToolCall(id="t", name="t", arguments="{}")
    model = Scripted(reply(call), reply(call))
    result = asyncio.run(Agent(model, [Flaky()], system_prompt="s").run("go", stop_after_tool="t"))
    assert model.requests == 2
    assert [row["is_error"] for row in result.tool_calls] == [True, False]
    assert result.stop_reason is StopReason.SUBMITTED and result.stop_reason == "submitted" and result.steps == 2
    with pytest.raises(AssertionError, match="asked again"):  # without it the loop asks the model a third time
        asyncio.run(Agent(Scripted(reply(call), reply(call)), [Flaky()], system_prompt="s").run("go"))
