"""OpenRouter's billed cost and serving provider, from its response to the run's done event; and the
OpenRouter config's ``extra_body`` (a provider pin) reaching the request."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionChunk

from nanoagent.core.agent import Agent
from nanoagent.inference.backends.sglang import ChatBackend
from nanoagent.inference.config import LeanInferConfig
from nanoagent.llm.openrouter import OpenRouterConfig
from nanoagent.profiler import normalize_usage
from nanoagent.runtime.model import Model

PIN = 'provider={"order": ["baidu/fp8"], "allow_fallbacks": false, "quantizations": ["fp8"]}'

# The second call of the Baidu-pinned cache probe of 2026-11-24 (10,752 of 10,831 prompt tokens read
# from the cache): its generation id, provider, token counts and billed cost as OpenRouter returned
# them, laid out as OpenRouter's final usage chunk of a stream. The probe recorded no reasoning count
# or upstream cost; OpenRouter reports upstream_inference_cost only for a request on the caller's
# own provider key (BYOK), and null otherwise.
RECORDED_USAGE_CHUNK = {
    "id": "gen-1795550195-RcCS7KC6kgY3b70XkLkr",
    "object": "chat.completion.chunk",
    "created": 1795550195,
    "model": "deepseek/deepseek-v4-flash",
    "provider": "Baidu",
    "choices": [],
    "usage": {
        "prompt_tokens": 10831,
        "completion_tokens": 200,
        "total_tokens": 11031,
        "cost": 4.6276664e-05,
        "is_byok": False,
        "prompt_tokens_details": {"cached_tokens": 10752},
        "cost_details": {"upstream_inference_cost": None},
    },
}


def test_the_recorded_usage_keeps_the_billed_cost_beside_the_tokens() -> None:
    chunk = ChatCompletionChunk.model_validate(RECORDED_USAGE_CHUNK)
    assert normalize_usage(chunk.usage) == {
        "prompt_tokens": 10831,
        "completion_tokens": 200,
        "total_tokens": 11031,
        "cached_tokens": 10752,
        "billed_cost_nanousd": 46277,  # $4.6276664e-05, in whole nano-dollars
    }


def test_upstream_cost_and_reasoning_are_kept_when_reported() -> None:
    usage = normalize_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 30},
            "cost": 0.0005182342,
            "cost_details": {"upstream_inference_cost": 0.00049},
        }
    )
    assert usage["reasoning_tokens"] == 30
    assert (usage["billed_cost_nanousd"], usage["upstream_cost_nanousd"]) == (518234, 490000)
    assert "billed_cost_nanousd" not in normalize_usage({"prompt_tokens": 1, "completion_tokens": 1})


def _stream(*chunks: dict):
    async def chunks_():
        for chunk in chunks:
            yield ChatCompletionChunk.model_validate(chunk)

    class Stream:
        def __aiter__(self):
            return chunks_()

        async def close(self) -> None:
            pass

    return Stream()


def _content_chunk(text: str) -> dict:
    return {**RECORDED_USAGE_CHUNK, "usage": None, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}]}


class StreamingCompletions:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def create(self, **request):
        self.requests.append(request)
        return _stream(_content_chunk("Done."), RECORDED_USAGE_CHUNK)


def _openrouter(completions, extra_body: dict | None = None) -> ChatBackend:
    backend = ChatBackend.from_config(
        LeanInferConfig(provider="openrouter", model="deepseek/deepseek-v4-flash", base_url="http://localhost/v1", api_key="k", extra_body=extra_body or {})
    )
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return backend


def test_the_done_event_records_billed_cost_provider_and_tokens(tmp_path: Path) -> None:
    from nanoagent.execution.events import EventWriter

    writer = EventWriter(tmp_path / "events.ndjson")
    agent = Agent(Model(_openrouter(StreamingCompletions())), [], system_prompt="s", events=writer)
    result = asyncio.run(agent.run("go", on_delta=lambda kind, text: None))
    writer.close()
    assert result.answer == "Done."
    [done] = [json.loads(line) for line in (tmp_path / "events.ndjson").read_text().splitlines() if '"done"' in line]
    assert done["usage"] == {
        "prompt_tokens": 10831,
        "completion_tokens": 200,
        "total_tokens": 11031,
        "cached_tokens": 10752,
        "billed_cost_nanousd": 46277,
    }
    assert done["metrics"]["providers"] == ["Baidu"]
    [call] = done["metrics"]["steps"][0]["model_calls"]
    assert call["provider"] == "Baidu" and call["usage"]["billed_cost_nanousd"] == 46277


def test_an_unstreamed_reply_and_an_answer_name_their_provider() -> None:
    message = SimpleNamespace(content='{"next": "a"}', tool_calls=None)
    response = SimpleNamespace(
        provider="Baidu",
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10, cost=1e-06),
    )

    class Completions:
        async def create(self, **request):
            return response

    backend = _openrouter(Completions())
    reply = asyncio.run(Model(backend).query([{"role": "user", "content": "hi"}], []))
    assert reply.provider == "Baidu" and reply.usage["billed_cost_nanousd"] == 1000
    schema = {"type": "object", "properties": {"next": {"type": "string"}}, "required": ["next"]}
    answer = asyncio.run(backend.answer([{"role": "user", "content": "pick"}], name="decide", schema=schema))
    assert answer.provider == "Baidu"


def test_a_profile_without_a_provider_keeps_its_record_unchanged() -> None:
    from nanoagent.profiler import Profiler

    profiler = Profiler()
    step = profiler.start_step(1)
    record = step.finish_model_call(step.start_model_call(), usage={"total_tokens": 3}, cost=0.0)
    step.finish()
    assert "provider" not in record.to_dict()
    assert "providers" not in profiler.snapshot().to_dict()


def _openrouter_config(**fields) -> OpenRouterConfig:
    base = dict(model="deepseek/deepseek-v4-flash", max_tokens=None, temperature=0.0, max_attempts=1, read_timeout=60.0, input_price=0.0, output_price=0.0)
    return OpenRouterConfig(**{**base, **fields})


def test_the_openrouter_extra_body_reaches_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert _openrouter_config().runtime().extra_body == []
    plain = Model.from_config(_openrouter_config().runtime())._backend
    assert "extra_body" not in plain._base_kwargs
    pinned = Model.from_config(_openrouter_config(extra_body=[PIN]).runtime())._backend
    assert pinned._base_kwargs["extra_body"] == {"provider": {"order": ["baidu/fp8"], "allow_fallbacks": False, "quantizations": ["fp8"]}}

    completions = StreamingCompletions()
    pinned._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    asyncio.run(Model(pinned).query([{"role": "user", "content": "hi"}], [], on_delta=lambda kind, text: None))
    [request] = completions.requests
    assert request["extra_body"]["provider"]["order"] == ["baidu/fp8"]
    assert "usage" not in request["extra_body"]  # the deprecated usage.include is not sent


def test_reasoning_effort_is_sent_as_openrouter_reasoning() -> None:
    from nanoagent.cli.plugin import _with_reasoning
    from nanoagent.runtime.config import parse_kv_list

    model = _openrouter_config(extra_body=[PIN]).runtime()
    assert _with_reasoning(model, None) is model
    low = _with_reasoning(model, "low")
    assert parse_kv_list(low.extra_body) == {**parse_kv_list([PIN]), "reasoning": {"effort": "low"}}
    with pytest.raises(ValueError, match="keep one"):
        _with_reasoning(_openrouter_config(extra_body=['reasoning={"effort": "high"}']).runtime(), "low")
    with pytest.raises(ValueError, match="provider is 'bedrock'"):
        from dataclasses import replace

        _with_reasoning(replace(model, provider="bedrock"), "low")
