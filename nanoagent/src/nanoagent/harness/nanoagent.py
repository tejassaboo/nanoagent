"""NanoAgent's in-process agent-loop harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from nanoagent.core import workspace
from nanoagent.core.agent import Agent, AgentResult, StopReason
from nanoagent.harness.base import ProgressSink, RunnerCapabilities, RunnerRequest, RunnerResult, RunnerUnavailableError
from nanoagent.inference.providers import MissingKeyError
from nanoagent.runtime.build import build_agent
from nanoagent.runtime.config import AgentConfig, AgentDefinitionConfig
from nanoagent.runtime.model import Model
from omegaconf import MISSING
from slimconfig import Config


@dataclass
class NanoAgentConfig(Config):
    """Configuration owned only by NanoAgent's agent loop."""

    max_steps: int = MISSING
    cost_limit: float | None = MISSING
    token_limit: int | None = MISSING
    context_window: int | None = MISSING
    hooks: list[str] = MISSING
    skills: str | None = MISSING
    context_files: list[str] = MISSING
    events: str | None = MISSING
    # The company policy a plugin agent sees: "off" (none, the baseline), "full" (the plugin's whole
    # policy.txt) or a method name from nanoagent.plugin.policy.METHODS (a compressed policy.txt).
    # Only plugin agent and decide tasks read it; every method uses the same model and tools.
    policy: str = "off"
    # A bot preset that sets every field below at once (nanoagent.plugin.bots.PRESETS): off, full,
    # full_plus (or full+), vanilla, vanilla_a (ladder rung V-a), variant, product or optimal. Each field below that is not null
    # replaces the preset's value; `policy` is the older name of `selector`. Quote "off" in YAML.
    bot: str = "off"
    selector: str | None = None  # off, full, bloom, scoped, rule_quota, bm25, bm25_frac[:f], bloom_chunk, sketch_bm25
    idf: str | None = None  # none, exact or cms (a count-min sketch)
    dedup: str | None = None  # none or minhash (drop near-copies of kept chunks)
    twin: str | None = None  # sketch, or exact (every sketch swapped for its exact structure)
    sketch_opts: list[str] | None = None  # any of probe_sizing, conservative_update, one_permutation, double_hash
    chunker: str | None = None  # plain or structured
    budget: str | None = None  # frac:<f>, or frac:<f>,words:<n> (the smaller); null = the selector's own
    backup: str | None = None  # none, or learned (a learned Bloom filter with a backup filter)
    stopwords: str | None = None  # none, or light (strip stopwords from the selected text)
    stop_on_submit: bool | None = None  # end an agent task right after its first accepted submit
    reasoning_effort: str | None = None  # low, medium or high; null sends nothing
    seed: int | None = None  # the sketches' hash seed
    # What a selector reads: the task's instructions (the system prompt), or those and the task's
    # input message (`instructions+input`; the selection is then made per task input).
    query: str | None = None
    # A token cap the run is measured against but not stopped by (e.g. a hosted 65536): the run
    # records the step at which a run capped there would have stopped. token_limit still stops it.
    hosted_token_limit: int | None = None

    def runtime(self, *, system_prompt: str) -> AgentConfig:
        return AgentConfig(
            system_prompt=system_prompt,
            max_steps=self.max_steps,
            cost_limit=self.cost_limit,
            token_limit=self.token_limit,
            context_window=self.context_window,
            hooks=self.hooks,
            skills=self.skills,
            context_files=self.context_files,
            events=self.events,
        )


class AgentFactory(Protocol):
    def __call__(self, instructions: str | None) -> tuple[Agent, str]: ...


class NanoAgentRunner:
    """Run requests through NanoAgent's in-process agent loop."""

    name = "nanoagent"
    capabilities = RunnerCapabilities(
        streaming=True,
        reasoning=True,
        tools=True,
        usage=True,
        cancellation=True,
        history=True,
    )

    def __init__(
        self,
        cfg: AgentDefinitionConfig,
        *,
        agent_factory: AgentFactory | None = None,
    ) -> None:
        self._cfg = cfg
        # Built on the first run, like a PI profile's credentials are only needed when it runs.
        self._model: Model | None = None
        self._factory = agent_factory or self._configured_agent

    def _configured_agent(self, instructions: str | None) -> tuple[Agent, str]:
        clean = instructions.strip() if instructions else ""
        suffix = f"Application instructions:\n{clean}" if clean else None
        if self._model is None:
            try:
                self._model = Model.from_config(self._cfg.model)
            except MissingKeyError as error:
                raise RunnerUnavailableError(str(error)) from None
        agent = build_agent(self._cfg, model=self._model, prompt_suffix=suffix)
        return agent, agent.system_prompt

    async def run(self, request: RunnerRequest, emit: ProgressSink) -> RunnerResult:
        agent, system_prompt = self._factory(request.instructions)
        transcript = [
            {"role": "system", "content": system_prompt},
            *[dict(message) for message in request.messages],
        ]
        emitted_tools = 0
        terminal_snapshot: AgentResult | None = None

        def on_delta(kind: str, text: str) -> None:
            emit({"type": "delta", "kind": kind, "text": text})

        def on_step(result: AgentResult) -> None:
            nonlocal emitted_tools, terminal_snapshot
            for row in result.tool_calls[emitted_tools:]:
                emit({"type": "tool", **row})
            emitted_tools = len(result.tool_calls)
            if result.stop_reason is StopReason.RUNNING:
                emit(
                    {
                        "type": "step",
                        "step": result.steps,
                        "usage": result.usage,
                        "cost": result.cost,
                        "metrics": (
                            result.profile.steps[-1].to_dict()
                            if result.profile is not None and result.profile.steps
                            else None
                        ),
                    }
                )
            else:
                terminal_snapshot = result

        try:
            if request.workspace is None:
                result = await agent.run(
                    request.input,
                    messages=transcript,
                    on_delta=on_delta,
                    on_step=on_step,
                    on_tool=emit,
                )
            else:
                with workspace.use(request.workspace):
                    result = await agent.run(
                        request.input,
                        messages=transcript,
                        on_delta=on_delta,
                        on_step=on_step,
                        on_tool=emit,
                    )
        except Exception:
            if terminal_snapshot is None:
                raise
            result = terminal_snapshot
        return RunnerResult.from_agent_result(result)

    def forget_model(self) -> Model | None:
        """Drop the model the first run built, so the next builds one with the credentials set now.

        Runs in flight may still use it, so the caller closes it later.
        """
        model, self._model = self._model, None
        return model

    async def aclose(self) -> None:
        if self._model is not None:
            await self._model.aclose()

    def availability(self) -> tuple[bool, str | None]:
        return True, None

    @property
    def definition(self) -> AgentDefinitionConfig:
        """Resolved NanoAgent definition, including profile model overrides."""
        return self._cfg


NanoAgentHarness = NanoAgentRunner

__all__ = ["AgentFactory", "NanoAgentConfig", "NanoAgentHarness", "NanoAgentRunner"]
