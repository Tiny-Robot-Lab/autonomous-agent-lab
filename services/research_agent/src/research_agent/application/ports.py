from typing import Any, Protocol, runtime_checkable

from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    AgentHeartbeat,
    CompletionCommand,
    InferenceProviderHealth,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskRequested,
    TaskResult,
    TerminalOutcome,
)

from research_agent.domain.models import ResearchResult, SynthesisResult


class MessageBus(Protocol):
    async def next_task(self) -> TaskRequested: ...
    async def publish(self, event: AgentEvent) -> None: ...
    async def register(self, metadata: AgentCapabilityMetadata) -> None: ...
    async def heartbeat(self, heartbeat: AgentHeartbeat) -> None: ...


class ResearchExecutor(Protocol):
    async def execute(self, task: TaskRequested) -> ResearchResult: ...


class TurnCompletionClient(Protocol):
    """Provider-neutral boundary for submitting fenced turn completions.

    Implementations POST the canonical CompletionCommand to the authenticated
    Orchestrator completion API with the detached assignment proof. Raw
    exceptions, prompts, and credentials must never cross this boundary.
    """

    async def submit_completion(
        self,
        task_id: str,
        command: CompletionCommand,
        assignment_proof: str,
    ) -> dict[str, Any]: ...


class InferenceProvider(Protocol):
    async def synthesize(self, question: str, evidence: str) -> SynthesisResult: ...


ResearchSynthesizer = InferenceProvider


@runtime_checkable
class InferenceHealthProbe(Protocol):
    """Application port for active or passive inference provider health probing."""

    async def check_health(self) -> InferenceProviderHealth: ...


@runtime_checkable
class RuntimeInferenceProvider(InferenceProvider, InferenceHealthProbe, Protocol):
    """Application port combining research synthesis and health probing capabilities."""


@runtime_checkable
class ContinuationContextReader(Protocol):
    """Port for loading authoritative parent and child context for resumed continuation."""

    async def get_task_input(self, task_id: str, version: int = 1) -> TaskInput | None: ...
    async def get_continuation(self, task_id: str, version: int = 1) -> TaskContinuation | None: ...
    async def get_stage_artifact(self, producing_task_id: str, artifact_id: str, artifact_version: int = 1) -> StageArtifact | None: ...
    async def get_terminal_outcome(self, task_id: str) -> TerminalOutcome | None: ...
    async def get_canonical_result(self, task_id: str) -> TaskResult | None: ...


@runtime_checkable
class DelegationClient(Protocol):
    """Port for staging parent artifacts and submitting delegation proposals."""

    async def stage_artifact(
        self,
        task_id: str,
        assignment_id: str,
        assignment_proof: str,
        artifact_id: str,
        artifact_type: str,
        content: dict[str, Any],
        artifact_version: int = 1,
        payload_schema_version: int = 1,
    ) -> dict[str, Any]: ...

    async def propose_delegation(
        self,
        task_id: str,
        assignment_id: str,
        assignment_proof: str,
        proposal_id: str,
        child_intent: str,
        artifact_reference: dict[str, Any],
        target_capability: str = "analysis",
        parent_input_version: int = 1,
        control_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...
