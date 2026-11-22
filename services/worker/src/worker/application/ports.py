from typing import Any, Protocol, runtime_checkable

from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    AgentHeartbeat,
    CompletionCommand,
    InferenceProviderHealth,
    InferenceResponse,
    TaskRequested,
)


class MessageBus(Protocol):
    async def next_task(self) -> TaskRequested: ...
    async def publish(self, event: AgentEvent) -> None: ...
    async def register(self, metadata: AgentCapabilityMetadata) -> None: ...
    async def heartbeat(self, heartbeat: AgentHeartbeat) -> None: ...


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
    async def generate(self, prompt: str) -> InferenceResponse: ...


@runtime_checkable
class InferenceHealthProbe(Protocol):
    """Application port for active or passive inference provider health probing."""

    async def check_health(self) -> InferenceProviderHealth: ...


@runtime_checkable
class RuntimeInferenceProvider(InferenceProvider, InferenceHealthProbe, Protocol):
    """Application port combining prompt generation and health probing capabilities."""
