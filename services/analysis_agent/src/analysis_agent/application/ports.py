from typing import Any, Protocol

from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    AgentHeartbeat,
    CompletionCommand,
    TaskRequested,
)

from analysis_agent.domain.analysis import AnalysisReport


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


class MaterialAnalyzer(Protocol):
    def analyze(self, material: str) -> AnalysisReport: ...
