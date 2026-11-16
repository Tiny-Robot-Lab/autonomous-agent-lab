from typing import Protocol

from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    AgentHeartbeat,
    TaskRequested,
    TaskResult,
)

from analysis_agent.domain.analysis import AnalysisReport


class MessageBus(Protocol):
    async def next_task(self) -> TaskRequested: ...
    async def publish(self, event: AgentEvent) -> None: ...
    async def register(self, metadata: AgentCapabilityMetadata) -> None: ...
    async def heartbeat(self, heartbeat: AgentHeartbeat) -> None: ...


class ResultStore(Protocol):
    async def save(self, result: TaskResult) -> None: ...


class MaterialAnalyzer(Protocol):
    def analyze(self, material: str) -> AnalysisReport: ...
