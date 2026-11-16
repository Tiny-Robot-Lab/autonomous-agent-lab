from __future__ import annotations

import logging

from agent_contracts import AgentEvent, TaskRequested, TaskResult

from analysis_agent.application.ports import MaterialAnalyzer, MessageBus, ResultStore

logger = logging.getLogger(__name__)


class AnalysisExecutionError(Exception):
    """Raised when an expected task failure occurs during analysis."""

    def __init__(self, task_id: str, original_error: Exception) -> None:
        self.task_id = task_id
        self.original_error = original_error
        super().__init__(f"Analysis task {task_id} failed: {original_error}")


class AnalysisProcessor:
    def __init__(
        self,
        agent_id: str,
        bus: MessageBus,
        analyzer: MaterialAnalyzer,
        result_store: ResultStore,
    ) -> None:
        self.agent_id = agent_id
        self.bus = bus
        self.analyzer = analyzer
        self.result_store = result_store

    async def run_once(self, task: TaskRequested) -> None:
        logger.info("Analysis agent %s processing task %s", self.agent_id, task.task_id)
        await self.bus.publish(
            AgentEvent.create(self.agent_id, "working", "Analyzing supplied material", task.task_id)
        )
        try:
            report = self.analyzer.analyze(task.prompt)
            await self.result_store.save(
                TaskResult(task_id=task.task_id, output=report.render())
            )
            await self.bus.publish(
                AgentEvent.create(self.agent_id, "idle", "Task completed", task.task_id)
            )
            logger.info("Analysis agent %s completed task %s", self.agent_id, task.task_id)
        except Exception as exc:
            logger.exception("Analysis task %s failed during execution", task.task_id)
            await self.bus.publish(
                AgentEvent.create(self.agent_id, "error", "Task failed", task.task_id)
            )
            raise AnalysisExecutionError(task.task_id, exc) from exc
