from __future__ import annotations

import logging
from uuid import uuid4

from agent_contracts import (
    AgentEvent,
    CompletionCommand,
    TaskRequested,
    TaskResult,
    classify_completion_failure,
)

from analysis_agent.application.ports import MaterialAnalyzer, MessageBus, TurnCompletionClient

logger = logging.getLogger(__name__)


class AnalysisExecutionError(Exception):
    """Raised when an expected task failure occurs during analysis."""

    def __init__(self, task_id: str, original_error: Exception) -> None:
        self.task_id = task_id
        self.original_error = original_error
        super().__init__(f"Analysis task {task_id} failed: {original_error}")


class AnalysisProcessor:
    """Assignment-backed delegated-child analysis turn processor.

    Every turn completes exclusively through the trusted Orchestrator
    completion boundary via the injected TurnCompletionClient. Canonical
    result/outcome keys are never written directly; failures submit only the
    platform-owned safe failure code derived from the local exception, which
    itself is never sent or persisted as canonical data.
    """

    def __init__(
        self,
        agent_id: str,
        bus: MessageBus,
        analyzer: MaterialAnalyzer,
        completion_client: TurnCompletionClient | None = None,
    ) -> None:
        if completion_client is None:
            raise TypeError("AnalysisProcessor requires a TurnCompletionClient")
        self.agent_id = agent_id
        self.bus = bus
        self.analyzer = analyzer
        self.completion_client = completion_client

    async def run_once(
        self,
        task: TaskRequested,
        *,
        assignment_id: str,
        assignment_proof: str,
        execution_kind: str = "delegated_child",
        continuation_version: int | None = None,
    ) -> None:
        """Execute one claimed delegated-child turn and submit its fenced completion."""
        if not assignment_id or not str(assignment_id).strip():
            raise ValueError("assignment_id must be a non-empty string")
        if not assignment_proof or not str(assignment_proof).strip():
            raise ValueError("assignment_proof must be a non-empty string")
        logger.info("Analysis agent %s processing task %s", self.agent_id, task.task_id)
        await self.bus.publish(
            AgentEvent.create(self.agent_id, "working", "Analyzing supplied material", task.task_id)
        )
        try:
            report = self.analyzer.analyze(task.prompt)
            command = CompletionCommand(
                completion_command_id=f"completion-{uuid4()}",
                assignment_id=str(assignment_id).strip(),
                execution_kind=execution_kind,
                continuation_version=continuation_version,
                completed_result=TaskResult(task_id=task.task_id, output=report.render()),
            )
        except Exception as exc:
            logger.exception("Analysis task %s failed during execution", task.task_id)
            failure_code = classify_completion_failure(exc)
            try:
                failure_command = CompletionCommand(
                    completion_command_id=f"completion-{uuid4()}",
                    assignment_id=str(assignment_id).strip(),
                    execution_kind=execution_kind,
                    continuation_version=continuation_version,
                    failure_code=failure_code,
                )
                await self.completion_client.submit_completion(
                    task_id=task.task_id,
                    command=failure_command,
                    assignment_proof=assignment_proof,
                )
            except Exception as submit_exc:  # noqa: BLE001 - best-effort failure submission must never mask the original error
                logger.error(
                    "Analysis agent %s failed to submit safe failure completion for task %s: %s",
                    self.agent_id,
                    task.task_id,
                    submit_exc,
                )
            await self.bus.publish(
                AgentEvent.create(self.agent_id, "error", "Task failed", task.task_id)
            )
            raise AnalysisExecutionError(task.task_id, exc) from exc

        try:
            await self.completion_client.submit_completion(
                task_id=task.task_id,
                command=command,
                assignment_proof=assignment_proof,
            )
        except Exception as submit_exc:
            logger.error(
                "Analysis agent %s failed to submit completion for task %s: %s",
                self.agent_id,
                task.task_id,
                submit_exc,
            )
            await self.bus.publish(
                AgentEvent.create(self.agent_id, "error", "Task failed", task.task_id)
            )
            raise AnalysisExecutionError(task.task_id, submit_exc) from submit_exc

        await self.bus.publish(
            AgentEvent.create(self.agent_id, "idle", "Task completed", task.task_id)
        )
        logger.info("Analysis agent %s completed task %s", self.agent_id, task.task_id)
