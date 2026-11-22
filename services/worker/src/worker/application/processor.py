from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agent_contracts import (
    AgentEvent,
    CompletionCommand,
    InferenceResponse,
    TaskRequested,
    TaskResult,
    classify_completion_failure,
)

from .ports import InferenceProvider, MessageBus, TurnCompletionClient

if TYPE_CHECKING:
    from worker.application.fallback import WorkerFallbackCoordinator

logger = logging.getLogger(__name__)


class TaskExecutionError(Exception):
    """Raised when an expected task failure occurs during execution."""

    def __init__(self, task_id: str, original_error: Exception) -> None:
        self.task_id = task_id
        self.original_error = original_error
        super().__init__(f"Task {task_id} failed: {original_error}")


class TaskProcessor:
    """Assignment-backed worker turn processor.

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
        inference_provider: InferenceProvider | None = None,
        fallback_coordinator: WorkerFallbackCoordinator | None = None,
        completion_client: TurnCompletionClient | None = None,
    ) -> None:
        if completion_client is None:
            raise TypeError("TaskProcessor requires a TurnCompletionClient")
        self.agent_id = agent_id
        self.bus = bus
        self.completion_client = completion_client
        self.inference_provider = inference_provider
        self.fallback_coordinator = fallback_coordinator

    async def process(self, task: TaskRequested) -> Any:
        """Execute the task prompt using the configured fallback coordinator or inference provider."""
        if self.fallback_coordinator is not None:
            return await self.fallback_coordinator.generate(task)
        if self.inference_provider is not None:
            res = await self.inference_provider.generate(task.prompt)
            if isinstance(res, InferenceResponse):
                return res.text
            return res
        return f"Completed: {task.prompt}"

    async def run_once(
        self,
        task: TaskRequested,
        *,
        assignment_id: str,
        assignment_proof: str,
        execution_kind: str = "initial",
        continuation_version: int | None = None,
    ) -> None:
        """Execute one claimed turn and submit its fenced completion."""
        if not assignment_id or not str(assignment_id).strip():
            raise ValueError("assignment_id must be a non-empty string")
        if not assignment_proof or not str(assignment_proof).strip():
            raise ValueError("assignment_proof must be a non-empty string")
        logger.info("Worker %s processing task %s", self.agent_id, task.task_id)
        await self.bus.publish(
            AgentEvent.create(
                self.agent_id, "working", "Processing task", task.task_id
            )
        )
        try:
            res = await self.process(task)
            if isinstance(res, TaskResult):
                result = res
            else:
                result_output = (
                    res
                    if isinstance(res, str) and res.strip()
                    else f"Completed: {task.prompt}"
                )
                result = TaskResult(task_id=task.task_id, output=result_output)

            command = CompletionCommand(
                completion_command_id=f"completion-{uuid4()}",
                assignment_id=str(assignment_id).strip(),
                execution_kind=execution_kind,
                continuation_version=continuation_version,
                completed_result=result,
            )
        except Exception as exc:
            logger.exception("Task %s failed during execution", task.task_id)
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
                    "Worker %s failed to submit safe failure completion for task %s: %s",
                    self.agent_id,
                    task.task_id,
                    submit_exc,
                )
            # 1. Record failure in event stream and set current state to error
            await self.bus.publish(
                AgentEvent.create(self.agent_id, "error", "Task failed", task.task_id)
            )
            # 2. Transition current state back to idle so the worker is available for next task
            await self.bus.publish(
                AgentEvent.create(self.agent_id, "idle", "Waiting for work")
            )
            raise TaskExecutionError(task.task_id, exc) from exc
        try:
            await self.completion_client.submit_completion(
                task_id=task.task_id,
                command=command,
                assignment_proof=assignment_proof,
            )
        except Exception as submit_exc:
            logger.error(
                "Worker %s failed to submit completion for task %s: %s",
                self.agent_id,
                task.task_id,
                submit_exc,
            )
            await self.bus.publish(
                AgentEvent.create(self.agent_id, "error", "Task failed", task.task_id)
            )
            await self.bus.publish(
                AgentEvent.create(self.agent_id, "idle", "Waiting for work")
            )
            raise TaskExecutionError(task.task_id, submit_exc) from submit_exc

        await self.bus.publish(
            AgentEvent.create(
                self.agent_id, "idle", "Task completed", task.task_id
            )
        )
        logger.info("Worker %s completed task %s successfully", self.agent_id, task.task_id)
