from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from agent_contracts import (
    AgentEvent,
    CompletionCommand,
    TaskRequested,
    TaskResult,
    classify_completion_failure,
)

from research_agent.application.ports import (
    InferenceProvider,
    MessageBus,
    ResearchExecutor,
    TurnCompletionClient,
)
from research_agent.domain.models import ResearchResult

if TYPE_CHECKING:
    from research_agent.application.fallback import ResearchFallbackCoordinator

logger = logging.getLogger(__name__)


class ResearchExecutionError(Exception):
    """Raised when an expected task failure occurs during research execution."""

    def __init__(self, task_id: str, original_error: Exception) -> None:
        self.task_id = task_id
        self.original_error = original_error
        super().__init__(f"Research task {task_id} failed: {original_error}")


class ResearchProcessor:
    """Assignment-backed research turn processor.

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
        executor: ResearchExecutor,
        completion_client: TurnCompletionClient | None = None,
        inference_provider: InferenceProvider | None = None,
        fallback_coordinator: ResearchFallbackCoordinator | None = None,
    ) -> None:
        if completion_client is None:
            raise TypeError("ResearchProcessor requires a TurnCompletionClient")
        self.agent_id = agent_id
        self.bus = bus
        self.executor = executor
        self.completion_client = completion_client
        self.inference_provider = inference_provider
        self.fallback_coordinator = fallback_coordinator

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
        logger.info("Research agent %s processing task %s", self.agent_id, task.task_id)
        await self.bus.publish(
            AgentEvent.create(
                self.agent_id, "working", "Processing task", task.task_id
            )
        )
        try:
            result = await self.executor.execute(task)

            if self.fallback_coordinator is not None:
                synthesis = await self.fallback_coordinator.synthesize(
                    question=task.prompt, evidence=result.output, task_id=task.task_id
                )
                result = ResearchResult(
                    task_id=task.task_id,
                    output=synthesis.content,
                    acknowledged=result.acknowledged,
                    truncated=synthesis.truncated,
                    sources=result.sources,
                )
            elif self.inference_provider is not None:
                synthesis = await self.inference_provider.synthesize(
                    question=task.prompt, evidence=result.output
                )
                result = ResearchResult(
                    task_id=task.task_id,
                    output=synthesis.content,
                    acknowledged=result.acknowledged,
                    truncated=synthesis.truncated,
                    sources=result.sources,
                )

            command = CompletionCommand(
                completion_command_id=f"completion-{uuid4()}",
                assignment_id=str(assignment_id).strip(),
                execution_kind=execution_kind,
                continuation_version=continuation_version,
                completed_result=TaskResult(
                    task_id=result.task_id,
                    output=result.output,
                    sources=[s.to_dict() for s in result.sources] or None,
                ),
            )
        except Exception as exc:
            logger.exception("Research task %s failed during execution", task.task_id)
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
                    "Research agent %s failed to submit safe failure completion for task %s: %s",
                    self.agent_id,
                    task.task_id,
                    submit_exc,
                )
            # Record failure in event stream and set current state to error.
            # Do not immediately overwrite error with idle.
            await self.bus.publish(
                AgentEvent.create(self.agent_id, "error", "Task failed", task.task_id)
            )
            raise ResearchExecutionError(task.task_id, exc) from exc

        try:
            await self.completion_client.submit_completion(
                task_id=task.task_id,
                command=command,
                assignment_proof=assignment_proof,
            )
        except Exception as submit_exc:
            logger.error(
                "Research agent %s failed to submit completion for task %s: %s",
                self.agent_id,
                task.task_id,
                submit_exc,
            )
            await self.bus.publish(
                AgentEvent.create(self.agent_id, "error", "Task failed", task.task_id)
            )
            raise ResearchExecutionError(task.task_id, submit_exc) from submit_exc

        await self.bus.publish(
            AgentEvent.create(
                self.agent_id, "idle", "Task completed", task.task_id
            )
        )
        logger.info(
            "Research agent %s completed task %s successfully",
            self.agent_id,
            task.task_id,
        )
