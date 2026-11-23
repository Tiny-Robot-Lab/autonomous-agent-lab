from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agent_contracts import (
    AgentEvent,
    CompletionCommand,
    TaskRequested,
    TaskResult,
    classify_completion_failure,
)

from research_agent.application.delegation import ResearchDelegationCoordinator
from research_agent.application.ports import (
    ContinuationContextReader,
    DelegationClient,
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
        context_reader: ContinuationContextReader | None = None,
        delegation_coordinator: ResearchDelegationCoordinator | None = None,
        delegation_client: DelegationClient | None = None,
    ) -> None:
        if completion_client is None:
            raise TypeError("ResearchProcessor requires a TurnCompletionClient")
        self.agent_id = agent_id
        self.bus = bus
        self.executor = executor
        self.completion_client = completion_client
        self.inference_provider = inference_provider
        self.fallback_coordinator = fallback_coordinator
        self.context_reader = context_reader
        self.delegation_coordinator = (
            delegation_coordinator
            if delegation_coordinator is not None
            else ResearchDelegationCoordinator("complete_locally")
        )
        self.delegation_client = delegation_client or (
            completion_client if isinstance(completion_client, DelegationClient) else None
        )

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
                self.agent_id,
                "working",
                "Processing task",
                task.task_id,
                assignment_id=assignment_id,
                execution_kind=execution_kind,
                continuation_version=continuation_version,
            )
        )
        try:
            if execution_kind == "resumed_continuation":
                if continuation_version != 1:
                    raise ValueError(
                        f"resumed_continuation requires continuation_version=1, got {continuation_version}"
                    )
                reader = self.context_reader or (
                    self.completion_client
                    if isinstance(self.completion_client, ContinuationContextReader)
                    else None
                )
                if reader is None:
                    raise RuntimeError("ContinuationContextReader is required for resumed_continuation execution")

                # 1. Authoritative TaskInput
                task_input = await reader.get_task_input(task.task_id, 1)
                if task_input is None:
                    raise RuntimeError(f"Parent TaskInput v1 not found for task '{task.task_id}'")

                # 2. Authoritative Continuation v1
                continuation = await reader.get_continuation(task.task_id, 1)
                if continuation is None or continuation.stage != "incorporate_analysis_child":
                    raise RuntimeError(
                        f"Continuation v1 with stage 'incorporate_analysis_child' not found for task '{task.task_id}'"
                    )
                child_task_id = continuation.child_task_id
                if not child_task_id:
                    raise RuntimeError(f"Child task ID missing in continuation v1 for task '{task.task_id}'")

                # 3. Sealed StageArtifact
                if not continuation.artifact_references:
                    raise RuntimeError(f"No artifact references in continuation v1 for task '{task.task_id}'")
                art_ref = continuation.artifact_references[0]
                stage_artifact = await reader.get_stage_artifact(
                    art_ref.producing_task_id, art_ref.artifact_id, art_ref.artifact_version
                )
                if stage_artifact is None or not stage_artifact.sealed:
                    raise RuntimeError(
                        f"Primary sealed stage artifact '{art_ref.artifact_id}' missing or unsealed"
                    )
                if not isinstance(stage_artifact.content, Mapping) or not stage_artifact.content:
                    raise RuntimeError(
                        f"Primary stage artifact '{art_ref.artifact_id}' content is missing or corrupt"
                    )

                primary_evidence_text = ""
                if "material" in stage_artifact.content:
                    primary_evidence_text = str(stage_artifact.content["material"])
                elif "summary" in stage_artifact.content:
                    primary_evidence_text = str(stage_artifact.content["summary"])
                elif "findings" in stage_artifact.content:
                    primary_evidence_text = str(stage_artifact.content["findings"])
                elif "text" in stage_artifact.content:
                    primary_evidence_text = str(stage_artifact.content["text"])
                else:
                    primary_evidence_text = json.dumps(stage_artifact.content)

                if not primary_evidence_text.strip():
                    raise RuntimeError("Primary research evidence text is empty")

                # 4. Child terminal outcome
                child_outcome = await reader.get_terminal_outcome(child_task_id)
                if child_outcome is None:
                    raise RuntimeError(f"Child outcome not found for child task '{child_task_id}'")

                # 5. Handle child outcomes: completed vs failed / claim_timed_out
                sources: list[Any] = []
                if child_outcome.terminal_status == "completed":
                    child_result = await reader.get_canonical_result(child_task_id)
                    if child_result is None or not child_result.output:
                        raise RuntimeError(
                            f"Completed child task '{child_task_id}' missing canonical result"
                        )
                    combined_evidence = (
                        f"Primary Research Evidence:\n{primary_evidence_text}\n\n"
                        f"Specialist Analysis Findings:\n{child_result.output}"
                    )
                    prompt_q = task_input.prompt
                    if self.fallback_coordinator is not None:
                        synthesis = await self.fallback_coordinator.synthesize(
                            question=prompt_q,
                            evidence=combined_evidence,
                            task_id=task.task_id,
                        )
                        final_output = synthesis.content
                    elif self.inference_provider is not None:
                        synthesis = await self.inference_provider.synthesize(
                            question=prompt_q, evidence=combined_evidence
                        )
                        final_output = synthesis.content
                    else:
                        final_output = (
                            "Synthesized research report combining primary findings and specialist analysis:\n\n"
                            f"{combined_evidence}"
                        )
                    if child_result.sources:
                        sources = list(child_result.sources)
                elif child_outcome.terminal_status in ("failed", "claim_timed_out"):
                    degraded_note = (
                        f"\n\n[Note: Specialist analysis was unavailable ({child_outcome.terminal_status}); "
                        "synthesizing final report based on primary sealed research evidence.]"
                    )
                    prompt_q = task_input.prompt
                    if self.fallback_coordinator is not None:
                        synthesis = await self.fallback_coordinator.synthesize(
                            question=prompt_q,
                            evidence=primary_evidence_text,
                            task_id=task.task_id,
                        )
                        final_output = synthesis.content + degraded_note
                    elif self.inference_provider is not None:
                        synthesis = await self.inference_provider.synthesize(
                            question=prompt_q, evidence=primary_evidence_text
                        )
                        final_output = synthesis.content + degraded_note
                    else:
                        final_output = primary_evidence_text + degraded_note
                else:
                    raise RuntimeError(
                        f"Unexpected child terminal status: '{child_outcome.terminal_status}'"
                    )

                result = ResearchResult(
                    task_id=task.task_id,
                    output=final_output,
                    acknowledged=True,
                    sources=sources,
                )
            else:
                result = await self.executor.execute(task)

                decision = self.delegation_coordinator.decide(task, execution_kind)
                if decision == "delegate_analysis":
                    client = self.delegation_client or (
                        self.completion_client
                        if isinstance(self.completion_client, DelegationClient)
                        else None
                    )
                    if client is None:
                        raise RuntimeError(
                            "DelegationClient is required for delegate_analysis execution"
                        )
                    evidence_text = result.output
                    if not evidence_text or not evidence_text.strip():
                        raise RuntimeError(
                            "Primary research evidence text is empty for delegation"
                        )
                    artifact_id = f"art-{uuid4().hex[:8]}"
                    await client.stage_artifact(
                        task_id=task.task_id,
                        assignment_id=assignment_id,
                        assignment_proof=assignment_proof,
                        artifact_id=artifact_id,
                        artifact_type="research_material",
                        content={"material": evidence_text},
                        artifact_version=1,
                        payload_schema_version=1,
                    )
                    proposal_id = f"prop-{uuid4().hex[:8]}"
                    art_ref = {
                        "artifact_id": artifact_id,
                        "producing_task_id": task.task_id,
                        "artifact_version": 1,
                    }
                    await client.propose_delegation(
                        task_id=task.task_id,
                        assignment_id=assignment_id,
                        assignment_proof=assignment_proof,
                        proposal_id=proposal_id,
                        child_intent="Analyze the sealed Research evidence for this delegated task.",
                        artifact_reference=art_ref,
                        target_capability="analysis",
                        parent_input_version=1,
                    )
                    await self.bus.publish(
                        AgentEvent.create(
                            self.agent_id,
                            "idle",
                            "Yielded turn for delegation",
                            task.task_id,
                            assignment_id=assignment_id,
                            execution_kind=execution_kind,
                            continuation_version=continuation_version,
                        )
                    )
                    logger.info(
                        "Research agent %s yielded turn for delegation on task %s",
                        self.agent_id,
                        task.task_id,
                    )
                    return

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
                    sources=[s.to_dict() if hasattr(s, "to_dict") else s for s in result.sources] or None,
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
                AgentEvent.create(
                    self.agent_id,
                    "error",
                    "Task failed",
                    task.task_id,
                    assignment_id=assignment_id,
                    execution_kind=execution_kind,
                    continuation_version=continuation_version,
                )
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
                AgentEvent.create(
                    self.agent_id,
                    "error",
                    "Task failed",
                    task.task_id,
                    assignment_id=assignment_id,
                    execution_kind=execution_kind,
                    continuation_version=continuation_version,
                )
            )
            raise ResearchExecutionError(task.task_id, submit_exc) from submit_exc

        await self.bus.publish(
            AgentEvent.create(
                self.agent_id,
                "idle",
                "Task completed",
                task.task_id,
                assignment_id=assignment_id,
                execution_kind=execution_kind,
                continuation_version=continuation_version,
            )
        )
        logger.info(
            "Research agent %s completed task %s successfully",
            self.agent_id,
            task.task_id,
        )
