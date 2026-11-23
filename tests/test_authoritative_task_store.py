import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from agent_contracts import (
    ActiveClaim,
    AgentAvailability,
    AgentAvailabilityState,
    AgentCandidate,
    AgentExecutionState,
    ArtifactReference,
    AssignmentCompletionRecord,
    AssignmentYieldRecord,
    ClaimExpiryRecord,
    CompletionCommand,
    CompletionRecord,
    DelegationGrant,
    DelegationProposalRecord,
    RecoveryDecisionCommand,
    RecoveryDecisionRecord,
    RequiredCapabilityRecord,
    RootAuthorityRecord,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskRequested,
    TaskResult,
    TerminalOutcome,
    WakeUpHint,
    build_claim_timed_out_terminal_outcome,
    build_completion_audit,
    build_terminal_outcome,
    compute_proposal_fingerprint,
    normalize_canonical_utc_timestamp,
)
from fastapi import Request
from orchestrator.application.ports import (
    AuthoritativeTaskStore,
    CapabilityMismatchError,
    ChildClaimDeadlineExpiredError,
    ClaimDeadlineNotExpiredError,
    ClaimDeliveryLeaseExpiredError,
    ClaimNotFoundError,
    ClaimTimeoutConflictError,
    CompletionConflictError,
    DelegationConflictError,
    FencingEpochRaceError,
    GrantAlreadyConsumedError,
    InvalidClaimProofError,
    RecoveryConflictError,
    RecoveryPreconditionFailedError,
    TaskAlreadyClaimedError,
    TaskDispatcher,
    TaskNotClaimableError,
    WaveBudgetExceededError,
)
from orchestrator.application.use_cases import RegisterChildTask, SubmitTask
from orchestrator.config import Settings
from orchestrator.domain.task import (
    ConflictingAuthorityGrantError,
    ConflictingChildTaskRegistrationError,
    ConflictingContinuationError,
    ConflictingRootAuthorityError,
    ConflictingRootTaskRegistrationError,
    ConflictingStageArtifactError,
    ConflictingTaskInputError,
    CorruptAuthoritativeRecordError,
    DelegationDepthExceededError,
    NonContiguousVersionError,
    ParentTaskNotFoundError,
    TaskLineage,
)
from orchestrator.infrastructure.redis_adapter import RedisGateway
from orchestrator.presentation.api import TaskInput as ApiTaskInput
from orchestrator.presentation.api import create_app, create_task

from tests.test_application import FakeTaskQueue


class InMemoryAuthoritativeTaskStore(AuthoritativeTaskStore):
    """Deterministic in-memory implementation of AuthoritativeTaskStore for tests."""

    def __init__(self) -> None:
        self.statuses: dict[str, str] = {}
        self.registrations: dict[str, dict[str, Any]] = {}
        self.lineages: dict[str, TaskLineage] = {}
        self.children: dict[str, set[str]] = {}
        self.inputs: dict[tuple[str, int], TaskInput] = {}
        self.artifacts: dict[tuple[str, str, int], StageArtifact] = {}
        self.continuations: dict[tuple[str, int], TaskContinuation] = {}
        self.latest_continuation: dict[str, int] = {}
        self.authorities: dict[str, RootAuthorityRecord] = {}
        self.authority_grants: dict[tuple[str, str], DelegationGrant] = {}
        self.required_capabilities: dict[str, RequiredCapabilityRecord] = {}
        self.claims: dict[str, ActiveClaim] = {}
        self.fences: dict[str, int] = {}
        self.proposals: dict[tuple[str, str], DelegationProposalRecord] = {}
        self.yield_records: dict[tuple[str, str], AssignmentYieldRecord] = {}
        self.consumed_waves: dict[str, set[str]] = {}
        self.consumed_grants: dict[str, set[str]] = {}
        self.claim_deadlines: dict[str, str] = {}
        self.hints: list[WakeUpHint] = []
        self.delegation_decisions: dict[str, set[str]] = {}
        self.completion_records: dict[tuple[str, str], CompletionRecord] = {}
        self.terminal_outcomes: dict[str, TerminalOutcome] = {}
        self.canonical_results: dict[str, TaskResult] = {}
        self.assignment_completions: dict[tuple[str, str], AssignmentCompletionRecord] = {}
        self.child_markers: dict[str, dict[str, str]] = {}
        self.claim_expiries: dict[str, ClaimExpiryRecord] = {}
        self.recovery_decisions: dict[tuple[str, str], RecoveryDecisionRecord] = {}
        self.fail_next_registration: bool = False
        self.root_registrations: list[tuple[TaskRequested, str]] = []
        self.child_registrations: list[tuple[TaskRequested, str]] = []

    async def register_root_task(
        self,
        task: TaskRequested,
        initial_status: str = "pending_dispatch",
        required_capability: str = "general",
    ) -> None:
        if task.parent_task_id is not None or task.root_task_id != task.task_id:
            raise ValueError(f"Task '{task.task_id}' is not a root task")
        if self.fail_next_registration:
            self.fail_next_registration = False
            raise RuntimeError("Simulated pipeline failure during root registration")

        required_cap_record = RequiredCapabilityRecord(
            task_id=task.task_id,
            required_capability=required_capability,
            created_at=task.created_at,
            version=1,
        )
        exists_marker = task.task_id in self.registrations
        exists_status = task.task_id in self.statuses
        exists_lineage = task.task_id in self.lineages
        exists_input = (task.task_id, 1) in self.inputs
        exists_authority = task.task_id in self.authorities
        exists_cap = task.task_id in self.required_capabilities

        if not (exists_marker or exists_status or exists_lineage or exists_input or exists_authority or exists_cap):
            self.registrations[task.task_id] = {
                "registered_at": task.created_at,
                "task_id": task.task_id,
            }
            self.statuses[task.task_id] = initial_status
            self.lineages[task.task_id] = TaskLineage(
                task_id=task.task_id,
                root_task_id=task.root_task_id,
                parent_task_id=None,
            )
            self.inputs[(task.task_id, 1)] = TaskInput(
                task_id=task.task_id,
                prompt=task.prompt,
                created_at=task.created_at,
                input_version=1,
                artifact_references=(),
                metadata={},
                version=1,
            )
            self.authorities[task.task_id] = RootAuthorityRecord.create_default(
                root_task_id=task.task_id,
                created_at=task.created_at,
            )
            self.required_capabilities[task.task_id] = required_cap_record
            self.root_registrations.append((task, initial_status))
            return

        if exists_marker and exists_status and exists_lineage and exists_input and exists_authority and exists_cap:
            existing_marker = self.registrations[task.task_id]
            existing_lineage = self.lineages[task.task_id]
            existing_input = self.inputs[(task.task_id, 1)]
            existing_authority = self.authorities[task.task_id]
            existing_status = self.statuses[task.task_id]
            existing_cap = self.required_capabilities[task.task_id]
            if (
                existing_marker.get("task_id") == task.task_id
                and existing_lineage.root_task_id == task.root_task_id
                and existing_input.prompt == task.prompt
                and existing_authority.root_task_id == task.task_id
                and existing_cap.required_capability == required_cap_record.required_capability
                and isinstance(existing_status, str)
            ):
                return
            raise ConflictingRootTaskRegistrationError(
                f"Root task '{task.task_id}' already exists with conflicting state"
            )

        raise ConflictingRootTaskRegistrationError(
            f"Root task '{task.task_id}' has inconsistent or partial authoritative state in storage"
        )

    async def register_child_task(
        self,
        task: TaskRequested,
        initial_status: str = "pending_dispatch",
        required_capability: str = "analysis",
    ) -> None:
        if task.parent_task_id is None or task.root_task_id == task.task_id:
            raise ValueError(f"Task '{task.task_id}' is not a child task")
        if self.fail_next_registration:
            self.fail_next_registration = False
            raise RuntimeError("Simulated pipeline failure during child registration")
        cap_rec = RequiredCapabilityRecord(
            task_id=task.task_id,
            required_capability=required_capability,
            created_at=task.created_at,
            version=1,
        )
        exists_status = task.task_id in self.statuses
        exists_lineage = task.task_id in self.lineages
        exists_capability = task.task_id in self.required_capabilities
        is_member = task.task_id in self.children.get(task.parent_task_id, set())

        # 1. All records absent: create atomically
        if not exists_status and not exists_lineage and not exists_capability and not is_member:
            self.statuses[task.task_id] = initial_status
            self.lineages[task.task_id] = TaskLineage(
                task_id=task.task_id,
                root_task_id=task.root_task_id,
                parent_task_id=task.parent_task_id,
            )
            self.required_capabilities[task.task_id] = cap_rec
            self.inputs[(task.task_id, 1)] = TaskInput(
                task_id=task.task_id,
                prompt=task.prompt,
                created_at=task.created_at,
                input_version=1,
                artifact_references=(),
                metadata={},
                version=1,
            )
            self.children.setdefault(task.parent_task_id, set()).add(task.task_id)
            self.child_registrations.append((task, initial_status))
            return

        # 2. All records present and byte-identical: idempotent success
        if exists_status and exists_lineage and exists_capability and is_member:
            cur_lineage = self.lineages[task.task_id]
            cur_cap = self.required_capabilities[task.task_id]
            if (
                cur_lineage.parent_task_id == task.parent_task_id
                and cur_lineage.root_task_id == task.root_task_id
                and cur_lineage.task_id == task.task_id
                and cur_cap.required_capability == cap_rec.required_capability
            ):
                return
            raise ConflictingChildTaskRegistrationError(
                f"Child task '{task.task_id}' already exists with conflicting state"
            )

        # 3. Partial existing registration: fail closed
        raise ConflictingChildTaskRegistrationError(
            f"Child task '{task.task_id}' has inconsistent or partial authoritative state in storage"
        )

    async def get_required_capability(
        self, task_id: str
    ) -> RequiredCapabilityRecord | None:
        return self.required_capabilities.get(task_id.strip())

    async def get_lineage(self, task_id: str) -> TaskLineage | None:
        return self.lineages.get(task_id.strip())

    async def get_children(self, parent_task_id: str) -> list[str]:
        return sorted(self.children.get(parent_task_id.strip(), set()))

    async def update_status(self, task_id: str, status: str) -> None:
        clean_id = task_id.strip()
        if not clean_id:
            raise ValueError("task_id must be a non-empty string")
        self.statuses[clean_id] = status.strip()

    async def get_status(self, task_id: str, now: str = "") -> str | None:
        clean_task_id = task_id.strip()
        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )
        deadline = self.claim_deadlines.get(clean_task_id)
        if deadline and now_iso > deadline and clean_task_id not in self.terminal_outcomes:
            try:
                await self.materialize_claim_timeout(clean_task_id, now_iso)
            except Exception:  # noqa: BLE001, S110
                pass
        claim = self.claims.get(clean_task_id)
        if (
            claim
            and claim.state == "acknowledged"
            and claim.execution_expires_at
            and now_iso > claim.execution_expires_at
            and clean_task_id not in self.terminal_outcomes
        ):
            try:
                await self.transition_expired_claim(clean_task_id, now_iso)
            except Exception:  # noqa: BLE001, S110
                pass
        outcome = self.terminal_outcomes.get(clean_task_id)
        if outcome is not None:
            return outcome.terminal_status
        return self.statuses.get(clean_task_id)

    async def get_result(self, task_id: str, now: str = "") -> dict[str, Any]:
        clean_task_id = task_id.strip()
        status = await self.get_status(clean_task_id, now=now)
        res = self.canonical_results.get(clean_task_id)
        return {
            "task_id": clean_task_id,
            "status": status,
            "output": res.output if res else None,
        }

    async def save_task_input(self, task_input: TaskInput) -> None:
        if task_input.input_version > 1:
            prior_key = (task_input.task_id, task_input.input_version - 1)
            if prior_key not in self.inputs:
                raise NonContiguousVersionError(
                    f"Cannot save TaskInput version {task_input.input_version} for task '{task_input.task_id}' "
                    f"because prior version {task_input.input_version - 1} does not exist"
                )
        key = (task_input.task_id, task_input.input_version)
        if key in self.inputs:
            if self.inputs[key] == task_input:
                return
            raise ConflictingTaskInputError(
                f"TaskInput for task '{task_input.task_id}' version {task_input.input_version} "
                "already exists with conflicting data"
            )
        self.inputs[key] = task_input

    async def get_task_input(
        self, task_id: str, input_version: int = 1
    ) -> TaskInput | None:
        return self.inputs.get((task_id.strip(), input_version))

    async def save_stage_artifact(
        self,
        artifact: StageArtifact,
        expected_assignment_id: str,
        expected_principal_id: str,
        expected_fencing_epoch: int,
    ) -> None:
        clean_assignment_id = str(expected_assignment_id).strip()
        clean_principal_id = str(expected_principal_id).strip()
        if not clean_assignment_id or not clean_principal_id or expected_fencing_epoch < 1:
            raise ValueError(
                "save_stage_artifact requires non-empty assignment_id, principal_id, and positive fencing_epoch"
            )

        claim = self.claims.get(artifact.producing_task_id)
        if not claim:
            raise ClaimNotFoundError(f"Active claim for task '{artifact.producing_task_id}' not found")
        if claim.assignment_id != clean_assignment_id or claim.runtime_principal_id != clean_principal_id:
            raise ClaimNotFoundError(
                f"Active claim does not match assignment '{clean_assignment_id}' or principal '{clean_principal_id}'"
            )
        if claim.state != "acknowledged":
            raise TaskNotClaimableError("Assignment must be acknowledged before staging artifacts")
        if claim.fencing_epoch != expected_fencing_epoch:
            raise FencingEpochRaceError(
                f"Fencing epoch mismatch: expected {expected_fencing_epoch}, claim has {claim.fencing_epoch}"
            )
        cur_fence = self.fences.get(artifact.producing_task_id, 0)
        if cur_fence != expected_fencing_epoch:
            raise FencingEpochRaceError(f"Fencing epoch mismatch: current fence is {cur_fence}")
        if artifact.artifact_type == "research_material" and (
            claim.execution_kind != "initial" or claim.assigned_capability != "research"
        ):
            raise CapabilityMismatchError(
                f"Artifact staging policy violation for artifact_type '{artifact.artifact_type}'"
            )

        if artifact.artifact_version > 1:
            prior_key = (artifact.producing_task_id, artifact.artifact_id, artifact.artifact_version - 1)
            if prior_key not in self.artifacts:
                raise NonContiguousVersionError(
                    f"Cannot save StageArtifact version {artifact.artifact_version} for task '{artifact.producing_task_id}', "
                    f"artifact '{artifact.artifact_id}' because prior version {artifact.artifact_version - 1} does not exist"
                )
        key = (artifact.producing_task_id, artifact.artifact_id, artifact.artifact_version)
        if key in self.artifacts:
            if self.artifacts[key] == artifact:
                return
            raise ConflictingStageArtifactError(
                f"StageArtifact '{artifact.artifact_id}' for task '{artifact.producing_task_id}' "
                f"version {artifact.artifact_version} already exists with conflicting data"
            )
        self.artifacts[key] = artifact

    async def get_stage_artifact(
        self, producing_task_id: str, artifact_id: str, artifact_version: int = 1
    ) -> StageArtifact | None:
        return self.artifacts.get((producing_task_id.strip(), artifact_id.strip(), artifact_version))

    async def save_continuation(self, continuation: TaskContinuation) -> None:
        key = (continuation.parent_task_id, continuation.continuation_version)
        if key not in self.continuations:
            if continuation.continuation_version == 1:
                if continuation.parent_task_id in self.latest_continuation:
                    raise CorruptAuthoritativeRecordError(
                        "Continuation latest pointer already exists for version 1 creation"
                    )
            else:
                if continuation.parent_task_id not in self.latest_continuation:
                    raise CorruptAuthoritativeRecordError(
                        f"Continuation latest pointer is missing for version {continuation.continuation_version}"
                    )
                cur_latest = self.latest_continuation[continuation.parent_task_id]
                if cur_latest >= continuation.continuation_version:
                    raise CorruptAuthoritativeRecordError(
                        f"Continuation latest pointer is corrupt: {cur_latest}"
                    )
                if cur_latest < continuation.continuation_version - 1:
                    raise NonContiguousVersionError(
                        f"Cannot save TaskContinuation version {continuation.continuation_version} "
                        f"for parent task '{continuation.parent_task_id}' because prior version "
                        f"{continuation.continuation_version - 1} does not exist"
                    )
                prior_key = (continuation.parent_task_id, continuation.continuation_version - 1)
                if prior_key not in self.continuations:
                    raise NonContiguousVersionError(
                        f"Cannot save TaskContinuation version {continuation.continuation_version} "
                        f"for parent task '{continuation.parent_task_id}' because prior version "
                        f"{continuation.continuation_version - 1} does not exist"
                    )
            self.continuations[key] = continuation
            self.latest_continuation[continuation.parent_task_id] = continuation.continuation_version
            return

        if self.continuations[key] == continuation:
            if continuation.parent_task_id not in self.latest_continuation:
                raise CorruptAuthoritativeRecordError(
                    "Continuation latest pointer is missing during replay"
                )
            cur_latest = self.latest_continuation[continuation.parent_task_id]
            if cur_latest < continuation.continuation_version:
                raise CorruptAuthoritativeRecordError(
                    f"Continuation latest pointer is corrupt during replay: {cur_latest}"
                )
            return
        raise ConflictingContinuationError(
            f"TaskContinuation for parent task '{continuation.parent_task_id}' "
            f"version {continuation.continuation_version} already exists with conflicting data"
        )

    async def get_continuation(
        self, parent_task_id: str, continuation_version: int = 1
    ) -> TaskContinuation | None:
        return self.continuations.get((parent_task_id.strip(), continuation_version))

    async def save_root_authority(self, authority: RootAuthorityRecord) -> None:
        key = authority.root_task_id
        if key in self.authorities:
            if self.authorities[key] == authority:
                return
            raise ConflictingRootAuthorityError(
                f"RootAuthorityRecord for root task '{authority.root_task_id}' "
                "already exists with conflicting data"
            )
        self.authorities[key] = authority

    async def get_root_authority(self, root_task_id: str) -> RootAuthorityRecord | None:
        return self.authorities.get(root_task_id.strip())

    async def save_authority_grant(self, grant: DelegationGrant) -> None:
        key = (grant.root_task_id, grant.grant_id)
        if key in self.authority_grants:
            if self.authority_grants[key] == grant:
                return
            raise ConflictingAuthorityGrantError(
                f"DelegationGrant for root task '{grant.root_task_id}' and grant '{grant.grant_id}' "
                "already exists with conflicting data"
            )
        self.authority_grants[key] = grant

    async def get_authority_grant(
        self, root_task_id: str, grant_id: str
    ) -> DelegationGrant | None:
        return self.authority_grants.get((root_task_id.strip(), grant_id.strip()))

    async def save_active_claim(self, claim: ActiveClaim) -> None:
        self.claims[claim.task_id] = claim

    async def get_active_claim(self, task_id: str) -> ActiveClaim | None:
        return self.claims.get(task_id.strip())

    async def get_fencing_epoch(self, task_id: str) -> int:
        return self.fences.get(task_id.strip(), 0)

    async def atomic_claim_task(
        self,
        task_id: str,
        claim: ActiveClaim,
        grant: DelegationGrant | None = None,
        now: str = "",
        expected_continuation_version: int = -1,
        is_root: bool = True,
    ) -> tuple[ActiveClaim, bool]:
        clean_task_id = task_id.strip()
        cur_status = self.statuses.get(clean_task_id)
        if not cur_status:
            raise TaskNotClaimableError(f"Task '{clean_task_id}' not found")

        req_cap = self.required_capabilities.get(clean_task_id)
        if req_cap and req_cap.required_capability != claim.assigned_capability:
            raise CapabilityMismatchError(
                f"Task '{clean_task_id}' requires capability '{req_cap.required_capability}', but requested '{claim.assigned_capability}'"
            )

        existing_claim = self.claims.get(clean_task_id)
        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )
        effective_status = cur_status

        if existing_claim:
            if (
                existing_claim.runtime_principal_id == claim.runtime_principal_id
                and existing_claim.request_id == claim.request_id
            ):
                if existing_claim.state == "acknowledged":
                    if existing_claim.execution_expires_at and now_iso > existing_claim.execution_expires_at:
                        await self.transition_expired_claim(clean_task_id, now_iso)
                        raise TaskNotClaimableError(
                            f"Task '{clean_task_id}' execution claim has expired and transitioned to recovery_required"
                        )
                    raise TaskAlreadyClaimedError(f"Task '{clean_task_id}' assignment is already acknowledged")
                if (
                    existing_claim.state == "claimed_unacknowledged"
                    and existing_claim.delivery_expires_at <= now_iso
                ):
                    effective_status = existing_claim.prior_claimable_status
                else:
                    return (existing_claim, True)
            else:
                if (
                    existing_claim.state == "claimed_unacknowledged"
                    and existing_claim.delivery_expires_at <= now_iso
                ):
                    effective_status = existing_claim.prior_claimable_status
                elif (
                    existing_claim.state == "acknowledged"
                    and existing_claim.execution_expires_at
                    and now_iso > existing_claim.execution_expires_at
                ):
                    await self.transition_expired_claim(clean_task_id, now_iso)
                    raise TaskNotClaimableError(
                        f"Task '{clean_task_id}' execution claim has expired and transitioned to recovery_required"
                    )
                else:
                    raise TaskAlreadyClaimedError(f"Task '{clean_task_id}' has an active claim")

        if claim.execution_kind == "initial":
            if not is_root or effective_status != "pending":
                raise TaskNotClaimableError(f"Task '{clean_task_id}' lifecycle mismatch for initial execution: status='{effective_status}', is_root={is_root}")
            if expected_continuation_version is not None and expected_continuation_version > 0:
                raise ValueError("continuation_version must remain absent for non-resumed claims")
        elif claim.execution_kind == "delegated_child":
            if is_root or effective_status != "pending":
                raise TaskNotClaimableError(f"Task '{clean_task_id}' lifecycle mismatch for delegated_child execution: status='{effective_status}', is_root={is_root}")
            if expected_continuation_version is not None and expected_continuation_version > 0:
                raise ValueError("continuation_version must remain absent for non-resumed claims")
            deadline = self.claim_deadlines.get(clean_task_id)
            if deadline and now_iso > deadline:
                try:
                    await self.materialize_claim_timeout(clean_task_id, now_iso)
                except Exception:  # noqa: BLE001, S110
                    pass
                raise ChildClaimDeadlineExpiredError(
                    f"Claim deadline for child task '{clean_task_id}' has expired ({deadline})"
                )
        elif claim.execution_kind == "resumed_continuation":
            if not is_root or effective_status != "resumable":
                raise TaskNotClaimableError(f"Task '{clean_task_id}' lifecycle mismatch for resumed_continuation execution: status='{effective_status}', is_root={is_root}")
            if (
                expected_continuation_version is None
                or not isinstance(expected_continuation_version, int)
                or type(expected_continuation_version) is bool
                or expected_continuation_version < 1
            ):
                raise ValueError("expected_continuation_version must be a positive integer for resumed_continuation")
            if clean_task_id not in self.latest_continuation:
                raise TaskNotClaimableError(f"No continuation checkpoint found for task '{clean_task_id}'")
            cur_latest = self.latest_continuation[clean_task_id]
            if (
                cur_latest is None
                or not isinstance(cur_latest, (int, float))
                or type(cur_latest) is bool
                or cur_latest < 1
                or int(cur_latest) != cur_latest
            ):
                raise CorruptAuthoritativeRecordError(f"Corrupt continuation latest record for task '{clean_task_id}'")
            if int(cur_latest) != int(expected_continuation_version):
                raise TaskNotClaimableError(f"Continuation version mismatch for task '{clean_task_id}': latest={cur_latest}, expected={expected_continuation_version}")

        cur_epoch = self.fences.get(clean_task_id, 0)
        expected_epoch = cur_epoch + 1
        if claim.fencing_epoch != expected_epoch:
            raise FencingEpochRaceError(
                f"Concurrent claim conflict on task '{clean_task_id}' (fencing epoch race)"
            )

        if grant:
            key = (grant.root_task_id, grant.grant_id)
            if key in self.authority_grants and self.authority_grants[key] != grant:
                raise ConflictingAuthorityGrantError(
                    f"DelegationGrant for root task '{grant.root_task_id}' and grant '{grant.grant_id}' "
                    "already exists with conflicting data"
                )

        # Commit phase (no failures permitted)
        self.fences[clean_task_id] = expected_epoch
        self.statuses[clean_task_id] = "working"
        self.claims[clean_task_id] = claim

        if grant:
            self.authority_grants[(grant.root_task_id, grant.grant_id)] = grant

        return (claim, False)

    async def acknowledge_claim(
        self,
        task_id: str,
        assignment_id: str,
        principal_id: str,
        acknowledged_at: str,
        execution_expires_at: str | None = None,
    ) -> ActiveClaim:
        clean_task_id = task_id.strip()
        claim = self.claims.get(clean_task_id)
        if not claim:
            raise ClaimNotFoundError(f"Active claim for task '{clean_task_id}' not found")
        if claim.assignment_id != assignment_id.strip() or claim.runtime_principal_id != principal_id.strip():
            raise ClaimNotFoundError("Assignment ID or principal mismatch")
        if claim.state == "acknowledged":
            return claim
        clean_now = normalize_canonical_utc_timestamp(acknowledged_at)
        if claim.delivery_expires_at <= clean_now:
            raise ClaimDeliveryLeaseExpiredError("Lease expired")

        ack_dict = claim.to_dict()
        ack_dict["state"] = "acknowledged"
        ack_dict["acknowledged_at"] = clean_now
        if execution_expires_at is not None:
            ack_dict["execution_expires_at"] = normalize_canonical_utc_timestamp(execution_expires_at)
        updated = ActiveClaim.from_dict(ack_dict)
        self.claims[clean_task_id] = updated
        return updated

    async def expire_delivery_lease(self, task_id: str, now: str = "") -> bool:
        clean_task_id = task_id.strip()
        claim = self.claims.get(clean_task_id)
        if not claim:
            return False
        if not hasattr(claim, "state") or not hasattr(claim, "delivery_expires_at"):
            raise CorruptAuthoritativeRecordError(f"Corrupt active claim on task '{clean_task_id}' during lease expiry")
        if claim.state not in ("claimed_unacknowledged", "acknowledged"):
            raise CorruptAuthoritativeRecordError(f"Corrupt active claim on task '{clean_task_id}' during lease expiry")
        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )
        if claim.state == "claimed_unacknowledged" and claim.delivery_expires_at <= now_iso:
            del self.claims[clean_task_id]
            self.statuses[clean_task_id] = claim.prior_claimable_status
            return True
        return False

    async def accept_delegation_proposal(
        self,
        parent_task_id: str,
        proposal: DelegationProposalRecord,
        child_task_input: TaskInput,
        parent_continuation: TaskContinuation,
        yield_record: AssignmentYieldRecord,
        wake_up_hint: WakeUpHint,
        expected_fencing_epoch: int,
        now: str,
    ) -> tuple[DelegationProposalRecord, bool]:
        clean_parent = parent_task_id.strip()
        proposal_key = (clean_parent, proposal.proposal_id.strip())

        # Phase 1: Replay check
        existing = self.proposals.get(proposal_key)
        if existing is not None:
            if (
                existing.assignment_id == proposal.assignment_id
                and existing.runtime_principal_id == proposal.runtime_principal_id
                and existing.bound_agent_id == proposal.bound_agent_id
                and existing.parent_task_id == clean_parent
                and existing.proposal_id == proposal.proposal_id
                and existing.proposal_fingerprint == proposal.proposal_fingerprint
            ):
                return (existing, True)
            raise DelegationConflictError(
                f"Conflicting proposal replay for proposal '{proposal.proposal_id}'"
            )

        # Phase 1: Validations
        cur_status = self.statuses.get(clean_parent)
        if not cur_status:
            raise ParentTaskNotFoundError(f"Parent task '{clean_parent}' not found")
        if cur_status != "working":
            raise TaskNotClaimableError(
                f"Parent task '{clean_parent}' must be in 'working' status, got '{cur_status}'"
            )

        waves = self.consumed_waves.setdefault(clean_parent, set())
        if "1" in waves:
            raise WaveBudgetExceededError(f"Wave budget exceeded for parent task '{clean_parent}'")

        grants = self.consumed_grants.setdefault(clean_parent, set())
        if proposal.delegation_grant_id in grants:
            raise GrantAlreadyConsumedError(
                f"Authority grant '{proposal.delegation_grant_id}' has already been consumed"
            )

        claim = self.claims.get(clean_parent)
        if not claim:
            raise ClaimNotFoundError(f"Active claim for parent task '{clean_parent}' not found")
        if claim.assignment_id != proposal.assignment_id or claim.runtime_principal_id != proposal.runtime_principal_id:
            raise ClaimNotFoundError("Assignment ID or principal mismatch on parent claim")
        if claim.bound_agent_id != proposal.bound_agent_id:
            raise ClaimNotFoundError("Bound agent mismatch on parent claim")
        if claim.state != "acknowledged":
            raise TaskNotClaimableError("Parent assignment must be acknowledged before proposing delegation")
        if claim.assigned_capability != "research":
            raise CapabilityMismatchError("Only research assignments may propose delegation")
        if claim.execution_kind != "initial":
            raise TaskNotClaimableError("Only initial research root claims may propose delegation")

        cur_fence = self.fences.get(clean_parent, 0)
        if cur_fence != expected_fencing_epoch or claim.fencing_epoch != expected_fencing_epoch:
            raise FencingEpochRaceError(
                f"Fencing epoch mismatch for parent task '{clean_parent}': expected {expected_fencing_epoch}, current {cur_fence}"
            )

        parent_lineage = self.lineages.get(clean_parent)
        if (
            not parent_lineage
            or not parent_lineage.is_root
            or parent_lineage.task_id != clean_parent
            or parent_lineage.root_task_id != clean_parent
            or bool(parent_lineage.parent_task_id)
        ):
            raise TaskNotClaimableError(f"Parent task '{clean_parent}' is not an eligible root task")

        # Verify root authority record
        root_auth = self.authorities.get(clean_parent)
        if not root_auth:
            raise ParentTaskNotFoundError(f"Root authority record for '{clean_parent}' not found")
        if root_auth.root_task_id != clean_parent:
            raise DelegationConflictError("Root authority binding mismatch")
        d = getattr(root_auth, "max_delegation_depth", None)
        w = getattr(root_auth, "max_delegation_waves", None)
        c = getattr(root_auth, "max_children_per_wave", None)
        if (
            type(d) is not int or d != 1
            or type(w) is not int or w != 1
            or type(c) is not int or c != 1
        ):
            raise WaveBudgetExceededError("Root authority policy bounds violation")
        if ("research", "analysis") not in root_auth.allowed_delegation_edges:
            raise DelegationConflictError("Delegation edge research->analysis not permitted by root authority")

        # Verify delegation grant record
        grant_rec = self.authority_grants.get((clean_parent, proposal.delegation_grant_id))
        if not grant_rec:
            raise ParentTaskNotFoundError(f"Delegation grant '{proposal.delegation_grant_id}' not found")
        if grant_rec.grant_id != proposal.delegation_grant_id:
            raise DelegationConflictError("Delegation grant ID mismatch")
        if grant_rec.root_task_id != clean_parent:
            raise DelegationConflictError("Delegation grant root task binding mismatch")
        if grant_rec.issued_to_assignment_id != proposal.assignment_id:
            raise ClaimNotFoundError("Delegation grant issued assignment mismatch")
        if grant_rec.issued_to_principal_id != proposal.runtime_principal_id:
            raise ClaimNotFoundError("Delegation grant issued principal mismatch")
        if not claim.turn_id or not grant_rec.turn_id or grant_rec.turn_id != claim.turn_id:
            raise ClaimNotFoundError("Delegation grant turn mismatch")
        if grant_rec.source_capability != "research" or grant_rec.target_capability != "analysis":
            raise CapabilityMismatchError("Delegation grant capability mismatch")
        if grant_rec.fencing_epoch != expected_fencing_epoch:
            raise FencingEpochRaceError("Delegation grant fencing epoch mismatch")
        gd = getattr(grant_rec, "max_depth", None)
        gw = getattr(grant_rec, "max_waves", None)
        gc = getattr(grant_rec, "max_children", None)
        if (
            type(gd) is not int or gd != 1
            or type(gw) is not int or gw != 1
            or type(gc) is not int or gc != 1
        ):
            raise WaveBudgetExceededError("Delegation grant bounds violation")

        # Verify sealed artifact exists
        if child_task_input.artifact_references:
            art_ref = child_task_input.artifact_references[0]
            art = self.artifacts.get((art_ref.producing_task_id, art_ref.artifact_id, art_ref.artifact_version))
            if not art or not art.sealed:
                raise ValueError("Referenced artifact not found or not sealed")
            if art.producing_task_id != clean_parent:
                raise ValueError("Artifact producing task mismatch")
            if art.artifact_type != "research_material":
                raise ValueError(f"Artifact type must be research_material, got {art.artifact_type}")
            if art.artifact_id != art_ref.artifact_id or art.artifact_version != art_ref.artifact_version:
                raise ValueError("Artifact identity mismatch")
            if not art.producing_turn_id or type(art.producing_turn_id) is not str:
                raise ValueError("Artifact producing_turn_id missing")
            if art.producing_turn_id != claim.turn_id:
                raise ValueError("Artifact producing_turn_id does not match active claim turn_id")
            if type(art.payload_schema_version) is not int or art.payload_schema_version != 1:
                raise ValueError(f"Unsupported artifact payload_schema_version: {art.payload_schema_version}")
            if type(art.version) is not int or art.version != 1:
                raise ValueError(f"Unsupported artifact contract version: {art.version}")
        else:
            raise ValueError("Child task input must reference at least one sealed artifact")

        # Verify parent TaskInput version exists and is valid
        parent_input = self.inputs.get((clean_parent, parent_continuation.input_version))
        if parent_input is None:
            raise ValueError(f"Parent TaskInput version {parent_continuation.input_version} not found")
        if parent_input.task_id != clean_parent:
            raise ValueError("Parent TaskInput task_id mismatch")
        if parent_input.input_version != parent_continuation.input_version:
            raise ValueError("Parent TaskInput input_version mismatch")
        if type(parent_input.version) is not int or parent_input.version != 1:
            raise CorruptAuthoritativeRecordError(f"Unsupported TaskInput contract version: {parent_input.version}")
        if not parent_input.prompt or not isinstance(parent_input.prompt, str):
            raise CorruptAuthoritativeRecordError("Parent TaskInput prompt is invalid")

        # Verify immutable target keys do not exist before write
        clean_child = proposal.child_task_id.strip()
        if (
            clean_child in self.statuses
            or clean_child in self.lineages
            or clean_child in self.required_capabilities
            or (clean_child, 1) in self.inputs
            or clean_child in self.claim_deadlines
            or clean_child in self.fences
        ):
            raise ConflictingChildTaskRegistrationError(f"Child task '{clean_child}' already exists")

        if (clean_parent, yield_record.assignment_id) in self.yield_records:
            raise ConflictingChildTaskRegistrationError(f"Yield record for '{clean_parent}' already exists")

        # Verify continuation version 1 does not exist
        if (clean_parent, 1) in self.continuations or clean_parent in self.latest_continuation:
            raise ConflictingContinuationError(f"Continuation version 1 already exists for task '{clean_parent}'")

        # Phase 2: Commits
        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )
        self.proposals[proposal_key] = proposal
        self.statuses[clean_child] = "pending"
        self.claim_deadlines[clean_child] = proposal.child_claim_deadline
        self.fences[clean_child] = 0
        self.required_capabilities[clean_child] = RequiredCapabilityRecord(
            task_id=clean_child,
            required_capability="analysis",
            created_at=now_iso,
            version=1,
        )
        self.lineages[clean_child] = TaskLineage(
            task_id=clean_child,
            root_task_id=parent_lineage.root_task_id,
            parent_task_id=clean_parent,
        )
        self.inputs[(clean_child, 1)] = child_task_input
        self.children.setdefault(clean_parent, set()).add(clean_child)
        self.continuations[(clean_parent, 1)] = parent_continuation
        self.latest_continuation[clean_parent] = 1
        self.statuses[clean_parent] = "waiting_for_children"
        del self.claims[clean_parent]
        self.yield_records[(clean_parent, yield_record.assignment_id)] = yield_record
        waves.add("1")
        grants.add(proposal.delegation_grant_id)
        self.delegation_decisions.setdefault(clean_parent, set()).add(proposal.delegation_decision_id)
        self.hints.append(wake_up_hint)

        return (proposal, False)

    async def complete_turn(
        self,
        task_id: str,
        command: CompletionCommand,
        proof_digest: str,
        fencing_epoch: int,
        parent_task_id: str | None,
        root_task_id: str,
        completed_at: str,
        runtime_principal_id: str,
        bound_agent_id: str,
    ) -> tuple[TerminalOutcome, TaskResult | None, bool]:
        clean_task = str(task_id).strip()
        clean_root = str(root_task_id).strip()
        if not clean_task or not clean_root:
            raise ValueError("task_id and root_task_id must be non-empty strings")
        clean_principal = str(runtime_principal_id).strip()
        clean_agent = str(bound_agent_id).strip()
        clean_digest = str(proof_digest).strip()
        if not clean_principal or not clean_agent or not clean_digest:
            raise ValueError("runtime_principal_id, bound_agent_id, and proof_digest must be non-empty strings")

        now_iso = (
            normalize_canonical_utc_timestamp(completed_at)
            if (completed_at and str(completed_at).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )

        is_child = command.execution_kind == "delegated_child"
        clean_parent = str(parent_task_id).strip() if parent_task_id else ""
        if is_child and not clean_parent:
            raise ValueError("parent_task_id is required for delegated_child completion")

        if command.completed_result is not None:
            if command.completed_result.task_id != clean_task:
                raise ValueError(
                    f"Completed TaskResult task_id '{command.completed_result.task_id}' "
                    f"does not match completion task '{clean_task}'"
                )
            if command.completed_result.status != "completed":
                raise ValueError(
                    f"Completed TaskResult status must be 'completed' for task '{clean_task}'"
                )

        # Phase 1: Replay check (first store-level operation, zero writes)
        record_key = (clean_task, command.completion_command_id.strip())
        existing_record = self.completion_records.get(record_key)
        if existing_record is not None:
            matches = (
                existing_record.task_id == clean_task
                and existing_record.assignment_id == command.assignment_id
                and existing_record.runtime_principal_id == clean_principal
                and existing_record.bound_agent_id == clean_agent
                and existing_record.execution_kind == command.execution_kind
                and existing_record.semantic_fingerprint == command.semantic_fingerprint()
                and existing_record.continuation_version == command.continuation_version
            )
            if matches:
                stored_outcome = self.terminal_outcomes.get(clean_task)
                if stored_outcome is None or stored_outcome.outcome_id != existing_record.outcome_id:
                    raise CorruptAuthoritativeRecordError(
                        f"Canonical terminal outcome missing for replayed task '{clean_task}'"
                    )
                if stored_outcome.task_id != clean_task:
                    raise CorruptAuthoritativeRecordError(
                        f"Canonical terminal outcome task mismatch for replayed task '{clean_task}'"
                    )
                stored_result = self.canonical_results.get(clean_task)
                if stored_outcome.terminal_status == "completed":
                    if stored_result is None:
                        raise CorruptAuthoritativeRecordError(
                            f"Canonical result missing for completed replayed task '{clean_task}'; "
                            "refusing to fabricate output"
                        )
                    if stored_result.task_id != clean_task or stored_result.status != "completed":
                        raise CorruptAuthoritativeRecordError(
                            f"Canonical result inconsistent for replayed task '{clean_task}'"
                        )
                elif stored_result is not None:
                    raise CorruptAuthoritativeRecordError(
                        f"Unexpected canonical result for failed replayed task '{clean_task}'"
                    )
                return (stored_outcome, stored_result, True)
            raise CompletionConflictError(
                f"Completion command '{command.completion_command_id}' on task '{clean_task}' "
                "conflicts with the already-accepted semantic payload"
            )

        # Phase 1 (cont.): strictly read-only validations
        cur_status = self.statuses.get(clean_task)
        if cur_status is None:
            raise TaskNotClaimableError(f"Task '{clean_task}' not found")
        if cur_status != "working":
            if cur_status in ("completed", "failed"):
                raise CompletionConflictError(
                    f"Task '{clean_task}' already reached canonical terminal status '{cur_status}'"
                )
            raise TaskNotClaimableError(
                f"Task '{clean_task}' is not in an eligible working turn (status '{cur_status}')"
            )

        claim = self.claims.get(clean_task)
        if claim is None:
            raise ClaimNotFoundError(
                f"Active acknowledged claim for assignment '{command.assignment_id}' "
                f"on task '{clean_task}' not found: CLAIM_NOT_FOUND"
            )
        if claim.assignment_id != command.assignment_id:
            raise ClaimNotFoundError("Active claim assignment mismatch: ASSIGNMENT_MISMATCH")
        if claim.task_id != clean_task:
            raise ClaimNotFoundError("Active claim task mismatch: TASK_MISMATCH")
        if claim.runtime_principal_id != clean_principal:
            raise ClaimNotFoundError("Active claim principal mismatch: PRINCIPAL_MISMATCH")
        if claim.bound_agent_id != clean_agent:
            raise ClaimNotFoundError("Active claim bound agent mismatch: BOUND_AGENT_MISMATCH")
        if claim.execution_kind != command.execution_kind:
            raise TaskNotClaimableError("Claimed execution kind mismatch: EXECUTION_KIND_MISMATCH")
        if claim.continuation_version != command.continuation_version:
            raise TaskNotClaimableError("Claimed continuation version mismatch: CONTINUATION_VERSION_MISMATCH")
        if claim.state != "acknowledged":
            raise ClaimNotFoundError("Assignment must be acknowledged: CLAIM_NOT_ACKNOWLEDGED")
        if claim.proof_digest != clean_digest:
            raise InvalidClaimProofError("Assignment proof verification failed")

        cur_fence = self.fences.get(clean_task)
        if cur_fence is None or cur_fence != fencing_epoch or claim.fencing_epoch != fencing_epoch:
            raise TaskAlreadyClaimedError(
                f"Fencing epoch mismatch for assignment '{command.assignment_id}' on task '{clean_task}'"
            )

        lineage = self.lineages.get(clean_task)
        if (
            lineage is None
            or lineage.task_id != clean_task
            or lineage.root_task_id != clean_root
        ):
            raise CorruptAuthoritativeRecordError(
                f"Authoritative lineage missing or inconsistent for task '{clean_task}'"
            )
        if is_child:
            if lineage.parent_task_id != clean_parent:
                raise TaskNotClaimableError("Child lineage parent mismatch: CHILD_LINEAGE_MISMATCH")
        elif lineage.parent_task_id is not None:
            raise TaskNotClaimableError("Non-child completion on child lineage: LINEAGE_KIND_MISMATCH")

        if clean_task in self.terminal_outcomes:
            raise CompletionConflictError(
                f"Conflicting canonical outcome already exists for task '{clean_task}': OUTCOME_CONFLICT"
            )
        if clean_task in self.canonical_results:
            raise CompletionConflictError(
                f"Conflicting canonical result already exists for task '{clean_task}': RESULT_CONFLICT"
            )
        if (clean_task, command.assignment_id) in self.assignment_completions:
            raise CompletionConflictError(
                f"Conflicting assignment completion already exists for task '{clean_task}': AUDIT_CONFLICT"
            )

        parent_children: set[str] = set()
        if is_child:
            parent_status = self.statuses.get(clean_parent)
            if parent_status is None:
                raise TaskNotClaimableError("Parent task status missing: PARENT_STATUS_NOT_FOUND")
            parent_lineage = self.lineages.get(clean_parent)
            if parent_lineage is None:
                raise TaskNotClaimableError("Parent lineage missing: PARENT_LINEAGE_NOT_FOUND")
            if (
                parent_lineage.task_id != clean_parent
                or parent_lineage.root_task_id != clean_root
                or parent_lineage.parent_task_id is not None
            ):
                raise CorruptAuthoritativeRecordError(
                    f"Authoritative parent lineage inconsistent for '{clean_parent}'"
                )
            if parent_status != "waiting_for_children":
                raise TaskNotClaimableError(
                    f"Parent task '{clean_parent}' is not waiting for children: PARENT_NOT_WAITING"
                )
            parent_children = self.children.get(clean_parent, set())
            if clean_task not in parent_children:
                raise TaskNotClaimableError("Child not member of parent: CHILD_NOT_MEMBER")
            continuation = self.continuations.get((clean_parent, 1))
            if continuation is None:
                raise TaskNotClaimableError("Parent continuation missing: MISSING_CONTINUATION")
            if continuation.stage != "incorporate_analysis_child":
                raise TaskNotClaimableError("Continuation stage mismatch: CONTINUATION_STAGE_MISMATCH")
            if continuation.child_task_id != clean_task:
                raise TaskNotClaimableError("Continuation child mismatch: CONTINUATION_CHILD_MISMATCH")
            if continuation.continuation_version != 1:
                raise TaskNotClaimableError("Continuation version mismatch: CONTINUATION_VERSION_MISMATCH")
            if self.latest_continuation.get(clean_parent) != 1:
                raise TaskNotClaimableError("Continuation latest mismatch: CONTINUATION_LATEST_MISMATCH")
            if clean_task in self.child_markers:
                raise CompletionConflictError(
                    f"Child completion already projected for '{clean_task}': MARKER_CONFLICT"
                )

        outcome, result = build_terminal_outcome(
            command=command,
            task_id=clean_task,
            runtime_principal_id=clean_principal,
            bound_agent_id=clean_agent,
            completed_at=now_iso,
        )
        audit = build_completion_audit(outcome=outcome, fencing_epoch=fencing_epoch)
        record = CompletionRecord(
            completion_command_id=command.completion_command_id,
            task_id=clean_task,
            assignment_id=command.assignment_id,
            runtime_principal_id=clean_principal,
            bound_agent_id=clean_agent,
            execution_kind=command.execution_kind,
            semantic_fingerprint=outcome.semantic_fingerprint,
            proof_digest=clean_digest,
            outcome_id=outcome.outcome_id,
            continuation_version=command.continuation_version,
        )

        # Phase 2: mechanically non-failing commit
        self.completion_records[record_key] = record
        self.terminal_outcomes[clean_task] = outcome
        if result is not None:
            self.canonical_results[clean_task] = result
        self.assignment_completions[(clean_task, command.assignment_id)] = audit
        self.statuses[clean_task] = outcome.terminal_status
        del self.claims[clean_task]
        if is_child:
            self.child_markers[clean_task] = {
                "child_task_id": clean_task,
                "completion_command_id": command.completion_command_id,
                "completed_at": now_iso,
                "outcome_id": outcome.outcome_id,
            }
            self.statuses[clean_parent] = "resumable"
            self.hints.append(
                WakeUpHint(
                    hint_id=f"hint-{uuid4()}",
                    capability="research",
                    created_at=now_iso,
                    execution_kind="resumed_continuation",
                    task_id=clean_parent,
                    expected_continuation_version=1,
                )
            )

        return (outcome, result, False)

    async def get_completion_record(
        self,
        task_id: str,
        completion_command_id: str,
    ) -> CompletionRecord | None:
        return self.completion_records.get((str(task_id).strip(), str(completion_command_id).strip()))

    async def get_terminal_outcome(self, task_id: str) -> TerminalOutcome | None:
        return self.terminal_outcomes.get(str(task_id).strip())

    async def get_canonical_result(self, task_id: str) -> TaskResult | None:
        clean_task = str(task_id).strip()
        result = self.canonical_results.get(clean_task)
        if result is not None and (
            result.task_id != clean_task or result.status != "completed"
        ):
            raise CorruptAuthoritativeRecordError(
                f"Canonical result is not bound to a completed turn of task '{clean_task}'"
            )
        return result

    async def get_assignment_completion(
        self,
        task_id: str,
        assignment_id: str,
    ) -> AssignmentCompletionRecord | None:
        return self.assignment_completions.get((str(task_id).strip(), str(assignment_id).strip()))

    async def get_delegation_proposal(
        self,
        parent_task_id: str,
        proposal_id: str,
    ) -> DelegationProposalRecord | None:
        clean_parent = str(parent_task_id).strip()
        clean_proposal_id = str(proposal_id).strip()
        return self.proposals.get((clean_parent, clean_proposal_id))

    async def materialize_claim_timeout(
        self,
        child_task_id: str,
        now: str,
    ) -> tuple[TerminalOutcome, bool]:
        clean_child = child_task_id.strip()
        clean_now = normalize_canonical_utc_timestamp(now)
        cur_status = self.statuses.get(clean_child)
        if not cur_status:
            raise ParentTaskNotFoundError(f"Child task '{clean_child}' not found")

        if cur_status == "claim_timed_out":
            outcome = self.terminal_outcomes.get(clean_child)
            if not outcome or outcome.terminal_status != "claim_timed_out" or outcome.task_id != clean_child:
                raise ClaimTimeoutConflictError(f"Child task '{clean_child}' missing or conflicting terminal outcome")
            if clean_child in self.canonical_results:
                raise ClaimTimeoutConflictError(f"Child task '{clean_child}' has unexpected canonical result")
            if clean_child in self.claims:
                raise ClaimTimeoutConflictError(f"Child task '{clean_child}' has active claim")

            lineage = self.lineages.get(clean_child)
            if not lineage or lineage.task_id != clean_child or not lineage.parent_task_id or not lineage.root_task_id:
                raise ClaimTimeoutConflictError(f"Child task '{clean_child}' missing or corrupt lineage")
            parent_task_id = lineage.parent_task_id

            children = self.children.get(parent_task_id, set())
            if clean_child not in children:
                raise ClaimTimeoutConflictError(f"Child task '{clean_child}' not in parent's children set")

            latest_ver = self.latest_continuation.get(parent_task_id, 0)
            if latest_ver != 1:
                raise ClaimTimeoutConflictError(f"Parent task '{parent_task_id}' latest continuation version is {latest_ver}, expected 1")

            cont = self.continuations.get((parent_task_id, 1))
            if (
                not cont
                or cont.parent_task_id != parent_task_id
                or cont.continuation_version != 1
                or cont.stage != "incorporate_analysis_child"
                or cont.child_task_id != clean_child
            ):
                raise ClaimTimeoutConflictError(f"Parent task '{parent_task_id}' continuation v1 does not match stage/child/version")

            marker = self.child_markers.get(clean_child)
            if (
                not marker
                or marker.get("child_task_id") != clean_child
                or marker.get("parent_task_id") != parent_task_id
                or marker.get("outcome_id") != outcome.outcome_id
                or marker.get("continuation_version") != 1
                or marker.get("status") != "claim_timed_out"
            ):
                raise ClaimTimeoutConflictError(f"Child task '{clean_child}' missing or conflicting completion marker")

            parent_st = self.statuses.get(parent_task_id)
            if parent_st != "resumable":
                raise ClaimTimeoutConflictError(f"Parent task '{parent_task_id}' is '{parent_st}', expected 'resumable'")

            return (outcome, True)

        if cur_status != "pending":
            raise ClaimTimeoutConflictError(f"Child task '{clean_child}' is in '{cur_status}' status, expected 'pending'")

        if clean_child in self.terminal_outcomes:
            raise ClaimTimeoutConflictError(f"Child task '{clean_child}' already has a terminal outcome")
        if clean_child in self.canonical_results:
            raise ClaimTimeoutConflictError(f"Child task '{clean_child}' already has a canonical result")
        if clean_child in self.child_markers:
            raise ClaimTimeoutConflictError(f"Child task '{clean_child}' already has a completion marker")

        claim = self.claims.get(clean_child)
        if claim:
            if claim.state == "acknowledged":
                raise ClaimTimeoutConflictError(f"Child task '{clean_child}' has active acknowledged claim")
            elif claim.state == "claimed_unacknowledged" and claim.delivery_expires_at > clean_now:
                raise ClaimTimeoutConflictError(f"Child task '{clean_child}' has active unacknowledged delivery lease")

        lineage = self.lineages.get(clean_child)
        if not lineage or not lineage.parent_task_id:
            raise ClaimTimeoutConflictError(f"Child task '{clean_child}' has no parent lineage")
        parent_task_id = lineage.parent_task_id

        deadline = self.claim_deadlines.get(clean_child)
        if not deadline:
            raise ClaimTimeoutConflictError(f"Child task '{clean_child}' has no claim deadline")
        if clean_now <= deadline:
            raise ClaimDeadlineNotExpiredError(f"Child task '{clean_child}' claim deadline {deadline} has not expired at {clean_now}")

        parent_status = self.statuses.get(parent_task_id)
        if not parent_status:
            raise ParentTaskNotFoundError(f"Parent task '{parent_task_id}' not found")
        if parent_status != "waiting_for_children":
            raise ClaimTimeoutConflictError(f"Parent task '{parent_task_id}' is '{parent_status}', expected 'waiting_for_children'")

        children = self.children.get(parent_task_id, set())
        if clean_child not in children:
            raise ClaimTimeoutConflictError(f"Child task '{clean_child}' not in parent's children set")

        latest_ver = self.latest_continuation.get(parent_task_id, 0)
        if latest_ver != 1:
            raise ClaimTimeoutConflictError(f"Parent task '{parent_task_id}' latest continuation version is {latest_ver}, expected 1")
        cont = self.continuations.get((parent_task_id, 1))
        if (
            not cont
            or cont.parent_task_id != parent_task_id
            or cont.continuation_version != latest_ver
            or cont.stage != "incorporate_analysis_child"
            or cont.child_task_id != clean_child
        ):
            raise ClaimTimeoutConflictError(f"Parent task '{parent_task_id}' continuation v1 does not match stage/child/version")

        outcome = build_claim_timed_out_terminal_outcome(
            task_id=clean_child,
            claim_deadline=deadline,
            completed_at=clean_now,
        )
        self.terminal_outcomes[clean_child] = outcome
        self.statuses[clean_child] = "claim_timed_out"
        self.child_markers[clean_child] = {
            "child_task_id": clean_child,
            "continuation_version": 1,
            "outcome_id": outcome.outcome_id,
            "parent_task_id": parent_task_id,
            "status": "claim_timed_out",
        }
        self.statuses[parent_task_id] = "resumable"
        hint = WakeUpHint(
            hint_id=f"hint-{uuid4()}",
            capability="research",
            created_at=clean_now,
            execution_kind="resumed_continuation",
            task_id=parent_task_id,
            expected_continuation_version=1,
        )
        self.hints.append(hint)
        return (outcome, False)

    async def transition_expired_claim(
        self,
        task_id: str,
        now: str,
    ) -> tuple[ClaimExpiryRecord, bool]:
        clean_task = task_id.strip()
        clean_now = normalize_canonical_utc_timestamp(now)
        cur_status = self.statuses.get(clean_task)
        if not cur_status:
            raise ParentTaskNotFoundError(f"Task '{clean_task}' not found")

        if cur_status == "recovery_required":
            existing = self.claim_expiries.get(clean_task)
            if existing is not None:
                return (existing, True)
            raise ClaimTimeoutConflictError(f"Task '{clean_task}' is recovery_required but missing expiry record")

        if cur_status != "working":
            raise ClaimTimeoutConflictError(f"Task '{clean_task}' is in '{cur_status}', expected 'working'")

        if clean_task in self.terminal_outcomes or clean_task in self.canonical_results:
            raise ClaimTimeoutConflictError(f"Task '{clean_task}' is already completed")

        claim = self.claims.get(clean_task)
        if not claim:
            raise ClaimNotFoundError(f"No active claim on task '{clean_task}'")
        if claim.state != "acknowledged":
            raise ClaimTimeoutConflictError(f"Claim on task '{clean_task}' is in state '{claim.state}', expected 'acknowledged'")

        if not claim.execution_expires_at:
            raise ClaimTimeoutConflictError(f"Claim on task '{clean_task}' has no execution_expires_at")
        if clean_now <= claim.execution_expires_at:
            raise ClaimTimeoutConflictError(f"Claim execution lease {claim.execution_expires_at} has not expired at {clean_now}")

        cur_fence = self.fences.get(clean_task, 0)
        if cur_fence != claim.fencing_epoch:
            raise FencingEpochRaceError(f"Fence mismatch: current {cur_fence} != claim {claim.fencing_epoch}")

        if claim.execution_kind == "resumed_continuation":
            if claim.prior_claimable_status != "resumable":
                raise ClaimTimeoutConflictError(f"Corrupt prior_claimable_status '{claim.prior_claimable_status}' for resumed_continuation")
        else:
            if claim.prior_claimable_status != "pending":
                raise ClaimTimeoutConflictError(f"Corrupt prior_claimable_status '{claim.prior_claimable_status}'")

        expiry_record = ClaimExpiryRecord(
            task_id=clean_task,
            claim_id=claim.claim_id,
            assignment_id=claim.assignment_id,
            principal_id=claim.runtime_principal_id,
            bound_agent_id=claim.bound_agent_id,
            execution_kind=claim.execution_kind,
            expired_fencing_epoch=claim.fencing_epoch,
            acknowledged_at=claim.acknowledged_at or clean_now,
            execution_expires_at=claim.execution_expires_at,
            expired_at=clean_now,
            prior_claimable_status=claim.prior_claimable_status,
            continuation_version=claim.continuation_version,
        )
        del self.claims[clean_task]
        self.claim_expiries[clean_task] = expiry_record
        self.statuses[clean_task] = "recovery_required"
        return (expiry_record, False)

    async def resolve_expired_claim(
        self,
        command: RecoveryDecisionCommand,
        operator_principal_id: str,
        now: str,
    ) -> tuple[RecoveryDecisionRecord, str]:
        clean_task = command.task_id.strip()
        clean_now = normalize_canonical_utc_timestamp(now)
        key = (clean_task, command.recovery_decision_id.strip())

        existing = self.recovery_decisions.get(key)
        if existing is not None:
            if existing.semantic_fingerprint == command.semantic_fingerprint():
                return (existing, existing.restored_status)
            raise RecoveryConflictError(f"Conflicting recovery decision '{command.recovery_decision_id}'")

        cur_status = self.statuses.get(clean_task)
        if not cur_status:
            raise ParentTaskNotFoundError(f"Task '{clean_task}' not found")
        if cur_status != "recovery_required":
            raise RecoveryPreconditionFailedError(f"Task '{clean_task}' is in '{cur_status}', expected 'recovery_required'")

        if clean_task in self.terminal_outcomes:
            raise RecoveryConflictError(f"Task '{clean_task}' has already completed")
        if clean_task in self.claims:
            raise RecoveryConflictError(f"Task '{clean_task}' has an active claim")

        expiry = self.claim_expiries.get(clean_task)
        if not expiry:
            raise RecoveryPreconditionFailedError(f"Task '{clean_task}' has no ClaimExpiryRecord")

        if (
            expiry.claim_id != command.expired_claim_id
            or expiry.assignment_id != command.expired_assignment_id
            or expiry.expired_fencing_epoch != command.expired_fencing_epoch
        ):
            raise RecoveryConflictError("Recovery decision does not match ClaimExpiryRecord")

        if command.disposition != "restore_claimable":
            raise RecoveryPreconditionFailedError(f"Unsupported recovery disposition '{command.disposition}'")

        cur_fence = self.fences.get(clean_task, 0)
        if cur_fence != command.expired_fencing_epoch:
            raise FencingEpochRaceError(f"Fence mismatch: current {cur_fence} != expected {command.expired_fencing_epoch}")

        restored_status = expiry.prior_claimable_status
        if restored_status not in ("pending", "resumable"):
            raise RecoveryPreconditionFailedError(f"Invalid prior_claimable_status '{restored_status}' in expiry record")

        record = RecoveryDecisionRecord(
            task_id=clean_task,
            recovery_decision_id=command.recovery_decision_id.strip(),
            operator_principal_id=operator_principal_id.strip(),
            expired_claim_id=command.expired_claim_id,
            expired_assignment_id=command.expired_assignment_id,
            expired_fencing_epoch=command.expired_fencing_epoch,
            disposition=command.disposition,
            prior_status="recovery_required",
            restored_status=restored_status,
            resolved_at=clean_now,
            semantic_fingerprint=command.semantic_fingerprint(),
        )
        self.recovery_decisions[key] = record
        self.statuses[clean_task] = restored_status
        return (record, restored_status)

    async def get_claim_expiry(self, task_id: str) -> ClaimExpiryRecord | None:
        return self.claim_expiries.get(task_id.strip())

    async def get_recovery_decision(
        self,
        task_id: str,
        recovery_decision_id: str,
    ) -> RecoveryDecisionRecord | None:
        return self.recovery_decisions.get((task_id.strip(), recovery_decision_id.strip()))


class MockAtomicDispatcher(TaskDispatcher):
    """Simulates an atomic dispatcher that enqueues and transitions status to pending."""

    def __init__(
        self,
        task_store: AuthoritativeTaskStore | None = None,
        should_fail: bool = False,
    ) -> None:
        self.task_store = task_store
        self.dispatches: list[tuple[TaskRequested, str]] = []
        self.should_fail = should_fail

    async def dispatch(
        self, task: TaskRequested, target: str, capability: str | None = None
    ) -> None:
        if self.should_fail:
            raise RuntimeError(f"Simulated dispatch transport failure targeting '{target}'")
        self.dispatches.append((task, target))
        if self.task_store is not None:
            await self.task_store.update_status(task.task_id, "pending")
            if hasattr(self.task_store, "hints"):
                self.task_store.hints.append(
                    WakeUpHint(
                        hint_id=f"hint-{uuid4().hex[:8]}",
                        capability=capability or "research",
                        created_at=datetime.now(UTC).isoformat(),
                        execution_kind="initial",
                        task_id=task.task_id,
                    )
                )


def make_candidate(
    agent_id: str,
    capability: str = "general",
    dispatch_target: str = "agent.tasks.general",
) -> AgentCandidate:
    now_str = datetime.now(UTC).isoformat()
    return AgentCandidate(
        agent_id=agent_id,
        capabilities=[capability],
        availability=AgentAvailability(
            state=AgentAvailabilityState.READY,
            execution_state=AgentExecutionState.IDLE,
            heartbeat_age_seconds=1.0,
            last_seen_at=now_str,
            as_of=now_str,
            reason="healthy",
        ),
        agent_type="worker",
        execution_state=AgentExecutionState.IDLE,
        dispatch_target=dispatch_target,
    )


class FakeCandidateFinder:
    def __init__(self, candidates: list[AgentCandidate]) -> None:
        self.candidates = candidates

    async def execute(self, capability: Any, as_of: Any = None) -> list[AgentCandidate]:
        return list(self.candidates)


# ==============================================================================
# 1. Domain Model: TaskLineage Invariants
# ==============================================================================


def test_task_lineage_root_invariants() -> None:
    lineage = TaskLineage(task_id="root-1", root_task_id="root-1", parent_task_id=None)
    assert lineage.task_id == "root-1"
    assert lineage.root_task_id == "root-1"
    assert lineage.parent_task_id is None
    assert lineage.is_root is True

    # Root task with mismatched root_task_id is rejected
    with pytest.raises(ValueError, match="root tasks must have root_task_id equal to task_id"):
        TaskLineage(task_id="root-1", root_task_id="different", parent_task_id=None)


def test_task_lineage_child_invariants() -> None:
    lineage = TaskLineage(task_id="child-1", root_task_id="root-1", parent_task_id="root-1")
    assert lineage.task_id == "child-1"
    assert lineage.root_task_id == "root-1"
    assert lineage.parent_task_id == "root-1"
    assert lineage.is_root is False

    # Child claiming root_task_id == task_id rejected
    with pytest.raises(ValueError, match="child tasks must not have root_task_id equal to task_id"):
        TaskLineage(task_id="c1", root_task_id="c1", parent_task_id="p1")

    # Self-parenting rejected
    with pytest.raises(ValueError, match="parent_task_id must not equal task_id"):
        TaskLineage(task_id="c1", root_task_id="r1", parent_task_id="c1")


def test_task_lineage_deeper_ancestry_representation() -> None:
    """Wire/lineage model represents deeper ancestry structurally."""
    lineage = TaskLineage(task_id="A3", root_task_id="A", parent_task_id="A2")
    assert lineage.task_id == "A3"
    assert lineage.root_task_id == "A"
    assert lineage.parent_task_id == "A2"
    assert lineage.is_root is False

    # Round trip dictionary
    data = lineage.to_dict()
    assert TaskLineage.from_dict(data) == lineage


# ==============================================================================
# 2. Production Composition & Wiring
# ==============================================================================


def test_submit_task_requires_authoritative_task_store() -> None:
    """SubmitTask constructor must require an AuthoritativeTaskStore without silent skip."""
    # Omitting task_store raises TypeError
    with pytest.raises(TypeError):
        SubmitTask()  # type: ignore[call-arg]

    # Passing None or non-store raises TypeError
    with pytest.raises(TypeError, match="SubmitTask requires an AuthoritativeTaskStore"):
        SubmitTask(task_store=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="task_store must implement AuthoritativeTaskStore"):
        SubmitTask(task_store="invalid-store")  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_production_composition_wires_redis_gateway_as_authoritative_store() -> None:
    """Demonstrates production-style create_app supplies RedisGateway as AuthoritativeTaskStore.

    A normal root submission invokes authoritative root registration (pending_dispatch).
    """
    settings = Settings(redis_url="redis://fake:6379/0", app_env="test")
    gateway = RedisGateway(settings.redis_url)

    # Mock client and pipeline on gateway
    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.rpush = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=None)

    mock_client = AsyncMock()
    mock_client.pipeline = MagicMock(return_value=mock_pipe)
    mock_client.set = AsyncMock()
    mock_client.eval = AsyncMock(return_value="OK")
    mock_client.xadd = AsyncMock()
    mock_client.aclose = AsyncMock()
    gateway._client = mock_client
    gateway.agents = AsyncMock(
        return_value=[
            {
                "agent_id": "worker-1",
                "capabilities": ["general"],
                "state": "idle",
                "dispatch_target": "agent.tasks.general",
                "updated_at": datetime.now(UTC).isoformat(),
            }
        ]
    )

    # In production composition with RedisGateway
    app = create_app(settings=settings, gateway=gateway)

    # Dispatch a root task through API layer
    req = Request({"type": "http", "app": app})
    res = await create_task(ApiTaskInput(prompt="Inspect cluster state"), request=req)

    assert res["status"] == "accepted"
    assert res["task_id"]

    # Verify atomic Lua script was evaluated for root registration with initial status pending_dispatch
    mock_client.eval.assert_awaited()
    # Verify dispatch pipeline pushed task and set status to pending
    mock_pipe.rpush.assert_called_once()


def test_create_app_fails_fast_when_task_store_provided_without_dispatch_path() -> None:
    """create_app(task_store=store) must fail fast if no dispatch path exists.

    It must not create a zombie app that registers tasks but cannot dispatch them.
    """
    store = InMemoryAuthoritativeTaskStore()
    with pytest.raises(TypeError, match="create_app requires a dispatch path"):
        create_app(task_store=store)


def test_create_app_fails_fast_when_no_authoritative_task_store_can_be_resolved() -> None:
    """create_app must fail fast if no AuthoritativeTaskStore can be resolved."""
    with pytest.raises(TypeError, match="create_app requires an AuthoritativeTaskStore"):
        create_app(task_store=None, gateway=None, task_queue=MagicMock())


# ==============================================================================
# 3. Root Lifecycle & Atomic Dispatch
# ==============================================================================


@pytest.mark.anyio
async def test_submit_task_records_root_lineage_and_lifecycle_transitions() -> None:
    """Verifies root registration establishes pending_dispatch + root lineage, and successful dispatch transitions to pending."""
    store = InMemoryAuthoritativeTaskStore()
    dispatcher = MockAtomicDispatcher(task_store=store, should_fail=False)
    candidate = make_candidate("worker-1", capability="general")
    finder = FakeCandidateFinder([candidate])

    submit_task = SubmitTask(
        task_store=store,
        dispatcher=dispatcher,
        candidate_finder=finder,
    )

    task = await submit_task.execute("Analyze cluster metrics")

    # Authoritative root lineage persists
    lineage = await store.get_lineage(task.task_id)
    assert lineage is not None
    assert lineage.task_id == task.task_id
    assert lineage.root_task_id == task.task_id
    assert lineage.parent_task_id is None

    # Status transitioned to pending via atomic dispatch
    assert await store.get_status(task.task_id) == "pending"


@pytest.mark.anyio
async def test_submit_task_dispatch_failure_transitions_to_dispatch_failed() -> None:
    """When dispatch fails, use case records dispatch_failed and re-raises the original exception."""
    store = InMemoryAuthoritativeTaskStore()
    dispatcher = MockAtomicDispatcher(task_store=store, should_fail=True)
    candidate = make_candidate("worker-1", capability="general")
    finder = FakeCandidateFinder([candidate])

    submit_task = SubmitTask(
        task_store=store,
        dispatcher=dispatcher,
        candidate_finder=finder,
    )

    original_msg = "Simulated dispatch transport failure targeting 'agent.tasks.general'"
    with pytest.raises(RuntimeError, match=original_msg):
        await submit_task.execute("Analyze cluster metrics")

    # The task was recorded authoritatively before dispatch attempt
    assert len(store.lineages) == 1
    task_id = next(iter(store.lineages.keys()))

    # Lineage is preserved
    lineage = await store.get_lineage(task_id)
    assert lineage is not None
    assert lineage.task_id == task_id

    # Status transitioned to dispatch_failed, preserving observable state
    assert await store.get_status(task_id) == "dispatch_failed"


@pytest.mark.anyio
async def test_legacy_enqueue_success_with_status_failure_does_not_relabel_dispatch_failed() -> None:
    """When legacy enqueue succeeds but later update_status('pending') fails:

    - status-update failure propagates;
    - task is NOT relabelled as dispatch_failed (since it is already committed to queue).
    """
    class FailingPendingStore(InMemoryAuthoritativeTaskStore):
        async def update_status(self, task_id: str, status: str) -> None:
            if status == "pending":
                raise RuntimeError("Simulated status update failure for pending")
            await super().update_status(task_id, status)

    store = FailingPendingStore()
    queue = FakeTaskQueue()
    submit = SubmitTask(task_store=store, queue=queue)

    with pytest.raises(RuntimeError, match="Simulated status update failure for pending"):
        await submit.execute("Legacy task with failing pending status")

    # Enqueue was committed
    assert len(queue.enqueued) == 1
    enqueued_task, _ = queue.enqueued[0]

    # Task status remains pending_dispatch and was NOT relabelled dispatch_failed
    assert await store.get_status(enqueued_task.task_id) == "pending_dispatch"


# ==============================================================================
# 4. Child Registration (RegisterChildTask) - Non-Executable Primitive
# ==============================================================================


@pytest.mark.anyio
async def test_register_child_requires_authoritative_parent_with_zero_state() -> None:
    store = InMemoryAuthoritativeTaskStore()
    register_child = RegisterChildTask(task_store=store)

    with pytest.raises(ParentTaskNotFoundError, match="Authoritative parent task not found"):
        await register_child.execute(parent_task_id="non-existent-parent", prompt="subtask")

    # Zero partial state exists
    assert store.statuses == {}
    assert store.lineages == {}
    assert store.children == {}


@pytest.mark.anyio
async def test_register_child_enforces_depth_1_policy() -> None:
    store = InMemoryAuthoritativeTaskStore()
    register_child = RegisterChildTask(task_store=store)

    # 1. Create root parent
    root = TaskRequested.create("Root task")
    await store.register_root_task(root)

    # 2. Register child of root (depth 1 - valid)
    child1 = await register_child.execute(parent_task_id=root.task_id, prompt="Child 1")
    assert child1.parent_task_id == root.task_id
    assert child1.root_task_id == root.task_id

    # 3. Attempt to register child of child1 (depth 2 - rejected)
    with pytest.raises(DelegationDepthExceededError, match="is already a child task"):
        await register_child.execute(parent_task_id=child1.task_id, prompt="Grandchild")

    # Verify no grandchild was registered in store
    assert await store.get_children(child1.task_id) == []


@pytest.mark.anyio
async def test_register_child_derives_root_and_mints_child_id() -> None:
    store = InMemoryAuthoritativeTaskStore()
    register_child = RegisterChildTask(task_store=store)

    root = TaskRequested.create("Root parent")
    await store.register_root_task(root)

    child = await register_child.execute(parent_task_id=root.task_id, prompt="Derived root child")
    assert child.root_task_id == root.task_id
    assert child.parent_task_id == root.task_id
    assert child.task_id != root.task_id
    assert child.version == 2

    # Status established as pending_dispatch
    assert await store.get_status(child.task_id) == "pending_dispatch"

    # Lineage in store reflects authoritative derivation
    lineage = await store.get_lineage(child.task_id)
    assert lineage is not None
    assert lineage.root_task_id == root.task_id
    assert lineage.parent_task_id == root.task_id


def test_register_child_primitive_is_non_executable() -> None:
    """Verifies RegisterChildTask accepts no dispatcher, no target parameter, and performs no queue dispatch."""
    import inspect

    # Constructor requires only task_store
    sig_init = inspect.signature(RegisterChildTask.__init__)
    assert list(sig_init.parameters.keys()) == ["self", "task_store"]

    # Execute accepts only parent_task_id and prompt (no target or queue parameters)
    sig_exec = inspect.signature(RegisterChildTask.execute)
    assert list(sig_exec.parameters.keys()) == ["self", "parent_task_id", "prompt"]


@pytest.mark.anyio
async def test_register_child_simulated_pipeline_failure_leaves_no_orphans() -> None:
    store = InMemoryAuthoritativeTaskStore()
    register_child = RegisterChildTask(task_store=store)

    root = TaskRequested.create("Root parent")
    await store.register_root_task(root)

    # Trigger failure during child registration
    store.fail_next_registration = True

    with pytest.raises(RuntimeError, match="Simulated pipeline failure"):
        await register_child.execute(parent_task_id=root.task_id, prompt="Failing child")

    # Parent has zero children registered (no phantom entry)
    assert await store.get_children(root.task_id) == []


@pytest.mark.anyio
async def test_parent_membership_duplicate_safe_and_deterministic() -> None:
    """Verifies that parent-to-child Redis set membership is duplicate-safe and retrieval is deterministically sorted.

    Note: Set membership is duplicate-safe via SADD. Full task re-registration idempotency
    across status and forward lineage is deferred to future work.
    """
    store = InMemoryAuthoritativeTaskStore()
    register_child = RegisterChildTask(task_store=store)

    root = TaskRequested.create("Root parent")
    await store.register_root_task(root)

    c1 = await register_child.execute(parent_task_id=root.task_id, prompt="Child 1")
    c2 = await register_child.execute(parent_task_id=root.task_id, prompt="Child 2")
    c3 = await register_child.execute(parent_task_id=root.task_id, prompt="Child 3")

    expected_sorted = sorted([c1.task_id, c2.task_id, c3.task_id])
    assert await store.get_children(root.task_id) == expected_sorted

    # Re-adding the child to the parent set does not duplicate the membership entry
    store.children[root.task_id].add(c1.task_id)
    assert await store.get_children(root.task_id) == expected_sorted


@pytest.mark.anyio
async def test_register_child_validates_and_normalizes_prompt() -> None:
    """RegisterChildTask must validate and normalize prompts through the Task domain model."""
    store = InMemoryAuthoritativeTaskStore()
    register_child = RegisterChildTask(task_store=store)

    root = TaskRequested.create("Root prompt")
    await store.register_root_task(root)

    # 1. Valid trimmed prompt
    child = await register_child.execute(root.task_id, "   child prompt with whitespace   \n")
    assert child.prompt == "child prompt with whitespace"

    # 2. Empty/blank prompt rejected
    with pytest.raises(ValueError, match="Prompt must not be empty"):
        await register_child.execute(root.task_id, "   ")

    # 3. Over 2,000 characters rejected
    with pytest.raises(ValueError, match="Prompt must not exceed 2000 characters"):
        await register_child.execute(root.task_id, "x" * 2001)

    # 4. Non-string prompt rejected
    with pytest.raises(TypeError, match="Prompt must be a string"):
        await register_child.execute(root.task_id, 12345)  # type: ignore[arg-type]


# ==============================================================================
# 5. RedisGateway Implementation & Transaction Semantics
# ==============================================================================


@pytest.mark.anyio
async def test_redis_gateway_dispatch_uses_atomic_pipeline_with_transaction_true() -> None:
    """Verifies that RedisGateway.dispatch executes RPUSH and status:pending inside a transaction=True pipeline."""
    gateway = RedisGateway("redis://localhost:6379/0")

    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.rpush = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=None)

    gateway._client.pipeline = MagicMock(return_value=mock_pipe)

    task = TaskRequested.create("Execute runtime task")
    target = "agent.tasks.custom-executor-1"

    await gateway.dispatch(task, target=target)

    # Verified that pipeline was requested with transaction=True
    gateway._client.pipeline.assert_called_once_with(transaction=True)

    # Sets status: pending and pushes to target via pipeline buffer
    mock_pipe.rpush.assert_called_once_with(target, task.to_json())
    mock_pipe.set.assert_called_once_with(
        f"task.status:{task.task_id}",
        json.dumps({"task_id": task.task_id, "status": "pending"}),
    )
    mock_pipe.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_redis_gateway_register_child_uses_atomic_lua_script() -> None:
    """Verifies that RedisGateway.register_child_task executes REGISTER_CHILD_TASK_LUA atomically across keys."""
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.eval = AsyncMock(return_value="OK")

    child = TaskRequested(
        task_id="child-123",
        prompt="Child prompt",
        created_at="2026-11-15T00:00:00+00:00",
        root_task_id="root-1",
        parent_task_id="root-1",
        version=2,
    )

    await gateway.register_child_task(child, initial_status="pending_dispatch", required_capability="analysis")

    gateway._client.eval.assert_awaited_once()
    assert gateway._client.eval.await_args is not None
    call_args = gateway._client.eval.await_args[0]
    assert call_args[1] == 4
    assert call_args[2] == "task.status:child-123"
    assert call_args[3] == "task.lineage:child-123"
    assert call_args[4] == "task.required_capability:child-123"
    assert call_args[5] == "task.children:root-1"


@pytest.mark.anyio
async def test_redis_gateway_get_children_determinism() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.smembers = AsyncMock(return_value={b"c-3", b"c-1", b"c-2"})

    children = await gateway.get_children("parent-1")
    assert children == ["c-1", "c-2", "c-3"]  # strictly sorted


@pytest.mark.anyio
async def test_redis_gateway_register_root_executes_atomic_lua_script() -> None:
    """Verifies that RedisGateway.register_root_task executes atomic Lua script across all 4 keys."""
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.eval = AsyncMock(return_value="OK")

    task = TaskRequested.create("Root task prompt")

    # 1. Fresh registration succeeds
    await gateway.register_root_task(task, initial_status="pending_dispatch")
    gateway._client.eval.assert_awaited_once()

    # 2. Idempotent replay succeeds
    gateway._client.eval = AsyncMock(return_value="IDEMPOTENT")
    await gateway.register_root_task(task, initial_status="pending_dispatch")

    # 3. Conflicting registration raises ConflictingRootTaskRegistrationError
    gateway._client.eval = AsyncMock(return_value="CONFLICT")
    with pytest.raises(ConflictingRootTaskRegistrationError, match="already exists with conflicting state"):
        await gateway.register_root_task(task)

    # 4. Partial state conflict raises ConflictingRootTaskRegistrationError
    gateway._client.eval = AsyncMock(return_value="PARTIAL_CONFLICT")
    with pytest.raises(ConflictingRootTaskRegistrationError, match="inconsistent or partial authoritative state"):
        await gateway.register_root_task(task)


@pytest.mark.anyio
async def test_simulated_dispatch_pipeline_failure_does_not_expose_queued_task_or_pending_status() -> None:
    """Simulated pipeline execute failure during dispatch does not expose enqueued task or pending status."""
    gateway = RedisGateway("redis://localhost:6379/0")

    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.rpush = MagicMock()
    mock_pipe.execute = AsyncMock(side_effect=RuntimeError("Simulated pipeline commit failure"))
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=None)

    gateway._client.pipeline = MagicMock(return_value=mock_pipe)

    task = TaskRequested.create("Root task")

    with pytest.raises(RuntimeError, match="Simulated pipeline commit failure"):
        await gateway.dispatch(task, target="agent.tasks.general")

    # Gateway client pipeline was invoked with transaction=True
    gateway._client.pipeline.assert_called_once_with(transaction=True)


# ==============================================================================
# 6. Durable TaskInput, StageArtifact, Continuation & Authority Storage Tests
# ==============================================================================


@pytest.mark.anyio
async def test_in_memory_task_input_write_once_semantics() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now = datetime.now(UTC).isoformat()
    input_v1 = TaskInput(
        task_id="t-1",
        prompt="Initial prompt",
        created_at=now,
        input_version=1,
    )

    # 1. Initial write
    await store.save_task_input(input_v1)
    retrieved = await store.get_task_input("t-1", input_version=1)
    assert retrieved == input_v1

    # 2. Idempotent write succeeds
    await store.save_task_input(input_v1)

    # 3. Conflicting write for same version raises ConflictingTaskInputError
    conflicting = TaskInput(
        task_id="t-1",
        prompt="Conflicting prompt",
        created_at=now,
        input_version=1,
    )
    with pytest.raises(ConflictingTaskInputError, match="already exists with conflicting data"):
        await store.save_task_input(conflicting)

    # 4. New version succeeds
    input_v2 = TaskInput(
        task_id="t-1",
        prompt="Updated prompt v2",
        created_at=now,
        input_version=2,
    )
    await store.save_task_input(input_v2)
    assert await store.get_task_input("t-1", input_version=2) == input_v2
    assert await store.get_task_input("t-1", input_version=1) == input_v1


@pytest.mark.anyio
async def test_in_memory_stage_artifact_write_once_semantics() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now = datetime.now(UTC).isoformat()
    future = "2099-01-01T00:00:00Z"
    claim = ActiveClaim(
        task_id="t-parent",
        assignment_id="assign-1",
        claim_id="claim-1",
        root_task_id="t-parent",
        turn_id="turn-1",
        runtime_principal_id="principal-1",
        bound_agent_id="research-1",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="a" * 64,
        request_id="req-1",
        claimed_at=now,
        delivery_expires_at=future,
        state="acknowledged",
        acknowledged_at=now,
    )
    store.claims["t-parent"] = claim
    store.fences["t-parent"] = 1

    art_v1 = StageArtifact(
        artifact_id="art-1",
        producing_task_id="t-parent",
        artifact_type="research_material",
        content={"material": "val1"},
        created_at=now,
        artifact_version=1,
    )

    await store.save_stage_artifact(art_v1, "assign-1", "principal-1", 1)
    retrieved = await store.get_stage_artifact("t-parent", "art-1", artifact_version=1)
    assert retrieved == art_v1

    # Idempotent write
    await store.save_stage_artifact(art_v1, "assign-1", "principal-1", 1)

    # Conflicting write
    conflicting = StageArtifact(
        artifact_id="art-1",
        producing_task_id="t-parent",
        artifact_type="research_material",
        content={"material": "val-conflict"},
        created_at=now,
        artifact_version=1,
    )
    with pytest.raises(ConflictingStageArtifactError, match="already exists with conflicting data"):
        await store.save_stage_artifact(conflicting, "assign-1", "principal-1", 1)


@pytest.mark.anyio
async def test_in_memory_continuation_write_once_semantics() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now = datetime.now(UTC).isoformat()
    cont_v1 = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-1",
        input_version=1,
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "focus-1"},
        continuation_version=1,
        created_at=now,
        updated_at=now,
    )

    await store.save_continuation(cont_v1)
    retrieved = await store.get_continuation("p-1", continuation_version=1)
    assert retrieved == cont_v1

    # Idempotent write
    await store.save_continuation(cont_v1)

    # Conflicting write
    conflicting = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-different",
        input_version=1,
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "focus-conflict"},
        continuation_version=1,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(ConflictingContinuationError, match="already exists with conflicting data"):
        await store.save_continuation(conflicting)


@pytest.mark.anyio
async def test_in_memory_root_authority_write_once_semantics() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now = datetime.now(UTC).isoformat()
    auth = RootAuthorityRecord(
        root_task_id="r-1",
        policy_context="default",
        created_at=now,
    )

    await store.save_root_authority(auth)
    retrieved = await store.get_root_authority("r-1")
    assert retrieved == auth

    # Idempotent write
    await store.save_root_authority(auth)

    # Conflicting write
    conflicting = RootAuthorityRecord.create_default(
        root_task_id="r-1",
        created_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(ConflictingRootAuthorityError, match="already exists with conflicting data"):
        await store.save_root_authority(conflicting)


@pytest.mark.anyio
async def test_submit_task_root_ingestion_persists_durable_input_and_authority() -> None:
    store = InMemoryAuthoritativeTaskStore()
    queue = FakeTaskQueue()
    submit = SubmitTask(task_store=store, queue=queue)

    task_requested = await submit.execute("Analyze system resilience under node failure")
    task_id = task_requested.task_id

    # 1. Authoritative status and lineage exist
    assert await store.get_status(task_id) == "pending"
    lineage = await store.get_lineage(task_id)
    assert lineage is not None
    assert lineage.is_root
    assert lineage.root_task_id == task_id

    # 2. Durable TaskInput v1 exists independently of queue message
    input_record = await store.get_task_input(task_id, input_version=1)
    assert input_record is not None
    assert input_record.task_id == task_id
    assert input_record.prompt == "Analyze system resilience under node failure"
    assert input_record.input_version == 1

    # 3. Default RootAuthorityRecord exists
    authority = await store.get_root_authority(task_id)
    assert authority is not None
    assert authority.root_task_id == task_id
    assert authority.max_delegation_depth == 1
    assert authority.max_delegation_waves == 1


@pytest.mark.anyio
async def test_redis_gateway_task_input_roundtrip_and_conflict() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    now = datetime.now(UTC).isoformat()
    record = TaskInput(
        task_id="t-1",
        prompt="Test prompt",
        created_at=now,
        input_version=1,
    )

    # 1. Initial write (Lua returns OK)
    gateway._client.eval = AsyncMock(return_value="OK")
    await gateway.save_task_input(record)
    gateway._client.eval.assert_awaited_once()

    # 2. Idempotent write (Lua returns IDEMPOTENT)
    gateway._client.eval = AsyncMock(return_value="IDEMPOTENT")
    await gateway.save_task_input(record)

    # 3. Conflicting write (Lua returns CONFLICT)
    gateway._client.eval = AsyncMock(return_value="CONFLICT")
    different = TaskInput(
        task_id="t-1",
        prompt="Different prompt",
        created_at=now,
        input_version=1,
    )
    with pytest.raises(ConflictingTaskInputError, match="already exists with conflicting data"):
        await gateway.save_task_input(different)

    # 4. Non-contiguous write (Lua returns NON_CONTIGUOUS)
    gateway._client.eval = AsyncMock(return_value="NON_CONTIGUOUS")
    v3 = TaskInput(
        task_id="t-1",
        prompt="Skip to v3",
        created_at=now,
        input_version=3,
    )
    with pytest.raises(NonContiguousVersionError, match="prior version 2 does not exist"):
        await gateway.save_task_input(v3)

    # 5. Retrieval
    gateway._client.get = AsyncMock(return_value=record.to_json())
    retrieved = await gateway.get_task_input("t-1", 1)
    assert retrieved == record

    # 6. Corrupt record detection
    gateway._client.get = AsyncMock(return_value="invalid JSON {")
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt TaskInput record"):
        await gateway.get_task_input("t-1", 1)


@pytest.mark.anyio
async def test_redis_gateway_stage_artifact_roundtrip_and_conflict() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    now = datetime.now(UTC).isoformat()
    art = StageArtifact(
        artifact_id="art-1",
        producing_task_id="t-1",
        artifact_type="research_material",
        content={"material": "sample"},
        created_at=now,
        artifact_version=1,
    )

    # 1. Save artifact (Lua returns OK)
    gateway._client.eval = AsyncMock(return_value="OK")
    await gateway.save_stage_artifact(art, "assign-1", "principal-1", 1)
    gateway._client.eval.assert_awaited_once()

    # 2. Idempotent write (Lua returns IDEMPOTENT)
    gateway._client.eval = AsyncMock(return_value="IDEMPOTENT")
    await gateway.save_stage_artifact(art, "assign-1", "principal-1", 1)

    # 3. Conflicting write (Lua returns CONFLICT)
    gateway._client.eval = AsyncMock(return_value="CONFLICT")
    conflict = StageArtifact(
        artifact_id="art-1",
        producing_task_id="t-1",
        artifact_type="research_material",
        content={"material": "conflicting_sample"},
        created_at=now,
        artifact_version=1,
    )
    with pytest.raises(ConflictingStageArtifactError, match="already exists with conflicting data"):
        await gateway.save_stage_artifact(conflict, "assign-1", "principal-1", 1)

    # 4. Non-contiguous write (Lua returns NON_CONTIGUOUS)
    gateway._client.eval = AsyncMock(return_value="NON_CONTIGUOUS")
    art_v3 = StageArtifact(
        artifact_id="art-1",
        producing_task_id="t-1",
        artifact_type="research_material",
        content={"material": "v3"},
        created_at=now,
        artifact_version=3,
    )
    with pytest.raises(NonContiguousVersionError, match="prior version 2 does not exist"):
        await gateway.save_stage_artifact(art_v3, "assign-1", "principal-1", 1)

    # 5. Retrieval
    gateway._client.get = AsyncMock(return_value=art.to_json())
    retrieved = await gateway.get_stage_artifact("t-1", "art-1", 1)
    assert retrieved == art

    # 6. Corrupt record detection
    gateway._client.get = AsyncMock(return_value='{"malformed": "content_missing"}')
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt StageArtifact record"):
        await gateway.get_stage_artifact("t-1", "art-1", 1)


@pytest.mark.anyio
async def test_redis_gateway_continuation_roundtrip_and_conflict() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    now = datetime.now(UTC).isoformat()
    cont = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-1",
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "sample focus"},
        continuation_version=1,
        created_at=now,
        updated_at=now,
    )

    # 1. Save continuation (Lua returns OK)
    gateway._client.eval = AsyncMock(return_value="OK")
    await gateway.save_continuation(cont)
    gateway._client.eval.assert_awaited_once()

    # 2. Idempotent write (Lua returns IDEMPOTENT)
    gateway._client.eval = AsyncMock(return_value="IDEMPOTENT")
    await gateway.save_continuation(cont)

    # 3. Conflicting write (Lua returns CONFLICT)
    gateway._client.eval = AsyncMock(return_value="CONFLICT")
    conflict = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-different",
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "different focus"},
        continuation_version=1,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(ConflictingContinuationError, match="already exists with conflicting data"):
        await gateway.save_continuation(conflict)

    # 4. Non-contiguous write (Lua returns NON_CONTIGUOUS)
    gateway._client.eval = AsyncMock(return_value="NON_CONTIGUOUS")
    cont_v3 = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-1",
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "focus-v3"},
        continuation_version=3,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(NonContiguousVersionError, match="prior version 2 does not exist"):
        await gateway.save_continuation(cont_v3)

    # 4b. Corrupt latest pointer write (Lua returns CORRUPT_POINTER)
    gateway._client.eval = AsyncMock(return_value="CORRUPT_POINTER")
    with pytest.raises(CorruptAuthoritativeRecordError, match="Continuation latest pointer"):
        await gateway.save_continuation(cont)

    # 5. Retrieval
    gateway._client.get = AsyncMock(return_value=cont.to_json())
    retrieved = await gateway.get_continuation("p-1", 1)
    assert retrieved == cont

    # 6. Corrupt record detection
    gateway._client.get = AsyncMock(return_value="corrupt continuation")
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt TaskContinuation record"):
        await gateway.get_continuation("p-1", 1)


@pytest.mark.anyio
async def test_redis_gateway_root_authority_roundtrip_and_conflict() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    now = datetime.now(UTC).isoformat()
    auth = RootAuthorityRecord.create_default("root-abc", created_at=now)

    # 1. Save authority (Lua returns OK)
    gateway._client.eval = AsyncMock(return_value="OK")
    await gateway.save_root_authority(auth)
    gateway._client.eval.assert_awaited_once()

    # 2. Idempotent write (Lua returns IDEMPOTENT)
    gateway._client.eval = AsyncMock(return_value="IDEMPOTENT")
    await gateway.save_root_authority(auth)

    # 3. Conflicting write (Lua returns CONFLICT)
    gateway._client.eval = AsyncMock(return_value="CONFLICT")
    conflict = RootAuthorityRecord.create_default(
        root_task_id="root-abc",
        created_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(ConflictingRootAuthorityError, match="already exists with conflicting data"):
        await gateway.save_root_authority(conflict)

    # 4. Retrieval
    gateway._client.get = AsyncMock(return_value=auth.to_json())
    retrieved = await gateway.get_root_authority("root-abc")
    assert retrieved == auth

    # 5. Corrupt record detection
    gateway._client.get = AsyncMock(return_value="corrupt authority data")
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt RootAuthorityRecord"):
        await gateway.get_root_authority("root-abc")


@pytest.mark.anyio
async def test_in_memory_version_contiguity_and_concurrency() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now = datetime.now(UTC).isoformat()

    # TaskInput contiguity: cannot save v2 without v1
    v2_input = TaskInput(task_id="t-c", prompt="v2 prompt", created_at=now, input_version=2)
    with pytest.raises(NonContiguousVersionError, match="prior version 1 does not exist"):
        await store.save_task_input(v2_input)

    # StageArtifact contiguity: cannot save v2 without v1
    claim_c = ActiveClaim(
        task_id="t-c",
        assignment_id="assign-c",
        claim_id="claim-c",
        root_task_id="t-c",
        turn_id="turn-c",
        runtime_principal_id="principal-c",
        bound_agent_id="research-c",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="c" * 64,
        request_id="req-c",
        claimed_at=now,
        delivery_expires_at="2099-01-01T00:00:00Z",
        state="acknowledged",
        acknowledged_at=now,
    )
    store.claims["t-c"] = claim_c
    store.fences["t-c"] = 1

    v2_art = StageArtifact(
        artifact_id="a-c",
        producing_task_id="t-c",
        artifact_type="research_material",
        content={"material": "v2"},
        created_at=now,
        artifact_version=2,
    )
    with pytest.raises(NonContiguousVersionError, match="prior version 1 does not exist"):
        await store.save_stage_artifact(v2_art, "assign-c", "principal-c", 1)

    # TaskContinuation latest pointer and contiguity:
    # Saving v2 without v1 (latest pointer missing) raises CorruptAuthoritativeRecordError
    v2_missing_cont = TaskContinuation(
        parent_task_id="t-missing",
        child_task_id="child-missing",
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "v2"},
        continuation_version=2,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(CorruptAuthoritativeRecordError):
        await store.save_continuation(v2_missing_cont)

    # Save v1, then attempt saving v3 (latest=1 < 3-1) raises NonContiguousVersionError
    v1_cont = TaskContinuation(
        parent_task_id="t-c",
        child_task_id="child-c",
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "v1"},
        continuation_version=1,
        created_at=now,
        updated_at=now,
    )
    await store.save_continuation(v1_cont)
    v3_cont = TaskContinuation(
        parent_task_id="t-c",
        child_task_id="child-c",
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "v3"},
        continuation_version=3,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(NonContiguousVersionError, match="prior version 2 does not exist"):
        await store.save_continuation(v3_cont)

    # Partial state conflict on root registration
    root = TaskRequested.create("Root task")
    # Manually inject only status (partial state)
    store.statuses[root.task_id] = "pending"
    with pytest.raises(ConflictingRootTaskRegistrationError, match="inconsistent or partial"):
        await store.register_root_task(root)


@pytest.mark.anyio
async def test_in_memory_authority_grant_write_once_and_conflict() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now = datetime.now(UTC).isoformat()
    grant = DelegationGrant(
        grant_id="grant-001",
        root_task_id="root-101",
        issued_to_assignment_id="assign-001",
        issued_to_principal_id="principal-research-1",
        fencing_epoch=1,
        turn_id="turn-001",
        issued_at=now,
    )
    await store.save_authority_grant(grant)

    # Retrieval by (root_task_id, grant_id)
    retrieved = await store.get_authority_grant("root-101", "grant-001")
    assert retrieved == grant

    # Non-existent grant_id returns None
    assert await store.get_authority_grant("root-101", "nonexistent") is None

    # Idempotent re-save
    await store.save_authority_grant(grant)

    # Conflicting save with same grant_id raises
    conflicting = DelegationGrant(
        grant_id="grant-001",
        root_task_id="root-101",
        issued_to_assignment_id="assign-002",
        issued_to_principal_id="principal-research-2",
        fencing_epoch=1,
        turn_id="turn-001",
        issued_at=now,
    )
    with pytest.raises(ConflictingAuthorityGrantError, match="already exists with conflicting data"):
        await store.save_authority_grant(conflicting)

    # Replacement assignment receives distinct grant bound to new assignment, epoch, turn
    grant_b = DelegationGrant(
        grant_id="grant-002",
        root_task_id="root-101",
        issued_to_assignment_id="assign-002",
        issued_to_principal_id="principal-research-2",
        fencing_epoch=2,
        turn_id="turn-002",
        issued_at=now,
    )
    await store.save_authority_grant(grant_b)

    # Both grants remain retrievable by exact grant_id and immutable
    assert await store.get_authority_grant("root-101", "grant-001") == grant
    assert await store.get_authority_grant("root-101", "grant-002") == grant_b
    assert grant.fencing_epoch != grant_b.fencing_epoch
    assert grant.issued_to_assignment_id != grant_b.issued_to_assignment_id


@pytest.mark.anyio
async def test_in_memory_atomic_claim_and_acknowledgement_lifecycle() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now = datetime.now(UTC).isoformat()
    future = "2099-01-01T00:00:00Z"

    root = TaskRequested.create("Root task")
    await store.register_root_task(root, required_capability="research")
    await store.update_status(root.task_id, "pending")

    grant = DelegationGrant(
        grant_id="grant-001",
        root_task_id=root.task_id,
        issued_to_assignment_id="assign-001",
        issued_to_principal_id="principal-research-1",
        fencing_epoch=1,
        turn_id="turn-001",
        issued_at=now,
    )

    claim = ActiveClaim(
        task_id=root.task_id,
        assignment_id="assign-001",
        claim_id="claim-001",
        root_task_id=root.task_id,
        turn_id="turn-001",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="digest-001",
        request_id="req-001",
        claimed_at=now,
        delivery_expires_at=future,
        delegation_grant_id="grant-001",
    )

    # 1. Successful initial claim
    active_claim, is_replay = await store.atomic_claim_task(
        task_id=root.task_id,
        claim=claim,
        grant=grant,
        now=now,
    )
    assert not is_replay
    assert active_claim.fencing_epoch == 1
    assert await store.get_status(root.task_id) == "working"
    assert await store.get_authority_grant(root.task_id, grant.grant_id) == grant

    # 2. Idempotent retry with same (principal_id, request_id)
    active_claim_2, is_replay_2 = await store.atomic_claim_task(
        task_id=root.task_id,
        claim=claim,
        grant=grant,
        now=now,
    )
    assert is_replay_2
    assert active_claim_2.assignment_id == "assign-001"

    # 3. Another principal attempts to claim while active
    other_claim = ActiveClaim(
        task_id=root.task_id,
        assignment_id="assign-002",
        claim_id="claim-002",
        root_task_id=root.task_id,
        turn_id="turn-002",
        runtime_principal_id="principal-research-2",
        bound_agent_id="research-agent-2",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=2,
        proof_digest="digest-002",
        request_id="req-002",
        claimed_at=now,
        delivery_expires_at=future,
    )
    with pytest.raises(TaskAlreadyClaimedError, match="active claim"):
        await store.atomic_claim_task(
            task_id=root.task_id,
            claim=other_claim,
            now=now,
        )

    # 4. Acknowledgement
    acked = await store.acknowledge_claim(
        task_id=root.task_id,
        assignment_id="assign-001",
        principal_id="principal-research-1",
        acknowledged_at=now,
    )
    assert acked.state == "acknowledged"
    assert acked.acknowledged_at == normalize_canonical_utc_timestamp(now)

    # Idempotent acknowledgement
    acked_again = await store.acknowledge_claim(
        task_id=root.task_id,
        assignment_id="assign-001",
        principal_id="principal-research-1",
        acknowledged_at=now,
    )
    assert acked_again.state == "acknowledged"

    # Acknowledgement mismatch (wrong assignment)
    with pytest.raises(ClaimNotFoundError, match="mismatch"):
        await store.acknowledge_claim(
            task_id=root.task_id,
            assignment_id="wrong-assignment",
            principal_id="principal-research-1",
            acknowledged_at=now,
        )

    # Acknowledgement mismatch (wrong principal)
    with pytest.raises(ClaimNotFoundError, match="mismatch"):
        await store.acknowledge_claim(
            task_id=root.task_id,
            assignment_id="assign-001",
            principal_id="wrong-principal",
            acknowledged_at=now,
        )


@pytest.mark.anyio
async def test_in_memory_delivery_lease_expiry_restores_claimability() -> None:
    store = InMemoryAuthoritativeTaskStore()
    past = "2020-01-01T00:00:00Z"
    now = "2020-01-01T00:00:31Z"

    root = TaskRequested.create("Root task")
    await store.register_root_task(root, required_capability="research")
    await store.update_status(root.task_id, "pending")

    claim = ActiveClaim(
        task_id=root.task_id,
        assignment_id="assign-001",
        claim_id="claim-001",
        root_task_id=root.task_id,
        turn_id="turn-001",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="digest-001",
        request_id="req-001",
        claimed_at=past,
        delivery_expires_at="2020-01-01T00:00:30Z",
    )

    await store.atomic_claim_task(
        task_id=root.task_id,
        claim=claim,
        now=past,
    )
    assert await store.get_status(root.task_id) == "working"

    # Acknowledgement fails if lease expired
    with pytest.raises(ClaimDeliveryLeaseExpiredError, match="expired"):
        await store.acknowledge_claim(
            task_id=root.task_id,
            assignment_id="assign-001",
            principal_id="principal-research-1",
            acknowledged_at=now,
        )

    # Expiry evaluation restores status: pending
    expired = await store.expire_delivery_lease(root.task_id, now=now)
    assert expired is True
    assert await store.get_status(root.task_id) == "pending"
    assert await store.get_active_claim(root.task_id) is None

    # Next claim receives incremented epoch 2
    claim_2 = ActiveClaim(
        task_id=root.task_id,
        assignment_id="assign-002",
        claim_id="claim-002",
        root_task_id=root.task_id,
        turn_id="turn-002",
        runtime_principal_id="principal-research-2",
        bound_agent_id="research-agent-2",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=2,
        proof_digest="digest-002",
        request_id="req-002",
        claimed_at=now,
        delivery_expires_at="2020-01-01T00:01:00Z",
    )
    new_claim, _ = await store.atomic_claim_task(
        task_id=root.task_id,
        claim=claim_2,
        now=now,
    )
    assert new_claim.fencing_epoch == 2


@pytest.mark.anyio
async def test_redis_gateway_authority_grant_roundtrip_and_conflict() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    now = datetime.now(UTC).isoformat()
    grant = DelegationGrant(
        grant_id="grant-001",
        root_task_id="root-101",
        issued_to_assignment_id="assign-001",
        issued_to_principal_id="principal-research-1",
        fencing_epoch=1,
        turn_id="turn-001",
        issued_at=now,
    )

    # 1. Save grant (Lua returns OK)
    gateway._client.eval = AsyncMock(return_value="OK")
    await gateway.save_authority_grant(grant)
    gateway._client.eval.assert_awaited_once()

    # 2. Idempotent write (Lua returns IDEMPOTENT)
    gateway._client.eval = AsyncMock(return_value="IDEMPOTENT")
    await gateway.save_authority_grant(grant)

    # 3. Conflicting write (Lua returns CONFLICT)
    gateway._client.eval = AsyncMock(return_value="CONFLICT")
    conflict = DelegationGrant(
        grant_id="grant-001",
        root_task_id="root-101",
        issued_to_assignment_id="assign-002",
        issued_to_principal_id="principal-research-2",
        fencing_epoch=1,
        turn_id="turn-001",
        issued_at=now,
    )
    with pytest.raises(ConflictingAuthorityGrantError, match="already exists with conflicting data"):
        await gateway.save_authority_grant(conflict)

    # 4. Retrieval
    gateway._client.get = AsyncMock(return_value=grant.to_json())
    retrieved = await gateway.get_authority_grant("root-101", "grant-001")
    assert retrieved == grant

    # 5. Corrupt record detection
    gateway._client.get = AsyncMock(return_value="corrupt grant data")
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt DelegationGrant"):
        await gateway.get_authority_grant("root-101", "grant-001")


@pytest.mark.anyio
async def test_redis_gateway_claim_and_acknowledgement_lua_branches() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    now = datetime.now(UTC).isoformat()
    claim = ActiveClaim(
        task_id="task-001",
        assignment_id="assign-001",
        claim_id="claim-001",
        root_task_id="task-001",
        turn_id="turn-001",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="digest-001",
        request_id="req-001",
        claimed_at=now,
        delivery_expires_at="2099-01-01T00:00:00Z",
    )

    # 1. NOT_CLAIMABLE
    gateway._client.eval = AsyncMock(return_value=["NOT_CLAIMABLE", "failed"])
    with pytest.raises(TaskNotClaimableError, match="non-claimable status 'failed'"):
        await gateway.atomic_claim_task("task-001", claim)

    # 2. ACTIVE_CLAIM_CONFLICT
    gateway._client.eval = AsyncMock(return_value=["ALREADY_CLAIMED", "{}"])
    with pytest.raises(TaskAlreadyClaimedError, match="active claim by another runtime"):
        await gateway.atomic_claim_task("task-001", claim)

    # 3. Success (Lua returns ["OK", "1"])
    return_dict = claim.to_dict()
    return_dict["fencing_epoch"] = 1
    gateway._client.eval = AsyncMock(return_value=["OK", "1"])
    claimed, is_replay = await gateway.atomic_claim_task("task-001", claim)
    assert is_replay is False
    assert claimed.fencing_epoch == 1

    # 4. Replay (Lua returns ["REPLAY", json])
    gateway._client.eval = AsyncMock(return_value=["REPLAY", json.dumps(return_dict)])
    claimed_replayed, is_replay_flag = await gateway.atomic_claim_task("task-001", claim)
    assert is_replay_flag is True
    assert claimed_replayed.assignment_id == "assign-001"

    # 4b. Replay after acknowledged (Lua returns ["ALREADY_ACKNOWLEDGED", json])
    gateway._client.eval = AsyncMock(return_value=["ALREADY_ACKNOWLEDGED", json.dumps(return_dict)])
    with pytest.raises(TaskAlreadyClaimedError, match="already acknowledged"):
        await gateway.atomic_claim_task("task-001", claim)

    # 4c. Fencing epoch race (Lua returns ["EPOCH_RACE", "1"])
    gateway._client.eval = AsyncMock(return_value=["EPOCH_RACE", "1"])
    with pytest.raises(TaskAlreadyClaimedError, match="fencing epoch race"):
        await gateway.atomic_claim_task("task-001", claim)

    # 4d. Conflicting authority grant (Lua returns ["CONFLICT_GRANT"])
    gateway._client.eval = AsyncMock(return_value=["CONFLICT_GRANT"])
    with pytest.raises(ConflictingAuthorityGrantError, match="already exists with conflicting data"):
        await gateway.atomic_claim_task("task-001", claim)

    # 4e. Corrupt or inconsistent grant inputs
    gateway._client.eval = AsyncMock(return_value=["CORRUPT_GRANT_INPUT"])
    with pytest.raises(CorruptAuthoritativeRecordError, match="Inconsistent authority grant inputs"):
        await gateway.atomic_claim_task("task-001", claim)

    # 4f. Corrupt authoritative records
    for corrupt_code in ("CORRUPT_CLAIM", "CORRUPT_FENCE", "CORRUPT_STATUS"):
        gateway._client.eval = AsyncMock(return_value=[corrupt_code])
        with pytest.raises(CorruptAuthoritativeRecordError, match=corrupt_code):
            await gateway.atomic_claim_task("task-001", claim)

    # 5. Acknowledge: CLAIM_NOT_FOUND
    gateway._client.eval = AsyncMock(return_value=["CLAIM_NOT_FOUND"])
    with pytest.raises(ClaimNotFoundError, match="not found"):
        await gateway.acknowledge_claim("task-001", "assign-001", "principal-1", now)

    # 6. Acknowledge: MISMATCH
    gateway._client.eval = AsyncMock(return_value=["MISMATCH"])
    with pytest.raises(ClaimNotFoundError, match="does not match"):
        await gateway.acknowledge_claim("task-001", "wrong-assign", "principal-1", now)

    # 7. Acknowledge: LEASE_EXPIRED
    gateway._client.eval = AsyncMock(return_value=["LEASE_EXPIRED"])
    with pytest.raises(ClaimDeliveryLeaseExpiredError, match="Delivery lease for assignment"):
        await gateway.acknowledge_claim("task-001", "assign-001", "principal-1", now)

    # 8. Acknowledge: Success
    acked_dict = claim.to_dict()
    acked_dict["state"] = "acknowledged"
    acked_dict["acknowledged_at"] = normalize_canonical_utc_timestamp(now)
    gateway._client.eval = AsyncMock(return_value=["OK", json.dumps(acked_dict)])
    acked_claim = await gateway.acknowledge_claim("task-001", "assign-001", "principal-1", now)
    assert acked_claim.state == "acknowledged"

    # 9. Expire delivery lease: returned 1
    gateway._client.eval = AsyncMock(return_value=1)
    assert await gateway.expire_delivery_lease("task-001", now) is True

    # 10. Expire delivery lease: returned 0
    gateway._client.eval = AsyncMock(return_value=0)
    assert await gateway.expire_delivery_lease("task-001", now) is False

    # 10b. Expire delivery lease: returned CORRUPT_CLAIM
    gateway._client.eval = AsyncMock(return_value=["CORRUPT_CLAIM"])
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt authoritative record"):
        await gateway.expire_delivery_lease("task-001", now)

    # 11. Dispatch hint
    gateway._client.rpush = AsyncMock()
    hint = WakeUpHint(
        hint_id="hint-1",
        capability="research",
        created_at=now,
        execution_kind="initial",
        task_id="task-001",
    )
    await gateway.dispatch_hint(hint, "agent.hints.research")
    gateway._client.rpush.assert_awaited_once_with("agent.hints.research", hint.to_json())


@pytest.mark.anyio
async def test_in_memory_atomic_claim_zero_mutation_invariants() -> None:
    store = InMemoryAuthoritativeTaskStore()
    past = "2020-01-01T00:00:00Z"
    now = "2020-01-01T00:00:15Z"
    future = "2099-01-01T00:00:00Z"

    root = TaskRequested.create("Root task for invariant tests")
    await store.register_root_task(root, required_capability="research")
    await store.update_status(root.task_id, "pending")

    grant_1 = DelegationGrant(
        grant_id="grant-001",
        root_task_id=root.task_id,
        issued_to_assignment_id="assign-001",
        issued_to_principal_id="principal-research-1",
        fencing_epoch=1,
        turn_id="turn-001",
        issued_at=past,
    )
    claim_1 = ActiveClaim(
        task_id=root.task_id,
        assignment_id="assign-001",
        claim_id="claim-001",
        root_task_id=root.task_id,
        turn_id="turn-001",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="a" * 64,
        request_id="req-001",
        claimed_at=past,
        delivery_expires_at=past,  # Expired lease
        delegation_grant_id=grant_1.grant_id,
    )

    # Initial claim
    await store.atomic_claim_task(root.task_id, claim_1, grant=grant_1, now=past)
    assert store.fences[root.task_id] == 1
    assert store.statuses[root.task_id] == "working"
    assert store.claims[root.task_id] == claim_1
    assert store.authority_grants[(root.task_id, grant_1.grant_id)] == grant_1

    # Status back to pending for re-claim
    await store.update_status(root.task_id, "pending")

    # Invariant 1: Expired claim + epoch race leaves expired claim and status unchanged
    claim_bad_epoch = ActiveClaim(
        task_id=root.task_id,
        assignment_id="assign-002",
        claim_id="claim-002",
        root_task_id=root.task_id,
        turn_id="turn-002",
        runtime_principal_id="principal-research-2",
        bound_agent_id="research-agent-2",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=99,  # Epoch race
        proof_digest="b" * 64,
        request_id="req-002",
        claimed_at=now,
        delivery_expires_at=future,
    )
    with pytest.raises(TaskAlreadyClaimedError, match="fencing epoch race"):
        await store.atomic_claim_task(root.task_id, claim_bad_epoch, now=now)

    assert store.fences[root.task_id] == 1
    assert store.statuses[root.task_id] == "pending"
    assert store.claims[root.task_id] == claim_1  # Claim 1 preserved, NOT deleted!

    # Invariant 2: Conflicting grant leaves fence, status, active claim, and existing grant unchanged
    grant_conflict = DelegationGrant(
        grant_id="grant-001",  # Same grant_id
        root_task_id=root.task_id,
        issued_to_assignment_id="assign-002",  # Different assignment
        issued_to_principal_id="principal-research-2",  # Different principal
        fencing_epoch=2,
        turn_id="turn-002",
        issued_at=now,
    )
    claim_2_candidate = ActiveClaim(
        task_id=root.task_id,
        assignment_id="assign-002",
        claim_id="claim-002",
        root_task_id=root.task_id,
        turn_id="turn-002",
        runtime_principal_id="principal-research-2",
        bound_agent_id="research-agent-2",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=2,
        proof_digest="c" * 64,
        request_id="req-002",
        claimed_at=now,
        delivery_expires_at=future,
        delegation_grant_id=grant_conflict.grant_id,
    )
    with pytest.raises(ConflictingAuthorityGrantError, match="already exists with conflicting data"):
        await store.atomic_claim_task(root.task_id, claim_2_candidate, grant=grant_conflict, now=now)

    assert store.fences[root.task_id] == 1  # Fence NOT incremented!
    assert store.statuses[root.task_id] == "pending"  # Status NOT working!
    assert store.claims[root.task_id] == claim_1  # Claim 1 preserved!
    assert store.authority_grants[(root.task_id, grant_1.grant_id)] == grant_1  # Grant 1 untouched!

    # Invariant 3: Successful replacement of an expired claim advances fence exactly once,
    # stores replacement claim, changes status to working, and preserves historical grant
    grant_2_valid = DelegationGrant(
        grant_id="grant-002",
        root_task_id=root.task_id,
        issued_to_assignment_id="assign-002",
        issued_to_principal_id="principal-research-2",
        fencing_epoch=2,
        turn_id="turn-002",
        issued_at=now,
    )
    claim_2_valid = ActiveClaim(
        task_id=root.task_id,
        assignment_id="assign-002",
        claim_id="claim-002",
        root_task_id=root.task_id,
        turn_id="turn-002",
        runtime_principal_id="principal-research-2",
        bound_agent_id="research-agent-2",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=2,
        proof_digest="c" * 64,
        request_id="req-002",
        claimed_at=now,
        delivery_expires_at=future,
        delegation_grant_id=grant_2_valid.grant_id,
    )
    res_claim, is_replay = await store.atomic_claim_task(root.task_id, claim_2_valid, grant=grant_2_valid, now=now)
    assert not is_replay
    assert res_claim.fencing_epoch == 2
    assert store.fences[root.task_id] == 2
    assert store.statuses[root.task_id] == "working"
    assert store.claims[root.task_id] == claim_2_valid
    assert store.authority_grants[(root.task_id, grant_1.grant_id)] == grant_1  # Historical grant preserved!
    assert store.authority_grants[(root.task_id, grant_2_valid.grant_id)] == grant_2_valid  # New grant stored!

    # Invariant 4: Malformed claim in expiry fails closed
    store.claims[root.task_id] = "corrupt_non_dict"  # type: ignore[assignment]
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt active claim"):
        await store.expire_delivery_lease(root.task_id, now=now)


@pytest.mark.anyio
async def test_continuation_version_numeric_hardening_and_zero_mutation() -> None:
    store = InMemoryAuthoritativeTaskStore()
    t_id = "root-cont-harden"
    now = "2026-11-17T12:00:00Z"
    future = "2099-01-01T00:00:00Z"

    root = TaskRequested.create("Root for continuation numeric hardening")
    object.__setattr__(root, "task_id", t_id)
    object.__setattr__(root, "root_task_id", t_id)
    await store.register_root_task(root, initial_status="pending_dispatch", required_capability="general")
    await store.update_status(t_id, "resumable")

    claim_candidate = ActiveClaim(
        task_id=t_id,
        assignment_id="assign-resumed-001",
        claim_id="claim-resumed-001",
        root_task_id=t_id,
        turn_id="turn-resumed-001",
        runtime_principal_id="principal-worker-1",
        bound_agent_id="worker-1",
        assigned_capability="general",
        execution_kind="resumed_continuation",
        fencing_epoch=1,
        proof_digest="a" * 64,
        request_id="req-resumed-001",
        claimed_at=now,
        delivery_expires_at=future,
        prior_claimable_status="resumable",
    )

    # 1. Zero and negative expected continuation versions rejected with ValueError, zero mutations
    for bad_ver in (0, -1, -5):
        with pytest.raises(ValueError, match="expected_continuation_version must be a positive integer"):
            await store.atomic_claim_task(t_id, claim_candidate, expected_continuation_version=bad_ver, now=now)

        assert store.statuses[t_id] == "resumable"
        assert store.fences.get(t_id, 0) == 0
        assert t_id not in store.claims

    # 3. Nonnumeric authoritative latest-pointer fails closed, zero mutations
    store.latest_continuation[t_id] = "corrupt_non_numeric"  # type: ignore[assignment]
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt continuation latest record"):
        await store.atomic_claim_task(t_id, claim_candidate, expected_continuation_version=1, now=now)

    assert store.statuses[t_id] == "resumable"
    assert store.fences.get(t_id, 0) == 0
    assert t_id not in store.claims

    # 4. Fractional authoritative latest-pointer fails closed, zero mutations
    store.latest_continuation[t_id] = 1.5  # type: ignore[assignment]
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt continuation latest record"):
        await store.atomic_claim_task(t_id, claim_candidate, expected_continuation_version=1, now=now)

    assert store.statuses[t_id] == "resumable"
    assert store.fences.get(t_id, 0) == 0
    assert t_id not in store.claims

    # 5. Zero and negative authoritative latest-pointer fail closed, zero mutations
    for bad_ptr in (0, -1, -5):
        store.latest_continuation[t_id] = bad_ptr
        with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt continuation latest record"):
            await store.atomic_claim_task(t_id, claim_candidate, expected_continuation_version=1, now=now)

        assert store.statuses[t_id] == "resumable"
        assert store.fences.get(t_id, 0) == 0
        assert t_id not in store.claims

    # 6. Mismatch returns continuation-version conflict and causes zero mutations
    store.latest_continuation[t_id] = 1
    with pytest.raises(TaskNotClaimableError, match="Continuation version mismatch"):
        await store.atomic_claim_task(t_id, claim_candidate, expected_continuation_version=2, now=now)

    assert store.statuses[t_id] == "resumable"
    assert store.fences.get(t_id, 0) == 0
    assert t_id not in store.claims

    # 7. Valid positive integer matching version succeeds and commits state
    res_claim, is_replay = await store.atomic_claim_task(t_id, claim_candidate, expected_continuation_version=1, now=now)
    assert not is_replay
    assert res_claim.fencing_epoch == 1
    assert store.fences[t_id] == 1
    assert store.statuses[t_id] == "working"
    assert store.claims[t_id] == claim_candidate


@pytest.mark.anyio
async def test_in_memory_child_registration_atomic_boundary_and_immutability() -> None:
    store = InMemoryAuthoritativeTaskStore()
    root_id = "root-child-test"
    child_id = "child-test-001"
    now = "2026-11-17T12:00:00Z"

    root = TaskRequested.create("Root task")
    object.__setattr__(root, "task_id", root_id)
    object.__setattr__(root, "root_task_id", root_id)
    await store.register_root_task(root)

    child = TaskRequested(
        task_id=child_id,
        prompt="Child prompt",
        created_at=now,
        root_task_id=root_id,
        parent_task_id=root_id,
        version=2,
    )

    # 1. First registration creates all records atomically
    await store.register_child_task(child, initial_status="pending_dispatch", required_capability="analysis")
    assert store.statuses[child_id] == "pending_dispatch"
    assert store.lineages[child_id].task_id == child_id
    assert store.required_capabilities[child_id].required_capability == "analysis"
    assert child_id in store.children[root_id]

    # 2. Byte-identical replay succeeds idempotently
    await store.register_child_task(child, initial_status="pending_dispatch", required_capability="analysis")
    assert store.statuses[child_id] == "pending_dispatch"

    # 3. Conflicting capability replay fails with ConflictingChildTaskRegistrationError and zero mutation
    with pytest.raises(ConflictingChildTaskRegistrationError, match="already exists with conflicting state"):
        await store.register_child_task(child, initial_status="pending_dispatch", required_capability="research")
    assert store.required_capabilities[child_id].required_capability == "analysis"

    # 4. Conflicting lineage replay fails with ConflictingChildTaskRegistrationError and zero mutation
    conflicting_child = TaskRequested(
        task_id=child_id,
        prompt="Child prompt",
        created_at=now,
        root_task_id="other-root",
        parent_task_id=root_id,
        version=2,
    )
    with pytest.raises(ConflictingChildTaskRegistrationError, match="already exists with conflicting state"):
        await store.register_child_task(conflicting_child, initial_status="pending_dispatch", required_capability="analysis")
    assert store.lineages[child_id].root_task_id == root_id

    # 5. Partial prior state fails closed with ConflictingChildTaskRegistrationError and zero mutation
    partial_child_id = "child-partial-001"
    store.statuses[partial_child_id] = "pending_dispatch"
    partial_child = TaskRequested(
        task_id=partial_child_id,
        prompt="Partial child",
        created_at=now,
        root_task_id=root_id,
        parent_task_id=root_id,
        version=2,
    )
    with pytest.raises(ConflictingChildTaskRegistrationError, match="inconsistent or partial authoritative state"):
        await store.register_child_task(partial_child, initial_status="pending_dispatch", required_capability="analysis")
    assert partial_child_id not in store.lineages
    assert partial_child_id not in store.required_capabilities
    assert partial_child_id not in store.children.get(root_id, set())


@pytest.mark.anyio
async def test_registration_capability_validation_and_zero_mutation() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now = datetime.now(UTC).isoformat()
    root = TaskRequested(
        task_id="root-val-1",
        prompt="Root prompt",
        created_at=now,
        root_task_id="root-val-1",
        parent_task_id=None,
        version=2,
    )
    child = TaskRequested(
        task_id="child-val-1",
        prompt="Child prompt",
        created_at=now,
        root_task_id="root-val-1",
        parent_task_id="root-val-1",
        version=2,
    )

    # 1. Explicit empty root capability does not become 'general'; raises ValueError before mutation
    with pytest.raises(ValueError, match="required_capability must be a non-empty string"):
        await store.register_root_task(root, required_capability="")
    assert "root-val-1" not in store.registrations
    assert "root-val-1" not in store.statuses
    assert "root-val-1" not in store.lineages
    assert "root-val-1" not in store.required_capabilities

    # 2. Whitespace-only root capability raises ValueError
    with pytest.raises(ValueError, match="required_capability must be a non-empty string"):
        await store.register_root_task(root, required_capability="   ")
    assert "root-val-1" not in store.registrations

    # 3. Unknown root capability raises ValueError
    with pytest.raises(ValueError, match="Unrecognized required_capability 'unknown'"):
        await store.register_root_task(root, required_capability="unknown")
    assert "root-val-1" not in store.registrations

    # 4. Explicit empty child capability does not become 'analysis'; raises ValueError before mutation
    with pytest.raises(ValueError, match="required_capability must be a non-empty string"):
        await store.register_child_task(child, required_capability="")
    assert "child-val-1" not in store.statuses
    assert "child-val-1" not in store.lineages
    assert "child-val-1" not in store.required_capabilities
    assert "child-val-1" not in store.children.get("root-val-1", set())

    # 5. Whitespace-only child capability raises ValueError
    with pytest.raises(ValueError, match="required_capability must be a non-empty string"):
        await store.register_child_task(child, required_capability="   ")
    assert "child-val-1" not in store.statuses

    # 6. Unknown child capability raises ValueError
    with pytest.raises(ValueError, match="Unrecognized required_capability 'fast'"):
        await store.register_child_task(child, required_capability="fast")
    assert "child-val-1" not in store.statuses

    # 7. Valid default registrations succeed unchanged
    await store.register_root_task(root)
    assert store.required_capabilities["root-val-1"].required_capability == "general"

    await store.register_child_task(child)
    assert store.required_capabilities["child-val-1"].required_capability == "analysis"

    # 8. Valid explicit capabilities succeed
    root_res = TaskRequested(
        task_id="root-val-2",
        prompt="Root prompt 2",
        created_at=now,
        root_task_id="root-val-2",
        parent_task_id=None,
        version=2,
    )
    await store.register_root_task(root_res, required_capability="research")
    assert store.required_capabilities["root-val-2"].required_capability == "research"

    child_res = TaskRequested(
        task_id="child-val-2",
        prompt="Child prompt 2",
        created_at=now,
        root_task_id="root-val-1",
        parent_task_id="root-val-1",
        version=2,
    )
    await store.register_child_task(child_res, required_capability="research")
    assert store.required_capabilities["child-val-2"].required_capability == "research"


@pytest.mark.anyio
async def test_in_memory_accept_delegation_proposal_and_replay():
    store = InMemoryAuthoritativeTaskStore()
    now = normalize_canonical_utc_timestamp(datetime.now(UTC))

    # Setup parent root task
    root_task = TaskRequested(
        task_id="parent-1",
        prompt="Conduct research",
        created_at=now,
        root_task_id="parent-1",
        parent_task_id=None,
        version=2,
    )
    await store.register_root_task(root_task, initial_status="pending", required_capability="research")

    # Claim parent task as Research
    claim = ActiveClaim(
        task_id="parent-1",
        assignment_id="asgn-1",
        claim_id="claim-1",
        root_task_id="parent-1",
        turn_id="turn-1",
        runtime_principal_id="research-agent-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="digest-1",
        request_id="req-1",
        claimed_at=now,
        delivery_expires_at="2099-01-01T00:00:00.000000Z",
        prior_claimable_status="pending",
    )
    grant = DelegationGrant(
        grant_id="grant-1",
        root_task_id="parent-1",
        issued_to_assignment_id="asgn-1",
        issued_to_principal_id="research-agent-1",
        fencing_epoch=1,
        turn_id="turn-1",
        issued_at=now,
    )
    await store.atomic_claim_task("parent-1", claim, grant=grant, now=now)
    await store.acknowledge_claim("parent-1", "asgn-1", "research-agent-1", now)

    # Save stage artifact
    art = StageArtifact(
        artifact_id="art-1",
        producing_task_id="parent-1",
        artifact_type="research_material",
        content={"material": "sample research"},
        created_at=now,
        producing_turn_id="turn-1",
        artifact_version=1,
    )
    await store.save_stage_artifact(art, "asgn-1", "research-agent-1", 1)

    # Prepare delegation proposal
    art_ref = ArtifactReference(
        artifact_id="art-1",
        artifact_version=1,
        producing_task_id="parent-1",
    )
    fp = compute_proposal_fingerprint(
        parent_task_id="parent-1",
        proposal_id="prop-1",
        assignment_id="asgn-1",
        target_capability="analysis",
        child_intent="Analyze anomalies",
        artifact_reference=art_ref,
        parent_input_version=1,
    )
    proposal = DelegationProposalRecord(
        parent_task_id="parent-1",
        proposal_id="prop-1",
        child_task_id="child-1",
        delegation_decision_id="dec-1",
        assignment_id="asgn-1",
        runtime_principal_id="research-agent-1",
        bound_agent_id="research-agent-1",
        proof_digest="digest-1",
        proposal_fingerprint=fp,
        delegation_grant_id="grant-1",
        child_claim_deadline="2099-01-01T00:02:00.000000Z",
        accepted_at=now,
        target_capability="analysis",
        continuation_version=1,
    )
    child_input = TaskInput(
        task_id="child-1",
        prompt="Analyze anomalies",
        created_at=now,
        input_version=1,
        artifact_references=(art_ref,),
    )
    continuation = TaskContinuation(
        parent_task_id="parent-1",
        child_task_id="child-1",
        created_at=now,
        updated_at=now,
        input_version=1,
        continuation_capability="research",
        stage="incorporate_analysis_child",
        artifact_references=(art_ref,),
        continuation_version=1,
    )
    yield_rec = AssignmentYieldRecord(
        assignment_id="asgn-1",
        parent_task_id="parent-1",
        principal_id="research-agent-1",
        proposal_id="prop-1",
        child_task_id="child-1",
        delegation_decision_id="dec-1",
        continuation_version=1,
        yielded_at=now,
    )
    hint = WakeUpHint(
        hint_id="hint-1",
        capability="analysis",
        created_at=now,
        execution_kind="delegated_child",
        task_id="child-1",
    )

    # 1. First acceptance
    accepted, is_replay = await store.accept_delegation_proposal(
        parent_task_id="parent-1",
        proposal=proposal,
        child_task_input=child_input,
        parent_continuation=continuation,
        yield_record=yield_rec,
        wake_up_hint=hint,
        expected_fencing_epoch=1,
        now=now,
    )
    assert not is_replay
    assert accepted == proposal
    assert store.statuses["parent-1"] == "waiting_for_children"
    assert store.statuses["child-1"] == "pending"
    assert "parent-1" not in store.claims  # Retired active claim
    assert store.yield_records[("parent-1", "asgn-1")] == yield_rec
    assert "1" in store.consumed_waves["parent-1"]
    assert "grant-1" in store.consumed_grants["parent-1"]
    assert len(store.hints) == 1

    # 2. Idempotent replay: returns stored proposal, zero mutations, no second hint
    replayed, is_replay_2 = await store.accept_delegation_proposal(
        parent_task_id="parent-1",
        proposal=proposal,
        child_task_input=child_input,
        parent_continuation=continuation,
        yield_record=yield_rec,
        wake_up_hint=hint,
        expected_fencing_epoch=1,
        now=now,
    )
    assert is_replay_2
    assert replayed == proposal
    assert len(store.hints) == 1

    # 3. Conflicting replay (mismatched fingerprint) fails closed
    conflicting_proposal = DelegationProposalRecord(
        parent_task_id="parent-1",
        proposal_id="prop-1",
        child_task_id="child-1",
        delegation_decision_id="dec-1",
        assignment_id="asgn-1",
        runtime_principal_id="research-agent-1",
        bound_agent_id="research-agent-1",
        proof_digest="digest-1",
        proposal_fingerprint="conflicting-fingerprint",
        delegation_grant_id="grant-1",
        child_claim_deadline="2099-01-01T00:02:00.000000Z",
        accepted_at=now,
        target_capability="analysis",
        continuation_version=1,
    )
    with pytest.raises(DelegationConflictError, match="Conflicting proposal replay"):
        await store.accept_delegation_proposal(
            parent_task_id="parent-1",
            proposal=conflicting_proposal,
            child_task_input=child_input,
            parent_continuation=continuation,
            yield_record=yield_rec,
            wake_up_hint=hint,
            expected_fencing_epoch=1,
            now=now,
        )


@pytest.mark.anyio
async def test_in_memory_child_claim_deadline_enforcement():
    store = InMemoryAuthoritativeTaskStore()
    now_past = "2026-01-01T00:00:00.000000Z"
    deadline = "2026-01-01T00:02:00.000000Z"
    now_after_deadline = "2026-01-01T00:02:01.000000Z"

    store.statuses["child-1"] = "pending"
    store.lineages["child-1"] = TaskLineage(task_id="child-1", root_task_id="root-1", parent_task_id="root-1")
    store.required_capabilities["child-1"] = RequiredCapabilityRecord(task_id="child-1", required_capability="analysis", created_at=now_past)
    store.claim_deadlines["child-1"] = deadline

    claim = ActiveClaim(
        task_id="child-1",
        assignment_id="asgn-c-1",
        claim_id="claim-c-1",
        root_task_id="root-1",
        turn_id="turn-c-1",
        runtime_principal_id="analysis-agent-1",
        bound_agent_id="analysis-agent-1",
        assigned_capability="analysis",
        execution_kind="delegated_child",
        fencing_epoch=1,
        proof_digest="digest-c",
        request_id="req-c-1",
        claimed_at=now_after_deadline,
        delivery_expires_at="2099-01-01T00:00:00.000000Z",
        prior_claimable_status="pending",
    )

    # Claim after deadline is rejected
    with pytest.raises(ChildClaimDeadlineExpiredError, match="Claim deadline for child task 'child-1' has expired"):
        await store.atomic_claim_task("child-1", claim, now=now_after_deadline, is_root=False)

    # Claim at or before deadline succeeds
    now_within = "2026-01-01T00:02:00.000000Z"
    active, is_rep = await store.atomic_claim_task("child-1", claim, now=now_within, is_root=False)
    assert not is_rep
    assert active.task_id == "child-1"
    assert store.statuses["child-1"] == "working"


def test_consolidated_exception_class_identity():
    from orchestrator.application import ports
    from orchestrator.domain import task
    from orchestrator.presentation import api

    # Proves the 4 exception classes are removed from domain/task.py
    for exc_name in (
        "DelegationConflictError",
        "WaveBudgetExceededError",
        "GrantAlreadyConsumedError",
        "ChildClaimDeadlineExpiredError",
    ):
        assert not hasattr(task, exc_name), f"{exc_name} must not be defined in domain.task"

    # Proves the exception raised by store/ports is identical to the one caught in presentation/api
    assert ports.DelegationConflictError is api.DelegationConflictError
    assert ports.WaveBudgetExceededError is api.WaveBudgetExceededError
    assert ports.GrantAlreadyConsumedError is api.GrantAlreadyConsumedError
    assert ports.ChildClaimDeadlineExpiredError is api.ChildClaimDeadlineExpiredError


@pytest.mark.anyio
async def test_redis_gateway_materialize_claim_timeout_branches() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    now_iso = "2026-11-22T12:00:01.000000Z"
    deadline_iso = "2026-11-22T12:00:00.000000Z"
    outcome = build_claim_timed_out_terminal_outcome(
        task_id="child-1",
        claim_deadline=deadline_iso,
        completed_at=now_iso,
    )
    hint = WakeUpHint(
        hint_id="hint-1",
        capability="research",
        created_at=now_iso,
        execution_kind="resumed_continuation",
        task_id="parent-1",
        expected_continuation_version=1,
    )

    gateway.get_lineage = AsyncMock(
        return_value=TaskLineage(task_id="child-1", root_task_id="parent-1", parent_task_id="parent-1")
    )
    gateway._client.get = AsyncMock(return_value=deadline_iso)

    # 1. CHILD_NOT_PENDING
    gateway._client.eval = AsyncMock(return_value=["CHILD_NOT_PENDING", "completed"])
    with pytest.raises(TaskNotClaimableError, match="status: got 'completed'"):
        await gateway.materialize_claim_timeout("child-1", now_iso)

    # 2. DEADLINE_NOT_EXPIRED
    gateway._client.eval = AsyncMock(return_value=["DEADLINE_NOT_EXPIRED", deadline_iso])
    with pytest.raises(TaskNotClaimableError, match="has not expired"):
        await gateway.materialize_claim_timeout("child-1", now_iso)

    # 3. PARENT_NOT_FOUND
    gateway._client.eval = AsyncMock(return_value=["PARENT_NOT_FOUND"])
    with pytest.raises(TaskNotClaimableError, match="Required record not found"):
        await gateway.materialize_claim_timeout("child-1", now_iso)

    # 4. ACTIVE_CLAIM_EXISTS
    gateway._client.eval = AsyncMock(return_value=["ACTIVE_CLAIM_EXISTS"])
    with pytest.raises(TaskAlreadyClaimedError, match="(?i)active claim exists"):
        await gateway.materialize_claim_timeout("child-1", now_iso)

    # 5. OK
    gateway._client.eval = AsyncMock(return_value=["OK", outcome.to_json(), hint.to_json()])
    res, is_rep = await gateway.materialize_claim_timeout("child-1", now_iso)
    assert not is_rep
    assert res.task_id == "child-1"
    assert res.terminal_status == "claim_timed_out"

    # 6. IDEMPOTENT (no hint emitted)
    gateway._client.eval = AsyncMock(return_value=["IDEMPOTENT", outcome.to_json(), ""])
    res2, is_rep2 = await gateway.materialize_claim_timeout("child-1", now_iso)
    assert is_rep2
    assert res2.outcome_id == outcome.outcome_id


@pytest.mark.anyio
async def test_redis_gateway_transition_expired_claim_branches() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    now_iso = "2026-11-22T12:00:01.000000Z"
    claim = ActiveClaim(
        task_id="task-1",
        root_task_id="task-1",
        assignment_id="asgn-1",
        claim_id="claim-1",
        turn_id="turn-1",
        fencing_epoch=2,
        runtime_principal_id="p-1",
        bound_agent_id="a-1",
        assigned_capability="research",
        execution_kind="initial",
        prior_claimable_status="pending",
        proof_digest="digest",
        request_id="req-1",
        delivery_expires_at="2026-11-22T12:00:00.000000Z",
        claimed_at="2026-11-22T11:00:00.000000Z",
        state="acknowledged",
        acknowledged_at="2026-11-22T11:00:00.000000Z",
        execution_expires_at="2026-11-22T12:00:00.000000Z",
    )
    rec = ClaimExpiryRecord(
        task_id="task-1",
        claim_id="claim-1",
        assignment_id="asgn-1",
        principal_id="p-1",
        bound_agent_id="a-1",
        execution_kind="initial",
        expired_fencing_epoch=2,
        acknowledged_at="2026-11-22T11:00:00.000000Z",
        execution_expires_at="2026-11-22T12:00:00.000000Z",
        expired_at=now_iso,
        prior_claimable_status="pending",
    )

    gateway.get_active_claim = AsyncMock(return_value=claim)

    # 1. NOT_WORKING
    gateway._client.eval = AsyncMock(return_value=["NOT_WORKING", "pending"])
    with pytest.raises(TaskNotClaimableError, match="not in 'working' status"):
        await gateway.transition_expired_claim("task-1", now_iso)

    # 2. EXECUTION_NOT_EXPIRED
    gateway._client.eval = AsyncMock(return_value=["EXECUTION_NOT_EXPIRED", "2099-01-01T00:00:00Z"])
    with pytest.raises(TaskNotClaimableError, match="has not expired"):
        await gateway.transition_expired_claim("task-1", now_iso)

    # 3. OK
    gateway._client.eval = AsyncMock(return_value=["OK", rec.to_json()])
    res, is_rep = await gateway.transition_expired_claim("task-1", now_iso)
    assert not is_rep
    assert res.task_id == "task-1"
    assert res.claim_id == "claim-1"

    # 4. IDEMPOTENT via Lua
    gateway._client.eval = AsyncMock(return_value=["IDEMPOTENT", rec.to_json()])
    res2, is_rep2 = await gateway.transition_expired_claim("task-1", now_iso)
    assert is_rep2
    assert res2.claim_id == "claim-1"

    # 5. IDEMPOTENT via pre-check (active claim absent, expiry record and recovery_required status present)
    gateway.get_active_claim = AsyncMock(return_value=None)
    gateway.get_claim_expiry = AsyncMock(return_value=rec)
    gateway.get_status = AsyncMock(return_value="recovery_required")
    res3, is_rep3 = await gateway.transition_expired_claim("task-1", now_iso)
    assert is_rep3
    assert res3.claim_id == "claim-1"


@pytest.mark.anyio
async def test_redis_gateway_resolve_expired_claim_branches() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    now_iso = "2026-11-22T12:00:01.000000Z"
    cmd = RecoveryDecisionCommand(
        recovery_decision_id="rec-1",
        task_id="task-1",
        expired_claim_id="claim-1",
        expired_assignment_id="asgn-1",
        expired_fencing_epoch=2,
        disposition="restore_claimable",
    )
    rec = RecoveryDecisionRecord(
        recovery_decision_id="rec-1",
        task_id="task-1",
        expired_claim_id="claim-1",
        expired_assignment_id="asgn-1",
        expired_fencing_epoch=2,
        disposition="restore_claimable",
        semantic_fingerprint=cmd.semantic_fingerprint(),
        prior_status="recovery_required",
        restored_status="pending",
        operator_principal_id="p-operator",
        resolved_at=now_iso,
    )
    expiry = ClaimExpiryRecord(
        task_id="task-1",
        claim_id="claim-1",
        assignment_id="asgn-1",
        principal_id="p-1",
        bound_agent_id="a-1",
        execution_kind="initial",
        expired_fencing_epoch=2,
        acknowledged_at=now_iso,
        execution_expires_at=now_iso,
        expired_at=now_iso,
        prior_claimable_status="pending",
    )

    gateway.get_claim_expiry = AsyncMock(return_value=expiry)

    # 1. NOT_RECOVERY_REQUIRED
    gateway._client.eval = AsyncMock(return_value=["NOT_RECOVERY_REQUIRED", "completed"])
    with pytest.raises(RecoveryPreconditionFailedError, match="not in 'recovery_required' status"):
        await gateway.resolve_expired_claim(cmd, "p-operator", now_iso)

    # 2. DECISION_CONFLICT
    gateway._client.eval = AsyncMock(return_value=["DECISION_CONFLICT"])
    with pytest.raises(RecoveryConflictError, match="conflicts with existing decision payload"):
        await gateway.resolve_expired_claim(cmd, "p-operator", now_iso)

    # 3. FENCE_MISMATCH
    gateway._client.eval = AsyncMock(return_value=["FENCE_MISMATCH"])
    with pytest.raises(RecoveryPreconditionFailedError, match="Fencing epoch mismatch"):
        await gateway.resolve_expired_claim(cmd, "p-operator", now_iso)

    # 4. OK
    gateway._client.eval = AsyncMock(return_value=["OK", rec.to_json()])
    resolved_rec, is_idempotent = await gateway.resolve_expired_claim(cmd, "p-operator", now_iso)
    assert resolved_rec.recovery_decision_id == "rec-1"
    assert not is_idempotent
    assert resolved_rec.restored_status == "pending"

    # 5. IDEMPOTENT
    gateway._client.eval = AsyncMock(return_value=["IDEMPOTENT", rec.to_json()])
    replayed_rec, is_idempotent2 = await gateway.resolve_expired_claim(cmd, "p-operator", now_iso)
    assert replayed_rec.recovery_decision_id == "rec-1"
    assert is_idempotent2
