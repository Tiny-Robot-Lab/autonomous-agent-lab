from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from agent_contracts import (
    AcceptedDelegationWave,
    ActiveClaim,
    AgentCandidate,
    AgentEvent,
    AssignmentCompletionRecord,
    AssignmentYieldRecord,
    ClaimExpiryRecord,
    CompletionCommand,
    CompletionRecord,
    DelegationGrant,
    DelegationGrantV2,
    DelegationProposalRecord,
    DelegationWaveProposalRecord,
    RecoveryDecisionCommand,
    RecoveryDecisionRecord,
    RequiredCapabilityRecord,
    ResolvedDelegationWave,
    ResumedContinuationContext,
    RootAuthorityRecord,
    RoutingDecision,
    StageArtifact,
    TaskContinuation,
    TaskContinuationV2,
    TaskInput,
    TaskRequested,
    TaskResult,
    TerminalOutcome,
    WakeUpHint,
    WaveCancellationCommand,
    WaveCancellationRecord,
    WaveTimeoutEvaluationCommand,
    WaveYieldRecord,
)

from orchestrator.domain.advisor import AdvisorRecommendation
from orchestrator.domain.assessment import RoutingAssessment
from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.task import TaskLineage


class TaskNotClaimableError(Exception):
    """Raised when a task is not in a claimable lifecycle status."""


class TaskAlreadyClaimedError(Exception):
    """Raised when a task has an active, unexpired claim by another request."""


class ClaimNotFoundError(Exception):
    """Raised when an expected active claim does not exist."""


class InvalidClaimProofError(Exception):
    """Raised when an assignment proof digest verification fails."""


class ClaimDeliveryLeaseExpiredError(Exception):
    """Raised when attempting an operation on an unacknowledged claim whose delivery lease has expired."""


class CapabilityMismatchError(Exception):
    """Raised when an authenticated principal attempts an operation for an unauthorized or mismatched capability."""


class FencingEpochRaceError(TaskAlreadyClaimedError):
    """Raised when a concurrent claim conflict or epoch race is detected."""


class ChildClaimDeadlineExpiredError(TaskNotClaimableError):
    """Raised when an attempt is made to claim a child task after its claim deadline has passed."""


class ClaimDeadlineNotExpiredError(TaskNotClaimableError):
    """Raised when an attempt is made to materialize claim timeout before or at the claim deadline."""


class DelegationConflictError(Exception):
    """Raised when a delegation proposal conflicts with existing authoritative state."""


class WaveBudgetExceededError(Exception):
    """Raised when a delegation attempt violates the maximum allowed waves for the task."""


class GrantAlreadyConsumedError(Exception):
    """Raised when an authority grant has already been consumed."""


class DelegationReplayAuthorizationError(Exception):
    """Raised when retired delegation proposal replay authorization fails."""


class CompletionConflictError(Exception):
    """Raised when a completion conflicts with canonical terminal state.

    Covers a changed semantic payload under an already-accepted completion
    command ID, a different command after canonical terminal state, and a
    pre-existing conflicting canonical result, outcome, or audit record.
    """


class CompletionReplayAuthorizationError(Exception):
    """Raised when retired completion replay authorization fails.

    Always surfaced to callers as one generic authorization failure that
    reveals no result, outcome, command, task-state, or existence information.
    """


class ClaimTimeoutConflictError(Exception):
    """Raised when materializing a claim timeout conflicts with existing terminal state."""


class RecoveryConflictError(Exception):
    """Raised when a recovery decision conflicts with an existing decision or semantic payload."""


class RecoveryPreconditionFailedError(Exception):
    """Raised when recovery preconditions (status, claim, fence, or expiry record) are not met."""


class WaveDeadlineNotReachedError(TaskNotClaimableError):
    """Raised when wave-deadline evaluation is attempted before the frozen wave_deadline."""


class WaveAlreadyResolvedError(DelegationConflictError):
    """Raised when a wave-resolving operation loses to a prior authoritative resolution."""


class WaveCancellationConflictError(DelegationConflictError):
    """Raised when a wave cancellation command conflicts with an existing cancellation decision."""


@runtime_checkable
class TaskQueue(Protocol):
    async def enqueue(
        self,
        task: TaskRequested,
        capability: AgentCapability = AgentCapability.GENERAL,
        target: str | None = None,
    ) -> None: ...


@runtime_checkable
class TaskDispatcher(Protocol):
    async def dispatch(
        self,
        task: TaskRequested,
        target: str,
        capability: str | None = None,
    ) -> None: ...


@runtime_checkable
class WakeUpHintDispatcher(Protocol):
    async def dispatch_hint(
        self,
        hint: WakeUpHint,
        target: str | None = None,
        capability: str | None = None,
    ) -> None: ...


@runtime_checkable
class AuthoritativeTaskStore(Protocol):
    """Authoritative platform storage port for task existence, forward lineage, and parent membership.

    Governed by ADR-0006:
    - Persists durable task existence and forward lineage;
    - Commits child task state, forward lineage, and parent membership as one consistency boundary;
    - Tracks lifecycle transitions: pending_dispatch -> pending / dispatch_failed.
    """

    async def register_root_task(
        self,
        task: TaskRequested,
        initial_status: str = "pending_dispatch",
        required_capability: str = "general",
    ) -> None: ...

    async def register_child_task(
        self,
        task: TaskRequested,
        initial_status: str = "pending_dispatch",
        required_capability: str = "analysis",
    ) -> None:
        """Atomically persist pre-validated child task status, forward lineage, and parent membership.

        Domain validation (parent existence and depth-1 policy) is enforced by the
        platform application use case (RegisterChildTask) prior to invoking this method.
        """
        ...

    async def get_required_capability(
        self, task_id: str
    ) -> RequiredCapabilityRecord | None: ...

    async def get_lineage(self, task_id: str) -> TaskLineage | None: ...

    async def get_children(self, parent_task_id: str) -> list[str]: ...

    async def update_status(self, task_id: str, status: str) -> None: ...

    async def get_status(self, task_id: str, now: str = "") -> str | None: ...

    async def save_task_input(self, task_input: TaskInput) -> None: ...

    async def get_task_input(
        self, task_id: str, input_version: int = 1
    ) -> TaskInput | None: ...

    async def save_stage_artifact(
        self,
        artifact: StageArtifact,
        expected_assignment_id: str,
        expected_principal_id: str,
        expected_fencing_epoch: int,
    ) -> None: ...

    async def get_stage_artifact(
        self, producing_task_id: str, artifact_id: str, artifact_version: int = 1
    ) -> StageArtifact | None: ...

    async def save_continuation(self, continuation: TaskContinuation) -> None: ...

    async def get_continuation(
        self, parent_task_id: str, continuation_version: int = 1
    ) -> TaskContinuation | TaskContinuationV2 | None: ...

    async def get_latest_continuation_version(self, parent_task_id: str) -> int | None: ...

    async def save_root_authority(self, authority: RootAuthorityRecord) -> None: ...

    async def get_root_authority(self, root_task_id: str) -> RootAuthorityRecord | None: ...

    async def save_authority_grant(self, grant: DelegationGrant | DelegationGrantV2) -> None: ...

    async def get_authority_grant(
        self, root_task_id: str, grant_id: str
    ) -> DelegationGrant | DelegationGrantV2 | None: ...

    async def save_active_claim(self, claim: ActiveClaim) -> None: ...

    async def get_active_claim(self, task_id: str) -> ActiveClaim | None: ...

    async def get_fencing_epoch(self, task_id: str) -> int: ...

    async def atomic_claim_task(
        self,
        task_id: str,
        claim: ActiveClaim,
        grant: DelegationGrant | DelegationGrantV2 | None = None,
        now: str = "",
        expected_continuation_version: int = -1,
        is_root: bool = True,
    ) -> tuple[ActiveClaim, bool]:
        """Atomically claim a task.

        Returns (active_claim, is_replay).
        Raises TaskNotClaimableError, TaskAlreadyClaimedError, etc.
        """
        ...

    async def acknowledge_claim(
        self,
        task_id: str,
        assignment_id: str,
        principal_id: str,
        acknowledged_at: str,
        execution_expires_at: str | None = None,
    ) -> ActiveClaim: ...

    async def expire_delivery_lease(self, task_id: str, now: str) -> bool: ...

    async def materialize_claim_timeout(
        self,
        child_task_id: str,
        now: str = "",
    ) -> tuple[TerminalOutcome, bool]:
        """Atomically materialize claim_timed_out terminal outcome for an unclaimed child.

        Returns (outcome, is_replay).
        """
        ...

    async def transition_expired_claim(
        self,
        task_id: str,
        now: str = "",
    ) -> tuple[ClaimExpiryRecord, bool]:
        """Atomically detect an expired acknowledged execution claim and transition task to recovery_required.

        Returns (claim_expiry_record, is_replay).
        """
        ...

    async def resolve_expired_claim(
        self,
        command: RecoveryDecisionCommand,
        operator_principal_id: str,
        now: str = "",
    ) -> tuple[RecoveryDecisionRecord, bool]:
        """Atomically apply an operator recovery decision to clear recovery_required.

        Returns (recovery_decision_record, is_replay).
        """
        ...

    async def get_claim_expiry(
        self,
        task_id: str,
    ) -> ClaimExpiryRecord | None: ...

    async def get_recovery_decision(
        self,
        task_id: str,
        recovery_decision_id: str,
    ) -> RecoveryDecisionRecord | None: ...

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
        """Atomically accept a research -> analysis delegation proposal.

        Returns (accepted_proposal_record, is_replay).
        """
        ...

    async def get_delegation_proposal(
        self,
        parent_task_id: str,
        proposal_id: str,
    ) -> DelegationProposalRecord | None:
        """Retrieve an accepted delegation proposal record by parent_task_id and proposal_id."""
        ...

    async def accept_delegation_wave(
        self,
        parent_task_id: str,
        proposal: DelegationWaveProposalRecord,
        accepted_wave: AcceptedDelegationWave,
        child_task_inputs: Sequence[TaskInput],
        parent_continuation: TaskContinuationV2,
        yield_record: WaveYieldRecord,
        wake_up_hints: Sequence[WakeUpHint],
        expected_fencing_epoch: int,
        now: str,
    ) -> tuple[DelegationWaveProposalRecord, bool]:
        """Atomically accept a bounded multi-child research -> analysis delegation wave.

        Returns (accepted_wave_proposal_record, is_replay).
        """
        ...

    async def get_delegation_wave_proposal(
        self,
        parent_task_id: str,
        proposal_id: str,
    ) -> DelegationWaveProposalRecord | None:
        """Retrieve an accepted wave proposal record by parent_task_id and proposal_id."""
        ...

    async def get_accepted_delegation_wave(
        self,
        parent_task_id: str,
        wave_id: str,
    ) -> AcceptedDelegationWave | None:
        """Retrieve an accepted delegation wave record by parent_task_id and wave_id."""
        ...

    async def get_resolved_delegation_wave(
        self,
        parent_task_id: str,
        wave_id: str,
    ) -> ResolvedDelegationWave | None:
        """Retrieve an immutable resolved delegation wave record by parent_task_id and wave_id."""
        ...

    async def get_child_completion_marker(
        self,
        child_task_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve a child completion projection marker by child_task_id."""
        ...

    async def get_wave_yield(
        self,
        parent_task_id: str,
        assignment_id: str,
    ) -> WaveYieldRecord | None:
        """Retrieve a persisted wave yield record by parent_task_id and assignment_id."""
        ...

    async def get_child_claim_deadline(
        self,
        child_task_id: str,
    ) -> str | None:
        """Retrieve a child task's authoritative claim deadline."""
        ...

    async def evaluate_wave_timeout(
        self,
        command: WaveTimeoutEvaluationCommand,
        now: str = "",
    ) -> tuple[ResolvedDelegationWave, bool]:
        """Atomically evaluate a frozen delegation-wave deadline and resolve TIMED_OUT if reached.

        Returns (resolved_wave, is_replay).
        """
        ...

    async def cancel_delegation_wave(
        self,
        command: WaveCancellationCommand,
        operator_principal_id: str,
        now: str = "",
    ) -> tuple[WaveCancellationRecord, ResolvedDelegationWave, bool]:
        """Atomically cancel an active delegation wave and terminate the collaborative parent.

        Returns (cancellation_record, resolved_wave, is_replay).
        """
        ...

    async def get_wave_cancellation_record(
        self,
        parent_task_id: str,
        cancellation_decision_id: str,
    ) -> WaveCancellationRecord | None:
        """Retrieve a persisted wave cancellation decision record."""
        ...

    async def read_resumed_continuation_v2_context(
        self,
        parent_task_id: str,
        assignment_id: str,
        principal_id: str,
        bound_agent_id: str,
        expected_fencing_epoch: int,
    ) -> ResumedContinuationContext:
        """Read and atomically validate a released v2 continuation context snapshot."""
        ...


@runtime_checkable
class TurnCompletionStore(Protocol):
    """Authoritative platform port for universal fenced turn completion.

    Governed by ADR-0007 (Decision 11) and Issue #222:
    - One single-node Lua script commits canonical TerminalOutcome,
      completion-command replay record, assignment-completion audit, terminal
      lifecycle status, active-claim retirement, and (for delegated children)
      parent projection with a single non-secret Research wake-up hint. All
      expected validation rejections happen in read-only Phase 1 with zero
      mutation; Phase 2 holds only prevalidated writes. Redis serializes
      script execution atomically with respect to other clients; no rollback
      is claimed for unexpected command errors, host failures,
      out-of-memory conditions, or persistence failures;
    - Failed turns persist no TaskResult;
    - Identical retries return the recorded outcome with zero writes.
    """

    async def complete_turn(
        self,
        task_id: str,
        command: CompletionCommand,
        proof_digest: str,
        fencing_epoch: int,
        parent_task_id: str | None,
        root_task_id: str,
        completed_at: str,
    ) -> tuple[TerminalOutcome, TaskResult | None, bool]:
        """Atomically commit a fenced turn completion.

        Returns (outcome, result_or_None, is_replay). Raises
        CompletionConflictError on conflicting terminal state, ClaimNotFoundError
        when the active claim is absent or mismatched, TaskAlreadyClaimedError on
        fencing epoch mismatch, and TaskNotClaimableError on lifecycle or
        lineage violations.
        """
        ...

    async def get_completion_record(
        self,
        task_id: str,
        completion_command_id: str,
    ) -> CompletionRecord | None:
        """Retrieve a completion-command replay record without mutating state."""
        ...

    async def get_terminal_outcome(self, task_id: str) -> TerminalOutcome | None:
        """Retrieve the canonical terminal outcome for a task, if any."""
        ...

    async def get_canonical_result(self, task_id: str) -> TaskResult | None:
        """Retrieve the canonical TaskResult for a completed task, if any."""
        ...

    async def get_assignment_completion(
        self,
        task_id: str,
        assignment_id: str,
    ) -> AssignmentCompletionRecord | None:
        """Retrieve the assignment-completion audit record, if any."""
        ...


class StateReader(Protocol):
    async def agents(self) -> list[dict[str, Any]]: ...
    async def events(self, limit: int = 50) -> list[dict[str, str]]: ...
    async def routing_decision(self, task_id: str) -> dict[str, Any] | None: ...
    async def task_inference_telemetry(self, task_id: str) -> dict[str, Any] | None: ...


class TelemetryStorageError(RuntimeError):
    """Raised when telemetry storage or event stream is unavailable or read fails."""


class EventPublisher(Protocol):
    async def publish(self, event: AgentEvent) -> None: ...


@runtime_checkable
class RoutingRecorder(Protocol):
    async def record_routing_decision(self, decision: RoutingDecision) -> None: ...


class CandidateFinder(Protocol):
    async def execute(
        self,
        capability: AgentCapability | str,
        as_of: Any = None,
    ) -> list[AgentCandidate]: ...


class CapabilityRouter(Protocol):
    def route(self, prompt: str) -> AgentCapability: ...


@runtime_checkable
class AssessingCapabilityRouter(CapabilityRouter, Protocol):
    def assess(self, prompt: str) -> RoutingAssessment: ...


@runtime_checkable
class RoutingAdvisor(Protocol):
    async def advise(
        self,
        prompt: str,
        assessment: RoutingAssessment,
    ) -> AdvisorRecommendation: ...


class ResultReader(Protocol):
    async def get_result(self, task_id: str, now: str = "") -> dict[str, Any] | None: ...
