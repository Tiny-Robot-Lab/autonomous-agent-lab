from typing import Any, Protocol, runtime_checkable

from agent_contracts import (
    ActiveClaim,
    AgentCandidate,
    AgentEvent,
    DelegationGrant,
    RootAuthorityRecord,
    RoutingDecision,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskRequested,
    WakeUpHint,
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
    ) -> None: ...


@runtime_checkable
class WakeUpHintDispatcher(Protocol):
    async def dispatch_hint(
        self,
        hint: WakeUpHint,
        target: str,
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
    ) -> None: ...

    async def register_child_task(
        self,
        task: TaskRequested,
        initial_status: str = "pending_dispatch",
    ) -> None:
        """Atomically persist pre-validated child task status, forward lineage, and parent membership.

        Domain validation (parent existence and depth-1 policy) is enforced by the
        platform application use case (RegisterChildTask) prior to invoking this method.
        """
        ...

    async def get_lineage(self, task_id: str) -> TaskLineage | None: ...

    async def get_children(self, parent_task_id: str) -> list[str]: ...

    async def update_status(self, task_id: str, status: str) -> None: ...

    async def get_status(self, task_id: str) -> str | None: ...

    async def save_task_input(self, task_input: TaskInput) -> None: ...

    async def get_task_input(
        self, task_id: str, input_version: int = 1
    ) -> TaskInput | None: ...

    async def save_stage_artifact(self, artifact: StageArtifact) -> None: ...

    async def get_stage_artifact(
        self, producing_task_id: str, artifact_id: str, artifact_version: int = 1
    ) -> StageArtifact | None: ...

    async def save_continuation(self, continuation: TaskContinuation) -> None: ...

    async def get_continuation(
        self, parent_task_id: str, continuation_version: int = 1
    ) -> TaskContinuation | None: ...

    async def save_root_authority(self, authority: RootAuthorityRecord) -> None: ...

    async def get_root_authority(self, root_task_id: str) -> RootAuthorityRecord | None: ...

    async def save_authority_grant(self, grant: DelegationGrant) -> None: ...

    async def get_authority_grant(
        self, root_task_id: str, grant_id: str
    ) -> DelegationGrant | None: ...

    async def save_active_claim(self, claim: ActiveClaim) -> None: ...

    async def get_active_claim(self, task_id: str) -> ActiveClaim | None: ...

    async def get_fencing_epoch(self, task_id: str) -> int: ...

    async def atomic_claim_task(
        self,
        task_id: str,
        claim: ActiveClaim,
        grant: DelegationGrant | None = None,
        now: str = "",
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
    ) -> ActiveClaim: ...

    async def expire_delivery_lease(self, task_id: str, now: str) -> bool: ...


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
    async def get_result(self, task_id: str) -> dict[str, Any] | None: ...
