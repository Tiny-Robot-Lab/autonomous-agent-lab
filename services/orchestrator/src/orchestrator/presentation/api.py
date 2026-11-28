from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from agent_contracts import (
    FORBIDDEN_AUTHORITY_FIELDS,
    ArtifactReference,
    DelegationWaveProposalItem,
    DelegationWaveProposalRequest,
    RecoveryDecisionCommand,
    RequiredSuccessFraction,
    WaveCancellationCommand,
    __version__,
    sanitize_redis_url,
)
from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from orchestrator.application.ports import (
    AuthoritativeTaskStore,
    CandidateFinder,
    CapabilityMismatchError,
    CapabilityRouter,
    ChildClaimDeadlineExpiredError,
    ClaimDeliveryLeaseExpiredError,
    ClaimNotFoundError,
    CompletionConflictError,
    CompletionReplayAuthorizationError,
    DelegationConflictError,
    DelegationReplayAuthorizationError,
    EventPublisher,
    FencingEpochRaceError,
    GrantAlreadyConsumedError,
    InvalidClaimProofError,
    RecoveryConflictError,
    RecoveryPreconditionFailedError,
    ResultReader,
    RoutingRecorder,
    StateReader,
    TaskAlreadyClaimedError,
    TaskDispatcher,
    TaskNotClaimableError,
    TaskQueue,
    TelemetryStorageError,
    TurnCompletionStore,
    WaveAlreadyResolvedError,
    WaveBudgetExceededError,
    WaveCancellationConflictError,
    WaveDeadlineNotReachedError,
)
from orchestrator.application.queries import (
    FindCandidatesByCapability,
    GetAgentStates,
    GetRecentEvents,
    GetRoutingDecision,
    GetTaskChildren,
    GetTaskInferenceTelemetry,
    GetTaskResult,
)
from orchestrator.application.use_cases import (
    AcceptDelegationProposal,
    AcceptDelegationWave,
    AcknowledgeAssignment,
    CancelDelegationWave,
    ClaimTask,
    CompleteTurn,
    EvaluateExpiry,
    EvaluateWaveTimeout,
    GetChildTaskInput,
    GetResumedContinuationContext,
    ResolveExpiredClaim,
    StageArtifactUseCase,
    SubmitTask,
    UnauthorizedPrincipalError,
)
from orchestrator.config import Settings
from orchestrator.domain.capability import (
    AgentCapability,
    CapabilityUnavailableError,
)
from orchestrator.domain.principal import (
    RuntimePrincipal,
    verify_token,
)
from orchestrator.domain.selection import CandidateSelectionPolicy
from orchestrator.domain.task import (
    ConflictingChildTaskRegistrationError,
    ConflictingContinuationError,
    ConflictingStageArtifactError,
    CorruptAuthoritativeRecordError,
    NonContiguousVersionError,
    ParentTaskNotFoundError,
)
from orchestrator.infrastructure.redis_adapter import RedisGateway


class TaskInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    capability: AgentCapability | None = None


async def health() -> dict[str, str]:
    return {"status": "ok"}


async def dashboard() -> str:
    return Path(__file__).with_name("index.html").read_text()


async def create_task(value: TaskInput, request: Request) -> dict[str, str]:
    submit_task: SubmitTask = request.app.state.submit_task
    try:
        task = await submit_task.execute(value.prompt, capability=value.capability)
        return {"task_id": task.task_id, "status": "accepted"}
    except CapabilityUnavailableError as exc:
        detail: dict[str, Any] = {
            "error": "capability_unavailable",
            "capability": str(exc.capability),
            "message": str(exc.message),
        }
        if exc.task_id:
            detail["task_id"] = exc.task_id
        raise HTTPException(
            status_code=503,
            detail=detail,
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def agents(request: Request) -> list[dict[str, Any]]:
    get_agent_states: GetAgentStates = request.app.state.get_agent_states
    return await get_agent_states.execute()


async def events(request: Request, limit: int = 25) -> list[dict[str, str]]:
    get_recent_events: GetRecentEvents = request.app.state.get_recent_events
    return await get_recent_events.execute(limit=limit)


async def task_result(task_id: str, request: Request) -> dict[str, Any]:
    get_task_result: GetTaskResult = request.app.state.get_task_result
    result = await get_task_result.execute(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return result


async def task_routing(task_id: str, request: Request) -> dict[str, Any]:
    get_routing_decision: GetRoutingDecision | None = request.app.state.get_routing_decision
    if get_routing_decision is None:
        raise HTTPException(status_code=503, detail="Routing decision reader not configured")
    decision = await get_routing_decision.execute(task_id)
    if decision is None:
        raise HTTPException(
            status_code=404, detail=f"Routing decision for task '{task_id}' not found"
        )
    return decision


async def task_telemetry(task_id: str, request: Request) -> dict[str, Any]:
    get_task_inference_telemetry: GetTaskInferenceTelemetry | None = (
        request.app.state.get_task_inference_telemetry
    )
    if get_task_inference_telemetry is None:
        raise HTTPException(status_code=503, detail="Task telemetry reader not configured")
    try:
        telemetry = await get_task_inference_telemetry.execute(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TelemetryStorageError as exc:
        raise HTTPException(status_code=503, detail="Task telemetry storage unavailable") from exc
    if telemetry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Inference telemetry for task '{task_id}' not found",
        )
    return telemetry


async def task_children(task_id: str, request: Request) -> dict[str, Any]:
    get_task_children: GetTaskChildren = request.app.state.get_task_children
    clean_id = str(task_id).strip()
    try:
        children = await get_task_children.execute(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if children is None:
        raise HTTPException(status_code=404, detail=f"Task '{clean_id}' not found")
    return {
        "parent_task_id": clean_id,
        "child_task_ids": children,
    }


async def candidates(capability: str, request: Request) -> list[dict[str, Any]]:
    find_candidates: FindCandidatesByCapability = request.app.state.find_candidates
    try:
        candidate_list = await find_candidates.execute(capability)
        return [c.to_dict() for c in candidate_list]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ClaimTaskInput(BaseModel):
    task_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    execution_kind: str = "initial"
    expected_continuation_version: int | None = None


class AcknowledgeAssignmentInput(BaseModel):
    task_id: str = Field(min_length=1)
    assignment_proof: str = Field(min_length=1)


class StageArtifactInput(BaseModel):
    assignment_id: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    content: dict[str, Any]
    artifact_version: int = 1
    payload_schema_version: int = 1

    model_config = {"extra": "forbid"}


class DelegationProposalInput(BaseModel):
    proposal_id: str = Field(min_length=1)
    assignment_id: str = Field(min_length=1)
    child_intent: str = Field(min_length=1, max_length=2000)
    artifact_reference: dict[str, Any]
    target_capability: str = "analysis"
    parent_input_version: int = 1
    control_fields: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class DelegationWaveProposalItemInput(BaseModel):
    item_id: str = Field(min_length=1)
    child_intent: str = Field(min_length=1)
    artifact_reference: dict[str, Any]
    target_capability: str = "analysis"
    control_fields: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class DelegationWaveProposalInput(BaseModel):
    proposal_id: str = Field(min_length=1)
    assignment_id: str = Field(min_length=1)
    items: list[DelegationWaveProposalItemInput] = Field(min_length=1)
    barrier_policy: str = "WAIT_ALL"
    required_success_fraction: dict[str, Any] | None = None
    work_complexity: str = "medium"
    uncertainty_level: str = "low"
    planned_parallelism: int | None = None
    parent_input_version: int = 1
    version: int = 2

    model_config = {"extra": "allow"}


class ResolveExpiredClaimInput(BaseModel):
    recovery_decision_id: str = Field(min_length=1, max_length=128)
    expired_claim_id: str = Field(min_length=1, max_length=128)
    expired_assignment_id: str = Field(min_length=1, max_length=128)
    expired_fencing_epoch: int = Field(ge=1)
    disposition: str = Field(min_length=1, max_length=64)

    model_config = {"extra": "forbid"}

    @field_validator("expired_fencing_epoch", mode="before")
    @classmethod
    def validate_expired_fencing_epoch(cls, v: Any) -> int:
        if type(v) is bool:
            raise TypeError("expired_fencing_epoch must not be a boolean")
        if not isinstance(v, int):
            raise TypeError("expired_fencing_epoch must be an integer")
        if v < 1:
            raise ValueError("expired_fencing_epoch must be >= 1")
        return v


class EvaluateExpiryInput(BaseModel):
    now: str | None = None

    model_config = {"extra": "forbid"}


class EvaluateWaveTimeoutInput(BaseModel):
    now: str | None = None

    model_config = {"extra": "forbid"}


class CancelDelegationWaveInput(BaseModel):
    cancellation_decision_id: str = Field(min_length=1, max_length=256)

    model_config = {"extra": "forbid"}


SAFE_CTX_KEYS = frozenset({
    "expected",
    "limit_value",
    "gt",
    "ge",
    "lt",
    "le",
    "min_length",
    "max_length",
    "max_digits",
    "decimal_places",
    "pattern",
})


def _sanitize_validation_item(item: Any) -> Any:
    if isinstance(item, dict):
        sanitized: dict[str, Any] = {}
        if "type" in item and isinstance(item["type"], str):
            sanitized["type"] = item["type"]
        if "loc" in item and isinstance(item["loc"], (tuple, list)):
            sanitized["loc"] = list(item["loc"])
        if "msg" in item and isinstance(item["msg"], str):
            sanitized["msg"] = item["msg"]
        if "ctx" in item and isinstance(item["ctx"], dict):
            safe_ctx: dict[str, Any] = {}
            for k, v in item["ctx"].items():
                if k in SAFE_CTX_KEYS and isinstance(v, (int, float, bool, str)):
                    safe_ctx[k] = v
                elif k == "error" and isinstance(v, Exception):
                    safe_ctx[k] = v.__class__.__name__
            if safe_ctx:
                sanitized["ctx"] = safe_ctx
        return sanitized
    elif isinstance(item, list):
        return [_sanitize_validation_item(x) for x in item]
    return item


def sanitize_validation_errors(errors: list[Any]) -> list[dict[str, Any]]:
    return [_sanitize_validation_item(err) for err in errors]


def get_current_runtime_principal(request: Request) -> RuntimePrincipal:
    auth_header = request.headers.get("Authorization", "").strip()
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth_header.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")

    settings: Settings = request.app.state.settings
    pepper = settings.runtime_credential_pepper
    principals = settings.runtime_principals

    for principal_id, cfg in principals.items():
        expected_digest = cfg.get("token_digest", "")
        if expected_digest and verify_token(token, pepper, expected_digest):
            if not cfg["active"]:
                raise HTTPException(status_code=403, detail="Principal is inactive")
            return RuntimePrincipal(
                principal_id=principal_id,
                bound_agent_id=cfg["bound_agent_id"],
                authorized_capabilities=tuple(cfg["authorized_capabilities"]),
                active=True,
            )

    raise HTTPException(status_code=401, detail="Invalid runtime credentials")


async def claim_task_endpoint(
    value: ClaimTaskInput,
    request: Request,
    principal: Annotated[RuntimePrincipal, Depends(get_current_runtime_principal)],
) -> dict[str, Any]:
    claim_task_uc: ClaimTask | None = request.app.state.claim_task
    if claim_task_uc is None:
        raise HTTPException(status_code=503, detail="Task claim service is not enabled")
    try:
        res = await claim_task_uc.execute(
            principal=principal,
            task_id=value.task_id,
            request_id=value.request_id,
            capability=value.capability,
            execution_kind=value.execution_kind,
            expected_continuation_version=value.expected_continuation_version,
        )
        data: dict[str, Any] = {"assignment": res.assignment.to_dict()}
        if res.assignment_proof is not None:
            data["assignment_proof"] = res.assignment_proof
        return data
    except UnauthorizedPrincipalError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except CapabilityMismatchError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (TaskAlreadyClaimedError, ClaimDeliveryLeaseExpiredError, ChildClaimDeadlineExpiredError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except TaskNotClaimableError as exc:
        if "not found" in str(exc).lower():
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CorruptAuthoritativeRecordError as exc:
        raise HTTPException(status_code=500, detail="Authoritative record corruption detected") from exc


async def acknowledge_assignment_endpoint(
    assignment_id: str,
    value: AcknowledgeAssignmentInput,
    request: Request,
    principal: Annotated[RuntimePrincipal, Depends(get_current_runtime_principal)],
) -> dict[str, Any]:
    ack_uc: AcknowledgeAssignment | None = request.app.state.acknowledge_assignment
    if ack_uc is None:
        raise HTTPException(status_code=503, detail="Task claim service is not enabled")
    try:
        claim = await ack_uc.execute(
            principal=principal,
            task_id=value.task_id,
            assignment_id=assignment_id,
            assignment_proof=value.assignment_proof,
        )
        return {
            "assignment_id": claim.assignment_id,
            "status": claim.state,
            "acknowledged_at": claim.acknowledged_at,
        }
    except UnauthorizedPrincipalError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except ClaimNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (InvalidClaimProofError, CapabilityMismatchError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (ClaimDeliveryLeaseExpiredError, TaskAlreadyClaimedError, TaskNotClaimableError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CorruptAuthoritativeRecordError as exc:
        raise HTTPException(status_code=500, detail="Authoritative record corruption detected") from exc


async def stage_artifact_endpoint(
    task_id: str,
    value: StageArtifactInput,
    request: Request,
    principal: Annotated[RuntimePrincipal, Depends(get_current_runtime_principal)],
) -> dict[str, Any]:
    stage_uc: StageArtifactUseCase | None = request.app.state.stage_artifact
    if stage_uc is None:
        raise HTTPException(status_code=503, detail="Artifact staging service is not enabled")

    header_proof = request.headers.get("X-Assignment-Proof", "").strip()
    if not header_proof:
        raise HTTPException(
            status_code=401, detail="Missing assignment proof in X-Assignment-Proof header"
        )

    try:
        artifact = await stage_uc.execute(
            principal=principal,
            task_id=task_id,
            assignment_id=value.assignment_id,
            assignment_proof=header_proof,
            artifact_id=value.artifact_id,
            artifact_type=value.artifact_type,
            content=value.content,
            artifact_version=value.artifact_version,
            payload_schema_version=value.payload_schema_version,
        )
        return {"artifact": artifact.to_dict()}
    except UnauthorizedPrincipalError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except ClaimNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (InvalidClaimProofError, CapabilityMismatchError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (TaskNotClaimableError, TaskAlreadyClaimedError, ConflictingStageArtifactError, NonContiguousVersionError, ClaimDeliveryLeaseExpiredError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CorruptAuthoritativeRecordError as exc:
        raise HTTPException(status_code=500, detail="Authoritative record corruption detected") from exc


async def complete_task_endpoint(
    task_id: str,
    request: Request,
    principal: Annotated[RuntimePrincipal, Depends(get_current_runtime_principal)],
) -> dict[str, Any]:
    complete_turn_uc: CompleteTurn | None = request.app.state.complete_turn
    if complete_turn_uc is None:
        raise HTTPException(status_code=503, detail="Turn completion service is not enabled")

    header_proof = request.headers.get("X-Assignment-Proof", "").strip()
    if not header_proof:
        raise HTTPException(status_code=401, detail="Missing assignment proof in X-Assignment-Proof header")

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    try:
        response = await complete_turn_uc.execute(
            principal=principal,
            task_id=task_id,
            command_payload=payload,
            assignment_proof=header_proof,
        )
        data: dict[str, Any] = {
            "outcome": response.outcome.to_dict(),
            "replayed": response.replayed,
        }
        if response.result is not None:
            data["result"] = response.result.to_dict()
        return data
    except CompletionReplayAuthorizationError as exc:
        raise HTTPException(status_code=403, detail="Completion replay authorization failed") from exc
    except UnauthorizedPrincipalError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except (InvalidClaimProofError, CapabilityMismatchError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ClaimNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TaskAlreadyClaimedError, TaskNotClaimableError, CompletionConflictError) as exc:
        if isinstance(exc, TaskNotClaimableError) and "not found" in str(exc).lower():
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CorruptAuthoritativeRecordError as exc:
        raise HTTPException(status_code=500, detail="Authoritative record corruption detected") from exc


async def accept_delegation_endpoint(
    task_id: str,
    value: DelegationProposalInput,
    request: Request,
    principal: Annotated[RuntimePrincipal, Depends(get_current_runtime_principal)],
) -> dict[str, Any]:
    delegation_uc: AcceptDelegationProposal | None = request.app.state.accept_delegation
    if delegation_uc is None:
        raise HTTPException(status_code=503, detail="Delegation service is not enabled")

    header_proof = request.headers.get("X-Assignment-Proof", "").strip()
    if not header_proof:
        raise HTTPException(status_code=401, detail="Missing assignment proof in X-Assignment-Proof header")

    try:
        extra = getattr(value, "__pydantic_extra__", None)
        if extra:
            if "assignment_proof" in extra:
                raise ValueError("assignment_proof must not be provided in request body; use X-Assignment-Proof header")
            forbidden = set(extra.keys()) & FORBIDDEN_AUTHORITY_FIELDS
            if forbidden:
                raise ValueError(f"Forbidden authority fields supplied: {sorted(forbidden)}")
            raise ValueError(f"Unexpected fields supplied: {sorted(extra.keys())}")

        raw_ref = value.artifact_reference
        if not isinstance(raw_ref, dict):
            raise TypeError("artifact_reference must be an object")
        art_ref = ArtifactReference.from_dict(raw_ref)

        resp = await delegation_uc.execute(
            principal=principal,
            parent_task_id=task_id,
            proposal_id=value.proposal_id,
            assignment_id=value.assignment_id,
            assignment_proof=header_proof,
            child_intent=value.child_intent,
            artifact_reference=art_ref,
            target_capability=value.target_capability,
            parent_input_version=value.parent_input_version,
            control_fields=value.control_fields,
        )
        return resp.to_dict()
    except DelegationReplayAuthorizationError as exc:
        raise HTTPException(status_code=403, detail="Delegation replay authorization failed") from exc
    except UnauthorizedPrincipalError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except (InvalidClaimProofError, CapabilityMismatchError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (ClaimNotFoundError, ParentTaskNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        TaskAlreadyClaimedError,
        TaskNotClaimableError,
        ClaimDeliveryLeaseExpiredError,
        DelegationConflictError,
        WaveBudgetExceededError,
        GrantAlreadyConsumedError,
        ConflictingContinuationError,
        ConflictingChildTaskRegistrationError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CorruptAuthoritativeRecordError as exc:
        raise HTTPException(status_code=500, detail="Authoritative record corruption detected") from exc


async def accept_delegation_wave_endpoint(
    task_id: str,
    value: DelegationWaveProposalInput,
    request: Request,
    principal: Annotated[RuntimePrincipal, Depends(get_current_runtime_principal)],
) -> dict[str, Any]:
    wave_uc: AcceptDelegationWave | None = getattr(
        request.app.state, "accept_delegation_wave", None
    )
    if wave_uc is None:
        raise HTTPException(
            status_code=503, detail="Delegation wave service is not enabled"
        )

    header_proof = request.headers.get("X-Assignment-Proof", "").strip()
    if not header_proof:
        raise HTTPException(
            status_code=401, detail="Missing assignment proof in X-Assignment-Proof header"
        )

    try:
        extra = getattr(value, "__pydantic_extra__", None)
        if extra:
            if "assignment_proof" in extra:
                raise ValueError(
                    "assignment_proof must not be provided in request body; use X-Assignment-Proof header"
                )
            forbidden = set(extra.keys()) & FORBIDDEN_AUTHORITY_FIELDS
            if forbidden:
                raise ValueError(f"Forbidden authority fields supplied: {sorted(forbidden)}")
            raise ValueError(f"Unexpected fields supplied: {sorted(extra.keys())}")

        parsed_items: list[DelegationWaveProposalItem] = []
        for item in value.items:
            item_extra = getattr(item, "__pydantic_extra__", None)
            if item_extra:
                if "assignment_proof" in item_extra:
                    raise ValueError(
                        "assignment_proof must not be provided in request body; use X-Assignment-Proof header"
                    )
                forbidden_item = set(item_extra.keys()) & FORBIDDEN_AUTHORITY_FIELDS
                if forbidden_item:
                    raise ValueError(
                        f"Forbidden authority fields supplied: {sorted(forbidden_item)}"
                    )
                raise ValueError(f"Unexpected fields supplied: {sorted(item_extra.keys())}")

            raw_ref = item.artifact_reference
            if not isinstance(raw_ref, dict):
                raise TypeError("artifact_reference must be an object")
            art_ref = ArtifactReference.from_dict(raw_ref)
            parsed_items.append(
                DelegationWaveProposalItem(
                    item_id=item.item_id,
                    child_intent=item.child_intent,
                    artifact_reference=art_ref,
                    target_capability=item.target_capability,
                    control_fields=item.control_fields,
                )
            )

        parsed_frac: RequiredSuccessFraction | None = None
        if value.required_success_fraction is not None:
            parsed_frac = RequiredSuccessFraction.from_dict(value.required_success_fraction)

        req = DelegationWaveProposalRequest(
            proposal_id=value.proposal_id,
            items=tuple(parsed_items),
            barrier_policy=value.barrier_policy,
            required_success_fraction=parsed_frac,
            work_complexity=value.work_complexity,
            uncertainty_level=value.uncertainty_level,
            planned_parallelism=value.planned_parallelism,
            parent_input_version=value.parent_input_version,
            version=value.version,
        )

        resp = await wave_uc.execute(
            principal=principal,
            parent_task_id=task_id,
            assignment_id=value.assignment_id,
            assignment_proof=header_proof,
            request=req,
        )
        return resp.to_dict()
    except DelegationReplayAuthorizationError as exc:
        raise HTTPException(status_code=403, detail="Delegation replay authorization failed") from exc
    except UnauthorizedPrincipalError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except (InvalidClaimProofError, CapabilityMismatchError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (ClaimNotFoundError, ParentTaskNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        TaskAlreadyClaimedError,
        TaskNotClaimableError,
        ClaimDeliveryLeaseExpiredError,
        DelegationConflictError,
        WaveBudgetExceededError,
        GrantAlreadyConsumedError,
        ConflictingContinuationError,
        ConflictingChildTaskRegistrationError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CorruptAuthoritativeRecordError as exc:
        raise HTTPException(status_code=500, detail="Authoritative record corruption detected") from exc


async def evaluate_expiry_endpoint(
    task_id: str,
    request: Request,
    principal: Annotated[RuntimePrincipal, Depends(get_current_runtime_principal)],
    value: Annotated[EvaluateExpiryInput | None, Body()] = None,
) -> dict[str, Any]:
    eval_uc: EvaluateExpiry | None = getattr(request.app.state, "evaluate_expiry", None)
    if eval_uc is None:
        raise HTTPException(status_code=503, detail="Expiry evaluation service is not enabled")
    now_str = value.now if (value and value.now) else ""
    try:
        res = await eval_uc.execute(principal=principal, task_id=task_id, now=now_str)
        data: dict[str, Any] = {
            "task_id": res.task_id,
            "action": res.action,
            "status": res.status,
        }
        if res.outcome is not None:
            data["outcome"] = res.outcome.to_dict()
        if res.expiry_record is not None:
            data["expiry_record"] = res.expiry_record.to_dict()
        return data
    except UnauthorizedPrincipalError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except CapabilityMismatchError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ParentTaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CorruptAuthoritativeRecordError as exc:
        raise HTTPException(status_code=500, detail="Authoritative record corruption detected") from exc


async def resolve_expired_claim_endpoint(
    task_id: str,
    value: ResolveExpiredClaimInput,
    request: Request,
    principal: Annotated[RuntimePrincipal, Depends(get_current_runtime_principal)],
) -> dict[str, Any]:
    res_uc: ResolveExpiredClaim | None = getattr(request.app.state, "resolve_expired_claim", None)
    if res_uc is None:
        raise HTTPException(status_code=503, detail="Recovery resolution service is not enabled")
    try:
        command = RecoveryDecisionCommand(
            task_id=task_id,
            recovery_decision_id=value.recovery_decision_id,
            expired_claim_id=value.expired_claim_id,
            expired_assignment_id=value.expired_assignment_id,
            expired_fencing_epoch=value.expired_fencing_epoch,
            disposition=value.disposition,
        )
        record, restored_status = await res_uc.execute(principal=principal, command=command)
        return {
            "record": record.to_dict(),
            "restored_status": restored_status,
        }
    except UnauthorizedPrincipalError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except CapabilityMismatchError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ParentTaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RecoveryConflictError, FencingEpochRaceError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (RecoveryPreconditionFailedError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CorruptAuthoritativeRecordError as exc:
        raise HTTPException(status_code=500, detail="Authoritative record corruption detected") from exc


async def evaluate_wave_timeout_endpoint(
    task_id: str,
    wave_id: str,
    request: Request,
    principal: Annotated[RuntimePrincipal, Depends(get_current_runtime_principal)],
    value: Annotated[EvaluateWaveTimeoutInput | None, Body()] = None,
) -> dict[str, Any]:
    eval_uc: EvaluateWaveTimeout | None = getattr(
        request.app.state, "evaluate_wave_timeout", None
    )
    if eval_uc is None:
        raise HTTPException(
            status_code=503, detail="Wave timeout evaluation service is not enabled"
        )
    now_str = value.now if (value and value.now) else ""
    try:
        resolved_wave, replayed = await eval_uc.execute(
            principal=principal,
            parent_task_id=task_id,
            wave_id=wave_id,
            now=now_str,
        )
        return {
            "resolved_wave": resolved_wave.to_dict(),
            "replayed": replayed,
        }
    except UnauthorizedPrincipalError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except CapabilityMismatchError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ParentTaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        WaveDeadlineNotReachedError,
        WaveAlreadyResolvedError,
        DelegationConflictError,
        TaskNotClaimableError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CorruptAuthoritativeRecordError as exc:
        raise HTTPException(
            status_code=500, detail="Authoritative record corruption detected"
        ) from exc


async def cancel_delegation_wave_endpoint(
    task_id: str,
    wave_id: str,
    value: CancelDelegationWaveInput,
    request: Request,
    principal: Annotated[RuntimePrincipal, Depends(get_current_runtime_principal)],
) -> dict[str, Any]:
    cancel_uc: CancelDelegationWave | None = getattr(
        request.app.state, "cancel_delegation_wave", None
    )
    if cancel_uc is None:
        raise HTTPException(
            status_code=503, detail="Wave cancellation service is not enabled"
        )
    try:
        command = WaveCancellationCommand(
            cancellation_decision_id=value.cancellation_decision_id,
            parent_task_id=task_id,
            wave_id=wave_id,
        )
        cancellation_rec, resolved_wave, replayed = await cancel_uc.execute(
            principal=principal,
            command=command,
        )
        return {
            "cancellation": cancellation_rec.to_dict(),
            "resolved_wave": resolved_wave.to_dict(),
            "replayed": replayed,
        }
    except UnauthorizedPrincipalError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except CapabilityMismatchError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ParentTaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        WaveCancellationConflictError,
        WaveAlreadyResolvedError,
        DelegationConflictError,
        TaskNotClaimableError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CorruptAuthoritativeRecordError as exc:
        raise HTTPException(
            status_code=500, detail="Authoritative record corruption detected"
        ) from exc


async def continuation_context_endpoint(
    task_id: str,
    assignment_id: str,
    request: Request,
    principal: Annotated[RuntimePrincipal, Depends(get_current_runtime_principal)],
) -> dict[str, Any]:
    get_ctx_uc: GetResumedContinuationContext | None = getattr(
        request.app.state, "get_resumed_continuation_context", None
    )
    if get_ctx_uc is None:
        raise HTTPException(
            status_code=503, detail="Continuation context service is not enabled"
        )

    header_proof = request.headers.get("X-Assignment-Proof", "").strip()
    if not header_proof:
        raise HTTPException(
            status_code=401, detail="Missing assignment proof in X-Assignment-Proof header"
        )

    try:
        context = await get_ctx_uc.execute(
            principal=principal,
            task_id=task_id,
            assignment_id=assignment_id,
            assignment_proof=header_proof,
        )
        return context.to_dict()
    except UnauthorizedPrincipalError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except (InvalidClaimProofError, CapabilityMismatchError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ClaimNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TaskAlreadyClaimedError, TaskNotClaimableError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CorruptAuthoritativeRecordError as exc:
        raise HTTPException(
            status_code=500, detail="Authoritative record corruption detected"
        ) from exc


async def child_task_input_endpoint(
    task_id: str,
    assignment_id: str,
    request: Request,
    principal: Annotated[RuntimePrincipal, Depends(get_current_runtime_principal)],
) -> dict[str, Any]:
    get_input_uc: GetChildTaskInput | None = getattr(
        request.app.state, "get_child_task_input", None
    )
    if get_input_uc is None:
        raise HTTPException(
            status_code=503, detail="Child task input service is not enabled"
        )

    header_proof = request.headers.get("X-Assignment-Proof", "").strip()
    if not header_proof:
        raise HTTPException(
            status_code=401, detail="Missing assignment proof in X-Assignment-Proof header"
        )

    try:
        material = await get_input_uc.execute(
            principal=principal,
            task_id=task_id,
            assignment_id=assignment_id,
            assignment_proof=header_proof,
        )
        return material.to_dict()
    except UnauthorizedPrincipalError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except CapabilityMismatchError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ClaimNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TaskAlreadyClaimedError, TaskNotClaimableError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidClaimProofError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CorruptAuthoritativeRecordError as exc:
        raise HTTPException(
            status_code=500, detail="Authoritative record corruption detected"
        ) from exc


class NoOpResultReader:
    async def get_result(self, task_id: str, now: str = "") -> dict[str, Any] | None:
        return None


def create_app(
    settings: Settings | None = None,
    gateway: RedisGateway | None = None,
    task_queue: TaskQueue | None = None,
    dispatcher: TaskDispatcher | None = None,
    state_reader: StateReader | None = None,
    result_reader: ResultReader | None = None,
    router: CapabilityRouter | None = None,
    candidate_finder: CandidateFinder | None = None,
    event_publisher: EventPublisher | None = None,
    enforce_availability: bool | None = None,
    selection_policy: CandidateSelectionPolicy | None = None,
    routing_recorder: RoutingRecorder | None = None,
    task_store: AuthoritativeTaskStore | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()

    # 1. Gateway resolution
    # In default production composition (or when neither task_store, dispatcher, nor task_queue
    # is provided), create exactly one RedisGateway used coherently across all ports.
    # When an explicit task_store, dispatcher, or task_queue identifies the backend,
    # do not auto-create an extra RedisGateway.
    # Fail closed: auto-construction requires a validated authenticated
    # RedisConnectionConfig. Never auto-create an unauthenticated gateway from
    # Settings.redis_url. Tests must inject an explicit backend or gateway.
    if gateway is None and task_store is None and dispatcher is None and task_queue is None:
        connection_config = resolved_settings.redis_connection
        if connection_config is None or not connection_config.is_authenticated:
            if connection_config is not None:
                endpoint = connection_config.sanitized_url
            else:
                endpoint = sanitize_redis_url(resolved_settings.redis_url)
            raise ValueError(
                "Invalid Redis configuration: create_app requires a validated "
                "authenticated RedisConnectionConfig to auto-construct RedisGateway "
                f"(endpoint {endpoint}); configure REDIS_USERNAME/REDIS_PASSWORD_FILE "
                "via Settings.from_env, or inject an explicit gateway/task_store backend"
            )
        gateway = RedisGateway(
            resolved_settings.redis_url,
            delivery_mode=resolved_settings.task_delivery_mode,
            connection_config=connection_config,
        )

    # 2. AuthoritativeTaskStore linear resolution
    if task_store is not None:
        resolved_task_store: AuthoritativeTaskStore | None = task_store
    elif isinstance(gateway, AuthoritativeTaskStore):
        resolved_task_store = gateway
    elif isinstance(task_queue, AuthoritativeTaskStore):
        resolved_task_store = task_queue
    elif isinstance(dispatcher, AuthoritativeTaskStore):
        resolved_task_store = dispatcher
    else:
        resolved_task_store = None

    if resolved_task_store is None:
        raise TypeError(
            "create_app requires an AuthoritativeTaskStore (e.g., via task_store, gateway, or an adapter implementing AuthoritativeTaskStore)"
        )

    # 3. Dispatch / Queue linear resolution
    dispatch_client = dispatcher or (gateway if isinstance(gateway, TaskDispatcher) else None)
    queue = task_queue or (gateway if isinstance(gateway, TaskQueue) else None)

    if dispatch_client is None and queue is None:
        raise TypeError(
            "create_app requires a dispatch path (e.g., via gateway, dispatcher, or task_queue)"
        )

    # 4. State reader, result reader, events, and routing recorder resolution
    reader = state_reader or gateway
    results = (
        result_reader
        or (task_store if hasattr(task_store, "get_result") else None)
        or gateway
        or NoOpResultReader()
    )
    events_out = event_publisher or gateway
    routing_recorder_client = routing_recorder or (
        gateway if isinstance(gateway, RoutingRecorder) else None
    )

    should_enforce = enforce_availability if enforce_availability is not None else (gateway is not None)
    resolved_finder = candidate_finder
    if resolved_finder is None and should_enforce and reader is not None:
        resolved_finder = FindCandidatesByCapability(reader)

    find_candidates_query = resolved_finder or (FindCandidatesByCapability(reader) if reader else None)

    submit_task = SubmitTask(
        task_store=resolved_task_store,
        queue=queue,
        dispatcher=dispatch_client,
        router=router,
        candidate_finder=resolved_finder,
        events=events_out,
        selection_policy=selection_policy,
        routing_recorder=routing_recorder_client,
    )
    get_agent_states = GetAgentStates(reader) if reader else None
    get_recent_events = GetRecentEvents(reader) if reader else None
    get_task_result = GetTaskResult(results)
    get_routing_decision = GetRoutingDecision(reader) if reader else None
    get_task_inference_telemetry = (
        GetTaskInferenceTelemetry(reader) if reader else None
    )
    get_task_children = GetTaskChildren(resolved_task_store)
    claim_task: ClaimTask | None = None
    acknowledge_assignment: AcknowledgeAssignment | None = None
    stage_artifact: StageArtifactUseCase | None = None
    accept_delegation: AcceptDelegationProposal | None = None
    accept_delegation_wave: AcceptDelegationWave | None = None
    complete_turn: CompleteTurn | None = None
    get_resumed_continuation_context: GetResumedContinuationContext | None = None
    get_child_task_input: GetChildTaskInput | None = None
    evaluate_expiry: EvaluateExpiry | None = None
    resolve_expired_claim: ResolveExpiredClaim | None = None
    evaluate_wave_timeout: EvaluateWaveTimeout | None = None
    cancel_delegation_wave: CancelDelegationWave | None = None

    assignment_lifecycle_configured = bool(
        resolved_settings.assignment_proof_pepper.strip()
    )
    executable_delegation_enabled = (
        resolved_settings.task_delivery_mode == "authenticated_claim"
        and assignment_lifecycle_configured
    )

    if resolved_settings.task_delivery_mode == "authenticated_claim":
        if not resolved_settings.runtime_credential_pepper.strip():
            raise ValueError(
                "runtime_credential_pepper must be configured when task_delivery_mode is 'authenticated_claim'"
            )
        if not resolved_settings.assignment_proof_pepper.strip():
            raise ValueError(
                "assignment_proof_pepper must be configured when task_delivery_mode is 'authenticated_claim'"
            )
        if not resolved_settings.runtime_principals:
            raise ValueError(
                "runtime_principals must be configured when task_delivery_mode is 'authenticated_claim'"
            )

    if assignment_lifecycle_configured:
        claim_task = ClaimTask(
            task_store=resolved_task_store,
            assignment_proof_pepper=resolved_settings.assignment_proof_pepper,
            delivery_lease_seconds=resolved_settings.claim_delivery_lease_seconds,
            allow_wave_delegation_grant=executable_delegation_enabled,
        )
        acknowledge_assignment = AcknowledgeAssignment(
            task_store=resolved_task_store,
            assignment_proof_pepper=resolved_settings.assignment_proof_pepper,
            execution_claim_timeout_seconds=resolved_settings.execution_claim_timeout_seconds,
        )
        completion_store: TurnCompletionStore | None = (
            resolved_task_store
            if isinstance(resolved_task_store, TurnCompletionStore)
            else None
        )
        if completion_store is not None:
            complete_turn = CompleteTurn(
                task_store=resolved_task_store,
                completion_store=completion_store,
                assignment_proof_pepper=resolved_settings.assignment_proof_pepper,
            )
        evaluate_expiry = EvaluateExpiry(task_store=resolved_task_store)
        resolve_expired_claim = ResolveExpiredClaim(task_store=resolved_task_store)
        evaluate_wave_timeout = EvaluateWaveTimeout(task_store=resolved_task_store)
        cancel_delegation_wave = CancelDelegationWave(task_store=resolved_task_store)

    if executable_delegation_enabled:
        stage_artifact = StageArtifactUseCase(
            task_store=resolved_task_store,
            assignment_proof_pepper=resolved_settings.assignment_proof_pepper,
        )
        accept_delegation = AcceptDelegationProposal(
            task_store=resolved_task_store,
            assignment_proof_pepper=resolved_settings.assignment_proof_pepper,
            child_claim_deadline_seconds=resolved_settings.child_claim_deadline_seconds,
        )
        accept_delegation_wave = AcceptDelegationWave(
            task_store=resolved_task_store,
            assignment_proof_pepper=resolved_settings.assignment_proof_pepper,
            child_claim_deadline_seconds=resolved_settings.child_claim_deadline_seconds,
        )
        get_resumed_continuation_context = GetResumedContinuationContext(
            task_store=resolved_task_store,
            assignment_proof_pepper=resolved_settings.assignment_proof_pepper,
        )
        get_child_task_input = GetChildTaskInput(
            task_store=resolved_task_store,
            assignment_proof_pepper=resolved_settings.assignment_proof_pepper,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if gateway is not None and hasattr(gateway, "aclose"):
            await gateway.aclose()

    app = FastAPI(title="Autonomous Agent Lab", version=__version__, lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        sanitized = sanitize_validation_errors(exc.errors())
        return JSONResponse(status_code=422, content={"detail": sanitized})

    app.state.settings = resolved_settings
    app.state.gateway = gateway
    app.state.assignment_lifecycle_configured = assignment_lifecycle_configured
    app.state.executable_delegation_enabled = executable_delegation_enabled
    app.state.submit_task = submit_task
    app.state.get_agent_states = get_agent_states
    app.state.get_recent_events = get_recent_events
    app.state.get_task_result = get_task_result
    app.state.get_routing_decision = get_routing_decision
    app.state.get_task_inference_telemetry = get_task_inference_telemetry
    app.state.get_task_children = get_task_children
    app.state.claim_task = claim_task
    app.state.acknowledge_assignment = acknowledge_assignment
    app.state.stage_artifact = stage_artifact
    app.state.accept_delegation = accept_delegation
    app.state.accept_delegation_wave = accept_delegation_wave
    app.state.complete_turn = complete_turn
    app.state.get_resumed_continuation_context = get_resumed_continuation_context
    app.state.get_child_task_input = get_child_task_input
    app.state.evaluate_expiry = evaluate_expiry
    app.state.resolve_expired_claim = resolve_expired_claim
    app.state.evaluate_wave_timeout = evaluate_wave_timeout
    app.state.cancel_delegation_wave = cancel_delegation_wave
    app.state.find_candidates = find_candidates_query

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        favicon_file = static_dir / "favicon.ico"
        if favicon_file.exists():

            async def favicon() -> FileResponse:
                return FileResponse(favicon_file)

            app.add_api_route(
                "/favicon.ico", favicon, methods=["GET"], include_in_schema=False
            )

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/", dashboard, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route("/api/tasks", create_task, methods=["POST"], status_code=202)
    app.add_api_route("/api/agents", agents, methods=["GET"])
    app.add_api_route("/api/events", events, methods=["GET"])
    app.add_api_route("/api/tasks/{task_id}/result", task_result, methods=["GET"])
    app.add_api_route("/api/tasks/{task_id}/routing", task_routing, methods=["GET"])
    app.add_api_route("/api/tasks/{task_id}/telemetry", task_telemetry, methods=["GET"])
    app.add_api_route("/api/tasks/{task_id}/children", task_children, methods=["GET"])
    app.add_api_route("/api/candidates", candidates, methods=["GET"])
    app.add_api_route("/api/tasks/claim", claim_task_endpoint, methods=["POST"])
    app.add_api_route(
        "/api/assignments/{assignment_id}/acknowledge",
        acknowledge_assignment_endpoint,
        methods=["POST"],
    )
    app.add_api_route(
        "/api/tasks/{task_id}/artifacts",
        stage_artifact_endpoint,
        methods=["POST"],
    )
    app.add_api_route(
        "/api/tasks/{task_id}/delegations",
        accept_delegation_endpoint,
        methods=["POST"],
    )
    app.add_api_route(
        "/api/tasks/{task_id}/delegation-waves",
        accept_delegation_wave_endpoint,
        methods=["POST"],
    )
    app.add_api_route(
        "/api/tasks/{task_id}/completion",
        complete_task_endpoint,
        methods=["POST"],
    )
    app.add_api_route(
        "/api/tasks/{task_id}/continuation-context",
        continuation_context_endpoint,
        methods=["GET"],
    )
    app.add_api_route(
        "/api/tasks/{task_id}/input",
        child_task_input_endpoint,
        methods=["GET"],
    )
    app.add_api_route(
        "/api/tasks/{task_id}/evaluate-expiry",
        evaluate_expiry_endpoint,
        methods=["POST"],
    )
    app.add_api_route(
        "/api/tasks/{task_id}/resolve-expired-claim",
        resolve_expired_claim_endpoint,
        methods=["POST"],
    )
    app.add_api_route(
        "/api/tasks/{task_id}/waves/{wave_id}/evaluate-timeout",
        evaluate_wave_timeout_endpoint,
        methods=["POST"],
    )
    app.add_api_route(
        "/api/tasks/{task_id}/waves/{wave_id}/cancel",
        cancel_delegation_wave_endpoint,
        methods=["POST"],
    )

    return app


# NOTE: The ASGI application is bound lazily (PEP 562 module __getattr__) rather
# than at module level so that importing this module has no Redis-configuration
# side effect. create_app() resolves Settings.from_env(), which fails closed
# without validated REDIS_USERNAME/REDIS_PASSWORD_FILE credentials or an
# explicit REDIS_AUTH_MODE='unauthenticated_local_test' loopback opt-in; a
# module-level `app = create_app()` would therefore raise during test
# collection (and whenever the live DB-15 REDIS_URL fixture is set without
# credentials). Uvicorn still resolves the "orchestrator.presentation.api:app"
# attribute path through this hook on first access, and the resolved app is
# cached for repeated access.
_default_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    if name == "app":
        global _default_app
        if _default_app is None:
            _default_app = create_app()
        return _default_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
