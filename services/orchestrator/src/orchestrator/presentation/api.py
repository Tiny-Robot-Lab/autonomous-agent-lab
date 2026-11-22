from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from agent_contracts import (
    FORBIDDEN_AUTHORITY_FIELDS,
    ArtifactReference,
    __version__,
)
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from orchestrator.application.ports import (
    AuthoritativeTaskStore,
    CandidateFinder,
    CapabilityMismatchError,
    CapabilityRouter,
    ChildClaimDeadlineExpiredError,
    ClaimDeliveryLeaseExpiredError,
    ClaimNotFoundError,
    DelegationConflictError,
    DelegationReplayAuthorizationError,
    EventPublisher,
    GrantAlreadyConsumedError,
    InvalidClaimProofError,
    ResultReader,
    RoutingRecorder,
    StateReader,
    TaskAlreadyClaimedError,
    TaskDispatcher,
    TaskNotClaimableError,
    TaskQueue,
    TelemetryStorageError,
    WaveBudgetExceededError,
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
    AcknowledgeAssignment,
    ClaimTask,
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
    assignment_proof: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    content: dict[str, Any]
    artifact_version: int = 1
    payload_schema_version: int = 1


class DelegationProposalInput(BaseModel):
    proposal_id: str = Field(min_length=1)
    assignment_id: str = Field(min_length=1)
    child_intent: str = Field(min_length=1, max_length=2000)
    artifact_reference: dict[str, Any]
    target_capability: str = "analysis"
    parent_input_version: int = 1
    control_fields: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


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
            if not cfg.get("active", True):
                raise HTTPException(status_code=403, detail="Principal is inactive")
            raw_caps = cfg.get("authorized_capabilities", ())
            caps = tuple(raw_caps) if isinstance(raw_caps, (list, tuple)) else ()
            return RuntimePrincipal(
                principal_id=principal_id,
                bound_agent_id=cfg.get("bound_agent_id", principal_id),
                authorized_capabilities=caps,
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
    try:
        artifact = await stage_uc.execute(
            principal=principal,
            task_id=task_id,
            assignment_id=value.assignment_id,
            assignment_proof=value.assignment_proof,
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


class NoOpResultReader:
    async def get_result(self, task_id: str) -> dict[str, Any] | None:
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
    if gateway is None and task_store is None and dispatcher is None and task_queue is None:
        gateway = RedisGateway(
            resolved_settings.redis_url,
            delivery_mode=resolved_settings.task_delivery_mode,
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
    results = result_reader or gateway or NoOpResultReader()
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
    if resolved_settings.assignment_proof_pepper.strip():
        claim_task = ClaimTask(
            task_store=resolved_task_store,
            assignment_proof_pepper=resolved_settings.assignment_proof_pepper,
            delivery_lease_seconds=resolved_settings.claim_delivery_lease_seconds,
        )
        acknowledge_assignment = AcknowledgeAssignment(
            task_store=resolved_task_store,
            assignment_proof_pepper=resolved_settings.assignment_proof_pepper,
        )
        stage_artifact = StageArtifactUseCase(
            task_store=resolved_task_store,
            assignment_proof_pepper=resolved_settings.assignment_proof_pepper,
        )
        accept_delegation = AcceptDelegationProposal(
            task_store=resolved_task_store,
            assignment_proof_pepper=resolved_settings.assignment_proof_pepper,
            child_claim_deadline_seconds=resolved_settings.child_claim_deadline_seconds,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if gateway is not None and hasattr(gateway, "aclose"):
            await gateway.aclose()

    app = FastAPI(title="Autonomous Agent Lab", version=__version__, lifespan=lifespan)

    app.state.settings = resolved_settings
    app.state.gateway = gateway
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

    return app


app = create_app()
