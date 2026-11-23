from datetime import UTC, datetime
from typing import Any

import pytest
from agent_contracts import (
    ActiveClaim,
    DelegationGrant,
    RequiredCapabilityRecord,
    RootAuthorityRecord,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskRequested,
    WakeUpHint,
    normalize_canonical_utc_timestamp,
)
from orchestrator.application.ports import (
    AuthoritativeTaskStore,
    ClaimDeliveryLeaseExpiredError,
    ClaimNotFoundError,
    FencingEpochRaceError,
    TaskAlreadyClaimedError,
    TaskNotClaimableError,
)
from orchestrator.application.queries import (
    GetAgentStates,
    GetRecentEvents,
    GetTaskResult,
)
from orchestrator.application.use_cases import SubmitTask
from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.task import ConflictingAuthorityGrantError, TaskLineage


class FakeTaskQueue(AuthoritativeTaskStore):
    def __init__(self, delivery_mode: str = "legacy_push") -> None:
        self._delivery_mode = delivery_mode
        self.enqueued: list[tuple[TaskRequested, AgentCapability]] = []
        self.dispatches: list[tuple[TaskRequested, str]] = []
        self.statuses: dict[str, str] = {}
        self.lineages: dict[str, TaskLineage] = {}
        self.children: dict[str, set[str]] = {}
        self.inputs: dict[tuple[str, int], TaskInput] = {}
        self.artifacts: dict[tuple[str, str, int], StageArtifact] = {}
        self.continuations: dict[tuple[str, int], TaskContinuation] = {}
        self.authorities: dict[str, RootAuthorityRecord] = {}
        self.authority_grants: dict[tuple[str, str], DelegationGrant] = {}
        self.required_capabilities: dict[str, RequiredCapabilityRecord] = {}
        self.claims: dict[str, ActiveClaim] = {}
        self.fences: dict[str, int] = {}
        self.hints: list[tuple[WakeUpHint, str]] = []
        self.fail_get_lineage: bool = False
        self.fail_get_children: bool = False

    async def register_root_task(
        self,
        task: TaskRequested,
        initial_status: str = "pending_dispatch",
        required_capability: str = "general",
    ) -> None:
        required_cap_record = RequiredCapabilityRecord(
            task_id=task.task_id,
            required_capability=required_capability,
            created_at=task.created_at,
            version=1,
        )
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

    async def register_child_task(
        self,
        task: TaskRequested,
        initial_status: str = "pending_dispatch",
        required_capability: str = "analysis",
    ) -> None:
        cap_rec = RequiredCapabilityRecord(
            task_id=task.task_id,
            required_capability=required_capability,
            created_at=task.created_at,
            version=1,
        )
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
        if task.parent_task_id:
            self.children.setdefault(task.parent_task_id, set()).add(task.task_id)

    async def get_required_capability(
        self, task_id: str
    ) -> RequiredCapabilityRecord | None:
        return self.required_capabilities.get(task_id.strip())

    async def get_lineage(self, task_id: str) -> TaskLineage | None:
        if self.fail_get_lineage:
            raise RuntimeError("Simulated get_lineage storage failure")
        return self.lineages.get(task_id.strip())

    async def get_children(self, parent_task_id: str) -> list[str]:
        if self.fail_get_children:
            raise RuntimeError("Simulated get_children storage failure")
        return sorted(self.children.get(parent_task_id.strip(), set()))

    async def update_status(self, task_id: str, status: str) -> None:
        self.statuses[task_id.strip()] = status.strip()

    async def get_status(self, task_id: str, now: str = "") -> str | None:
        return self.statuses.get(task_id.strip())

    async def save_task_input(self, task_input: TaskInput) -> None:
        self.inputs[(task_input.task_id, task_input.input_version)] = task_input

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
        self.artifacts[(artifact.producing_task_id, artifact.artifact_id, artifact.artifact_version)] = artifact

    async def get_stage_artifact(
        self, producing_task_id: str, artifact_id: str, artifact_version: int = 1
    ) -> StageArtifact | None:
        return self.artifacts.get((producing_task_id.strip(), artifact_id.strip(), artifact_version))

    async def save_continuation(self, continuation: TaskContinuation) -> None:
        self.continuations[(continuation.parent_task_id, continuation.continuation_version)] = continuation

    async def get_continuation(
        self, parent_task_id: str, continuation_version: int = 1
    ) -> TaskContinuation | None:
        return self.continuations.get((parent_task_id.strip(), continuation_version))

    async def save_root_authority(self, authority: RootAuthorityRecord) -> None:
        self.authorities[authority.root_task_id] = authority

    async def get_root_authority(self, root_task_id: str) -> RootAuthorityRecord | None:
        return self.authorities.get(root_task_id.strip())

    async def save_authority_grant(self, grant: DelegationGrant) -> None:
        self.authority_grants[(grant.root_task_id, grant.grant_id)] = grant

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
        expected_continuation_version: int | None = None,
        is_root: bool = True,
    ) -> tuple[ActiveClaim, bool]:
        clean_task_id = task_id.strip()
        cur_status = self.statuses.get(clean_task_id)
        if not cur_status:
            raise TaskNotClaimableError(f"Task '{clean_task_id}' not found")

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
                else:
                    raise TaskAlreadyClaimedError(f"Task '{clean_task_id}' has an active claim")

        if effective_status not in ("pending", "resumable"):
            raise TaskNotClaimableError(f"Task '{clean_task_id}' is in non-claimable status '{effective_status}'")

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
        clean_time = normalize_canonical_utc_timestamp(acknowledged_at)
        if claim.delivery_expires_at <= clean_time:
            raise ClaimDeliveryLeaseExpiredError("Lease expired")

        ack_dict = claim.to_dict()
        ack_dict["state"] = "acknowledged"
        ack_dict["acknowledged_at"] = clean_time
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
        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )
        if claim.state == "claimed_unacknowledged" and claim.delivery_expires_at <= now_iso:
            del self.claims[clean_task_id]
            self.statuses[clean_task_id] = "pending"
            return True
        return False

    async def dispatch_hint(self, hint: WakeUpHint, target: str) -> None:
        self.hints.append((hint, target))

    async def materialize_claim_timeout(self, child_task_id: str, now: str) -> Any:
        raise NotImplementedError

    async def transition_expired_claim(self, task_id: str, now: str) -> Any:
        raise NotImplementedError

    async def resolve_expired_claim(self, command: Any, operator_principal_id: str, now: str) -> Any:
        raise NotImplementedError

    async def get_claim_expiry(self, task_id: str) -> Any:
        return None

    async def get_recovery_decision(self, task_id: str, recovery_decision_id: str) -> Any:
        return None

    async def enqueue(
        self,
        task: TaskRequested,
        capability: AgentCapability = AgentCapability.GENERAL,
        target: str | None = None,
    ) -> None:
        self.enqueued.append((task, capability))
        if target:
            self.dispatches.append((task, target))

    async def dispatch(
        self, task: TaskRequested, target: str, capability: str | None = None
    ) -> None:
        if self._delivery_mode == "authenticated_claim":
            recognized = frozenset({"general", "research", "analysis"})
            if capability is not None and str(capability).strip():
                cand = str(capability).strip()
                if cand in recognized:
                    resolved_cap = cand
                else:
                    raise ValueError(f"Unrecognized canonical capability: '{cand}'")
            else:
                inv_map = {
                    "agent.tasks.general": "general",
                    "agent.tasks.research": "research",
                }
                mapped = inv_map.get(target)
                if mapped in recognized:
                    resolved_cap = mapped
                else:
                    raise ValueError(
                        f"Cannot derive recognized canonical capability from target '{target}'"
                    )
            hint_target = f"agent.hints.{resolved_cap}"
            hint = WakeUpHint(
                hint_id=f"hint-{task.task_id}",
                capability=resolved_cap,
                created_at=datetime.now(UTC).isoformat(),
                execution_kind="initial",
                task_id=task.task_id,
            )
            self.hints.append((hint, hint_target))
        else:
            self.dispatches.append((task, target))
            capability = (
                AgentCapability.RESEARCH if "research" in target else AgentCapability.GENERAL
            )
            self.enqueued.append((task, capability))
        self.statuses[task.task_id] = "pending"


class FakeStateReader:
    def __init__(
        self,
        agents_data: list[dict[str, Any]] | None = None,
        events_data: list[dict[str, str]] | None = None,
        routing_decisions: dict[str, dict[str, Any]] | None = None,
        task_results: dict[str, Any] | None = None,
        task_statuses: dict[str, Any] | None = None,
    ) -> None:
        self.agents_data = agents_data or []
        self.events_data = events_data or []
        self.routing_decisions = routing_decisions or {}
        self.task_results = task_results or {}
        self.task_statuses = task_statuses or {}
        self.last_events_limit: int | None = None

    async def agents(self) -> list[dict[str, str]]:
        return self.agents_data

    async def events(self, limit: int = 50) -> list[dict[str, str]]:
        self.last_events_limit = limit
        return self.events_data[:limit]

    async def routing_decision(self, task_id: str) -> dict[str, Any] | None:
        if task_id in self.routing_decisions:
            return self.routing_decisions[task_id]
        for event in self.events_data:
            if event.get("task_id") == task_id and event.get("routing_decision"):
                rd = event["routing_decision"]
                import json

                return json.loads(rd) if isinstance(rd, str) else rd
        return None

    async def task_inference_telemetry(self, task_id: str) -> dict[str, Any] | None:
        import json

        from agent_contracts import FallbackTransition, InferenceExecutionTelemetry
        from orchestrator.application.inference_history import (
            TaskEventObservation,
            project_task_inference_history,
        )

        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            return None

        matching = [e for e in self.events_data if e.get("task_id") == clean_task_id]
        task_completed = clean_task_id in self.task_results
        task_exists_durable = (
            task_completed
            or (clean_task_id in self.routing_decisions)
            or (clean_task_id in self.task_statuses)
        )
        task_failed = False
        if clean_task_id in self.task_statuses:
            s = self.task_statuses[clean_task_id]
            if (isinstance(s, dict) and s.get("status") in ("error", "failed")) or s in (
                "error",
                "failed",
            ):
                task_failed = True

        observations: list[TaskEventObservation] = []
        for stream_order, e in enumerate(matching):
            event_id = str(e.get("event_id", f"1-{stream_order}"))
            state = e.get("state")
            is_terminal_lifecycle = state in ("idle", "completed")
            is_terminal_error = state == "error"

            parsed_telem: InferenceExecutionTelemetry | None = None
            raw_telem = e.get("inference_telemetry")
            is_malformed = False
            if raw_telem:
                try:
                    telem_data = json.loads(raw_telem) if isinstance(raw_telem, str) else raw_telem
                    if isinstance(telem_data, dict):
                        parsed_telem = InferenceExecutionTelemetry.from_dict(telem_data)
                    else:
                        is_malformed = True
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    is_malformed = True

            parsed_trans: FallbackTransition | None = None
            raw_trans = e.get("fallback_transition")
            if raw_trans:
                try:
                    trans_data = json.loads(raw_trans) if isinstance(raw_trans, str) else raw_trans
                    if isinstance(trans_data, dict):
                        parsed_trans = FallbackTransition.from_dict(trans_data)
                    else:
                        is_malformed = True
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    is_malformed = True

            observations.append(
                TaskEventObservation(
                    event_id=event_id,
                    task_id=clean_task_id,
                    stream_position=(
                        int(event_id.split("-")[0]) if event_id.split("-")[0].isdigit() else 0,
                        int(event_id.split("-")[1]) if len(event_id.split("-")) > 1 and event_id.split("-")[1].isdigit() else 0,
                    ),
                    stream_order=stream_order,
                    state=state,
                    inference_telemetry=parsed_telem,
                    fallback_transition=parsed_trans,
                    is_terminal_lifecycle=is_terminal_lifecycle,
                    is_terminal_error=is_terminal_error,
                    is_malformed=is_malformed,
                )
            )

        return project_task_inference_history(
            task_id=clean_task_id,
            observations=observations,
            task_exists_durable=task_exists_durable,
            task_completed=task_completed,
            task_failed=task_failed,
        )


class FakeResultReader:
    def __init__(
        self,
        results: dict[str, dict[str, Any]] | None = None,
        pending: set[str] | None = None,
    ) -> None:
        self.results = results or {}
        self.pending = pending or set()

    async def get_result(self, task_id: str, now: str = "") -> dict[str, Any] | None:
        if task_id in self.results:
            return self.results[task_id]
        if task_id in self.pending:
            return {"task_id": task_id, "status": "pending", "output": None}
        return None


@pytest.mark.anyio
async def test_submit_task_use_case() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(task_store=queue, queue=queue)
    task = await use_case.execute("test task prompt")

    assert task.prompt == "test task prompt"
    assert len(queue.enqueued) == 1

    enqueued_task, capability = queue.enqueued[0]

    assert enqueued_task.prompt == "test task prompt"
    assert capability == AgentCapability.GENERAL

@pytest.mark.anyio
async def test_get_agent_states_query() -> None:
    expected = [{"agent_id": "worker-1", "state": "idle", "detail": "Waiting"}]
    reader = FakeStateReader(agents_data=expected)
    query = GetAgentStates(reader)

    result = await query.execute()
    assert result == expected


@pytest.mark.anyio
async def test_get_recent_events_bounds_limits() -> None:
    sample_events = [{"event_id": str(i), "detail": f"event {i}"} for i in range(150)]
    reader = FakeStateReader(events_data=sample_events)
    query = GetRecentEvents(reader)

    # Test default
    await query.execute()
    assert reader.last_events_limit == 50

    # Test lower bound (<= 0 maps to 1)
    await query.execute(limit=0)
    assert reader.last_events_limit == 1

    await query.execute(limit=-10)
    assert reader.last_events_limit == 1

    # Test upper bound (> 100 maps to 100)
    await query.execute(limit=250)
    assert reader.last_events_limit == 100

    # Test valid within-range limit
    await query.execute(limit=25)
    assert reader.last_events_limit == 25

@pytest.mark.anyio
async def test_submit_research_task_routes_to_research_capability() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(task_store=queue, queue=queue)

    task = await use_case.execute(
        "Compare local and cloud LLMs",
        capability=AgentCapability.RESEARCH,
    )

    assert len(queue.enqueued) == 1

    enqueued_task, capability = queue.enqueued[0]

    assert enqueued_task == task
    assert capability == AgentCapability.RESEARCH


@pytest.mark.anyio
async def test_submit_task_automatic_routing_research() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(task_store=queue, queue=queue)

    # Under Phase B routing policy, genuine research prompts require external retrieval/freshness/source evidence rather than weak verbs alone
    task = await use_case.execute("Compare current features and pricing of local and cloud LLMs")

    assert len(queue.enqueued) == 1
    enqueued_task, capability = queue.enqueued[0]
    assert enqueued_task == task
    assert capability == AgentCapability.RESEARCH


@pytest.mark.anyio
async def test_submit_task_automatic_routing_general() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(task_store=queue, queue=queue)

    task = await use_case.execute("build agent")

    assert len(queue.enqueued) == 1
    enqueued_task, capability = queue.enqueued[0]
    assert enqueued_task == task
    assert capability == AgentCapability.GENERAL


@pytest.mark.anyio
async def test_submit_task_routing_v2_freshness_and_explanatory() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(task_store=queue, queue=queue)

    # Freshness prompt automatically resolves to RESEARCH
    task_research = await use_case.execute(
        "What are the latest developments in battery technology?"
    )
    assert queue.enqueued[0] == (task_research, AgentCapability.RESEARCH)

    # General explanatory prompt automatically resolves to GENERAL
    task_general = await use_case.execute(
        "Explain the difference between containers and virtual machines."
    )
    assert queue.enqueued[1] == (task_general, AgentCapability.GENERAL)



@pytest.mark.anyio
async def test_submit_task_preserves_explicit_capability_override() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(task_store=queue, queue=queue)

    # Research prompt explicitly directed to GENERAL
    task1 = await use_case.execute(
        "Investigate system logs",
        capability=AgentCapability.GENERAL,
    )
    assert queue.enqueued[0] == (task1, AgentCapability.GENERAL)

    # General prompt explicitly directed to RESEARCH
    task2 = await use_case.execute(
        "build agent",
        capability=AgentCapability.RESEARCH,
    )
    assert queue.enqueued[1] == (task2, AgentCapability.RESEARCH)


class FakeCapabilityRouter:
    def __init__(self, capability_to_return: AgentCapability) -> None:
        self.capability_to_return = capability_to_return
        self.routed_prompts: list[str] = []

    def route(self, prompt: str) -> AgentCapability:
        self.routed_prompts.append(prompt)
        return self.capability_to_return


@pytest.mark.anyio
async def test_submit_task_uses_injected_router() -> None:
    queue = FakeTaskQueue()
    fake_router = FakeCapabilityRouter(AgentCapability.RESEARCH)
    use_case = SubmitTask(task_store=queue, queue=queue, router=fake_router)

    # Non-research prompt routed to RESEARCH via injected router
    task = await use_case.execute("simple task")

    assert fake_router.routed_prompts == ["simple task"]
    assert queue.enqueued[0] == (task, AgentCapability.RESEARCH)


@pytest.mark.anyio
async def test_get_task_result_query_completed() -> None:
    expected = {
        "task_id": "task-1",
        "status": "completed",
        "output": "research findings",
    }
    reader = FakeResultReader(results={"task-1": expected})
    query = GetTaskResult(reader)

    result = await query.execute("task-1")
    assert result == expected


@pytest.mark.anyio
async def test_get_task_result_query_pending() -> None:
    reader = FakeResultReader(pending={"task-pending-1"})
    query = GetTaskResult(reader)

    result = await query.execute("task-pending-1")
    assert result == {"task_id": "task-pending-1", "status": "pending", "output": None}


@pytest.mark.anyio
async def test_get_task_result_query_not_found() -> None:
    reader = FakeResultReader()
    query = GetTaskResult(reader)

    result = await query.execute("non-existent-task")
    assert result is None


@pytest.mark.anyio
async def test_get_task_inference_telemetry_query_success() -> None:
    from orchestrator.application.queries import GetTaskInferenceTelemetry

    expected_telem = {
        "provider_name": "cloudflare",
        "attempt_index": 0,
        "latency_ms": 94.3,
        "status": "succeeded",
        "model_name": "@cf/meta/llama-3.1-8b-instruct",
        "finish_reason": "stop",
    }
    reader = FakeStateReader(
        events_data=[
            {
                "task_id": "task-telemetry-1",
                "inference_telemetry": expected_telem,
            }
        ]
    )
    query = GetTaskInferenceTelemetry(reader)
    result = await query.execute("task-telemetry-1")
    assert result is not None
    assert result["task_id"] == "task-telemetry-1"
    assert result["inference_telemetry"]["provider_name"] == "cloudflare"
    assert result["inference_telemetry"]["status"] == "succeeded"


@pytest.mark.anyio
async def test_get_task_inference_telemetry_query_not_found() -> None:
    from orchestrator.application.queries import GetTaskInferenceTelemetry

    reader = FakeStateReader(events_data=[])
    query = GetTaskInferenceTelemetry(reader)
    result = await query.execute("task-missing")
    assert result is None


@pytest.mark.anyio
async def test_get_task_inference_telemetry_query_empty_task_id_raises() -> None:
    from orchestrator.application.queries import GetTaskInferenceTelemetry

    reader = FakeStateReader()
    query = GetTaskInferenceTelemetry(reader)
    with pytest.raises(ValueError, match="task_id must not be empty"):
        await query.execute("")
    with pytest.raises(ValueError, match="task_id must not be empty"):
        await query.execute("   ")


@pytest.mark.anyio
async def test_get_task_children_query_lifecycle() -> None:
    from orchestrator.application.queries import GetTaskChildren

    store = FakeTaskQueue()
    root = TaskRequested.create("Root task")
    await store.register_root_task(root)

    child1 = TaskRequested(
        task_id="child-b",
        prompt="Child B",
        created_at=root.created_at,
        root_task_id=root.task_id,
        parent_task_id=root.task_id,
        version=2,
    )
    child2 = TaskRequested(
        task_id="child-a",
        prompt="Child A",
        created_at=root.created_at,
        root_task_id=root.task_id,
        parent_task_id=root.task_id,
        version=2,
    )
    await store.register_child_task(child1)
    await store.register_child_task(child2)

    query = GetTaskChildren(store)

    # 1. Successful query with lexicographical sorting
    children = await query.execute(root.task_id)
    assert children == ["child-a", "child-b"]

    # 2. Existing parent with zero children returns []
    empty_root = TaskRequested.create("Empty root")
    await store.register_root_task(empty_root)
    assert await query.execute(empty_root.task_id) == []

    # 3. Child queried as parent returns []
    assert await query.execute(child1.task_id) == []

    # 4. Unknown parent returns None
    assert await query.execute("unknown-parent") is None

    # 5. Empty/blank task_id raises ValueError
    with pytest.raises(ValueError, match="task_id must not be empty"):
        await query.execute("")
    with pytest.raises(ValueError, match="task_id must not be empty"):
        await query.execute("   ")

    # 6. Stripping surrounding whitespace
    assert await query.execute(f"  {root.task_id}  ") == ["child-a", "child-b"]


@pytest.mark.anyio
async def test_get_task_children_get_lineage_failure_propagates() -> None:
    from orchestrator.application.queries import GetTaskChildren

    store = FakeTaskQueue()
    store.fail_get_lineage = True

    query = GetTaskChildren(store)
    with pytest.raises(RuntimeError, match="Simulated get_lineage storage failure"):
        await query.execute("root-1")


@pytest.mark.anyio
async def test_get_task_children_get_children_failure_propagates() -> None:
    from orchestrator.application.queries import GetTaskChildren

    store = FakeTaskQueue()
    root = TaskRequested.create("Root task")
    await store.register_root_task(root)
    store.fail_get_children = True

    query = GetTaskChildren(store)
    with pytest.raises(RuntimeError, match="Simulated get_children storage failure"):
        await query.execute(root.task_id)

