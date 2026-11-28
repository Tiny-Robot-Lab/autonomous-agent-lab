import json
from datetime import datetime
from typing import Any

from agent_contracts import (
    AgentCandidate,
    LivenessPolicy,
    TaskContinuationV2,
    evaluate_agent_record,
)
from agent_contracts.capabilities import DISALLOWED_METADATA_KEYS

from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.task import CorruptAuthoritativeRecordError

from .ports import (
    AuthoritativeTaskStore,
    ResultReader,
    StateReader,
    TurnCompletionStore,
)


class GetAgentStates:
    def __init__(self, reader: StateReader) -> None:
        self._reader = reader

    async def execute(self) -> list[dict[str, Any]]:
        return await self._reader.agents()


class GetRecentEvents:
    def __init__(self, reader: StateReader) -> None:
        self._reader = reader

    async def execute(self, limit: int = 50) -> list[dict[str, str]]:
        bounded_limit = min(max(limit, 1), 100)
        return await self._reader.events(limit=bounded_limit)


class GetTaskResult:
    def __init__(self, reader: ResultReader) -> None:
        self._reader = reader

    async def execute(self, task_id: str, now: str = "") -> dict[str, Any] | None:
        return await self._reader.get_result(task_id, now=now)


class GetRoutingDecision:
    """Query a recorded routing decision by its associated task ID."""

    def __init__(self, reader: StateReader) -> None:
        self._reader = reader

    async def execute(self, task_id: str) -> dict[str, Any] | None:
        if not task_id or not str(task_id).strip():
            raise ValueError("task_id must not be empty")
        return await self._reader.routing_decision(str(task_id).strip())


class GetTaskInferenceTelemetry:
    """Query task-scoped inference execution telemetry and fallback history."""

    def __init__(self, reader: StateReader) -> None:
        self._reader = reader

    async def execute(self, task_id: str) -> dict[str, Any] | None:
        if not task_id or not str(task_id).strip():
            raise ValueError("task_id must not be empty")
        return await self._reader.task_inference_telemetry(str(task_id).strip())


class GetTaskChildren:
    """Query immediate child task IDs for an authoritative parent task."""

    def __init__(self, task_store: AuthoritativeTaskStore) -> None:
        self._task_store = task_store

    async def execute(self, parent_task_id: str) -> list[str] | None:
        if not parent_task_id or not str(parent_task_id).strip():
            raise ValueError("task_id must not be empty")
        clean_id = str(parent_task_id).strip()

        lineage = await self._task_store.get_lineage(clean_id)
        if lineage is None:
            return None

        return await self._task_store.get_children(clean_id)


class FindCandidatesByCapability:
    """Query registered agents providing a required capability and eligible for task dispatch."""

    def __init__(
        self,
        reader: StateReader,
        policy: LivenessPolicy | None = None,
    ) -> None:
        self._reader = reader
        self._policy = policy or LivenessPolicy()

    async def execute(
        self,
        capability: AgentCapability | str,
        as_of: datetime | str | None = None,
    ) -> list[AgentCandidate]:
        if isinstance(capability, AgentCapability):
            target_cap = capability.value.lower()
        elif isinstance(capability, str):
            target_cap = capability.strip().lower()
            if not target_cap:
                raise ValueError("capability must not be empty")
        else:
            raise TypeError(
                f"capability must be an AgentCapability or str, got {type(capability).__name__}"
            )

        raw_agents = await self._reader.agents()
        candidates: list[AgentCandidate] = []

        for entry in raw_agents:
            if not isinstance(entry, dict):
                continue

            raw_agent_id = entry.get("agent_id")
            if not raw_agent_id or not str(raw_agent_id).strip():
                continue

            raw_capabilities = entry.get("capabilities", [])
            if isinstance(raw_capabilities, str):
                try:
                    raw_capabilities = json.loads(raw_capabilities)
                except (json.JSONDecodeError, TypeError):
                    raw_capabilities = [raw_capabilities]

            if not isinstance(raw_capabilities, list):
                continue

            normalized_capabilities: list[str] = [
                str(c).strip().lower()
                for c in raw_capabilities
                if isinstance(c, (str, AgentCapability)) and str(c).strip()
            ]

            # Step 1: Capability match check
            if target_cap not in normalized_capabilities:
                continue

            # Step 2: Availability evaluation using runtime liveness policy
            availability = evaluate_agent_record(
                entry,
                as_of=as_of,
                policy=self._policy,
            )

            # Step 3: Eligibility check (only READY agents with fresh heartbeats are eligible)
            if not availability.is_eligible:
                continue

            # Step 4: Extract and sanitize provider metadata
            raw_providers = entry.get("providers", {})
            if isinstance(raw_providers, str):
                try:
                    raw_providers = json.loads(raw_providers)
                except (json.JSONDecodeError, TypeError):
                    raw_providers = {}

            sanitized_providers: dict[str, str] = {}
            if isinstance(raw_providers, dict):
                for k, v in raw_providers.items():
                    k_str = str(k).strip()
                    k_lower = k_str.lower()
                    if any(disallowed in k_lower for disallowed in DISALLOWED_METADATA_KEYS):
                        continue
                    sanitized_providers[k_str] = str(v)

            raw_dispatch_target = entry.get("dispatch_target")

            candidates.append(
                AgentCandidate(
                    agent_id=str(raw_agent_id).strip(),
                    capabilities=normalized_capabilities,
                    agent_type=str(entry.get("agent_type", "agent")),
                    execution_state=availability.execution_state,
                    availability=availability,
                    providers=sanitized_providers,
                    dispatch_target=(
                        str(raw_dispatch_target).strip()
                        if raw_dispatch_target
                        else None
                    ),
                )
            )

        return candidates


QueryAgentsByCapability = FindCandidatesByCapability


class GetTaskCollaboration:
    """Query authoritative collaboration lifecycle state for a parent task."""

    def __init__(
        self,
        task_store: AuthoritativeTaskStore,
        completion_store: TurnCompletionStore | None = None,
    ) -> None:
        self._task_store = task_store
        self._completion_store = (
            completion_store
            if completion_store is not None
            else (task_store if isinstance(task_store, TurnCompletionStore) else None)
        )

    async def execute(self, parent_task_id: str) -> dict[str, Any] | None:
        if not parent_task_id or not str(parent_task_id).strip():
            raise ValueError("task_id must not be empty")
        clean_id = str(parent_task_id).strip()

        # 1. Authoritative parent status and existence
        parent_status = await self._task_store.get_status(clean_id)
        if parent_status is None:
            # Check if parent task exists via lineage or input
            lineage = await self._task_store.get_lineage(clean_id)
            if lineage is not None:
                raise CorruptAuthoritativeRecordError(
                    f"Parent task '{clean_id}' exists in lineage but status record is missing"
                )
            parent_input = await self._task_store.get_task_input(clean_id, 1)
            if parent_input is not None:
                raise CorruptAuthoritativeRecordError(
                    f"Parent task '{clean_id}' exists in task input but status record is missing"
                )
            return None

        # 2. Check continuation version and record
        latest_ver = await self._task_store.get_latest_continuation_version(clean_id)
        if latest_ver is None or latest_ver < 2:
            return {
                "parent_task_id": clean_id,
                "parent_status": parent_status,
                "has_collaboration": False,
                "continuation_version": None,
                "continuation_state": None,
                "wave_id": None,
                "barrier_policy": None,
                "wave_resolution": None,
                "children": [],
            }

        continuation = await self._task_store.get_continuation(
            clean_id, continuation_version=latest_ver
        )
        if continuation is None:
            raise CorruptAuthoritativeRecordError(
                f"Continuation version {latest_ver} recorded for task '{clean_id}' but continuation record is missing"
            )

        if not isinstance(continuation, TaskContinuationV2):
            raise CorruptAuthoritativeRecordError(
                f"Continuation version {latest_ver} for task '{clean_id}' is not TaskContinuationV2 (got {type(continuation).__name__})"
            )

        wave_id = continuation.wave_id
        accepted_wave = await self._task_store.get_accepted_delegation_wave(
            clean_id, wave_id
        )
        if accepted_wave is None:
            raise CorruptAuthoritativeRecordError(
                f"Accepted delegation wave '{wave_id}' missing for collaborative task '{clean_id}'"
            )

        resolved_wave = await self._task_store.get_resolved_delegation_wave(
            clean_id, wave_id
        )
        wave_resolution = (
            resolved_wave.resolution
            if resolved_wave is not None
            else continuation.wave_resolution
        )

        children_list: list[dict[str, Any]] = []
        for child_id in accepted_wave.required_child_task_ids:
            child_status = await self._task_store.get_status(child_id)
            if child_status is None:
                raise CorruptAuthoritativeRecordError(
                    f"Required child '{child_id}' for wave '{wave_id}' missing status record"
                )

            child_input = await self._task_store.get_task_input(
                child_id, input_version=1
            )
            if child_input is None:
                raise CorruptAuthoritativeRecordError(
                    f"Required child '{child_id}' for wave '{wave_id}' missing TaskInput v1"
                )
            intent = child_input.prompt

            terminal_status = None
            if self._completion_store is not None:
                outcome = await self._completion_store.get_terminal_outcome(child_id)
                if outcome is not None:
                    terminal_status = outcome.terminal_status

            children_list.append({
                "task_id": child_id,
                "intent": intent,
                "status": child_status,
                "terminal_status": terminal_status,
            })

        return {
            "parent_task_id": clean_id,
            "parent_status": parent_status,
            "has_collaboration": True,
            "continuation_version": continuation.continuation_version,
            "continuation_state": continuation.state,
            "wave_id": wave_id,
            "barrier_policy": accepted_wave.barrier_policy,
            "wave_resolution": wave_resolution,
            "children": children_list,
        }
