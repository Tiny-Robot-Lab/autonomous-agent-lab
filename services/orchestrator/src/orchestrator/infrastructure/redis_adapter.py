import json
import logging
from typing import Any

import redis.asyncio as redis
from agent_contracts import (
    AgentEvent,
    FallbackTransition,
    InferenceExecutionTelemetry,
    RootAuthorityRecord,
    RoutingDecision,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskRequested,
    canonical_json_dumps,
)

from orchestrator.application.inference_history import (
    TaskEventObservation,
    project_task_inference_history,
)
from orchestrator.application.ports import TelemetryStorageError
from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.task import (
    ConflictingContinuationError,
    ConflictingRootAuthorityError,
    ConflictingRootTaskRegistrationError,
    ConflictingStageArtifactError,
    ConflictingTaskInputError,
    CorruptAuthoritativeRecordError,
    NonContiguousVersionError,
    TaskLineage,
)

logger = logging.getLogger(__name__)


def _parse_redis_stream_id(event_id: str) -> tuple[int, int]:
    """Parse a Redis stream ID ('<ms>-<seq>') into an integer tuple for chronological sorting."""
    try:
        parts = event_id.strip().split("-", 1)
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except (AttributeError, ValueError):
        return (0, 0)

CAPABILITY_QUEUE_MAP: dict[AgentCapability, str] = {
    AgentCapability.GENERAL: "agent.tasks.general",
    AgentCapability.RESEARCH: "agent.tasks.research",
}

REGISTER_ROOT_TASK_LUA = """
local marker_key = KEYS[1]
local status_key = KEYS[2]
local lineage_key = KEYS[3]
local input_key = KEYS[4]
local authority_key = KEYS[5]

local marker_payload = ARGV[1]
local initial_status_json = ARGV[2]
local lineage_json = ARGV[3]
local input_json = ARGV[4]
local authority_json = ARGV[5]
local expected_task_id = ARGV[6]

local exists_marker = redis.call('EXISTS', marker_key)
local exists_status = redis.call('EXISTS', status_key)
local exists_lineage = redis.call('EXISTS', lineage_key)
local exists_input = redis.call('EXISTS', input_key)
local exists_authority = redis.call('EXISTS', authority_key)

if exists_marker == 0 and exists_status == 0 and exists_lineage == 0 and exists_input == 0 and exists_authority == 0 then
    redis.call('SET', marker_key, marker_payload)
    redis.call('SET', status_key, initial_status_json)
    redis.call('SET', lineage_key, lineage_json)
    redis.call('SET', input_key, input_json)
    redis.call('SET', authority_key, authority_json)
    return 'OK'
end

if exists_marker == 1 and exists_status == 1 and exists_lineage == 1 and exists_input == 1 and exists_authority == 1 then
    local cur_marker = redis.call('GET', marker_key)
    local cur_lineage = redis.call('GET', lineage_key)
    local cur_input = redis.call('GET', input_key)
    local cur_authority = redis.call('GET', authority_key)
    if cur_marker == marker_payload and cur_lineage == lineage_json and cur_input == input_json and cur_authority == authority_json then
        local cur_status = redis.call('GET', status_key)
        local ok, status_tbl = pcall(cjson.decode, cur_status)
        if ok and type(status_tbl) == 'table' and status_tbl['task_id'] == expected_task_id and type(status_tbl['status']) == 'string' then
            return 'IDEMPOTENT'
        end
    end
    return 'CONFLICT'
end

return 'PARTIAL_CONFLICT'
"""

SAVE_TASK_INPUT_LUA = """
local target_key = KEYS[1]
local version = tonumber(ARGV[1])
local payload = ARGV[2]

if version > 1 then
    local prior_key = KEYS[2]
    if (not prior_key) or redis.call('EXISTS', prior_key) == 0 then
        return 'NON_CONTIGUOUS'
    end
end

if redis.call('EXISTS', target_key) == 0 then
    redis.call('SET', target_key, payload)
    return 'OK'
else
    local existing = redis.call('GET', target_key)
    if existing == payload then
        return 'IDEMPOTENT'
    else
        return 'CONFLICT'
    end
end
"""

SAVE_STAGE_ARTIFACT_LUA = """
local target_key = KEYS[1]
local index_key = KEYS[2]
local version = tonumber(ARGV[1])
local payload = ARGV[2]
local index_member = ARGV[3]

if version > 1 then
    local prior_key = KEYS[3]
    if (not prior_key) or redis.call('EXISTS', prior_key) == 0 then
        return 'NON_CONTIGUOUS'
    end
end

if redis.call('EXISTS', target_key) == 0 then
    redis.call('SET', target_key, payload)
    redis.call('SADD', index_key, index_member)
    return 'OK'
else
    local existing = redis.call('GET', target_key)
    if existing == payload then
        redis.call('SADD', index_key, index_member)
        return 'IDEMPOTENT'
    else
        return 'CONFLICT'
    end
end
"""

SAVE_CONTINUATION_LUA = """
local target_key = KEYS[1]
local latest_key = KEYS[2]
local version = tonumber(ARGV[1])
local payload = ARGV[2]

if redis.call('EXISTS', target_key) == 0 then
    if version == 1 then
        if redis.call('EXISTS', latest_key) == 1 then
            return 'CORRUPT_POINTER'
        end
    else
        if redis.call('EXISTS', latest_key) == 0 then
            return 'CORRUPT_POINTER'
        end
        local cur_latest_raw = redis.call('GET', latest_key)
        if (not cur_latest_raw) or (not string.match(cur_latest_raw, '^%d+$')) then
            return 'CORRUPT_POINTER'
        end
        local cur_latest = tonumber(cur_latest_raw)
        if cur_latest >= version then
            return 'CORRUPT_POINTER'
        end
        if cur_latest < version - 1 then
            return 'NON_CONTIGUOUS'
        end
        local prior_key = KEYS[3]
        if (not prior_key) or redis.call('EXISTS', prior_key) == 0 then
            return 'NON_CONTIGUOUS'
        end
    end

    redis.call('SET', target_key, payload)
    redis.call('SET', latest_key, tostring(version))
    return 'OK'
else
    local existing = redis.call('GET', target_key)
    if existing == payload then
        if redis.call('EXISTS', latest_key) == 0 then
            return 'CORRUPT_POINTER'
        end
        local cur_latest_raw = redis.call('GET', latest_key)
        if (not cur_latest_raw) or (not string.match(cur_latest_raw, '^%d+$')) then
            return 'CORRUPT_POINTER'
        end
        local cur_latest = tonumber(cur_latest_raw)
        if cur_latest < version then
            return 'CORRUPT_POINTER'
        end
        return 'IDEMPOTENT'
    else
        return 'CONFLICT'
    end
end
"""

SAVE_ROOT_AUTHORITY_LUA = """
local key = KEYS[1]
local payload = ARGV[1]

if redis.call('EXISTS', key) == 0 then
    redis.call('SET', key, payload)
    return 'OK'
else
    local existing = redis.call('GET', key)
    if existing == payload then
        return 'IDEMPOTENT'
    else
        return 'CONFLICT'
    end
end
"""


class RedisGateway:
    def __init__(self, url: str) -> None:
        self._client = redis.from_url(url, decode_responses=True)

    async def register_root_task(
        self, task: TaskRequested, initial_status: str = "pending_dispatch"
    ) -> None:
        if task.parent_task_id is not None or task.root_task_id != task.task_id:
            raise ValueError(f"Task '{task.task_id}' is not a root task")

        task_input = TaskInput(
            task_id=task.task_id,
            prompt=task.prompt,
            created_at=task.created_at,
            input_version=1,
            artifact_references=(),
            metadata={},
            version=1,
        )
        root_authority = RootAuthorityRecord.create_default(
            root_task_id=task.task_id,
            created_at=task.created_at,
        )

        marker_key = f"task.registration:{task.task_id}"
        status_key = f"task.status:{task.task_id}"
        lineage_key = f"task.lineage:{task.task_id}"
        input_key = f"task.input:{task.task_id}:1"
        authority_key = f"task.authority:{task.task_id}"

        marker_json = canonical_json_dumps({
            "registered_at": task.created_at,
            "task_id": task.task_id,
        })
        status_json = canonical_json_dumps({
            "status": initial_status,
            "task_id": task.task_id,
        })
        lineage_json = canonical_json_dumps({
            "parent_task_id": None,
            "root_task_id": task.root_task_id,
            "task_id": task.task_id,
        })
        input_json = task_input.to_json()
        authority_json = root_authority.to_json()

        raw_result = await self._client.eval(
            REGISTER_ROOT_TASK_LUA,
            5,
            marker_key,
            status_key,
            lineage_key,
            input_key,
            authority_key,
            marker_json,
            status_json,
            lineage_json,
            input_json,
            authority_json,
            task.task_id,
        )
        result = (
            raw_result.decode("utf-8")
            if isinstance(raw_result, (bytes, bytearray))
            else str(raw_result)
        )
        if result in ("OK", "IDEMPOTENT"):
            return
        if result == "CONFLICT":
            raise ConflictingRootTaskRegistrationError(
                f"Root task '{task.task_id}' already exists with conflicting state"
            )
        if result == "PARTIAL_CONFLICT":
            raise ConflictingRootTaskRegistrationError(
                f"Root task '{task.task_id}' has inconsistent or partial authoritative state in storage"
            )
        raise RuntimeError(f"Unexpected Redis register_root_task result: {result}")

    async def register_child_task(
        self, task: TaskRequested, initial_status: str = "pending_dispatch"
    ) -> None:
        """Atomically persist pre-validated child task status, forward lineage, and parent membership.

        Domain validation (parent existence and depth-1 policy) is enforced by the
        platform application use case (RegisterChildTask) prior to invoking this method.
        This method executes the atomic storage consistency boundary across status, lineage, and parent set.
        """
        if task.parent_task_id is None or task.root_task_id == task.task_id:
            raise ValueError(f"Task '{task.task_id}' is not a child task")
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.set(
                f"task.status:{task.task_id}",
                canonical_json_dumps({"status": initial_status, "task_id": task.task_id}),
            )
            pipe.set(
                f"task.lineage:{task.task_id}",
                canonical_json_dumps({
                    "parent_task_id": task.parent_task_id,
                    "root_task_id": task.root_task_id,
                    "task_id": task.task_id,
                }),
            )
            pipe.sadd(f"task.children:{task.parent_task_id}", task.task_id)
            await pipe.execute()

    async def get_lineage(self, task_id: str) -> TaskLineage | None:
        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            return None
        key = f"task.lineage:{clean_task_id}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise TypeError("Lineage payload must be a JSON object")
            return TaskLineage.from_dict(data)
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt TaskLineage record at key '{key}': {exc}"
            ) from exc

    async def get_children(self, parent_task_id: str) -> list[str]:
        clean_parent_task_id = str(parent_task_id).strip()
        if not clean_parent_task_id:
            return []
        members = await self._client.smembers(f"task.children:{clean_parent_task_id}")
        decoded = [
            m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m)
            for m in members
        ]
        return sorted(decoded)

    async def update_status(self, task_id: str, status: str) -> None:
        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            raise ValueError("task_id must be a non-empty string")
        clean_status = str(status).strip()
        if not clean_status:
            raise ValueError("status must be a non-empty string")
        await self._client.set(
            f"task.status:{clean_task_id}",
            canonical_json_dumps({"status": clean_status, "task_id": clean_task_id}),
        )

    async def get_status(self, task_id: str) -> str | None:
        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            return None
        key = f"task.status:{clean_task_id}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            if isinstance(data, dict) and isinstance(data.get("status"), str):
                return data["status"]
            raise ValueError("Status record must be a JSON object containing a string 'status'")
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt task status record at key '{key}': {exc}"
            ) from exc

    async def save_task_input(self, task_input: TaskInput) -> None:
        target_key = f"task.input:{task_input.task_id}:{task_input.input_version}"
        payload = task_input.to_json()
        if task_input.input_version == 1:
            raw_result = await self._client.eval(
                SAVE_TASK_INPUT_LUA,
                1,
                target_key,
                task_input.input_version,
                payload,
            )
        else:
            prior_key = f"task.input:{task_input.task_id}:{task_input.input_version - 1}"
            raw_result = await self._client.eval(
                SAVE_TASK_INPUT_LUA,
                2,
                target_key,
                prior_key,
                task_input.input_version,
                payload,
            )
        result = (
            raw_result.decode("utf-8")
            if isinstance(raw_result, (bytes, bytearray))
            else str(raw_result)
        )
        if result in ("OK", "IDEMPOTENT"):
            return
        if result == "NON_CONTIGUOUS":
            raise NonContiguousVersionError(
                f"Cannot save TaskInput version {task_input.input_version} for task "
                f"'{task_input.task_id}' because prior version {task_input.input_version - 1} does not exist"
            )
        if result == "CONFLICT":
            raise ConflictingTaskInputError(
                f"TaskInput for task '{task_input.task_id}' version {task_input.input_version} "
                "already exists with conflicting data"
            )
        raise RuntimeError(f"Unexpected Redis save_task_input result: {result}")

    async def get_task_input(
        self, task_id: str, input_version: int = 1
    ) -> TaskInput | None:
        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            return None
        key = f"task.input:{clean_task_id}:{input_version}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise TypeError("Payload must be a JSON object")
            return TaskInput.from_dict(data)
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt TaskInput record at key '{key}': {exc}"
            ) from exc

    async def save_stage_artifact(self, artifact: StageArtifact) -> None:
        target_key = (
            f"task.artifact:{artifact.producing_task_id}:"
            f"{artifact.artifact_id}:{artifact.artifact_version}"
        )
        index_key = f"task.artifacts:{artifact.producing_task_id}"
        payload = artifact.to_json()
        index_member = f"{artifact.artifact_id}:{artifact.artifact_version}"
        if artifact.artifact_version == 1:
            raw_result = await self._client.eval(
                SAVE_STAGE_ARTIFACT_LUA,
                2,
                target_key,
                index_key,
                artifact.artifact_version,
                payload,
                index_member,
            )
        else:
            prior_key = (
                f"task.artifact:{artifact.producing_task_id}:"
                f"{artifact.artifact_id}:{artifact.artifact_version - 1}"
            )
            raw_result = await self._client.eval(
                SAVE_STAGE_ARTIFACT_LUA,
                3,
                target_key,
                index_key,
                prior_key,
                artifact.artifact_version,
                payload,
                index_member,
            )
        result = (
            raw_result.decode("utf-8")
            if isinstance(raw_result, (bytes, bytearray))
            else str(raw_result)
        )
        if result in ("OK", "IDEMPOTENT"):
            return
        if result == "NON_CONTIGUOUS":
            raise NonContiguousVersionError(
                f"Cannot save StageArtifact version {artifact.artifact_version} for task "
                f"'{artifact.producing_task_id}', artifact '{artifact.artifact_id}' because "
                f"prior version {artifact.artifact_version - 1} does not exist"
            )
        if result == "CONFLICT":
            raise ConflictingStageArtifactError(
                f"StageArtifact '{artifact.artifact_id}' for task '{artifact.producing_task_id}' "
                f"version {artifact.artifact_version} already exists with conflicting data"
            )
        raise RuntimeError(f"Unexpected Redis save_stage_artifact result: {result}")

    async def get_stage_artifact(
        self, producing_task_id: str, artifact_id: str, artifact_version: int = 1
    ) -> StageArtifact | None:
        clean_task_id = str(producing_task_id).strip()
        clean_artifact_id = str(artifact_id).strip()
        if not clean_task_id or not clean_artifact_id:
            return None
        key = f"task.artifact:{clean_task_id}:{clean_artifact_id}:{artifact_version}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise TypeError("Payload must be a JSON object")
            return StageArtifact.from_dict(data)
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt StageArtifact record at key '{key}': {exc}"
            ) from exc

    async def save_continuation(self, continuation: TaskContinuation) -> None:
        target_key = (
            f"task.continuation:{continuation.parent_task_id}:"
            f"{continuation.continuation_version}"
        )
        latest_key = f"task.continuation_latest:{continuation.parent_task_id}"
        payload = continuation.to_json()
        if continuation.continuation_version == 1:
            raw_result = await self._client.eval(
                SAVE_CONTINUATION_LUA,
                2,
                target_key,
                latest_key,
                continuation.continuation_version,
                payload,
            )
        else:
            prior_key = (
                f"task.continuation:{continuation.parent_task_id}:"
                f"{continuation.continuation_version - 1}"
            )
            raw_result = await self._client.eval(
                SAVE_CONTINUATION_LUA,
                3,
                target_key,
                latest_key,
                prior_key,
                continuation.continuation_version,
                payload,
            )
        result = (
            raw_result.decode("utf-8")
            if isinstance(raw_result, (bytes, bytearray))
            else str(raw_result)
        )
        if result in ("OK", "IDEMPOTENT"):
            return
        if result == "NON_CONTIGUOUS":
            raise NonContiguousVersionError(
                f"Cannot save TaskContinuation version {continuation.continuation_version} "
                f"for parent task '{continuation.parent_task_id}' because prior version "
                f"{continuation.continuation_version - 1} does not exist"
            )
        if result == "CORRUPT_POINTER":
            raise CorruptAuthoritativeRecordError(
                f"Continuation latest pointer for parent task '{continuation.parent_task_id}' "
                f"is corrupt or inconsistent with version {continuation.continuation_version}"
            )
        if result == "CONFLICT":
            raise ConflictingContinuationError(
                f"TaskContinuation for parent task '{continuation.parent_task_id}' "
                f"version {continuation.continuation_version} already exists with conflicting data"
            )
        raise RuntimeError(f"Unexpected Redis save_continuation result: {result}")

    async def get_continuation(
        self, parent_task_id: str, continuation_version: int = 1
    ) -> TaskContinuation | None:
        clean_parent_task_id = str(parent_task_id).strip()
        if not clean_parent_task_id:
            return None
        key = f"task.continuation:{clean_parent_task_id}:{continuation_version}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise TypeError("Payload must be a JSON object")
            return TaskContinuation.from_dict(data)
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt TaskContinuation record at key '{key}': {exc}"
            ) from exc

    async def save_root_authority(self, authority: RootAuthorityRecord) -> None:
        target_key = f"task.authority:{authority.root_task_id}"
        payload = authority.to_json()
        raw_result = await self._client.eval(
            SAVE_ROOT_AUTHORITY_LUA,
            1,
            target_key,
            payload,
        )
        result = (
            raw_result.decode("utf-8")
            if isinstance(raw_result, (bytes, bytearray))
            else str(raw_result)
        )
        if result in ("OK", "IDEMPOTENT"):
            return
        if result == "CONFLICT":
            raise ConflictingRootAuthorityError(
                f"RootAuthorityRecord for root task '{authority.root_task_id}' "
                "already exists with conflicting data"
            )
        raise RuntimeError(f"Unexpected Redis save_root_authority result: {result}")

    async def get_root_authority(self, root_task_id: str) -> RootAuthorityRecord | None:
        clean_root_task_id = str(root_task_id).strip()
        if not clean_root_task_id:
            return None
        key = f"task.authority:{clean_root_task_id}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise TypeError("Payload must be a JSON object")
            return RootAuthorityRecord.from_dict(data)
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt RootAuthorityRecord at key '{key}': {exc}"
            ) from exc

    async def dispatch(self, task: TaskRequested, target: str) -> None:
        """Atomically enqueue task payload and establish pending status.

        In accordance with ADR-0006:
        - RPUSH of TaskRequested and transition of task.status to 'pending' commit together
          in an atomic transaction pipeline (transaction=True).
        - If the transaction fails, neither the queue insertion nor 'pending' status is committed.
        - Failure handling and transition to 'dispatch_failed' is coordinated by the application use case.
        """
        if not target or not isinstance(target, str) or not target.strip():
            raise ValueError("Dispatch target must be a non-empty string")
        clean_target = target.strip()
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.rpush(clean_target, task.to_json())
            pipe.set(
                f"task.status:{task.task_id}",
                json.dumps({"task_id": task.task_id, "status": "pending"}),
            )
            await pipe.execute()

    async def enqueue(
        self,
        task: TaskRequested,
        capability: AgentCapability = AgentCapability.GENERAL,
        target: str | None = None,
    ) -> None:
        queue_name = target or CAPABILITY_QUEUE_MAP.get(capability)
        if not queue_name:
            raise ValueError(f"Unknown capability: {capability}")
        await self.dispatch(task, queue_name)

    async def get_result(self, task_id: str) -> dict[str, Any] | None:
        raw_result = await self._client.get(f"task.result:{task_id}")
        if raw_result is not None:
            return json.loads(raw_result)

        raw_status = await self._client.get(f"task.status:{task_id}")
        if raw_status is not None:
            status_data = json.loads(raw_status)
            return {
                "task_id": task_id,
                "status": status_data.get("status", "pending"),
                "output": None,
            }

        events = await self.events(limit=100)
        matching = [e for e in events if e.get("task_id") == task_id and e.get("state") is not None]
        if matching:
            latest = matching[0]
            state = latest.get("state", "pending")
            return {
                "task_id": task_id,
                "status": "pending" if state == "working" else state,
                "output": None,
            }

        return None

    async def agents(self) -> list[dict[str, Any]]:
        ids = await self._client.smembers("agent.ids")
        agents_list: list[dict[str, Any]] = []
        for agent_id in sorted(ids):
            data = await self._client.hgetall(f"agent.state:{agent_id}")
            if not data:
                continue
            entry: dict[str, Any] = dict(data)

            if "capabilities" in entry and isinstance(entry["capabilities"], str):
                try:
                    entry["capabilities"] = json.loads(entry["capabilities"])
                except (json.JSONDecodeError, TypeError):
                    entry["capabilities"] = [entry["capabilities"]]
            elif "capabilities" not in entry:
                entry["capabilities"] = []

            if "providers" in entry and isinstance(entry["providers"], str):
                try:
                    entry["providers"] = json.loads(entry["providers"])
                except (json.JSONDecodeError, TypeError):
                    pass
            elif "providers" not in entry:
                entry["providers"] = {}

            agents_list.append(entry)
        return agents_list

    async def events(self, limit: int = 50) -> list[dict[str, str]]:
        rows = await self._client.xrevrange("agent.events", count=limit)
        return [dict(fields, event_id=event_id) for event_id, fields in rows]

    async def record_routing_decision(self, decision: RoutingDecision) -> None:
        await self._client.set(
            f"task.routing:{decision.task_id}",
            decision.to_json(),
        )

    async def routing_decision(self, task_id: str) -> dict[str, Any] | None:
        raw_decision = await self._client.get(f"task.routing:{task_id}")
        if raw_decision is not None:
            try:
                return json.loads(raw_decision)
            except (json.JSONDecodeError, TypeError):
                return None

        events = await self.events(limit=100)
        for e in events:
            if e.get("task_id") == task_id and e.get("routing_decision"):
                raw_rd = e["routing_decision"]
                if isinstance(raw_rd, dict):
                    return raw_rd
                try:
                    return json.loads(raw_rd)
                except (json.JSONDecodeError, TypeError):
                    pass

        return None

    async def task_inference_telemetry(self, task_id: str) -> dict[str, Any] | None:
        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            return None

        # Query durable evidence if available
        try:
            raw_result = await self._client.get(f"task.result:{clean_task_id}")
            raw_status = await self._client.get(f"task.status:{clean_task_id}")
            raw_routing = await self._client.get(f"task.routing:{clean_task_id}")
        except (redis.RedisError, ConnectionError, OSError) as exc:
            logger.error("Failed querying durable keys for task %s: %s", clean_task_id, exc.__class__.__name__)
            raise TelemetryStorageError(
                f"Task durable state storage unavailable: {exc.__class__.__name__}"
            ) from exc

        task_completed = raw_result is not None
        task_exists_durable = task_completed or (raw_status is not None) or (raw_routing is not None)
        task_failed = False
        if raw_status is not None:
            try:
                status_data = json.loads(raw_status)
                if isinstance(status_data, dict) and status_data.get("status") in ("error", "failed"):
                    task_failed = True
            except (json.JSONDecodeError, TypeError):
                pass

        try:
            rows = await self._client.xrevrange("agent.events")
        except (redis.RedisError, ConnectionError, OSError) as exc:
            logger.error("Failed querying agent.events stream for task %s: %s", clean_task_id, exc.__class__.__name__)
            raise TelemetryStorageError(
                f"Event stream storage unavailable: {exc.__class__.__name__}"
            ) from exc

        chronological_rows = list(reversed(rows))
        observations: list[TaskEventObservation] = []
        for stream_order, (raw_event_id, raw_fields) in enumerate(chronological_rows):
            event_id = (
                raw_event_id.decode("utf-8")
                if isinstance(raw_event_id, (bytes, bytearray))
                else str(raw_event_id)
            )
            fields: dict[str, Any] = {}
            for k, v in raw_fields.items():
                key_str = k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)
                val_str = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
                fields[key_str] = val_str

            if fields.get("task_id") != clean_task_id:
                continue

            state = fields.get("state")
            is_terminal_lifecycle = state in ("idle", "completed")
            is_terminal_error = state == "error"

            is_malformed = False
            parsed_telem: InferenceExecutionTelemetry | None = None
            raw_telem = fields.get("inference_telemetry")
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
            raw_trans = fields.get("fallback_transition")
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
                    stream_position=_parse_redis_stream_id(event_id),
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

    async def publish(self, event: AgentEvent) -> None:
        mapping = {}
        for key, value in event.to_dict().items():
            if value is None:
                mapping[key] = ""
            elif isinstance(value, (dict, list)):
                mapping[key] = json.dumps(value)
            else:
                mapping[key] = str(value)
        await self._client.xadd("agent.events", mapping, maxlen=500, approximate=True)

    async def aclose(self) -> None:
        await self._client.aclose()
