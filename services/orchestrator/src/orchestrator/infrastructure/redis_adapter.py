import json
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import redis.asyncio as redis
from agent_contracts import (
    ActiveClaim,
    AgentEvent,
    DelegationGrant,
    FallbackTransition,
    InferenceExecutionTelemetry,
    RootAuthorityRecord,
    RoutingDecision,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskRequested,
    WakeUpHint,
    canonical_json_dumps,
    normalize_canonical_utc_timestamp,
)

from orchestrator.application.inference_history import (
    TaskEventObservation,
    project_task_inference_history,
)
from orchestrator.application.ports import (
    ClaimDeliveryLeaseExpiredError,
    ClaimNotFoundError,
    TaskAlreadyClaimedError,
    TaskNotClaimableError,
    TelemetryStorageError,
)
from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.task import (
    ConflictingAuthorityGrantError,
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

SAVE_AUTHORITY_GRANT_LUA = """
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

CLAIM_TASK_LUA = """
local status_key = KEYS[1]
local claim_key = KEYS[2]
local fence_key = KEYS[3]
local grant_key = KEYS[4]

local task_id = ARGV[1]
local now_iso = ARGV[2]
local principal_id = ARGV[3]
local request_id = ARGV[4]
local working_status_json = ARGV[5]
local pending_status_json = ARGV[6]
local claim_payload = ARGV[7]
local grant_payload = ARGV[8]
local expected_epoch = tonumber(ARGV[9])

-- =============================================================================
-- PHASE 1: Reads and Validations (Strictly Read-Only, Zero Mutations)
-- =============================================================================

-- 1. Status key exists
if redis.call('EXISTS', status_key) == 0 then
    return {'NOT_FOUND'}
end

-- 2. Status JSON is valid and structurally complete
local cur_status_raw = redis.call('GET', status_key)
local ok_s, status_tbl = pcall(cjson.decode, cur_status_raw)
if not ok_s or type(status_tbl) ~= 'table' then
    return {'CORRUPT_STATUS'}
end
local cur_status = status_tbl['status']
if type(cur_status) ~= 'string' or cur_status == '' then
    return {'CORRUPT_STATUS'}
end

-- 3. Existing claim JSON is valid and structurally complete
local effective_status = cur_status
local has_expired_claim = false

if redis.call('EXISTS', claim_key) == 1 then
    local cur_claim_raw = redis.call('GET', claim_key)
    local ok_c, cur_claim = pcall(cjson.decode, cur_claim_raw)
    if not ok_c or type(cur_claim) ~= 'table' then
        return {'CORRUPT_CLAIM'}
    end

    local c_principal_id = cur_claim['runtime_principal_id']
    local c_request_id = cur_claim['request_id']
    local c_assignment_id = cur_claim['assignment_id']
    local c_state = cur_claim['state']
    local c_expires_at = cur_claim['delivery_expires_at']
    local c_fencing_epoch = cur_claim['fencing_epoch']

    if type(c_principal_id) ~= 'string' or c_principal_id == '' then
        return {'CORRUPT_CLAIM'}
    end
    if type(c_request_id) ~= 'string' or c_request_id == '' then
        return {'CORRUPT_CLAIM'}
    end
    if type(c_assignment_id) ~= 'string' or c_assignment_id == '' then
        return {'CORRUPT_CLAIM'}
    end
    if type(c_expires_at) ~= 'string' or c_expires_at == '' then
        return {'CORRUPT_CLAIM'}
    end
    if type(c_fencing_epoch) ~= 'number' or c_fencing_epoch < 1 or math.floor(c_fencing_epoch) ~= c_fencing_epoch then
        return {'CORRUPT_CLAIM'}
    end
    if type(c_state) ~= 'string' or (c_state ~= 'claimed_unacknowledged' and c_state ~= 'acknowledged') then
        return {'CORRUPT_CLAIM'}
    end

    if c_principal_id == principal_id and c_request_id == request_id then
        if c_state == 'acknowledged' then
            return {'ALREADY_ACKNOWLEDGED', cur_claim_raw}
        elseif c_state == 'claimed_unacknowledged' then
            if c_expires_at > now_iso then
                -- Active unacknowledged delivery lease: idempotent replay
                return {'REPLAY', cur_claim_raw}
            else
                has_expired_claim = true
                effective_status = 'pending'
            end
        end
    else
        -- Different principal or different request_id
        if c_state == 'claimed_unacknowledged' and c_expires_at <= now_iso then
            has_expired_claim = true
            effective_status = 'pending'
        else
            return {'ALREADY_CLAIMED', cur_claim_raw}
        end
    end
end

-- 4. Effective status is claimable
if effective_status ~= 'pending' and effective_status ~= 'resumable' then
    return {'NOT_CLAIMABLE', effective_status}
end

-- 5. Fence is absent or a valid nonnegative integer
local cur_fence = 0
if redis.call('EXISTS', fence_key) == 1 then
    local fence_raw = redis.call('GET', fence_key)
    cur_fence = tonumber(fence_raw)
    if not cur_fence or cur_fence < 0 or math.floor(cur_fence) ~= cur_fence then
        return {'CORRUPT_FENCE'}
    end
end

-- 6. Expected epoch equals current fence plus one
if not expected_epoch or expected_epoch < 1 or math.floor(expected_epoch) ~= expected_epoch then
    return {'CORRUPT_FENCE'}
end

if cur_fence + 1 ~= expected_epoch then
    return {'EPOCH_RACE', tostring(cur_fence)}
end

-- 7. Grant key and payload pairing: both absent/empty or both present/nonempty
local has_grant_key = (grant_key ~= nil and grant_key ~= '')
local has_grant_payload = (grant_payload ~= nil and grant_payload ~= '')

if has_grant_key ~= has_grant_payload then
    return {'CORRUPT_GRANT_INPUT'}
end
local has_grant = has_grant_key and has_grant_payload

-- 8. Existing grant is absent or byte-identical to candidate grant
if has_grant then
    if redis.call('EXISTS', grant_key) == 1 then
        local existing_grant = redis.call('GET', grant_key)
        if existing_grant ~= grant_payload then
            return {'CONFLICT_GRANT'}
        end
    end
end

-- 9. Candidate inputs are valid enough for required invariants
if type(claim_payload) ~= 'string' or claim_payload == '' then
    return {'CORRUPT_CLAIM_INPUT'}
end
if type(working_status_json) ~= 'string' or working_status_json == '' then
    return {'CORRUPT_STATUS_INPUT'}
end

-- =============================================================================
-- PHASE 2: State Commit (Atomic Mutations; No Failures Permitted After First Mutation)
-- =============================================================================

-- 1. Advance fencing epoch (First mutating command in successful path)
redis.call('SET', fence_key, tostring(expected_epoch))

-- 2. Transition task status to working
redis.call('SET', status_key, working_status_json)

-- 3. Store active claim (replaces any previous claim, expired or otherwise)
redis.call('SET', claim_key, claim_payload)

-- 4. Store authority grant if present
if has_grant then
    redis.call('SET', grant_key, grant_payload)
end

return {'OK', tostring(expected_epoch)}
"""

ACKNOWLEDGE_CLAIM_LUA = """
local claim_key = KEYS[1]
local assignment_id = ARGV[1]
local principal_id = ARGV[2]
local now_iso = ARGV[3]

if redis.call('EXISTS', claim_key) == 0 then
    return {'CLAIM_NOT_FOUND'}
end

local cur_claim_raw = redis.call('GET', claim_key)
local ok, cur_claim = pcall(cjson.decode, cur_claim_raw)
if not ok or type(cur_claim) ~= 'table' then
    return {'CORRUPT_CLAIM'}
end

local c_assignment_id = cur_claim['assignment_id']
local c_principal_id = cur_claim['runtime_principal_id']
local c_state = cur_claim['state']
local c_expires_at = cur_claim['delivery_expires_at']
local c_fencing_epoch = cur_claim['fencing_epoch']

if type(c_assignment_id) ~= 'string' or c_assignment_id == '' then
    return {'CORRUPT_CLAIM'}
end
if type(c_principal_id) ~= 'string' or c_principal_id == '' then
    return {'CORRUPT_CLAIM'}
end
if type(c_state) ~= 'string' or (c_state ~= 'claimed_unacknowledged' and c_state ~= 'acknowledged') then
    return {'CORRUPT_CLAIM'}
end
if type(c_expires_at) ~= 'string' or c_expires_at == '' then
    return {'CORRUPT_CLAIM'}
end
if type(c_fencing_epoch) ~= 'number' or c_fencing_epoch < 1 or math.floor(c_fencing_epoch) ~= c_fencing_epoch then
    return {'CORRUPT_CLAIM'}
end

if c_assignment_id ~= assignment_id or c_principal_id ~= principal_id then
    return {'MISMATCH'}
end

if c_state == 'acknowledged' then
    return {'IDEMPOTENT', cur_claim_raw}
end

if c_expires_at <= now_iso then
    return {'LEASE_EXPIRED'}
end

-- State commit (Single mutating command)
cur_claim['state'] = 'acknowledged'
cur_claim['acknowledged_at'] = now_iso
local updated_raw = cjson.encode(cur_claim)
redis.call('SET', claim_key, updated_raw)

return {'OK', updated_raw}
"""

EXPIRE_DELIVERY_LEASE_LUA = """
local claim_key = KEYS[1]
local status_key = KEYS[2]
local now_iso = ARGV[1]
local pending_status_json = ARGV[2]

if redis.call('EXISTS', claim_key) == 0 then
    return {'NOOP'}
end

local cur_claim_raw = redis.call('GET', claim_key)
local ok, cur_claim = pcall(cjson.decode, cur_claim_raw)
if not ok or type(cur_claim) ~= 'table' then
    return {'CORRUPT_CLAIM'}
end

local c_state = cur_claim['state']
local c_expires_at = cur_claim['delivery_expires_at']

if type(c_state) ~= 'string' or (c_state ~= 'claimed_unacknowledged' and c_state ~= 'acknowledged') then
    return {'CORRUPT_CLAIM'}
end

if type(c_expires_at) ~= 'string' or c_expires_at == '' then
    return {'CORRUPT_CLAIM'}
end

if c_state == 'acknowledged' then
    return {'NOOP'}
end

if c_state == 'claimed_unacknowledged' then
    if c_expires_at > now_iso then
        return {'NOOP'}
    end

    -- Lease has expired: atomically delete claim and restore status to pending
    redis.call('DEL', claim_key)
    redis.call('SET', status_key, pending_status_json)
    return {'EXPIRED'}
end

return {'CORRUPT_CLAIM'}
"""


class RedisGateway:
    def __init__(self, url: str, delivery_mode: str = "legacy_push") -> None:
        self._client = redis.from_url(url, decode_responses=True)
        self._delivery_mode = delivery_mode

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

    async def save_authority_grant(self, grant: DelegationGrant) -> None:
        target_key = f"task.authority_grant:{grant.root_task_id}:{grant.grant_id}"
        payload = grant.to_json()
        raw_result = await self._client.eval(
            SAVE_AUTHORITY_GRANT_LUA,
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
            raise ConflictingAuthorityGrantError(
                f"DelegationGrant for root task '{grant.root_task_id}' and grant '{grant.grant_id}' "
                "already exists with conflicting data"
            )
        raise RuntimeError(f"Unexpected Redis save_authority_grant result: {result}")

    async def get_authority_grant(
        self, root_task_id: str, grant_id: str
    ) -> DelegationGrant | None:
        clean_root_task_id = str(root_task_id).strip()
        clean_grant_id = str(grant_id).strip()
        if not clean_root_task_id or not clean_grant_id:
            return None
        key = f"task.authority_grant:{clean_root_task_id}:{clean_grant_id}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise TypeError("Payload must be a JSON object")
            return DelegationGrant.from_dict(data)
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt DelegationGrant at key '{key}': {exc}"
            ) from exc

    async def save_active_claim(self, claim: ActiveClaim) -> None:
        key = f"task.claim:{claim.task_id}"
        await self._client.set(key, claim.to_json())

    async def get_active_claim(self, task_id: str) -> ActiveClaim | None:
        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            return None
        key = f"task.claim:{clean_task_id}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise TypeError("Payload must be a JSON object")
            return ActiveClaim.from_dict(data)
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt ActiveClaim at key '{key}': {exc}"
            ) from exc

    async def get_fencing_epoch(self, task_id: str) -> int:
        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            return 0
        raw = await self._client.get(f"task.fence:{clean_task_id}")
        if raw is None:
            return 0
        try:
            val = int(raw if isinstance(raw, str) else raw.decode("utf-8"))
            return val
        except (TypeError, ValueError) as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt fencing epoch at key 'task.fence:{clean_task_id}': {exc}"
            ) from exc

    async def atomic_claim_task(
        self,
        task_id: str,
        claim: ActiveClaim,
        grant: DelegationGrant | None = None,
        now: str = "",
    ) -> tuple[ActiveClaim, bool]:
        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            raise ValueError("task_id must be a non-empty string")
        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )
        status_key = f"task.status:{clean_task_id}"
        claim_key = f"task.claim:{clean_task_id}"
        fence_key = f"task.fence:{clean_task_id}"
        grant_key = f"task.authority_grant:{claim.root_task_id}:{grant.grant_id}" if grant else ""

        claim_payload = claim.to_json()
        grant_payload = grant.to_json() if grant else ""

        working_status_json = canonical_json_dumps({
            "status": "working",
            "task_id": clean_task_id,
        })
        pending_status_json = canonical_json_dumps({
            "status": "pending",
            "task_id": clean_task_id,
        })

        keys = [status_key, claim_key, fence_key, grant_key or ""]

        raw_res = await self._client.eval(
            CLAIM_TASK_LUA,
            4,
            *keys,
            clean_task_id,
            now_iso,
            claim.runtime_principal_id,
            claim.request_id,
            working_status_json,
            pending_status_json,
            claim_payload,
            grant_payload,
            str(claim.fencing_epoch),
        )

        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(f"Unexpected Redis claim_task response: {raw_res}")

        code = raw_res[0] if isinstance(raw_res[0], str) else raw_res[0].decode("utf-8")

        if code == "OK":
            return (claim, False)

        if code == "REPLAY":
            replay_raw = raw_res[1] if isinstance(raw_res[1], str) else raw_res[1].decode("utf-8")
            existing_claim = ActiveClaim.from_json(replay_raw)
            return (existing_claim, True)

        if code == "ALREADY_ACKNOWLEDGED":
            raise TaskAlreadyClaimedError(f"Task '{clean_task_id}' assignment is already acknowledged")

        if code == "NOT_FOUND":
            raise TaskNotClaimableError(f"Task '{clean_task_id}' not found")

        if code == "NOT_CLAIMABLE":
            cur_status = raw_res[1] if len(raw_res) > 1 else "unknown"
            if not isinstance(cur_status, str):
                cur_status = cur_status.decode("utf-8")
            raise TaskNotClaimableError(f"Task '{clean_task_id}' is in non-claimable status '{cur_status}'")

        if code == "ALREADY_CLAIMED":
            raise TaskAlreadyClaimedError(f"Task '{clean_task_id}' has an active claim by another runtime")

        if code == "CONFLICT_GRANT":
            raise ConflictingAuthorityGrantError(
                f"DelegationGrant for root task '{claim.root_task_id}' and grant '{grant.grant_id if grant else ''}' "
                "already exists with conflicting data"
            )

        if code == "CORRUPT_GRANT_INPUT":
            raise CorruptAuthoritativeRecordError(
                f"Inconsistent authority grant inputs provided for task '{clean_task_id}'"
            )

        if code in ("CORRUPT_CLAIM_INPUT", "CORRUPT_STATUS_INPUT"):
            raise CorruptAuthoritativeRecordError(
                f"Invalid candidate payload inputs provided for task '{clean_task_id}': {code}"
            )

        if code == "EPOCH_RACE":
            raise TaskAlreadyClaimedError(f"Concurrent claim conflict on task '{clean_task_id}' (fencing epoch race)")

        if code in ("CORRUPT_CLAIM", "CORRUPT_FENCE", "CORRUPT_STATUS"):
            raise CorruptAuthoritativeRecordError(f"Corrupt authoritative record found on task '{clean_task_id}': {code}")

        raise RuntimeError(f"Unexpected claim result code: {code}")

    async def acknowledge_claim(
        self,
        task_id: str,
        assignment_id: str,
        principal_id: str,
        acknowledged_at: str,
    ) -> ActiveClaim:
        clean_task_id = str(task_id).strip()
        clean_assignment_id = str(assignment_id).strip()
        clean_principal_id = str(principal_id).strip()
        clean_time = normalize_canonical_utc_timestamp(acknowledged_at)

        claim_key = f"task.claim:{clean_task_id}"

        raw_res = await self._client.eval(
            ACKNOWLEDGE_CLAIM_LUA,
            1,
            claim_key,
            clean_assignment_id,
            clean_principal_id,
            clean_time,
        )
        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(f"Unexpected Redis acknowledge_claim response: {raw_res}")

        code = raw_res[0] if isinstance(raw_res[0], str) else raw_res[0].decode("utf-8")

        if code in ("OK", "IDEMPOTENT"):
            raw_json = raw_res[1] if isinstance(raw_res[1], str) else raw_res[1].decode("utf-8")
            return ActiveClaim.from_json(raw_json)

        if code == "CLAIM_NOT_FOUND":
            raise ClaimNotFoundError(f"Active claim for task '{clean_task_id}' not found")

        if code == "MISMATCH":
            raise ClaimNotFoundError(
                f"Active claim does not match assignment '{clean_assignment_id}' or principal '{clean_principal_id}'"
            )

        if code == "LEASE_EXPIRED":
            raise ClaimDeliveryLeaseExpiredError(f"Delivery lease for assignment '{clean_assignment_id}' has expired")

        if code == "CORRUPT_CLAIM":
            raise CorruptAuthoritativeRecordError(
                f"Corrupt ActiveClaim at key '{claim_key}': malformed claim record during acknowledge"
            )

        raise RuntimeError(f"Unexpected acknowledge result code: {code}")

    async def expire_delivery_lease(self, task_id: str, now: str = "") -> bool:
        clean_task_id = str(task_id).strip()
        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )
        claim_key = f"task.claim:{clean_task_id}"
        status_key = f"task.status:{clean_task_id}"
        pending_status_json = canonical_json_dumps({
            "status": "pending",
            "task_id": clean_task_id,
        })
        res = await self._client.eval(
            EXPIRE_DELIVERY_LEASE_LUA,
            2,
            claim_key,
            status_key,
            now_iso,
            pending_status_json,
        )
        if not isinstance(res, (list, tuple)) or len(res) == 0:
            if res == 1:
                return True
            if res == 0:
                return False
            raise RuntimeError(f"Unexpected Redis expire_delivery_lease response: {res}")

        code = res[0] if isinstance(res[0], str) else res[0].decode("utf-8")
        if code in ("EXPIRED", 1, "1"):
            return True
        if code in ("NOOP", 0, "0"):
            return False
        if code == "CORRUPT_CLAIM":
            raise CorruptAuthoritativeRecordError(
                f"Corrupt active claim on task '{clean_task_id}' during lease expiry"
            )

        raise RuntimeError(f"Unexpected expire_delivery_lease result code: {code}")

    async def dispatch_hint(self, hint: WakeUpHint, target: str) -> None:
        clean_target = str(target).strip()
        if not clean_target:
            raise ValueError("target queue must be a non-empty string")
        await self._client.rpush(clean_target, hint.to_json())

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
            if self._delivery_mode == "authenticated_claim":
                capability = (
                    clean_target.removeprefix("agent.tasks.")
                    if clean_target.startswith("agent.tasks.")
                    else clean_target
                )
                hint_target = f"agent.hints.{capability}"
                hint = WakeUpHint(
                    hint_id=str(uuid4()),
                    capability=capability,
                    created_at=datetime.now(UTC).isoformat(),
                    execution_kind="initial",
                    task_id=task.task_id,
                )
                pipe.rpush(hint_target, hint.to_json())
            else:
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
