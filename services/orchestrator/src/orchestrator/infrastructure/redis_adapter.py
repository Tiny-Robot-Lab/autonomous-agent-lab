import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import redis.asyncio as redis
from agent_contracts import (
    CONTINUATION_V2_STATE_CLOSED,
    CONTINUATION_V2_STATE_RELEASED,
    MAX_WAVE_CHILDREN,
    MIN_WAVE_CHILDREN,
    AcceptedDelegationWave,
    ActiveClaim,
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
    FallbackTransition,
    InferenceExecutionTelemetry,
    RecoveryDecisionCommand,
    RecoveryDecisionRecord,
    RedisConnectionConfig,
    RequiredCapabilityRecord,
    ResolvedDelegationWave,
    ResumedContinuationContext,
    RootAuthorityRecord,
    RoutingDecision,
    StageArtifact,
    TaskContinuation,
    TaskContinuationV2,
    TaskInput,
    TaskLifecycleStatus,
    TaskRequested,
    TaskResult,
    TerminalOutcome,
    WakeUpHint,
    WaveCancellationCommand,
    WaveCancellationRecord,
    WaveResolution,
    WaveTimeoutEvaluationCommand,
    WaveYieldRecord,
    build_claim_timed_out_terminal_outcome,
    build_completion_audit,
    build_terminal_outcome,
    canonical_json_dumps,
    deserialize_continuation,
    deserialize_delegation_grant,
    normalize_canonical_utc_timestamp,
    resolve_client_connection_kwargs,
)
from redis.exceptions import AuthenticationError, NoPermissionError, ResponseError

from orchestrator.application.inference_history import (
    TaskEventObservation,
    project_task_inference_history,
)
from orchestrator.application.ports import (
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
    TaskNotClaimableError,
    TelemetryStorageError,
    WaveAlreadyResolvedError,
    WaveBudgetExceededError,
    WaveCancellationConflictError,
    WaveDeadlineNotReachedError,
)
from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.task import (
    ConflictingAuthorityGrantError,
    ConflictingChildTaskRegistrationError,
    ConflictingContinuationError,
    ConflictingRootAuthorityError,
    ConflictingRootTaskRegistrationError,
    ConflictingStageArtifactError,
    ConflictingTaskInputError,
    CorruptAuthoritativeRecordError,
    NonContiguousVersionError,
    ParentTaskNotFoundError,
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

LOCAL_COMPOSE_CAPABILITY_QUEUE_MAP: dict[str, str] = {
    "general": "agent.tasks.general",
    "research": "agent.tasks.research",
    "analysis": "agent.tasks.analysis",
}
LOCAL_COMPOSE_CAPABILITY_HINT_MAP: dict[str, str] = {
    "general": "agent.hints.general",
    "research": "agent.hints.research",
    "analysis": "agent.hints.analysis",
}
CAPABILITY_QUEUE_MAP: dict[AgentCapability, str] = {
    AgentCapability.GENERAL: LOCAL_COMPOSE_CAPABILITY_QUEUE_MAP["general"],
    AgentCapability.RESEARCH: LOCAL_COMPOSE_CAPABILITY_QUEUE_MAP["research"],
}
RECOGNIZED_CAPABILITIES: frozenset[str] = frozenset({"general", "research", "analysis"})


def _validate_capability_contract(contract: dict[str, str], *, kind: str) -> dict[str, str]:
    """Validate an explicitly finite capability-to-destination contract.

    Queue names must be exact: no globs, control characters, or empty
    entries. The contract is never sourced from runtime-writable Redis
    state, and there is no production validation-disable mode.
    """
    if not contract:
        raise ValueError(f"Capability-to-{kind} contract must be finite and non-empty")
    for contract_capability, queue in contract.items():
        if not isinstance(contract_capability, str) or not contract_capability:
            raise ValueError("Capability contract keys must be non-empty strings")
        if not isinstance(queue, str) or not queue:
            raise ValueError("Capability contract destinations must be non-empty strings")
        if any(ord(character) < 32 or ord(character) == 127 for character in queue):
            raise ValueError("Capability contract destinations must not contain control characters")
        if any(glob_character in queue for glob_character in "*?[\\"):
            raise ValueError("Capability contract destinations must be exact queue names")
    return contract

REGISTER_ROOT_TASK_LUA = """
local marker_key = KEYS[1]
local status_key = KEYS[2]
local lineage_key = KEYS[3]
local input_key = KEYS[4]
local authority_key = KEYS[5]
local capability_key = KEYS[6]

local marker_payload = ARGV[1]
local initial_status_json = ARGV[2]
local lineage_json = ARGV[3]
local input_json = ARGV[4]
local authority_json = ARGV[5]
local expected_task_id = ARGV[6]
local capability_json = ARGV[7]

local exists_marker = redis.call('EXISTS', marker_key)
local exists_status = redis.call('EXISTS', status_key)
local exists_lineage = redis.call('EXISTS', lineage_key)
local exists_input = redis.call('EXISTS', input_key)
local exists_authority = redis.call('EXISTS', authority_key)
local exists_capability = redis.call('EXISTS', capability_key)

if exists_marker == 0 and exists_status == 0 and exists_lineage == 0 and exists_input == 0 and exists_authority == 0 and exists_capability == 0 then
    redis.call('SET', marker_key, marker_payload)
    redis.call('SET', status_key, initial_status_json)
    redis.call('SET', lineage_key, lineage_json)
    redis.call('SET', input_key, input_json)
    redis.call('SET', authority_key, authority_json)
    redis.call('SET', capability_key, capability_json)
    return 'OK'
end

if exists_marker == 1 and exists_status == 1 and exists_lineage == 1 and exists_input == 1 and exists_authority == 1 and exists_capability == 1 then
    local cur_marker = redis.call('GET', marker_key)
    local cur_lineage = redis.call('GET', lineage_key)
    local cur_input = redis.call('GET', input_key)
    local cur_authority = redis.call('GET', authority_key)
    local cur_capability = redis.call('GET', capability_key)
    if cur_marker == marker_payload and cur_lineage == lineage_json and cur_input == input_json and cur_authority == authority_json and cur_capability == capability_json then
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

REGISTER_CHILD_TASK_LUA = """
local status_key = KEYS[1]
local lineage_key = KEYS[2]
local capability_key = KEYS[3]
local parent_children_key = KEYS[4]

local initial_status_json = ARGV[1]
local lineage_json = ARGV[2]
local capability_json = ARGV[3]
local expected_task_id = ARGV[4]

local exists_status = redis.call('EXISTS', status_key)
local exists_lineage = redis.call('EXISTS', lineage_key)
local exists_capability = redis.call('EXISTS', capability_key)
local is_member = redis.call('SISMEMBER', parent_children_key, expected_task_id)

-- 1. All records absent: create all records atomically
if exists_status == 0 and exists_lineage == 0 and exists_capability == 0 and is_member == 0 then
    redis.call('SET', status_key, initial_status_json)
    redis.call('SET', lineage_key, lineage_json)
    redis.call('SET', capability_key, capability_json)
    redis.call('SADD', parent_children_key, expected_task_id)
    return 'OK'
end

-- 2. All records present and byte-identical: idempotent success
if exists_status == 1 and exists_lineage == 1 and exists_capability == 1 and is_member == 1 then
    local cur_lineage = redis.call('GET', lineage_key)
    local cur_capability = redis.call('GET', capability_key)
    if cur_lineage == lineage_json and cur_capability == capability_json then
        local cur_status = redis.call('GET', status_key)
        local ok, status_tbl = pcall(cjson.decode, cur_status)
        if ok and type(status_tbl) == 'table' and status_tbl['task_id'] == expected_task_id and type(status_tbl['status']) == 'string' then
            return 'IDEMPOTENT'
        end
    end
    return 'CONFLICT'
end

-- 3. Partial existing registration: fail closed
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
local claim_key = KEYS[3]
local fence_key = KEYS[4]
local version = tonumber(ARGV[1])
local payload = ARGV[2]
local index_member = ARGV[3]
local expected_task_id = ARGV[4]
local expected_assignment_id = ARGV[5]
local expected_principal_id = ARGV[6]
local expected_epoch = tonumber(ARGV[7])
local artifact_type = ARGV[8]

-- Phase 1: Validation
if expected_assignment_id == '' or expected_principal_id == '' or (not expected_epoch) or expected_epoch < 1 then
    return 'INVALID_CLAIM_INPUT'
end

if redis.call('EXISTS', claim_key) == 0 then
    return 'CLAIM_NOT_FOUND'
end

local claim_raw = redis.call('GET', claim_key)
local ok_c, claim = pcall(cjson.decode, claim_raw)
if not ok_c or type(claim) ~= 'table' then
    return 'CORRUPT_CLAIM'
end

if claim['task_id'] ~= expected_task_id then
    return 'CLAIM_TASK_MISMATCH'
end

if claim['assignment_id'] ~= expected_assignment_id then
    return 'ASSIGNMENT_MISMATCH'
end

if claim['runtime_principal_id'] ~= expected_principal_id then
    return 'PRINCIPAL_MISMATCH'
end

if claim['state'] ~= 'acknowledged' then
    return 'CLAIM_NOT_ACKNOWLEDGED'
end

local c_epoch = tonumber(claim['fencing_epoch'])
if not c_epoch or c_epoch ~= expected_epoch then
    return 'FENCE_MISMATCH'
end

if redis.call('EXISTS', fence_key) == 0 then
    return 'FENCE_NOT_FOUND'
end

local cur_fence = tonumber(redis.call('GET', fence_key))
if not cur_fence or cur_fence ~= expected_epoch then
    return 'STALE_FENCE'
end

if artifact_type == 'research_material' then
    if claim['assigned_capability'] ~= 'research' or claim['execution_kind'] ~= 'initial' then
        return 'CAPABILITY_POLICY_VIOLATION'
    end
else
    return 'CAPABILITY_POLICY_VIOLATION'
end

if version > 1 then
    local prior_key = KEYS[5]
    if (not prior_key or prior_key == '') or redis.call('EXISTS', prior_key) == 0 then
        return 'NON_CONTIGUOUS'
    end
end

if redis.call('EXISTS', target_key) == 1 then
    local existing = redis.call('GET', target_key)
    if existing == payload then
        redis.call('SADD', index_key, index_member)
        return 'IDEMPOTENT'
    else
        return 'CONFLICT'
    end
end

-- Phase 2: Mutation
redis.call('SET', target_key, payload)
redis.call('SADD', index_key, index_member)
return 'OK'
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
local capability_key = KEYS[4]
local continuation_latest_key = KEYS[5]
local authority_key = KEYS[6]
local deadline_key = KEYS[7]
local grant_key = KEYS[8]
local continuation_key = KEYS[9]
local wave_resolution_key = KEYS[10]

local task_id = ARGV[1]
local now_iso = ARGV[2]
local principal_id = ARGV[3]
local request_id = ARGV[4]
local working_status_json = ARGV[5]
local claim_payload = ARGV[6]
local grant_payload = ARGV[7]
local expected_epoch = tonumber(ARGV[8])
local requested_capability = ARGV[9]
local execution_kind = ARGV[10]
local is_root = ARGV[11]
local exp_cont_ver = tonumber(ARGV[12])

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

-- 3. Authoritative required capability key exists, is valid, and matches requested
if redis.call('EXISTS', capability_key) == 0 then
    return {'MISSING_REQUIRED_CAPABILITY'}
end
local cur_cap_raw = redis.call('GET', capability_key)
local ok_cap, cap_tbl = pcall(cjson.decode, cur_cap_raw)
if not ok_cap or type(cap_tbl) ~= 'table' then
    return {'CORRUPT_REQUIRED_CAPABILITY'}
end
local task_required_cap = cap_tbl['required_capability']
if type(task_required_cap) ~= 'string' or task_required_cap == '' then
    return {'CORRUPT_REQUIRED_CAPABILITY'}
end
if task_required_cap ~= requested_capability then
    return {'CAPABILITY_MISMATCH', task_required_cap}
end

-- 4. Existing claim JSON is valid and structurally complete
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
    local c_prior_status = cur_claim['prior_claimable_status']

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
    if c_prior_status ~= nil and c_prior_status ~= 'pending' and c_prior_status ~= 'resumable' then
        return {'CORRUPT_CLAIM'}
    end

    local resolved_prior = c_prior_status or 'pending'

    local c_exec_expires_at = cur_claim['execution_expires_at']
    if c_state == 'acknowledged' and type(c_exec_expires_at) == 'string' and c_exec_expires_at ~= '' and now_iso > c_exec_expires_at then
        return {'EXECUTION_CLAIM_EXPIRED', cur_claim_raw}
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
                effective_status = resolved_prior
            end
        end
    else
        -- Different principal or different request_id
        if c_state == 'claimed_unacknowledged' and c_expires_at <= now_iso then
            has_expired_claim = true
            effective_status = resolved_prior
        else
            return {'ALREADY_CLAIMED', cur_claim_raw}
        end
    end
end

-- 5. Execution-kind and lifecycle compatibility
if execution_kind == 'initial' then
    if is_root ~= '1' or effective_status ~= 'pending' then
        return {'LIFECYCLE_MISMATCH', effective_status}
    end
elseif execution_kind == 'delegated_child' then
    if is_root == '1' or effective_status ~= 'pending' then
        return {'LIFECYCLE_MISMATCH', effective_status}
    end
    local deadline_val = nil
    if deadline_key and deadline_key ~= '' and redis.call('EXISTS', deadline_key) == 1 then
        deadline_val = redis.call('GET', deadline_key)
    elseif status_tbl['claim_deadline'] and status_tbl['claim_deadline'] ~= '' then
        deadline_val = status_tbl['claim_deadline']
    end
    if deadline_val and deadline_val ~= '' and now_iso > deadline_val then
        return {'CHILD_CLAIM_DEADLINE_EXPIRED', deadline_val}
    end
elseif execution_kind == 'resumed_continuation' then
    if is_root ~= '1' or effective_status ~= 'resumable' then
        return {'LIFECYCLE_MISMATCH', effective_status}
    end
else
    return {'INVALID_EXECUTION_KIND', execution_kind}
end

-- 6. Continuation version validation (for resumed_continuation)
if execution_kind == 'resumed_continuation' then
    if not exp_cont_ver or exp_cont_ver < 1 or math.floor(exp_cont_ver) ~= exp_cont_ver then
        return {'INVALID_CONTINUATION_VERSION'}
    end
    if redis.call('EXISTS', continuation_latest_key) == 0 then
        return {'MISSING_CONTINUATION_LATEST'}
    end
    local cur_latest_raw = redis.call('GET', continuation_latest_key)
    local cur_latest_ver = tonumber(cur_latest_raw)
    if not cur_latest_ver or cur_latest_ver < 1 or math.floor(cur_latest_ver) ~= cur_latest_ver then
        return {'CORRUPT_CONTINUATION_LATEST'}
    end
    if cur_latest_ver ~= exp_cont_ver then
        return {'CONTINUATION_VERSION_MISMATCH', tostring(cur_latest_ver)}
    end
    if exp_cont_ver == 2 then
        if type(continuation_key) ~= 'string' or continuation_key == '' then
            return {'MISSING_CONTINUATION'}
        end
        if redis.call('EXISTS', continuation_key) == 0 then
            return {'MISSING_CONTINUATION'}
        end
        local cont_t = redis.call('TYPE', continuation_key)
        if type(cont_t) == 'table' then cont_t = cont_t['ok'] else cont_t = tostring(cont_t) end
        if cont_t ~= 'string' then
            return {'CORRUPT_CONTINUATION'}
        end
        local ok_cnt, cont_tbl = pcall(cjson.decode, redis.call('GET', continuation_key))
        if not ok_cnt or type(cont_tbl) ~= 'table' then
            return {'CORRUPT_CONTINUATION'}
        end
        if cont_tbl['version'] ~= 2 or cont_tbl['continuation_version'] ~= 2 or
           cont_tbl['parent_task_id'] ~= task_id or
           cont_tbl['continuation_capability'] ~= requested_capability or
           cont_tbl['stage'] ~= 'incorporate_analysis_child' then
            return {'CORRUPT_CONTINUATION'}
        end
        local c_wave_id = cont_tbl['wave_id']
        local c_children = cont_tbl['required_child_task_ids']
        if type(c_wave_id) ~= 'string' or c_wave_id == '' or
           type(c_children) ~= 'table' or #c_children < 2 or #c_children > 4 then
            return {'CORRUPT_CONTINUATION'}
        end
        local c_state = cont_tbl['state']
        if c_state ~= 'released' then
            return {'CONTINUATION_NOT_CLAIMABLE', tostring(c_state)}
        end
        local c_res = cont_tbl['wave_resolution']
        if c_res ~= 'SATISFIED' and c_res ~= 'UNSATISFIABLE' and c_res ~= 'TIMED_OUT' then
            return {'UNSUPPORTED_WAVE_RESOLUTION', tostring(c_res)}
        end
        if type(wave_resolution_key) ~= 'string' or wave_resolution_key == '' then
            return {'MISSING_RESOLVED_WAVE'}
        end
        if wave_resolution_key ~= ('task.delegation_wave_resolution:' .. task_id .. ':' .. c_wave_id) then
            return {'WAVE_RESOLUTION_MISMATCH'}
        end
        if redis.call('EXISTS', wave_resolution_key) == 0 then
            return {'MISSING_RESOLVED_WAVE'}
        end
        local res_t = redis.call('TYPE', wave_resolution_key)
        if type(res_t) == 'table' then res_t = res_t['ok'] else res_t = tostring(res_t) end
        if res_t ~= 'string' then
            return {'CORRUPT_RESOLVED_WAVE'}
        end
        local ok_rw, rw_tbl = pcall(cjson.decode, redis.call('GET', wave_resolution_key))
        if not ok_rw or type(rw_tbl) ~= 'table' then
            return {'CORRUPT_RESOLVED_WAVE'}
        end
        if rw_tbl['version'] ~= 2 or rw_tbl['parent_task_id'] ~= task_id or
           rw_tbl['wave_id'] ~= c_wave_id or rw_tbl['resolution'] ~= c_res or
           (rw_tbl['resolution'] ~= 'SATISFIED' and rw_tbl['resolution'] ~= 'UNSATISFIABLE' and rw_tbl['resolution'] ~= 'TIMED_OUT') then
            return {'WAVE_RESOLUTION_MISMATCH'}
        end
        if (rw_tbl['barrier_policy'] ~= 'WAIT_ALL' and rw_tbl['barrier_policy'] ~= 'PARTIAL') or
           type(rw_tbl['resolved_at']) ~= 'string' or rw_tbl['resolved_at'] == '' or
           type(rw_tbl['accepted_at']) ~= 'string' or rw_tbl['accepted_at'] == '' or
           type(rw_tbl['wave_deadline']) ~= 'string' or rw_tbl['wave_deadline'] == '' or
           type(rw_tbl['accepted_wave_fingerprint']) ~= 'string' or rw_tbl['accepted_wave_fingerprint'] == '' then
            return {'CORRUPT_RESOLVED_WAVE'}
        end
        local rw_children = rw_tbl['required_child_task_ids']
        if type(rw_children) ~= 'table' or #rw_children ~= #c_children then
            return {'WAVE_RESOLUTION_MISMATCH'}
        end
        for idx = 1, #c_children do
            if rw_children[idx] ~= c_children[idx] then
                return {'WAVE_RESOLUTION_MISMATCH'}
            end
        end
    end
else
    if exp_cont_ver and exp_cont_ver > 0 then
        return {'UNEXPECTED_CONTINUATION_VERSION'}
    end
end

-- 7. Fence is absent or a valid nonnegative integer
local cur_fence = 0
if redis.call('EXISTS', fence_key) == 1 then
    local fence_raw = redis.call('GET', fence_key)
    cur_fence = tonumber(fence_raw)
    if not cur_fence or cur_fence < 0 or math.floor(cur_fence) ~= cur_fence then
        return {'CORRUPT_FENCE'}
    end
end

-- 8. Expected epoch equals current fence plus one
if not expected_epoch or expected_epoch < 1 or math.floor(expected_epoch) ~= expected_epoch then
    return {'CORRUPT_FENCE'}
end

if cur_fence + 1 ~= expected_epoch then
    return {'EPOCH_RACE', tostring(cur_fence)}
end

-- 9. Grant key and payload pairing: both absent/empty or both present/nonempty
local has_grant_key = (type(grant_key) == "string" and grant_key ~= "")
local has_grant_payload = (grant_payload ~= nil and grant_payload ~= '')

if execution_kind == 'resumed_continuation' then
    if has_grant_payload then
        return {'CORRUPT_GRANT_INPUT'}
    end
    has_grant_key = false
elseif has_grant_key ~= has_grant_payload then
    return {'CORRUPT_GRANT_INPUT'}
end
local has_grant = has_grant_key and has_grant_payload

-- 10. Existing grant and RootAuthorityRecord bounds validation
if has_grant then
    if redis.call('EXISTS', authority_key) == 0 then
        return {'MISSING_ROOT_AUTHORITY'}
    end
    local auth_raw = redis.call('GET', authority_key)
    local ok_auth, auth_tbl = pcall(cjson.decode, auth_raw)
    if not ok_auth or type(auth_tbl) ~= 'table' then
        return {'CORRUPT_ROOT_AUTHORITY'}
    end
    local auth_ver = auth_tbl['version']
    local auth_depth = auth_tbl['max_delegation_depth']
    local auth_waves = auth_tbl['max_delegation_waves']
    local auth_children = auth_tbl['max_children_per_wave']
    if type(auth_ver) ~= 'number' or math.floor(auth_ver) ~= auth_ver or
       type(auth_depth) ~= 'number' or math.floor(auth_depth) ~= auth_depth or
       type(auth_waves) ~= 'number' or math.floor(auth_waves) ~= auth_waves or
       type(auth_children) ~= 'number' or math.floor(auth_children) ~= auth_children then
        return {'INVALID_ROOT_AUTHORITY_BOUNDS'}
    end
    if auth_depth ~= 1 or auth_waves ~= 1 then
        return {'INVALID_ROOT_AUTHORITY_BOUNDS'}
    end
    if auth_ver == 1 then
        if auth_children ~= 1 then
            return {'INVALID_ROOT_AUTHORITY_BOUNDS'}
        end
    elseif auth_ver == 2 then
        if auth_children < 2 or auth_children > 4 then
            return {'INVALID_ROOT_AUTHORITY_BOUNDS'}
        end
    else
        return {'INVALID_ROOT_AUTHORITY_BOUNDS'}
    end
    local edges = auth_tbl['allowed_delegation_edges']
    local has_research_analysis_edge = false
    if type(edges) == 'table' then
        for _, edge in ipairs(edges) do
            if type(edge) == 'table' and edge[1] == 'research' and edge[2] == 'analysis' then
                has_research_analysis_edge = true
                break
            end
        end
    end
    if not has_research_analysis_edge then
        return {'INVALID_ROOT_AUTHORITY_EDGES'}
    end

    local ok_gr, grant_tbl = pcall(cjson.decode, grant_payload)
    if not ok_gr or type(grant_tbl) ~= 'table' then
        return {'CORRUPT_GRANT_INPUT'}
    end
    local g_ver = grant_tbl['version']
    local g_depth = grant_tbl['max_depth']
    local g_waves = grant_tbl['max_waves']
    local g_children = grant_tbl['max_children']
    local g_epoch = grant_tbl['fencing_epoch']
    if type(g_ver) ~= 'number' or math.floor(g_ver) ~= g_ver or
       type(g_depth) ~= 'number' or math.floor(g_depth) ~= g_depth or
       type(g_waves) ~= 'number' or math.floor(g_waves) ~= g_waves or
       type(g_children) ~= 'number' or math.floor(g_children) ~= g_children or
       type(g_epoch) ~= 'number' or math.floor(g_epoch) ~= g_epoch then
        return {'CORRUPT_GRANT_INPUT'}
    end
    if g_depth ~= 1 or g_waves ~= 1 or g_epoch ~= expected_epoch or
       grant_tbl['root_task_id'] ~= task_id or
       grant_tbl['source_capability'] ~= 'research' or
       grant_tbl['target_capability'] ~= 'analysis' then
        return {'INVALID_GRANT_BOUNDS'}
    end
    if g_ver == 1 then
        if auth_ver ~= 1 or g_children ~= 1 then
            return {'INVALID_GRANT_BOUNDS'}
        end
    elseif g_ver == 2 then
        if auth_ver ~= 2 or g_children < 2 or g_children > auth_children or g_children > 4 then
            return {'INVALID_GRANT_BOUNDS'}
        end
    else
        return {'INVALID_GRANT_BOUNDS'}
    end

    if redis.call('EXISTS', grant_key) == 1 then
        local existing_grant = redis.call('GET', grant_key)
        if existing_grant ~= grant_payload then
            return {'CONFLICT_GRANT'}
        end
    end
end

-- 11. Candidate inputs are valid enough for required invariants
if type(claim_payload) ~= 'string' or claim_payload == '' then
    return {'CORRUPT_CLAIM_INPUT'}
end
local ok_cand, cand_claim = pcall(cjson.decode, claim_payload)
if not ok_cand or type(cand_claim) ~= 'table' then
    return {'CORRUPT_CLAIM_INPUT'}
end
if cand_claim['runtime_principal_id'] ~= principal_id or
   cand_claim['request_id'] ~= request_id or
   cand_claim['fencing_epoch'] ~= expected_epoch or
   cand_claim['assigned_capability'] ~= requested_capability or
   cand_claim['execution_kind'] ~= execution_kind or
   cand_claim['prior_claimable_status'] ~= effective_status then
    return {'CORRUPT_CLAIM_INPUT'}
end
if execution_kind == 'resumed_continuation' and cand_claim['continuation_version'] ~= exp_cont_ver then
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

ACCEPT_DELEGATION_LUA = """
local proposal_key = KEYS[1]
local parent_status_key = KEYS[2]
local parent_claim_key = KEYS[3]
local parent_fence_key = KEYS[4]
local parent_lineage_key = KEYS[5]
local root_authority_key = KEYS[6]
local grant_record_key = KEYS[7]
local waves_key = KEYS[8]
local grants_key = KEYS[9]
local decisions_key = KEYS[10]
local yield_key = KEYS[11]
local continuation_key = KEYS[12]
local continuation_latest_key = KEYS[13]
local artifact_key = KEYS[14]
local parent_input_key = KEYS[15]
local parent_children_key = KEYS[16]
local child_status_key = KEYS[17]
local child_lineage_key = KEYS[18]
local child_capability_key = KEYS[19]
local child_input_key = KEYS[20]
local child_deadline_key = KEYS[21]
local child_fence_key = KEYS[22]
local hint_queue_key = KEYS[23]

local parent_task_id = ARGV[1]
local proposal_id = ARGV[2]
local assignment_id = ARGV[3]
local principal_id = ARGV[4]
local proposal_fingerprint = ARGV[5]
local delegation_grant_id = ARGV[6]
local expected_epoch = tonumber(ARGV[7])
local child_task_id = ARGV[8]
local delegation_decision_id = ARGV[9]
local now_iso = ARGV[10]
local child_deadline_iso = ARGV[11]
local ref_artifact_id = ARGV[12]
local ref_artifact_version = tonumber(ARGV[13])
local bound_agent_id = ARGV[14]
local expected_parent_input_version = tonumber(ARGV[15])
local pre_serialized_proposal_json = ARGV[16]
local pre_serialized_child_status_json = ARGV[17]
local pre_serialized_child_lineage_json = ARGV[18]
local pre_serialized_child_capability_json = ARGV[19]
local pre_serialized_child_input_json = ARGV[20]
local pre_serialized_continuation_json = ARGV[21]
local pre_serialized_parent_status_json = ARGV[22]
local pre_serialized_yield_json = ARGV[23]
local pre_serialized_hint_json = ARGV[24]

-- =============================================================================
-- PHASE 1: Strictly Read-Only Validations (Zero Mutations)
-- =============================================================================

-- 1. Idempotent Replay Check (FIRST STORE-LEVEL OPERATION)
if redis.call('EXISTS', proposal_key) == 1 then
    local t = redis.call('TYPE', proposal_key)
    local type_str = (type(t) == 'table' and t['ok']) or tostring(t)
    if type_str ~= 'string' then return {'WRONG_KEY_TYPE_PROPOSAL'} end
    local stored_raw = redis.call('GET', proposal_key)
    local ok_p, stored = pcall(cjson.decode, stored_raw)
    if not ok_p or type(stored) ~= 'table' then
        return {'CORRUPT_PROPOSAL'}
    end
    if stored['assignment_id'] == assignment_id and
       stored['runtime_principal_id'] == principal_id and
       stored['bound_agent_id'] == bound_agent_id and
       stored['parent_task_id'] == parent_task_id and
       stored['proposal_id'] == proposal_id and
       stored['proposal_fingerprint'] == proposal_fingerprint then
        return {'IDEMPOTENT', stored_raw}
    else
        return {'PROPOSAL_CONFLICT'}
    end
end

-- 2. Validate Parent Task Existence & Status
if redis.call('EXISTS', parent_status_key) == 0 then
    return {'PARENT_NOT_FOUND'}
end
local t = redis.call('TYPE', parent_status_key)
local type_str = (type(t) == 'table' and t['ok']) or tostring(t)
if type_str ~= 'string' then return {'WRONG_KEY_TYPE_PARENT_STATUS'} end

local parent_status_raw = redis.call('GET', parent_status_key)
local ok_ps, parent_status = pcall(cjson.decode, parent_status_raw)
if not ok_ps or type(parent_status) ~= 'table' then
    return {'CORRUPT_PARENT_STATUS'}
end
if parent_status['status'] ~= 'working' then
    return {'PARENT_NOT_WORKING', parent_status['status']}
end

-- 3. Validate Active Claim
if redis.call('EXISTS', parent_claim_key) == 0 then
    return {'CLAIM_NOT_FOUND'}
end
t = redis.call('TYPE', parent_claim_key)
type_str = (type(t) == 'table' and t['ok']) or tostring(t)
if type_str ~= 'string' then return {'WRONG_KEY_TYPE_CLAIM'} end

local claim_raw = redis.call('GET', parent_claim_key)
local ok_c, claim = pcall(cjson.decode, claim_raw)
if not ok_c or type(claim) ~= 'table' then
    return {'CORRUPT_CLAIM'}
end
if type(claim['turn_id']) ~= 'string' or claim['turn_id'] == '' then
    return {'CORRUPT_CLAIM'}
end
if claim['assignment_id'] ~= assignment_id then
    return {'ASSIGNMENT_MISMATCH'}
end
if claim['runtime_principal_id'] ~= principal_id then
    return {'PRINCIPAL_MISMATCH'}
end
if claim['bound_agent_id'] ~= bound_agent_id then
    return {'BOUND_AGENT_MISMATCH'}
end
if claim['state'] ~= 'acknowledged' then
    return {'CLAIM_NOT_ACKNOWLEDGED'}
end
if claim['assigned_capability'] ~= 'research' then
    return {'CAPABILITY_POLICY_VIOLATION'}
end
if claim['execution_kind'] ~= 'initial' then
    return {'EXECUTION_KIND_MISMATCH'}
end

-- 4. Wave & Grant Budget Validation
if redis.call('EXISTS', waves_key) == 1 then
    t = redis.call('TYPE', waves_key)
    type_str = (type(t) == 'table' and t['ok']) or tostring(t)
    if type_str ~= 'set' then return {'WRONG_KEY_TYPE_WAVES'} end
    if redis.call('SISMEMBER', waves_key, '1') == 1 then
        return {'WAVE_ALREADY_CONSUMED'}
    end
end
if redis.call('EXISTS', grants_key) == 1 then
    t = redis.call('TYPE', grants_key)
    type_str = (type(t) == 'table' and t['ok']) or tostring(t)
    if type_str ~= 'set' then return {'WRONG_KEY_TYPE_GRANTS'} end
    if redis.call('SISMEMBER', grants_key, delegation_grant_id) == 1 then
        return {'GRANT_ALREADY_CONSUMED'}
    end
end
if redis.call('EXISTS', decisions_key) == 1 then
    t = redis.call('TYPE', decisions_key)
    type_str = (type(t) == 'table' and t['ok']) or tostring(t)
    if type_str ~= 'set' then return {'WRONG_KEY_TYPE_DECISIONS'} end
end
if redis.call('EXISTS', parent_children_key) == 1 then
    t = redis.call('TYPE', parent_children_key)
    type_str = (type(t) == 'table' and t['ok']) or tostring(t)
    if type_str ~= 'set' then return {'WRONG_KEY_TYPE_CHILDREN'} end
    local count = redis.call('SCARD', parent_children_key)
    if count and count >= 1 then
        return {'CHILD_LIMIT_EXCEEDED'}
    end
end

-- 5. Validate Required Keys Existence & Types
if redis.call('EXISTS', parent_fence_key) == 0 then
    return {'FENCE_NOT_FOUND'}
end
t = redis.call('TYPE', parent_fence_key)
type_str = (type(t) == 'table' and t['ok']) or tostring(t)
if type_str ~= 'string' then return {'WRONG_KEY_TYPE_FENCE'} end

if redis.call('EXISTS', parent_lineage_key) == 0 then
    return {'PARENT_LINEAGE_NOT_FOUND'}
end
t = redis.call('TYPE', parent_lineage_key)
type_str = (type(t) == 'table' and t['ok']) or tostring(t)
if type_str ~= 'string' then return {'WRONG_KEY_TYPE_PARENT_LINEAGE'} end

if redis.call('EXISTS', root_authority_key) == 0 then
    return {'ROOT_AUTHORITY_NOT_FOUND'}
end
t = redis.call('TYPE', root_authority_key)
type_str = (type(t) == 'table' and t['ok']) or tostring(t)
if type_str ~= 'string' then return {'WRONG_KEY_TYPE_ROOT_AUTHORITY'} end

if redis.call('EXISTS', grant_record_key) == 0 then
    return {'GRANT_NOT_FOUND'}
end
t = redis.call('TYPE', grant_record_key)
type_str = (type(t) == 'table' and t['ok']) or tostring(t)
if type_str ~= 'string' then return {'WRONG_KEY_TYPE_GRANT_RECORD'} end

if redis.call('EXISTS', artifact_key) == 0 then
    return {'ARTIFACT_NOT_FOUND'}
end
t = redis.call('TYPE', artifact_key)
type_str = (type(t) == 'table' and t['ok']) or tostring(t)
if type_str ~= 'string' then return {'WRONG_KEY_TYPE_ARTIFACT'} end

if redis.call('EXISTS', parent_input_key) == 0 then
    return {'PARENT_INPUT_NOT_FOUND'}
end
t = redis.call('TYPE', parent_input_key)
type_str = (type(t) == 'table' and t['ok']) or tostring(t)
if type_str ~= 'string' then return {'WRONG_KEY_TYPE_PARENT_INPUT'} end

if redis.call('EXISTS', continuation_latest_key) == 1 then
    t = redis.call('TYPE', continuation_latest_key)
    type_str = (type(t) == 'table' and t['ok']) or tostring(t)
    if type_str ~= 'string' then return {'WRONG_KEY_TYPE_CONTINUATION_LATEST'} end
end
if redis.call('EXISTS', hint_queue_key) == 1 then
    t = redis.call('TYPE', hint_queue_key)
    type_str = (type(t) == 'table' and t['ok']) or tostring(t)
    if type_str ~= 'list' then return {'WRONG_KEY_TYPE_HINT_QUEUE'} end
end

-- 6. Validate Target Keys Do Not Exist Before Write
if redis.call('EXISTS', yield_key) == 1 then
    return {'YIELD_ALREADY_EXISTS'}
end
if redis.call('EXISTS', continuation_key) == 1 or redis.call('EXISTS', continuation_latest_key) == 1 then
    return {'CONTINUATION_EXISTS'}
end
if redis.call('EXISTS', child_status_key) == 1 or
   redis.call('EXISTS', child_lineage_key) == 1 or
   redis.call('EXISTS', child_capability_key) == 1 or
   redis.call('EXISTS', child_input_key) == 1 or
   redis.call('EXISTS', child_deadline_key) == 1 or
   redis.call('EXISTS', child_fence_key) == 1 then
    return {'CHILD_TASK_ALREADY_EXISTS'}
end

-- 7. Fencing Epoch Validation
local cur_fence = tonumber(redis.call('GET', parent_fence_key))
if not cur_fence or not expected_epoch or cur_fence ~= expected_epoch then
    return {'FENCE_MISMATCH', tostring(cur_fence)}
end
local claim_epoch = tonumber(claim['fencing_epoch'])
if not claim_epoch or claim_epoch ~= expected_epoch then
    return {'FENCE_MISMATCH', tostring(claim_epoch)}
end

-- 8. Parent Lineage (must be root)
local lineage_raw = redis.call('GET', parent_lineage_key)
local ok_l, lineage = pcall(cjson.decode, lineage_raw)
if not ok_l or type(lineage) ~= 'table' then
    return {'CORRUPT_LINEAGE'}
end
local has_task_id = false
local has_root_task_id = false
local has_parent_task_id = false
for k, _ in pairs(lineage) do
    if k == 'task_id' then has_task_id = true end
    if k == 'root_task_id' then has_root_task_id = true end
    if k == 'parent_task_id' then has_parent_task_id = true end
end
if not has_task_id or not has_root_task_id or not has_parent_task_id then
    return {'CORRUPT_PARENT_LINEAGE'}
end
if type(lineage['task_id']) ~= 'string' or type(lineage['root_task_id']) ~= 'string' then
    return {'CORRUPT_PARENT_LINEAGE'}
end
if lineage['task_id'] ~= parent_task_id or lineage['root_task_id'] ~= parent_task_id then
    return {'CORRUPT_PARENT_LINEAGE'}
end
if lineage['parent_task_id'] ~= cjson.null then
    return {'PARENT_NOT_ROOT'}
end

-- 9. Root Authority Validation
local auth_raw = redis.call('GET', root_authority_key)
local ok_ra, auth = pcall(cjson.decode, auth_raw)
if not ok_ra or type(auth) ~= 'table' then
    return {'CORRUPT_ROOT_AUTHORITY'}
end
if auth['root_task_id'] ~= parent_task_id then
    return {'ROOT_AUTHORITY_BINDING_MISMATCH'}
end
local d = auth['max_delegation_depth']
local w = auth['max_delegation_waves']
local c = auth['max_children_per_wave']
if type(d) ~= 'number' or d ~= 1 or math.floor(d) ~= d or
   type(w) ~= 'number' or w ~= 1 or math.floor(w) ~= w or
   type(c) ~= 'number' or c ~= 1 or math.floor(c) ~= c then
    return {'POLICY_BOUNDS_VIOLATION'}
end
local edges = auth['allowed_delegation_edges']
if type(edges) ~= 'table' then
    return {'CORRUPT_ROOT_AUTHORITY'}
end
local edge_allowed = false
for _, edge in ipairs(edges) do
    if type(edge) == 'table' and edge[1] == 'research' and edge[2] == 'analysis' then
        edge_allowed = true
        break
    end
end
if not edge_allowed then
    return {'DELEGATION_EDGE_NOT_PERMITTED'}
end

-- 10. Authoritative Delegation Grant Validation
local grant_raw = redis.call('GET', grant_record_key)
local ok_g, grant = pcall(cjson.decode, grant_raw)
if not ok_g or type(grant) ~= 'table' then
    return {'CORRUPT_GRANT'}
end
if type(grant['turn_id']) ~= 'string' or grant['turn_id'] == '' then
    return {'CORRUPT_GRANT'}
end
if grant['turn_id'] ~= claim['turn_id'] then
    return {'GRANT_TURN_MISMATCH'}
end
if grant['grant_id'] ~= delegation_grant_id then
    return {'GRANT_ID_MISMATCH'}
end
if grant['root_task_id'] ~= parent_task_id then
    return {'GRANT_ROOT_BINDING_MISMATCH'}
end
if grant['issued_to_assignment_id'] ~= assignment_id then
    return {'GRANT_ASSIGNMENT_MISMATCH'}
end
if grant['issued_to_principal_id'] ~= principal_id then
    return {'GRANT_PRINCIPAL_MISMATCH'}
end
if grant['source_capability'] ~= 'research' or grant['target_capability'] ~= 'analysis' then
    return {'GRANT_CAPABILITY_MISMATCH'}
end
local g_epoch = tonumber(grant['fencing_epoch'])
if not g_epoch or g_epoch ~= expected_epoch then
    return {'GRANT_EPOCH_MISMATCH'}
end
local gd = grant['max_depth']
local gw = grant['max_waves']
local gc = grant['max_children']
if type(gd) ~= 'number' or gd ~= 1 or math.floor(gd) ~= gd or
   type(gw) ~= 'number' or gw ~= 1 or math.floor(gw) ~= gw or
   type(gc) ~= 'number' or gc ~= 1 or math.floor(gc) ~= gc then
    return {'GRANT_BOUNDS_VIOLATION'}
end

-- 11. Sealed Artifact Validation
local art_raw = redis.call('GET', artifact_key)
local ok_a, art = pcall(cjson.decode, art_raw)
if not ok_a or type(art) ~= 'table' then
    return {'CORRUPT_ARTIFACT'}
end
if art['sealed'] ~= true then
    return {'ARTIFACT_NOT_SEALED'}
end
if art['producing_task_id'] ~= parent_task_id then
    return {'ARTIFACT_PROVENANCE_MISMATCH'}
end
if art['artifact_type'] ~= 'research_material' then
    return {'ARTIFACT_TYPE_MISMATCH'}
end
if art['artifact_id'] ~= ref_artifact_id or tonumber(art['artifact_version']) ~= ref_artifact_version then
    return {'ARTIFACT_IDENTITY_MISMATCH'}
end
if type(art['producing_turn_id']) ~= 'string' or art['producing_turn_id'] == '' then
    return {'ARTIFACT_MISSING_TURN_ID'}
end
if art['producing_turn_id'] ~= claim['turn_id'] then
    return {'ARTIFACT_TURN_MISMATCH'}
end
local art_schema_ver = art['payload_schema_version']
if type(art_schema_ver) ~= 'number' or math.floor(art_schema_ver) ~= art_schema_ver then
    return {'CORRUPT_ARTIFACT'}
end
if art_schema_ver ~= 1 then
    return {'UNSUPPORTED_ARTIFACT_SCHEMA_VERSION'}
end
local art_contract_ver = art['version']
if type(art_contract_ver) ~= 'number' or math.floor(art_contract_ver) ~= art_contract_ver or art_contract_ver ~= 1 then
    return {'UNSUPPORTED_ARTIFACT_CONTRACT_VERSION'}
end
if type(art['created_at']) ~= 'string' or art['created_at'] == '' then
    return {'CORRUPT_ARTIFACT'}
end
if type(art['content']) ~= 'table' then
    return {'CORRUPT_ARTIFACT'}
end
if art['metadata'] ~= nil and type(art['metadata']) ~= 'table' then
    return {'CORRUPT_ARTIFACT'}
end

-- 12. Authoritative Parent TaskInput Validation
local parent_input_raw = redis.call('GET', parent_input_key)
local ok_pi, pinput = pcall(cjson.decode, parent_input_raw)
if not ok_pi or type(pinput) ~= 'table' then
    return {'CORRUPT_PARENT_INPUT'}
end
if type(pinput['task_id']) ~= 'string' or pinput['task_id'] == '' then
    return {'CORRUPT_PARENT_INPUT'}
end
if pinput['task_id'] ~= parent_task_id then
    return {'PARENT_INPUT_TASK_MISMATCH'}
end
local pi_ver = pinput['input_version']
if type(pi_ver) ~= 'number' or math.floor(pi_ver) ~= pi_ver then
    return {'CORRUPT_PARENT_INPUT'}
end
if not expected_parent_input_version or pi_ver ~= expected_parent_input_version then
    return {'PARENT_INPUT_VERSION_MISMATCH'}
end
local pi_cver = pinput['version']
if type(pi_cver) ~= 'number' or math.floor(pi_cver) ~= pi_cver or pi_cver ~= 1 then
    return {'UNSUPPORTED_PARENT_INPUT_VERSION'}
end
if type(pinput['prompt']) ~= 'string' or string.match(pinput['prompt'], '^%s*$') then
    return {'CORRUPT_PARENT_INPUT'}
end
if type(pinput['created_at']) ~= 'string' or pinput['created_at'] == '' then
    return {'CORRUPT_PARENT_INPUT'}
end
if pinput['metadata'] ~= nil and type(pinput['metadata']) ~= 'table' then
    return {'CORRUPT_PARENT_INPUT'}
end
if pinput['artifact_references'] ~= nil and type(pinput['artifact_references']) ~= 'table' then
    return {'CORRUPT_PARENT_INPUT'}
end

-- =============================================================================
-- PHASE 2: Mechanically Non-Failing Writes (Atomic Commit)
-- =============================================================================
redis.call('SET', proposal_key, pre_serialized_proposal_json)
redis.call('SET', child_status_key, pre_serialized_child_status_json)
redis.call('SET', child_lineage_key, pre_serialized_child_lineage_json)
redis.call('SET', child_capability_key, pre_serialized_child_capability_json)
redis.call('SET', child_input_key, pre_serialized_child_input_json)
redis.call('SET', child_deadline_key, child_deadline_iso)
redis.call('SET', child_fence_key, '0')
redis.call('SADD', parent_children_key, child_task_id)
redis.call('SET', continuation_key, pre_serialized_continuation_json)
redis.call('SET', continuation_latest_key, '1')
redis.call('SET', parent_status_key, pre_serialized_parent_status_json)
redis.call('DEL', parent_claim_key)
redis.call('SET', yield_key, pre_serialized_yield_json)
redis.call('SADD', waves_key, '1')
redis.call('SADD', grants_key, delegation_grant_id)
redis.call('SADD', decisions_key, delegation_decision_id)
redis.call('RPUSH', hint_queue_key, pre_serialized_hint_json)

return {'OK', pre_serialized_proposal_json}
"""

ACCEPT_DELEGATION_WAVE_LUA = """
local proposal_key = KEYS[1]
local wave_key = KEYS[2]
local parent_status_key = KEYS[3]
local parent_claim_key = KEYS[4]
local parent_fence_key = KEYS[5]
local parent_lineage_key = KEYS[6]
local root_authority_key = KEYS[7]
local grant_record_key = KEYS[8]
local waves_key = KEYS[9]
local grants_key = KEYS[10]
local decisions_key = KEYS[11]
local yield_key = KEYS[12]
local continuation_key = KEYS[13]
local continuation_latest_key = KEYS[14]
local parent_input_key = KEYS[15]
local parent_children_key = KEYS[16]
local hint_queue_key = KEYS[17]

local c1_status_key = KEYS[18]
local c1_lineage_key = KEYS[19]
local c1_capability_key = KEYS[20]
local c1_input_key = KEYS[21]
local c1_deadline_key = KEYS[22]
local c1_fence_key = KEYS[23]

local c2_status_key = KEYS[24]
local c2_lineage_key = KEYS[25]
local c2_capability_key = KEYS[26]
local c2_input_key = KEYS[27]
local c2_deadline_key = KEYS[28]
local c2_fence_key = KEYS[29]

local c3_status_key = KEYS[30]
local c3_lineage_key = KEYS[31]
local c3_capability_key = KEYS[32]
local c3_input_key = KEYS[33]
local c3_deadline_key = KEYS[34]
local c3_fence_key = KEYS[35]

local c4_status_key = KEYS[36]
local c4_lineage_key = KEYS[37]
local c4_capability_key = KEYS[38]
local c4_input_key = KEYS[39]
local c4_deadline_key = KEYS[40]
local c4_fence_key = KEYS[41]

local art1_key = KEYS[42]
local art2_key = KEYS[43]
local art3_key = KEYS[44]
local art4_key = KEYS[45]

local parent_task_id = ARGV[1]
local proposal_id = ARGV[2]
local wave_id = ARGV[3]
local assignment_id = ARGV[4]
local principal_id = ARGV[5]
local bound_agent_id = ARGV[6]
local proposal_fingerprint = ARGV[7]
local delegation_grant_id = ARGV[8]
local expected_epoch = tonumber(ARGV[9])
local delegation_decision_id = ARGV[10]
local child_deadline_iso = ARGV[11]
local expected_parent_input_version = tonumber(ARGV[12])
local child_count = tonumber(ARGV[13])
local barrier_policy = ARGV[14]
local req_success_count_raw = ARGV[15]

local pre_proposal_json = ARGV[16]
local pre_wave_json = ARGV[17]
local pre_continuation_json = ARGV[18]
local pre_parent_status_json = ARGV[19]
local pre_yield_json = ARGV[20]

local c1_id = ARGV[21]
local c1_status_json = ARGV[22]
local c1_lineage_json = ARGV[23]
local c1_cap_json = ARGV[24]
local c1_input_json = ARGV[25]
local c1_hint_json = ARGV[26]
local c1_art_id = ARGV[27]
local c1_art_ver = tonumber(ARGV[28])

local c2_id = ARGV[29]
local c2_status_json = ARGV[30]
local c2_lineage_json = ARGV[31]
local c2_cap_json = ARGV[32]
local c2_input_json = ARGV[33]
local c2_hint_json = ARGV[34]
local c2_art_id = ARGV[35]
local c2_art_ver = tonumber(ARGV[36])

local c3_id = ARGV[37]
local c3_status_json = ARGV[38]
local c3_lineage_json = ARGV[39]
local c3_cap_json = ARGV[40]
local c3_input_json = ARGV[41]
local c3_hint_json = ARGV[42]
local c3_art_id = ARGV[43]
local c3_art_ver = tonumber(ARGV[44])

local c4_id = ARGV[45]
local c4_status_json = ARGV[46]
local c4_lineage_json = ARGV[47]
local c4_cap_json = ARGV[48]
local c4_input_json = ARGV[49]
local c4_hint_json = ARGV[50]
local c4_art_id = ARGV[51]
local c4_art_ver = tonumber(ARGV[52])

local expected_frac_num_raw = ARGV[53]
local expected_frac_den_raw = ARGV[54]

local function key_type(key)
    local t = redis.call('TYPE', key)
    if type(t) == 'table' then return t['ok'] end
    return tostring(t)
end

local function validate_fraction_value(frac_tbl, policy, exp_num_raw, exp_den_raw)
    if policy == 'WAIT_ALL' then
        if exp_num_raw ~= '' or exp_den_raw ~= '' then
            return false
        end
        return frac_tbl == cjson.null or frac_tbl == nil
    elseif policy == 'PARTIAL' then
        local exp_num = tonumber(exp_num_raw)
        local exp_den = tonumber(exp_den_raw)
        if not exp_num or not exp_den or
           math.floor(exp_num) ~= exp_num or math.floor(exp_den) ~= exp_den or
           exp_num < 1 or exp_den < 1 or exp_num > exp_den or exp_den > 10000 then
            return false
        end
        if type(frac_tbl) ~= 'table' then
            return false
        end
        local f_num = frac_tbl['numerator']
        local f_den = frac_tbl['denominator']
        if type(f_num) ~= 'number' or type(f_den) ~= 'number' or
           math.floor(f_num) ~= f_num or math.floor(f_den) ~= f_den then
            return false
        end
        return f_num == exp_num and f_den == exp_den
    end
    return false
end

local function validate_artifact(akey, expected_id, expected_ver, claim_turn_id)
    if redis.call('EXISTS', akey) == 0 then
        return 'ARTIFACT_NOT_FOUND'
    end
    if key_type(akey) ~= 'string' then return 'WRONG_KEY_TYPE_ARTIFACT' end
    local ok_a, art = pcall(cjson.decode, redis.call('GET', akey))
    if not ok_a or type(art) ~= 'table' then return 'CORRUPT_ARTIFACT' end
    if art['sealed'] ~= true then return 'ARTIFACT_NOT_SEALED' end
    if art['producing_task_id'] ~= parent_task_id then return 'ARTIFACT_PROVENANCE_MISMATCH' end
    if art['artifact_type'] ~= 'research_material' then return 'ARTIFACT_TYPE_MISMATCH' end
    if art['artifact_id'] ~= expected_id or tonumber(art['artifact_version']) ~= expected_ver then
        return 'ARTIFACT_IDENTITY_MISMATCH'
    end
    if type(art['producing_turn_id']) ~= 'string' or art['producing_turn_id'] == '' then
        return 'ARTIFACT_MISSING_TURN_ID'
    end
    if art['producing_turn_id'] ~= claim_turn_id then return 'ARTIFACT_TURN_MISMATCH' end
    local sv = art['payload_schema_version']
    if type(sv) ~= 'number' or math.floor(sv) ~= sv or sv ~= 1 then
        return 'UNSUPPORTED_ARTIFACT_SCHEMA_VERSION'
    end
    local cv = art['version']
    if type(cv) ~= 'number' or math.floor(cv) ~= cv or cv ~= 1 then
        return 'UNSUPPORTED_ARTIFACT_CONTRACT_VERSION'
    end
    if type(art['created_at']) ~= 'string' or art['created_at'] == '' then return 'CORRUPT_ARTIFACT' end
    if type(art['content']) ~= 'table' then return 'CORRUPT_ARTIFACT' end
    if art['metadata'] ~= nil and type(art['metadata']) ~= 'table' then return 'CORRUPT_ARTIFACT' end
    return nil
end

local function child_keys_exist(sk, lk, ck, ik, dk, fk)
    return redis.call('EXISTS', sk) == 1 or
           redis.call('EXISTS', lk) == 1 or
           redis.call('EXISTS', ck) == 1 or
           redis.call('EXISTS', ik) == 1 or
           redis.call('EXISTS', dk) == 1 or
           redis.call('EXISTS', fk) == 1
end

local function validate_candidate_child(
    cid, expected_cid, status_json, lineage_json, cap_json, input_json, hint_json, art_id, art_ver_raw
)
    if type(cid) ~= 'string' or cid == '' or cid ~= expected_cid or cid == parent_task_id then
        return 'WAVE_CONSISTENCY_MISMATCH'
    end
    local art_ver = tonumber(art_ver_raw)
    if type(art_ver) ~= 'number' or math.floor(art_ver) ~= art_ver or art_ver < 1 then
        return 'CORRUPT_CANDIDATE_PAYLOAD'
    end
    local ok_st, st = pcall(cjson.decode, status_json)
    local ok_ln, ln = pcall(cjson.decode, lineage_json)
    local ok_cp, cp = pcall(cjson.decode, cap_json)
    local ok_in, inp = pcall(cjson.decode, input_json)
    local ok_hn, hn = pcall(cjson.decode, hint_json)
    if not ok_st or type(st) ~= 'table' or
       not ok_ln or type(ln) ~= 'table' or
       not ok_cp or type(cp) ~= 'table' or
       not ok_in or type(inp) ~= 'table' or
       not ok_hn or type(hn) ~= 'table' then
        return 'CORRUPT_CANDIDATE_PAYLOAD'
    end
    if st['task_id'] ~= cid or st['status'] ~= 'pending' or st['claim_deadline'] ~= child_deadline_iso then
        return 'WAVE_CONSISTENCY_MISMATCH'
    end
    if ln['task_id'] ~= cid or ln['parent_task_id'] ~= parent_task_id or ln['root_task_id'] ~= parent_task_id then
        return 'WAVE_CONSISTENCY_MISMATCH'
    end
    if cp['task_id'] ~= cid or cp['required_capability'] ~= 'analysis' then
        return 'WAVE_CONSISTENCY_MISMATCH'
    end
    if inp['task_id'] ~= cid or inp['input_version'] ~= 1 or
       type(inp['prompt']) ~= 'string' or string.match(inp['prompt'], '^%s*$') or
       type(inp['artifact_references']) ~= 'table' or #inp['artifact_references'] < 1 or
       type(inp['artifact_references'][1]) ~= 'table' or
       inp['artifact_references'][1]['artifact_id'] ~= art_id or
       inp['artifact_references'][1]['artifact_version'] ~= art_ver then
        return 'WAVE_CONSISTENCY_MISMATCH'
    end
    if hn['task_id'] ~= cid or hn['capability'] ~= 'analysis' or hn['execution_kind'] ~= 'delegated_child' then
        return 'WAVE_CONSISTENCY_MISMATCH'
    end
    return nil
end

-- =============================================================================
-- PHASE 1: Strictly Read-Only Validations (Zero Mutations)
-- =============================================================================

-- 1. Idempotent Replay Check (FIRST STORE-LEVEL OPERATION)
if redis.call('EXISTS', proposal_key) == 1 then
    if key_type(proposal_key) ~= 'string' then return {'WRONG_KEY_TYPE_PROPOSAL'} end
    local stored_raw = redis.call('GET', proposal_key)
    local ok_p, stored = pcall(cjson.decode, stored_raw)
    if not ok_p or type(stored) ~= 'table' then return {'CORRUPT_PROPOSAL'} end
    if stored['assignment_id'] ~= assignment_id or
       stored['runtime_principal_id'] ~= principal_id or
       stored['bound_agent_id'] ~= bound_agent_id or
       stored['parent_task_id'] ~= parent_task_id or
       stored['proposal_id'] ~= proposal_id or
       stored['proposal_fingerprint'] ~= proposal_fingerprint or
       stored['fencing_epoch'] ~= expected_epoch then
        return {'PROPOSAL_CONFLICT'}
    end

    local s_wave_id = stored['wave_id']
    local s_children = stored['required_child_task_ids']
    local s_deadline = stored['child_claim_deadline']
    if type(s_wave_id) ~= 'string' or s_wave_id == '' or
       type(s_children) ~= 'table' or #s_children < 2 or #s_children > 4 or
       type(s_deadline) ~= 'string' or s_deadline == '' then
        return {'CORRUPT_PROPOSAL'}
    end

    local s_wave_key = 'task.delegation_wave:' .. parent_task_id .. ':' .. s_wave_id
    if redis.call('EXISTS', s_wave_key) == 0 then return {'CORRUPT_WAVE_PROJECTION'} end
    if key_type(s_wave_key) ~= 'string' then return {'WRONG_KEY_TYPE_WAVE'} end
    local ok_sw, sw = pcall(cjson.decode, redis.call('GET', s_wave_key))
    if not ok_sw or type(sw) ~= 'table' then return {'CORRUPT_WAVE_PROJECTION'} end
    if sw['wave_id'] ~= s_wave_id or sw['parent_task_id'] ~= parent_task_id or
       sw['state'] ~= 'unresolved' or sw['barrier_policy'] ~= stored['barrier_policy'] or
       stored['barrier_policy'] ~= barrier_policy or
       sw['required_success_count'] ~= stored['required_success_count'] or
       not validate_fraction_value(stored['required_success_fraction'], barrier_policy, expected_frac_num_raw, expected_frac_den_raw) or
       not validate_fraction_value(sw['required_success_fraction'], barrier_policy, expected_frac_num_raw, expected_frac_den_raw) or
       sw['accepted_at'] ~= stored['accepted_at'] or sw['wave_deadline'] ~= stored['wave_deadline'] or
       type(sw['required_child_task_ids']) ~= 'table' or #sw['required_child_task_ids'] ~= #s_children then
        return {'CORRUPT_WAVE_PROJECTION'}
    end

    local s_resolved_key = 'task.delegation_wave_resolution:' .. parent_task_id .. ':' .. s_wave_id
    local has_resolved = (redis.call('EXISTS', s_resolved_key) == 1)
    local srw = nil
    if has_resolved then
        if key_type(s_resolved_key) ~= 'string' then return {'WRONG_KEY_TYPE_RESOLVED_WAVE'} end
        local ok_srw, decoded_srw = pcall(cjson.decode, redis.call('GET', s_resolved_key))
        if not ok_srw or type(decoded_srw) ~= 'table' then return {'CORRUPT_RESOLVED_WAVE'} end
        srw = decoded_srw
        if srw['version'] ~= 2 or srw['wave_id'] ~= s_wave_id or srw['parent_task_id'] ~= parent_task_id or
           srw['barrier_policy'] ~= stored['barrier_policy'] or
           srw['required_success_count'] ~= stored['required_success_count'] or
           not validate_fraction_value(srw['required_success_fraction'], barrier_policy, expected_frac_num_raw, expected_frac_den_raw) or
           srw['accepted_at'] ~= stored['accepted_at'] or srw['wave_deadline'] ~= stored['wave_deadline'] or
           (srw['resolution'] ~= 'SATISFIED' and srw['resolution'] ~= 'UNSATISFIABLE' and srw['resolution'] ~= 'TIMED_OUT' and srw['resolution'] ~= 'CANCELLED') or
           type(srw['required_child_task_ids']) ~= 'table' or #srw['required_child_task_ids'] ~= #s_children or
           type(srw['child_outcomes']) ~= 'table' then
            return {'CORRUPT_RESOLVED_WAVE'}
        end
    end

    if redis.call('EXISTS', continuation_key) == 0 then return {'CORRUPT_CONTINUATION'} end
    if key_type(continuation_key) ~= 'string' then return {'WRONG_KEY_TYPE_CONTINUATION'} end
    local ok_sc, sc = pcall(cjson.decode, redis.call('GET', continuation_key))
    if not ok_sc or type(sc) ~= 'table' then return {'CORRUPT_CONTINUATION'} end
    if sc['parent_task_id'] ~= parent_task_id or sc['wave_id'] ~= s_wave_id or
       sc['continuation_version'] ~= 2 or sc['version'] ~= 2 or
       type(sc['required_child_task_ids']) ~= 'table' or #sc['required_child_task_ids'] ~= #s_children then
        return {'CORRUPT_CONTINUATION'}
    end
    if not has_resolved then
        if sc['state'] ~= 'active' or sc['wave_resolution'] ~= cjson.null then
            return {'CORRUPT_CONTINUATION'}
        end
    elseif srw['resolution'] == 'CANCELLED' then
        if sc['state'] ~= 'closed' or sc['wave_resolution'] ~= cjson.null then
            return {'CORRUPT_CONTINUATION'}
        end
    else
        if sc['state'] ~= 'released' or sc['wave_resolution'] ~= srw['resolution'] then
            return {'CORRUPT_CONTINUATION'}
        end
    end

    if redis.call('EXISTS', continuation_latest_key) == 0 then return {'CORRUPT_CONTINUATION_LATEST'} end
    if key_type(continuation_latest_key) ~= 'string' then return {'WRONG_KEY_TYPE_CONTINUATION_LATEST'} end
    if redis.call('GET', continuation_latest_key) ~= '2' then return {'CORRUPT_CONTINUATION_LATEST'} end

    if redis.call('EXISTS', parent_status_key) == 0 then return {'CORRUPT_PARENT_STATUS'} end
    if key_type(parent_status_key) ~= 'string' then return {'WRONG_KEY_TYPE_PARENT_STATUS'} end
    local ok_sps, sps = pcall(cjson.decode, redis.call('GET', parent_status_key))
    if not ok_sps or type(sps) ~= 'table' or sps['task_id'] ~= parent_task_id then
        return {'CORRUPT_PARENT_STATUS'}
    end
    local p_st = sps['status']
    if not has_resolved then
        if p_st ~= 'waiting_for_children' then
            return {'CORRUPT_PARENT_STATUS'}
        end
    elseif srw['resolution'] == 'CANCELLED' then
        if p_st ~= 'cancelled' then
            return {'CORRUPT_PARENT_STATUS'}
        end
    else
        if p_st ~= 'resumable' and p_st ~= 'working' and p_st ~= 'completed' and p_st ~= 'failed' and p_st ~= 'recovery_required' then
            return {'CORRUPT_PARENT_STATUS'}
        end
    end

    local s_yield_key = 'task.assignment_yield:' .. parent_task_id .. ':' .. stored['assignment_id']
    if redis.call('EXISTS', s_yield_key) == 0 then return {'CORRUPT_YIELD'} end
    if key_type(s_yield_key) ~= 'string' then return {'WRONG_KEY_TYPE_YIELD'} end
    local ok_sy, sy = pcall(cjson.decode, redis.call('GET', s_yield_key))
    if not ok_sy or type(sy) ~= 'table' or
       sy['wave_id'] ~= s_wave_id or sy['assignment_id'] ~= stored['assignment_id'] or
       sy['proposal_id'] ~= stored['proposal_id'] or sy['parent_task_id'] ~= parent_task_id or
       sy['delegation_decision_id'] ~= stored['delegation_decision_id'] or
       sy['continuation_version'] ~= 2 or
       type(sy['required_child_task_ids']) ~= 'table' or #sy['required_child_task_ids'] ~= #s_children then
        return {'CORRUPT_YIELD'}
    end

    if redis.call('EXISTS', parent_children_key) == 0 then return {'CORRUPT_CHILDREN'} end
    if key_type(parent_children_key) ~= 'set' then return {'WRONG_KEY_TYPE_CHILDREN'} end
    if redis.call('SCARD', parent_children_key) ~= #s_children then return {'CORRUPT_CHILDREN'} end

    for idx = 1, #s_children do
        local cid = s_children[idx]
        if type(cid) ~= 'string' or cid == '' or
           sw['required_child_task_ids'][idx] ~= cid or
           sc['required_child_task_ids'][idx] ~= cid or
           sy['required_child_task_ids'][idx] ~= cid or
           (has_resolved and srw['required_child_task_ids'][idx] ~= cid) then
            return {'CORRUPT_WAVE_PROJECTION'}
        end
        if redis.call('SISMEMBER', parent_children_key, cid) ~= 1 then return {'CORRUPT_CHILDREN'} end

        local cs_k = 'task.status:' .. cid
        local cl_k = 'task.lineage:' .. cid
        local cc_k = 'task.required_capability:' .. cid
        local ci_k = 'task.input:' .. cid .. ':1'
        local cd_k = 'task.claim_deadline:' .. cid
        local cf_k = 'task.fence:' .. cid
        local co_k = 'task.outcome:' .. cid
        local cr_k = 'task.result:' .. cid
        local ccl_k = 'task.claim:' .. cid
        local cm_k = 'task.child_completion:' .. cid
        if redis.call('EXISTS', cs_k) == 0 or key_type(cs_k) ~= 'string' or
           redis.call('EXISTS', cl_k) == 0 or key_type(cl_k) ~= 'string' or
           redis.call('EXISTS', cc_k) == 0 or key_type(cc_k) ~= 'string' or
           redis.call('EXISTS', ci_k) == 0 or key_type(ci_k) ~= 'string' or
           redis.call('EXISTS', cd_k) == 0 or key_type(cd_k) ~= 'string' or
           redis.call('EXISTS', cf_k) == 0 or key_type(cf_k) ~= 'string' then
            return {'CORRUPT_CHILD_PROJECTION'}
        end
        local ok_cs, cs = pcall(cjson.decode, redis.call('GET', cs_k))
        local ok_cl2, cl2 = pcall(cjson.decode, redis.call('GET', cl_k))
        local ok_cc, cc = pcall(cjson.decode, redis.call('GET', cc_k))
        local ok_ci, ci = pcall(cjson.decode, redis.call('GET', ci_k))
        local cf_num = tonumber(redis.call('GET', cf_k))
        if not ok_cs or type(cs) ~= 'table' or cs['task_id'] ~= cid or type(cs['status']) ~= 'string' or
           not ok_cl2 or type(cl2) ~= 'table' or cl2['task_id'] ~= cid or cl2['parent_task_id'] ~= parent_task_id or cl2['root_task_id'] ~= parent_task_id or
           not ok_cc or type(cc) ~= 'table' or cc['task_id'] ~= cid or cc['required_capability'] ~= 'analysis' or
           not ok_ci or type(ci) ~= 'table' or ci['task_id'] ~= cid or ci['input_version'] ~= 1 or
           type(ci['artifact_references']) ~= 'table' or #ci['artifact_references'] < 1 or
           type(ci['artifact_references'][1]) ~= 'table' or ci['artifact_references'][1]['producing_task_id'] ~= parent_task_id or
           redis.call('GET', cd_k) ~= s_deadline or
           not cf_num or cf_num < 0 or math.floor(cf_num) ~= cf_num then
            return {'CORRUPT_CHILD_PROJECTION'}
        end

        local c_st = cs['status']
        local claim_exists = (redis.call('EXISTS', ccl_k) == 1)
        local outcome_exists = (redis.call('EXISTS', co_k) == 1)
        local result_exists = (redis.call('EXISTS', cr_k) == 1)
        local marker_exists = (redis.call('EXISTS', cm_k) == 1)

        if c_st == 'pending' then
            if has_resolved or cf_num ~= 0 or claim_exists or outcome_exists or result_exists or marker_exists then
                return {'CORRUPT_CHILD_PROJECTION'}
            end
        elseif c_st == 'working' then
            if has_resolved or cf_num < 1 or not claim_exists or outcome_exists or result_exists or marker_exists then
                return {'CORRUPT_CHILD_PROJECTION'}
            end
            if key_type(ccl_k) ~= 'string' then return {'CORRUPT_CHILD_PROJECTION'} end
            local ok_ccl, ccl = pcall(cjson.decode, redis.call('GET', ccl_k))
            if not ok_ccl or type(ccl) ~= 'table' or ccl['task_id'] ~= cid or
               ccl['fencing_epoch'] ~= cf_num or ccl['execution_kind'] ~= 'delegated_child' or
               ccl['assigned_capability'] ~= 'analysis' then
                return {'CORRUPT_CHILD_PROJECTION'}
            end
        elseif c_st == 'recovery_required' then
            if has_resolved or cf_num < 1 or claim_exists or outcome_exists or result_exists or marker_exists then
                return {'CORRUPT_CHILD_PROJECTION'}
            end
            local ce_k = 'task.claim_expiry:' .. cid
            if redis.call('EXISTS', ce_k) == 0 or key_type(ce_k) ~= 'string' then
                return {'CORRUPT_CHILD_PROJECTION'}
            end
            local ok_ce, ce = pcall(cjson.decode, redis.call('GET', ce_k))
            if not ok_ce or type(ce) ~= 'table' or ce['task_id'] ~= cid or
               ce['expired_fencing_epoch'] ~= cf_num or
               ce['prior_claimable_status'] ~= 'pending' or
               ce['execution_kind'] ~= 'delegated_child' then
                return {'CORRUPT_CHILD_PROJECTION'}
            end
        elseif c_st == 'completed' or c_st == 'failed' or c_st == 'claim_timed_out' then
            if claim_exists or not outcome_exists or not marker_exists then
                return {'CORRUPT_CHILD_PROJECTION'}
            end
            if key_type(co_k) ~= 'string' or key_type(cm_k) ~= 'string' then
                return {'CORRUPT_CHILD_PROJECTION'}
            end
            local ok_co, co = pcall(cjson.decode, redis.call('GET', co_k))
            local ok_cm, cm = pcall(cjson.decode, redis.call('GET', cm_k))
            if not ok_co or type(co) ~= 'table' or co['task_id'] ~= cid or co['terminal_status'] ~= c_st or
               not ok_cm or type(cm) ~= 'table' or cm['child_task_id'] ~= cid or
               cm['parent_task_id'] ~= parent_task_id or cm['outcome_id'] ~= co['outcome_id'] or
               cm['continuation_version'] ~= 2 then
                return {'CORRUPT_CHILD_PROJECTION'}
            end
            if c_st == 'completed' then
                if cf_num < 1 or not result_exists or key_type(cr_k) ~= 'string' then
                    return {'CORRUPT_CHILD_PROJECTION'}
                end
                local ok_cr, cr = pcall(cjson.decode, redis.call('GET', cr_k))
                if not ok_cr or type(cr) ~= 'table' or cr['task_id'] ~= cid or cr['status'] ~= 'completed' then
                    return {'CORRUPT_CHILD_PROJECTION'}
                end
            elseif result_exists then
                return {'CORRUPT_CHILD_PROJECTION'}
            end
            if has_resolved then
                local snap = srw['child_outcomes'][cid]
                if type(snap) ~= 'table' or snap['status'] ~= c_st or snap['outcome_id'] ~= co['outcome_id'] then
                    return {'CORRUPT_CHILD_PROJECTION'}
                end
            end
        elseif c_st == 'ended / barrier_resolved' then
            if not has_resolved or claim_exists or outcome_exists or result_exists or marker_exists then
                return {'CORRUPT_CHILD_PROJECTION'}
            end
            local snap = srw['child_outcomes'][cid]
            if type(snap) ~= 'table' or snap['status'] ~= 'ended / barrier_resolved' or snap['outcome_id'] ~= cjson.null then
                return {'CORRUPT_CHILD_PROJECTION'}
            end
        else
            return {'CORRUPT_CHILD_PROJECTION'}
        end
    end

    return {'IDEMPOTENT', stored_raw}
end

-- 2. Validate child_count and preserialized payloads
if not child_count or math.floor(child_count) ~= child_count or child_count < 2 or child_count > 4 then
    return {'POLICY_BOUNDS_VIOLATION'}
end
local ok_w_pre, w_pre = pcall(cjson.decode, pre_wave_json)
local ok_c_pre, c_pre = pcall(cjson.decode, pre_continuation_json)
local ok_p_pre, p_pre = pcall(cjson.decode, pre_proposal_json)
local ok_ps_pre, ps_pre = pcall(cjson.decode, pre_parent_status_json)
local ok_y_pre, y_pre = pcall(cjson.decode, pre_yield_json)
if not ok_w_pre or type(w_pre) ~= 'table' or
   not ok_c_pre or type(c_pre) ~= 'table' or
   not ok_p_pre or type(p_pre) ~= 'table' or
   not ok_ps_pre or type(ps_pre) ~= 'table' or
   not ok_y_pre or type(y_pre) ~= 'table' then
    return {'CORRUPT_CANDIDATE_PAYLOAD'}
end
if w_pre['wave_id'] ~= wave_id or c_pre['wave_id'] ~= wave_id or p_pre['wave_id'] ~= wave_id or y_pre['wave_id'] ~= wave_id or
   w_pre['parent_task_id'] ~= parent_task_id or c_pre['parent_task_id'] ~= parent_task_id or p_pre['parent_task_id'] ~= parent_task_id or y_pre['parent_task_id'] ~= parent_task_id or
   w_pre['barrier_policy'] ~= barrier_policy or p_pre['barrier_policy'] ~= barrier_policy or
   w_pre['state'] ~= 'unresolved' or c_pre['state'] ~= 'active' or
   c_pre['continuation_version'] ~= 2 or c_pre['version'] ~= 2 or
   p_pre['fencing_epoch'] ~= expected_epoch or p_pre['delegation_grant_id'] ~= delegation_grant_id or p_pre['delegation_decision_id'] ~= delegation_decision_id or
   ps_pre['task_id'] ~= parent_task_id or ps_pre['status'] ~= 'waiting_for_children' or
   y_pre['assignment_id'] ~= assignment_id or y_pre['proposal_id'] ~= proposal_id or y_pre['delegation_decision_id'] ~= delegation_decision_id or y_pre['principal_id'] ~= principal_id or y_pre['continuation_version'] ~= 2 or y_pre['version'] ~= 2 or y_pre['yield_reason'] ~= 'delegation_wave' then
    return {'WAVE_CONSISTENCY_MISMATCH'}
end
if type(w_pre['required_child_task_ids']) ~= 'table' or #w_pre['required_child_task_ids'] ~= child_count or
   type(c_pre['required_child_task_ids']) ~= 'table' or #c_pre['required_child_task_ids'] ~= child_count or
   type(p_pre['required_child_task_ids']) ~= 'table' or #p_pre['required_child_task_ids'] ~= child_count or
   type(y_pre['required_child_task_ids']) ~= 'table' or #y_pre['required_child_task_ids'] ~= child_count then
    return {'WAVE_CONSISTENCY_MISMATCH'}
end
for idx = 1, child_count do
    if w_pre['required_child_task_ids'][idx] ~= c_pre['required_child_task_ids'][idx] or
       w_pre['required_child_task_ids'][idx] ~= p_pre['required_child_task_ids'][idx] or
       w_pre['required_child_task_ids'][idx] ~= y_pre['required_child_task_ids'][idx] then
        return {'WAVE_CONSISTENCY_MISMATCH'}
    end
end
if barrier_policy == 'PARTIAL' then
    local req_sc = tonumber(req_success_count_raw)
    if not req_sc or req_sc < 1 or req_sc > child_count or w_pre['required_success_count'] ~= req_sc or p_pre['required_success_count'] ~= req_sc then
        return {'POLICY_BOUNDS_VIOLATION'}
    end
end
if not validate_fraction_value(w_pre['required_success_fraction'], barrier_policy, expected_frac_num_raw, expected_frac_den_raw) or
   not validate_fraction_value(p_pre['required_success_fraction'], barrier_policy, expected_frac_num_raw, expected_frac_den_raw) then
    return {'POLICY_BOUNDS_VIOLATION'}
end

local cand_err = validate_candidate_child(c1_id, w_pre['required_child_task_ids'][1], c1_status_json, c1_lineage_json, c1_cap_json, c1_input_json, c1_hint_json, c1_art_id, c1_art_ver)
if cand_err then return {cand_err} end
cand_err = validate_candidate_child(c2_id, w_pre['required_child_task_ids'][2], c2_status_json, c2_lineage_json, c2_cap_json, c2_input_json, c2_hint_json, c2_art_id, c2_art_ver)
if cand_err then return {cand_err} end
if c1_id == c2_id then return {'WAVE_CONSISTENCY_MISMATCH'} end
if child_count >= 3 then
    cand_err = validate_candidate_child(c3_id, w_pre['required_child_task_ids'][3], c3_status_json, c3_lineage_json, c3_cap_json, c3_input_json, c3_hint_json, c3_art_id, c3_art_ver)
    if cand_err then return {cand_err} end
    if c3_id == c1_id or c3_id == c2_id then return {'WAVE_CONSISTENCY_MISMATCH'} end
end
if child_count >= 4 then
    cand_err = validate_candidate_child(c4_id, w_pre['required_child_task_ids'][4], c4_status_json, c4_lineage_json, c4_cap_json, c4_input_json, c4_hint_json, c4_art_id, c4_art_ver)
    if cand_err then return {cand_err} end
    if c4_id == c1_id or c4_id == c2_id or c4_id == c3_id then return {'WAVE_CONSISTENCY_MISMATCH'} end
end

-- 3. Validate Parent Task Existence & Status
if redis.call('EXISTS', parent_status_key) == 0 then return {'PARENT_NOT_FOUND'} end
if key_type(parent_status_key) ~= 'string' then return {'WRONG_KEY_TYPE_PARENT_STATUS'} end
local ok_ps, parent_status = pcall(cjson.decode, redis.call('GET', parent_status_key))
if not ok_ps or type(parent_status) ~= 'table' then return {'CORRUPT_PARENT_STATUS'} end
if parent_status['status'] ~= 'working' then
    return {'PARENT_NOT_WORKING', parent_status['status']}
end

-- 4. Validate Active Claim
if redis.call('EXISTS', parent_claim_key) == 0 then return {'CLAIM_NOT_FOUND'} end
if key_type(parent_claim_key) ~= 'string' then return {'WRONG_KEY_TYPE_CLAIM'} end
local ok_cl, claim = pcall(cjson.decode, redis.call('GET', parent_claim_key))
if not ok_cl or type(claim) ~= 'table' then return {'CORRUPT_CLAIM'} end
if type(claim['turn_id']) ~= 'string' or claim['turn_id'] == '' then return {'CORRUPT_CLAIM'} end
if claim['task_id'] ~= parent_task_id then return {'CLAIM_NOT_FOUND'} end
if claim['assignment_id'] ~= assignment_id then return {'ASSIGNMENT_MISMATCH'} end
if claim['runtime_principal_id'] ~= principal_id then return {'PRINCIPAL_MISMATCH'} end
if claim['bound_agent_id'] ~= bound_agent_id then return {'BOUND_AGENT_MISMATCH'} end
if claim['state'] ~= 'acknowledged' then return {'CLAIM_NOT_ACKNOWLEDGED'} end
if claim['assigned_capability'] ~= 'research' then return {'CAPABILITY_POLICY_VIOLATION'} end
if claim['execution_kind'] ~= 'initial' then return {'EXECUTION_KIND_MISMATCH'} end
if claim['delegation_grant_id'] ~= delegation_grant_id then return {'GRANT_ID_MISMATCH'} end

-- 5. Wave, Grant & Children Budget Validation
if redis.call('EXISTS', waves_key) == 1 then
    if key_type(waves_key) ~= 'set' then return {'WRONG_KEY_TYPE_WAVES'} end
    if redis.call('SISMEMBER', waves_key, '1') == 1 then return {'WAVE_ALREADY_CONSUMED'} end
end
if redis.call('EXISTS', grants_key) == 1 then
    if key_type(grants_key) ~= 'set' then return {'WRONG_KEY_TYPE_GRANTS'} end
    if redis.call('SISMEMBER', grants_key, delegation_grant_id) == 1 then return {'GRANT_ALREADY_CONSUMED'} end
end
if redis.call('EXISTS', decisions_key) == 1 then
    if key_type(decisions_key) ~= 'set' then return {'WRONG_KEY_TYPE_DECISIONS'} end
end
if redis.call('EXISTS', parent_children_key) == 1 then
    if key_type(parent_children_key) ~= 'set' then return {'WRONG_KEY_TYPE_CHILDREN'} end
    local count = redis.call('SCARD', parent_children_key)
    if count and count >= 1 then return {'CHILD_LIMIT_EXCEEDED'} end
end

-- 6. Validate Required Keys Existence & Types
if redis.call('EXISTS', parent_fence_key) == 0 then return {'FENCE_NOT_FOUND'} end
if key_type(parent_fence_key) ~= 'string' then return {'WRONG_KEY_TYPE_FENCE'} end

if redis.call('EXISTS', parent_lineage_key) == 0 then return {'PARENT_LINEAGE_NOT_FOUND'} end
if key_type(parent_lineage_key) ~= 'string' then return {'WRONG_KEY_TYPE_PARENT_LINEAGE'} end

if redis.call('EXISTS', root_authority_key) == 0 then return {'ROOT_AUTHORITY_NOT_FOUND'} end
if key_type(root_authority_key) ~= 'string' then return {'WRONG_KEY_TYPE_ROOT_AUTHORITY'} end

if redis.call('EXISTS', grant_record_key) == 0 then return {'GRANT_NOT_FOUND'} end
if key_type(grant_record_key) ~= 'string' then return {'WRONG_KEY_TYPE_GRANT_RECORD'} end

if redis.call('EXISTS', parent_input_key) == 0 then return {'PARENT_INPUT_NOT_FOUND'} end
if key_type(parent_input_key) ~= 'string' then return {'WRONG_KEY_TYPE_PARENT_INPUT'} end

if redis.call('EXISTS', continuation_latest_key) == 1 then
    if key_type(continuation_latest_key) ~= 'string' then return {'WRONG_KEY_TYPE_CONTINUATION_LATEST'} end
end
if redis.call('EXISTS', hint_queue_key) == 1 then
    if key_type(hint_queue_key) ~= 'list' then return {'WRONG_KEY_TYPE_HINT_QUEUE'} end
end

-- 7. Validate Target Keys Do Not Exist Before Write
if redis.call('EXISTS', wave_key) == 1 then return {'WAVE_ALREADY_EXISTS'} end
if redis.call('EXISTS', yield_key) == 1 then return {'YIELD_ALREADY_EXISTS'} end
if redis.call('EXISTS', continuation_key) == 1 or redis.call('EXISTS', continuation_latest_key) == 1 then
    return {'CONTINUATION_EXISTS'}
end
if child_keys_exist(c1_status_key, c1_lineage_key, c1_capability_key, c1_input_key, c1_deadline_key, c1_fence_key) or
   child_keys_exist(c2_status_key, c2_lineage_key, c2_capability_key, c2_input_key, c2_deadline_key, c2_fence_key) then
    return {'CHILD_TASK_ALREADY_EXISTS'}
end
if child_count >= 3 and child_keys_exist(c3_status_key, c3_lineage_key, c3_capability_key, c3_input_key, c3_deadline_key, c3_fence_key) then
    return {'CHILD_TASK_ALREADY_EXISTS'}
end
if child_count >= 4 and child_keys_exist(c4_status_key, c4_lineage_key, c4_capability_key, c4_input_key, c4_deadline_key, c4_fence_key) then
    return {'CHILD_TASK_ALREADY_EXISTS'}
end

-- 8. Fencing Epoch Validation
local cur_fence = tonumber(redis.call('GET', parent_fence_key))
if not cur_fence or not expected_epoch or cur_fence ~= expected_epoch then
    return {'FENCE_MISMATCH', tostring(cur_fence)}
end
local claim_epoch = tonumber(claim['fencing_epoch'])
if not claim_epoch or claim_epoch ~= expected_epoch then
    return {'FENCE_MISMATCH', tostring(claim_epoch)}
end

-- 9. Parent Lineage (must be root)
local ok_l, lineage = pcall(cjson.decode, redis.call('GET', parent_lineage_key))
if not ok_l or type(lineage) ~= 'table' then return {'CORRUPT_LINEAGE'} end
if type(lineage['task_id']) ~= 'string' or type(lineage['root_task_id']) ~= 'string' then
    return {'CORRUPT_PARENT_LINEAGE'}
end
if lineage['task_id'] ~= parent_task_id or lineage['root_task_id'] ~= parent_task_id then
    return {'CORRUPT_PARENT_LINEAGE'}
end
if lineage['parent_task_id'] ~= cjson.null then
    return {'PARENT_NOT_ROOT'}
end

-- 10. Root Authority Validation (v2 wave bounds)
local ok_ra, auth = pcall(cjson.decode, redis.call('GET', root_authority_key))
if not ok_ra or type(auth) ~= 'table' then return {'CORRUPT_ROOT_AUTHORITY'} end
if auth['root_task_id'] ~= parent_task_id then return {'ROOT_AUTHORITY_BINDING_MISMATCH'} end
local d = auth['max_delegation_depth']
local w = auth['max_delegation_waves']
local c = auth['max_children_per_wave']
local av = auth['version']
if type(av) ~= 'number' or av ~= 2 or
   type(d) ~= 'number' or d ~= 1 or math.floor(d) ~= d or
   type(w) ~= 'number' or w ~= 1 or math.floor(w) ~= w or
   type(c) ~= 'number' or math.floor(c) ~= c or c < 2 or c > 4 or child_count > c then
    return {'POLICY_BOUNDS_VIOLATION'}
end
local edges = auth['allowed_delegation_edges']
if type(edges) ~= 'table' then return {'CORRUPT_ROOT_AUTHORITY'} end
local edge_allowed = false
for _, edge in ipairs(edges) do
    if type(edge) == 'table' and edge[1] == 'research' and edge[2] == 'analysis' then
        edge_allowed = true
        break
    end
end
if not edge_allowed then return {'DELEGATION_EDGE_NOT_PERMITTED'} end

-- 11. Authoritative Delegation Grant V2 Validation
local ok_g, grant = pcall(cjson.decode, redis.call('GET', grant_record_key))
if not ok_g or type(grant) ~= 'table' then return {'CORRUPT_GRANT'} end
if type(grant['turn_id']) ~= 'string' or grant['turn_id'] == '' then return {'CORRUPT_GRANT'} end
if grant['turn_id'] ~= claim['turn_id'] then return {'GRANT_TURN_MISMATCH'} end
if grant['grant_id'] ~= delegation_grant_id then return {'GRANT_ID_MISMATCH'} end
if grant['root_task_id'] ~= parent_task_id then return {'GRANT_ROOT_BINDING_MISMATCH'} end
if grant['issued_to_assignment_id'] ~= assignment_id then return {'GRANT_ASSIGNMENT_MISMATCH'} end
if grant['issued_to_principal_id'] ~= principal_id then return {'GRANT_PRINCIPAL_MISMATCH'} end
if grant['source_capability'] ~= 'research' or grant['target_capability'] ~= 'analysis' then
    return {'GRANT_CAPABILITY_MISMATCH'}
end
local g_epoch = tonumber(grant['fencing_epoch'])
if not g_epoch or g_epoch ~= expected_epoch then return {'GRANT_EPOCH_MISMATCH'} end
local gv = grant['version']
local gd = grant['max_depth']
local gw = grant['max_waves']
local gc = grant['max_children']
if type(gv) ~= 'number' or gv ~= 2 or
   type(gd) ~= 'number' or gd ~= 1 or math.floor(gd) ~= gd or
   type(gw) ~= 'number' or gw ~= 1 or math.floor(gw) ~= gw or
   type(gc) ~= 'number' or math.floor(gc) ~= gc or gc < 2 or gc > c or child_count > gc then
    return {'GRANT_BOUNDS_VIOLATION'}
end

-- 12. Sealed Artifact Validation for all wave items
local art_err = validate_artifact(art1_key, c1_art_id, c1_art_ver, claim['turn_id'])
if art_err then return {art_err} end
art_err = validate_artifact(art2_key, c2_art_id, c2_art_ver, claim['turn_id'])
if art_err then return {art_err} end
if child_count >= 3 then
    art_err = validate_artifact(art3_key, c3_art_id, c3_art_ver, claim['turn_id'])
    if art_err then return {art_err} end
end
if child_count >= 4 then
    art_err = validate_artifact(art4_key, c4_art_id, c4_art_ver, claim['turn_id'])
    if art_err then return {art_err} end
end

-- 13. Authoritative Parent TaskInput Validation
local ok_pi, pinput = pcall(cjson.decode, redis.call('GET', parent_input_key))
if not ok_pi or type(pinput) ~= 'table' then return {'CORRUPT_PARENT_INPUT'} end
if type(pinput['task_id']) ~= 'string' or pinput['task_id'] ~= parent_task_id then
    return {'PARENT_INPUT_TASK_MISMATCH'}
end
local pi_ver = pinput['input_version']
if type(pi_ver) ~= 'number' or math.floor(pi_ver) ~= pi_ver or pi_ver ~= expected_parent_input_version then
    return {'PARENT_INPUT_VERSION_MISMATCH'}
end
local pi_cver = pinput['version']
if type(pi_cver) ~= 'number' or math.floor(pi_cver) ~= pi_cver or pi_cver ~= 1 then
    return {'UNSUPPORTED_PARENT_INPUT_VERSION'}
end
if type(pinput['prompt']) ~= 'string' or string.match(pinput['prompt'], '^%s*$') then
    return {'CORRUPT_PARENT_INPUT'}
end
if type(pinput['created_at']) ~= 'string' or pinput['created_at'] == '' then
    return {'CORRUPT_PARENT_INPUT'}
end

-- =============================================================================
-- PHASE 2: Mechanically Non-Failing Writes (Atomic Commit)
-- =============================================================================
redis.call('SET', proposal_key, pre_proposal_json)
redis.call('SET', wave_key, pre_wave_json)

redis.call('SET', c1_status_key, c1_status_json)
redis.call('SET', c1_lineage_key, c1_lineage_json)
redis.call('SET', c1_capability_key, c1_cap_json)
redis.call('SET', c1_input_key, c1_input_json)
redis.call('SET', c1_deadline_key, child_deadline_iso)
redis.call('SET', c1_fence_key, '0')
redis.call('SADD', parent_children_key, c1_id)
redis.call('RPUSH', hint_queue_key, c1_hint_json)

redis.call('SET', c2_status_key, c2_status_json)
redis.call('SET', c2_lineage_key, c2_lineage_json)
redis.call('SET', c2_capability_key, c2_cap_json)
redis.call('SET', c2_input_key, c2_input_json)
redis.call('SET', c2_deadline_key, child_deadline_iso)
redis.call('SET', c2_fence_key, '0')
redis.call('SADD', parent_children_key, c2_id)
redis.call('RPUSH', hint_queue_key, c2_hint_json)

if child_count >= 3 then
    redis.call('SET', c3_status_key, c3_status_json)
    redis.call('SET', c3_lineage_key, c3_lineage_json)
    redis.call('SET', c3_capability_key, c3_cap_json)
    redis.call('SET', c3_input_key, c3_input_json)
    redis.call('SET', c3_deadline_key, child_deadline_iso)
    redis.call('SET', c3_fence_key, '0')
    redis.call('SADD', parent_children_key, c3_id)
    redis.call('RPUSH', hint_queue_key, c3_hint_json)
end

if child_count >= 4 then
    redis.call('SET', c4_status_key, c4_status_json)
    redis.call('SET', c4_lineage_key, c4_lineage_json)
    redis.call('SET', c4_capability_key, c4_cap_json)
    redis.call('SET', c4_input_key, c4_input_json)
    redis.call('SET', c4_deadline_key, child_deadline_iso)
    redis.call('SET', c4_fence_key, '0')
    redis.call('SADD', parent_children_key, c4_id)
    redis.call('RPUSH', hint_queue_key, c4_hint_json)
end

redis.call('SET', continuation_key, pre_continuation_json)
redis.call('SET', continuation_latest_key, '2')
redis.call('SET', parent_status_key, pre_parent_status_json)
redis.call('DEL', parent_claim_key)
redis.call('SET', yield_key, pre_yield_json)
redis.call('SADD', waves_key, '1')
redis.call('SADD', grants_key, delegation_grant_id)
redis.call('SADD', decisions_key, delegation_decision_id)

return {'OK', pre_proposal_json}
"""

COMPLETE_TURN_LUA = """
local completion_key = KEYS[1]
local status_key = KEYS[2]
local claim_key = KEYS[3]
local fence_key = KEYS[4]
local outcome_key = KEYS[5]
local result_key = KEYS[6]
local audit_key = KEYS[7]
local lineage_key = KEYS[8]
local parent_status_key = KEYS[9]
local parent_children_key = KEYS[10]
local continuation_key = KEYS[11]
local continuation_latest_key = KEYS[12]
local child_marker_key = KEYS[13]
local hint_queue_key = KEYS[14]

local task_id = ARGV[1]
local assignment_id = ARGV[2]
local principal_id = ARGV[3]
local bound_agent_id = ARGV[4]
local execution_kind = ARGV[5]
local cont_ver_raw = ARGV[6]
local epoch_raw = ARGV[7]
local completion_command_id = ARGV[8]
local semantic_fingerprint = ARGV[9]
local proof_digest = ARGV[10]
local terminal_status = ARGV[11]
local failure_code = ARGV[12]
local outcome_json = ARGV[13]
local result_json = ARGV[14]
local audit_json = ARGV[15]
local parent_task_id = ARGV[16]
local root_task_id = ARGV[17]
local binding_json = ARGV[18]
local hint_json = ARGV[19]
local marker_json = ARGV[20]
local status_json = ARGV[21]
local parent_status_json = ARGV[22]

local is_child = (execution_kind == 'delegated_child')

local function key_type(key)
    local t = redis.call('TYPE', key)
    if type(t) == 'table' then return t['ok'] end
    return tostring(t)
end

local function decode_object(raw)
    local ok, tbl = pcall(cjson.decode, raw)
    if not ok or type(tbl) ~= 'table' then return nil end
    return tbl
end

local function has_cont_ver(stored_cv)
    return stored_cv ~= nil and stored_cv ~= cjson.null
end

-- =============================================================================
-- PHASE 1: Completion-Command Replay Check (FIRST STORE-LEVEL OPERATION)
-- =============================================================================
if redis.call('EXISTS', completion_key) == 1 then
    if key_type(completion_key) ~= 'string' then return {'WRONG_KEY_TYPE_COMPLETION'} end
    local stored = decode_object(redis.call('GET', completion_key))
    if not stored then return {'CORRUPT_COMPLETION'} end
    local stored_cv = stored['continuation_version']
    local stored_has_cv = has_cont_ver(stored_cv)
    local arg_has_cv = (cont_ver_raw ~= '')
    local matches = (
        stored['completion_command_id'] == completion_command_id and
        stored['task_id'] == task_id and
        stored['assignment_id'] == assignment_id and
        stored['runtime_principal_id'] == principal_id and
        stored['bound_agent_id'] == bound_agent_id and
        stored['execution_kind'] == execution_kind and
        stored['semantic_fingerprint'] == semantic_fingerprint and
        stored_has_cv == arg_has_cv and
        (not arg_has_cv or tonumber(stored_cv) == tonumber(cont_ver_raw))
    )
    if matches then
        if key_type(outcome_key) ~= 'string' then return {'CORRUPT_OUTCOME'} end
        local stored_outcome = decode_object(redis.call('GET', outcome_key))
        if not stored_outcome then return {'CORRUPT_OUTCOME'} end
        if stored_outcome['task_id'] ~= task_id then return {'CORRUPT_OUTCOME'} end
        local stored_status = stored_outcome['terminal_status']
        if stored_status ~= 'completed' and stored_status ~= 'failed' then
            return {'CORRUPT_OUTCOME'}
        end
        local result_exists = (redis.call('EXISTS', result_key) == 1)
        if stored_status == 'completed' and not result_exists then
            return {'CORRUPT_RESULT'}
        end
        if stored_status == 'failed' and result_exists then
            return {'CORRUPT_RESULT'}
        end
        local stored_result_raw = ''
        if result_exists then
            if key_type(result_key) ~= 'string' then return {'CORRUPT_RESULT'} end
            stored_result_raw = redis.call('GET', result_key)
            local stored_result_tbl = decode_object(stored_result_raw)
            if not stored_result_tbl then return {'CORRUPT_RESULT'} end
            if stored_result_tbl['task_id'] ~= task_id or stored_result_tbl['status'] ~= 'completed' then
                return {'CORRUPT_RESULT'}
            end
        end
        return {'IDEMPOTENT', redis.call('GET', outcome_key), stored_result_raw}
    end
    return {'COMMAND_CONFLICT'}
end

-- =============================================================================
-- PHASE 1 (cont.): Strictly Read-Only Validations (Zero Mutations)
-- =============================================================================

-- Candidate input well-formedness: numerics, status/result consistency, JSON payloads
local exp_epoch = tonumber(epoch_raw)
if not exp_epoch or exp_epoch < 1 or math.floor(exp_epoch) ~= exp_epoch then
    return {'CORRUPT_COMPLETION_INPUT'}
end
local arg_cv = nil
if cont_ver_raw ~= '' then
    arg_cv = tonumber(cont_ver_raw)
    if not arg_cv or arg_cv < 1 or math.floor(arg_cv) ~= arg_cv then
        return {'CORRUPT_COMPLETION_INPUT'}
    end
end
local has_result = (result_json ~= '')
if has_result and terminal_status ~= 'completed' then
    return {'CORRUPT_COMPLETION_INPUT'}
end
if (not has_result) and terminal_status ~= 'failed' then
    return {'CORRUPT_COMPLETION_INPUT'}
end
if not decode_object(outcome_json) then return {'CORRUPT_COMPLETION_INPUT'} end
if not decode_object(audit_json) then return {'CORRUPT_COMPLETION_INPUT'} end
if not decode_object(binding_json) then return {'CORRUPT_COMPLETION_INPUT'} end
if not decode_object(status_json) then return {'CORRUPT_COMPLETION_INPUT'} end
if has_result then
    local result_tbl = decode_object(result_json)
    if not result_tbl then return {'CORRUPT_COMPLETION_INPUT'} end
    if result_tbl['task_id'] ~= task_id then return {'RESULT_TASK_MISMATCH'} end
    if result_tbl['status'] ~= 'completed' then return {'RESULT_STATUS_MISMATCH'} end
end
if is_child then
    if not decode_object(hint_json) then return {'CORRUPT_COMPLETION_INPUT'} end
    if not decode_object(marker_json) then return {'CORRUPT_COMPLETION_INPUT'} end
    if not decode_object(parent_status_json) then return {'CORRUPT_COMPLETION_INPUT'} end
end

-- Live task status must exist and be an eligible nonterminal working turn
if redis.call('EXISTS', status_key) == 0 then
    return {'TASK_NOT_FOUND'}
end
if key_type(status_key) ~= 'string' then return {'WRONG_KEY_TYPE_STATUS'} end
local status_tbl = decode_object(redis.call('GET', status_key))
if not status_tbl then return {'CORRUPT_STATUS'} end
local cur_status = status_tbl['status']
if type(cur_status) ~= 'string' or cur_status == '' then
    return {'CORRUPT_STATUS'}
end
if status_tbl['task_id'] ~= task_id then
    return {'CORRUPT_STATUS'}
end
if cur_status ~= 'working' then
    return {'NOT_WORKING', cur_status}
end

-- Active claim must exist, be acknowledged, and bind every turn identity field
if redis.call('EXISTS', claim_key) == 0 then
    return {'CLAIM_NOT_FOUND'}
end
if key_type(claim_key) ~= 'string' then return {'WRONG_KEY_TYPE_CLAIM'} end
local claim = decode_object(redis.call('GET', claim_key))
if not claim then return {'CORRUPT_CLAIM'} end
if claim['assignment_id'] ~= assignment_id then return {'ASSIGNMENT_MISMATCH'} end
if claim['task_id'] ~= task_id then return {'TASK_MISMATCH'} end
if claim['runtime_principal_id'] ~= principal_id then return {'PRINCIPAL_MISMATCH'} end
if claim['bound_agent_id'] ~= bound_agent_id then return {'BOUND_AGENT_MISMATCH'} end
if claim['execution_kind'] ~= execution_kind then return {'EXECUTION_KIND_MISMATCH'} end
local claim_cv = claim['continuation_version']
local claim_has_cv = has_cont_ver(claim_cv)
if claim_has_cv ~= (arg_cv ~= nil) then return {'CONTINUATION_VERSION_MISMATCH'} end
if arg_cv and tonumber(claim_cv) ~= arg_cv then return {'CONTINUATION_VERSION_MISMATCH'} end
if claim['state'] ~= 'acknowledged' then return {'CLAIM_NOT_ACKNOWLEDGED'} end
if claim['proof_digest'] ~= proof_digest then return {'PROOF_MISMATCH'} end

-- Current fencing epoch must equal the claimed epoch and the presented epoch
if redis.call('EXISTS', fence_key) == 0 then
    return {'FENCE_NOT_FOUND'}
end
if key_type(fence_key) ~= 'string' then return {'WRONG_KEY_TYPE_FENCE'} end
local cur_fence = tonumber(redis.call('GET', fence_key))
if not cur_fence or cur_fence ~= exp_epoch then
    return {'FENCE_MISMATCH', tostring(cur_fence)}
end
if tonumber(claim['fencing_epoch']) ~= exp_epoch then
    return {'FENCE_MISMATCH', tostring(claim['fencing_epoch'])}
end

-- Authoritative lineage must bind the completing task (and its parent for children)
if redis.call('EXISTS', lineage_key) == 0 then
    return {'LINEAGE_NOT_FOUND'}
end
if key_type(lineage_key) ~= 'string' then return {'WRONG_KEY_TYPE_LINEAGE'} end
local lineage = decode_object(redis.call('GET', lineage_key))
if not lineage then return {'CORRUPT_LINEAGE'} end
if lineage['task_id'] ~= task_id or lineage['root_task_id'] ~= root_task_id then
    return {'CORRUPT_LINEAGE'}
end
local lineage_parent = lineage['parent_task_id']
if is_child then
    if lineage_parent ~= parent_task_id then return {'CHILD_LINEAGE_MISMATCH'} end
else
    if lineage_parent ~= cjson.null then return {'LINEAGE_KIND_MISMATCH'} end
end

-- No conflicting canonical outcome, result, or audit may already exist
if redis.call('EXISTS', outcome_key) == 1 then
    return {'OUTCOME_CONFLICT'}
end
if redis.call('EXISTS', result_key) == 1 then
    return {'RESULT_CONFLICT'}
end
if redis.call('EXISTS', audit_key) == 1 then
    return {'AUDIT_CONFLICT'}
end

-- Delegated-child parent projection preconditions
if is_child then
    if redis.call('EXISTS', parent_status_key) == 0 then
        return {'PARENT_STATUS_NOT_FOUND'}
    end
    if key_type(parent_status_key) ~= 'string' then return {'WRONG_KEY_TYPE_PARENT_STATUS'} end
    local parent_status = decode_object(redis.call('GET', parent_status_key))
    if not parent_status then return {'CORRUPT_PARENT_STATUS'} end
    if parent_status['task_id'] ~= parent_task_id then
        return {'CORRUPT_PARENT_STATUS'}
    end
    if parent_status['status'] ~= 'waiting_for_children' then
        return {'PARENT_NOT_WAITING', parent_status['status']}
    end

    if redis.call('EXISTS', parent_children_key) == 0 then
        return {'CHILD_NOT_MEMBER'}
    end
    if key_type(parent_children_key) ~= 'set' then return {'WRONG_KEY_TYPE_CHILDREN'} end
    if redis.call('SISMEMBER', parent_children_key, task_id) == 0 then
        return {'CHILD_NOT_MEMBER'}
    end

    if redis.call('EXISTS', continuation_key) == 0 then
        return {'MISSING_CONTINUATION'}
    end
    if key_type(continuation_key) ~= 'string' then return {'WRONG_KEY_TYPE_CONTINUATION'} end
    local continuation = decode_object(redis.call('GET', continuation_key))
    if not continuation then return {'CORRUPT_CONTINUATION'} end
    if continuation['stage'] ~= 'incorporate_analysis_child' then
        return {'CONTINUATION_STAGE_MISMATCH'}
    end
    if continuation['child_task_id'] ~= task_id then
        return {'CONTINUATION_CHILD_MISMATCH'}
    end
    if tonumber(continuation['continuation_version']) ~= 1 then
        return {'CONTINUATION_VERSION_MISMATCH'}
    end
    if redis.call('EXISTS', continuation_latest_key) == 0 then
        return {'CONTINUATION_LATEST_MISMATCH'}
    end
    if key_type(continuation_latest_key) ~= 'string' then return {'WRONG_KEY_TYPE_CONTINUATION_LATEST'} end
    if redis.call('GET', continuation_latest_key) ~= '1' then
        return {'CONTINUATION_LATEST_MISMATCH'}
    end

    if redis.call('EXISTS', child_marker_key) == 1 then
        return {'MARKER_CONFLICT'}
    end

    if redis.call('EXISTS', hint_queue_key) == 1 then
        if key_type(hint_queue_key) ~= 'list' then return {'WRONG_KEY_TYPE_HINT_QUEUE'} end
    end
end

-- =============================================================================
-- PHASE 2: Write Commit (Prevalidated Writes Only)
--
-- Every value stored below was validated in read-only Phase 1, so this section
-- performs only plain SET/DEL/RPUSH commands with pre-serialized values and no
-- decoding, numeric conversion, or fallible branches. Redis executes the script
-- atomically with respect to other clients, so concurrent completions observe
-- either none or all of these writes. No claim is made about rollback after an
-- unexpected command error, host failure, out-of-memory condition, or
-- persistence failure: safety against THOSE hazards comes from keeping every
-- expected validation rejection inside Phase 1, where zero mutation occurs.
-- =============================================================================
redis.call('SET', completion_key, binding_json)
redis.call('SET', outcome_key, outcome_json)
if has_result then
    redis.call('SET', result_key, result_json)
end
redis.call('SET', audit_key, audit_json)
redis.call('SET', status_key, status_json)
redis.call('DEL', claim_key)
if is_child then
    redis.call('SET', child_marker_key, marker_json)
    redis.call('SET', parent_status_key, parent_status_json)
    redis.call('RPUSH', hint_queue_key, hint_json)
end

return {'OK', outcome_json, result_json}
"""

COMPLETE_WAVE_CHILD_TURN_LUA = """
local completion_key = KEYS[1]
local status_key = KEYS[2]
local claim_key = KEYS[3]
local fence_key = KEYS[4]
local outcome_key = KEYS[5]
local result_key = KEYS[6]
local audit_key = KEYS[7]
local lineage_key = KEYS[8]
local child_marker_key = KEYS[9]
local parent_status_key = KEYS[10]
local parent_children_key = KEYS[11]
local parent_lineage_key = KEYS[12]
local continuation_key = KEYS[13]
local continuation_latest_key = KEYS[14]
local wave_key = KEYS[15]
local resolved_wave_key = KEYS[16]
local hint_queue_key = KEYS[17]

local s1_status_key = KEYS[18]
local s1_claim_key = KEYS[19]
local s1_fence_key = KEYS[20]
local s1_outcome_key = KEYS[21]
local s1_result_key = KEYS[22]
local s1_lineage_key = KEYS[23]
local s1_cap_key = KEYS[24]
local s1_input_key = KEYS[25]
local s1_marker_key = KEYS[26]

local s2_status_key = KEYS[27]
local s2_claim_key = KEYS[28]
local s2_fence_key = KEYS[29]
local s2_outcome_key = KEYS[30]
local s2_result_key = KEYS[31]
local s2_lineage_key = KEYS[32]
local s2_cap_key = KEYS[33]
local s2_input_key = KEYS[34]
local s2_marker_key = KEYS[35]

local s3_status_key = KEYS[36]
local s3_claim_key = KEYS[37]
local s3_fence_key = KEYS[38]
local s3_outcome_key = KEYS[39]
local s3_result_key = KEYS[40]
local s3_lineage_key = KEYS[41]
local s3_cap_key = KEYS[42]
local s3_input_key = KEYS[43]
local s3_marker_key = KEYS[44]

local s4_status_key = KEYS[45]
local s4_claim_key = KEYS[46]
local s4_fence_key = KEYS[47]
local s4_outcome_key = KEYS[48]
local s4_result_key = KEYS[49]
local s4_lineage_key = KEYS[50]
local s4_cap_key = KEYS[51]
local s4_input_key = KEYS[52]
local s4_marker_key = KEYS[53]

local task_id = ARGV[1]
local assignment_id = ARGV[2]
local principal_id = ARGV[3]
local bound_agent_id = ARGV[4]
local execution_kind = ARGV[5]
local cont_ver_raw = ARGV[6]
local epoch_raw = ARGV[7]
local completion_command_id = ARGV[8]
local semantic_fingerprint = ARGV[9]
local proof_digest = ARGV[10]
local terminal_status = ARGV[11]
local failure_code = ARGV[12]
local outcome_json = ARGV[13]
local result_json = ARGV[14]
local audit_json = ARGV[15]
local parent_task_id = ARGV[16]
local root_task_id = ARGV[17]
local binding_json = ARGV[18]
local hint_json = ARGV[19]
local marker_json = ARGV[20]
local status_json = ARGV[21]
local parent_status_json = ARGV[22]
local wave_id = ARGV[23]
local completed_at = ARGV[24]
local child_count = tonumber(ARGV[25])
local accepted_wave_fp = ARGV[26]
local pre_cont_sat_json = ARGV[27]
local pre_cont_unsat_json = ARGV[28]
local s1_id = ARGV[29]
local s2_id = ARGV[30]
local s3_id = ARGV[31]
local s4_id = ARGV[32]

local function key_type(key)
    local t = redis.call('TYPE', key)
    if type(t) == 'table' then return t['ok'] end
    return tostring(t)
end

local function decode_object(raw)
    local ok, tbl = pcall(cjson.decode, raw)
    if not ok or type(tbl) ~= 'table' then return nil end
    return tbl
end

local function validate_sibling_readonly(
    sid,
    st_k, cl_k, fn_k, out_k, res_k, lin_k, cap_k, inp_k, mrk_k,
    trig_outcome_id
)
    if redis.call('SISMEMBER', parent_children_key, sid) ~= 1 then
        return 'CHILD_NOT_MEMBER', nil, nil, 0, false
    end

    if redis.call('EXISTS', lin_k) == 0 then return 'LINEAGE_NOT_FOUND', nil, nil, 0, false end
    if key_type(lin_k) ~= 'string' then return 'WRONG_KEY_TYPE_LINEAGE', nil, nil, 0, false end
    local lin = decode_object(redis.call('GET', lin_k))
    if not lin or lin['task_id'] ~= sid or lin['parent_task_id'] ~= parent_task_id or lin['root_task_id'] ~= root_task_id then
        return 'CHILD_LINEAGE_MISMATCH', nil, nil, 0, false
    end

    if redis.call('EXISTS', cap_k) == 0 then return 'MISSING_REQUIRED_CAPABILITY', nil, nil, 0, false end
    if key_type(cap_k) ~= 'string' then return 'WRONG_KEY_TYPE_CAPABILITY', nil, nil, 0, false end
    local cap = decode_object(redis.call('GET', cap_k))
    if not cap or cap['task_id'] ~= sid or cap['required_capability'] ~= 'analysis' then
        return 'CORRUPT_CHILD_CAPABILITY', nil, nil, 0, false
    end

    if redis.call('EXISTS', inp_k) == 0 then return 'MISSING_CHILD_INPUT', nil, nil, 0, false end
    if key_type(inp_k) ~= 'string' then return 'WRONG_KEY_TYPE_INPUT', nil, nil, 0, false end
    local inp = decode_object(redis.call('GET', inp_k))
    if not inp or inp['task_id'] ~= sid or inp['input_version'] ~= 1 or
       type(inp['prompt']) ~= 'string' or string.match(inp['prompt'], '^%s*$') or
       type(inp['artifact_references']) ~= 'table' or #inp['artifact_references'] < 1 then
        return 'CORRUPT_CHILD_INPUT', nil, nil, 0, false
    end

    if redis.call('EXISTS', fn_k) == 0 then return 'FENCE_NOT_FOUND', nil, nil, 0, false end
    if key_type(fn_k) ~= 'string' then return 'WRONG_KEY_TYPE_FENCE', nil, nil, 0, false end
    local s_fence = tonumber(redis.call('GET', fn_k))
    if not s_fence or s_fence < 0 or math.floor(s_fence) ~= s_fence then
        return 'CORRUPT_FENCE', nil, nil, 0, false
    end

    if redis.call('EXISTS', st_k) == 0 then return 'TASK_NOT_FOUND', nil, nil, 0, false end
    if key_type(st_k) ~= 'string' then return 'WRONG_KEY_TYPE_STATUS', nil, nil, 0, false end
    local st = decode_object(redis.call('GET', st_k))
    if not st or st['task_id'] ~= sid or type(st['status']) ~= 'string' or st['status'] == '' then
        return 'CORRUPT_STATUS', nil, nil, 0, false
    end
    local cur_st = st['status']

    if sid == task_id then
        return nil, terminal_status, trig_outcome_id, s_fence, true
    end

    local claim_exists = (redis.call('EXISTS', cl_k) == 1)
    local outcome_exists = (redis.call('EXISTS', out_k) == 1)
    local result_exists = (redis.call('EXISTS', res_k) == 1)
    local marker_exists = (redis.call('EXISTS', mrk_k) == 1)

    if cur_st == 'completed' then
        if claim_exists or s_fence < 1 or not outcome_exists or not result_exists or not marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        if key_type(out_k) ~= 'string' or key_type(res_k) ~= 'string' or key_type(mrk_k) ~= 'string' then
            return 'WRONG_KEY_TYPE_CHILD_OUTCOME', nil, nil, 0, false
        end
        local out_tbl = decode_object(redis.call('GET', out_k))
        local res_tbl = decode_object(redis.call('GET', res_k))
        local mrk_tbl = decode_object(redis.call('GET', mrk_k))
        if not out_tbl or out_tbl['task_id'] ~= sid or out_tbl['terminal_status'] ~= 'completed' or
           type(out_tbl['outcome_id']) ~= 'string' or out_tbl['outcome_id'] == '' or
           not res_tbl or res_tbl['task_id'] ~= sid or res_tbl['status'] ~= 'completed' or
           not mrk_tbl or mrk_tbl['child_task_id'] ~= sid or mrk_tbl['parent_task_id'] ~= parent_task_id or
           mrk_tbl['outcome_id'] ~= out_tbl['outcome_id'] or mrk_tbl['continuation_version'] ~= 2 then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, 'completed', out_tbl['outcome_id'], s_fence, false
    elseif cur_st == 'failed' or cur_st == 'claim_timed_out' then
        if claim_exists or not outcome_exists or result_exists or not marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        if key_type(out_k) ~= 'string' or key_type(mrk_k) ~= 'string' then
            return 'WRONG_KEY_TYPE_CHILD_OUTCOME', nil, nil, 0, false
        end
        local out_tbl = decode_object(redis.call('GET', out_k))
        local mrk_tbl = decode_object(redis.call('GET', mrk_k))
        if not out_tbl or out_tbl['task_id'] ~= sid or out_tbl['terminal_status'] ~= cur_st or
           type(out_tbl['outcome_id']) ~= 'string' or out_tbl['outcome_id'] == '' or
           not mrk_tbl or mrk_tbl['child_task_id'] ~= sid or mrk_tbl['parent_task_id'] ~= parent_task_id or
           mrk_tbl['outcome_id'] ~= out_tbl['outcome_id'] or mrk_tbl['continuation_version'] ~= 2 then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, cur_st, out_tbl['outcome_id'], s_fence, false
    elseif cur_st == 'pending' then
        if claim_exists or outcome_exists or result_exists or marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, 'pending', nil, s_fence, false
    elseif cur_st == 'working' then
        if not claim_exists or outcome_exists or result_exists or marker_exists or s_fence < 1 then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        if key_type(cl_k) ~= 'string' then return 'WRONG_KEY_TYPE_CLAIM', nil, nil, 0, false end
        local cl_tbl = decode_object(redis.call('GET', cl_k))
        if not cl_tbl or cl_tbl['task_id'] ~= sid or cl_tbl['execution_kind'] ~= 'delegated_child' or
           cl_tbl['fencing_epoch'] ~= s_fence or cl_tbl['assigned_capability'] ~= 'analysis' then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, 'working', nil, s_fence, true
    elseif cur_st == 'recovery_required' then
        if claim_exists or outcome_exists or result_exists or marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, 'recovery_required', nil, s_fence, false
    end
    return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
end

-- =============================================================================
-- PHASE 1: Replay Check & Strictly Read-Only Validations (Zero Mutations)
-- =============================================================================

if redis.call('EXISTS', completion_key) == 1 then
    if key_type(completion_key) ~= 'string' then return {'WRONG_KEY_TYPE_COMPLETION'} end
    local stored = decode_object(redis.call('GET', completion_key))
    if not stored then return {'CORRUPT_COMPLETION'} end
    local stored_cv = stored['continuation_version']
    local matches = (
        stored['completion_command_id'] == completion_command_id and
        stored['task_id'] == task_id and
        stored['assignment_id'] == assignment_id and
        stored['runtime_principal_id'] == principal_id and
        stored['bound_agent_id'] == bound_agent_id and
        stored['execution_kind'] == 'delegated_child' and
        stored['semantic_fingerprint'] == semantic_fingerprint and
        stored['proof_digest'] == proof_digest and
        (stored_cv == nil or stored_cv == cjson.null)
    )
    if matches then
        if redis.call('EXISTS', outcome_key) == 0 or key_type(outcome_key) ~= 'string' then
            return {'CORRUPT_OUTCOME'}
        end
        local stored_outcome = decode_object(redis.call('GET', outcome_key))
        if not stored_outcome or stored_outcome['task_id'] ~= task_id or
           stored_outcome['outcome_id'] ~= stored['outcome_id'] then
            return {'CORRUPT_OUTCOME'}
        end
        local stored_status = stored_outcome['terminal_status']
        if stored_status ~= 'completed' and stored_status ~= 'failed' then
            return {'CORRUPT_OUTCOME'}
        end
        local result_exists = (redis.call('EXISTS', result_key) == 1)
        if stored_status == 'completed' and not result_exists then
            return {'CORRUPT_RESULT'}
        end
        if stored_status == 'failed' and result_exists then
            return {'CORRUPT_RESULT'}
        end
        local stored_result_raw = ''
        if result_exists then
            if key_type(result_key) ~= 'string' then return {'CORRUPT_RESULT'} end
            stored_result_raw = redis.call('GET', result_key)
            local stored_result_tbl = decode_object(stored_result_raw)
            if not stored_result_tbl or stored_result_tbl['task_id'] ~= task_id or stored_result_tbl['status'] ~= 'completed' then
                return {'CORRUPT_RESULT'}
            end
        end
        if redis.call('EXISTS', child_marker_key) == 0 or key_type(child_marker_key) ~= 'string' then
            return {'CORRUPT_CHILD_MARKER'}
        end
        local stored_mrk = decode_object(redis.call('GET', child_marker_key))
        if not stored_mrk or stored_mrk['child_task_id'] ~= task_id or
           stored_mrk['parent_task_id'] ~= parent_task_id or
           stored_mrk['outcome_id'] ~= stored['outcome_id'] or
           stored_mrk['continuation_version'] ~= 2 then
            return {'CORRUPT_CHILD_MARKER'}
        end
        return {'IDEMPOTENT', redis.call('GET', outcome_key), stored_result_raw}
    end
    return {'COMMAND_CONFLICT'}
end

-- Candidate inputs validation
if execution_kind ~= 'delegated_child' or cont_ver_raw ~= '' then
    return {'CORRUPT_COMPLETION_INPUT'}
end
local exp_epoch = tonumber(epoch_raw)
if not exp_epoch or exp_epoch < 1 or math.floor(exp_epoch) ~= exp_epoch then
    return {'CORRUPT_COMPLETION_INPUT'}
end
if not child_count or math.floor(child_count) ~= child_count or child_count < 2 or child_count > 4 then
    return {'CORRUPT_COMPLETION_INPUT'}
end
local has_result = (result_json ~= '')
if has_result and terminal_status ~= 'completed' then
    return {'CORRUPT_COMPLETION_INPUT'}
end
if (not has_result) and terminal_status ~= 'failed' then
    return {'CORRUPT_COMPLETION_INPUT'}
end
local cand_outcome = decode_object(outcome_json)
if not cand_outcome or type(cand_outcome['outcome_id']) ~= 'string' or cand_outcome['outcome_id'] == '' then
    return {'CORRUPT_COMPLETION_INPUT'}
end
if not decode_object(audit_json) then return {'CORRUPT_COMPLETION_INPUT'} end
if not decode_object(binding_json) then return {'CORRUPT_COMPLETION_INPUT'} end
if not decode_object(status_json) then return {'CORRUPT_COMPLETION_INPUT'} end
if not decode_object(hint_json) then return {'CORRUPT_COMPLETION_INPUT'} end
if not decode_object(marker_json) then return {'CORRUPT_COMPLETION_INPUT'} end
if not decode_object(parent_status_json) then return {'CORRUPT_COMPLETION_INPUT'} end
local pre_cont_sat = decode_object(pre_cont_sat_json)
local pre_cont_unsat = decode_object(pre_cont_unsat_json)
if not pre_cont_sat or not pre_cont_unsat or
   pre_cont_sat['state'] ~= 'released' or pre_cont_sat['wave_resolution'] ~= 'SATISFIED' or
   pre_cont_unsat['state'] ~= 'released' or pre_cont_unsat['wave_resolution'] ~= 'UNSATISFIABLE' then
    return {'CORRUPT_COMPLETION_INPUT'}
end
if has_result then
    local result_tbl = decode_object(result_json)
    if not result_tbl then return {'CORRUPT_COMPLETION_INPUT'} end
    if result_tbl['task_id'] ~= task_id then return {'RESULT_TASK_MISMATCH'} end
    if result_tbl['status'] ~= 'completed' then return {'RESULT_STATUS_MISMATCH'} end
end

-- Triggering child status, claim, fence, lineage, and conflict checks
if redis.call('EXISTS', status_key) == 0 then return {'TASK_NOT_FOUND'} end
if key_type(status_key) ~= 'string' then return {'WRONG_KEY_TYPE_STATUS'} end
local status_tbl = decode_object(redis.call('GET', status_key))
if not status_tbl or status_tbl['task_id'] ~= task_id or type(status_tbl['status']) ~= 'string' or status_tbl['status'] == '' then
    return {'CORRUPT_STATUS'}
end
if status_tbl['status'] ~= 'working' then
    return {'NOT_WORKING', status_tbl['status']}
end

if redis.call('EXISTS', claim_key) == 0 then return {'CLAIM_NOT_FOUND'} end
if key_type(claim_key) ~= 'string' then return {'WRONG_KEY_TYPE_CLAIM'} end
local claim = decode_object(redis.call('GET', claim_key))
if not claim then return {'CORRUPT_CLAIM'} end
if claim['assignment_id'] ~= assignment_id then return {'ASSIGNMENT_MISMATCH'} end
if claim['task_id'] ~= task_id then return {'TASK_MISMATCH'} end
if claim['runtime_principal_id'] ~= principal_id then return {'PRINCIPAL_MISMATCH'} end
if claim['bound_agent_id'] ~= bound_agent_id then return {'BOUND_AGENT_MISMATCH'} end
if claim['execution_kind'] ~= 'delegated_child' then return {'EXECUTION_KIND_MISMATCH'} end
local claim_cv = claim['continuation_version']
if claim_cv ~= nil and claim_cv ~= cjson.null then return {'CONTINUATION_VERSION_MISMATCH'} end
if claim['state'] ~= 'acknowledged' then return {'CLAIM_NOT_ACKNOWLEDGED'} end
if claim['proof_digest'] ~= proof_digest then return {'PROOF_MISMATCH'} end

if redis.call('EXISTS', fence_key) == 0 then return {'FENCE_NOT_FOUND'} end
if key_type(fence_key) ~= 'string' then return {'WRONG_KEY_TYPE_FENCE'} end
local cur_fence = tonumber(redis.call('GET', fence_key))
if not cur_fence or cur_fence ~= exp_epoch or tonumber(claim['fencing_epoch']) ~= exp_epoch then
    return {'FENCE_MISMATCH', tostring(cur_fence)}
end

if redis.call('EXISTS', lineage_key) == 0 then return {'LINEAGE_NOT_FOUND'} end
if key_type(lineage_key) ~= 'string' then return {'WRONG_KEY_TYPE_LINEAGE'} end
local lineage = decode_object(redis.call('GET', lineage_key))
if not lineage or lineage['task_id'] ~= task_id or lineage['root_task_id'] ~= root_task_id then
    return {'CORRUPT_LINEAGE'}
end
if lineage['parent_task_id'] ~= parent_task_id then return {'CHILD_LINEAGE_MISMATCH'} end

if redis.call('EXISTS', outcome_key) == 1 then return {'OUTCOME_CONFLICT'} end
if redis.call('EXISTS', result_key) == 1 then return {'RESULT_CONFLICT'} end
if redis.call('EXISTS', audit_key) == 1 then return {'AUDIT_CONFLICT'} end
if redis.call('EXISTS', child_marker_key) == 1 then return {'MARKER_CONFLICT'} end

-- Parent, continuation v2, accepted wave, and resolved wave checks
if redis.call('EXISTS', resolved_wave_key) == 1 then
    if key_type(resolved_wave_key) ~= 'string' then return {'WRONG_KEY_TYPE_RESOLVED_WAVE'} end
    return {'WAVE_ALREADY_RESOLVED'}
end

if redis.call('EXISTS', parent_status_key) == 0 then return {'PARENT_STATUS_NOT_FOUND'} end
if key_type(parent_status_key) ~= 'string' then return {'WRONG_KEY_TYPE_PARENT_STATUS'} end
local parent_status = decode_object(redis.call('GET', parent_status_key))
if not parent_status or parent_status['task_id'] ~= parent_task_id then return {'CORRUPT_PARENT_STATUS'} end
if parent_status['status'] ~= 'waiting_for_children' then
    return {'PARENT_NOT_WAITING', tostring(parent_status['status'])}
end

if redis.call('EXISTS', parent_lineage_key) == 0 then return {'PARENT_LINEAGE_NOT_FOUND'} end
if key_type(parent_lineage_key) ~= 'string' then return {'WRONG_KEY_TYPE_PARENT_LINEAGE'} end
local p_lin = decode_object(redis.call('GET', parent_lineage_key))
if not p_lin or p_lin['task_id'] ~= parent_task_id or p_lin['root_task_id'] ~= root_task_id or
   p_lin['parent_task_id'] ~= cjson.null then
    return {'CORRUPT_PARENT_LINEAGE'}
end

if redis.call('EXISTS', parent_children_key) == 0 then return {'CHILD_NOT_MEMBER'} end
if key_type(parent_children_key) ~= 'set' then return {'WRONG_KEY_TYPE_CHILDREN'} end
if redis.call('SCARD', parent_children_key) ~= child_count then return {'CORRUPT_CHILDREN'} end
if redis.call('SISMEMBER', parent_children_key, task_id) == 0 then return {'CHILD_NOT_MEMBER'} end

if redis.call('EXISTS', continuation_latest_key) == 0 then return {'CONTINUATION_LATEST_MISMATCH'} end
if key_type(continuation_latest_key) ~= 'string' then return {'WRONG_KEY_TYPE_CONTINUATION_LATEST'} end
if redis.call('GET', continuation_latest_key) ~= '2' then return {'CONTINUATION_LATEST_MISMATCH'} end

if redis.call('EXISTS', continuation_key) == 0 then return {'MISSING_CONTINUATION'} end
if key_type(continuation_key) ~= 'string' then return {'WRONG_KEY_TYPE_CONTINUATION'} end
local sc = decode_object(redis.call('GET', continuation_key))
if not sc or sc['version'] ~= 2 or sc['continuation_version'] ~= 2 or
   sc['parent_task_id'] ~= parent_task_id or sc['wave_id'] ~= wave_id or
   sc['stage'] ~= 'incorporate_analysis_child' or sc['continuation_capability'] ~= 'research' or
   sc['created_at'] ~= pre_cont_sat['created_at'] or
   sc['input_version'] ~= pre_cont_sat['input_version'] then
    return {'CORRUPT_CONTINUATION'}
end
if sc['state'] ~= 'active' or sc['wave_resolution'] ~= cjson.null then
    return {'CONTINUATION_NOT_ACTIVE', tostring(sc['state'])}
end
if type(sc['required_child_task_ids']) ~= 'table' or #sc['required_child_task_ids'] ~= child_count then
    return {'CORRUPT_CONTINUATION'}
end

if redis.call('EXISTS', wave_key) == 0 then return {'MISSING_ACCEPTED_WAVE'} end
if key_type(wave_key) ~= 'string' then return {'WRONG_KEY_TYPE_WAVE'} end
local sw = decode_object(redis.call('GET', wave_key))
if not sw or sw['version'] ~= 1 or sw['wave_id'] ~= wave_id or sw['parent_task_id'] ~= parent_task_id or
   sw['state'] ~= 'unresolved' or type(sw['accepted_at']) ~= 'string' or sw['accepted_at'] == '' or
   type(sw['wave_deadline']) ~= 'string' or sw['wave_deadline'] == '' or
   type(sw['required_child_task_ids']) ~= 'table' or #sw['required_child_task_ids'] ~= child_count then
    return {'CORRUPT_WAVE_PROJECTION'}
end

local dd = sw['deadline_derivation']
if type(dd) ~= 'table' or dd['policy_id'] ~= 'deterministic_bounded_baseline_v1' or
   dd['policy_version'] ~= 1 or dd['required_child_count'] ~= child_count or
   dd['accepted_at'] ~= sw['accepted_at'] or dd['wave_deadline'] ~= sw['wave_deadline'] then
    return {'CORRUPT_WAVE_PROJECTION'}
end

local policy = sw['barrier_policy']
local frac_num = nil
local frac_den = nil
local derived_req_count = nil
if policy == 'WAIT_ALL' then
    if sw['required_success_fraction'] ~= cjson.null or sw['required_success_count'] ~= cjson.null then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
elseif policy == 'PARTIAL' then
    local rf = sw['required_success_fraction']
    if type(rf) ~= 'table' or type(rf['numerator']) ~= 'number' or type(rf['denominator']) ~= 'number' then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
    frac_num = rf['numerator']
    frac_den = rf['denominator']
    if math.floor(frac_num) ~= frac_num or math.floor(frac_den) ~= frac_den or
       frac_num < 1 or frac_den < 1 or frac_num > frac_den or frac_den > 10000 then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
    derived_req_count = math.floor((child_count * frac_num + frac_den - 1) / frac_den)
    if derived_req_count < 1 or derived_req_count > child_count or sw['required_success_count'] ~= derived_req_count then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
else
    return {'CORRUPT_WAVE_PROJECTION'}
end

if redis.call('EXISTS', hint_queue_key) == 1 and key_type(hint_queue_key) ~= 'list' then
    return {'WRONG_KEY_TYPE_HINT_QUEUE'}
end

-- Check exact sibling membership order matches wave & continuation
local slot_ids = {s1_id, s2_id}
if child_count >= 3 then slot_ids[3] = s3_id end
if child_count >= 4 then slot_ids[4] = s4_id end
local trig_found = false
for idx = 1, child_count do
    local sid = slot_ids[idx]
    if type(sid) ~= 'string' or sid == '' or
       sw['required_child_task_ids'][idx] ~= sid or
       sc['required_child_task_ids'][idx] ~= sid then
        return {'WAVE_CONSISTENCY_MISMATCH'}
    end
    if sid == task_id then trig_found = true end
end
if not trig_found then return {'CHILD_NOT_MEMBER'} end

-- Per-sibling authoritative validation
local trig_outcome_id = cand_outcome['outcome_id']
local e1, st1, oid1, fn1, hc1 = validate_sibling_readonly(
    s1_id, s1_status_key, s1_claim_key, s1_fence_key, s1_outcome_key, s1_result_key, s1_lineage_key, s1_cap_key, s1_input_key, s1_marker_key, trig_outcome_id
)
if e1 then return {e1} end
local e2, st2, oid2, fn2, hc2 = validate_sibling_readonly(
    s2_id, s2_status_key, s2_claim_key, s2_fence_key, s2_outcome_key, s2_result_key, s2_lineage_key, s2_cap_key, s2_input_key, s2_marker_key, trig_outcome_id
)
if e2 then return {e2} end
local st3, oid3, fn3, hc3 = nil, nil, 0, false
if child_count >= 3 then
    local e3
    e3, st3, oid3, fn3, hc3 = validate_sibling_readonly(
        s3_id, s3_status_key, s3_claim_key, s3_fence_key, s3_outcome_key, s3_result_key, s3_lineage_key, s3_cap_key, s3_input_key, s3_marker_key, trig_outcome_id
    )
    if e3 then return {e3} end
end
local st4, oid4, fn4, hc4 = nil, nil, 0, false
if child_count >= 4 then
    local e4
    e4, st4, oid4, fn4, hc4 = validate_sibling_readonly(
        s4_id, s4_status_key, s4_claim_key, s4_fence_key, s4_outcome_key, s4_result_key, s4_lineage_key, s4_cap_key, s4_input_key, s4_marker_key, trig_outcome_id
    )
    if e4 then return {e4} end
end

-- Pure canonical barrier evaluation over post-commit child states
local slot_statuses = {st1, st2}
local slot_outcomes = {oid1, oid2}
local slot_fences = {fn1, fn2}
local slot_has_claim = {hc1, hc2}
if child_count >= 3 then
    slot_statuses[3] = st3
    slot_outcomes[3] = oid3
    slot_fences[3] = fn3
    slot_has_claim[3] = hc3
end
if child_count >= 4 then
    slot_statuses[4] = st4
    slot_outcomes[4] = oid4
    slot_fences[4] = fn4
    slot_has_claim[4] = hc4
end

local completed_count = 0
local terminal_count = 0
local unresolved_count = 0
local potentially_successful_count = 0
for idx = 1, child_count do
    local st_val = slot_statuses[idx]
    if st_val == 'completed' then
        completed_count = completed_count + 1
        terminal_count = terminal_count + 1
    elseif st_val == 'failed' or st_val == 'claim_timed_out' then
        terminal_count = terminal_count + 1
    elseif st_val == 'pending' or st_val == 'working' or st_val == 'recovery_required' then
        unresolved_count = unresolved_count + 1
        potentially_successful_count = potentially_successful_count + 1
    else
        return {'CORRUPT_CHILD_STATE'}
    end
end

local wave_resolution = 'UNRESOLVED'
if policy == 'WAIT_ALL' then
    if terminal_count == child_count then
        wave_resolution = 'SATISFIED'
    end
else
    if completed_count >= derived_req_count then
        wave_resolution = 'SATISFIED'
    elseif completed_count + potentially_successful_count < derived_req_count then
        wave_resolution = 'UNSATISFIABLE'
    end
end

local resolve_now = (wave_resolution ~= 'UNRESOLVED')
local resolved_wave_json = ''
local released_cont_json = ''

local s1_close = false
local s1_del_claim = false
local s1_inc_fence = false
local s1_next_fence = ''
local s1_res_status_json = ''

local s2_close = false
local s2_del_claim = false
local s2_inc_fence = false
local s2_next_fence = ''
local s2_res_status_json = ''

local s3_close = false
local s3_del_claim = false
local s3_inc_fence = false
local s3_next_fence = ''
local s3_res_status_json = ''

local s4_close = false
local s4_del_claim = false
local s4_inc_fence = false
local s4_next_fence = ''
local s4_res_status_json = ''

if resolve_now then
    if wave_resolution == 'SATISFIED' then
        released_cont_json = pre_cont_sat_json
    else
        released_cont_json = pre_cont_unsat_json
    end

    local child_outcomes_tbl = {}
    local residual_ids_tbl = {}
    local close_flags = {false, false, false, false}
    local del_claim_flags = {false, false, false, false}
    local inc_fence_flags = {false, false, false, false}
    local next_fences = {'', '', '', ''}
    local res_status_jsons = {'', '', '', ''}

    for idx = 1, child_count do
        local sid = slot_ids[idx]
        local st_val = slot_statuses[idx]
        if st_val == 'completed' or st_val == 'failed' or st_val == 'claim_timed_out' then
            child_outcomes_tbl[sid] = {
                outcome_id = slot_outcomes[idx],
                status = st_val,
            }
        else
            child_outcomes_tbl[sid] = {
                outcome_id = cjson.null,
                status = 'ended / barrier_resolved',
            }
            table.insert(residual_ids_tbl, sid)
            close_flags[idx] = true
            del_claim_flags[idx] = slot_has_claim[idx]
            if slot_fences[idx] > 0 or slot_has_claim[idx] then
                inc_fence_flags[idx] = true
                next_fences[idx] = tostring(slot_fences[idx] + 1)
            end
            res_status_jsons[idx] = cjson.encode({
                status = 'ended / barrier_resolved',
                task_id = sid,
                updated_at = completed_at,
                wave_id = wave_id,
            })
        end
    end

    s1_close, s1_del_claim, s1_inc_fence, s1_next_fence, s1_res_status_json = close_flags[1], del_claim_flags[1], inc_fence_flags[1], next_fences[1], res_status_jsons[1]
    s2_close, s2_del_claim, s2_inc_fence, s2_next_fence, s2_res_status_json = close_flags[2], del_claim_flags[2], inc_fence_flags[2], next_fences[2], res_status_jsons[2]
    s3_close, s3_del_claim, s3_inc_fence, s3_next_fence, s3_res_status_json = close_flags[3], del_claim_flags[3], inc_fence_flags[3], next_fences[3], res_status_jsons[3]
    s4_close, s4_del_claim, s4_inc_fence, s4_next_fence, s4_res_status_json = close_flags[4], del_claim_flags[4], inc_fence_flags[4], next_fences[4], res_status_jsons[4]

    local req_count_val = cjson.null
    local req_frac_val = cjson.null
    if policy == 'PARTIAL' then
        req_count_val = derived_req_count
        req_frac_val = {
            denominator = frac_den,
            numerator = frac_num,
        }
    end

    local rw_residual_val = residual_ids_tbl
    if #residual_ids_tbl == 0 then
        rw_residual_val = '__TAL_EMPTY_ARRAY_SENTINEL__'
    end

    local rw_tbl = {
        accepted_at = sw['accepted_at'],
        accepted_wave_fingerprint = accepted_wave_fp,
        barrier_policy = policy,
        child_outcomes = child_outcomes_tbl,
        completed_count = completed_count,
        deadline_policy_id = dd['policy_id'],
        deadline_policy_version = dd['policy_version'],
        parent_task_id = parent_task_id,
        required_child_task_ids = sw['required_child_task_ids'],
        required_success_count = req_count_val,
        required_success_fraction = req_frac_val,
        residual_child_task_ids = rw_residual_val,
        residual_disposition = 'ended / barrier_resolved',
        resolution = wave_resolution,
        resolved_at = completed_at,
        terminal_count = terminal_count,
        triggering_assignment_id = assignment_id,
        triggering_child_task_id = task_id,
        triggering_fencing_epoch = exp_epoch,
        unresolved_count = unresolved_count,
        version = 2,
        wave_deadline = sw['wave_deadline'],
        wave_id = wave_id,
    }
    local ok_rw_enc, encoded_rw = pcall(cjson.encode, rw_tbl)
    if not ok_rw_enc or type(encoded_rw) ~= 'string' then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
    if #residual_ids_tbl == 0 then
        encoded_rw = string.gsub(encoded_rw, '"residual_child_task_ids":"__TAL_EMPTY_ARRAY_SENTINEL__"', '"residual_child_task_ids":[]', 1)
    end
    if #encoded_rw > 16384 then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
    resolved_wave_json = encoded_rw
end

-- =============================================================================
-- PHASE 2: Write Commit (Prevalidated Writes Only; Mechanically Non-Failing)
-- =============================================================================
redis.call('SET', completion_key, binding_json)
redis.call('SET', outcome_key, outcome_json)
if has_result then
    redis.call('SET', result_key, result_json)
end
redis.call('SET', audit_key, audit_json)
redis.call('SET', status_key, status_json)
redis.call('DEL', claim_key)
redis.call('SET', child_marker_key, marker_json)

if resolve_now then
    redis.call('SET', resolved_wave_key, resolved_wave_json)
    redis.call('SET', continuation_key, released_cont_json)
    redis.call('SET', parent_status_key, parent_status_json)
    redis.call('RPUSH', hint_queue_key, hint_json)

    if s1_close then
        redis.call('SET', s1_status_key, s1_res_status_json)
        if s1_del_claim then redis.call('DEL', s1_claim_key) end
        if s1_inc_fence then redis.call('SET', s1_fence_key, s1_next_fence) end
    end
    if s2_close then
        redis.call('SET', s2_status_key, s2_res_status_json)
        if s2_del_claim then redis.call('DEL', s2_claim_key) end
        if s2_inc_fence then redis.call('SET', s2_fence_key, s2_next_fence) end
    end
    if child_count >= 3 and s3_close then
        redis.call('SET', s3_status_key, s3_res_status_json)
        if s3_del_claim then redis.call('DEL', s3_claim_key) end
        if s3_inc_fence then redis.call('SET', s3_fence_key, s3_next_fence) end
    end
    if child_count >= 4 and s4_close then
        redis.call('SET', s4_status_key, s4_res_status_json)
        if s4_del_claim then redis.call('DEL', s4_claim_key) end
        if s4_inc_fence then redis.call('SET', s4_fence_key, s4_next_fence) end
    end
end

return {'OK', outcome_json, result_json}
"""

ACKNOWLEDGE_CLAIM_LUA = """
local claim_key = KEYS[1]
local assignment_id = ARGV[1]
local principal_id = ARGV[2]
local now_iso = ARGV[3]
local execution_expires_at = ARGV[4]

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
if execution_expires_at and execution_expires_at ~= '' then
    cur_claim['execution_expires_at'] = execution_expires_at
end
local updated_raw = cjson.encode(cur_claim)
redis.call('SET', claim_key, updated_raw)

return {'OK', updated_raw}
"""

EXPIRE_DELIVERY_LEASE_LUA = """
local claim_key = KEYS[1]
local status_key = KEYS[2]
local now_iso = ARGV[1]

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
local prior_status = cur_claim['prior_claimable_status'] or 'pending'

if type(c_state) ~= 'string' or (c_state ~= 'claimed_unacknowledged' and c_state ~= 'acknowledged') then
    return {'CORRUPT_CLAIM'}
end

if type(c_expires_at) ~= 'string' or c_expires_at == '' then
    return {'CORRUPT_CLAIM'}
end

if prior_status ~= 'pending' and prior_status ~= 'resumable' then
    return {'CORRUPT_CLAIM'}
end

if c_state == 'acknowledged' then
    return {'NOOP'}
end

if c_state == 'claimed_unacknowledged' then
    if c_expires_at > now_iso then
        return {'NOOP'}
    end

    if redis.call('EXISTS', status_key) == 0 then
        return {'CORRUPT_STATUS'}
    end
    local status_raw = redis.call('GET', status_key)
    local ok_s, status_tbl = pcall(cjson.decode, status_raw)
    if not ok_s or type(status_tbl) ~= 'table' then
        return {'CORRUPT_STATUS'}
    end

    -- Lease has expired: atomically delete claim and restore status to prior_claimable_status
    status_tbl['status'] = prior_status
    local restored_status_json = cjson.encode(status_tbl)

    redis.call('DEL', claim_key)
    redis.call('SET', status_key, restored_status_json)
    return {'EXPIRED', prior_status}
end

return {'CORRUPT_CLAIM'}
"""

MATERIALIZE_CLAIM_TIMEOUT_LUA = """
local child_status_key = KEYS[1]
local child_claim_key = KEYS[2]
local child_deadline_key = KEYS[3]
local child_outcome_key = KEYS[4]
local child_result_key = KEYS[5]
local child_lineage_key = KEYS[6]
local parent_status_key = KEYS[7]
local parent_children_key = KEYS[8]
local parent_continuation_key = KEYS[9]
local parent_continuation_latest_key = KEYS[10]
local child_marker_key = KEYS[11]
local hint_queue_key = KEYS[12]

local child_task_id = ARGV[1]
local parent_task_id = ARGV[2]
local now_iso = ARGV[3]
local child_outcome_json = ARGV[4]
local child_status_json = ARGV[5]
local parent_status_json = ARGV[6]
local marker_json = ARGV[7]
local hint_json = ARGV[8]

-- =============================================================================
-- PHASE 1: Reads and Validations (Strictly Read-Only, Zero Mutations)
-- =============================================================================

-- 0. Key type validations
for _, k in ipairs({child_status_key, child_claim_key, child_deadline_key, child_outcome_key, child_result_key, child_lineage_key, parent_status_key, parent_continuation_key, parent_continuation_latest_key, child_marker_key}) do
    local t = redis.call('TYPE', k)['ok']
    if t ~= 'none' and t ~= 'string' then
        return {'WRONG_KEY_TYPE_' .. k, t}
    end
end
local children_type = redis.call('TYPE', parent_children_key)['ok']
if children_type ~= 'none' and children_type ~= 'set' then
    return {'WRONG_KEY_TYPE_' .. parent_children_key, children_type}
end
local hint_type = redis.call('TYPE', hint_queue_key)['ok']
if hint_type ~= 'none' and hint_type ~= 'list' then
    return {'WRONG_KEY_TYPE_' .. hint_queue_key, hint_type}
end

-- 1. Check child status exists
if redis.call('EXISTS', child_status_key) == 0 then
    return {'CHILD_NOT_FOUND'}
end
local child_status_raw = redis.call('GET', child_status_key)
local ok_cs, child_status_tbl = pcall(cjson.decode, child_status_raw)
if not ok_cs or type(child_status_tbl) ~= 'table' then
    return {'CORRUPT_CHILD_STATUS'}
end
local child_status = child_status_tbl['status']

-- 2. Idempotent check: if child is already claim_timed_out
if child_status == 'claim_timed_out' then
    if redis.call('EXISTS', child_outcome_key) == 0 then
        return {'OUTCOME_CONFLICT'}
    end
    local existing_outcome_raw = redis.call('GET', child_outcome_key)
    local ok_out, existing_out = pcall(cjson.decode, existing_outcome_raw)
    if not ok_out or type(existing_out) ~= 'table' then
        return {'CORRUPT_TERMINAL_OUTCOME'}
    end
    if existing_out['task_id'] ~= child_task_id or existing_out['terminal_status'] ~= 'claim_timed_out' then
        return {'OUTCOME_CONFLICT'}
    end
    local expected_outcome_id = existing_out['outcome_id']

    -- No TaskResult may exist
    if redis.call('EXISTS', child_result_key) == 1 then
        return {'RESULT_CONFLICT'}
    end

    -- No active claim may exist
    if redis.call('EXISTS', child_claim_key) == 1 then
        return {'ACTIVE_CLAIM_EXISTS'}
    end

    -- Lineage and membership checks:
    if redis.call('EXISTS', child_lineage_key) == 0 then
        return {'CHILD_LINEAGE_NOT_FOUND'}
    end
    local lineage_raw = redis.call('GET', child_lineage_key)
    local ok_l, lineage_tbl = pcall(cjson.decode, lineage_raw)
    if not ok_l or type(lineage_tbl) ~= 'table' then
        return {'CORRUPT_CHILD_LINEAGE'}
    end
    if lineage_tbl['task_id'] ~= child_task_id then
        return {'LINEAGE_CHILD_MISMATCH'}
    end
    if lineage_tbl['parent_task_id'] ~= parent_task_id then
        return {'LINEAGE_PARENT_MISMATCH'}
    end
    if lineage_tbl['root_task_id'] == nil or lineage_tbl['root_task_id'] == cjson.null or lineage_tbl['root_task_id'] == '' then
        return {'CORRUPT_CHILD_LINEAGE'}
    end

    -- Parent children set must include child
    if redis.call('EXISTS', parent_children_key) == 0 or redis.call('SISMEMBER', parent_children_key, child_task_id) ~= 1 then
        return {'CHILD_NOT_MEMBER'}
    end

    -- Continuation checks:
    if redis.call('EXISTS', parent_continuation_latest_key) == 0 then
        return {'MISSING_CONTINUATION_LATEST'}
    end
    local latest_raw = redis.call('GET', parent_continuation_latest_key)
    if tonumber(latest_raw) ~= 1 then
        return {'CONTINUATION_LATEST_MISMATCH', tostring(latest_raw)}
    end

    if redis.call('EXISTS', parent_continuation_key) == 0 then
        return {'MISSING_CONTINUATION'}
    end
    local cont_raw = redis.call('GET', parent_continuation_key)
    local ok_cnt, cont_tbl = pcall(cjson.decode, cont_raw)
    if not ok_cnt or type(cont_tbl) ~= 'table' then
        return {'CORRUPT_CONTINUATION'}
    end
    if cont_tbl['parent_task_id'] ~= parent_task_id then
        return {'CONTINUATION_PARENT_MISMATCH', tostring(cont_tbl['parent_task_id'])}
    end
    if tonumber(cont_tbl['continuation_version']) ~= 1 then
        return {'CONTINUATION_VERSION_MISMATCH', tostring(cont_tbl['continuation_version'])}
    end
    if cont_tbl['stage'] ~= 'incorporate_analysis_child' then
        return {'CONTINUATION_STAGE_MISMATCH', tostring(cont_tbl['stage'])}
    end
    if cont_tbl['child_task_id'] ~= child_task_id then
        return {'CONTINUATION_CHILD_MISMATCH', tostring(cont_tbl['child_task_id'])}
    end

    -- Projection checks:
    -- Child marker must exist and bind exact child, parent, outcome, and continuation version
    if redis.call('EXISTS', child_marker_key) == 0 then
        return {'INCOMPLETE_REPLAY_PROJECTION'}
    end
    local marker_raw = redis.call('GET', child_marker_key)
    local ok_m, marker_tbl = pcall(cjson.decode, marker_raw)
    if not ok_m or type(marker_tbl) ~= 'table' then
        return {'CORRUPT_CHILD_MARKER'}
    end
    if marker_tbl['child_task_id'] ~= child_task_id then
        return {'INCOMPLETE_REPLAY_PROJECTION'}
    end
    if marker_tbl['parent_task_id'] ~= parent_task_id then
        return {'INCOMPLETE_REPLAY_PROJECTION'}
    end
    if marker_tbl['outcome_id'] ~= expected_outcome_id then
        return {'INCOMPLETE_REPLAY_PROJECTION'}
    end
    if tonumber(marker_tbl['continuation_version']) ~= 1 then
        return {'INCOMPLETE_REPLAY_PROJECTION'}
    end
    if marker_tbl['status'] ~= 'claim_timed_out' then
        return {'INCOMPLETE_REPLAY_PROJECTION'}
    end

    -- Parent status must be exactly 'resumable'
    if redis.call('EXISTS', parent_status_key) == 0 then
        return {'PARENT_NOT_FOUND'}
    end
    local parent_st_raw = redis.call('GET', parent_status_key)
    local ok_pst, parent_st_tbl = pcall(cjson.decode, parent_st_raw)
    if not ok_pst or type(parent_st_tbl) ~= 'table' then
        return {'CORRUPT_PARENT_STATUS'}
    end
    if parent_st_tbl['status'] ~= 'resumable' then
        return {'INCOMPLETE_REPLAY_PROJECTION'}
    end

    -- Valid replay: return existing outcome with IDEMPOTENT flag (no writes, no hint emitted)
    return {'IDEMPOTENT', existing_outcome_raw}
end

-- 3. Child must be in pending status
if child_status ~= 'pending' then
    return {'CHILD_NOT_PENDING', child_status}
end

-- 4. Check no terminal outcome, result, or marker already exists
if redis.call('EXISTS', child_outcome_key) == 1 then
    return {'OUTCOME_CONFLICT'}
end
if redis.call('EXISTS', child_result_key) == 1 then
    return {'RESULT_CONFLICT'}
end
if redis.call('EXISTS', child_marker_key) == 1 then
    return {'MARKER_CONFLICT'}
end

-- 5. Check active claim on child
if redis.call('EXISTS', child_claim_key) == 1 then
    local claim_raw = redis.call('GET', child_claim_key)
    local ok_c, claim_tbl = pcall(cjson.decode, claim_raw)
    if not ok_c or type(claim_tbl) ~= 'table' then
        return {'CORRUPT_CLAIM'}
    end
    if claim_tbl['task_id'] ~= child_task_id then
        return {'CORRUPT_CLAIM'}
    end
    local c_state = claim_tbl['state']
    local c_expires_at = claim_tbl['delivery_expires_at']
    if c_state == 'acknowledged' then
        return {'ACTIVE_CLAIM_EXISTS'}
    elseif c_state == 'claimed_unacknowledged' then
        if type(c_expires_at) == 'string' and c_expires_at > now_iso then
            return {'ACTIVE_CLAIM_EXISTS'}
        end
    else
        return {'CORRUPT_CLAIM'}
    end
end

-- 6. Child lineage validation
if redis.call('EXISTS', child_lineage_key) == 0 then
    return {'CHILD_LINEAGE_NOT_FOUND'}
end
local lineage_raw = redis.call('GET', child_lineage_key)
local ok_l, lineage_tbl = pcall(cjson.decode, lineage_raw)
if not ok_l or type(lineage_tbl) ~= 'table' then
    return {'CORRUPT_CHILD_LINEAGE'}
end
if lineage_tbl['task_id'] ~= child_task_id then
    return {'LINEAGE_CHILD_MISMATCH'}
end
if lineage_tbl['parent_task_id'] == nil or lineage_tbl['parent_task_id'] == cjson.null or lineage_tbl['parent_task_id'] ~= parent_task_id then
    return {'LINEAGE_PARENT_MISMATCH'}
end

-- 7. Child claim deadline validation
local deadline_val = nil
if redis.call('EXISTS', child_deadline_key) == 1 then
    deadline_val = redis.call('GET', child_deadline_key)
elseif child_status_tbl['claim_deadline'] and child_status_tbl['claim_deadline'] ~= '' then
    deadline_val = child_status_tbl['claim_deadline']
end
if not deadline_val or deadline_val == '' then
    return {'MISSING_CLAIM_DEADLINE'}
end

-- Boundary check: now_iso must be strictly > deadline_val. At deadline (now_iso <= deadline_val), child remains claimable.
if now_iso <= deadline_val then
    return {'DEADLINE_NOT_EXPIRED', deadline_val}
end

-- 8. Parent status validation
if redis.call('EXISTS', parent_status_key) == 0 then
    return {'PARENT_NOT_FOUND'}
end
local parent_status_raw = redis.call('GET', parent_status_key)
local ok_ps, parent_status_tbl = pcall(cjson.decode, parent_status_raw)
if not ok_ps or type(parent_status_tbl) ~= 'table' then
    return {'CORRUPT_PARENT_STATUS'}
end
if parent_status_tbl['task_id'] and parent_status_tbl['task_id'] ~= parent_task_id then
    return {'CORRUPT_PARENT_STATUS'}
end
if parent_status_tbl['status'] ~= 'waiting_for_children' then
    return {'PARENT_NOT_WAITING', parent_status_tbl['status']}
end

-- 9. Parent children membership validation
if redis.call('EXISTS', parent_children_key) == 0 then
    return {'PARENT_CHILDREN_NOT_FOUND'}
end
if redis.call('SISMEMBER', parent_children_key, child_task_id) ~= 1 then
    return {'CHILD_NOT_MEMBER'}
end

-- 10. Parent continuation validation
if redis.call('EXISTS', parent_continuation_latest_key) == 0 then
    return {'MISSING_CONTINUATION_LATEST'}
end
local latest_raw = redis.call('GET', parent_continuation_latest_key)
local latest_ver = tonumber(latest_raw)
if not latest_ver or latest_ver ~= 1 then
    return {'CONTINUATION_LATEST_MISMATCH', tostring(latest_ver or latest_raw)}
end

if redis.call('EXISTS', parent_continuation_key) == 0 then
    return {'MISSING_CONTINUATION'}
end
local cont_raw = redis.call('GET', parent_continuation_key)
local ok_cnt, cont_tbl = pcall(cjson.decode, cont_raw)
if not ok_cnt or type(cont_tbl) ~= 'table' then
    return {'CORRUPT_CONTINUATION'}
end
if cont_tbl['parent_task_id'] ~= parent_task_id then
    return {'CONTINUATION_PARENT_MISMATCH', tostring(cont_tbl['parent_task_id'])}
end
local cont_ver = tonumber(cont_tbl['continuation_version'])
if not cont_ver or cont_ver ~= latest_ver then
    return {'CONTINUATION_VERSION_MISMATCH', tostring(cont_tbl['continuation_version'])}
end
if cont_tbl['stage'] ~= 'incorporate_analysis_child' then
    return {'CONTINUATION_STAGE_MISMATCH', tostring(cont_tbl['stage'])}
end
if cont_tbl['child_task_id'] ~= child_task_id then
    return {'CONTINUATION_CHILD_MISMATCH', tostring(cont_tbl['child_task_id'])}
end

-- =============================================================================
-- PHASE 2: State Commit (Atomic Mutations; No Failures Permitted After First Mutation)
-- =============================================================================

-- 1. Store canonical TerminalOutcome for child
redis.call('SET', child_outcome_key, child_outcome_json)

-- 2. Transition child status to claim_timed_out
redis.call('SET', child_status_key, child_status_json)

-- 3. Write child completion marker
redis.call('SET', child_marker_key, marker_json)

-- 4. Transition parent status from waiting_for_children to resumable
redis.call('SET', parent_status_key, parent_status_json)

-- 5. Enqueue research wake-up hint
redis.call('RPUSH', hint_queue_key, hint_json)

return {'OK', child_outcome_json}
"""

TRANSITION_EXPIRED_CLAIM_LUA = """
local status_key = KEYS[1]
local claim_key = KEYS[2]
local fence_key = KEYS[3]
local expiry_key = KEYS[4]
local outcome_key = KEYS[5]
local result_key = KEYS[6]
local events_key = KEYS[7]

local task_id = ARGV[1]
local now_iso = ARGV[2]
local expiry_record_json = ARGV[3]
local recovery_status_json = ARGV[4]
local stream_event_json = ARGV[5]

-- =============================================================================
-- PHASE 1: Reads and Validations (Strictly Read-Only, Zero Mutations)
-- =============================================================================

-- 0. Key type validation
for _, k in ipairs({status_key, claim_key, fence_key, expiry_key, outcome_key, result_key}) do
    local t = redis.call('TYPE', k)['ok']
    if t ~= 'none' and t ~= 'string' then
        return {'WRONG_KEY_TYPE_' .. k, t}
    end
end
local ev_type = redis.call('TYPE', events_key)['ok']
if ev_type ~= 'none' and ev_type ~= 'stream' then
    return {'WRONG_KEY_TYPE_' .. events_key, ev_type}
end

-- 1. Status key exists
if redis.call('EXISTS', status_key) == 0 then
    return {'NOT_FOUND'}
end
local status_raw = redis.call('GET', status_key)
local ok_s, status_tbl = pcall(cjson.decode, status_raw)
if not ok_s or type(status_tbl) ~= 'table' then
    return {'CORRUPT_STATUS'}
end
local cur_status = status_tbl['status']

-- 2. Idempotent check: if already in recovery_required and expiry record exists
if cur_status == 'recovery_required' then
    if redis.call('EXISTS', expiry_key) == 1 then
        local existing_expiry = redis.call('GET', expiry_key)
        return {'IDEMPOTENT', existing_expiry}
    end
    return {'EXPIRY_RECORD_MISSING'}
end

-- 3. Task must be in working status
if cur_status ~= 'working' then
    return {'NOT_WORKING', cur_status}
end

-- 4. Check no terminal outcome or result has already won
if redis.call('EXISTS', outcome_key) == 1 then
    return {'ALREADY_COMPLETED'}
end
if redis.call('EXISTS', result_key) == 1 then
    return {'ALREADY_COMPLETED'}
end

-- 5. Active claim must exist
if redis.call('EXISTS', claim_key) == 0 then
    return {'CLAIM_NOT_FOUND'}
end
local claim_raw = redis.call('GET', claim_key)
local ok_c, claim_tbl = pcall(cjson.decode, claim_raw)
if not ok_c or type(claim_tbl) ~= 'table' then
    return {'CORRUPT_CLAIM'}
end

-- 6. Claim must be in acknowledged state
if claim_tbl['state'] ~= 'acknowledged' then
    return {'CLAIM_NOT_ACKNOWLEDGED', tostring(claim_tbl['state'])}
end

-- 7. execution_expires_at must exist and be expired
local exec_expires = claim_tbl['execution_expires_at']
if not exec_expires or type(exec_expires) ~= 'string' or exec_expires == '' then
    return {'MISSING_EXECUTION_EXPIRES_AT'}
end
if now_iso <= exec_expires then
    return {'EXECUTION_NOT_EXPIRED', exec_expires}
end

-- 8. Fencing epoch must match current fence
if redis.call('EXISTS', fence_key) == 0 then
    return {'FENCE_NOT_FOUND'}
end
local fence_raw = redis.call('GET', fence_key)
local cur_fence = tonumber(fence_raw)
local claim_epoch = tonumber(claim_tbl['fencing_epoch'])
if not cur_fence or not claim_epoch or cur_fence ~= claim_epoch then
    return {'FENCE_MISMATCH', tostring(cur_fence), tostring(claim_epoch)}
end

-- 9. Validate prior_claimable_status matches execution_kind
local ekind = claim_tbl['execution_kind']
local prior_s = claim_tbl['prior_claimable_status']
if ekind == 'resumed_continuation' then
    if prior_s ~= 'resumable' then
        return {'CORRUPT_PRIOR_STATUS', tostring(prior_s)}
    end
else
    if prior_s ~= 'pending' then
        return {'CORRUPT_PRIOR_STATUS', tostring(prior_s)}
    end
end

-- =============================================================================
-- PHASE 2: State Commit (Atomic Mutations; No Failures Permitted After First Mutation)
-- =============================================================================

-- 1. Invalidate/delete the ActiveClaim
redis.call('DEL', claim_key)

-- 2. Store the ClaimExpiryRecord
redis.call('SET', expiry_key, expiry_record_json)

-- 3. Transition task status to recovery_required
redis.call('SET', status_key, recovery_status_json)

-- NOTE: As required by approved decision 5: DO NOT increment task.fence!

-- 4. Publish bounded non-secret observation to event stream if event provided
if stream_event_json and stream_event_json ~= '' then
    local ok_ev, ev_tbl = pcall(cjson.decode, stream_event_json)
    if ok_ev and type(ev_tbl) == 'table' then
        local args = {'XADD', events_key, 'MAXLEN', '~', '500', '*'}
        for k, v in pairs(ev_tbl) do
            table.insert(args, tostring(k))
            table.insert(args, tostring(v))
        end
        redis.call(unpack(args))
    end
end

return {'OK', expiry_record_json}
"""

RESOLVE_EXPIRED_CLAIM_LUA = """
local status_key = KEYS[1]
local claim_key = KEYS[2]
local fence_key = KEYS[3]
local expiry_key = KEYS[4]
local decision_key = KEYS[5]
local outcome_key = KEYS[6]

local task_id = ARGV[1]
local recovery_decision_id = ARGV[2]
local expired_claim_id = ARGV[3]
local expired_assignment_id = ARGV[4]
local expired_epoch = tonumber(ARGV[5])
local disposition = ARGV[6]
local decision_record_json = ARGV[7]
local decision_fingerprint = ARGV[8]
local restored_status_json = ARGV[9]

-- =============================================================================
-- PHASE 1: Reads and Validations (Strictly Read-Only, Zero Mutations)
-- =============================================================================

-- 0. Key type validation
for _, k in ipairs({status_key, claim_key, fence_key, expiry_key, decision_key, outcome_key}) do
    local t = redis.call('TYPE', k)['ok']
    if t ~= 'none' and t ~= 'string' then
        return {'WRONG_KEY_TYPE_' .. k, t}
    end
end

-- 1. Check idempotent replay of recovery decision
if redis.call('EXISTS', decision_key) == 1 then
    local existing_dec_raw = redis.call('GET', decision_key)
    local ok_d, existing_dec = pcall(cjson.decode, existing_dec_raw)
    if ok_d and type(existing_dec) == 'table' then
        if existing_dec['semantic_fingerprint'] == decision_fingerprint then
            return {'IDEMPOTENT', existing_dec_raw, existing_dec['restored_status']}
        else
            return {'DECISION_CONFLICT'}
        end
    end
    return {'CORRUPT_DECISION'}
end

-- 2. Status key exists
if redis.call('EXISTS', status_key) == 0 then
    return {'NOT_FOUND'}
end
local status_raw = redis.call('GET', status_key)
local ok_s, status_tbl = pcall(cjson.decode, status_raw)
if not ok_s or type(status_tbl) ~= 'table' then
    return {'CORRUPT_STATUS'}
end
local cur_status = status_tbl['status']

-- 3. Task must be in recovery_required status
if cur_status ~= 'recovery_required' then
    return {'NOT_RECOVERY_REQUIRED', cur_status}
end

-- 4. Check no terminal outcome has won
if redis.call('EXISTS', outcome_key) == 1 then
    return {'ALREADY_COMPLETED'}
end

-- 5. No active claim must exist
if redis.call('EXISTS', claim_key) == 1 then
    return {'ACTIVE_CLAIM_EXISTS'}
end

-- 6. ClaimExpiryRecord must exist and match command
if redis.call('EXISTS', expiry_key) == 0 then
    return {'EXPIRY_RECORD_NOT_FOUND'}
end
local expiry_raw = redis.call('GET', expiry_key)
local ok_exp, expiry_tbl = pcall(cjson.decode, expiry_raw)
if not ok_exp or type(expiry_tbl) ~= 'table' then
    return {'CORRUPT_EXPIRY_RECORD'}
end

if expiry_tbl['task_id'] ~= task_id or
   expiry_tbl['claim_id'] ~= expired_claim_id or
   expiry_tbl['assignment_id'] ~= expired_assignment_id or
   tonumber(expiry_tbl['expired_fencing_epoch']) ~= expired_epoch then
    return {'EXPIRY_RECORD_MISMATCH'}
end

-- 7. Disposition must be restore_claimable
if disposition ~= 'restore_claimable' then
    return {'UNSUPPORTED_DISPOSITION', disposition}
end

-- 8. Fencing epoch must still match expired_fencing_epoch
if redis.call('EXISTS', fence_key) == 0 then
    return {'FENCE_NOT_FOUND'}
end
local fence_raw = redis.call('GET', fence_key)
local cur_fence = tonumber(fence_raw)
if not cur_fence or cur_fence ~= expired_epoch then
    return {'FENCE_MISMATCH', tostring(cur_fence), tostring(expired_epoch)}
end

-- 9. Verify prior_claimable_status in expiry record is valid
local restored_status = expiry_tbl['prior_claimable_status']
if restored_status ~= 'pending' and restored_status ~= 'resumable' then
    return {'CORRUPT_PRIOR_STATUS', tostring(restored_status)}
end

-- =============================================================================
-- PHASE 2: State Commit (Atomic Mutations; No Failures Permitted After First Mutation)
-- =============================================================================

-- 1. Store the RecoveryDecisionRecord
redis.call('SET', decision_key, decision_record_json)

-- 2. Restore task status to prior_claimable_status
redis.call('SET', status_key, restored_status_json)

-- NOTE: As required by approved decision 5: DO NOT increment task.fence!

return {'OK', decision_record_json, restored_status}
"""

EVALUATE_WAVE_TIMEOUT_LUA = """
local parent_status_key = KEYS[1]
local parent_children_key = KEYS[2]
local parent_lineage_key = KEYS[3]
local continuation_key = KEYS[4]
local continuation_latest_key = KEYS[5]
local wave_key = KEYS[6]
local resolved_wave_key = KEYS[7]
local hint_queue_key = KEYS[8]

local s1_status_key = KEYS[9]
local s1_claim_key = KEYS[10]
local s1_fence_key = KEYS[11]
local s1_outcome_key = KEYS[12]
local s1_result_key = KEYS[13]
local s1_lineage_key = KEYS[14]
local s1_cap_key = KEYS[15]
local s1_input_key = KEYS[16]
local s1_marker_key = KEYS[17]
local s1_expiry_key = KEYS[18]

local s2_status_key = KEYS[19]
local s2_claim_key = KEYS[20]
local s2_fence_key = KEYS[21]
local s2_outcome_key = KEYS[22]
local s2_result_key = KEYS[23]
local s2_lineage_key = KEYS[24]
local s2_cap_key = KEYS[25]
local s2_input_key = KEYS[26]
local s2_marker_key = KEYS[27]
local s2_expiry_key = KEYS[28]

local s3_status_key = KEYS[29]
local s3_claim_key = KEYS[30]
local s3_fence_key = KEYS[31]
local s3_outcome_key = KEYS[32]
local s3_result_key = KEYS[33]
local s3_lineage_key = KEYS[34]
local s3_cap_key = KEYS[35]
local s3_input_key = KEYS[36]
local s3_marker_key = KEYS[37]
local s3_expiry_key = KEYS[38]

local s4_status_key = KEYS[39]
local s4_claim_key = KEYS[40]
local s4_fence_key = KEYS[41]
local s4_outcome_key = KEYS[42]
local s4_result_key = KEYS[43]
local s4_lineage_key = KEYS[44]
local s4_cap_key = KEYS[45]
local s4_input_key = KEYS[46]
local s4_marker_key = KEYS[47]
local s4_expiry_key = KEYS[48]

local parent_task_id = ARGV[1]
local wave_id = ARGV[2]
local now_iso = ARGV[3]
local child_count = tonumber(ARGV[4])
local accepted_wave_fp = ARGV[5]
local pre_cont_to_json = ARGV[6]
local parent_status_json = ARGV[7]
local hint_json = ARGV[8]
local s1_id = ARGV[9]
local s2_id = ARGV[10]
local s3_id = ARGV[11]
local s4_id = ARGV[12]

local function key_type(key)
    local t = redis.call('TYPE', key)
    if type(t) == 'table' then return t['ok'] end
    return tostring(t)
end

local function decode_object(raw)
    local ok, tbl = pcall(cjson.decode, raw)
    if not ok or type(tbl) ~= 'table' then return nil end
    return tbl
end

local function check_str_key(str_k)
    local kt = key_type(str_k)
    if kt ~= 'none' and kt ~= 'string' then
        return 'WRONG_KEY_TYPE_' .. str_k
    end
    return nil
end

local function validate_sibling_state(
    sid,
    st_k, cl_k, fn_k, out_k, res_k, lin_k, cap_k, inp_k, mrk_k, exp_k,
    resolved_mode, rw_outcomes
)
    if redis.call('SISMEMBER', parent_children_key, sid) ~= 1 then
        return 'CHILD_NOT_MEMBER', nil, nil, 0, false
    end
    local kt_err = check_str_key(lin_k) or check_str_key(cap_k) or check_str_key(inp_k) or
                   check_str_key(fn_k) or check_str_key(st_k) or check_str_key(cl_k) or
                   check_str_key(out_k) or check_str_key(res_k) or check_str_key(mrk_k) or check_str_key(exp_k)
    if kt_err then
        return kt_err, nil, nil, 0, false
    end
    if redis.call('EXISTS', lin_k) == 0 then return 'LINEAGE_NOT_FOUND', nil, nil, 0, false end
    local lin = decode_object(redis.call('GET', lin_k))
    if not lin or lin['task_id'] ~= sid or lin['parent_task_id'] ~= parent_task_id or lin['root_task_id'] ~= parent_task_id then
        return 'CHILD_LINEAGE_MISMATCH', nil, nil, 0, false
    end
    if redis.call('EXISTS', cap_k) == 0 then return 'MISSING_REQUIRED_CAPABILITY', nil, nil, 0, false end
    local cap = decode_object(redis.call('GET', cap_k))
    if not cap or cap['task_id'] ~= sid or cap['required_capability'] ~= 'analysis' then
        return 'CORRUPT_CHILD_CAPABILITY', nil, nil, 0, false
    end
    if redis.call('EXISTS', inp_k) == 0 then return 'MISSING_CHILD_INPUT', nil, nil, 0, false end
    local inp = decode_object(redis.call('GET', inp_k))
    if not inp or inp['task_id'] ~= sid or inp['input_version'] ~= 1 or
       type(inp['prompt']) ~= 'string' or string.match(inp['prompt'], '^%s*$') or
       type(inp['artifact_references']) ~= 'table' or #inp['artifact_references'] < 1 then
        return 'CORRUPT_CHILD_INPUT', nil, nil, 0, false
    end
    if redis.call('EXISTS', fn_k) == 0 then return 'FENCE_NOT_FOUND', nil, nil, 0, false end
    local s_fence = tonumber(redis.call('GET', fn_k))
    if not s_fence or s_fence < 0 or math.floor(s_fence) ~= s_fence then
        return 'CORRUPT_FENCE', nil, nil, 0, false
    end
    if redis.call('EXISTS', st_k) == 0 then return 'TASK_NOT_FOUND', nil, nil, 0, false end
    local st = decode_object(redis.call('GET', st_k))
    if not st or st['task_id'] ~= sid or type(st['status']) ~= 'string' or st['status'] == '' then
        return 'CORRUPT_STATUS', nil, nil, 0, false
    end
    local cur_st = st['status']
    local claim_exists = (redis.call('EXISTS', cl_k) == 1)
    local outcome_exists = (redis.call('EXISTS', out_k) == 1)
    local result_exists = (redis.call('EXISTS', res_k) == 1)
    local marker_exists = (redis.call('EXISTS', mrk_k) == 1)
    local expiry_exists = (redis.call('EXISTS', exp_k) == 1)

    if cur_st == 'completed' then
        if claim_exists or s_fence < 1 or not outcome_exists or not result_exists or not marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        local out_tbl = decode_object(redis.call('GET', out_k))
        local res_tbl = decode_object(redis.call('GET', res_k))
        local mrk_tbl = decode_object(redis.call('GET', mrk_k))
        if not out_tbl or out_tbl['task_id'] ~= sid or out_tbl['terminal_status'] ~= 'completed' or
           type(out_tbl['outcome_id']) ~= 'string' or out_tbl['outcome_id'] == '' or
           not res_tbl or res_tbl['task_id'] ~= sid or res_tbl['status'] ~= 'completed' or
           not mrk_tbl or mrk_tbl['child_task_id'] ~= sid or mrk_tbl['parent_task_id'] ~= parent_task_id or
           mrk_tbl['outcome_id'] ~= out_tbl['outcome_id'] or mrk_tbl['continuation_version'] ~= 2 then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        if resolved_mode then
            local snap = rw_outcomes[sid]
            if type(snap) ~= 'table' or snap['status'] ~= 'completed' or snap['outcome_id'] ~= out_tbl['outcome_id'] then
                return 'CORRUPT_WAVE_PROJECTION', nil, nil, 0, false
            end
        end
        return nil, 'completed', out_tbl['outcome_id'], s_fence, false
    elseif cur_st == 'failed' or cur_st == 'claim_timed_out' then
        if claim_exists or not outcome_exists or result_exists or not marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        local out_tbl = decode_object(redis.call('GET', out_k))
        local mrk_tbl = decode_object(redis.call('GET', mrk_k))
        if not out_tbl or out_tbl['task_id'] ~= sid or out_tbl['terminal_status'] ~= cur_st or
           type(out_tbl['outcome_id']) ~= 'string' or out_tbl['outcome_id'] == '' or
           not mrk_tbl or mrk_tbl['child_task_id'] ~= sid or mrk_tbl['parent_task_id'] ~= parent_task_id or
           mrk_tbl['outcome_id'] ~= out_tbl['outcome_id'] or mrk_tbl['continuation_version'] ~= 2 then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        if resolved_mode then
            local snap = rw_outcomes[sid]
            if type(snap) ~= 'table' or snap['status'] ~= cur_st or snap['outcome_id'] ~= out_tbl['outcome_id'] then
                return 'CORRUPT_WAVE_PROJECTION', nil, nil, 0, false
            end
        end
        return nil, cur_st, out_tbl['outcome_id'], s_fence, false
    elseif cur_st == 'pending' then
        if resolved_mode or claim_exists or outcome_exists or result_exists or marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, 'pending', nil, s_fence, false
    elseif cur_st == 'working' then
        if resolved_mode or not claim_exists or outcome_exists or result_exists or marker_exists or s_fence < 1 then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        local cl_tbl = decode_object(redis.call('GET', cl_k))
        if not cl_tbl or cl_tbl['task_id'] ~= sid or cl_tbl['execution_kind'] ~= 'delegated_child' or
           cl_tbl['fencing_epoch'] ~= s_fence or cl_tbl['assigned_capability'] ~= 'analysis' then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, 'working', nil, s_fence, true
    elseif cur_st == 'recovery_required' then
        if resolved_mode or claim_exists or outcome_exists or result_exists or marker_exists or s_fence < 1 or not expiry_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        local ce_tbl = decode_object(redis.call('GET', exp_k))
        if not ce_tbl or ce_tbl['task_id'] ~= sid or ce_tbl['expired_fencing_epoch'] ~= s_fence or
           ce_tbl['prior_claimable_status'] ~= 'pending' or ce_tbl['execution_kind'] ~= 'delegated_child' then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, 'recovery_required', nil, s_fence, false
    elseif cur_st == 'ended / barrier_resolved' then
        if not resolved_mode or claim_exists or outcome_exists or result_exists or marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        local snap = rw_outcomes[sid]
        if type(snap) ~= 'table' or snap['status'] ~= 'ended / barrier_resolved' or snap['outcome_id'] ~= cjson.null then
            return 'CORRUPT_WAVE_PROJECTION', nil, nil, 0, false
        end
        return nil, 'ended / barrier_resolved', nil, s_fence, false
    end
    return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
end

-- =============================================================================
-- PHASE 1: Strictly Read-Only Validations & Exact Replay Check
-- =============================================================================

if not child_count or math.floor(child_count) ~= child_count or child_count < 2 or child_count > 4 then
    return {'CORRUPT_INPUT'}
end
local pre_cont_to = decode_object(pre_cont_to_json)
local pre_ps = decode_object(parent_status_json)
local pre_hint = decode_object(hint_json)
if not pre_cont_to or not pre_ps or not pre_hint or
   pre_cont_to['version'] ~= 2 or pre_cont_to['continuation_version'] ~= 2 or
   pre_cont_to['parent_task_id'] ~= parent_task_id or pre_cont_to['wave_id'] ~= wave_id or
   pre_cont_to['state'] ~= 'released' or pre_cont_to['wave_resolution'] ~= 'TIMED_OUT' or
   pre_ps['task_id'] ~= parent_task_id or pre_ps['status'] ~= 'resumable' or
   pre_hint['task_id'] ~= parent_task_id or pre_hint['capability'] ~= 'research' or
   pre_hint['execution_kind'] ~= 'resumed_continuation' or pre_hint['expected_continuation_version'] ~= 2 then
    return {'CORRUPT_INPUT'}
end

for _, k in ipairs({parent_status_key, parent_lineage_key, continuation_key, continuation_latest_key, wave_key, resolved_wave_key}) do
    local kt = key_type(k)
    if kt ~= 'none' and kt ~= 'string' then
        return {'WRONG_KEY_TYPE_' .. k, kt}
    end
end
local children_t = key_type(parent_children_key)
if children_t ~= 'none' and children_t ~= 'set' then
    return {'WRONG_KEY_TYPE_' .. parent_children_key, children_t}
end
local hint_t = key_type(hint_queue_key)
if hint_t ~= 'none' and hint_t ~= 'list' then
    return {'WRONG_KEY_TYPE_' .. hint_queue_key, hint_t}
end

if redis.call('EXISTS', parent_lineage_key) == 0 then return {'PARENT_NOT_FOUND'} end
local p_lin = decode_object(redis.call('GET', parent_lineage_key))
if not p_lin or p_lin['task_id'] ~= parent_task_id or p_lin['root_task_id'] ~= parent_task_id or
   p_lin['parent_task_id'] ~= cjson.null then
    return {'CORRUPT_PARENT_LINEAGE'}
end

if redis.call('EXISTS', parent_status_key) == 0 then return {'PARENT_NOT_FOUND'} end
local parent_status = decode_object(redis.call('GET', parent_status_key))
if not parent_status or parent_status['task_id'] ~= parent_task_id or type(parent_status['status']) ~= 'string' then
    return {'CORRUPT_PARENT_STATUS'}
end

if redis.call('EXISTS', continuation_latest_key) == 0 then return {'CONTINUATION_LATEST_MISMATCH'} end
if redis.call('GET', continuation_latest_key) ~= '2' then return {'CONTINUATION_LATEST_MISMATCH'} end

if redis.call('EXISTS', continuation_key) == 0 then return {'MISSING_CONTINUATION'} end
local sc = decode_object(redis.call('GET', continuation_key))
if not sc or sc['version'] ~= 2 or sc['continuation_version'] ~= 2 or
   sc['parent_task_id'] ~= parent_task_id or sc['wave_id'] ~= wave_id or
   sc['stage'] ~= 'incorporate_analysis_child' or sc['continuation_capability'] ~= 'research' or
   sc['created_at'] ~= pre_cont_to['created_at'] or
   sc['input_version'] ~= pre_cont_to['input_version'] or
   type(sc['required_child_task_ids']) ~= 'table' or #sc['required_child_task_ids'] ~= child_count then
    return {'CORRUPT_CONTINUATION'}
end

if redis.call('EXISTS', wave_key) == 0 then return {'MISSING_ACCEPTED_WAVE'} end
local sw = decode_object(redis.call('GET', wave_key))
if not sw or sw['version'] ~= 1 or sw['wave_id'] ~= wave_id or sw['parent_task_id'] ~= parent_task_id or
   sw['state'] ~= 'unresolved' or type(sw['accepted_at']) ~= 'string' or sw['accepted_at'] == '' or
   type(sw['wave_deadline']) ~= 'string' or sw['wave_deadline'] == '' or
   type(sw['required_child_task_ids']) ~= 'table' or #sw['required_child_task_ids'] ~= child_count then
    return {'CORRUPT_WAVE_PROJECTION'}
end

local dd = sw['deadline_derivation']
if type(dd) ~= 'table' or dd['policy_id'] ~= 'deterministic_bounded_baseline_v1' or
   dd['policy_version'] ~= 1 or dd['required_child_count'] ~= child_count or
   dd['accepted_at'] ~= sw['accepted_at'] or dd['wave_deadline'] ~= sw['wave_deadline'] then
    return {'CORRUPT_WAVE_PROJECTION'}
end

local policy = sw['barrier_policy']
local frac_num = nil
local frac_den = nil
local derived_req_count = nil
if policy == 'WAIT_ALL' then
    if sw['required_success_fraction'] ~= cjson.null or sw['required_success_count'] ~= cjson.null then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
elseif policy == 'PARTIAL' then
    local rf = sw['required_success_fraction']
    if type(rf) ~= 'table' or type(rf['numerator']) ~= 'number' or type(rf['denominator']) ~= 'number' then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
    frac_num = rf['numerator']
    frac_den = rf['denominator']
    if math.floor(frac_num) ~= frac_num or math.floor(frac_den) ~= frac_den or
       frac_num < 1 or frac_den < 1 or frac_num > frac_den or frac_den > 10000 then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
    derived_req_count = math.floor((child_count * frac_num + frac_den - 1) / frac_den)
    if derived_req_count < 1 or derived_req_count > child_count or sw['required_success_count'] ~= derived_req_count then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
else
    return {'CORRUPT_WAVE_PROJECTION'}
end

if redis.call('EXISTS', parent_children_key) == 0 or redis.call('SCARD', parent_children_key) ~= child_count then
    return {'CORRUPT_CHILDREN'}
end

local slot_ids = {s1_id, s2_id}
if child_count >= 3 then slot_ids[3] = s3_id end
if child_count >= 4 then slot_ids[4] = s4_id end
for idx = 1, child_count do
    local sid = slot_ids[idx]
    if type(sid) ~= 'string' or sid == '' or
       sw['required_child_task_ids'][idx] ~= sid or
       sc['required_child_task_ids'][idx] ~= sid then
        return {'WAVE_CONSISTENCY_MISMATCH'}
    end
end

local has_resolved = (redis.call('EXISTS', resolved_wave_key) == 1)
if has_resolved then
    local rw_raw = redis.call('GET', resolved_wave_key)
    local rw_tbl = decode_object(rw_raw)
    if not rw_tbl or rw_tbl['version'] ~= 2 or rw_tbl['wave_id'] ~= wave_id or
       rw_tbl['parent_task_id'] ~= parent_task_id or rw_tbl['barrier_policy'] ~= policy or
       rw_tbl['accepted_at'] ~= sw['accepted_at'] or rw_tbl['wave_deadline'] ~= sw['wave_deadline'] or
       rw_tbl['accepted_wave_fingerprint'] ~= accepted_wave_fp or
       type(rw_tbl['child_outcomes']) ~= 'table' or type(rw_tbl['required_child_task_ids']) ~= 'table' or
       #rw_tbl['required_child_task_ids'] ~= child_count then
        return {'CORRUPT_RESOLVED_WAVE'}
    end
    local existing_res = rw_tbl['resolution']
    if existing_res ~= 'TIMED_OUT' then
        return {'WAVE_ALREADY_RESOLVED', tostring(existing_res)}
    end
    if sc['state'] ~= 'released' or sc['wave_resolution'] ~= 'TIMED_OUT' then
        return {'CORRUPT_CONTINUATION'}
    end
    local p_st = parent_status['status']
    if p_st ~= 'resumable' and p_st ~= 'working' and p_st ~= 'completed' and p_st ~= 'failed' and p_st ~= 'recovery_required' then
        return {'CORRUPT_PARENT_STATUS'}
    end
    local rw_out = rw_tbl['child_outcomes']
    local e1 = validate_sibling_state(s1_id, s1_status_key, s1_claim_key, s1_fence_key, s1_outcome_key, s1_result_key, s1_lineage_key, s1_cap_key, s1_input_key, s1_marker_key, s1_expiry_key, true, rw_out)
    if e1 then return {e1} end
    local e2 = validate_sibling_state(s2_id, s2_status_key, s2_claim_key, s2_fence_key, s2_outcome_key, s2_result_key, s2_lineage_key, s2_cap_key, s2_input_key, s2_marker_key, s2_expiry_key, true, rw_out)
    if e2 then return {e2} end
    if child_count >= 3 then
        local e3 = validate_sibling_state(s3_id, s3_status_key, s3_claim_key, s3_fence_key, s3_outcome_key, s3_result_key, s3_lineage_key, s3_cap_key, s3_input_key, s3_marker_key, s3_expiry_key, true, rw_out)
        if e3 then return {e3} end
    end
    if child_count >= 4 then
        local e4 = validate_sibling_state(s4_id, s4_status_key, s4_claim_key, s4_fence_key, s4_outcome_key, s4_result_key, s4_lineage_key, s4_cap_key, s4_input_key, s4_marker_key, s4_expiry_key, true, rw_out)
        if e4 then return {e4} end
    end
    return {'IDEMPOTENT', rw_raw}
end

if parent_status['status'] ~= 'waiting_for_children' then
    return {'PARENT_NOT_WAITING', tostring(parent_status['status'])}
end
if sc['state'] ~= 'active' or sc['wave_resolution'] ~= cjson.null then
    return {'CONTINUATION_NOT_ACTIVE', tostring(sc['state'])}
end

-- ADR-0011: TIMED_OUT requires now strictly at or beyond the frozen authoritative wave_deadline
if now_iso < sw['wave_deadline'] then
    return {'WAVE_DEADLINE_NOT_REACHED', sw['wave_deadline']}
end

local e1, st1, oid1, fn1, hc1 = validate_sibling_state(s1_id, s1_status_key, s1_claim_key, s1_fence_key, s1_outcome_key, s1_result_key, s1_lineage_key, s1_cap_key, s1_input_key, s1_marker_key, s1_expiry_key, false, nil)
if e1 then return {e1} end
local e2, st2, oid2, fn2, hc2 = validate_sibling_state(s2_id, s2_status_key, s2_claim_key, s2_fence_key, s2_outcome_key, s2_result_key, s2_lineage_key, s2_cap_key, s2_input_key, s2_marker_key, s2_expiry_key, false, nil)
if e2 then return {e2} end
local st3, oid3, fn3, hc3 = nil, nil, 0, false
if child_count >= 3 then
    local e3
    e3, st3, oid3, fn3, hc3 = validate_sibling_state(s3_id, s3_status_key, s3_claim_key, s3_fence_key, s3_outcome_key, s3_result_key, s3_lineage_key, s3_cap_key, s3_input_key, s3_marker_key, s3_expiry_key, false, nil)
    if e3 then return {e3} end
end
local st4, oid4, fn4, hc4 = nil, nil, 0, false
if child_count >= 4 then
    local e4
    e4, st4, oid4, fn4, hc4 = validate_sibling_state(s4_id, s4_status_key, s4_claim_key, s4_fence_key, s4_outcome_key, s4_result_key, s4_lineage_key, s4_cap_key, s4_input_key, s4_marker_key, s4_expiry_key, false, nil)
    if e4 then return {e4} end
end

local slot_statuses = {st1, st2}
local slot_outcomes = {oid1, oid2}
local slot_fences = {fn1, fn2}
local slot_has_claim = {hc1, hc2}
if child_count >= 3 then
    slot_statuses[3] = st3; slot_outcomes[3] = oid3; slot_fences[3] = fn3; slot_has_claim[3] = hc3
end
if child_count >= 4 then
    slot_statuses[4] = st4; slot_outcomes[4] = oid4; slot_fences[4] = fn4; slot_has_claim[4] = hc4
end

local completed_count = 0
local terminal_count = 0
local unresolved_count = 0
local potentially_successful_count = 0
for idx = 1, child_count do
    local st_val = slot_statuses[idx]
    if st_val == 'completed' then
        completed_count = completed_count + 1
        terminal_count = terminal_count + 1
    elseif st_val == 'failed' or st_val == 'claim_timed_out' then
        terminal_count = terminal_count + 1
    elseif st_val == 'pending' or st_val == 'working' or st_val == 'recovery_required' then
        unresolved_count = unresolved_count + 1
        potentially_successful_count = potentially_successful_count + 1
    else
        return {'CORRUPT_CHILD_STATE'}
    end
end

if unresolved_count < 1 then
    return {'CORRUPT_WAVE_PROJECTION'}
end
if policy == 'WAIT_ALL' then
    if terminal_count >= child_count then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
else
    if completed_count >= derived_req_count or completed_count + potentially_successful_count < derived_req_count then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
end

local child_outcomes_tbl = {}
local residual_ids_tbl = {}
local close_flags = {false, false, false, false}
local del_claim_flags = {false, false, false, false}
local inc_fence_flags = {false, false, false, false}
local next_fences = {'', '', '', ''}
local res_status_jsons = {'', '', '', ''}

for idx = 1, child_count do
    local sid = slot_ids[idx]
    local st_val = slot_statuses[idx]
    if st_val == 'completed' or st_val == 'failed' or st_val == 'claim_timed_out' then
        child_outcomes_tbl[sid] = {
            outcome_id = slot_outcomes[idx],
            status = st_val,
        }
    else
        child_outcomes_tbl[sid] = {
            outcome_id = cjson.null,
            status = 'ended / barrier_resolved',
        }
        table.insert(residual_ids_tbl, sid)
        close_flags[idx] = true
        del_claim_flags[idx] = slot_has_claim[idx]
        if slot_fences[idx] > 0 or slot_has_claim[idx] then
            inc_fence_flags[idx] = true
            next_fences[idx] = tostring(slot_fences[idx] + 1)
        end
        res_status_jsons[idx] = cjson.encode({
            status = 'ended / barrier_resolved',
            task_id = sid,
            updated_at = now_iso,
            wave_id = wave_id,
        })
    end
end

local s1_close, s1_del_claim, s1_inc_fence, s1_next_fence, s1_res_status_json = close_flags[1], del_claim_flags[1], inc_fence_flags[1], next_fences[1], res_status_jsons[1]
local s2_close, s2_del_claim, s2_inc_fence, s2_next_fence, s2_res_status_json = close_flags[2], del_claim_flags[2], inc_fence_flags[2], next_fences[2], res_status_jsons[2]
local s3_close, s3_del_claim, s3_inc_fence, s3_next_fence, s3_res_status_json = close_flags[3], del_claim_flags[3], inc_fence_flags[3], next_fences[3], res_status_jsons[3]
local s4_close, s4_del_claim, s4_inc_fence, s4_next_fence, s4_res_status_json = close_flags[4], del_claim_flags[4], inc_fence_flags[4], next_fences[4], res_status_jsons[4]

local req_count_val = cjson.null
local req_frac_val = cjson.null
if policy == 'PARTIAL' then
    req_count_val = derived_req_count
    req_frac_val = {
        denominator = frac_den,
        numerator = frac_num,
    }
end

local rw_tbl = {
    accepted_at = sw['accepted_at'],
    accepted_wave_fingerprint = accepted_wave_fp,
    barrier_policy = policy,
    child_outcomes = child_outcomes_tbl,
    completed_count = completed_count,
    deadline_policy_id = dd['policy_id'],
    deadline_policy_version = dd['policy_version'],
    parent_task_id = parent_task_id,
    required_child_task_ids = sw['required_child_task_ids'],
    required_success_count = req_count_val,
    required_success_fraction = req_frac_val,
    residual_child_task_ids = residual_ids_tbl,
    residual_disposition = 'ended / barrier_resolved',
    resolution = 'TIMED_OUT',
    resolved_at = now_iso,
    terminal_count = terminal_count,
    triggering_assignment_id = cjson.null,
    triggering_child_task_id = cjson.null,
    triggering_fencing_epoch = cjson.null,
    unresolved_count = unresolved_count,
    version = 2,
    wave_deadline = sw['wave_deadline'],
    wave_id = wave_id,
}
local ok_rw_enc, resolved_wave_json = pcall(cjson.encode, rw_tbl)
if not ok_rw_enc or type(resolved_wave_json) ~= 'string' or #resolved_wave_json > 16384 then
    return {'CORRUPT_WAVE_PROJECTION'}
end

-- =============================================================================
-- PHASE 2: Prevalidated Atomic Writes Only
-- =============================================================================
redis.call('SET', resolved_wave_key, resolved_wave_json)
redis.call('SET', continuation_key, pre_cont_to_json)
redis.call('SET', parent_status_key, parent_status_json)
redis.call('RPUSH', hint_queue_key, hint_json)

if s1_close then
    redis.call('SET', s1_status_key, s1_res_status_json)
    if s1_del_claim then redis.call('DEL', s1_claim_key) end
    if s1_inc_fence then redis.call('SET', s1_fence_key, s1_next_fence) end
end
if s2_close then
    redis.call('SET', s2_status_key, s2_res_status_json)
    if s2_del_claim then redis.call('DEL', s2_claim_key) end
    if s2_inc_fence then redis.call('SET', s2_fence_key, s2_next_fence) end
end
if child_count >= 3 and s3_close then
    redis.call('SET', s3_status_key, s3_res_status_json)
    if s3_del_claim then redis.call('DEL', s3_claim_key) end
    if s3_inc_fence then redis.call('SET', s3_fence_key, s3_next_fence) end
end
if child_count >= 4 and s4_close then
    redis.call('SET', s4_status_key, s4_res_status_json)
    if s4_del_claim then redis.call('DEL', s4_claim_key) end
    if s4_inc_fence then redis.call('SET', s4_fence_key, s4_next_fence) end
end

return {'OK', resolved_wave_json}
"""

CANCEL_DELEGATION_WAVE_LUA = """
local cancellation_key = KEYS[1]
local parent_status_key = KEYS[2]
local parent_children_key = KEYS[3]
local parent_lineage_key = KEYS[4]
local continuation_key = KEYS[5]
local continuation_latest_key = KEYS[6]
local wave_key = KEYS[7]
local resolved_wave_key = KEYS[8]

local s1_status_key = KEYS[9]
local s1_claim_key = KEYS[10]
local s1_fence_key = KEYS[11]
local s1_outcome_key = KEYS[12]
local s1_result_key = KEYS[13]
local s1_lineage_key = KEYS[14]
local s1_cap_key = KEYS[15]
local s1_input_key = KEYS[16]
local s1_marker_key = KEYS[17]
local s1_expiry_key = KEYS[18]

local s2_status_key = KEYS[19]
local s2_claim_key = KEYS[20]
local s2_fence_key = KEYS[21]
local s2_outcome_key = KEYS[22]
local s2_result_key = KEYS[23]
local s2_lineage_key = KEYS[24]
local s2_cap_key = KEYS[25]
local s2_input_key = KEYS[26]
local s2_marker_key = KEYS[27]
local s2_expiry_key = KEYS[28]

local s3_status_key = KEYS[29]
local s3_claim_key = KEYS[30]
local s3_fence_key = KEYS[31]
local s3_outcome_key = KEYS[32]
local s3_result_key = KEYS[33]
local s3_lineage_key = KEYS[34]
local s3_cap_key = KEYS[35]
local s3_input_key = KEYS[36]
local s3_marker_key = KEYS[37]
local s3_expiry_key = KEYS[38]

local s4_status_key = KEYS[39]
local s4_claim_key = KEYS[40]
local s4_fence_key = KEYS[41]
local s4_outcome_key = KEYS[42]
local s4_result_key = KEYS[43]
local s4_lineage_key = KEYS[44]
local s4_cap_key = KEYS[45]
local s4_input_key = KEYS[46]
local s4_marker_key = KEYS[47]
local s4_expiry_key = KEYS[48]

local parent_task_id = ARGV[1]
local wave_id = ARGV[2]
local cancellation_decision_id = ARGV[3]
local operator_principal_id = ARGV[4]
local semantic_fingerprint = ARGV[5]
local now_iso = ARGV[6]
local child_count = tonumber(ARGV[7])
local accepted_wave_fp = ARGV[8]
local cancellation_record_json = ARGV[9]
local pre_cont_closed_json = ARGV[10]
local parent_cancelled_status_json = ARGV[11]
local s1_id = ARGV[12]
local s2_id = ARGV[13]
local s3_id = ARGV[14]
local s4_id = ARGV[15]

local function key_type(key)
    local t = redis.call('TYPE', key)
    if type(t) == 'table' then return t['ok'] end
    return tostring(t)
end

local function decode_object(raw)
    local ok, tbl = pcall(cjson.decode, raw)
    if not ok or type(tbl) ~= 'table' then return nil end
    return tbl
end

local function check_str_key(str_k)
    local kt = key_type(str_k)
    if kt ~= 'none' and kt ~= 'string' then
        return 'WRONG_KEY_TYPE_' .. str_k
    end
    return nil
end

local function validate_sibling_state(
    sid,
    st_k, cl_k, fn_k, out_k, res_k, lin_k, cap_k, inp_k, mrk_k, exp_k,
    resolved_mode, rw_outcomes
)
    if redis.call('SISMEMBER', parent_children_key, sid) ~= 1 then
        return 'CHILD_NOT_MEMBER', nil, nil, 0, false
    end
    local kt_err = check_str_key(lin_k) or check_str_key(cap_k) or check_str_key(inp_k) or
                   check_str_key(fn_k) or check_str_key(st_k) or check_str_key(cl_k) or
                   check_str_key(out_k) or check_str_key(res_k) or check_str_key(mrk_k) or check_str_key(exp_k)
    if kt_err then
        return kt_err, nil, nil, 0, false
    end
    if redis.call('EXISTS', lin_k) == 0 then return 'LINEAGE_NOT_FOUND', nil, nil, 0, false end
    local lin = decode_object(redis.call('GET', lin_k))
    if not lin or lin['task_id'] ~= sid or lin['parent_task_id'] ~= parent_task_id or lin['root_task_id'] ~= parent_task_id then
        return 'CHILD_LINEAGE_MISMATCH', nil, nil, 0, false
    end
    if redis.call('EXISTS', cap_k) == 0 then return 'MISSING_REQUIRED_CAPABILITY', nil, nil, 0, false end
    local cap = decode_object(redis.call('GET', cap_k))
    if not cap or cap['task_id'] ~= sid or cap['required_capability'] ~= 'analysis' then
        return 'CORRUPT_CHILD_CAPABILITY', nil, nil, 0, false
    end
    if redis.call('EXISTS', inp_k) == 0 then return 'MISSING_CHILD_INPUT', nil, nil, 0, false end
    local inp = decode_object(redis.call('GET', inp_k))
    if not inp or inp['task_id'] ~= sid or inp['input_version'] ~= 1 or
       type(inp['prompt']) ~= 'string' or string.match(inp['prompt'], '^%s*$') or
       type(inp['artifact_references']) ~= 'table' or #inp['artifact_references'] < 1 then
        return 'CORRUPT_CHILD_INPUT', nil, nil, 0, false
    end
    if redis.call('EXISTS', fn_k) == 0 then return 'FENCE_NOT_FOUND', nil, nil, 0, false end
    local s_fence = tonumber(redis.call('GET', fn_k))
    if not s_fence or s_fence < 0 or math.floor(s_fence) ~= s_fence then
        return 'CORRUPT_FENCE', nil, nil, 0, false
    end
    if redis.call('EXISTS', st_k) == 0 then return 'TASK_NOT_FOUND', nil, nil, 0, false end
    local st = decode_object(redis.call('GET', st_k))
    if not st or st['task_id'] ~= sid or type(st['status']) ~= 'string' or st['status'] == '' then
        return 'CORRUPT_STATUS', nil, nil, 0, false
    end
    local cur_st = st['status']
    local claim_exists = (redis.call('EXISTS', cl_k) == 1)
    local outcome_exists = (redis.call('EXISTS', out_k) == 1)
    local result_exists = (redis.call('EXISTS', res_k) == 1)
    local marker_exists = (redis.call('EXISTS', mrk_k) == 1)
    local expiry_exists = (redis.call('EXISTS', exp_k) == 1)

    if cur_st == 'completed' then
        if claim_exists or s_fence < 1 or not outcome_exists or not result_exists or not marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        local out_tbl = decode_object(redis.call('GET', out_k))
        local res_tbl = decode_object(redis.call('GET', res_k))
        local mrk_tbl = decode_object(redis.call('GET', mrk_k))
        if not out_tbl or out_tbl['task_id'] ~= sid or out_tbl['terminal_status'] ~= 'completed' or
           type(out_tbl['outcome_id']) ~= 'string' or out_tbl['outcome_id'] == '' or
           not res_tbl or res_tbl['task_id'] ~= sid or res_tbl['status'] ~= 'completed' or
           not mrk_tbl or mrk_tbl['child_task_id'] ~= sid or mrk_tbl['parent_task_id'] ~= parent_task_id or
           mrk_tbl['outcome_id'] ~= out_tbl['outcome_id'] or mrk_tbl['continuation_version'] ~= 2 then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        if resolved_mode then
            local snap = rw_outcomes[sid]
            if type(snap) ~= 'table' or snap['status'] ~= 'completed' or snap['outcome_id'] ~= out_tbl['outcome_id'] then
                return 'CORRUPT_WAVE_PROJECTION', nil, nil, 0, false
            end
        end
        return nil, 'completed', out_tbl['outcome_id'], s_fence, false
    elseif cur_st == 'failed' or cur_st == 'claim_timed_out' then
        if claim_exists or not outcome_exists or result_exists or not marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        local out_tbl = decode_object(redis.call('GET', out_k))
        local mrk_tbl = decode_object(redis.call('GET', mrk_k))
        if not out_tbl or out_tbl['task_id'] ~= sid or out_tbl['terminal_status'] ~= cur_st or
           type(out_tbl['outcome_id']) ~= 'string' or out_tbl['outcome_id'] == '' or
           not mrk_tbl or mrk_tbl['child_task_id'] ~= sid or mrk_tbl['parent_task_id'] ~= parent_task_id or
           mrk_tbl['outcome_id'] ~= out_tbl['outcome_id'] or mrk_tbl['continuation_version'] ~= 2 then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        if resolved_mode then
            local snap = rw_outcomes[sid]
            if type(snap) ~= 'table' or snap['status'] ~= cur_st or snap['outcome_id'] ~= out_tbl['outcome_id'] then
                return 'CORRUPT_WAVE_PROJECTION', nil, nil, 0, false
            end
        end
        return nil, cur_st, out_tbl['outcome_id'], s_fence, false
    elseif cur_st == 'pending' then
        if resolved_mode or claim_exists or outcome_exists or result_exists or marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, 'pending', nil, s_fence, false
    elseif cur_st == 'working' then
        if resolved_mode or not claim_exists or outcome_exists or result_exists or marker_exists or s_fence < 1 then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        local cl_tbl = decode_object(redis.call('GET', cl_k))
        if not cl_tbl or cl_tbl['task_id'] ~= sid or cl_tbl['execution_kind'] ~= 'delegated_child' or
           cl_tbl['fencing_epoch'] ~= s_fence or cl_tbl['assigned_capability'] ~= 'analysis' then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, 'working', nil, s_fence, true
    elseif cur_st == 'recovery_required' then
        if resolved_mode or claim_exists or outcome_exists or result_exists or marker_exists or s_fence < 1 or not expiry_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        local ce_tbl = decode_object(redis.call('GET', exp_k))
        if not ce_tbl or ce_tbl['task_id'] ~= sid or ce_tbl['expired_fencing_epoch'] ~= s_fence or
           ce_tbl['prior_claimable_status'] ~= 'pending' or ce_tbl['execution_kind'] ~= 'delegated_child' then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, 'recovery_required', nil, s_fence, false
    elseif cur_st == 'ended / barrier_resolved' then
        if not resolved_mode or claim_exists or outcome_exists or result_exists or marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        local snap = rw_outcomes[sid]
        if type(snap) ~= 'table' or snap['status'] ~= 'ended / barrier_resolved' or snap['outcome_id'] ~= cjson.null then
            return 'CORRUPT_WAVE_PROJECTION', nil, nil, 0, false
        end
        return nil, 'ended / barrier_resolved', nil, s_fence, false
    end
    return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
end

-- =============================================================================
-- PHASE 1: Strictly Read-Only Validations & Exact Replay Check
-- =============================================================================

if not child_count or math.floor(child_count) ~= child_count or child_count < 2 or child_count > 4 then
    return {'CORRUPT_INPUT'}
end

for _, k in ipairs({cancellation_key, parent_status_key, parent_lineage_key, continuation_key, continuation_latest_key, wave_key, resolved_wave_key}) do
    local kt = key_type(k)
    if kt ~= 'none' and kt ~= 'string' then
        return {'WRONG_KEY_TYPE_' .. k, kt}
    end
end
local children_t = key_type(parent_children_key)
if children_t ~= 'none' and children_t ~= 'set' then
    return {'WRONG_KEY_TYPE_' .. parent_children_key, children_t}
end

local is_replay = (redis.call('EXISTS', cancellation_key) == 1)
local stored_dec_raw = nil
if is_replay then
    stored_dec_raw = redis.call('GET', cancellation_key)
    local stored_dec = decode_object(stored_dec_raw)
    if not stored_dec then return {'CORRUPT_CANCELLATION_RECORD'} end
    if stored_dec['cancellation_decision_id'] ~= cancellation_decision_id or
       stored_dec['parent_task_id'] ~= parent_task_id or
       stored_dec['wave_id'] ~= wave_id or
       stored_dec['operator_principal_id'] ~= operator_principal_id or
       stored_dec['semantic_fingerprint'] ~= semantic_fingerprint or
       stored_dec['resolution'] ~= 'CANCELLED' or
       stored_dec['prior_parent_status'] ~= 'waiting_for_children' or
       stored_dec['terminal_parent_status'] ~= 'cancelled' then
        return {'CANCELLATION_CONFLICT'}
    end
end

local pre_dec = decode_object(cancellation_record_json)
local pre_cont = decode_object(pre_cont_closed_json)
local pre_ps = decode_object(parent_cancelled_status_json)
if not pre_dec or not pre_cont or not pre_ps or
   pre_dec['cancellation_decision_id'] ~= cancellation_decision_id or
   pre_dec['parent_task_id'] ~= parent_task_id or
   pre_dec['wave_id'] ~= wave_id or
   pre_dec['operator_principal_id'] ~= operator_principal_id or
   pre_dec['semantic_fingerprint'] ~= semantic_fingerprint or
   pre_cont['version'] ~= 2 or pre_cont['continuation_version'] ~= 2 or
   pre_cont['parent_task_id'] ~= parent_task_id or pre_cont['wave_id'] ~= wave_id or
   pre_cont['state'] ~= 'closed' or pre_cont['wave_resolution'] ~= cjson.null or
   pre_ps['task_id'] ~= parent_task_id or pre_ps['status'] ~= 'cancelled' then
    return {'CORRUPT_INPUT'}
end

if redis.call('EXISTS', parent_lineage_key) == 0 then return {'PARENT_NOT_FOUND'} end
local p_lin = decode_object(redis.call('GET', parent_lineage_key))
if not p_lin or p_lin['task_id'] ~= parent_task_id or p_lin['root_task_id'] ~= parent_task_id or
   p_lin['parent_task_id'] ~= cjson.null then
    return {'CORRUPT_PARENT_LINEAGE'}
end

if redis.call('EXISTS', parent_status_key) == 0 then return {'PARENT_NOT_FOUND'} end
local parent_status = decode_object(redis.call('GET', parent_status_key))
if not parent_status or parent_status['task_id'] ~= parent_task_id or type(parent_status['status']) ~= 'string' then
    return {'CORRUPT_PARENT_STATUS'}
end

if redis.call('EXISTS', continuation_latest_key) == 0 then return {'CONTINUATION_LATEST_MISMATCH'} end
if redis.call('GET', continuation_latest_key) ~= '2' then return {'CONTINUATION_LATEST_MISMATCH'} end

if redis.call('EXISTS', continuation_key) == 0 then return {'MISSING_CONTINUATION'} end
local sc = decode_object(redis.call('GET', continuation_key))
if not sc or sc['version'] ~= 2 or sc['continuation_version'] ~= 2 or
   sc['parent_task_id'] ~= parent_task_id or sc['wave_id'] ~= wave_id or
   sc['stage'] ~= 'incorporate_analysis_child' or sc['continuation_capability'] ~= 'research' or
   sc['created_at'] ~= pre_cont['created_at'] or
   sc['input_version'] ~= pre_cont['input_version'] or
   type(sc['required_child_task_ids']) ~= 'table' or #sc['required_child_task_ids'] ~= child_count then
    return {'CORRUPT_CONTINUATION'}
end

if redis.call('EXISTS', wave_key) == 0 then return {'MISSING_ACCEPTED_WAVE'} end
local sw = decode_object(redis.call('GET', wave_key))
if not sw or sw['version'] ~= 1 or sw['wave_id'] ~= wave_id or sw['parent_task_id'] ~= parent_task_id or
   sw['state'] ~= 'unresolved' or type(sw['accepted_at']) ~= 'string' or sw['accepted_at'] == '' or
   type(sw['wave_deadline']) ~= 'string' or sw['wave_deadline'] == '' or
   type(sw['required_child_task_ids']) ~= 'table' or #sw['required_child_task_ids'] ~= child_count then
    return {'CORRUPT_WAVE_PROJECTION'}
end

local dd = sw['deadline_derivation']
if type(dd) ~= 'table' or dd['policy_id'] ~= 'deterministic_bounded_baseline_v1' or
   dd['policy_version'] ~= 1 or dd['required_child_count'] ~= child_count or
   dd['accepted_at'] ~= sw['accepted_at'] or dd['wave_deadline'] ~= sw['wave_deadline'] then
    return {'CORRUPT_WAVE_PROJECTION'}
end

local policy = sw['barrier_policy']
local frac_num = nil
local frac_den = nil
local derived_req_count = nil
if policy == 'WAIT_ALL' then
    if sw['required_success_fraction'] ~= cjson.null or sw['required_success_count'] ~= cjson.null then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
elseif policy == 'PARTIAL' then
    local rf = sw['required_success_fraction']
    if type(rf) ~= 'table' or type(rf['numerator']) ~= 'number' or type(rf['denominator']) ~= 'number' then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
    frac_num = rf['numerator']
    frac_den = rf['denominator']
    if math.floor(frac_num) ~= frac_num or math.floor(frac_den) ~= frac_den or
       frac_num < 1 or frac_den < 1 or frac_num > frac_den or frac_den > 10000 then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
    derived_req_count = math.floor((child_count * frac_num + frac_den - 1) / frac_den)
    if derived_req_count < 1 or derived_req_count > child_count or sw['required_success_count'] ~= derived_req_count then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
else
    return {'CORRUPT_WAVE_PROJECTION'}
end

if redis.call('EXISTS', parent_children_key) == 0 or redis.call('SCARD', parent_children_key) ~= child_count then
    return {'CORRUPT_CHILDREN'}
end

local slot_ids = {s1_id, s2_id}
if child_count >= 3 then slot_ids[3] = s3_id end
if child_count >= 4 then slot_ids[4] = s4_id end
for idx = 1, child_count do
    local sid = slot_ids[idx]
    if type(sid) ~= 'string' or sid == '' or
       sw['required_child_task_ids'][idx] ~= sid or
       sc['required_child_task_ids'][idx] ~= sid then
        return {'WAVE_CONSISTENCY_MISMATCH'}
    end
end

local has_resolved = (redis.call('EXISTS', resolved_wave_key) == 1)
if is_replay then
    if not has_resolved then
        return {'CORRUPT_CANCELLATION_RECORD'}
    end
    local rw_raw = redis.call('GET', resolved_wave_key)
    local rw_tbl = decode_object(rw_raw)
    if not rw_tbl or rw_tbl['version'] ~= 2 or rw_tbl['wave_id'] ~= wave_id or
       rw_tbl['parent_task_id'] ~= parent_task_id or rw_tbl['resolution'] ~= 'CANCELLED' or
       rw_tbl['barrier_policy'] ~= policy or rw_tbl['accepted_at'] ~= sw['accepted_at'] or
       rw_tbl['wave_deadline'] ~= sw['wave_deadline'] or
       rw_tbl['accepted_wave_fingerprint'] ~= accepted_wave_fp or
       type(rw_tbl['child_outcomes']) ~= 'table' or type(rw_tbl['required_child_task_ids']) ~= 'table' or
       #rw_tbl['required_child_task_ids'] ~= child_count then
        return {'CORRUPT_RESOLVED_WAVE'}
    end
    if sc['state'] ~= 'closed' or sc['wave_resolution'] ~= cjson.null then
        return {'CORRUPT_CONTINUATION'}
    end
    if parent_status['status'] ~= 'cancelled' then
        return {'CORRUPT_PARENT_STATUS'}
    end
    local rw_out = rw_tbl['child_outcomes']
    local e1 = validate_sibling_state(s1_id, s1_status_key, s1_claim_key, s1_fence_key, s1_outcome_key, s1_result_key, s1_lineage_key, s1_cap_key, s1_input_key, s1_marker_key, s1_expiry_key, true, rw_out)
    if e1 then return {e1} end
    local e2 = validate_sibling_state(s2_id, s2_status_key, s2_claim_key, s2_fence_key, s2_outcome_key, s2_result_key, s2_lineage_key, s2_cap_key, s2_input_key, s2_marker_key, s2_expiry_key, true, rw_out)
    if e2 then return {e2} end
    if child_count >= 3 then
        local e3 = validate_sibling_state(s3_id, s3_status_key, s3_claim_key, s3_fence_key, s3_outcome_key, s3_result_key, s3_lineage_key, s3_cap_key, s3_input_key, s3_marker_key, s3_expiry_key, true, rw_out)
        if e3 then return {e3} end
    end
    if child_count >= 4 then
        local e4 = validate_sibling_state(s4_id, s4_status_key, s4_claim_key, s4_fence_key, s4_outcome_key, s4_result_key, s4_lineage_key, s4_cap_key, s4_input_key, s4_marker_key, s4_expiry_key, true, rw_out)
        if e4 then return {e4} end
    end
    return {'IDEMPOTENT', stored_dec_raw, rw_raw}
end

if has_resolved then
    local rw_tbl = decode_object(redis.call('GET', resolved_wave_key))
    if not rw_tbl then return {'CORRUPT_RESOLVED_WAVE'} end
    return {'WAVE_ALREADY_RESOLVED', tostring(rw_tbl['resolution'])}
end

if parent_status['status'] ~= 'waiting_for_children' then
    return {'PARENT_NOT_WAITING', tostring(parent_status['status'])}
end
if sc['state'] ~= 'active' or sc['wave_resolution'] ~= cjson.null then
    return {'CONTINUATION_NOT_ACTIVE', tostring(sc['state'])}
end

local e1, st1, oid1, fn1, hc1 = validate_sibling_state(s1_id, s1_status_key, s1_claim_key, s1_fence_key, s1_outcome_key, s1_result_key, s1_lineage_key, s1_cap_key, s1_input_key, s1_marker_key, s1_expiry_key, false, nil)
if e1 then return {e1} end
local e2, st2, oid2, fn2, hc2 = validate_sibling_state(s2_id, s2_status_key, s2_claim_key, s2_fence_key, s2_outcome_key, s2_result_key, s2_lineage_key, s2_cap_key, s2_input_key, s2_marker_key, s2_expiry_key, false, nil)
if e2 then return {e2} end
local st3, oid3, fn3, hc3 = nil, nil, 0, false
if child_count >= 3 then
    local e3
    e3, st3, oid3, fn3, hc3 = validate_sibling_state(s3_id, s3_status_key, s3_claim_key, s3_fence_key, s3_outcome_key, s3_result_key, s3_lineage_key, s3_cap_key, s3_input_key, s3_marker_key, s3_expiry_key, false, nil)
    if e3 then return {e3} end
end
local st4, oid4, fn4, hc4 = nil, nil, 0, false
if child_count >= 4 then
    local e4
    e4, st4, oid4, fn4, hc4 = validate_sibling_state(s4_id, s4_status_key, s4_claim_key, s4_fence_key, s4_outcome_key, s4_result_key, s4_lineage_key, s4_cap_key, s4_input_key, s4_marker_key, s4_expiry_key, false, nil)
    if e4 then return {e4} end
end

local slot_statuses = {st1, st2}
local slot_outcomes = {oid1, oid2}
local slot_fences = {fn1, fn2}
local slot_has_claim = {hc1, hc2}
if child_count >= 3 then
    slot_statuses[3] = st3; slot_outcomes[3] = oid3; slot_fences[3] = fn3; slot_has_claim[3] = hc3
end
if child_count >= 4 then
    slot_statuses[4] = st4; slot_outcomes[4] = oid4; slot_fences[4] = fn4; slot_has_claim[4] = hc4
end

local completed_count = 0
local terminal_count = 0
local unresolved_count = 0
local potentially_successful_count = 0
for idx = 1, child_count do
    local st_val = slot_statuses[idx]
    if st_val == 'completed' then
        completed_count = completed_count + 1
        terminal_count = terminal_count + 1
    elseif st_val == 'failed' or st_val == 'claim_timed_out' then
        terminal_count = terminal_count + 1
    elseif st_val == 'pending' or st_val == 'working' or st_val == 'recovery_required' then
        unresolved_count = unresolved_count + 1
        potentially_successful_count = potentially_successful_count + 1
    else
        return {'CORRUPT_CHILD_STATE'}
    end
end

if unresolved_count < 1 then
    return {'CORRUPT_WAVE_PROJECTION'}
end
if policy == 'WAIT_ALL' then
    if terminal_count >= child_count then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
else
    if completed_count >= derived_req_count or completed_count + potentially_successful_count < derived_req_count then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
end

local child_outcomes_tbl = {}
local residual_ids_tbl = {}
local close_flags = {false, false, false, false}
local del_claim_flags = {false, false, false, false}
local inc_fence_flags = {false, false, false, false}
local next_fences = {'', '', '', ''}
local res_status_jsons = {'', '', '', ''}

for idx = 1, child_count do
    local sid = slot_ids[idx]
    local st_val = slot_statuses[idx]
    if st_val == 'completed' or st_val == 'failed' or st_val == 'claim_timed_out' then
        child_outcomes_tbl[sid] = {
            outcome_id = slot_outcomes[idx],
            status = st_val,
        }
    else
        child_outcomes_tbl[sid] = {
            outcome_id = cjson.null,
            status = 'ended / barrier_resolved',
        }
        table.insert(residual_ids_tbl, sid)
        close_flags[idx] = true
        del_claim_flags[idx] = slot_has_claim[idx]
        if slot_fences[idx] > 0 or slot_has_claim[idx] then
            inc_fence_flags[idx] = true
            next_fences[idx] = tostring(slot_fences[idx] + 1)
        end
        res_status_jsons[idx] = cjson.encode({
            status = 'ended / barrier_resolved',
            task_id = sid,
            updated_at = now_iso,
            wave_id = wave_id,
        })
    end
end

local s1_close, s1_del_claim, s1_inc_fence, s1_next_fence, s1_res_status_json = close_flags[1], del_claim_flags[1], inc_fence_flags[1], next_fences[1], res_status_jsons[1]
local s2_close, s2_del_claim, s2_inc_fence, s2_next_fence, s2_res_status_json = close_flags[2], del_claim_flags[2], inc_fence_flags[2], next_fences[2], res_status_jsons[2]
local s3_close, s3_del_claim, s3_inc_fence, s3_next_fence, s3_res_status_json = close_flags[3], del_claim_flags[3], inc_fence_flags[3], next_fences[3], res_status_jsons[3]
local s4_close, s4_del_claim, s4_inc_fence, s4_next_fence, s4_res_status_json = close_flags[4], del_claim_flags[4], inc_fence_flags[4], next_fences[4], res_status_jsons[4]

local req_count_val = cjson.null
local req_frac_val = cjson.null
if policy == 'PARTIAL' then
    req_count_val = derived_req_count
    req_frac_val = {
        denominator = frac_den,
        numerator = frac_num,
    }
end

local rw_tbl = {
    accepted_at = sw['accepted_at'],
    accepted_wave_fingerprint = accepted_wave_fp,
    barrier_policy = policy,
    child_outcomes = child_outcomes_tbl,
    completed_count = completed_count,
    deadline_policy_id = dd['policy_id'],
    deadline_policy_version = dd['policy_version'],
    parent_task_id = parent_task_id,
    required_child_task_ids = sw['required_child_task_ids'],
    required_success_count = req_count_val,
    required_success_fraction = req_frac_val,
    residual_child_task_ids = residual_ids_tbl,
    residual_disposition = 'ended / barrier_resolved',
    resolution = 'CANCELLED',
    resolved_at = now_iso,
    terminal_count = terminal_count,
    triggering_assignment_id = cjson.null,
    triggering_child_task_id = cjson.null,
    triggering_fencing_epoch = cjson.null,
    unresolved_count = unresolved_count,
    version = 2,
    wave_deadline = sw['wave_deadline'],
    wave_id = wave_id,
}
local ok_rw_enc, resolved_wave_json = pcall(cjson.encode, rw_tbl)
if not ok_rw_enc or type(resolved_wave_json) ~= 'string' or #resolved_wave_json > 16384 then
    return {'CORRUPT_WAVE_PROJECTION'}
end

-- =============================================================================
-- PHASE 2: Prevalidated Atomic Writes Only (Zero Hint, Zero Parent Outcome)
-- =============================================================================
redis.call('SET', cancellation_key, cancellation_record_json)
redis.call('SET', resolved_wave_key, resolved_wave_json)
redis.call('SET', continuation_key, pre_cont_closed_json)
redis.call('SET', parent_status_key, parent_cancelled_status_json)

if s1_close then
    redis.call('SET', s1_status_key, s1_res_status_json)
    if s1_del_claim then redis.call('DEL', s1_claim_key) end
    if s1_inc_fence then redis.call('SET', s1_fence_key, s1_next_fence) end
end
if s2_close then
    redis.call('SET', s2_status_key, s2_res_status_json)
    if s2_del_claim then redis.call('DEL', s2_claim_key) end
    if s2_inc_fence then redis.call('SET', s2_fence_key, s2_next_fence) end
end
if child_count >= 3 and s3_close then
    redis.call('SET', s3_status_key, s3_res_status_json)
    if s3_del_claim then redis.call('DEL', s3_claim_key) end
    if s3_inc_fence then redis.call('SET', s3_fence_key, s3_next_fence) end
end
if child_count >= 4 and s4_close then
    redis.call('SET', s4_status_key, s4_res_status_json)
    if s4_del_claim then redis.call('DEL', s4_claim_key) end
    if s4_inc_fence then redis.call('SET', s4_fence_key, s4_next_fence) end
end

return {'OK', cancellation_record_json, resolved_wave_json}
"""

MATERIALIZE_WAVE_CHILD_CLAIM_TIMEOUT_LUA = """
local child_status_key = KEYS[1]
local child_claim_key = KEYS[2]
local child_deadline_key = KEYS[3]
local child_outcome_key = KEYS[4]
local child_result_key = KEYS[5]
local child_lineage_key = KEYS[6]
local child_marker_key = KEYS[7]
local child_expiry_key = KEYS[8]
local parent_status_key = KEYS[9]
local parent_children_key = KEYS[10]
local parent_lineage_key = KEYS[11]
local continuation_key = KEYS[12]
local continuation_latest_key = KEYS[13]
local wave_key = KEYS[14]
local resolved_wave_key = KEYS[15]
local hint_queue_key = KEYS[16]

local s1_status_key = KEYS[17]
local s1_claim_key = KEYS[18]
local s1_fence_key = KEYS[19]
local s1_outcome_key = KEYS[20]
local s1_result_key = KEYS[21]
local s1_lineage_key = KEYS[22]
local s1_cap_key = KEYS[23]
local s1_input_key = KEYS[24]
local s1_marker_key = KEYS[25]
local s1_expiry_key = KEYS[26]

local s2_status_key = KEYS[27]
local s2_claim_key = KEYS[28]
local s2_fence_key = KEYS[29]
local s2_outcome_key = KEYS[30]
local s2_result_key = KEYS[31]
local s2_lineage_key = KEYS[32]
local s2_cap_key = KEYS[33]
local s2_input_key = KEYS[34]
local s2_marker_key = KEYS[35]
local s2_expiry_key = KEYS[36]

local s3_status_key = KEYS[37]
local s3_claim_key = KEYS[38]
local s3_fence_key = KEYS[39]
local s3_outcome_key = KEYS[40]
local s3_result_key = KEYS[41]
local s3_lineage_key = KEYS[42]
local s3_cap_key = KEYS[43]
local s3_input_key = KEYS[44]
local s3_marker_key = KEYS[45]
local s3_expiry_key = KEYS[46]

local s4_status_key = KEYS[47]
local s4_claim_key = KEYS[48]
local s4_fence_key = KEYS[49]
local s4_outcome_key = KEYS[50]
local s4_result_key = KEYS[51]
local s4_lineage_key = KEYS[52]
local s4_cap_key = KEYS[53]
local s4_input_key = KEYS[54]
local s4_marker_key = KEYS[55]
local s4_expiry_key = KEYS[56]

local child_task_id = ARGV[1]
local parent_task_id = ARGV[2]
local wave_id = ARGV[3]
local now_iso = ARGV[4]
local child_outcome_json = ARGV[5]
local child_status_json = ARGV[6]
local parent_status_json = ARGV[7]
local marker_json = ARGV[8]
local hint_json = ARGV[9]
local child_count = tonumber(ARGV[10])
local accepted_wave_fp = ARGV[11]
local pre_cont_sat_json = ARGV[12]
local pre_cont_unsat_json = ARGV[13]
local s1_id = ARGV[14]
local s2_id = ARGV[15]
local s3_id = ARGV[16]
local s4_id = ARGV[17]

local function key_type(key)
    local t = redis.call('TYPE', key)
    if type(t) == 'table' then return t['ok'] end
    return tostring(t)
end

local function decode_object(raw)
    local ok, tbl = pcall(cjson.decode, raw)
    if not ok or type(tbl) ~= 'table' then return nil end
    return tbl
end

local function check_str_key(str_k)
    local kt = key_type(str_k)
    if kt ~= 'none' and kt ~= 'string' then
        return 'WRONG_KEY_TYPE_' .. str_k
    end
    return nil
end

local function validate_sibling_readonly(
    sid,
    st_k, cl_k, fn_k, out_k, res_k, lin_k, cap_k, inp_k, mrk_k, exp_k,
    trig_outcome_id
)
    if redis.call('SISMEMBER', parent_children_key, sid) ~= 1 then
        return 'CHILD_NOT_MEMBER', nil, nil, 0, false
    end
    local kt_err = check_str_key(lin_k) or check_str_key(cap_k) or check_str_key(inp_k) or
                   check_str_key(fn_k) or check_str_key(st_k) or check_str_key(cl_k) or
                   check_str_key(out_k) or check_str_key(res_k) or check_str_key(mrk_k) or check_str_key(exp_k)
    if kt_err then
        return kt_err, nil, nil, 0, false
    end
    if redis.call('EXISTS', lin_k) == 0 then return 'CHILD_LINEAGE_NOT_FOUND', nil, nil, 0, false end
    local lin = decode_object(redis.call('GET', lin_k))
    if not lin or lin['task_id'] ~= sid or lin['parent_task_id'] ~= parent_task_id or lin['root_task_id'] ~= parent_task_id then
        return 'LINEAGE_PARENT_MISMATCH', nil, nil, 0, false
    end
    if redis.call('EXISTS', cap_k) == 0 then return 'MISSING_REQUIRED_CAPABILITY', nil, nil, 0, false end
    local cap = decode_object(redis.call('GET', cap_k))
    if not cap or cap['task_id'] ~= sid or cap['required_capability'] ~= 'analysis' then
        return 'CORRUPT_CHILD_CAPABILITY', nil, nil, 0, false
    end
    if redis.call('EXISTS', inp_k) == 0 then return 'MISSING_CHILD_INPUT', nil, nil, 0, false end
    local inp = decode_object(redis.call('GET', inp_k))
    if not inp or inp['task_id'] ~= sid or inp['input_version'] ~= 1 or
       type(inp['prompt']) ~= 'string' or string.match(inp['prompt'], '^%s*$') or
       type(inp['artifact_references']) ~= 'table' or #inp['artifact_references'] < 1 then
        return 'CORRUPT_CHILD_INPUT', nil, nil, 0, false
    end
    if redis.call('EXISTS', fn_k) == 0 then return 'FENCE_NOT_FOUND', nil, nil, 0, false end
    local s_fence = tonumber(redis.call('GET', fn_k))
    if not s_fence or s_fence < 0 or math.floor(s_fence) ~= s_fence then
        return 'CORRUPT_FENCE', nil, nil, 0, false
    end
    if redis.call('EXISTS', st_k) == 0 then return 'CHILD_NOT_FOUND', nil, nil, 0, false end
    local st = decode_object(redis.call('GET', st_k))
    if not st or st['task_id'] ~= sid or type(st['status']) ~= 'string' or st['status'] == '' then
        return 'CORRUPT_CHILD_STATUS', nil, nil, 0, false
    end
    local cur_st = st['status']

    if sid == child_task_id then
        return nil, 'claim_timed_out', trig_outcome_id, s_fence, false
    end

    local claim_exists = (redis.call('EXISTS', cl_k) == 1)
    local outcome_exists = (redis.call('EXISTS', out_k) == 1)
    local result_exists = (redis.call('EXISTS', res_k) == 1)
    local marker_exists = (redis.call('EXISTS', mrk_k) == 1)
    local expiry_exists = (redis.call('EXISTS', exp_k) == 1)

    if cur_st == 'completed' then
        if claim_exists or s_fence < 1 or not outcome_exists or not result_exists or not marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        local out_tbl = decode_object(redis.call('GET', out_k))
        local res_tbl = decode_object(redis.call('GET', res_k))
        local mrk_tbl = decode_object(redis.call('GET', mrk_k))
        if not out_tbl or out_tbl['task_id'] ~= sid or out_tbl['terminal_status'] ~= 'completed' or
           type(out_tbl['outcome_id']) ~= 'string' or out_tbl['outcome_id'] == '' or
           not res_tbl or res_tbl['task_id'] ~= sid or res_tbl['status'] ~= 'completed' or
           not mrk_tbl or mrk_tbl['child_task_id'] ~= sid or mrk_tbl['parent_task_id'] ~= parent_task_id or
           mrk_tbl['outcome_id'] ~= out_tbl['outcome_id'] or mrk_tbl['continuation_version'] ~= 2 then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, 'completed', out_tbl['outcome_id'], s_fence, false
    elseif cur_st == 'failed' or cur_st == 'claim_timed_out' then
        if claim_exists or not outcome_exists or result_exists or not marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        local out_tbl = decode_object(redis.call('GET', out_k))
        local mrk_tbl = decode_object(redis.call('GET', mrk_k))
        if not out_tbl or out_tbl['task_id'] ~= sid or out_tbl['terminal_status'] ~= cur_st or
           type(out_tbl['outcome_id']) ~= 'string' or out_tbl['outcome_id'] == '' or
           not mrk_tbl or mrk_tbl['child_task_id'] ~= sid or mrk_tbl['parent_task_id'] ~= parent_task_id or
           mrk_tbl['outcome_id'] ~= out_tbl['outcome_id'] or mrk_tbl['continuation_version'] ~= 2 then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, cur_st, out_tbl['outcome_id'], s_fence, false
    elseif cur_st == 'pending' then
        if claim_exists or outcome_exists or result_exists or marker_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, 'pending', nil, s_fence, false
    elseif cur_st == 'working' then
        if not claim_exists or outcome_exists or result_exists or marker_exists or s_fence < 1 then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        local cl_tbl = decode_object(redis.call('GET', cl_k))
        if not cl_tbl or cl_tbl['task_id'] ~= sid or cl_tbl['execution_kind'] ~= 'delegated_child' or
           cl_tbl['fencing_epoch'] ~= s_fence or cl_tbl['assigned_capability'] ~= 'analysis' then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, 'working', nil, s_fence, true
    elseif cur_st == 'recovery_required' then
        if claim_exists or outcome_exists or result_exists or marker_exists or s_fence < 1 or not expiry_exists then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        local ce_tbl = decode_object(redis.call('GET', exp_k))
        if not ce_tbl or ce_tbl['task_id'] ~= sid or ce_tbl['expired_fencing_epoch'] ~= s_fence or
           ce_tbl['prior_claimable_status'] ~= 'pending' or ce_tbl['execution_kind'] ~= 'delegated_child' then
            return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
        end
        return nil, 'recovery_required', nil, s_fence, false
    end
    return 'CORRUPT_CHILD_STATE', nil, nil, 0, false
end

-- =============================================================================
-- PHASE 1: Strictly Read-Only Validations & Replay
-- =============================================================================

for _, k in ipairs({child_status_key, child_claim_key, child_deadline_key, child_outcome_key, child_result_key, child_lineage_key, child_marker_key, child_expiry_key, parent_status_key, parent_lineage_key, continuation_key, continuation_latest_key, wave_key, resolved_wave_key}) do
    local t = key_type(k)
    if t ~= 'none' and t ~= 'string' then
        return {'WRONG_KEY_TYPE_' .. k, t}
    end
end
local children_t = key_type(parent_children_key)
if children_t ~= 'none' and children_t ~= 'set' then
    return {'WRONG_KEY_TYPE_' .. parent_children_key, children_t}
end
local hint_t = key_type(hint_queue_key)
if hint_t ~= 'none' and hint_t ~= 'list' then
    return {'WRONG_KEY_TYPE_' .. hint_queue_key, hint_t}
end

if not child_count or math.floor(child_count) ~= child_count or child_count < 2 or child_count > 4 then
    return {'CORRUPT_INPUT'}
end

if redis.call('EXISTS', child_status_key) == 0 then return {'CHILD_NOT_FOUND'} end
local child_status_raw = redis.call('GET', child_status_key)
local child_status_tbl = decode_object(child_status_raw)
if not child_status_tbl or child_status_tbl['task_id'] ~= child_task_id or type(child_status_tbl['status']) ~= 'string' then
    return {'CORRUPT_CHILD_STATUS'}
end
local child_status = child_status_tbl['status']

if redis.call('EXISTS', child_lineage_key) == 0 then return {'CHILD_LINEAGE_NOT_FOUND'} end
local lineage_tbl = decode_object(redis.call('GET', child_lineage_key))
if not lineage_tbl or lineage_tbl['task_id'] ~= child_task_id or
   lineage_tbl['parent_task_id'] ~= parent_task_id or lineage_tbl['root_task_id'] ~= parent_task_id then
    return {'LINEAGE_PARENT_MISMATCH'}
end

if redis.call('EXISTS', parent_lineage_key) == 0 then return {'PARENT_NOT_FOUND'} end
local p_lin = decode_object(redis.call('GET', parent_lineage_key))
if not p_lin or p_lin['task_id'] ~= parent_task_id or p_lin['root_task_id'] ~= parent_task_id or
   p_lin['parent_task_id'] ~= cjson.null then
    return {'CORRUPT_PARENT_LINEAGE'}
end

if redis.call('EXISTS', parent_children_key) == 0 or
   redis.call('SCARD', parent_children_key) ~= child_count or
   redis.call('SISMEMBER', parent_children_key, child_task_id) ~= 1 then
    return {'CHILD_NOT_MEMBER'}
end

if redis.call('EXISTS', continuation_latest_key) == 0 then return {'MISSING_CONTINUATION_LATEST'} end
if redis.call('GET', continuation_latest_key) ~= '2' then return {'CONTINUATION_LATEST_MISMATCH'} end

if redis.call('EXISTS', continuation_key) == 0 then return {'MISSING_CONTINUATION'} end
local sc = decode_object(redis.call('GET', continuation_key))
if not sc or sc['version'] ~= 2 or sc['continuation_version'] ~= 2 or
   sc['parent_task_id'] ~= parent_task_id or sc['wave_id'] ~= wave_id or
   sc['stage'] ~= 'incorporate_analysis_child' or sc['continuation_capability'] ~= 'research' or
   type(sc['required_child_task_ids']) ~= 'table' or #sc['required_child_task_ids'] ~= child_count then
    return {'CORRUPT_CONTINUATION'}
end

if redis.call('EXISTS', wave_key) == 0 then return {'MISSING_ACCEPTED_WAVE'} end
local sw = decode_object(redis.call('GET', wave_key))
if not sw or sw['version'] ~= 1 or sw['wave_id'] ~= wave_id or sw['parent_task_id'] ~= parent_task_id or
   sw['state'] ~= 'unresolved' or type(sw['accepted_at']) ~= 'string' or sw['accepted_at'] == '' or
   type(sw['wave_deadline']) ~= 'string' or sw['wave_deadline'] == '' or
   type(sw['required_child_task_ids']) ~= 'table' or #sw['required_child_task_ids'] ~= child_count then
    return {'CORRUPT_WAVE_PROJECTION'}
end

local dd = sw['deadline_derivation']
if type(dd) ~= 'table' or dd['policy_id'] ~= 'deterministic_bounded_baseline_v1' or
   dd['policy_version'] ~= 1 or dd['required_child_count'] ~= child_count or
   dd['accepted_at'] ~= sw['accepted_at'] or dd['wave_deadline'] ~= sw['wave_deadline'] then
    return {'CORRUPT_WAVE_PROJECTION'}
end

if redis.call('EXISTS', parent_status_key) == 0 then return {'PARENT_NOT_FOUND'} end
local parent_status_tbl = decode_object(redis.call('GET', parent_status_key))
if not parent_status_tbl or parent_status_tbl['task_id'] ~= parent_task_id or type(parent_status_tbl['status']) ~= 'string' then
    return {'CORRUPT_PARENT_STATUS'}
end

-- Exact Replay when child already reached canonical claim_timed_out
if child_status == 'claim_timed_out' then
    if redis.call('EXISTS', child_outcome_key) == 0 then return {'OUTCOME_CONFLICT'} end
    local existing_outcome_raw = redis.call('GET', child_outcome_key)
    local existing_out = decode_object(existing_outcome_raw)
    if not existing_out or existing_out['task_id'] ~= child_task_id or existing_out['terminal_status'] ~= 'claim_timed_out' or
       type(existing_out['outcome_id']) ~= 'string' or existing_out['outcome_id'] == '' then
        return {'CORRUPT_TERMINAL_OUTCOME'}
    end
    if redis.call('EXISTS', child_result_key) == 1 then return {'RESULT_CONFLICT'} end
    if redis.call('EXISTS', child_claim_key) == 1 then return {'ACTIVE_CLAIM_EXISTS'} end
    if redis.call('EXISTS', child_marker_key) == 0 then return {'INCOMPLETE_REPLAY_PROJECTION'} end
    local mrk_tbl = decode_object(redis.call('GET', child_marker_key))
    if not mrk_tbl or mrk_tbl['child_task_id'] ~= child_task_id or mrk_tbl['parent_task_id'] ~= parent_task_id or
       mrk_tbl['outcome_id'] ~= existing_out['outcome_id'] or mrk_tbl['continuation_version'] ~= 2 or
       mrk_tbl['status'] ~= 'claim_timed_out' or mrk_tbl['wave_id'] ~= wave_id then
        return {'INCOMPLETE_REPLAY_PROJECTION'}
    end
    if redis.call('EXISTS', resolved_wave_key) == 1 then
        local rw_tbl = decode_object(redis.call('GET', resolved_wave_key))
        if not rw_tbl or rw_tbl['version'] ~= 2 or rw_tbl['wave_id'] ~= wave_id or rw_tbl['parent_task_id'] ~= parent_task_id or
           type(rw_tbl['child_outcomes']) ~= 'table' then
            return {'CORRUPT_RESOLVED_WAVE'}
        end
        local snap = rw_tbl['child_outcomes'][child_task_id]
        if type(snap) ~= 'table' or snap['status'] ~= 'claim_timed_out' or snap['outcome_id'] ~= existing_out['outcome_id'] then
            return {'INCOMPLETE_REPLAY_PROJECTION'}
        end
        if rw_tbl['resolution'] == 'CANCELLED' then
            if sc['state'] ~= 'closed' or sc['wave_resolution'] ~= cjson.null or parent_status_tbl['status'] ~= 'cancelled' then
                return {'INCOMPLETE_REPLAY_PROJECTION'}
            end
        else
            if sc['state'] ~= 'released' or sc['wave_resolution'] ~= rw_tbl['resolution'] then
                return {'INCOMPLETE_REPLAY_PROJECTION'}
            end
            local p_st = parent_status_tbl['status']
            if p_st ~= 'resumable' and p_st ~= 'working' and p_st ~= 'completed' and p_st ~= 'failed' and p_st ~= 'recovery_required' then
                return {'INCOMPLETE_REPLAY_PROJECTION'}
            end
        end
    else
        if sc['state'] ~= 'active' or sc['wave_resolution'] ~= cjson.null or parent_status_tbl['status'] ~= 'waiting_for_children' then
            return {'INCOMPLETE_REPLAY_PROJECTION'}
        end
    end
    return {'IDEMPOTENT', existing_outcome_raw}
end

-- First materialization preconditions (fail closed after wave resolution or residual closure)
if redis.call('EXISTS', resolved_wave_key) == 1 then
    return {'WAVE_ALREADY_RESOLVED'}
end
if child_status ~= 'pending' then
    return {'CHILD_NOT_PENDING', child_status}
end
if redis.call('EXISTS', child_outcome_key) == 1 then return {'OUTCOME_CONFLICT'} end
if redis.call('EXISTS', child_result_key) == 1 then return {'RESULT_CONFLICT'} end
if redis.call('EXISTS', child_marker_key) == 1 then return {'MARKER_CONFLICT'} end
if redis.call('EXISTS', child_claim_key) == 1 then return {'ACTIVE_CLAIM_EXISTS'} end
if redis.call('EXISTS', child_expiry_key) == 1 then return {'RECOVERY_RECORD_CONFLICT'} end

local deadline_val = nil
if redis.call('EXISTS', child_deadline_key) == 1 then
    deadline_val = redis.call('GET', child_deadline_key)
elseif child_status_tbl['claim_deadline'] and child_status_tbl['claim_deadline'] ~= '' then
    deadline_val = child_status_tbl['claim_deadline']
end
if not deadline_val or deadline_val == '' then
    return {'MISSING_CLAIM_DEADLINE'}
end
if now_iso <= deadline_val then
    return {'DEADLINE_NOT_EXPIRED', deadline_val}
end

if parent_status_tbl['status'] ~= 'waiting_for_children' then
    return {'PARENT_NOT_WAITING', tostring(parent_status_tbl['status'])}
end
if sc['state'] ~= 'active' or sc['wave_resolution'] ~= cjson.null then
    return {'CONTINUATION_NOT_ACTIVE', tostring(sc['state'])}
end

local cand_outcome = decode_object(child_outcome_json)
local cand_status = decode_object(child_status_json)
local cand_marker = decode_object(marker_json)
local cand_parent_st = decode_object(parent_status_json)
local cand_hint = decode_object(hint_json)
local pre_cont_sat = decode_object(pre_cont_sat_json)
local pre_cont_unsat = decode_object(pre_cont_unsat_json)
if not cand_outcome or cand_outcome['task_id'] ~= child_task_id or cand_outcome['terminal_status'] ~= 'claim_timed_out' or
   type(cand_outcome['outcome_id']) ~= 'string' or cand_outcome['outcome_id'] == '' or
   not cand_status or cand_status['task_id'] ~= child_task_id or cand_status['status'] ~= 'claim_timed_out' or
   not cand_marker or cand_marker['child_task_id'] ~= child_task_id or cand_marker['continuation_version'] ~= 2 or
   not cand_parent_st or cand_parent_st['task_id'] ~= parent_task_id or cand_parent_st['status'] ~= 'resumable' or
   not cand_hint or cand_hint['task_id'] ~= parent_task_id or cand_hint['expected_continuation_version'] ~= 2 or
   not pre_cont_sat or pre_cont_sat['state'] ~= 'released' or pre_cont_sat['wave_resolution'] ~= 'SATISFIED' or
   not pre_cont_unsat or pre_cont_unsat['state'] ~= 'released' or pre_cont_unsat['wave_resolution'] ~= 'UNSATISFIABLE' then
    return {'CORRUPT_INPUT'}
end

local policy = sw['barrier_policy']
local frac_num = nil
local frac_den = nil
local derived_req_count = nil
if policy == 'WAIT_ALL' then
    if sw['required_success_fraction'] ~= cjson.null or sw['required_success_count'] ~= cjson.null then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
elseif policy == 'PARTIAL' then
    local rf = sw['required_success_fraction']
    if type(rf) ~= 'table' or type(rf['numerator']) ~= 'number' or type(rf['denominator']) ~= 'number' then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
    frac_num = rf['numerator']
    frac_den = rf['denominator']
    if math.floor(frac_num) ~= frac_num or math.floor(frac_den) ~= frac_den or
       frac_num < 1 or frac_den < 1 or frac_num > frac_den or frac_den > 10000 then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
    derived_req_count = math.floor((child_count * frac_num + frac_den - 1) / frac_den)
    if derived_req_count < 1 or derived_req_count > child_count or sw['required_success_count'] ~= derived_req_count then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
else
    return {'CORRUPT_WAVE_PROJECTION'}
end

local slot_ids = {s1_id, s2_id}
if child_count >= 3 then slot_ids[3] = s3_id end
if child_count >= 4 then slot_ids[4] = s4_id end
local trig_found = false
for idx = 1, child_count do
    local sid = slot_ids[idx]
    if type(sid) ~= 'string' or sid == '' or
       sw['required_child_task_ids'][idx] ~= sid or
       sc['required_child_task_ids'][idx] ~= sid then
        return {'WAVE_CONSISTENCY_MISMATCH'}
    end
    if sid == child_task_id then trig_found = true end
end
if not trig_found then return {'CHILD_NOT_MEMBER'} end

local trig_outcome_id = cand_outcome['outcome_id']
local e1, st1, oid1, fn1, hc1 = validate_sibling_readonly(s1_id, s1_status_key, s1_claim_key, s1_fence_key, s1_outcome_key, s1_result_key, s1_lineage_key, s1_cap_key, s1_input_key, s1_marker_key, s1_expiry_key, trig_outcome_id)
if e1 then return {e1} end
local e2, st2, oid2, fn2, hc2 = validate_sibling_readonly(s2_id, s2_status_key, s2_claim_key, s2_fence_key, s2_outcome_key, s2_result_key, s2_lineage_key, s2_cap_key, s2_input_key, s2_marker_key, s2_expiry_key, trig_outcome_id)
if e2 then return {e2} end
local st3, oid3, fn3, hc3 = nil, nil, 0, false
if child_count >= 3 then
    local e3
    e3, st3, oid3, fn3, hc3 = validate_sibling_readonly(s3_id, s3_status_key, s3_claim_key, s3_fence_key, s3_outcome_key, s3_result_key, s3_lineage_key, s3_cap_key, s3_input_key, s3_marker_key, s3_expiry_key, trig_outcome_id)
    if e3 then return {e3} end
end
local st4, oid4, fn4, hc4 = nil, nil, 0, false
if child_count >= 4 then
    local e4
    e4, st4, oid4, fn4, hc4 = validate_sibling_readonly(s4_id, s4_status_key, s4_claim_key, s4_fence_key, s4_outcome_key, s4_result_key, s4_lineage_key, s4_cap_key, s4_input_key, s4_marker_key, s4_expiry_key, trig_outcome_id)
    if e4 then return {e4} end
end

local slot_statuses = {st1, st2}
local slot_outcomes = {oid1, oid2}
local slot_fences = {fn1, fn2}
local slot_has_claim = {hc1, hc2}
if child_count >= 3 then
    slot_statuses[3] = st3; slot_outcomes[3] = oid3; slot_fences[3] = fn3; slot_has_claim[3] = hc3
end
if child_count >= 4 then
    slot_statuses[4] = st4; slot_outcomes[4] = oid4; slot_fences[4] = fn4; slot_has_claim[4] = hc4
end

local completed_count = 0
local terminal_count = 0
local unresolved_count = 0
local potentially_successful_count = 0
for idx = 1, child_count do
    local st_val = slot_statuses[idx]
    if st_val == 'completed' then
        completed_count = completed_count + 1
        terminal_count = terminal_count + 1
    elseif st_val == 'failed' or st_val == 'claim_timed_out' then
        terminal_count = terminal_count + 1
    elseif st_val == 'pending' or st_val == 'working' or st_val == 'recovery_required' then
        unresolved_count = unresolved_count + 1
        potentially_successful_count = potentially_successful_count + 1
    else
        return {'CORRUPT_CHILD_STATE'}
    end
end

local wave_resolution = 'UNRESOLVED'
if policy == 'WAIT_ALL' then
    if terminal_count == child_count then
        wave_resolution = 'SATISFIED'
    end
else
    if completed_count >= derived_req_count then
        wave_resolution = 'SATISFIED'
    elseif completed_count + potentially_successful_count < derived_req_count then
        wave_resolution = 'UNSATISFIABLE'
    end
end

local resolve_now = (wave_resolution ~= 'UNRESOLVED')
local resolved_wave_json = ''
local released_cont_json = ''

local s1_close, s1_del_claim, s1_inc_fence, s1_next_fence, s1_res_status_json = false, false, false, '', ''
local s2_close, s2_del_claim, s2_inc_fence, s2_next_fence, s2_res_status_json = false, false, false, '', ''
local s3_close, s3_del_claim, s3_inc_fence, s3_next_fence, s3_res_status_json = false, false, false, '', ''
local s4_close, s4_del_claim, s4_inc_fence, s4_next_fence, s4_res_status_json = false, false, false, '', ''

if resolve_now then
    if wave_resolution == 'SATISFIED' then
        released_cont_json = pre_cont_sat_json
    else
        released_cont_json = pre_cont_unsat_json
    end

    local child_outcomes_tbl = {}
    local residual_ids_tbl = {}
    local close_flags = {false, false, false, false}
    local del_claim_flags = {false, false, false, false}
    local inc_fence_flags = {false, false, false, false}
    local next_fences = {'', '', '', ''}
    local res_status_jsons = {'', '', '', ''}

    for idx = 1, child_count do
        local sid = slot_ids[idx]
        local st_val = slot_statuses[idx]
        if st_val == 'completed' or st_val == 'failed' or st_val == 'claim_timed_out' then
            child_outcomes_tbl[sid] = {
                outcome_id = slot_outcomes[idx],
                status = st_val,
            }
        else
            child_outcomes_tbl[sid] = {
                outcome_id = cjson.null,
                status = 'ended / barrier_resolved',
            }
            table.insert(residual_ids_tbl, sid)
            close_flags[idx] = true
            del_claim_flags[idx] = slot_has_claim[idx]
            if slot_fences[idx] > 0 or slot_has_claim[idx] then
                inc_fence_flags[idx] = true
                next_fences[idx] = tostring(slot_fences[idx] + 1)
            end
            res_status_jsons[idx] = cjson.encode({
                status = 'ended / barrier_resolved',
                task_id = sid,
                updated_at = now_iso,
                wave_id = wave_id,
            })
        end
    end

    s1_close, s1_del_claim, s1_inc_fence, s1_next_fence, s1_res_status_json = close_flags[1], del_claim_flags[1], inc_fence_flags[1], next_fences[1], res_status_jsons[1]
    s2_close, s2_del_claim, s2_inc_fence, s2_next_fence, s2_res_status_json = close_flags[2], del_claim_flags[2], inc_fence_flags[2], next_fences[2], res_status_jsons[2]
    s3_close, s3_del_claim, s3_inc_fence, s3_next_fence, s3_res_status_json = close_flags[3], del_claim_flags[3], inc_fence_flags[3], next_fences[3], res_status_jsons[3]
    s4_close, s4_del_claim, s4_inc_fence, s4_next_fence, s4_res_status_json = close_flags[4], del_claim_flags[4], inc_fence_flags[4], next_fences[4], res_status_jsons[4]

    local req_count_val = cjson.null
    local req_frac_val = cjson.null
    if policy == 'PARTIAL' then
        req_count_val = derived_req_count
        req_frac_val = {
            denominator = frac_den,
            numerator = frac_num,
        }
    end

    local rw_residual_val = residual_ids_tbl
    if #residual_ids_tbl == 0 then
        rw_residual_val = '__TAL_EMPTY_ARRAY_SENTINEL__'
    end

    local rw_tbl = {
        accepted_at = sw['accepted_at'],
        accepted_wave_fingerprint = accepted_wave_fp,
        barrier_policy = policy,
        child_outcomes = child_outcomes_tbl,
        completed_count = completed_count,
        deadline_policy_id = dd['policy_id'],
        deadline_policy_version = dd['policy_version'],
        parent_task_id = parent_task_id,
        required_child_task_ids = sw['required_child_task_ids'],
        required_success_count = req_count_val,
        required_success_fraction = req_frac_val,
        residual_child_task_ids = rw_residual_val,
        residual_disposition = 'ended / barrier_resolved',
        resolution = wave_resolution,
        resolved_at = now_iso,
        terminal_count = terminal_count,
        triggering_assignment_id = cjson.null,
        triggering_child_task_id = child_task_id,
        triggering_fencing_epoch = cjson.null,
        unresolved_count = unresolved_count,
        version = 2,
        wave_deadline = sw['wave_deadline'],
        wave_id = wave_id,
    }
    local ok_rw_enc, encoded_rw = pcall(cjson.encode, rw_tbl)
    if not ok_rw_enc or type(encoded_rw) ~= 'string' then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
    if #residual_ids_tbl == 0 then
        encoded_rw = string.gsub(encoded_rw, '"residual_child_task_ids":"__TAL_EMPTY_ARRAY_SENTINEL__"', '"residual_child_task_ids":[]', 1)
    end
    if #encoded_rw > 16384 then
        return {'CORRUPT_WAVE_PROJECTION'}
    end
    resolved_wave_json = encoded_rw
end

-- =============================================================================
-- PHASE 2: Prevalidated Atomic Writes Only
-- =============================================================================
redis.call('SET', child_outcome_key, child_outcome_json)
redis.call('SET', child_status_key, child_status_json)
redis.call('SET', child_marker_key, marker_json)

if resolve_now then
    redis.call('SET', resolved_wave_key, resolved_wave_json)
    redis.call('SET', continuation_key, released_cont_json)
    redis.call('SET', parent_status_key, parent_status_json)
    redis.call('RPUSH', hint_queue_key, hint_json)

    if s1_close then
        redis.call('SET', s1_status_key, s1_res_status_json)
        if s1_del_claim then redis.call('DEL', s1_claim_key) end
        if s1_inc_fence then redis.call('SET', s1_fence_key, s1_next_fence) end
    end
    if s2_close then
        redis.call('SET', s2_status_key, s2_res_status_json)
        if s2_del_claim then redis.call('DEL', s2_claim_key) end
        if s2_inc_fence then redis.call('SET', s2_fence_key, s2_next_fence) end
    end
    if child_count >= 3 and s3_close then
        redis.call('SET', s3_status_key, s3_res_status_json)
        if s3_del_claim then redis.call('DEL', s3_claim_key) end
        if s3_inc_fence then redis.call('SET', s3_fence_key, s3_next_fence) end
    end
    if child_count >= 4 and s4_close then
        redis.call('SET', s4_status_key, s4_res_status_json)
        if s4_del_claim then redis.call('DEL', s4_claim_key) end
        if s4_inc_fence then redis.call('SET', s4_fence_key, s4_next_fence) end
    end
end

return {'OK', child_outcome_json}
"""

TRANSITION_WAVE_CHILD_EXPIRED_CLAIM_LUA = """
local status_key = KEYS[1]
local claim_key = KEYS[2]
local fence_key = KEYS[3]
local expiry_key = KEYS[4]
local outcome_key = KEYS[5]
local result_key = KEYS[6]
local events_key = KEYS[7]
local lineage_key = KEYS[8]
local marker_key = KEYS[9]
local parent_status_key = KEYS[10]
local parent_children_key = KEYS[11]
local continuation_key = KEYS[12]
local continuation_latest_key = KEYS[13]
local wave_key = KEYS[14]
local resolved_wave_key = KEYS[15]

local task_id = ARGV[1]
local parent_task_id = ARGV[2]
local wave_id = ARGV[3]
local now_iso = ARGV[4]
local expiry_record_json = ARGV[5]
local recovery_status_json = ARGV[6]
local stream_event_json = ARGV[7]

local function key_type(key)
    local t = redis.call('TYPE', key)
    if type(t) == 'table' then return t['ok'] end
    return tostring(t)
end

local function decode_object(raw)
    local ok, tbl = pcall(cjson.decode, raw)
    if not ok or type(tbl) ~= 'table' then return nil end
    return tbl
end

-- =============================================================================
-- PHASE 1: Reads and Validations (Strictly Read-Only, Zero Mutations)
-- =============================================================================

for _, k in ipairs({status_key, claim_key, fence_key, expiry_key, outcome_key, result_key, lineage_key, marker_key, parent_status_key, continuation_key, continuation_latest_key, wave_key, resolved_wave_key}) do
    local t = key_type(k)
    if t ~= 'none' and t ~= 'string' then
        return {'WRONG_KEY_TYPE_' .. k, t}
    end
end
local pc_t = key_type(parent_children_key)
if pc_t ~= 'none' and pc_t ~= 'set' then
    return {'WRONG_KEY_TYPE_' .. parent_children_key, pc_t}
end
local ev_type = key_type(events_key)
if ev_type ~= 'none' and ev_type ~= 'stream' then
    return {'WRONG_KEY_TYPE_' .. events_key, ev_type}
end

if redis.call('EXISTS', status_key) == 0 then return {'NOT_FOUND'} end
local status_tbl = decode_object(redis.call('GET', status_key))
if not status_tbl or status_tbl['task_id'] ~= task_id or type(status_tbl['status']) ~= 'string' then
    return {'CORRUPT_STATUS'}
end
local cur_status = status_tbl['status']

if redis.call('EXISTS', lineage_key) == 0 then return {'LINEAGE_NOT_FOUND'} end
local lin_tbl = decode_object(redis.call('GET', lineage_key))
if not lin_tbl or lin_tbl['task_id'] ~= task_id or lin_tbl['parent_task_id'] ~= parent_task_id or lin_tbl['root_task_id'] ~= parent_task_id then
    return {'CHILD_LINEAGE_MISMATCH'}
end

if redis.call('EXISTS', parent_children_key) == 0 or redis.call('SISMEMBER', parent_children_key, task_id) ~= 1 then
    return {'CHILD_NOT_MEMBER'}
end

if redis.call('EXISTS', continuation_latest_key) == 0 or redis.call('GET', continuation_latest_key) ~= '2' then
    return {'CONTINUATION_LATEST_MISMATCH'}
end

if redis.call('EXISTS', continuation_key) == 0 then return {'MISSING_CONTINUATION'} end
local cont_tbl = decode_object(redis.call('GET', continuation_key))
if not cont_tbl or cont_tbl['version'] ~= 2 or cont_tbl['continuation_version'] ~= 2 or
   cont_tbl['parent_task_id'] ~= parent_task_id or cont_tbl['wave_id'] ~= wave_id then
    return {'CORRUPT_CONTINUATION'}
end

if redis.call('EXISTS', wave_key) == 0 then return {'MISSING_ACCEPTED_WAVE'} end
local wave_tbl = decode_object(redis.call('GET', wave_key))
if not wave_tbl or wave_tbl['version'] ~= 1 or wave_tbl['wave_id'] ~= wave_id or wave_tbl['parent_task_id'] ~= parent_task_id or
   wave_tbl['state'] ~= 'unresolved' then
    return {'CORRUPT_WAVE_PROJECTION'}
end

if redis.call('EXISTS', parent_status_key) == 0 then return {'PARENT_NOT_FOUND'} end
local ps_tbl = decode_object(redis.call('GET', parent_status_key))
if not ps_tbl or ps_tbl['task_id'] ~= parent_task_id or type(ps_tbl['status']) ~= 'string' then
    return {'CORRUPT_PARENT_STATUS'}
end

if redis.call('EXISTS', resolved_wave_key) == 1 then
    return {'WAVE_ALREADY_RESOLVED'}
end
if ps_tbl['status'] ~= 'waiting_for_children' then
    return {'PARENT_NOT_WAITING', tostring(ps_tbl['status'])}
end
if cont_tbl['state'] ~= 'active' or cont_tbl['wave_resolution'] ~= cjson.null then
    return {'CONTINUATION_NOT_ACTIVE', tostring(cont_tbl['state'])}
end

if cur_status == 'recovery_required' then
    if redis.call('EXISTS', claim_key) == 1 then return {'ACTIVE_CLAIM_EXISTS'} end
    if redis.call('EXISTS', outcome_key) == 1 or redis.call('EXISTS', result_key) == 1 or redis.call('EXISTS', marker_key) == 1 then
        return {'ALREADY_COMPLETED'}
    end
    if redis.call('EXISTS', expiry_key) == 1 then
        local existing_expiry = redis.call('GET', expiry_key)
        local exp_tbl = decode_object(existing_expiry)
        if not exp_tbl or exp_tbl['task_id'] ~= task_id or exp_tbl['execution_kind'] ~= 'delegated_child' then
            return {'CORRUPT_EXPIRY_RECORD'}
        end
        return {'IDEMPOTENT', existing_expiry}
    end
    return {'EXPIRY_RECORD_MISSING'}
end

if cur_status ~= 'working' then
    return {'NOT_WORKING', cur_status}
end

if redis.call('EXISTS', outcome_key) == 1 or redis.call('EXISTS', result_key) == 1 or redis.call('EXISTS', marker_key) == 1 then
    return {'ALREADY_COMPLETED'}
end

if redis.call('EXISTS', claim_key) == 0 then return {'CLAIM_NOT_FOUND'} end
local claim_tbl = decode_object(redis.call('GET', claim_key))
if not claim_tbl or claim_tbl['task_id'] ~= task_id then return {'CORRUPT_CLAIM'} end
if claim_tbl['state'] ~= 'acknowledged' then
    return {'CLAIM_NOT_ACKNOWLEDGED', tostring(claim_tbl['state'])}
end

local exec_expires = claim_tbl['execution_expires_at']
if not exec_expires or type(exec_expires) ~= 'string' or exec_expires == '' then
    return {'MISSING_EXECUTION_EXPIRES_AT'}
end
if now_iso <= exec_expires then
    return {'EXECUTION_NOT_EXPIRED', exec_expires}
end

if redis.call('EXISTS', fence_key) == 0 then return {'FENCE_NOT_FOUND'} end
local cur_fence = tonumber(redis.call('GET', fence_key))
local claim_epoch = tonumber(claim_tbl['fencing_epoch'])
if not cur_fence or not claim_epoch or cur_fence ~= claim_epoch then
    return {'FENCE_MISMATCH', tostring(cur_fence), tostring(claim_epoch)}
end

if claim_tbl['execution_kind'] ~= 'delegated_child' or claim_tbl['prior_claimable_status'] ~= 'pending' then
    return {'CORRUPT_PRIOR_STATUS', tostring(claim_tbl['prior_claimable_status'])}
end

local cand_exp = decode_object(expiry_record_json)
if not cand_exp or cand_exp['task_id'] ~= task_id or cand_exp['claim_id'] ~= claim_tbl['claim_id'] or
   cand_exp['assignment_id'] ~= claim_tbl['assignment_id'] or cand_exp['principal_id'] ~= claim_tbl['runtime_principal_id'] or
   tonumber(cand_exp['expired_fencing_epoch']) ~= cur_fence then
    return {'CORRUPT_EXPIRY_RECORD'}
end

-- =============================================================================
-- PHASE 2: State Commit
-- =============================================================================
redis.call('DEL', claim_key)
redis.call('SET', expiry_key, expiry_record_json)
redis.call('SET', status_key, recovery_status_json)

if stream_event_json and stream_event_json ~= '' then
    local ev_tbl = decode_object(stream_event_json)
    if ev_tbl then
        local args = {'XADD', events_key, 'MAXLEN', '~', '500', '*'}
        for k, v in pairs(ev_tbl) do
            table.insert(args, tostring(k))
            table.insert(args, tostring(v))
        end
        redis.call(unpack(args))
    end
end

return {'OK', expiry_record_json}
"""

RESOLVE_WAVE_CHILD_EXPIRED_CLAIM_LUA = """
local status_key = KEYS[1]
local claim_key = KEYS[2]
local fence_key = KEYS[3]
local expiry_key = KEYS[4]
local decision_key = KEYS[5]
local outcome_key = KEYS[6]
local lineage_key = KEYS[7]
local marker_key = KEYS[8]
local parent_status_key = KEYS[9]
local parent_children_key = KEYS[10]
local continuation_key = KEYS[11]
local continuation_latest_key = KEYS[12]
local wave_key = KEYS[13]
local resolved_wave_key = KEYS[14]

local task_id = ARGV[1]
local parent_task_id = ARGV[2]
local wave_id = ARGV[3]
local recovery_decision_id = ARGV[4]
local expired_claim_id = ARGV[5]
local expired_assignment_id = ARGV[6]
local expired_epoch = tonumber(ARGV[7])
local disposition = ARGV[8]
local decision_record_json = ARGV[9]
local decision_fingerprint = ARGV[10]
local restored_status_json = ARGV[11]

local function key_type(key)
    local t = redis.call('TYPE', key)
    if type(t) == 'table' then return t['ok'] end
    return tostring(t)
end

local function decode_object(raw)
    local ok, tbl = pcall(cjson.decode, raw)
    if not ok or type(tbl) ~= 'table' then return nil end
    return tbl
end

-- =============================================================================
-- PHASE 1: Reads and Validations (Strictly Read-Only, Zero Mutations)
-- =============================================================================

for _, k in ipairs({status_key, claim_key, fence_key, expiry_key, decision_key, outcome_key, lineage_key, marker_key, parent_status_key, continuation_key, continuation_latest_key, wave_key, resolved_wave_key}) do
    local t = key_type(k)
    if t ~= 'none' and t ~= 'string' then
        return {'WRONG_KEY_TYPE_' .. k, t}
    end
end
local pc_t = key_type(parent_children_key)
if pc_t ~= 'none' and pc_t ~= 'set' then
    return {'WRONG_KEY_TYPE_' .. parent_children_key, pc_t}
end

if redis.call('EXISTS', lineage_key) == 0 then return {'LINEAGE_NOT_FOUND'} end
local lin_tbl = decode_object(redis.call('GET', lineage_key))
if not lin_tbl or lin_tbl['task_id'] ~= task_id or lin_tbl['parent_task_id'] ~= parent_task_id or lin_tbl['root_task_id'] ~= parent_task_id then
    return {'CHILD_LINEAGE_MISMATCH'}
end
if redis.call('EXISTS', parent_children_key) == 0 or redis.call('SISMEMBER', parent_children_key, task_id) ~= 1 then
    return {'CHILD_NOT_MEMBER'}
end
if redis.call('EXISTS', continuation_latest_key) == 0 or redis.call('GET', continuation_latest_key) ~= '2' then
    return {'CONTINUATION_LATEST_MISMATCH'}
end
if redis.call('EXISTS', continuation_key) == 0 then return {'MISSING_CONTINUATION'} end
local cont_tbl = decode_object(redis.call('GET', continuation_key))
if not cont_tbl or cont_tbl['version'] ~= 2 or cont_tbl['continuation_version'] ~= 2 or
   cont_tbl['parent_task_id'] ~= parent_task_id or cont_tbl['wave_id'] ~= wave_id then
    return {'CORRUPT_CONTINUATION'}
end
if redis.call('EXISTS', wave_key) == 0 then return {'MISSING_ACCEPTED_WAVE'} end
local wave_tbl = decode_object(redis.call('GET', wave_key))
if not wave_tbl or wave_tbl['version'] ~= 1 or wave_tbl['wave_id'] ~= wave_id or wave_tbl['parent_task_id'] ~= parent_task_id or
   wave_tbl['state'] ~= 'unresolved' then
    return {'CORRUPT_WAVE_PROJECTION'}
end
if redis.call('EXISTS', parent_status_key) == 0 then return {'PARENT_NOT_FOUND'} end
local ps_tbl = decode_object(redis.call('GET', parent_status_key))
if not ps_tbl or ps_tbl['task_id'] ~= parent_task_id or type(ps_tbl['status']) ~= 'string' then
    return {'CORRUPT_PARENT_STATUS'}
end

if redis.call('EXISTS', expiry_key) == 0 then return {'EXPIRY_RECORD_NOT_FOUND'} end
local expiry_tbl = decode_object(redis.call('GET', expiry_key))
if not expiry_tbl then return {'CORRUPT_EXPIRY_RECORD'} end
if expiry_tbl['task_id'] ~= task_id or
   expiry_tbl['claim_id'] ~= expired_claim_id or
   expiry_tbl['assignment_id'] ~= expired_assignment_id or
   tonumber(expiry_tbl['expired_fencing_epoch']) ~= expired_epoch or
   expiry_tbl['execution_kind'] ~= 'delegated_child' or
   expiry_tbl['prior_claimable_status'] ~= 'pending' then
    return {'EXPIRY_RECORD_MISMATCH'}
end

-- Check idempotent replay of existing recovery decision
if redis.call('EXISTS', decision_key) == 1 then
    local existing_dec_raw = redis.call('GET', decision_key)
    local existing_dec = decode_object(existing_dec_raw)
    if existing_dec then
        if existing_dec['semantic_fingerprint'] == decision_fingerprint and
           existing_dec['task_id'] == task_id and
           existing_dec['recovery_decision_id'] == recovery_decision_id and
           existing_dec['restored_status'] == 'pending' then
            return {'IDEMPOTENT', existing_dec_raw, existing_dec['restored_status']}
        end
        return {'DECISION_CONFLICT'}
    end
    return {'CORRUPT_DECISION'}
end

-- First resolution requires active unresolved wave & recovery_required child
if redis.call('EXISTS', resolved_wave_key) == 1 then
    return {'WAVE_ALREADY_RESOLVED'}
end
if ps_tbl['status'] ~= 'waiting_for_children' then
    return {'PARENT_NOT_WAITING', tostring(ps_tbl['status'])}
end
if cont_tbl['state'] ~= 'active' or cont_tbl['wave_resolution'] ~= cjson.null then
    return {'CONTINUATION_NOT_ACTIVE', tostring(cont_tbl['state'])}
end

if redis.call('EXISTS', status_key) == 0 then return {'NOT_FOUND'} end
local status_tbl = decode_object(redis.call('GET', status_key))
if not status_tbl or status_tbl['task_id'] ~= task_id or type(status_tbl['status']) ~= 'string' then
    return {'CORRUPT_STATUS'}
end
if status_tbl['status'] ~= 'recovery_required' then
    return {'NOT_RECOVERY_REQUIRED', status_tbl['status']}
end

if redis.call('EXISTS', outcome_key) == 1 or redis.call('EXISTS', marker_key) == 1 then
    return {'ALREADY_COMPLETED'}
end
if redis.call('EXISTS', claim_key) == 1 then
    return {'ACTIVE_CLAIM_EXISTS'}
end
if disposition ~= 'restore_claimable' then
    return {'UNSUPPORTED_DISPOSITION', disposition}
end
if redis.call('EXISTS', fence_key) == 0 then return {'FENCE_NOT_FOUND'} end
local cur_fence = tonumber(redis.call('GET', fence_key))
if not cur_fence or cur_fence ~= expired_epoch then
    return {'FENCE_MISMATCH', tostring(cur_fence), tostring(expired_epoch)}
end

-- =============================================================================
-- PHASE 2: State Commit
-- =============================================================================
redis.call('SET', decision_key, decision_record_json)
redis.call('SET', status_key, restored_status_json)

return {'OK', decision_record_json, 'pending'}
"""

READ_RESUMED_CONTINUATION_V2_LUA = """
local parent_status_key = KEYS[1]
local claim_key = KEYS[2]
local fence_key = KEYS[3]
local lineage_key = KEYS[4]
local continuation_latest_key = KEYS[5]
local continuation_key = KEYS[6]
local wave_key = KEYS[7]
local resolved_wave_key = KEYS[8]
local task_input_key = KEYS[9]
local stage_artifact_key = KEYS[10]

local parent_task_id = ARGV[1]
local expected_assignment_id = ARGV[2]
local expected_principal_id = ARGV[3]
local expected_bound_agent_id = ARGV[4]
local expected_fencing_epoch = tonumber(ARGV[5])
local expected_wave_id = ARGV[6]
local expected_input_version = tonumber(ARGV[7])
local expected_artifact_id = ARGV[8]
local expected_artifact_version = tonumber(ARGV[9])

local function key_type(key)
    local t = redis.call('TYPE', key)
    if type(t) == 'table' then return t['ok'] end
    return tostring(t)
end

local function decode_object(raw)
    local ok, tbl = pcall(cjson.decode, raw)
    if not ok or type(tbl) ~= 'table' then return nil end
    return tbl
end

local function check_str_key(k)
    local kt = key_type(k)
    if kt ~= 'none' and kt ~= 'string' then
        return 'WRONG_KEY_TYPE_' .. k
    end
    return nil
end

local kt_err = check_str_key(parent_status_key) or check_str_key(claim_key) or
               check_str_key(fence_key) or check_str_key(lineage_key) or
               check_str_key(continuation_latest_key) or check_str_key(continuation_key) or
               check_str_key(wave_key) or check_str_key(resolved_wave_key) or
               check_str_key(task_input_key) or check_str_key(stage_artifact_key)
if kt_err then
    return {kt_err}
end

if redis.call('EXISTS', claim_key) == 0 then return {'CLAIM_NOT_FOUND'} end
local claim_tbl = decode_object(redis.call('GET', claim_key))
if not claim_tbl or claim_tbl['task_id'] ~= parent_task_id or
   claim_tbl['assignment_id'] ~= expected_assignment_id or
   claim_tbl['runtime_principal_id'] ~= expected_principal_id or
   claim_tbl['bound_agent_id'] ~= expected_bound_agent_id then
    return {'CLAIM_NOT_FOUND'}
end
if claim_tbl['assigned_capability'] ~= 'research' then
    return {'CAPABILITY_MISMATCH'}
end
if claim_tbl['state'] ~= 'acknowledged' then
    return {'CLAIM_NOT_ACKNOWLEDGED'}
end
if claim_tbl['execution_kind'] ~= 'resumed_continuation' then
    return {'EXECUTION_KIND_MISMATCH'}
end
if claim_tbl['continuation_version'] ~= 2 then
    return {'CONTINUATION_VERSION_MISMATCH'}
end
local claim_fence = tonumber(claim_tbl['fencing_epoch'])
if not expected_fencing_epoch or not claim_fence or claim_fence < 1 or claim_fence ~= expected_fencing_epoch then
    return {'FENCE_MISMATCH'}
end

if redis.call('EXISTS', fence_key) == 0 then return {'FENCE_MISMATCH'} end
local cur_fence = tonumber(redis.call('GET', fence_key))
if not cur_fence or cur_fence ~= expected_fencing_epoch then
    return {'FENCE_MISMATCH'}
end

if redis.call('EXISTS', parent_status_key) == 0 then return {'CORRUPT_PARENT_STATUS'} end
local ps_tbl = decode_object(redis.call('GET', parent_status_key))
if not ps_tbl or ps_tbl['task_id'] ~= parent_task_id or type(ps_tbl['status']) ~= 'string' then
    return {'CORRUPT_PARENT_STATUS'}
end
if ps_tbl['status'] ~= 'working' then
    return {'PARENT_NOT_WORKING', ps_tbl['status']}
end

if redis.call('EXISTS', lineage_key) == 0 then return {'CORRUPT_LINEAGE'} end
local lin_tbl = decode_object(redis.call('GET', lineage_key))
if not lin_tbl or lin_tbl['task_id'] ~= parent_task_id or lin_tbl['parent_task_id'] ~= cjson.null or
   type(lin_tbl['root_task_id']) ~= 'string' or lin_tbl['root_task_id'] == '' or
   lin_tbl['root_task_id'] ~= claim_tbl['root_task_id'] then
    return {'CORRUPT_LINEAGE'}
end

if redis.call('EXISTS', continuation_latest_key) == 0 then return {'CONTINUATION_LATEST_MISMATCH'} end
if redis.call('GET', continuation_latest_key) ~= '2' then return {'CONTINUATION_LATEST_MISMATCH'} end

if redis.call('EXISTS', continuation_key) == 0 then return {'CORRUPT_CONTINUATION'} end
local cont_raw = redis.call('GET', continuation_key)
local cont_tbl = decode_object(cont_raw)
if not cont_tbl or cont_tbl['version'] ~= 2 or cont_tbl['continuation_version'] ~= 2 or
   cont_tbl['parent_task_id'] ~= parent_task_id or
   cont_tbl['stage'] ~= 'incorporate_analysis_child' or
   cont_tbl['continuation_capability'] ~= 'research' or
   type(cont_tbl['wave_id']) ~= 'string' or cont_tbl['wave_id'] == '' or
   type(cont_tbl['input_version']) ~= 'number' or cont_tbl['input_version'] < 1 or
   type(cont_tbl['required_child_task_ids']) ~= 'table' or #cont_tbl['required_child_task_ids'] < 1 then
    return {'CORRUPT_CONTINUATION'}
end
if cont_tbl['wave_id'] ~= expected_wave_id or cont_tbl['input_version'] ~= expected_input_version then
    return {'CORRUPT_CONTINUATION'}
end
local cont_state = cont_tbl['state']
if cont_state == 'closed' then
    return {'CONTINUATION_CLOSED'}
end
if cont_state ~= 'released' then
    return {'CONTINUATION_NOT_RELEASED', tostring(cont_state)}
end
local cont_res = cont_tbl['wave_resolution']
if cont_res ~= 'SATISFIED' and cont_res ~= 'UNSATISFIABLE' and cont_res ~= 'TIMED_OUT' then
    return {'UNSUPPORTED_WAVE_RESOLUTION', tostring(cont_res)}
end
if type(cont_tbl['artifact_references']) ~= 'table' or #cont_tbl['artifact_references'] < 1 then
    return {'CORRUPT_CONTINUATION'}
end
local match_count = 0
for _, ref in ipairs(cont_tbl['artifact_references']) do
    if type(ref) == 'table' and ref['producing_task_id'] == parent_task_id and
       ref['artifact_id'] == expected_artifact_id and ref['artifact_version'] == expected_artifact_version then
        match_count = match_count + 1
    end
end
if match_count ~= 1 then
    return {'CORRUPT_CONTINUATION'}
end

if redis.call('EXISTS', wave_key) == 0 then return {'CORRUPT_WAVE_PROJECTION'} end
local wave_tbl = decode_object(redis.call('GET', wave_key))
if not wave_tbl or wave_tbl['version'] ~= 1 or wave_tbl['wave_id'] ~= cont_tbl['wave_id'] or
   wave_tbl['parent_task_id'] ~= parent_task_id or wave_tbl['state'] ~= 'unresolved' or
   type(wave_tbl['required_child_task_ids']) ~= 'table' or
   #wave_tbl['required_child_task_ids'] ~= #cont_tbl['required_child_task_ids'] then
    return {'CORRUPT_WAVE_PROJECTION'}
end

if redis.call('EXISTS', resolved_wave_key) == 0 then return {'CORRUPT_RESOLVED_WAVE'} end
local rw_raw = redis.call('GET', resolved_wave_key)
local rw_tbl = decode_object(rw_raw)
if not rw_tbl or rw_tbl['version'] ~= 2 or rw_tbl['wave_id'] ~= cont_tbl['wave_id'] or
   rw_tbl['parent_task_id'] ~= parent_task_id or rw_tbl['resolution'] ~= cont_res or
   rw_tbl['barrier_policy'] ~= wave_tbl['barrier_policy'] or
   rw_tbl['accepted_at'] ~= wave_tbl['accepted_at'] or
   rw_tbl['wave_deadline'] ~= wave_tbl['wave_deadline'] or
   type(rw_tbl['accepted_wave_fingerprint']) ~= 'string' or rw_tbl['accepted_wave_fingerprint'] == '' or
   type(rw_tbl['required_child_task_ids']) ~= 'table' or
   #rw_tbl['required_child_task_ids'] ~= #cont_tbl['required_child_task_ids'] then
    return {'CORRUPT_RESOLVED_WAVE'}
end
for idx = 1, #cont_tbl['required_child_task_ids'] do
    local cid = cont_tbl['required_child_task_ids'][idx]
    if wave_tbl['required_child_task_ids'][idx] ~= cid or rw_tbl['required_child_task_ids'][idx] ~= cid then
        return {'CORRUPT_RESOLVED_WAVE'}
    end
end

if redis.call('EXISTS', task_input_key) == 0 then return {'CORRUPT_PARENT_INPUT'} end
local input_raw = redis.call('GET', task_input_key)
local input_tbl = decode_object(input_raw)
if not input_tbl or input_tbl['task_id'] ~= parent_task_id or input_tbl['input_version'] ~= cont_tbl['input_version'] then
    return {'CORRUPT_PARENT_INPUT'}
end

if redis.call('EXISTS', stage_artifact_key) == 0 then return {'CORRUPT_STAGE_ARTIFACT'} end
local art_raw = redis.call('GET', stage_artifact_key)
local art_tbl = decode_object(art_raw)
if not art_tbl or art_tbl['producing_task_id'] ~= parent_task_id or
   art_tbl['artifact_id'] ~= expected_artifact_id or
   art_tbl['artifact_version'] ~= expected_artifact_version or
   art_tbl['sealed'] ~= true then
    return {'CORRUPT_STAGE_ARTIFACT'}
end

return {'OK', lin_tbl['root_task_id'], input_raw, cont_raw, rw_raw, art_raw}
"""


class RedisGateway:
    def __init__(
        self,
        url: str,
        delivery_mode: str = "legacy_push",
        *,
        connection_config: RedisConnectionConfig | None = None,
        username: str | None = None,
        password: str | None = None,
        allow_unauthenticated_local_test: bool = False,
        capability_queue_contract: Mapping[str, str] | None = None,
        capability_hint_contract: Mapping[str, str] | None = None,
    ) -> None:
        resolved_url, client_kwargs = resolve_client_connection_kwargs(
            url,
            connection_config=connection_config,
            username=username,
            password=password,
            allow_unauthenticated_local_test=allow_unauthenticated_local_test,
        )
        self._client = redis.from_url(resolved_url, **client_kwargs)
        self._delivery_mode = delivery_mode
        if capability_queue_contract is None:
            queue_contract = dict(LOCAL_COMPOSE_CAPABILITY_QUEUE_MAP)
        else:
            queue_contract = dict(capability_queue_contract)
        if capability_hint_contract is None:
            hint_contract = dict(LOCAL_COMPOSE_CAPABILITY_HINT_MAP)
        else:
            hint_contract = dict(capability_hint_contract)
        self._capability_queue_contract = _validate_capability_contract(
            queue_contract, kind="destination"
        )
        self._capability_hint_contract = _validate_capability_contract(
            hint_contract, kind="hint-destination"
        )

    async def register_root_task(
        self,
        task: TaskRequested,
        initial_status: str = "pending_dispatch",
        required_capability: str = "general",
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
        required_cap_record = RequiredCapabilityRecord(
            task_id=task.task_id,
            required_capability=required_capability,
            created_at=task.created_at,
            version=1,
        )

        marker_key = f"task.registration:{task.task_id}"
        status_key = f"task.status:{task.task_id}"
        lineage_key = f"task.lineage:{task.task_id}"
        input_key = f"task.input:{task.task_id}:1"
        authority_key = f"task.authority:{task.task_id}"
        capability_key = f"task.required_capability:{task.task_id}"

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
        capability_json = required_cap_record.to_json()

        try:
            raw_result = await self._client.eval(
                REGISTER_ROOT_TASK_LUA,
                6,
                marker_key,
                status_key,
                lineage_key,
                input_key,
                authority_key,
                capability_key,
                marker_json,
                status_json,
                lineage_json,
                input_json,
                authority_json,
                task.task_id,
                capability_json,
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during register_root_task"
            ) from exc
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
        self,
        task: TaskRequested,
        initial_status: str = "pending_dispatch",
        required_capability: str = "analysis",
    ) -> None:
        """Atomically persist pre-validated child task status, forward lineage, and parent membership.

        Domain validation (parent existence and depth-1 policy) is enforced by the
        platform application use case (RegisterChildTask) prior to invoking this method.
        This method executes the atomic storage consistency boundary across status, lineage, capability, and parent set.
        """
        if task.parent_task_id is None or task.root_task_id == task.task_id:
            raise ValueError(f"Task '{task.task_id}' is not a child task")
        cap_rec = RequiredCapabilityRecord(
            task_id=task.task_id,
            required_capability=required_capability,
            created_at=task.created_at,
            version=1,
        )
        keys = [
            f"task.status:{task.task_id}",
            f"task.lineage:{task.task_id}",
            f"task.required_capability:{task.task_id}",
            f"task.children:{task.parent_task_id}",
        ]
        status_json = canonical_json_dumps({"status": initial_status, "task_id": task.task_id})
        lineage_json = canonical_json_dumps({
            "parent_task_id": task.parent_task_id,
            "root_task_id": task.root_task_id,
            "task_id": task.task_id,
        })
        capability_json = cap_rec.to_json()

        try:
            raw_result = await self._client.eval(
                REGISTER_CHILD_TASK_LUA,
                4,
                *keys,
                status_json,
                lineage_json,
                capability_json,
                task.task_id,
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during register_child_task"
            ) from exc
        result = (
            raw_result.decode("utf-8")
            if isinstance(raw_result, (bytes, bytearray))
            else str(raw_result)
        )
        if result in ("OK", "IDEMPOTENT"):
            return
        if result == "CONFLICT":
            raise ConflictingChildTaskRegistrationError(
                f"Child task '{task.task_id}' already exists with conflicting state"
            )
        if result == "PARTIAL_CONFLICT":
            raise ConflictingChildTaskRegistrationError(
                f"Child task '{task.task_id}' has inconsistent or partial authoritative state in storage"
            )
        raise RuntimeError(f"Unexpected Redis register_child_task result: {result}")

    async def get_required_capability(
        self, task_id: str
    ) -> RequiredCapabilityRecord | None:
        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            return None
        key = f"task.required_capability:{clean_task_id}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise TypeError("Required capability payload must be a JSON object")
            return RequiredCapabilityRecord.from_dict(data)
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt RequiredCapabilityRecord at key '{key}': {exc}"
            ) from exc

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
        except (NoPermissionError, AuthenticationError):
            raise
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

    async def get_status(self, task_id: str, now: str = "") -> str | None:
        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            return None

        try:
            raw_outcome = await self._client.get(f"task.outcome:{clean_task_id}")
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt terminal outcome at task.outcome:{clean_task_id}: {exc}"
            ) from exc
        if raw_outcome is not None:
            try:
                outcome_obj = TerminalOutcome.from_json(raw_outcome)
                return outcome_obj.terminal_status
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise CorruptAuthoritativeRecordError(
                    f"Corrupt terminal outcome at task.outcome:{clean_task_id}: {exc}"
                ) from exc

        key = f"task.status:{clean_task_id}"
        try:
            raw = await self._client.get(key)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt task status record at key '{key}': {exc}"
            ) from exc
        if raw is None:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            if isinstance(data, dict) and isinstance(data.get("status"), str):
                cur_s = data["status"]
            else:
                raise TypeError("Status record must be a JSON object containing a string 'status'")
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt task status record at key '{key}': {exc}"
            ) from exc

        # Targeted task-specific expiry evaluation for pending/working tasks
        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )
        if cur_s == "pending":
            try:
                raw_deadline = await self._client.get(f"task.claim_deadline:{clean_task_id}")
            except (NoPermissionError, AuthenticationError):
                raise
            except ResponseError as exc:
                raise CorruptAuthoritativeRecordError(
                    f"Corrupt claim deadline record at task.claim_deadline:{clean_task_id}: {exc}"
                ) from exc
            if raw_deadline is not None:
                deadline_str = raw_deadline if isinstance(raw_deadline, str) else raw_deadline.decode("utf-8")
                if deadline_str and now_iso > deadline_str:
                    try:
                        await self.materialize_claim_timeout(clean_task_id, now_iso)
                        return "claim_timed_out"
                    except (
                        TaskNotClaimableError,
                        CompletionConflictError,
                        ClaimTimeoutConflictError,
                        WaveAlreadyResolvedError,
                    ):
                        pass
                    except CorruptAuthoritativeRecordError:
                        raise
        elif cur_s == "working":
            try:
                raw_claim = await self._client.get(f"task.claim:{clean_task_id}")
            except (NoPermissionError, AuthenticationError):
                raise
            except ResponseError as exc:
                raise CorruptAuthoritativeRecordError(
                    f"Corrupt claim record at task.claim:{clean_task_id}: {exc}"
                ) from exc
            if raw_claim is not None:
                try:
                    claim_obj = ActiveClaim.from_json(raw_claim)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise CorruptAuthoritativeRecordError(
                        f"Corrupt claim record at task.claim:{clean_task_id}: {exc}"
                    ) from exc
                if (
                    claim_obj.state == "acknowledged"
                    and claim_obj.execution_expires_at
                    and now_iso > claim_obj.execution_expires_at
                ):
                    try:
                        await self.transition_expired_claim(clean_task_id, now_iso)
                        return "recovery_required"
                    except (
                        TaskNotClaimableError,
                        CompletionConflictError,
                        WaveAlreadyResolvedError,
                    ):
                        pass
                    except CorruptAuthoritativeRecordError:
                        raise

        return cur_s

    async def save_task_input(self, task_input: TaskInput) -> None:
        target_key = f"task.input:{task_input.task_id}:{task_input.input_version}"
        payload = task_input.to_json()
        try:
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
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during save_task_input"
            ) from exc
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
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt TaskInput record at key '{key}': {exc}"
            ) from exc

    async def save_stage_artifact(
        self,
        artifact: StageArtifact,
        expected_assignment_id: str,
        expected_principal_id: str,
        expected_fencing_epoch: int,
    ) -> None:
        clean_task_id = str(artifact.producing_task_id).strip()
        clean_assignment_id = str(expected_assignment_id).strip()
        clean_principal_id = str(expected_principal_id).strip()
        if not clean_assignment_id or not clean_principal_id or expected_fencing_epoch < 1:
            raise ValueError(
                "save_stage_artifact requires non-empty assignment_id, principal_id, and positive fencing_epoch"
            )

        target_key = (
            f"task.artifact:{clean_task_id}:"
            f"{artifact.artifact_id}:{artifact.artifact_version}"
        )
        index_key = f"task.artifacts:{clean_task_id}"
        claim_key = f"task.claim:{clean_task_id}"
        fence_key = f"task.fence:{clean_task_id}"
        payload = artifact.to_json()
        index_member = f"{artifact.artifact_id}:{artifact.artifact_version}"

        keys = [
            target_key,
            index_key,
            claim_key,
            fence_key,
        ]
        if artifact.artifact_version > 1:
            keys.append(
                f"task.artifact:{clean_task_id}:"
                f"{artifact.artifact_id}:{artifact.artifact_version - 1}"
            )

        try:
            raw_result = await self._client.eval(
                SAVE_STAGE_ARTIFACT_LUA,
                len(keys),
                *keys,
                artifact.artifact_version,
                payload,
                index_member,
                clean_task_id,
                clean_assignment_id,
                clean_principal_id,
                str(expected_fencing_epoch),
                artifact.artifact_type,
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during save_stage_artifact"
            ) from exc
        result = (
            raw_result.decode("utf-8")
            if isinstance(raw_result, (bytes, bytearray))
            else str(raw_result)
        )
        if result in ("OK", "IDEMPOTENT"):
            return
        if result == "INVALID_CLAIM_INPUT":
            raise ValueError(
                "save_stage_artifact requires non-empty assignment_id, principal_id, and positive fencing_epoch"
            )
        if result == "CLAIM_NOT_FOUND":
            raise ClaimNotFoundError(f"Active claim for task '{artifact.producing_task_id}' not found")
        if result in ("CLAIM_TASK_MISMATCH", "ASSIGNMENT_MISMATCH", "PRINCIPAL_MISMATCH"):
            raise ClaimNotFoundError(f"Assignment mismatch for artifact staging on task '{artifact.producing_task_id}': {result}")
        if result == "CLAIM_NOT_ACKNOWLEDGED":
            raise TaskNotClaimableError("Assignment must be acknowledged before staging artifacts")
        if result in ("FENCE_NOT_FOUND", "FENCE_MISMATCH", "STALE_FENCE"):
            raise FencingEpochRaceError(f"Fencing epoch mismatch on task '{artifact.producing_task_id}': {result}")
        if result == "CAPABILITY_POLICY_VIOLATION":
            raise CapabilityMismatchError(f"Artifact staging policy violation for artifact_type '{artifact.artifact_type}'")
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
        if result == "CORRUPT_CLAIM":
            raise CorruptAuthoritativeRecordError(f"Corrupt active claim on task '{artifact.producing_task_id}'")
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
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt StageArtifact record at key '{key}': {exc}"
            ) from exc

    async def save_continuation(self, continuation: TaskContinuation) -> None:
        if not isinstance(continuation, TaskContinuation) or getattr(continuation, "version", None) != 1:
            raise ValueError("Standalone save_continuation supports TaskContinuation v1 only")
        target_key = (
            f"task.continuation:{continuation.parent_task_id}:"
            f"{continuation.continuation_version}"
        )
        latest_key = f"task.continuation_latest:{continuation.parent_task_id}"
        payload = continuation.to_json()
        try:
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
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during save_continuation"
            ) from exc
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
    ) -> TaskContinuation | TaskContinuationV2 | None:
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
            return deserialize_continuation(
                data, expected_continuation_version=continuation_version
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt TaskContinuation record at key '{key}': {exc}"
            ) from exc

    async def get_latest_continuation_version(self, parent_task_id: str) -> int | None:
        clean_parent_task_id = str(parent_task_id).strip()
        if not clean_parent_task_id:
            return None
        key = f"task.continuation_latest:{clean_parent_task_id}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            val = int(text)
            if val < 1 or str(val) != text.strip():
                raise ValueError(f"Invalid continuation_latest value: {text!r}")
            return val
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt continuation_latest at key '{key}': {exc}"
            ) from exc

    async def save_root_authority(self, authority: RootAuthorityRecord) -> None:
        target_key = f"task.authority:{authority.root_task_id}"
        payload = authority.to_json()
        try:
            raw_result = await self._client.eval(
                SAVE_ROOT_AUTHORITY_LUA,
                1,
                target_key,
                payload,
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during save_root_authority"
            ) from exc
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
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt RootAuthorityRecord at key '{key}': {exc}"
            ) from exc

    async def save_authority_grant(self, grant: DelegationGrant | DelegationGrantV2) -> None:
        target_key = f"task.authority_grant:{grant.root_task_id}:{grant.grant_id}"
        payload = grant.to_json()
        try:
            raw_result = await self._client.eval(
                SAVE_AUTHORITY_GRANT_LUA,
                1,
                target_key,
                payload,
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during save_authority_grant"
            ) from exc
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
    ) -> DelegationGrant | DelegationGrantV2 | None:
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
            return deserialize_delegation_grant(data)
        except (NoPermissionError, AuthenticationError):
            raise
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
        try:
            raw = await self._client.get(key)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt ActiveClaim at key '{key}': {exc}"
            ) from exc
        if raw is None:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise TypeError("Payload must be a JSON object")
            return ActiveClaim.from_dict(data)
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt ActiveClaim at key '{key}': {exc}"
            ) from exc

    async def get_fencing_epoch(self, task_id: str) -> int:
        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            return 0
        try:
            raw = await self._client.get(f"task.fence:{clean_task_id}")
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt fencing epoch at key 'task.fence:{clean_task_id}': {exc}"
            ) from exc
        if raw is None:
            return 0
        try:
            val = int(raw if isinstance(raw, str) else raw.decode("utf-8"))
            return val
        except (TypeError, ValueError) as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt fencing epoch at key 'task.fence:{clean_task_id}': {exc}"
            ) from exc

    async def _peek_wave_id_for_resumed_claim(self, task_id: str) -> str:
        """Best-effort read of wave_id so static KEYS can bind task.delegation_wave_resolution.

        All authoritative validation occurs inside CLAIM_TASK_LUA; missing or
        wrongly typed continuation keys fall back to '_missing' so Lua handles
        the rejection atomically without Python-side TOCTOU.
        """
        key = f"task.continuation:{task_id}:2"
        try:
            ktype = await self._client.type(key)
            ktype_str = ktype if isinstance(ktype, str) else ktype.decode("utf-8")
            if ktype_str != "string":
                return "_missing"
            raw = await self._client.get(key)
            if raw is None:
                return "_missing"
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            data = json.loads(text)
            if isinstance(data, dict):
                wid = data.get("wave_id")
                if isinstance(wid, str) and wid.strip():
                    return wid.strip()
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception:  # noqa: BLE001, S110
            pass
        return "_missing"

    async def _eval_claim_resumed_v2(
        self,
        clean_task_id: str,
        claim: ActiveClaim,
        now_iso: str,
        working_status_json: str,
        claim_payload: str,
        expected_continuation_version: int,
        is_root: bool,
    ) -> Any:
        wave_id = await self._peek_wave_id_for_resumed_claim(clean_task_id)
        v2_keys = [
            f"task.status:{clean_task_id}",
            f"task.claim:{clean_task_id}",
            f"task.fence:{clean_task_id}",
            f"task.required_capability:{clean_task_id}",
            f"task.continuation_latest:{clean_task_id}",
            f"task.authority:{claim.root_task_id}",
            f"task.claim_deadline:{clean_task_id}",
            f"task.authority_grant:{claim.root_task_id}:unused",
            f"task.continuation:{clean_task_id}:2",
            f"task.delegation_wave_resolution:{clean_task_id}:{wave_id}",
        ]
        try:
            return await self._client.eval(
                CLAIM_TASK_LUA,
                len(v2_keys),
                *v2_keys,
                clean_task_id,
                now_iso,
                claim.runtime_principal_id,
                claim.request_id,
                working_status_json,
                claim_payload,
                "",
                str(claim.fencing_epoch),
                claim.assigned_capability,
                claim.execution_kind,
                "1" if is_root else "0",
                str(expected_continuation_version),
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during atomic_claim_task"
            ) from exc

    async def atomic_claim_task(
        self,
        task_id: str,
        claim: ActiveClaim,
        grant: DelegationGrant | DelegationGrantV2 | None = None,
        now: str = "",
        expected_continuation_version: int = -1,
        is_root: bool = True,
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
        capability_key = f"task.required_capability:{clean_task_id}"
        continuation_latest_key = f"task.continuation_latest:{clean_task_id}"
        authority_key = f"task.authority:{claim.root_task_id}"
        deadline_key = f"task.claim_deadline:{clean_task_id}"

        claim_payload = claim.to_json()
        grant_payload = grant.to_json() if grant else ""

        working_status_json = canonical_json_dumps({
            "status": "working",
            "task_id": clean_task_id,
        })

        # Keep the seven always-required keys at KEYS[1..7]; append the
        # authority grant key as KEYS[8] only when a grant is supplied.
        keys = [
            status_key,
            claim_key,
            fence_key,
            capability_key,
            continuation_latest_key,
            authority_key,
            deadline_key,
        ]
        if grant is not None:
            keys.append(f"task.authority_grant:{claim.root_task_id}:{grant.grant_id}")

        try:
            if (
                claim.execution_kind == "resumed_continuation"
                and expected_continuation_version == 2
                and grant is None
            ):
                raw_res = await self._eval_claim_resumed_v2(
                    clean_task_id=clean_task_id,
                    claim=claim,
                    now_iso=now_iso,
                    working_status_json=working_status_json,
                    claim_payload=claim_payload,
                    expected_continuation_version=expected_continuation_version,
                    is_root=is_root,
                )
            else:
                raw_res = await self._client.eval(
                    CLAIM_TASK_LUA,
                    len(keys),
                    *keys,
                    clean_task_id,
                    now_iso,
                    claim.runtime_principal_id,
                    claim.request_id,
                    working_status_json,
                    claim_payload,
                    grant_payload,
                    str(claim.fencing_epoch),
                    claim.assigned_capability,
                    claim.execution_kind,
                    "1" if is_root else "0",
                    str(expected_continuation_version),
                )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during atomic_claim_task"
            ) from exc

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

        if code == "MISSING_REQUIRED_CAPABILITY":
            raise CorruptAuthoritativeRecordError(f"Missing required capability record for task '{clean_task_id}'")

        if code == "CORRUPT_REQUIRED_CAPABILITY":
            raise CorruptAuthoritativeRecordError(f"Corrupt required capability record for task '{clean_task_id}'")

        if code == "CAPABILITY_MISMATCH":
            auth_cap = raw_res[1] if len(raw_res) > 1 else "unknown"
            if not isinstance(auth_cap, str):
                auth_cap = auth_cap.decode("utf-8")
            raise CapabilityMismatchError(
                f"Task '{clean_task_id}' requires capability '{auth_cap}', but '{claim.assigned_capability}' was requested"
            )

        if code == "LIFECYCLE_MISMATCH":
            cur_status = raw_res[1] if len(raw_res) > 1 else "unknown"
            if not isinstance(cur_status, str):
                cur_status = cur_status.decode("utf-8")
            raise TaskNotClaimableError(
                f"Task '{clean_task_id}' lifecycle status '{cur_status}' is incompatible with execution_kind '{claim.execution_kind}'"
            )

        if code == "INVALID_EXECUTION_KIND":
            raise ValueError(f"Invalid execution kind: '{claim.execution_kind}'")

        if code == "INVALID_CONTINUATION_VERSION":
            raise ValueError("expected_continuation_version must be a positive integer for resumed_continuation")

        if code == "MISSING_CONTINUATION_LATEST":
            raise TaskNotClaimableError(f"No continuation checkpoint found for task '{clean_task_id}'")

        if code == "CORRUPT_CONTINUATION_LATEST":
            raise CorruptAuthoritativeRecordError(f"Corrupt continuation latest record for task '{clean_task_id}'")

        if code == "CONTINUATION_VERSION_MISMATCH":
            latest_ver = raw_res[1] if len(raw_res) > 1 else "unknown"
            if not isinstance(latest_ver, str):
                latest_ver = latest_ver.decode("utf-8")
            raise TaskNotClaimableError(
                f"Continuation version mismatch for task '{clean_task_id}': latest={latest_ver}, expected={expected_continuation_version}"
            )

        if code == "UNEXPECTED_CONTINUATION_VERSION":
            raise ValueError("continuation_version must remain absent for non-resumed claims")

        if code in ("MISSING_CONTINUATION", "CONTINUATION_NOT_CLAIMABLE", "UNSUPPORTED_WAVE_RESOLUTION"):
            detail = raw_res[1] if len(raw_res) > 1 else code
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8")
            raise TaskNotClaimableError(
                f"Task '{clean_task_id}' continuation v2 is not claimable ({code}: {detail})"
            )

        if code in ("CORRUPT_CONTINUATION", "MISSING_RESOLVED_WAVE", "CORRUPT_RESOLVED_WAVE", "WAVE_RESOLUTION_MISMATCH"):
            raise CorruptAuthoritativeRecordError(
                f"Authoritative continuation or resolved wave validation failed on task '{clean_task_id}': {code}"
            )

        if code == "ALREADY_CLAIMED":
            raise TaskAlreadyClaimedError(f"Task '{clean_task_id}' has an active claim by another runtime")

        if code == "EXECUTION_CLAIM_EXPIRED":
            try:
                await self.transition_expired_claim(clean_task_id, now_iso)
            except (TaskNotClaimableError, CompletionConflictError, WaveAlreadyResolvedError):
                pass
            except CorruptAuthoritativeRecordError:
                raise
            raise TaskNotClaimableError(
                f"Task '{clean_task_id}' execution claim has expired and transitioned to recovery_required"
            )

        if code == "CHILD_CLAIM_DEADLINE_EXPIRED":
            deadline_str = raw_res[1] if len(raw_res) > 1 else ""
            if isinstance(deadline_str, bytes):
                deadline_str = deadline_str.decode("utf-8")
            try:
                await self.materialize_claim_timeout(clean_task_id, now_iso)
            except (
                TaskNotClaimableError,
                CompletionConflictError,
                ClaimTimeoutConflictError,
                WaveAlreadyResolvedError,
            ):
                pass
            except CorruptAuthoritativeRecordError:
                raise
            raise ChildClaimDeadlineExpiredError(
                f"Claim deadline for child task '{clean_task_id}' has expired ({deadline_str})"
            )

        if code in ("MISSING_ROOT_AUTHORITY", "CORRUPT_ROOT_AUTHORITY", "INVALID_GRANT_BOUNDS"):
            raise DelegationConflictError(f"Root authority or grant validation failed for task '{clean_task_id}': {code}")
            raise CorruptAuthoritativeRecordError(
                f"Corrupt or missing RootAuthorityRecord on root task '{claim.root_task_id}'"
            )

        if code in ("INVALID_ROOT_AUTHORITY_BOUNDS", "INVALID_ROOT_AUTHORITY_EDGES"):
            raise TaskNotClaimableError(
                f"RootAuthorityRecord policy does not permit delegation grant for root task '{claim.root_task_id}'"
            )

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
            raise FencingEpochRaceError(f"Concurrent claim conflict on task '{clean_task_id}' (fencing epoch race)")

        if code in ("CORRUPT_CLAIM", "CORRUPT_FENCE", "CORRUPT_STATUS"):
            raise CorruptAuthoritativeRecordError(f"Corrupt authoritative record found on task '{clean_task_id}': {code}")

        raise RuntimeError(f"Unexpected claim result code: {code}")

    async def acknowledge_claim(
        self,
        task_id: str,
        assignment_id: str,
        principal_id: str,
        acknowledged_at: str,
        execution_expires_at: str | None = None,
    ) -> ActiveClaim:
        clean_task_id = str(task_id).strip()
        clean_assignment_id = str(assignment_id).strip()
        clean_principal_id = str(principal_id).strip()
        clean_time = normalize_canonical_utc_timestamp(acknowledged_at)
        clean_exec_expires = (
            normalize_canonical_utc_timestamp(execution_expires_at)
            if execution_expires_at
            else ""
        )

        claim_key = f"task.claim:{clean_task_id}"

        try:
            raw_res = await self._client.eval(
                ACKNOWLEDGE_CLAIM_LUA,
                1,
                claim_key,
                clean_assignment_id,
                clean_principal_id,
                clean_time,
                clean_exec_expires,
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during acknowledge_claim"
            ) from exc
        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(f"Unexpected Redis acknowledge_claim response: {raw_res}")

        code = raw_res[0] if isinstance(raw_res[0], str) else raw_res[0].decode("utf-8")

        if code in ("OK", "IDEMPOTENT"):
            raw_json = raw_res[1] if isinstance(raw_res[1], str) else raw_res[1].decode("utf-8")
            return ActiveClaim.from_json(raw_json)

        if code == "CLAIM_NOT_FOUND":
            raise ClaimNotFoundError(f"Active claim for task '{clean_task_id}' not found")

        if code in ("ASSIGNMENT_MISMATCH", "PRINCIPAL_MISMATCH", "MISMATCH"):
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
        try:
            res = await self._client.eval(
                EXPIRE_DELIVERY_LEASE_LUA,
                2,
                claim_key,
                status_key,
                now_iso,
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during expire_delivery_lease"
            ) from exc
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
        if code in ("CORRUPT_CLAIM", "CORRUPT_STATUS"):
            raise CorruptAuthoritativeRecordError(
                f"Corrupt authoritative record on task '{clean_task_id}' during lease expiry: {code}"
            )

        raise RuntimeError(f"Unexpected expire_delivery_lease result code: {code}")

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
        clean_parent = str(parent_task_id).strip()
        clean_child = str(proposal.child_task_id).strip()
        clean_proposal_id = str(proposal.proposal_id).strip()

        if not clean_parent or not clean_child or not clean_proposal_id:
            raise ValueError("parent_task_id, child_task_id, and proposal_id must be non-empty strings")

        if not child_task_input.artifact_references:
            raise ValueError("Child task input must reference at least one sealed artifact")
        art_ref = child_task_input.artifact_references[0]

        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )

        child_lineage = TaskLineage(
            task_id=clean_child,
            root_task_id=clean_parent,
            parent_task_id=clean_parent,
        )
        child_cap_record = RequiredCapabilityRecord(
            task_id=clean_child,
            required_capability="analysis",
            created_at=now_iso,
            version=1,
        )
        child_status_dict = {
            "claim_deadline": proposal.child_claim_deadline,
            "created_at": now_iso,
            "status": "pending",
            "task_id": clean_child,
        }
        parent_status_dict = {
            "status": "waiting_for_children",
            "task_id": clean_parent,
            "updated_at": now_iso,
        }

        keys = [
            f"task.delegation_proposal:{clean_parent}:{clean_proposal_id}",
            f"task.status:{clean_parent}",
            f"task.claim:{clean_parent}",
            f"task.fence:{clean_parent}",
            f"task.lineage:{clean_parent}",
            f"task.authority:{clean_parent}",
            f"task.authority_grant:{clean_parent}:{proposal.delegation_grant_id.strip()}",
            f"task.consumed_waves:{clean_parent}",
            f"task.consumed_grants:{clean_parent}",
            f"task.delegation_decisions:{clean_parent}",
            f"task.assignment_yield:{clean_parent}:{yield_record.assignment_id.strip()}",
            f"task.continuation:{clean_parent}:{parent_continuation.continuation_version}",
            f"task.continuation_latest:{clean_parent}",
            f"task.artifact:{art_ref.producing_task_id}:{art_ref.artifact_id}:{art_ref.artifact_version}",
            f"task.input:{clean_parent}:{parent_continuation.input_version}",
            f"task.children:{clean_parent}",
            f"task.status:{clean_child}",
            f"task.lineage:{clean_child}",
            f"task.required_capability:{clean_child}",
            f"task.input:{clean_child}:1",
            f"task.claim_deadline:{clean_child}",
            f"task.fence:{clean_child}",
            "agent.hints.analysis",
        ]

        argv = [
            clean_parent,
            clean_proposal_id,
            proposal.assignment_id.strip(),
            proposal.runtime_principal_id.strip(),
            proposal.proposal_fingerprint.strip(),
            proposal.delegation_grant_id.strip(),
            str(expected_fencing_epoch),
            clean_child,
            proposal.delegation_decision_id.strip(),
            now_iso,
            proposal.child_claim_deadline,
            art_ref.artifact_id.strip(),
            str(art_ref.artifact_version),
            proposal.bound_agent_id.strip(),
            str(parent_continuation.input_version),
            proposal.to_json(),
            canonical_json_dumps(child_status_dict),
            canonical_json_dumps(child_lineage.to_dict()),
            child_cap_record.to_json(),
            child_task_input.to_json(),
            parent_continuation.to_json(),
            canonical_json_dumps(parent_status_dict),
            yield_record.to_json(),
            wake_up_hint.to_json(),
        ]

        try:
            raw_res = await self._client.eval(ACCEPT_DELEGATION_LUA, len(keys), *keys, *argv)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during accept_delegation_proposal"
            ) from exc
        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(f"Unexpected Redis accept_delegation_proposal response: {raw_res}")

        code = raw_res[0] if isinstance(raw_res[0], str) else raw_res[0].decode("utf-8")

        if code == "OK":
            return (proposal, False)

        if code == "IDEMPOTENT":
            stored_raw = raw_res[1] if isinstance(raw_res[1], str) else raw_res[1].decode("utf-8")
            stored = DelegationProposalRecord.from_json(stored_raw)
            return (stored, True)

        if code == "PROPOSAL_CONFLICT":
            raise DelegationConflictError(
                f"Conflicting proposal replay for proposal '{clean_proposal_id}'"
            )

        if code in ("PARENT_NOT_FOUND", "ROOT_AUTHORITY_NOT_FOUND", "GRANT_NOT_FOUND"):
            raise ParentTaskNotFoundError(f"Parent task or authoritative policy record '{clean_parent}' not found: {code}")

        if code == "PARENT_NOT_WORKING":
            cur_s = raw_res[1] if len(raw_res) > 1 else "unknown"
            if isinstance(cur_s, bytes):
                cur_s = cur_s.decode("utf-8")
            raise TaskNotClaimableError(
                f"Parent task '{clean_parent}' must be in 'working' status, got '{cur_s}'"
            )

        if code in ("WAVE_ALREADY_CONSUMED", "CHILD_LIMIT_EXCEEDED", "POLICY_BOUNDS_VIOLATION", "GRANT_BOUNDS_VIOLATION"):
            raise WaveBudgetExceededError(f"Delegation policy or wave budget exceeded for task '{clean_parent}': {code}")

        if code == "GRANT_ALREADY_CONSUMED":
            raise GrantAlreadyConsumedError(
                f"Authority grant '{proposal.delegation_grant_id}' has already been consumed"
            )

        if code in (
            "CLAIM_NOT_FOUND",
            "ASSIGNMENT_MISMATCH",
            "PRINCIPAL_MISMATCH",
            "BOUND_AGENT_MISMATCH",
            "GRANT_ASSIGNMENT_MISMATCH",
            "GRANT_PRINCIPAL_MISMATCH",
            "GRANT_TURN_MISMATCH",
        ):
            raise ClaimNotFoundError(f"Active claim or grant verification failed on parent task '{clean_parent}': {code}")

        if code == "CLAIM_NOT_ACKNOWLEDGED":
            raise TaskNotClaimableError("Parent assignment must be acknowledged before proposing delegation")

        if code in ("CAPABILITY_POLICY_VIOLATION", "GRANT_CAPABILITY_MISMATCH"):
            raise CapabilityMismatchError(f"Delegation capability mismatch on parent task '{clean_parent}': {code}")

        if code in ("FENCE_MISMATCH", "GRANT_EPOCH_MISMATCH"):
            f_epoch = raw_res[1] if len(raw_res) > 1 else "unknown"
            if isinstance(f_epoch, bytes):
                f_epoch = f_epoch.decode("utf-8")
            raise FencingEpochRaceError(
                f"Fencing epoch race or grant mismatch for parent task '{clean_parent}': expected {expected_fencing_epoch}, got {f_epoch} ({code})"
            )

        if code in ("PARENT_NOT_ROOT", "EXECUTION_KIND_MISMATCH"):
            raise TaskNotClaimableError(f"Parent task '{clean_parent}' is not an eligible root task: {code}")

        if code in (
            "ARTIFACT_NOT_FOUND",
            "ARTIFACT_NOT_SEALED",
            "ARTIFACT_PROVENANCE_MISMATCH",
            "ARTIFACT_TYPE_MISMATCH",
            "ARTIFACT_IDENTITY_MISMATCH",
            "ARTIFACT_MISSING_TURN_ID",
            "ARTIFACT_TURN_MISMATCH",
            "UNSUPPORTED_ARTIFACT_SCHEMA_VERSION",
            "UNSUPPORTED_ARTIFACT_CONTRACT_VERSION",
            "PARENT_INPUT_NOT_FOUND",
            "PARENT_INPUT_TASK_MISMATCH",
            "PARENT_INPUT_VERSION_MISMATCH",
        ):
            raise ValueError(f"Referenced artifact or input check failed on parent task '{clean_parent}': {code}")

        if code == "CONTINUATION_EXISTS":
            raise ConflictingContinuationError(f"Continuation version 1 already exists for task '{clean_parent}'")

        if code in ("CHILD_TASK_ALREADY_EXISTS", "YIELD_ALREADY_EXISTS"):
            raise ConflictingChildTaskRegistrationError(f"Child task or yield record for '{clean_child}' already exists: {code}")

        if code in ("ROOT_AUTHORITY_BINDING_MISMATCH", "GRANT_ROOT_BINDING_MISMATCH", "GRANT_ID_MISMATCH", "DELEGATION_EDGE_NOT_PERMITTED"):
            raise DelegationConflictError(f"Delegation authority conflict on parent task '{clean_parent}': {code}")

        if code.startswith(("WRONG_KEY_TYPE_", "CORRUPT_", "UNSUPPORTED_PARENT_INPUT_")):
            raise CorruptAuthoritativeRecordError(f"Authoritative record or type corruption on task '{clean_parent}': {code}")

        raise RuntimeError(f"Unexpected accept_delegation_proposal result code: {code}")

    async def get_delegation_wave_proposal(
        self,
        parent_task_id: str,
        proposal_id: str,
    ) -> DelegationWaveProposalRecord | None:
        clean_parent = str(parent_task_id).strip()
        clean_proposal_id = str(proposal_id).strip()
        if not clean_parent or not clean_proposal_id:
            return None
        key = f"task.delegation_wave_proposal:{clean_parent}:{clean_proposal_id}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            return DelegationWaveProposalRecord.from_json(text)
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt DelegationWaveProposalRecord at key '{key}': {exc}"
            ) from exc

    async def get_accepted_delegation_wave(
        self,
        parent_task_id: str,
        wave_id: str,
    ) -> AcceptedDelegationWave | None:
        clean_parent = str(parent_task_id).strip()
        clean_wave_id = str(wave_id).strip()
        if not clean_parent or not clean_wave_id:
            return None
        key = f"task.delegation_wave:{clean_parent}:{clean_wave_id}"
        try:
            raw = await self._client.get(key)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt AcceptedDelegationWave at key '{key}': {exc}"
            ) from exc
        if raw is None:
            return None
        try:
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            return AcceptedDelegationWave.from_json(text)
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt AcceptedDelegationWave at key '{key}': {exc}"
            ) from exc

    async def get_resolved_delegation_wave(
        self,
        parent_task_id: str,
        wave_id: str,
    ) -> ResolvedDelegationWave | None:
        clean_parent = str(parent_task_id).strip()
        clean_wave_id = str(wave_id).strip()
        if not clean_parent or not clean_wave_id:
            return None
        key = f"task.delegation_wave_resolution:{clean_parent}:{clean_wave_id}"
        try:
            raw = await self._client.get(key)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt ResolvedDelegationWave at key '{key}': {exc}"
            ) from exc
        if raw is None:
            return None
        try:
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            return ResolvedDelegationWave.from_json(text)
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt ResolvedDelegationWave at key '{key}': {exc}"
            ) from exc

    async def get_child_completion_marker(
        self,
        child_task_id: str,
    ) -> dict[str, Any] | None:
        clean_child = str(child_task_id).strip()
        if not clean_child:
            return None
        key = f"task.child_completion:{clean_child}"
        try:
            raw = await self._client.get(key)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt child completion marker at key '{key}': {exc}"
            ) from exc
        if raw is None:
            return None
        try:
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise TypeError("Child completion marker must be a JSON object")
            return parsed
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt child completion marker at key '{key}': {exc}"
            ) from exc

    async def get_wave_yield(
        self,
        parent_task_id: str,
        assignment_id: str,
    ) -> WaveYieldRecord | None:
        clean_parent = str(parent_task_id).strip()
        clean_assign = str(assignment_id).strip()
        if not clean_parent or not clean_assign:
            return None
        key = f"task.assignment_yield:{clean_parent}:{clean_assign}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            return WaveYieldRecord.from_json(text)
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt WaveYieldRecord at key '{key}': {exc}"
            ) from exc

    async def get_child_claim_deadline(
        self,
        child_task_id: str,
    ) -> str | None:
        clean_child = str(child_task_id).strip()
        if not clean_child:
            return None
        key = f"task.claim_deadline:{clean_child}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        text = raw if isinstance(raw, str) else raw.decode("utf-8")
        return text.strip()

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
        clean_parent = str(parent_task_id).strip()
        clean_proposal_id = str(proposal.proposal_id).strip()
        clean_wave_id = str(accepted_wave.wave_id).strip()
        if not clean_parent or not clean_proposal_id or not clean_wave_id:
            raise ValueError("parent_task_id, proposal_id, and wave_id must be non-empty strings")

        if not (MIN_WAVE_CHILDREN <= len(child_task_inputs) <= MAX_WAVE_CHILDREN):
            raise ValueError(
                f"child_task_inputs must contain {MIN_WAVE_CHILDREN} to {MAX_WAVE_CHILDREN} items"
            )
        if len(child_task_inputs) != len(wake_up_hints):
            raise ValueError("wake_up_hints length must match child_task_inputs length")

        inputs_by_id = {ti.task_id: ti for ti in child_task_inputs}
        hints_by_id = {h.task_id: h for h in wake_up_hints}
        if tuple(sorted(inputs_by_id.keys())) != accepted_wave.required_child_task_ids:
            raise ValueError("child_task_inputs IDs must match accepted_wave.required_child_task_ids")
        if tuple(sorted(hints_by_id.keys())) != accepted_wave.required_child_task_ids:
            raise ValueError("wake_up_hints IDs must match accepted_wave.required_child_task_ids")
        if parent_continuation.required_child_task_ids != accepted_wave.required_child_task_ids:
            raise ValueError("parent_continuation membership must match accepted_wave")
        if parent_continuation.wave_id != clean_wave_id or parent_continuation.parent_task_id != clean_parent:
            raise ValueError("parent_continuation wave/parent binding mismatch")
        if parent_continuation.continuation_version != 2 or parent_continuation.version != 2:
            raise ValueError("accept_delegation_wave requires TaskContinuationV2 with continuation_version=2")

        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )

        ordered_children = list(accepted_wave.required_child_task_ids)
        child_count = len(ordered_children)

        def _child_bundle(idx: int) -> tuple[str, str, str, str, str, str, str, str, str]:
            cid = ordered_children[idx]
            ti = inputs_by_id[cid]
            if not ti.artifact_references:
                raise ValueError(f"Child task input '{cid}' must reference at least one sealed artifact")
            art = ti.artifact_references[0]
            hint = hints_by_id[cid]
            lineage = TaskLineage(task_id=cid, root_task_id=clean_parent, parent_task_id=clean_parent)
            cap = RequiredCapabilityRecord(
                task_id=cid,
                required_capability="analysis",
                created_at=now_iso,
                version=1,
            )
            status_json = canonical_json_dumps({
                "claim_deadline": proposal.child_claim_deadline,
                "created_at": now_iso,
                "status": "pending",
                "task_id": cid,
            })
            return (
                cid,
                status_json,
                canonical_json_dumps(lineage.to_dict()),
                cap.to_json(),
                ti.to_json(),
                hint.to_json(),
                f"task.artifact:{art.producing_task_id}:{art.artifact_id}:{art.artifact_version}",
                art.artifact_id.strip(),
                str(art.artifact_version),
            )

        (
            c1_id, c1_status_json, c1_lineage_json, c1_cap_json, c1_input_json,
            c1_hint_json, _art1_key, c1_art_id, c1_art_ver,
        ) = _child_bundle(0)
        (
            c2_id, c2_status_json, c2_lineage_json, c2_cap_json, c2_input_json,
            c2_hint_json, _art2_key, c2_art_id, c2_art_ver,
        ) = _child_bundle(1)
        (
            c3_id, c3_status_json, c3_lineage_json, c3_cap_json, c3_input_json,
            c3_hint_json, _art3_key, c3_art_id, c3_art_ver,
        ) = _child_bundle(2) if child_count >= 3 else _child_bundle(0)
        (
            c4_id, c4_status_json, c4_lineage_json, c4_cap_json, c4_input_json,
            c4_hint_json, _art4_key, c4_art_id, c4_art_ver,
        ) = _child_bundle(3) if child_count >= 4 else _child_bundle(0)

        art1 = inputs_by_id[c1_id].artifact_references[0]
        art2 = inputs_by_id[c2_id].artifact_references[0]
        art3 = inputs_by_id[c3_id].artifact_references[0]
        art4 = inputs_by_id[c4_id].artifact_references[0]

        parent_status_dict = {
            "status": "waiting_for_children",
            "task_id": clean_parent,
            "updated_at": now_iso,
        }

        keys = [
            f"task.delegation_wave_proposal:{clean_parent}:{clean_proposal_id}",
            f"task.delegation_wave:{clean_parent}:{clean_wave_id}",
            f"task.status:{clean_parent}",
            f"task.claim:{clean_parent}",
            f"task.fence:{clean_parent}",
            f"task.lineage:{clean_parent}",
            f"task.authority:{clean_parent}",
            f"task.authority_grant:{clean_parent}:{proposal.delegation_grant_id.strip()}",
            f"task.consumed_waves:{clean_parent}",
            f"task.consumed_grants:{clean_parent}",
            f"task.delegation_decisions:{clean_parent}",
            f"task.assignment_yield:{clean_parent}:{yield_record.assignment_id.strip()}",
            f"task.continuation:{clean_parent}:2",
            f"task.continuation_latest:{clean_parent}",
            f"task.input:{clean_parent}:{parent_continuation.input_version}",
            f"task.children:{clean_parent}",
            "agent.hints.analysis",
            f"task.status:{c1_id}",
            f"task.lineage:{c1_id}",
            f"task.required_capability:{c1_id}",
            f"task.input:{c1_id}:1",
            f"task.claim_deadline:{c1_id}",
            f"task.fence:{c1_id}",
            f"task.status:{c2_id}",
            f"task.lineage:{c2_id}",
            f"task.required_capability:{c2_id}",
            f"task.input:{c2_id}:1",
            f"task.claim_deadline:{c2_id}",
            f"task.fence:{c2_id}",
            f"task.status:{c3_id}",
            f"task.lineage:{c3_id}",
            f"task.required_capability:{c3_id}",
            f"task.input:{c3_id}:1",
            f"task.claim_deadline:{c3_id}",
            f"task.fence:{c3_id}",
            f"task.status:{c4_id}",
            f"task.lineage:{c4_id}",
            f"task.required_capability:{c4_id}",
            f"task.input:{c4_id}:1",
            f"task.claim_deadline:{c4_id}",
            f"task.fence:{c4_id}",
            f"task.artifact:{art1.producing_task_id}:{art1.artifact_id}:{art1.artifact_version}",
            f"task.artifact:{art2.producing_task_id}:{art2.artifact_id}:{art2.artifact_version}",
            f"task.artifact:{art3.producing_task_id}:{art3.artifact_id}:{art3.artifact_version}",
            f"task.artifact:{art4.producing_task_id}:{art4.artifact_id}:{art4.artifact_version}",
        ]

        argv = [
            clean_parent,
            clean_proposal_id,
            clean_wave_id,
            proposal.assignment_id.strip(),
            proposal.runtime_principal_id.strip(),
            proposal.bound_agent_id.strip(),
            proposal.proposal_fingerprint.strip(),
            proposal.delegation_grant_id.strip(),
            str(expected_fencing_epoch),
            proposal.delegation_decision_id.strip(),
            proposal.child_claim_deadline,
            str(parent_continuation.input_version),
            str(child_count),
            accepted_wave.barrier_policy,
            "" if accepted_wave.required_success_count is None else str(accepted_wave.required_success_count),
            proposal.to_json(),
            accepted_wave.to_json(),
            parent_continuation.to_json(),
            canonical_json_dumps(parent_status_dict),
            yield_record.to_json(),
            c1_id, c1_status_json, c1_lineage_json, c1_cap_json, c1_input_json, c1_hint_json, c1_art_id, c1_art_ver,
            c2_id, c2_status_json, c2_lineage_json, c2_cap_json, c2_input_json, c2_hint_json, c2_art_id, c2_art_ver,
            c3_id, c3_status_json, c3_lineage_json, c3_cap_json, c3_input_json, c3_hint_json, c3_art_id, c3_art_ver,
            c4_id, c4_status_json, c4_lineage_json, c4_cap_json, c4_input_json, c4_hint_json, c4_art_id, c4_art_ver,
            "" if accepted_wave.required_success_fraction is None else str(accepted_wave.required_success_fraction.numerator),
            "" if accepted_wave.required_success_fraction is None else str(accepted_wave.required_success_fraction.denominator),
        ]

        try:
            raw_res = await self._client.eval(ACCEPT_DELEGATION_WAVE_LUA, len(keys), *keys, *argv)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during accept_delegation_wave"
            ) from exc
        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(f"Unexpected Redis accept_delegation_wave response: {raw_res}")

        code = raw_res[0] if isinstance(raw_res[0], str) else raw_res[0].decode("utf-8")

        if code == "OK":
            return (proposal, False)
        if code == "IDEMPOTENT":
            stored_raw = raw_res[1] if isinstance(raw_res[1], str) else raw_res[1].decode("utf-8")
            stored = DelegationWaveProposalRecord.from_json(stored_raw)
            return (stored, True)
        if code == "PROPOSAL_CONFLICT":
            raise DelegationConflictError(
                f"Conflicting wave proposal replay for proposal '{clean_proposal_id}'"
            )
        if code in ("PARENT_NOT_FOUND", "ROOT_AUTHORITY_NOT_FOUND", "GRANT_NOT_FOUND"):
            raise ParentTaskNotFoundError(
                f"Parent task or authoritative policy record '{clean_parent}' not found: {code}"
            )
        if code == "PARENT_NOT_WORKING":
            cur_s = raw_res[1] if len(raw_res) > 1 else "unknown"
            if isinstance(cur_s, bytes):
                cur_s = cur_s.decode("utf-8")
            raise TaskNotClaimableError(
                f"Parent task '{clean_parent}' must be in 'working' status, got '{cur_s}'"
            )
        if code in ("WAVE_ALREADY_CONSUMED", "CHILD_LIMIT_EXCEEDED", "POLICY_BOUNDS_VIOLATION", "GRANT_BOUNDS_VIOLATION"):
            raise WaveBudgetExceededError(
                f"Delegation wave policy or budget exceeded for task '{clean_parent}': {code}"
            )
        if code == "GRANT_ALREADY_CONSUMED":
            raise GrantAlreadyConsumedError(
                f"Authority grant '{proposal.delegation_grant_id}' has already been consumed"
            )
        if code in (
            "CLAIM_NOT_FOUND",
            "ASSIGNMENT_MISMATCH",
            "PRINCIPAL_MISMATCH",
            "BOUND_AGENT_MISMATCH",
            "GRANT_ASSIGNMENT_MISMATCH",
            "GRANT_PRINCIPAL_MISMATCH",
            "GRANT_TURN_MISMATCH",
        ):
            raise ClaimNotFoundError(
                f"Active claim or grant verification failed on parent task '{clean_parent}': {code}"
            )
        if code == "CLAIM_NOT_ACKNOWLEDGED":
            raise TaskNotClaimableError("Parent assignment must be acknowledged before proposing delegation wave")
        if code in ("CAPABILITY_POLICY_VIOLATION", "GRANT_CAPABILITY_MISMATCH"):
            raise CapabilityMismatchError(
                f"Delegation capability mismatch on parent task '{clean_parent}': {code}"
            )
        if code in ("FENCE_MISMATCH", "GRANT_EPOCH_MISMATCH"):
            f_epoch = raw_res[1] if len(raw_res) > 1 else "unknown"
            if isinstance(f_epoch, bytes):
                f_epoch = f_epoch.decode("utf-8")
            raise FencingEpochRaceError(
                f"Fencing epoch race or grant mismatch for parent task '{clean_parent}': expected {expected_fencing_epoch}, got {f_epoch} ({code})"
            )
        if code in ("PARENT_NOT_ROOT", "EXECUTION_KIND_MISMATCH"):
            raise TaskNotClaimableError(
                f"Parent task '{clean_parent}' is not an eligible root task: {code}"
            )
        if code in (
            "ARTIFACT_NOT_FOUND",
            "ARTIFACT_NOT_SEALED",
            "ARTIFACT_PROVENANCE_MISMATCH",
            "ARTIFACT_TYPE_MISMATCH",
            "ARTIFACT_IDENTITY_MISMATCH",
            "ARTIFACT_MISSING_TURN_ID",
            "ARTIFACT_TURN_MISMATCH",
            "UNSUPPORTED_ARTIFACT_SCHEMA_VERSION",
            "UNSUPPORTED_ARTIFACT_CONTRACT_VERSION",
            "PARENT_INPUT_NOT_FOUND",
            "PARENT_INPUT_TASK_MISMATCH",
            "PARENT_INPUT_VERSION_MISMATCH",
        ):
            raise ValueError(f"Referenced artifact or input check failed on parent task '{clean_parent}': {code}")
        if code == "CONTINUATION_EXISTS":
            raise ConflictingContinuationError(f"Continuation already exists for task '{clean_parent}'")
        if code in ("CHILD_TASK_ALREADY_EXISTS", "YIELD_ALREADY_EXISTS", "WAVE_ALREADY_EXISTS"):
            raise ConflictingChildTaskRegistrationError(
                f"Child task, wave, or yield record for '{clean_parent}' already exists: {code}"
            )
        if code in (
            "ROOT_AUTHORITY_BINDING_MISMATCH",
            "GRANT_ROOT_BINDING_MISMATCH",
            "GRANT_ID_MISMATCH",
            "DELEGATION_EDGE_NOT_PERMITTED",
            "WAVE_CONSISTENCY_MISMATCH",
        ):
            raise DelegationConflictError(
                f"Delegation wave authority conflict on parent task '{clean_parent}': {code}"
            )
        if code.startswith(("WRONG_KEY_TYPE_", "CORRUPT_", "UNSUPPORTED_PARENT_INPUT_")):
            raise CorruptAuthoritativeRecordError(
                f"Authoritative record or type corruption on task '{clean_parent}': {code}"
            )
        raise RuntimeError(f"Unexpected accept_delegation_wave result code: {code}")

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
        """Commit a fenced turn completion via COMPLETE_TURN_LUA.

        Builds the canonical outcome, result, audit, replay-record, status,
        hint, and marker payloads from the validated command, then commits
        them in one single-node Lua script. All expected type, decode,
        authorization, fencing, conflict, lifecycle, and policy failures are
        rejected in read-only Phase 1 with zero mutation; Phase 2 performs
        only prevalidated Redis writes. Redis serializes script execution
        atomically with respect to other clients. No rollback is claimed for
        unexpected command errors, host failures, out-of-memory conditions,
        or persistence failures. For non-child turns the
        parent/projection KEYS alias the completing task's own keys and are
        never read or written by the non-child script path.
        """
        clean_task = str(task_id).strip()
        clean_root = str(root_task_id).strip()
        if not clean_task or not clean_root:
            raise ValueError("task_id and root_task_id must be non-empty strings")
        if type(fencing_epoch) is bool or not isinstance(fencing_epoch, int):
            raise TypeError("fencing_epoch must be an int")
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

        is_child = command.execution_kind == "delegated_child"
        clean_parent = str(parent_task_id).strip() if parent_task_id else ""
        if is_child and not clean_parent:
            raise ValueError("parent_task_id is required for delegated_child completion")

        status_json = canonical_json_dumps({
            "status": outcome.terminal_status,
            "task_id": clean_task,
        })

        if is_child:
            latest_key = f"task.continuation_latest:{clean_parent}"
            try:
                raw_latest = await self._client.get(latest_key)
            except (NoPermissionError, AuthenticationError):
                raise
            except ResponseError as exc:
                raise CorruptAuthoritativeRecordError(
                    f"Corrupt continuation_latest at key '{latest_key}'"
                ) from exc
            if raw_latest is not None:
                latest_text = raw_latest if isinstance(raw_latest, str) else raw_latest.decode("utf-8")
                try:
                    latest_ver = int(latest_text)
                    if latest_ver < 1 or str(latest_ver) != latest_text.strip():
                        raise ValueError("non-canonical continuation_latest")
                except ValueError as exc:
                    raise CorruptAuthoritativeRecordError(
                        f"Corrupt continuation_latest at key '{latest_key}': {latest_text!r}"
                    ) from exc
                if latest_ver == 2:
                    return await self._complete_wave_child_turn(
                        clean_task=clean_task,
                        clean_parent=clean_parent,
                        clean_root=clean_root,
                        clean_principal=clean_principal,
                        clean_agent=clean_agent,
                        clean_digest=clean_digest,
                        fencing_epoch=fencing_epoch,
                        now_iso=now_iso,
                        command=command,
                        outcome=outcome,
                        result=result,
                        audit=audit,
                        record=record,
                        status_json=status_json,
                    )
                if latest_ver != 1:
                    raise CorruptAuthoritativeRecordError(
                        f"Unsupported continuation_latest '{latest_ver}' for parent '{clean_parent}'"
                    )

        if is_child:
            parent_status_key = f"task.status:{clean_parent}"
            parent_children_key = f"task.children:{clean_parent}"
            continuation_key = f"task.continuation:{clean_parent}:1"
            continuation_latest_key = f"task.continuation_latest:{clean_parent}"
            child_marker_key = f"task.child_completion:{clean_task}"
            hint_queue_key = "agent.hints.research"
            hint = WakeUpHint(
                hint_id=f"hint-{uuid4()}",
                capability="research",
                created_at=now_iso,
                execution_kind="resumed_continuation",
                task_id=clean_parent,
                expected_continuation_version=1,
            )
            hint_json = hint.to_json()
            marker_json = canonical_json_dumps({
                "child_task_id": clean_task,
                "completed_at": now_iso,
                "completion_command_id": command.completion_command_id,
                "continuation_version": 1,
                "outcome_id": outcome.outcome_id,
                "parent_task_id": clean_parent,
                "status": outcome.terminal_status,
            })
            parent_status_json = canonical_json_dumps({
                "status": "resumable",
                "task_id": clean_parent,
                "updated_at": now_iso,
            })
        else:
            parent_status_key = f"task.status:{clean_task}"
            parent_children_key = f"task.children:{clean_task}"
            continuation_key = f"task.continuation:{clean_task}:1"
            continuation_latest_key = f"task.continuation_latest:{clean_task}"
            child_marker_key = f"task.child_completion:{clean_task}"
            hint_queue_key = "agent.hints.research"
            hint_json = ""
            marker_json = ""
            parent_status_json = ""

        keys = [
            f"task.completion:{clean_task}:{command.completion_command_id}",
            f"task.status:{clean_task}",
            f"task.claim:{clean_task}",
            f"task.fence:{clean_task}",
            f"task.outcome:{clean_task}",
            f"task.result:{clean_task}",
            f"task.assignment_completion:{clean_task}:{command.assignment_id}",
            f"task.lineage:{clean_task}",
            parent_status_key,
            parent_children_key,
            continuation_key,
            continuation_latest_key,
            child_marker_key,
            hint_queue_key,
        ]

        argv = [
            clean_task,
            command.assignment_id,
            clean_principal,
            clean_agent,
            command.execution_kind,
            "" if command.continuation_version is None else str(command.continuation_version),
            str(fencing_epoch),
            command.completion_command_id,
            outcome.semantic_fingerprint,
            clean_digest,
            outcome.terminal_status,
            "" if outcome.failure_code is None else outcome.failure_code,
            outcome.to_json(),
            "" if result is None else result.to_json(),
            audit.to_json(),
            clean_parent,
            clean_root,
            record.to_json(),
            hint_json,
            marker_json,
            status_json,
            parent_status_json,
        ]

        try:
            raw_res = await self._client.eval(COMPLETE_TURN_LUA, len(keys), *keys, *argv)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during complete_turn"
            ) from exc
        return self._handle_complete_turn_response(clean_task, command, raw_res)

    async def _complete_wave_child_turn(
        self,
        *,
        clean_task: str,
        clean_parent: str,
        clean_root: str,
        clean_principal: str,
        clean_agent: str,
        clean_digest: str,
        fencing_epoch: int,
        now_iso: str,
        command: CompletionCommand,
        outcome: TerminalOutcome,
        result: TaskResult | None,
        audit: AssignmentCompletionRecord,
        record: CompletionRecord,
        status_json: str,
    ) -> tuple[TerminalOutcome, TaskResult | None, bool]:
        cont = await self.get_continuation(clean_parent, 2)
        if cont is None or not isinstance(cont, TaskContinuationV2):
            raise CorruptAuthoritativeRecordError(
                f"Missing or invalid TaskContinuationV2 for parent '{clean_parent}'"
            )
        ordered_children = cont.required_child_task_ids
        child_count = len(ordered_children)
        if not (MIN_WAVE_CHILDREN <= child_count <= MAX_WAVE_CHILDREN):
            raise CorruptAuthoritativeRecordError(
                f"TaskContinuationV2 child count {child_count} out of bounds for parent '{clean_parent}'"
            )
        wave = await self.get_accepted_delegation_wave(clean_parent, cont.wave_id)
        if wave is None:
            raise CorruptAuthoritativeRecordError(
                f"Missing AcceptedDelegationWave '{cont.wave_id}' for parent '{clean_parent}'"
            )
        accepted_wave_fp = wave.semantic_fingerprint()
        try:
            cont_base = cont.to_dict()
            cont_sat = TaskContinuationV2.from_dict({
                **cont_base,
                "state": CONTINUATION_V2_STATE_RELEASED,
                "updated_at": now_iso,
                "wave_resolution": WaveResolution.SATISFIED.value,
            })
            cont_unsat = TaskContinuationV2.from_dict({
                **cont_base,
                "state": CONTINUATION_V2_STATE_RELEASED,
                "updated_at": now_iso,
                "wave_resolution": WaveResolution.UNSATISFIABLE.value,
            })
        except (ValueError, TypeError) as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt TaskContinuationV2 for parent '{clean_parent}': {exc}"
            ) from exc

        hint = WakeUpHint(
            hint_id=f"hint-{uuid4()}",
            capability="research",
            created_at=now_iso,
            execution_kind="resumed_continuation",
            task_id=clean_parent,
            expected_continuation_version=2,
        )
        marker_json = canonical_json_dumps({
            "child_task_id": clean_task,
            "completed_at": now_iso,
            "completion_command_id": command.completion_command_id,
            "continuation_version": 2,
            "outcome_id": outcome.outcome_id,
            "parent_task_id": clean_parent,
            "status": outcome.terminal_status,
            "wave_id": cont.wave_id,
        })
        parent_status_json = canonical_json_dumps({
            "status": "resumable",
            "task_id": clean_parent,
            "updated_at": now_iso,
        })

        s1_id = ordered_children[0]
        s2_id = ordered_children[1]
        s3_id = ordered_children[2] if child_count >= 3 else ordered_children[0]
        s4_id = ordered_children[3] if child_count >= 4 else ordered_children[0]

        keys = [
            f"task.completion:{clean_task}:{command.completion_command_id}",
            f"task.status:{clean_task}",
            f"task.claim:{clean_task}",
            f"task.fence:{clean_task}",
            f"task.outcome:{clean_task}",
            f"task.result:{clean_task}",
            f"task.assignment_completion:{clean_task}:{command.assignment_id}",
            f"task.lineage:{clean_task}",
            f"task.child_completion:{clean_task}",
            f"task.status:{clean_parent}",
            f"task.children:{clean_parent}",
            f"task.lineage:{clean_parent}",
            f"task.continuation:{clean_parent}:2",
            f"task.continuation_latest:{clean_parent}",
            f"task.delegation_wave:{clean_parent}:{cont.wave_id}",
            f"task.delegation_wave_resolution:{clean_parent}:{cont.wave_id}",
            "agent.hints.research",
            f"task.status:{s1_id}",
            f"task.claim:{s1_id}",
            f"task.fence:{s1_id}",
            f"task.outcome:{s1_id}",
            f"task.result:{s1_id}",
            f"task.lineage:{s1_id}",
            f"task.required_capability:{s1_id}",
            f"task.input:{s1_id}:1",
            f"task.child_completion:{s1_id}",
            f"task.status:{s2_id}",
            f"task.claim:{s2_id}",
            f"task.fence:{s2_id}",
            f"task.outcome:{s2_id}",
            f"task.result:{s2_id}",
            f"task.lineage:{s2_id}",
            f"task.required_capability:{s2_id}",
            f"task.input:{s2_id}:1",
            f"task.child_completion:{s2_id}",
            f"task.status:{s3_id}",
            f"task.claim:{s3_id}",
            f"task.fence:{s3_id}",
            f"task.outcome:{s3_id}",
            f"task.result:{s3_id}",
            f"task.lineage:{s3_id}",
            f"task.required_capability:{s3_id}",
            f"task.input:{s3_id}:1",
            f"task.child_completion:{s3_id}",
            f"task.status:{s4_id}",
            f"task.claim:{s4_id}",
            f"task.fence:{s4_id}",
            f"task.outcome:{s4_id}",
            f"task.result:{s4_id}",
            f"task.lineage:{s4_id}",
            f"task.required_capability:{s4_id}",
            f"task.input:{s4_id}:1",
            f"task.child_completion:{s4_id}",
        ]

        argv = [
            clean_task,
            command.assignment_id,
            clean_principal,
            clean_agent,
            command.execution_kind,
            "" if command.continuation_version is None else str(command.continuation_version),
            str(fencing_epoch),
            command.completion_command_id,
            outcome.semantic_fingerprint,
            clean_digest,
            outcome.terminal_status,
            "" if outcome.failure_code is None else outcome.failure_code,
            outcome.to_json(),
            "" if result is None else result.to_json(),
            audit.to_json(),
            clean_parent,
            clean_root,
            record.to_json(),
            hint.to_json(),
            marker_json,
            status_json,
            parent_status_json,
            cont.wave_id,
            now_iso,
            str(child_count),
            accepted_wave_fp,
            cont_sat.to_json(),
            cont_unsat.to_json(),
            s1_id,
            s2_id,
            s3_id,
            s4_id,
        ]

        try:
            raw_res = await self._client.eval(
                COMPLETE_WAVE_CHILD_TURN_LUA, len(keys), *keys, *argv
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during complete_turn"
            ) from exc
        return self._handle_complete_turn_response(clean_task, command, raw_res)

    def _handle_complete_turn_response(
        self,
        clean_task: str,
        command: CompletionCommand,
        raw_res: Any,
    ) -> tuple[TerminalOutcome, TaskResult | None, bool]:
        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(f"Unexpected Redis complete_turn response: {raw_res}")

        def _s(val: object) -> str:
            return val if isinstance(val, str) else val.decode("utf-8")  # type: ignore[union-attr]

        code = _s(raw_res[0])

        if code in ("OK", "IDEMPOTENT"):
            outcome_raw = _s(raw_res[1]) if len(raw_res) > 1 else ""
            result_raw = _s(raw_res[2]) if len(raw_res) > 2 else ""
            try:
                stored_outcome = TerminalOutcome.from_json(outcome_raw)
                stored_result = TaskResult.from_json(result_raw) if result_raw else None
            except (ValueError, TypeError) as exc:
                raise CorruptAuthoritativeRecordError(
                    f"Corrupt canonical completion payload for task '{clean_task}': {code}"
                ) from exc
            if stored_outcome.task_id != clean_task:
                raise CorruptAuthoritativeRecordError(
                    f"Canonical outcome task mismatch for task '{clean_task}': {code}"
                )
            if stored_outcome.terminal_status == "completed":
                if stored_result is None:
                    raise CorruptAuthoritativeRecordError(
                        f"Canonical result missing for completed task '{clean_task}': {code}"
                    )
                if stored_result.task_id != clean_task or stored_result.status != "completed":
                    raise CorruptAuthoritativeRecordError(
                        f"Canonical result inconsistent for task '{clean_task}': {code}"
                    )
            elif stored_result is not None:
                raise CorruptAuthoritativeRecordError(
                    f"Unexpected canonical result for failed task '{clean_task}': {code}"
                )
            return (stored_outcome, stored_result, code == "IDEMPOTENT")

        if code == "COMMAND_CONFLICT":
            raise CompletionConflictError(
                f"Completion command '{command.completion_command_id}' on task '{clean_task}' "
                "conflicts with the already-accepted semantic payload"
            )

        if code in ("OUTCOME_CONFLICT", "RESULT_CONFLICT", "AUDIT_CONFLICT", "MARKER_CONFLICT"):
            raise CompletionConflictError(
                f"Conflicting canonical completion state already exists for task '{clean_task}': {code}"
            )

        if code in ("RESULT_TASK_MISMATCH", "RESULT_STATUS_MISMATCH"):
            raise ValueError(
                f"Completed TaskResult is not bound to task '{clean_task}': {code}"
            )

        if code == "NOT_WORKING":
            cur_s = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            if cur_s in ("completed", "failed"):
                raise CompletionConflictError(
                    f"Task '{clean_task}' already reached canonical terminal status '{cur_s}'"
                )
            raise TaskNotClaimableError(
                f"Task '{clean_task}' is not in an eligible working turn (status '{cur_s}')"
            )

        if code == "TASK_NOT_FOUND":
            raise TaskNotClaimableError(f"Task '{clean_task}' not found")

        if code in (
            "CLAIM_NOT_FOUND",
            "ASSIGNMENT_MISMATCH",
            "PRINCIPAL_MISMATCH",
            "BOUND_AGENT_MISMATCH",
            "TASK_MISMATCH",
            "CLAIM_NOT_ACKNOWLEDGED",
        ):
            raise ClaimNotFoundError(
                f"Active acknowledged claim for assignment '{command.assignment_id}' "
                f"on task '{clean_task}' not found: {code}"
            )

        if code == "PROOF_MISMATCH":
            raise InvalidClaimProofError("Assignment proof verification failed")

        if code in ("FENCE_NOT_FOUND", "FENCE_MISMATCH"):
            raise TaskAlreadyClaimedError(
                f"Fencing epoch mismatch for assignment '{command.assignment_id}' on task '{clean_task}': {code}"
            )

        if code in (
            "EXECUTION_KIND_MISMATCH",
            "CONTINUATION_VERSION_MISMATCH",
            "CONTINUATION_LATEST_MISMATCH",
            "CONTINUATION_STAGE_MISMATCH",
            "CONTINUATION_CHILD_MISMATCH",
            "LINEAGE_KIND_MISMATCH",
            "CHILD_LINEAGE_MISMATCH",
            "CHILD_NOT_MEMBER",
            "MISSING_CONTINUATION",
            "PARENT_STATUS_NOT_FOUND",
            "PARENT_NOT_WAITING",
            "LINEAGE_NOT_FOUND",
            "PARENT_LINEAGE_NOT_FOUND",
            "WAVE_ALREADY_RESOLVED",
            "CONTINUATION_NOT_ACTIVE",
        ):
            raise TaskNotClaimableError(
                f"Completion preconditions not satisfied for task '{clean_task}': {code}"
            )

        if code.startswith(("WRONG_KEY_TYPE_", "CORRUPT_", "MISSING_", "WAVE_CONSISTENCY_")):
            raise CorruptAuthoritativeRecordError(
                f"Authoritative record corruption detected on task '{clean_task}': {code}"
            )

        raise RuntimeError(f"Unexpected complete_turn result code: {code}")

    async def get_completion_record(
        self,
        task_id: str,
        completion_command_id: str,
    ) -> CompletionRecord | None:
        clean_task = str(task_id).strip()
        clean_cmd = str(completion_command_id).strip()
        if not clean_task or not clean_cmd:
            return None
        try:
            raw = await self._client.get(f"task.completion:{clean_task}:{clean_cmd}")
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt completion record at task.completion:{clean_task}:{clean_cmd}"
            ) from exc
        if raw is None:
            return None
        try:
            return CompletionRecord.from_json(raw)
        except (ValueError, TypeError) as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt completion record at task.completion:{clean_task}:{clean_cmd}"
            ) from exc

    async def get_terminal_outcome(self, task_id: str) -> TerminalOutcome | None:
        clean_task = str(task_id).strip()
        if not clean_task:
            return None
        try:
            raw = await self._client.get(f"task.outcome:{clean_task}")
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt terminal outcome at task.outcome:{clean_task}"
            ) from exc
        if raw is None:
            return None
        try:
            return TerminalOutcome.from_json(raw)
        except (ValueError, TypeError) as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt terminal outcome at task.outcome:{clean_task}"
            ) from exc

    async def get_canonical_result(self, task_id: str) -> TaskResult | None:
        clean_task = str(task_id).strip()
        if not clean_task:
            return None
        try:
            raw = await self._client.get(f"task.result:{clean_task}")
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt canonical result at task.result:{clean_task}"
            ) from exc
        if raw is None:
            return None
        try:
            result = TaskResult.from_json(raw)
        except (ValueError, TypeError) as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt canonical result at task.result:{clean_task}"
            ) from exc
        if result.task_id != clean_task or result.status != "completed":
            raise CorruptAuthoritativeRecordError(
                f"Canonical result at task.result:{clean_task} is not bound to a "
                "completed turn of the requested task"
            )
        return result

    async def get_assignment_completion(
        self,
        task_id: str,
        assignment_id: str,
    ) -> AssignmentCompletionRecord | None:
        clean_task = str(task_id).strip()
        clean_assign = str(assignment_id).strip()
        if not clean_task or not clean_assign:
            return None
        try:
            raw = await self._client.get(
                f"task.assignment_completion:{clean_task}:{clean_assign}"
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt assignment completion at task.assignment_completion:{clean_task}:{clean_assign}"
            ) from exc
        if raw is None:
            return None
        try:
            return AssignmentCompletionRecord.from_json(raw)
        except (ValueError, TypeError) as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt assignment completion at task.assignment_completion:{clean_task}:{clean_assign}"
            ) from exc

    async def get_delegation_proposal(
        self,
        parent_task_id: str,
        proposal_id: str,
    ) -> DelegationProposalRecord | None:
        clean_parent = str(parent_task_id).strip()
        clean_proposal_id = str(proposal_id).strip()
        if not clean_parent or not clean_proposal_id:
            return None
        target_key = f"task.delegation_proposal:{clean_parent}:{clean_proposal_id}"
        if await self._client.exists(target_key) == 0:
            return None
        key_type = await self._client.type(target_key)
        type_str = key_type.decode("utf-8") if isinstance(key_type, bytes) else str(key_type)
        if type_str != "string":
            raise CorruptAuthoritativeRecordError(
                f"Wrong Redis type '{type_str}' for delegation proposal at key '{target_key}'"
            )
        raw = await self._client.get(target_key)
        if raw is None:
            return None
        try:
            raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            return DelegationProposalRecord.from_json(raw_str)
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Failed to decode delegation proposal at key '{target_key}': {exc}"
            ) from exc

    async def materialize_claim_timeout(
        self,
        child_task_id: str,
        now: str = "",
    ) -> tuple[TerminalOutcome, bool]:
        clean_child = str(child_task_id).strip()
        if not clean_child:
            raise ValueError("child_task_id must be a non-empty string")

        child_lineage = await self.get_lineage(clean_child)
        if child_lineage is None or not child_lineage.parent_task_id:
            raise TaskNotClaimableError(f"Child task '{clean_child}' has no authoritative parent")
        clean_parent = child_lineage.parent_task_id

        deadline_key = f"task.claim_deadline:{clean_child}"
        try:
            deadline_raw = await self._client.get(deadline_key)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(f"Corrupt claim deadline at {deadline_key}: {exc}") from exc
        if not deadline_raw:
            raise TaskNotClaimableError(f"Child task '{clean_child}' has no claim deadline")
        clean_deadline = str(deadline_raw).strip()

        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )

        # Route v2 wave children through the barrier-aware atomic timeout script.
        # Catch ResponseError here so wrong-type corruption falls through to Lua validation
        # and mocked v1 gateway unit tests are unaffected.
        latest_v2 = False
        try:
            raw_latest = await self._client.get(f"task.continuation_latest:{clean_parent}")
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError:
            raw_latest = None
        if raw_latest is not None:
            latest_str = raw_latest.decode("utf-8") if isinstance(raw_latest, bytes) else str(raw_latest)
            if latest_str.strip() == "2":
                latest_v2 = True

        if latest_v2:
            return await self._materialize_wave_child_claim_timeout(
                clean_child=clean_child,
                clean_parent=clean_parent,
                clean_deadline=clean_deadline,
                now_iso=now_iso,
            )

        outcome = build_claim_timed_out_terminal_outcome(
            task_id=clean_child,
            claim_deadline=clean_deadline,
            completed_at=now_iso,
        )
        child_status_json = canonical_json_dumps({
            "claim_deadline": clean_deadline,
            "status": "claim_timed_out",
            "task_id": clean_child,
        })
        parent_status_json = canonical_json_dumps({
            "status": "resumable",
            "task_id": clean_parent,
            "updated_at": now_iso,
        })
        marker_json = canonical_json_dumps({
            "child_task_id": clean_child,
            "continuation_version": 1,
            "outcome_id": outcome.outcome_id,
            "parent_task_id": clean_parent,
            "status": "claim_timed_out",
        })
        hint = WakeUpHint(
            hint_id=f"hint-{uuid4()}",
            capability="research",
            created_at=now_iso,
            execution_kind="resumed_continuation",
            task_id=clean_parent,
            expected_continuation_version=1,
        )

        keys = [
            f"task.status:{clean_child}",
            f"task.claim:{clean_child}",
            f"task.claim_deadline:{clean_child}",
            f"task.outcome:{clean_child}",
            f"task.result:{clean_child}",
            f"task.lineage:{clean_child}",
            f"task.status:{clean_parent}",
            f"task.children:{clean_parent}",
            f"task.continuation:{clean_parent}:1",
            f"task.continuation_latest:{clean_parent}",
            f"task.child_completion:{clean_child}",
            "agent.hints.research",
        ]

        argv = [
            clean_child,
            clean_parent,
            now_iso,
            outcome.to_json(),
            child_status_json,
            parent_status_json,
            marker_json,
            hint.to_json(),
        ]

        try:
            raw_res = await self._client.eval(MATERIALIZE_CLAIM_TIMEOUT_LUA, len(keys), *keys, *argv)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during materialize_claim_timeout"
            ) from exc
        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(f"Unexpected Redis materialize_claim_timeout response: {raw_res}")

        def _s(val: object) -> str:
            return val if isinstance(val, str) else val.decode("utf-8")

        code = _s(raw_res[0])
        if code in ("OK", "IDEMPOTENT"):
            outcome_raw = _s(raw_res[1])
            return (TerminalOutcome.from_json(outcome_raw), code == "IDEMPOTENT")

        if code in ("OUTCOME_CONFLICT", "RESULT_CONFLICT", "MARKER_CONFLICT", "INCOMPLETE_REPLAY_PROJECTION"):
            raise ClaimTimeoutConflictError(
                f"Conflicting terminal completion state on task '{clean_child}': {code}"
            )

        if code == "DEADLINE_NOT_EXPIRED":
            deadline_val = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise ClaimDeadlineNotExpiredError(
                f"Claim deadline for task '{clean_child}' has not expired: deadline={deadline_val}, now={now_iso}"
            )

        if code == "CHILD_NOT_PENDING":
            cur_s = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise TaskNotClaimableError(
                f"Child task '{clean_child}' is not in 'pending' status: got '{cur_s}'"
            )

        if code == "ACTIVE_CLAIM_EXISTS":
            raise TaskAlreadyClaimedError(f"Active claim exists for child task '{clean_child}'")

        if code == "PARENT_NOT_WAITING":
            cur_s = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise TaskNotClaimableError(
                f"Parent task '{clean_parent}' is not 'waiting_for_children': got '{cur_s}'"
            )

        if code in (
            "CHILD_NOT_FOUND",
            "PARENT_NOT_FOUND",
            "CHILD_LINEAGE_NOT_FOUND",
            "PARENT_CHILDREN_NOT_FOUND",
            "MISSING_CLAIM_DEADLINE",
        ):
            raise TaskNotClaimableError(
                f"Required record not found for timeout materialization on task '{clean_child}': {code}"
            )

        if code in (
            "LINEAGE_CHILD_MISMATCH",
            "LINEAGE_PARENT_MISMATCH",
            "CHILD_NOT_MEMBER",
            "MISSING_CONTINUATION_LATEST",
            "CONTINUATION_LATEST_MISMATCH",
            "MISSING_CONTINUATION",
            "CONTINUATION_STAGE_MISMATCH",
            "CONTINUATION_CHILD_MISMATCH",
            "CONTINUATION_PARENT_MISMATCH",
            "CONTINUATION_VERSION_MISMATCH",
        ):
            raise TaskNotClaimableError(
                f"Structural validation failed during timeout materialization on task '{clean_child}': {code}"
            )

        if code.startswith(("WRONG_KEY_TYPE_", "CORRUPT_")):
            raise CorruptAuthoritativeRecordError(
                f"Authoritative record corruption detected on task '{clean_child}': {code}"
            )

        raise RuntimeError(f"Unexpected materialize_claim_timeout result code: {code}")

    async def _materialize_wave_child_claim_timeout(
        self,
        *,
        clean_child: str,
        clean_parent: str,
        clean_deadline: str,
        now_iso: str,
    ) -> tuple[TerminalOutcome, bool]:
        cont = await self.get_continuation(clean_parent, 2)
        if cont is None or not isinstance(cont, TaskContinuationV2):
            raise CorruptAuthoritativeRecordError(
                f"Missing or invalid TaskContinuationV2 for parent '{clean_parent}'"
            )
        ordered_children = cont.required_child_task_ids
        child_count = len(ordered_children)
        if not (MIN_WAVE_CHILDREN <= child_count <= MAX_WAVE_CHILDREN):
            raise CorruptAuthoritativeRecordError(
                f"TaskContinuationV2 child count {child_count} out of bounds for parent '{clean_parent}'"
            )
        wave = await self.get_accepted_delegation_wave(clean_parent, cont.wave_id)
        if wave is None:
            raise CorruptAuthoritativeRecordError(
                f"Missing AcceptedDelegationWave '{cont.wave_id}' for parent '{clean_parent}'"
            )
        accepted_wave_fp = wave.semantic_fingerprint()
        try:
            cont_base = cont.to_dict()
            cont_sat = TaskContinuationV2.from_dict({
                **cont_base,
                "state": CONTINUATION_V2_STATE_RELEASED,
                "updated_at": now_iso,
                "wave_resolution": WaveResolution.SATISFIED.value,
            })
            cont_unsat = TaskContinuationV2.from_dict({
                **cont_base,
                "state": CONTINUATION_V2_STATE_RELEASED,
                "updated_at": now_iso,
                "wave_resolution": WaveResolution.UNSATISFIABLE.value,
            })
        except (ValueError, TypeError) as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt TaskContinuationV2 for parent '{clean_parent}': {exc}"
            ) from exc

        outcome = build_claim_timed_out_terminal_outcome(
            task_id=clean_child,
            claim_deadline=clean_deadline,
            completed_at=now_iso,
        )
        child_status_json = canonical_json_dumps({
            "claim_deadline": clean_deadline,
            "status": "claim_timed_out",
            "task_id": clean_child,
        })
        parent_status_json = canonical_json_dumps({
            "status": "resumable",
            "task_id": clean_parent,
            "updated_at": now_iso,
        })
        marker_json = canonical_json_dumps({
            "child_task_id": clean_child,
            "claim_deadline": clean_deadline,
            "continuation_version": 2,
            "materialized_at": now_iso,
            "outcome_id": outcome.outcome_id,
            "parent_task_id": clean_parent,
            "status": "claim_timed_out",
            "wave_id": cont.wave_id,
        })
        hint = WakeUpHint(
            hint_id=f"hint-{uuid4()}",
            capability="research",
            created_at=now_iso,
            execution_kind="resumed_continuation",
            task_id=clean_parent,
            expected_continuation_version=2,
        )

        s1_id = ordered_children[0]
        s2_id = ordered_children[1]
        s3_id = ordered_children[2] if child_count >= 3 else ordered_children[0]
        s4_id = ordered_children[3] if child_count >= 4 else ordered_children[0]

        keys = [
            f"task.status:{clean_child}",
            f"task.claim:{clean_child}",
            f"task.claim_deadline:{clean_child}",
            f"task.outcome:{clean_child}",
            f"task.result:{clean_child}",
            f"task.lineage:{clean_child}",
            f"task.child_completion:{clean_child}",
            f"task.claim_expiry:{clean_child}",
            f"task.status:{clean_parent}",
            f"task.children:{clean_parent}",
            f"task.lineage:{clean_parent}",
            f"task.continuation:{clean_parent}:2",
            f"task.continuation_latest:{clean_parent}",
            f"task.delegation_wave:{clean_parent}:{cont.wave_id}",
            f"task.delegation_wave_resolution:{clean_parent}:{cont.wave_id}",
            "agent.hints.research",
            f"task.status:{s1_id}",
            f"task.claim:{s1_id}",
            f"task.fence:{s1_id}",
            f"task.outcome:{s1_id}",
            f"task.result:{s1_id}",
            f"task.lineage:{s1_id}",
            f"task.required_capability:{s1_id}",
            f"task.input:{s1_id}:1",
            f"task.child_completion:{s1_id}",
            f"task.claim_expiry:{s1_id}",
            f"task.status:{s2_id}",
            f"task.claim:{s2_id}",
            f"task.fence:{s2_id}",
            f"task.outcome:{s2_id}",
            f"task.result:{s2_id}",
            f"task.lineage:{s2_id}",
            f"task.required_capability:{s2_id}",
            f"task.input:{s2_id}:1",
            f"task.child_completion:{s2_id}",
            f"task.claim_expiry:{s2_id}",
            f"task.status:{s3_id}",
            f"task.claim:{s3_id}",
            f"task.fence:{s3_id}",
            f"task.outcome:{s3_id}",
            f"task.result:{s3_id}",
            f"task.lineage:{s3_id}",
            f"task.required_capability:{s3_id}",
            f"task.input:{s3_id}:1",
            f"task.child_completion:{s3_id}",
            f"task.claim_expiry:{s3_id}",
            f"task.status:{s4_id}",
            f"task.claim:{s4_id}",
            f"task.fence:{s4_id}",
            f"task.outcome:{s4_id}",
            f"task.result:{s4_id}",
            f"task.lineage:{s4_id}",
            f"task.required_capability:{s4_id}",
            f"task.input:{s4_id}:1",
            f"task.child_completion:{s4_id}",
            f"task.claim_expiry:{s4_id}",
        ]
        argv = [
            clean_child,
            clean_parent,
            cont.wave_id,
            now_iso,
            outcome.to_json(),
            child_status_json,
            parent_status_json,
            marker_json,
            hint.to_json(),
            str(child_count),
            accepted_wave_fp,
            cont_sat.to_json(),
            cont_unsat.to_json(),
            s1_id,
            s2_id,
            s3_id,
            s4_id,
        ]

        try:
            raw_res = await self._client.eval(
                MATERIALIZE_WAVE_CHILD_CLAIM_TIMEOUT_LUA, len(keys), *keys, *argv
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during materialize_claim_timeout"
            ) from exc
        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(f"Unexpected Redis materialize_claim_timeout response: {raw_res}")

        def _s(val: object) -> str:
            return val if isinstance(val, str) else val.decode("utf-8")  # type: ignore[union-attr]

        code = _s(raw_res[0])
        if code in ("OK", "IDEMPOTENT"):
            outcome_raw = _s(raw_res[1])
            return (TerminalOutcome.from_json(outcome_raw), code == "IDEMPOTENT")

        if code in (
            "OUTCOME_CONFLICT",
            "RESULT_CONFLICT",
            "MARKER_CONFLICT",
            "RECOVERY_RECORD_CONFLICT",
            "INCOMPLETE_REPLAY_PROJECTION",
        ):
            raise ClaimTimeoutConflictError(
                f"Conflicting terminal completion state on task '{clean_child}': {code}"
            )

        if code == "WAVE_ALREADY_RESOLVED":
            raise WaveAlreadyResolvedError(
                f"Delegation wave '{cont.wave_id}' on parent '{clean_parent}' is already resolved"
            )

        if code == "DEADLINE_NOT_EXPIRED":
            deadline_val = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise ClaimDeadlineNotExpiredError(
                f"Claim deadline for task '{clean_child}' has not expired: deadline={deadline_val}, now={now_iso}"
            )

        if code == "CHILD_NOT_PENDING":
            cur_s = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise TaskNotClaimableError(
                f"Child task '{clean_child}' is not in 'pending' status: got '{cur_s}'"
            )

        if code == "ACTIVE_CLAIM_EXISTS":
            raise TaskAlreadyClaimedError(f"Active claim exists for child task '{clean_child}'")

        if code == "PARENT_NOT_WAITING":
            cur_s = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise TaskNotClaimableError(
                f"Parent task '{clean_parent}' is not 'waiting_for_children': got '{cur_s}'"
            )

        if code in (
            "CHILD_NOT_FOUND",
            "PARENT_NOT_FOUND",
            "CHILD_LINEAGE_NOT_FOUND",
            "MISSING_CLAIM_DEADLINE",
            "LINEAGE_PARENT_MISMATCH",
            "CHILD_NOT_MEMBER",
            "MISSING_CONTINUATION_LATEST",
            "CONTINUATION_LATEST_MISMATCH",
            "MISSING_CONTINUATION",
            "MISSING_ACCEPTED_WAVE",
            "MISSING_REQUIRED_CAPABILITY",
            "MISSING_CHILD_INPUT",
            "FENCE_NOT_FOUND",
            "CONTINUATION_NOT_ACTIVE",
        ):
            raise TaskNotClaimableError(
                f"Wave child timeout preconditions failed on task '{clean_child}': {code}"
            )

        if code.startswith(("WRONG_KEY_TYPE_", "CORRUPT_", "WAVE_CONSISTENCY_")):
            raise CorruptAuthoritativeRecordError(
                f"Authoritative record corruption detected on task '{clean_child}': {code}"
            )

        raise RuntimeError(f"Unexpected _materialize_wave_child_claim_timeout result code: {code}")

    async def transition_expired_claim(
        self,
        task_id: str,
        now: str = "",
    ) -> tuple[ClaimExpiryRecord, bool]:
        clean_task = str(task_id).strip()
        if not clean_task:
            raise ValueError("task_id must be a non-empty string")

        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )

        claim = await self.get_active_claim(clean_task)
        if claim is None:
            existing_expiry = await self.get_claim_expiry(clean_task)
            cur_status = await self.get_status(clean_task, now=now_iso)
            if existing_expiry is not None and cur_status == "recovery_required":
                if existing_expiry.execution_kind == "delegated_child":
                    lineage = await self.get_lineage(clean_task)
                    if lineage is not None and lineage.parent_task_id:
                        latest_ver = await self.get_latest_continuation_version(lineage.parent_task_id)
                        if latest_ver == 2:
                            cont = await self.get_continuation(lineage.parent_task_id, 2)
                            if not isinstance(cont, TaskContinuationV2):
                                raise CorruptAuthoritativeRecordError(
                                    f"Missing or invalid TaskContinuationV2 for parent '{lineage.parent_task_id}'"
                                )
                            rw = await self.get_resolved_delegation_wave(lineage.parent_task_id, cont.wave_id)
                            if rw is not None:
                                raise WaveAlreadyResolvedError(
                                    f"Delegation wave '{cont.wave_id}' on parent '{lineage.parent_task_id}' is already resolved"
                                )
                return (existing_expiry, True)
            raise ClaimNotFoundError(f"Active claim for task '{clean_task}' not found")

        if claim.state != "acknowledged":
            raise TaskNotClaimableError(
                f"Claim for task '{clean_task}' is not acknowledged: state='{claim.state}'"
            )

        if not claim.execution_expires_at:
            raise TaskNotClaimableError(
                f"Claim for task '{clean_task}' has no execution_expires_at recorded"
            )

        if now_iso <= claim.execution_expires_at:
            raise TaskNotClaimableError(
                f"Execution claim for task '{clean_task}' has not expired: expires_at={claim.execution_expires_at}, now={now_iso}"
            )

        expiry_record = ClaimExpiryRecord(
            task_id=clean_task,
            claim_id=claim.claim_id,
            assignment_id=claim.assignment_id,
            principal_id=claim.runtime_principal_id,
            bound_agent_id=claim.bound_agent_id,
            execution_kind=claim.execution_kind,
            expired_fencing_epoch=claim.fencing_epoch,
            acknowledged_at=claim.acknowledged_at or now_iso,
            execution_expires_at=claim.execution_expires_at,
            expired_at=now_iso,
            prior_claimable_status=claim.prior_claimable_status,
            continuation_version=claim.continuation_version,
            reason_code="EXECUTION_CLAIM_EXPIRED",
        )
        recovery_status_json = canonical_json_dumps({
            "status": "recovery_required",
            "task_id": clean_task,
            "updated_at": now_iso,
        })

        event_payload = canonical_json_dumps({
            "agent_id": claim.bound_agent_id,
            "state": "recovery_required",
            "detail": "Execution claim expired and transitioned to recovery_required",
            "task_id": clean_task,
            "occurred_at": now_iso,
        })

        if claim.execution_kind == "delegated_child":
            lineage = await self.get_lineage(clean_task)
            if lineage is not None and lineage.parent_task_id:
                latest_ver = await self.get_latest_continuation_version(lineage.parent_task_id)
                if latest_ver == 2:
                    return await self._transition_wave_child_expired_claim(
                        clean_task=clean_task,
                        clean_parent=lineage.parent_task_id,
                        now_iso=now_iso,
                        expiry_record=expiry_record,
                        recovery_status_json=recovery_status_json,
                        event_payload=event_payload,
                    )

        keys = [
            f"task.status:{clean_task}",
            f"task.claim:{clean_task}",
            f"task.fence:{clean_task}",
            f"task.claim_expiry:{clean_task}",
            f"task.outcome:{clean_task}",
            f"task.result:{clean_task}",
            "agent.events",
        ]
        argv = [
            clean_task,
            now_iso,
            expiry_record.to_json(),
            recovery_status_json,
            event_payload,
        ]

        try:
            raw_res = await self._client.eval(TRANSITION_EXPIRED_CLAIM_LUA, len(keys), *keys, *argv)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during transition_expired_claim"
            ) from exc
        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(f"Unexpected Redis transition_expired_claim response: {raw_res}")

        def _s(val: object) -> str:
            return val if isinstance(val, str) else val.decode("utf-8")

        code = _s(raw_res[0])
        if code in ("OK", "IDEMPOTENT"):
            rec_raw = _s(raw_res[1])
            return (ClaimExpiryRecord.from_json(rec_raw), code == "IDEMPOTENT")

        if code == "ALREADY_COMPLETED":
            raise CompletionConflictError(
                f"Task '{clean_task}' already reached canonical terminal completion"
            )

        if code == "NOT_WORKING":
            cur_s = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise TaskNotClaimableError(
                f"Task '{clean_task}' is not in 'working' status: got '{cur_s}'"
            )

        if code == "CLAIM_NOT_FOUND":
            raise ClaimNotFoundError(f"Active claim for task '{clean_task}' not found")

        if code == "CLAIM_NOT_ACKNOWLEDGED":
            cur_s = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise TaskNotClaimableError(
                f"Claim for task '{clean_task}' is not acknowledged: got '{cur_s}'"
            )

        if code == "MISSING_EXECUTION_EXPIRES_AT":
            raise TaskNotClaimableError(f"Claim for task '{clean_task}' lacks execution_expires_at")

        if code == "EXECUTION_NOT_EXPIRED":
            exp_val = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise TaskNotClaimableError(
                f"Execution claim for task '{clean_task}' has not expired: expires={exp_val}, now={now_iso}"
            )

        if code == "FENCE_MISMATCH":
            raise TaskAlreadyClaimedError(
                f"Fencing epoch mismatch during expiry transition for task '{clean_task}'"
            )

        if code.startswith(("WRONG_KEY_TYPE_", "CORRUPT_")):
            raise CorruptAuthoritativeRecordError(
                f"Authoritative record corruption detected on task '{clean_task}': {code}"
            )

        raise RuntimeError(f"Unexpected transition_expired_claim result code: {code}")

    async def _transition_wave_child_expired_claim(
        self,
        *,
        clean_task: str,
        clean_parent: str,
        now_iso: str,
        expiry_record: ClaimExpiryRecord,
        recovery_status_json: str,
        event_payload: str,
    ) -> tuple[ClaimExpiryRecord, bool]:
        cont = await self.get_continuation(clean_parent, 2)
        if cont is None or not isinstance(cont, TaskContinuationV2):
            raise CorruptAuthoritativeRecordError(
                f"Missing or invalid TaskContinuationV2 for parent '{clean_parent}'"
            )

        keys = [
            f"task.status:{clean_task}",
            f"task.claim:{clean_task}",
            f"task.fence:{clean_task}",
            f"task.claim_expiry:{clean_task}",
            f"task.outcome:{clean_task}",
            f"task.result:{clean_task}",
            "agent.events",
            f"task.lineage:{clean_task}",
            f"task.child_completion:{clean_task}",
            f"task.status:{clean_parent}",
            f"task.children:{clean_parent}",
            f"task.continuation:{clean_parent}:2",
            f"task.continuation_latest:{clean_parent}",
            f"task.delegation_wave:{clean_parent}:{cont.wave_id}",
            f"task.delegation_wave_resolution:{clean_parent}:{cont.wave_id}",
        ]
        argv = [
            clean_task,
            clean_parent,
            cont.wave_id,
            now_iso,
            expiry_record.to_json(),
            recovery_status_json,
            event_payload,
        ]

        try:
            raw_res = await self._client.eval(
                TRANSITION_WAVE_CHILD_EXPIRED_CLAIM_LUA, len(keys), *keys, *argv
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during transition_expired_claim"
            ) from exc
        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(f"Unexpected Redis transition_expired_claim response: {raw_res}")

        def _s(val: object) -> str:
            return val if isinstance(val, str) else val.decode("utf-8")  # type: ignore[union-attr]

        code = _s(raw_res[0])
        if code in ("OK", "IDEMPOTENT"):
            rec_raw = _s(raw_res[1])
            return (ClaimExpiryRecord.from_json(rec_raw), code == "IDEMPOTENT")

        if code == "ALREADY_COMPLETED":
            raise CompletionConflictError(
                f"Task '{clean_task}' already reached canonical terminal completion"
            )

        if code == "WAVE_ALREADY_RESOLVED":
            raise WaveAlreadyResolvedError(
                f"Delegation wave '{cont.wave_id}' on parent '{clean_parent}' is already resolved"
            )

        if code == "NOT_WORKING":
            cur_s = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise TaskNotClaimableError(
                f"Task '{clean_task}' is not in 'working' status: got '{cur_s}'"
            )

        if code in ("CLAIM_NOT_FOUND", "ACTIVE_CLAIM_EXISTS", "EXPIRY_RECORD_MISSING"):
            raise ClaimNotFoundError(
                f"Active claim precondition failed for task '{clean_task}': {code}"
            )

        if code == "CLAIM_NOT_ACKNOWLEDGED":
            cur_s = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise TaskNotClaimableError(
                f"Claim for task '{clean_task}' is not acknowledged: got '{cur_s}'"
            )

        if code == "MISSING_EXECUTION_EXPIRES_AT":
            raise TaskNotClaimableError(f"Claim for task '{clean_task}' lacks execution_expires_at")

        if code == "EXECUTION_NOT_EXPIRED":
            exp_val = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise TaskNotClaimableError(
                f"Execution claim for task '{clean_task}' has not expired: expires={exp_val}, now={now_iso}"
            )

        if code == "FENCE_MISMATCH":
            raise TaskAlreadyClaimedError(
                f"Fencing epoch mismatch during expiry transition for task '{clean_task}'"
            )

        if code in (
            "NOT_FOUND",
            "LINEAGE_NOT_FOUND",
            "CHILD_LINEAGE_MISMATCH",
            "CHILD_NOT_MEMBER",
            "CONTINUATION_LATEST_MISMATCH",
            "MISSING_CONTINUATION",
            "MISSING_ACCEPTED_WAVE",
            "PARENT_NOT_FOUND",
            "PARENT_NOT_WAITING",
            "CONTINUATION_NOT_ACTIVE",
            "FENCE_NOT_FOUND",
        ):
            raise TaskNotClaimableError(
                f"Wave child expiry transition preconditions failed on task '{clean_task}': {code}"
            )

        if code.startswith(("WRONG_KEY_TYPE_", "CORRUPT_")):
            raise CorruptAuthoritativeRecordError(
                f"Authoritative record corruption detected on task '{clean_task}': {code}"
            )

        raise RuntimeError(f"Unexpected _transition_wave_child_expired_claim result code: {code}")

    async def resolve_expired_claim(
        self,
        command: RecoveryDecisionCommand,
        operator_principal_id: str,
        now: str = "",
    ) -> tuple[RecoveryDecisionRecord, bool]:
        clean_task = str(command.task_id).strip()
        clean_dec_id = str(command.recovery_decision_id).strip()
        clean_operator = str(operator_principal_id).strip()
        if not clean_task or not clean_dec_id or not clean_operator:
            raise ValueError(
                "task_id, recovery_decision_id, and operator_principal_id must be non-empty strings"
            )

        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )

        fp = command.semantic_fingerprint()

        expiry_rec = await self.get_claim_expiry(clean_task)
        if expiry_rec is None:
            raise RecoveryPreconditionFailedError(
                f"ClaimExpiryRecord for task '{clean_task}' not found"
            )

        restored_status = expiry_rec.prior_claimable_status
        decision_rec = RecoveryDecisionRecord(
            recovery_decision_id=clean_dec_id,
            task_id=clean_task,
            expired_claim_id=command.expired_claim_id,
            expired_assignment_id=command.expired_assignment_id,
            expired_fencing_epoch=command.expired_fencing_epoch,
            disposition=command.disposition,
            semantic_fingerprint=fp,
            prior_status="recovery_required",
            restored_status=restored_status,
            operator_principal_id=clean_operator,
            resolved_at=now_iso,
        )

        restored_status_json = canonical_json_dumps({
            "status": restored_status,
            "task_id": clean_task,
            "updated_at": now_iso,
        })

        if expiry_rec.execution_kind == "delegated_child":
            lineage = await self.get_lineage(clean_task)
            if lineage is not None and lineage.parent_task_id:
                latest_ver = await self.get_latest_continuation_version(lineage.parent_task_id)
                if latest_ver == 2:
                    return await self._resolve_wave_child_expired_claim(
                        command=command,
                        clean_task=clean_task,
                        clean_parent=lineage.parent_task_id,
                        clean_dec_id=clean_dec_id,
                        decision_rec=decision_rec,
                        fp=fp,
                        restored_status_json=restored_status_json,
                    )

        keys = [
            f"task.status:{clean_task}",
            f"task.claim:{clean_task}",
            f"task.fence:{clean_task}",
            f"task.claim_expiry:{clean_task}",
            f"task.recovery_decision:{clean_task}:{clean_dec_id}",
            f"task.outcome:{clean_task}",
        ]
        argv = [
            clean_task,
            clean_dec_id,
            command.expired_claim_id,
            command.expired_assignment_id,
            str(command.expired_fencing_epoch),
            command.disposition,
            decision_rec.to_json(),
            fp,
            restored_status_json,
        ]

        try:
            raw_res = await self._client.eval(RESOLVE_EXPIRED_CLAIM_LUA, len(keys), *keys, *argv)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during resolve_expired_claim"
            ) from exc
        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(f"Unexpected Redis resolve_expired_claim response: {raw_res}")

        def _s(val: object) -> str:
            return val if isinstance(val, str) else val.decode("utf-8")

        code = _s(raw_res[0])
        if code in ("OK", "IDEMPOTENT"):
            rec_raw = _s(raw_res[1])
            return (RecoveryDecisionRecord.from_json(rec_raw), code == "IDEMPOTENT")

        if code == "DECISION_CONFLICT":
            raise RecoveryConflictError(
                f"Recovery decision '{clean_dec_id}' on task '{clean_task}' conflicts with existing decision payload"
            )

        if code == "ALREADY_COMPLETED":
            raise CompletionConflictError(
                f"Task '{clean_task}' already reached canonical terminal completion"
            )

        if code == "NOT_RECOVERY_REQUIRED":
            cur_s = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise RecoveryPreconditionFailedError(
                f"Task '{clean_task}' is not in 'recovery_required' status: got '{cur_s}'"
            )

        if code == "ACTIVE_CLAIM_EXISTS":
            raise RecoveryPreconditionFailedError(
                f"Active claim unexpectedly present on task '{clean_task}'"
            )

        if code in ("EXPIRY_RECORD_NOT_FOUND", "EXPIRY_RECORD_MISMATCH"):
            raise RecoveryPreconditionFailedError(
                f"ClaimExpiryRecord validation failed for task '{clean_task}': {code}"
            )

        if code == "UNSUPPORTED_DISPOSITION":
            disp = _s(raw_res[1]) if len(raw_res) > 1 else command.disposition
            raise RecoveryPreconditionFailedError(f"Unsupported recovery disposition: '{disp}'")

        if code == "FENCE_MISMATCH":
            raise RecoveryPreconditionFailedError(
                f"Fencing epoch mismatch during recovery resolution for task '{clean_task}'"
            )

        if code.startswith(("WRONG_KEY_TYPE_", "CORRUPT_")):
            raise CorruptAuthoritativeRecordError(
                f"Authoritative record corruption detected on task '{clean_task}': {code}"
            )

        raise RuntimeError(f"Unexpected resolve_expired_claim result code: {code}")

    async def _resolve_wave_child_expired_claim(
        self,
        *,
        command: RecoveryDecisionCommand,
        clean_task: str,
        clean_parent: str,
        clean_dec_id: str,
        decision_rec: RecoveryDecisionRecord,
        fp: str,
        restored_status_json: str,
    ) -> tuple[RecoveryDecisionRecord, bool]:
        cont = await self.get_continuation(clean_parent, 2)
        if cont is None or not isinstance(cont, TaskContinuationV2):
            raise CorruptAuthoritativeRecordError(
                f"Missing or invalid TaskContinuationV2 for parent '{clean_parent}'"
            )

        keys = [
            f"task.status:{clean_task}",
            f"task.claim:{clean_task}",
            f"task.fence:{clean_task}",
            f"task.claim_expiry:{clean_task}",
            f"task.recovery_decision:{clean_task}:{clean_dec_id}",
            f"task.outcome:{clean_task}",
            f"task.lineage:{clean_task}",
            f"task.child_completion:{clean_task}",
            f"task.status:{clean_parent}",
            f"task.children:{clean_parent}",
            f"task.continuation:{clean_parent}:2",
            f"task.continuation_latest:{clean_parent}",
            f"task.delegation_wave:{clean_parent}:{cont.wave_id}",
            f"task.delegation_wave_resolution:{clean_parent}:{cont.wave_id}",
        ]
        argv = [
            clean_task,
            clean_parent,
            cont.wave_id,
            clean_dec_id,
            command.expired_claim_id,
            command.expired_assignment_id,
            str(command.expired_fencing_epoch),
            command.disposition,
            decision_rec.to_json(),
            fp,
            restored_status_json,
        ]

        try:
            raw_res = await self._client.eval(
                RESOLVE_WAVE_CHILD_EXPIRED_CLAIM_LUA, len(keys), *keys, *argv
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during resolve_expired_claim"
            ) from exc
        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(f"Unexpected Redis resolve_expired_claim response: {raw_res}")

        def _s(val: object) -> str:
            return val if isinstance(val, str) else val.decode("utf-8")  # type: ignore[union-attr]

        code = _s(raw_res[0])
        if code in ("OK", "IDEMPOTENT"):
            rec_raw = _s(raw_res[1])
            return (RecoveryDecisionRecord.from_json(rec_raw), code == "IDEMPOTENT")

        if code == "DECISION_CONFLICT":
            raise RecoveryConflictError(
                f"Recovery decision '{clean_dec_id}' on task '{clean_task}' conflicts with existing decision payload"
            )

        if code == "ALREADY_COMPLETED":
            raise CompletionConflictError(
                f"Task '{clean_task}' already reached canonical terminal completion"
            )

        if code == "NOT_RECOVERY_REQUIRED":
            cur_s = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise RecoveryPreconditionFailedError(
                f"Task '{clean_task}' is not in 'recovery_required' status: got '{cur_s}'"
            )

        if code in (
            "ACTIVE_CLAIM_EXISTS",
            "EXPIRY_RECORD_NOT_FOUND",
            "EXPIRY_RECORD_MISMATCH",
            "FENCE_MISMATCH",
            "FENCE_NOT_FOUND",
            "NOT_FOUND",
            "LINEAGE_NOT_FOUND",
            "CHILD_LINEAGE_MISMATCH",
            "CHILD_NOT_MEMBER",
            "CONTINUATION_LATEST_MISMATCH",
            "MISSING_CONTINUATION",
            "MISSING_ACCEPTED_WAVE",
            "PARENT_NOT_FOUND",
            "WAVE_ALREADY_RESOLVED",
            "PARENT_NOT_WAITING",
            "CONTINUATION_NOT_ACTIVE",
        ):
            raise RecoveryPreconditionFailedError(
                f"Wave child recovery preconditions failed for task '{clean_task}': {code}"
            )

        if code == "UNSUPPORTED_DISPOSITION":
            disp = _s(raw_res[1]) if len(raw_res) > 1 else command.disposition
            raise RecoveryPreconditionFailedError(f"Unsupported recovery disposition: '{disp}'")

        if code.startswith(("WRONG_KEY_TYPE_", "CORRUPT_")):
            raise CorruptAuthoritativeRecordError(
                f"Authoritative record corruption detected on task '{clean_task}': {code}"
            )

        raise RuntimeError(f"Unexpected _resolve_wave_child_expired_claim result code: {code}")

    async def get_wave_cancellation_record(
        self,
        parent_task_id: str,
        cancellation_decision_id: str,
    ) -> WaveCancellationRecord | None:
        clean_parent = str(parent_task_id).strip()
        clean_dec = str(cancellation_decision_id).strip()
        if not clean_parent or not clean_dec:
            return None
        target_key = f"task.delegation_wave_cancellation:{clean_parent}:{clean_dec}"
        try:
            raw = await self._client.get(target_key)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(f"Corrupt record at {target_key}") from exc
        if raw is None:
            return None
        try:
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            return WaveCancellationRecord.from_json(text)
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Failed to decode WaveCancellationRecord at {target_key}"
            ) from exc

    async def read_resumed_continuation_v2_context(
        self,
        parent_task_id: str,
        assignment_id: str,
        principal_id: str,
        bound_agent_id: str,
        expected_fencing_epoch: int,
    ) -> ResumedContinuationContext:
        clean_parent = str(parent_task_id).strip()
        clean_assignment = str(assignment_id).strip()
        clean_principal = str(principal_id).strip()
        clean_agent = str(bound_agent_id).strip()
        if (
            not clean_parent
            or not clean_assignment
            or not clean_principal
            or not clean_agent
            or type(expected_fencing_epoch) is bool
            or not isinstance(expected_fencing_epoch, int)
            or expected_fencing_epoch < 1
        ):
            raise ValueError("Invalid parameters for read_resumed_continuation_v2_context")

        # Best-effort peek so static KEYS bind the wave, input, and primary artifact keys;
        # all authoritative validations and cross-record checks run atomically inside Lua.
        peek_wave_id = "_missing"
        peek_input_version = 1
        peek_art_id = "_missing"
        peek_art_ver = 1
        try:
            cont_raw = await self._client.get(f"task.continuation:{clean_parent}:2")
            if cont_raw is not None:
                text = cont_raw if isinstance(cont_raw, str) else cont_raw.decode("utf-8")
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    wid = parsed.get("wave_id")
                    if isinstance(wid, str) and wid.strip():
                        peek_wave_id = wid.strip()
                    iv = parsed.get("input_version")
                    if isinstance(iv, int) and type(iv) is not bool and iv >= 1:
                        peek_input_version = iv
                    refs = parsed.get("artifact_references")
                    if isinstance(refs, list) and refs and isinstance(refs[0], dict):
                        aid = refs[0].get("artifact_id")
                        av = refs[0].get("artifact_version")
                        if isinstance(aid, str) and aid.strip():
                            peek_art_id = aid.strip()
                        if isinstance(av, int) and type(av) is not bool and av >= 1:
                            peek_art_ver = av
        except (NoPermissionError, AuthenticationError):
            raise
        except (ResponseError, ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
            peek_wave_id = "_missing"

        keys = [
            f"task.status:{clean_parent}",
            f"task.claim:{clean_parent}",
            f"task.fence:{clean_parent}",
            f"task.lineage:{clean_parent}",
            f"task.continuation_latest:{clean_parent}",
            f"task.continuation:{clean_parent}:2",
            f"task.delegation_wave:{clean_parent}:{peek_wave_id}",
            f"task.delegation_wave_resolution:{clean_parent}:{peek_wave_id}",
            f"task.input:{clean_parent}:{peek_input_version}",
            f"task.artifact:{clean_parent}:{peek_art_id}:{peek_art_ver}",
        ]
        argv = [
            clean_parent,
            clean_assignment,
            clean_principal,
            clean_agent,
            str(expected_fencing_epoch),
            peek_wave_id,
            str(peek_input_version),
            peek_art_id,
            str(peek_art_ver),
        ]

        try:
            raw_res = await self._client.eval(
                READ_RESUMED_CONTINUATION_V2_LUA, len(keys), *keys, *argv
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during read_resumed_continuation_v2_context"
            ) from exc
        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(
                f"Unexpected Redis read_resumed_continuation_v2_context response: {raw_res}"
            )

        def _s(val: object) -> str:
            return val if isinstance(val, str) else val.decode("utf-8")  # type: ignore[union-attr]

        code = _s(raw_res[0])
        if code == "OK":
            if len(raw_res) < 6:
                raise CorruptAuthoritativeRecordError(
                    f"Incomplete v2 continuation context snapshot for parent '{clean_parent}'"
                )
            try:
                root_id = _s(raw_res[1])
                parent_input = TaskInput.from_json(_s(raw_res[2]))
                continuation = TaskContinuationV2.from_json(_s(raw_res[3]))
                resolved_wave = ResolvedDelegationWave.from_json(_s(raw_res[4]))
                stage_artifact = StageArtifact.from_json(_s(raw_res[5]))
                return ResumedContinuationContext(
                    parent_task_input=parent_input,
                    continuation=continuation,
                    stage_artifact=stage_artifact,
                    resolved_wave=resolved_wave,
                    root_task_id=root_id,
                    assignment_id=clean_assignment,
                    fencing_epoch=expected_fencing_epoch,
                    version=2,
                )
            except (ValueError, TypeError) as exc:
                raise CorruptAuthoritativeRecordError(
                    f"Corrupt v2 continuation context payload for parent '{clean_parent}': {exc}"
                ) from exc

        if code == "CLAIM_NOT_FOUND":
            raise ClaimNotFoundError(
                f"Active claim for task '{clean_parent}' and assignment '{clean_assignment}' not found"
            )
        if code == "CAPABILITY_MISMATCH":
            raise CapabilityMismatchError("Claim assigned_capability is not 'research'")
        if code == "FENCE_MISMATCH":
            raise TaskAlreadyClaimedError(
                f"Fencing epoch mismatch for assignment '{clean_assignment}' on task '{clean_parent}'"
            )
        if code in (
            "CLAIM_NOT_ACKNOWLEDGED",
            "EXECUTION_KIND_MISMATCH",
            "CONTINUATION_VERSION_MISMATCH",
            "PARENT_NOT_WORKING",
            "CONTINUATION_LATEST_MISMATCH",
            "CONTINUATION_CLOSED",
            "CONTINUATION_NOT_RELEASED",
            "UNSUPPORTED_WAVE_RESOLUTION",
        ):
            raise TaskNotClaimableError(
                f"Resumed v2 continuation context unavailable for task '{clean_parent}': {code}"
            )
        if code.startswith(("WRONG_KEY_TYPE_", "CORRUPT_")):
            raise CorruptAuthoritativeRecordError(
                f"Authoritative record corruption detected on task '{clean_parent}': {code}"
            )

        raise RuntimeError(f"Unexpected read_resumed_continuation_v2_context result code: {code}")

    async def evaluate_wave_timeout(
        self,
        command: WaveTimeoutEvaluationCommand,
        now: str = "",
    ) -> tuple[ResolvedDelegationWave, bool]:
        if not isinstance(command, WaveTimeoutEvaluationCommand):
            raise TypeError(
                f"command must be a WaveTimeoutEvaluationCommand, got {type(command).__name__}"
            )
        clean_parent = command.parent_task_id
        clean_wave = command.wave_id
        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )

        parent_lineage = await self.get_lineage(clean_parent)
        if parent_lineage is None:
            raise ParentTaskNotFoundError(f"Parent task '{clean_parent}' not found")

        cont = await self.get_continuation(clean_parent, 2)
        if cont is None or not isinstance(cont, TaskContinuationV2):
            raise CorruptAuthoritativeRecordError(
                f"Missing or invalid TaskContinuationV2 for parent '{clean_parent}'"
            )
        if cont.wave_id != clean_wave:
            raise DelegationConflictError(
                f"Parent '{clean_parent}' active continuation binds to wave '{cont.wave_id}', not '{clean_wave}'"
            )
        ordered_children = cont.required_child_task_ids
        child_count = len(ordered_children)
        if not (MIN_WAVE_CHILDREN <= child_count <= MAX_WAVE_CHILDREN):
            raise CorruptAuthoritativeRecordError(
                f"TaskContinuationV2 child count {child_count} out of bounds for parent '{clean_parent}'"
            )
        wave = await self.get_accepted_delegation_wave(clean_parent, clean_wave)
        if wave is None:
            raise CorruptAuthoritativeRecordError(
                f"Missing AcceptedDelegationWave '{clean_wave}' for parent '{clean_parent}'"
            )
        accepted_wave_fp = wave.semantic_fingerprint()
        try:
            cont_base = cont.to_dict()
            cont_to = TaskContinuationV2.from_dict({
                **cont_base,
                "state": CONTINUATION_V2_STATE_RELEASED,
                "updated_at": now_iso,
                "wave_resolution": WaveResolution.TIMED_OUT.value,
            })
        except (ValueError, TypeError) as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt TaskContinuationV2 for parent '{clean_parent}': {exc}"
            ) from exc

        parent_status_json = canonical_json_dumps({
            "status": "resumable",
            "task_id": clean_parent,
            "updated_at": now_iso,
        })
        hint = WakeUpHint(
            hint_id=f"hint-{uuid4()}",
            capability="research",
            created_at=now_iso,
            execution_kind="resumed_continuation",
            task_id=clean_parent,
            expected_continuation_version=2,
        )

        s1_id = ordered_children[0]
        s2_id = ordered_children[1]
        s3_id = ordered_children[2] if child_count >= 3 else ordered_children[0]
        s4_id = ordered_children[3] if child_count >= 4 else ordered_children[0]

        keys = [
            f"task.status:{clean_parent}",
            f"task.children:{clean_parent}",
            f"task.lineage:{clean_parent}",
            f"task.continuation:{clean_parent}:2",
            f"task.continuation_latest:{clean_parent}",
            f"task.delegation_wave:{clean_parent}:{clean_wave}",
            f"task.delegation_wave_resolution:{clean_parent}:{clean_wave}",
            "agent.hints.research",
            f"task.status:{s1_id}",
            f"task.claim:{s1_id}",
            f"task.fence:{s1_id}",
            f"task.outcome:{s1_id}",
            f"task.result:{s1_id}",
            f"task.lineage:{s1_id}",
            f"task.required_capability:{s1_id}",
            f"task.input:{s1_id}:1",
            f"task.child_completion:{s1_id}",
            f"task.claim_expiry:{s1_id}",
            f"task.status:{s2_id}",
            f"task.claim:{s2_id}",
            f"task.fence:{s2_id}",
            f"task.outcome:{s2_id}",
            f"task.result:{s2_id}",
            f"task.lineage:{s2_id}",
            f"task.required_capability:{s2_id}",
            f"task.input:{s2_id}:1",
            f"task.child_completion:{s2_id}",
            f"task.claim_expiry:{s2_id}",
            f"task.status:{s3_id}",
            f"task.claim:{s3_id}",
            f"task.fence:{s3_id}",
            f"task.outcome:{s3_id}",
            f"task.result:{s3_id}",
            f"task.lineage:{s3_id}",
            f"task.required_capability:{s3_id}",
            f"task.input:{s3_id}:1",
            f"task.child_completion:{s3_id}",
            f"task.claim_expiry:{s3_id}",
            f"task.status:{s4_id}",
            f"task.claim:{s4_id}",
            f"task.fence:{s4_id}",
            f"task.outcome:{s4_id}",
            f"task.result:{s4_id}",
            f"task.lineage:{s4_id}",
            f"task.required_capability:{s4_id}",
            f"task.input:{s4_id}:1",
            f"task.child_completion:{s4_id}",
            f"task.claim_expiry:{s4_id}",
        ]
        argv = [
            clean_parent,
            clean_wave,
            now_iso,
            str(child_count),
            accepted_wave_fp,
            cont_to.to_json(),
            parent_status_json,
            hint.to_json(),
            s1_id,
            s2_id,
            s3_id,
            s4_id,
        ]

        try:
            raw_res = await self._client.eval(
                EVALUATE_WAVE_TIMEOUT_LUA, len(keys), *keys, *argv
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during evaluate_wave_timeout"
            ) from exc
        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(f"Unexpected Redis evaluate_wave_timeout response: {raw_res}")

        def _s(val: object) -> str:
            return val if isinstance(val, str) else val.decode("utf-8")  # type: ignore[union-attr]

        code = _s(raw_res[0])
        if code in ("OK", "IDEMPOTENT"):
            rw_raw = _s(raw_res[1])
            return (ResolvedDelegationWave.from_json(rw_raw), code == "IDEMPOTENT")

        if code == "PARENT_NOT_FOUND":
            raise ParentTaskNotFoundError(f"Parent task '{clean_parent}' not found")

        if code == "WAVE_DEADLINE_NOT_REACHED":
            deadline_val = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise WaveDeadlineNotReachedError(
                f"Wave '{clean_wave}' deadline has not been reached: deadline={deadline_val}, now={now_iso}"
            )

        if code == "WAVE_ALREADY_RESOLVED":
            existing_res = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise WaveAlreadyResolvedError(
                f"Delegation wave '{clean_wave}' on parent '{clean_parent}' is already resolved ({existing_res})"
            )

        if code in ("PARENT_NOT_WAITING", "CONTINUATION_NOT_ACTIVE"):
            raise TaskNotClaimableError(
                f"Delegation wave '{clean_wave}' preconditions not met on parent '{clean_parent}': {code}"
            )

        if code.startswith(("WRONG_KEY_TYPE_", "CORRUPT_", "MISSING_", "WAVE_CONSISTENCY_", "CHILD_", "LINEAGE_", "FENCE_")) or code in ("TASK_NOT_FOUND", "CONTINUATION_LATEST_MISMATCH"):
            raise CorruptAuthoritativeRecordError(
                f"Authoritative record validation failed during evaluate_wave_timeout on parent '{clean_parent}': {code}"
            )

        raise RuntimeError(f"Unexpected evaluate_wave_timeout result code: {code}")

    async def cancel_delegation_wave(
        self,
        command: WaveCancellationCommand,
        operator_principal_id: str,
        now: str = "",
    ) -> tuple[WaveCancellationRecord, ResolvedDelegationWave, bool]:
        if not isinstance(command, WaveCancellationCommand):
            raise TypeError(
                f"command must be a WaveCancellationCommand, got {type(command).__name__}"
            )
        clean_parent = command.parent_task_id
        clean_wave = command.wave_id
        clean_dec_id = command.cancellation_decision_id
        clean_operator = str(operator_principal_id).strip()
        if not clean_operator:
            raise ValueError("operator_principal_id must be a non-empty string")
        now_iso = (
            normalize_canonical_utc_timestamp(now)
            if (now and str(now).strip())
            else normalize_canonical_utc_timestamp(datetime.now(UTC))
        )

        parent_lineage = await self.get_lineage(clean_parent)
        if parent_lineage is None:
            raise ParentTaskNotFoundError(f"Parent task '{clean_parent}' not found")

        cont = await self.get_continuation(clean_parent, 2)
        if cont is None or not isinstance(cont, TaskContinuationV2):
            raise CorruptAuthoritativeRecordError(
                f"Missing or invalid TaskContinuationV2 for parent '{clean_parent}'"
            )
        if cont.wave_id != clean_wave:
            raise DelegationConflictError(
                f"Parent '{clean_parent}' continuation binds to wave '{cont.wave_id}', not '{clean_wave}'"
            )
        ordered_children = cont.required_child_task_ids
        child_count = len(ordered_children)
        if not (MIN_WAVE_CHILDREN <= child_count <= MAX_WAVE_CHILDREN):
            raise CorruptAuthoritativeRecordError(
                f"TaskContinuationV2 child count {child_count} out of bounds for parent '{clean_parent}'"
            )
        wave = await self.get_accepted_delegation_wave(clean_parent, clean_wave)
        if wave is None:
            raise CorruptAuthoritativeRecordError(
                f"Missing AcceptedDelegationWave '{clean_wave}' for parent '{clean_parent}'"
            )
        accepted_wave_fp = wave.semantic_fingerprint()
        fp = command.semantic_fingerprint()

        dec_rec = WaveCancellationRecord(
            cancellation_decision_id=clean_dec_id,
            parent_task_id=clean_parent,
            wave_id=clean_wave,
            operator_principal_id=clean_operator,
            semantic_fingerprint=fp,
            cancelled_at=now_iso,
        )
        try:
            cont_base = cont.to_dict()
            cont_closed = TaskContinuationV2.from_dict({
                **cont_base,
                "state": CONTINUATION_V2_STATE_CLOSED,
                "updated_at": now_iso,
                "wave_resolution": None,
            })
        except (ValueError, TypeError) as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt TaskContinuationV2 for parent '{clean_parent}': {exc}"
            ) from exc

        parent_cancelled_status_json = canonical_json_dumps({
            "status": TaskLifecycleStatus.CANCELLED.value,
            "task_id": clean_parent,
            "updated_at": now_iso,
        })

        s1_id = ordered_children[0]
        s2_id = ordered_children[1]
        s3_id = ordered_children[2] if child_count >= 3 else ordered_children[0]
        s4_id = ordered_children[3] if child_count >= 4 else ordered_children[0]

        keys = [
            f"task.delegation_wave_cancellation:{clean_parent}:{clean_dec_id}",
            f"task.status:{clean_parent}",
            f"task.children:{clean_parent}",
            f"task.lineage:{clean_parent}",
            f"task.continuation:{clean_parent}:2",
            f"task.continuation_latest:{clean_parent}",
            f"task.delegation_wave:{clean_parent}:{clean_wave}",
            f"task.delegation_wave_resolution:{clean_parent}:{clean_wave}",
            f"task.status:{s1_id}",
            f"task.claim:{s1_id}",
            f"task.fence:{s1_id}",
            f"task.outcome:{s1_id}",
            f"task.result:{s1_id}",
            f"task.lineage:{s1_id}",
            f"task.required_capability:{s1_id}",
            f"task.input:{s1_id}:1",
            f"task.child_completion:{s1_id}",
            f"task.claim_expiry:{s1_id}",
            f"task.status:{s2_id}",
            f"task.claim:{s2_id}",
            f"task.fence:{s2_id}",
            f"task.outcome:{s2_id}",
            f"task.result:{s2_id}",
            f"task.lineage:{s2_id}",
            f"task.required_capability:{s2_id}",
            f"task.input:{s2_id}:1",
            f"task.child_completion:{s2_id}",
            f"task.claim_expiry:{s2_id}",
            f"task.status:{s3_id}",
            f"task.claim:{s3_id}",
            f"task.fence:{s3_id}",
            f"task.outcome:{s3_id}",
            f"task.result:{s3_id}",
            f"task.lineage:{s3_id}",
            f"task.required_capability:{s3_id}",
            f"task.input:{s3_id}:1",
            f"task.child_completion:{s3_id}",
            f"task.claim_expiry:{s3_id}",
            f"task.status:{s4_id}",
            f"task.claim:{s4_id}",
            f"task.fence:{s4_id}",
            f"task.outcome:{s4_id}",
            f"task.result:{s4_id}",
            f"task.lineage:{s4_id}",
            f"task.required_capability:{s4_id}",
            f"task.input:{s4_id}:1",
            f"task.child_completion:{s4_id}",
            f"task.claim_expiry:{s4_id}",
        ]
        argv = [
            clean_parent,
            clean_wave,
            clean_dec_id,
            clean_operator,
            fp,
            now_iso,
            str(child_count),
            accepted_wave_fp,
            dec_rec.to_json(),
            cont_closed.to_json(),
            parent_cancelled_status_json,
            s1_id,
            s2_id,
            s3_id,
            s4_id,
        ]

        try:
            raw_res = await self._client.eval(
                CANCEL_DELEGATION_WAVE_LUA, len(keys), *keys, *argv
            )
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                "Redis command failed during cancel_delegation_wave"
            ) from exc
        if not isinstance(raw_res, (list, tuple)) or len(raw_res) == 0:
            raise RuntimeError(f"Unexpected Redis cancel_delegation_wave response: {raw_res}")

        def _s(val: object) -> str:
            return val if isinstance(val, str) else val.decode("utf-8")  # type: ignore[union-attr]

        code = _s(raw_res[0])
        if code in ("OK", "IDEMPOTENT"):
            dec_raw = _s(raw_res[1])
            rw_raw = _s(raw_res[2])
            return (
                WaveCancellationRecord.from_json(dec_raw),
                ResolvedDelegationWave.from_json(rw_raw),
                code == "IDEMPOTENT",
            )

        if code == "CANCELLATION_CONFLICT":
            raise WaveCancellationConflictError(
                f"Cancellation decision '{clean_dec_id}' on parent '{clean_parent}' conflicts with existing cancellation"
            )

        if code == "WAVE_ALREADY_RESOLVED":
            existing_res = _s(raw_res[1]) if len(raw_res) > 1 else "unknown"
            raise WaveAlreadyResolvedError(
                f"Delegation wave '{clean_wave}' on parent '{clean_parent}' is already resolved ({existing_res})"
            )

        if code == "PARENT_NOT_FOUND":
            raise ParentTaskNotFoundError(f"Parent task '{clean_parent}' not found")

        if code in ("PARENT_NOT_WAITING", "CONTINUATION_NOT_ACTIVE"):
            raise TaskNotClaimableError(
                f"Delegation wave '{clean_wave}' cancellation preconditions not met on parent '{clean_parent}': {code}"
            )

        if code.startswith(("WRONG_KEY_TYPE_", "CORRUPT_", "MISSING_", "WAVE_CONSISTENCY_", "CHILD_", "LINEAGE_", "FENCE_")) or code in ("TASK_NOT_FOUND", "CONTINUATION_LATEST_MISMATCH"):
            raise CorruptAuthoritativeRecordError(
                f"Authoritative record validation failed during cancel_delegation_wave on parent '{clean_parent}': {code}"
            )

        raise RuntimeError(f"Unexpected cancel_delegation_wave result code: {code}")

    async def get_claim_expiry(self, task_id: str) -> ClaimExpiryRecord | None:
        clean_task = str(task_id).strip()
        if not clean_task:
            return None
        target_key = f"task.claim_expiry:{clean_task}"
        try:
            raw = await self._client.get(target_key)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(f"Corrupt record at {target_key}") from exc
        if raw is None:
            return None
        try:
            return ClaimExpiryRecord.from_json(raw)
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Failed to decode ClaimExpiryRecord at {target_key}"
            ) from exc

    async def get_recovery_decision(
        self, task_id: str, recovery_decision_id: str
    ) -> RecoveryDecisionRecord | None:
        clean_task = str(task_id).strip()
        clean_dec = str(recovery_decision_id).strip()
        if not clean_task or not clean_dec:
            return None
        target_key = f"task.recovery_decision:{clean_task}:{clean_dec}"
        try:
            raw = await self._client.get(target_key)
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(f"Corrupt record at {target_key}") from exc
        if raw is None:
            return None
        try:
            return RecoveryDecisionRecord.from_json(raw)
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Failed to decode RecoveryDecisionRecord at {target_key}"
            ) from exc

    async def dispatch_hint(
        self,
        hint: WakeUpHint,
        target: str | None = None,
        capability: str | None = None,
    ) -> None:
        """Publish a wake-up hint to the governed finite hint destination.

        Authorization binds to the authoritative Orchestrator-supplied
        capability, never to a runtime-self-advertised capability or to the
        untrusted target text. The target is validated before normalization
        by exact raw equality against the canonical approved hint
        destination, and only the canonical constant reaches RPUSH. An
        omitted (``None``) target resolves from the authoritative
        capability alone. No Redis write occurs unless validation succeeds.
        """
        if capability is None or not isinstance(capability, str):
            raise ValueError("Authoritative hint capability is required")
        if capability not in self._capability_hint_contract:
            raise ValueError("Unrecognized authoritative hint capability")
        approved_target = self._capability_hint_contract[capability]
        if target is None:
            resolved_target = approved_target
        else:
            if not isinstance(target, str):
                raise ValueError("Hint target must be a string or omitted")
            if target != approved_target:
                raise ValueError("Hint target does not match authoritative capability")
            resolved_target = approved_target
        await self._client.rpush(resolved_target, hint.to_json())

    async def dispatch(
        self,
        task: TaskRequested,
        target: str | None,
        capability: str | None = None,
    ) -> None:
        """Atomically enqueue task payload or dispatch wake-up hint and establish pending status.

        In accordance with ADR-0006 and ADR-0007:
        - In authenticated_claim mode, publishes exactly one WakeUpHint to agent.hints.{canonical_capability}.
        - In legacy_push mode, RPUSH of TaskRequested and transition of task.status to 'pending' commit together
          in an atomic transaction pipeline (transaction=True).
        """
        if capability is None or not isinstance(capability, str):
            raise ValueError("Authoritative dispatch capability is required")
        if capability not in self._capability_queue_contract:
            raise ValueError("Unrecognized authoritative dispatch capability")
        approved_target = self._capability_queue_contract[capability]

        if target is not None:
            if not isinstance(target, str):
                raise ValueError("Dispatch target must be a string or omitted")
            if target != approved_target:
                raise ValueError("Dispatch target does not match authoritative capability")

        if self._delivery_mode == "authenticated_claim":
            canonical_cap = capability
            hint_target = f"agent.hints.{canonical_cap}"
            hint = WakeUpHint(
                hint_id=str(uuid4()),
                capability=canonical_cap,
                created_at=datetime.now(UTC).isoformat(),
                execution_kind="initial",
                task_id=task.task_id,
            )
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.rpush(hint_target, hint.to_json())
                pipe.set(
                    f"task.status:{task.task_id}",
                    json.dumps({"task_id": task.task_id, "status": "pending"}),
                )
                await pipe.execute()
        else:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.rpush(approved_target, task.to_json())
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
        capability_name = capability.value if hasattr(capability, "value") else str(capability)
        queue_name = target if target is not None else self._capability_queue_contract.get(capability_name)
        if queue_name is None:
            raise ValueError(f"Unknown capability: {capability}")
        await self.dispatch(task, queue_name, capability=capability_name)

    async def get_result(self, task_id: str, now: str = "") -> dict[str, Any] | None:
        """Read task completion state with the canonical outcome as authority.

        Governed by Issue #222:
        - A canonical TerminalOutcome controls completed/failed state, and its
          task identity is validated against the requested task;
        - completed outcomes require the canonical TaskResult to exist, be a
          string key holding valid JSON, satisfy the TaskResult contract, and
          bind to the requested task with completed status — any violation
          fails closed instead of fabricating output;
        - failed outcomes require no canonical TaskResult; an unexpected result
          key fails closed;
        - agent.events are observational only and are never a canonical fallback;
        - legacy task.result records written before the completion boundary
          remain readable with task-ID binding validation.
        """
        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            return None

        try:
            raw_outcome = await self._client.get(f"task.outcome:{clean_task_id}")
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt terminal outcome at task.outcome:{clean_task_id}"
            ) from exc
        if raw_outcome is not None:
            try:
                outcome = TerminalOutcome.from_json(raw_outcome)
            except (ValueError, TypeError) as exc:
                raise CorruptAuthoritativeRecordError(
                    f"Corrupt terminal outcome at task.outcome:{clean_task_id}"
                ) from exc
            if outcome.task_id != clean_task_id:
                raise CorruptAuthoritativeRecordError(
                    f"Terminal outcome at task.outcome:{clean_task_id} "
                    "is not bound to the requested task"
                )
            if outcome.terminal_status == "completed":
                result = await self.get_canonical_result(clean_task_id)
                if result is None:
                    raise CorruptAuthoritativeRecordError(
                        f"Canonical result missing for completed task '{clean_task_id}'; "
                        "refusing to fabricate output"
                    )
                return result.to_dict()
            try:
                unexpected = await self._client.get(f"task.result:{clean_task_id}")
            except (NoPermissionError, AuthenticationError):
                raise
            except ResponseError as exc:
                raise CorruptAuthoritativeRecordError(
                    f"Corrupt canonical result at task.result:{clean_task_id}"
                ) from exc
            if unexpected is not None:
                raise CorruptAuthoritativeRecordError(
                    f"Unexpected canonical result for failed task '{clean_task_id}'"
                )
            return {
                "task_id": clean_task_id,
                "status": outcome.terminal_status,
                "output": None,
            }

        try:
            raw_result = await self._client.get(f"task.result:{clean_task_id}")
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt legacy result at task.result:{clean_task_id}"
            ) from exc
        if raw_result is not None:
            try:
                legacy_data = json.loads(raw_result)
            except (json.JSONDecodeError, TypeError) as exc:
                raise CorruptAuthoritativeRecordError(
                    f"Corrupt legacy result at task.result:{clean_task_id}"
                ) from exc
            if (
                not isinstance(legacy_data, dict)
                or not isinstance(legacy_data.get("task_id"), str)
                or legacy_data.get("task_id") != clean_task_id
            ):
                raise CorruptAuthoritativeRecordError(
                    f"Legacy result at task.result:{clean_task_id} "
                    "is not bound to the requested task"
                )
            return legacy_data

        try:
            raw_status = await self._client.get(f"task.status:{clean_task_id}")
        except (NoPermissionError, AuthenticationError):
            raise
        except ResponseError as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt task status at task.status:{clean_task_id}"
            ) from exc
        if raw_status is not None:
            try:
                status_data = json.loads(raw_status)
            except (json.JSONDecodeError, TypeError) as exc:
                raise CorruptAuthoritativeRecordError(
                    f"Corrupt task status at task.status:{clean_task_id}"
                ) from exc
            if not isinstance(status_data, dict):
                raise CorruptAuthoritativeRecordError(
                    f"Corrupt task status at task.status:{clean_task_id}"
                )
            cur_s = status_data.get("status", "pending")

            # Targeted task-specific expiry evaluation for pending/working tasks
            now_iso = (
                normalize_canonical_utc_timestamp(now)
                if (now and str(now).strip())
                else normalize_canonical_utc_timestamp(datetime.now(UTC))
            )
            if cur_s == "pending":
                try:
                    raw_deadline = await self._client.get(f"task.claim_deadline:{clean_task_id}")
                except (NoPermissionError, AuthenticationError):
                    raise
                except ResponseError as exc:
                    raise CorruptAuthoritativeRecordError(
                        f"Corrupt claim deadline record at task.claim_deadline:{clean_task_id}: {exc}"
                    ) from exc
                if raw_deadline is not None:
                    deadline_str = raw_deadline if isinstance(raw_deadline, str) else raw_deadline.decode("utf-8")
                    if deadline_str and now_iso > deadline_str:
                        try:
                            await self.materialize_claim_timeout(clean_task_id, now_iso)
                            return {
                                "task_id": clean_task_id,
                                "status": "claim_timed_out",
                                "output": None,
                            }
                        except (
                            TaskNotClaimableError,
                            CompletionConflictError,
                            ClaimTimeoutConflictError,
                            WaveAlreadyResolvedError,
                        ):
                            pass
                        except CorruptAuthoritativeRecordError:
                            raise
            elif cur_s == "working":
                try:
                    raw_claim = await self._client.get(f"task.claim:{clean_task_id}")
                except (NoPermissionError, AuthenticationError):
                    raise
                except ResponseError as exc:
                    raise CorruptAuthoritativeRecordError(
                        f"Corrupt claim record at task.claim:{clean_task_id}: {exc}"
                    ) from exc
                if raw_claim is not None:
                    try:
                        claim_obj = ActiveClaim.from_json(raw_claim)
                    except (ValueError, TypeError, json.JSONDecodeError) as exc:
                        raise CorruptAuthoritativeRecordError(
                            f"Corrupt claim record at task.claim:{clean_task_id}: {exc}"
                        ) from exc
                    if (
                        claim_obj.state == "acknowledged"
                        and claim_obj.execution_expires_at
                        and now_iso > claim_obj.execution_expires_at
                    ):
                        try:
                            await self.transition_expired_claim(clean_task_id, now_iso)
                            return {
                                "task_id": clean_task_id,
                                "status": "recovery_required",
                                "output": None,
                            }
                        except (
                            TaskNotClaimableError,
                            CompletionConflictError,
                            WaveAlreadyResolvedError,
                        ):
                            pass
                        except CorruptAuthoritativeRecordError:
                            raise

            return {
                "task_id": clean_task_id,
                "status": cur_s,
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

            obs_turn_id = (parsed_telem.turn_id if parsed_telem and parsed_telem.turn_id else (parsed_trans.turn_id if parsed_trans and parsed_trans.turn_id else fields.get("turn_id"))) or None
            obs_assignment_id = (parsed_telem.assignment_id if parsed_telem and parsed_telem.assignment_id else (parsed_trans.assignment_id if parsed_trans and parsed_trans.assignment_id else fields.get("assignment_id"))) or None
            obs_execution_kind = (parsed_telem.execution_kind if parsed_telem and parsed_telem.execution_kind else (parsed_trans.execution_kind if parsed_trans and parsed_trans.execution_kind else fields.get("execution_kind"))) or None
            raw_cver = (parsed_telem.continuation_version if parsed_telem and parsed_telem.continuation_version is not None else (parsed_trans.continuation_version if parsed_trans and parsed_trans.continuation_version is not None else fields.get("continuation_version")))
            obs_continuation_version = int(raw_cver) if (raw_cver is not None and str(raw_cver).strip().isdigit()) else None

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
                    turn_id=obs_turn_id,
                    assignment_id=obs_assignment_id,
                    execution_kind=obs_execution_kind,
                    continuation_version=obs_continuation_version,
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
