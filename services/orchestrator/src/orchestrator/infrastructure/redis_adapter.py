import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import redis.asyncio as redis
from agent_contracts import (
    ActiveClaim,
    AgentEvent,
    AssignmentCompletionRecord,
    AssignmentYieldRecord,
    ClaimExpiryRecord,
    CompletionCommand,
    CompletionRecord,
    DelegationGrant,
    DelegationProposalRecord,
    FallbackTransition,
    InferenceExecutionTelemetry,
    RecoveryDecisionCommand,
    RecoveryDecisionRecord,
    RedisConnectionConfig,
    RequiredCapabilityRecord,
    RootAuthorityRecord,
    RoutingDecision,
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
    canonical_json_dumps,
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
    WaveBudgetExceededError,
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

if has_grant_key ~= has_grant_payload then
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
    if auth_tbl['max_delegation_depth'] ~= 1 or auth_tbl['max_delegation_waves'] ~= 1 or auth_tbl['max_children_per_wave'] ~= 1 then
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
                    except (TaskNotClaimableError, CompletionConflictError, ClaimTimeoutConflictError):
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
                    except (TaskNotClaimableError, CompletionConflictError):
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
        except (NoPermissionError, AuthenticationError):
            raise
        except Exception as exc:
            raise CorruptAuthoritativeRecordError(
                f"Corrupt TaskContinuation record at key '{key}': {exc}"
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

    async def save_authority_grant(self, grant: DelegationGrant) -> None:
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

    async def atomic_claim_task(
        self,
        task_id: str,
        claim: ActiveClaim,
        grant: DelegationGrant | None = None,
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

        if code == "ALREADY_CLAIMED":
            raise TaskAlreadyClaimedError(f"Task '{clean_task_id}' has an active claim by another runtime")

        if code == "EXECUTION_CLAIM_EXPIRED":
            try:
                await self.transition_expired_claim(clean_task_id, now_iso)
            except (TaskNotClaimableError, CompletionConflictError):
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
            except (TaskNotClaimableError, CompletionConflictError, ClaimTimeoutConflictError):
                pass
            except CorruptAuthoritativeRecordError:
                raise
            raise ChildClaimDeadlineExpiredError(
                f"Claim deadline for child task '{clean_task_id}' has expired ({deadline_str})"
            )

        if code in ("MISSING_ROOT_AUTHORITY", "CORRUPT_ROOT_AUTHORITY"):
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
        ):
            raise TaskNotClaimableError(
                f"Completion preconditions not satisfied for task '{clean_task}': {code}"
            )

        if code.startswith(("WRONG_KEY_TYPE_", "CORRUPT_")):
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
                        except (TaskNotClaimableError, CompletionConflictError, ClaimTimeoutConflictError):
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
                        except (TaskNotClaimableError, CompletionConflictError):
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
