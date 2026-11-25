"""Dedicated live Redis 7 integration tests for Issue #224.

Governed by ADR-0005, ADR-0006, ADR-0007, and Issue #224:
- Dedicated to database 15 (REDIS_URL=redis://127.0.0.1:6379/15);
- Never runs FLUSHALL (flushes only DB 15);
- Exercises complete Research -> Analysis -> Research delegation and continuation flow on real Redis 7;
- Exercises cross-instance Research reference topology: research-agent-1 yields, research-agent-2 resumes;
- Verifies parent executor-slot invariant on real Redis (waiting parent has NO active claim; worker processes unrelated task);
- Verifies all 3 child outcomes on real Redis:
  1. completed -> combined synthesis;
  2. failed -> degraded synthesis with primary sealed evidence;
  3. claim_timed_out -> degraded synthesis after evaluated expiry;
- Proves real-Redis Lua fencing and proof boundaries:
  - epoch 1 on initial claim;
  - epoch 2 on resumed claim;
  - stale completion from research-agent-1 rejected at detached-proof authorization boundary (exact HTTP 403; assignment existence not disclosed);
  - second delegation wave rejected by real Redis;
  - non-root / depth > 1 delegation rejected at authoritative missing-grant boundary by real Redis (ADR-0006 Items 8, 19; ADR-0007 Decision 8);
  - unauthorized capability rejected by real Redis;
- Proves multi-turn telemetry records distinct turn IDs and execution kinds without collision;
- Performs full DB 15 key/value scan asserting zero bearer tokens, peppers, or plaintext proofs exist in any key or stream.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import pytest
import redis.asyncio as redis
from agent_contracts import (
    AgentEvent,
    CompletionCommand,
    InferenceExecutionTelemetry,
    TaskInput,
    TaskRequested,
    TaskResult,
    normalize_canonical_utc_timestamp,
)
from agent_infrastructure import CompletionSubmissionError, OrchestratorClient
from orchestrator.application.use_cases import EvaluateExpiry
from orchestrator.config import Settings
from orchestrator.domain.principal import (
    RuntimePrincipal,
    compute_token_digest,
)
from orchestrator.infrastructure.redis_adapter import RedisGateway
from orchestrator.presentation.api import create_app

pytestmark = [pytest.mark.anyio, pytest.mark.redis]

REQUIRED_TEST_DB = "15"
PEPPER = "test_issue_224_redis_pepper_1234567890"
PROOF_PEPPER = "test_issue_224_redis_proof_pepper_0987654321"

RESEARCH_1_TOKEN = "secret-token-issue-224-redis-research-1"
RESEARCH_2_TOKEN = "secret-token-issue-224-redis-research-2"
ANALYSIS_1_TOKEN = "secret-token-issue-224-redis-analysis-1"
WORKER_1_TOKEN = "secret-token-issue-224-redis-worker-1"

PRINCIPALS_CONFIG: dict[str, dict[str, Any]] = {
    "principal-research-1": {
        "principal_id": "principal-research-1",
        "bound_agent_id": "research-agent-1",
        "authorized_capabilities": ["research"],
        "token_digest": compute_token_digest(RESEARCH_1_TOKEN, PEPPER),
        "active": True,
    },
    "principal-research-2": {
        "principal_id": "principal-research-2",
        "bound_agent_id": "research-agent-2",
        "authorized_capabilities": ["research"],
        "token_digest": compute_token_digest(RESEARCH_2_TOKEN, PEPPER),
        "active": True,
    },
    "principal-analysis-1": {
        "principal_id": "principal-analysis-1",
        "bound_agent_id": "analysis-agent-1",
        "authorized_capabilities": ["analysis"],
        "token_digest": compute_token_digest(ANALYSIS_1_TOKEN, PEPPER),
        "active": True,
    },
    "principal-worker-1": {
        "principal_id": "principal-worker-1",
        "bound_agent_id": "worker-1",
        "authorized_capabilities": ["general"],
        "token_digest": compute_token_digest(WORKER_1_TOKEN, PEPPER),
        "active": True,
    },
}


def _validate_test_redis_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("redis", "rediss"):
        raise ValueError(
            f"Redis integration tests require a redis:// URL, got scheme '{parsed.scheme}'"
        )
    if parsed.path.strip("/") != REQUIRED_TEST_DB:
        raise ValueError(
            "Redis integration tests require database 15 to prevent accidental data loss, "
            f"got '{parsed.path}'"
        )


@pytest.fixture
def redis_url() -> str:
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        pytest.skip("Redis integration tests disabled: REDIS_URL environment variable is not set")
    try:
        _validate_test_redis_url(url)
    except ValueError as exc:
        pytest.fail(str(exc))
    return url


@pytest.fixture
async def clean_gateway(redis_url: str) -> AsyncGenerator[RedisGateway, None]:
    gateway = RedisGateway(redis_url, allow_unauthenticated_local_test=True)
    try:
        await asyncio.wait_for(gateway._client.ping(), timeout=1.0)
    except (redis.ConnectionError, OSError) as exc:
        await gateway.aclose()
        pytest.fail(f"Configured Redis endpoint is unreachable within 1.0s timeout: {exc}")
    await gateway._client.flushdb()
    try:
        yield gateway
    finally:
        try:
            await gateway._client.flushdb()
        finally:
            await gateway.aclose()


def _build_test_settings(redis_url: str) -> Settings:
    return Settings(
        redis_url=redis_url,
        app_env="test",
        task_delivery_mode="authenticated_claim",
        claim_delivery_lease_seconds=30.0,
        child_claim_deadline_seconds=120.0,
        runtime_credential_pepper=PEPPER,
        assignment_proof_pepper=PROOF_PEPPER,
        runtime_principals=PRINCIPALS_CONFIG,
    )


async def _register_root_research_task(gateway: RedisGateway, task_id: str, prompt: str) -> None:
    now_iso = normalize_canonical_utc_timestamp(datetime.now(UTC))
    task = TaskRequested(
        task_id=task_id,
        prompt=prompt,
        created_at=now_iso,
        root_task_id=task_id,
        parent_task_id=None,
        version=2,
    )
    await gateway.register_root_task(task, required_capability="research")
    await gateway.update_status(task_id, "pending")


async def assert_database_zero_secret_leakage(
    gateway: RedisGateway,
    secrets_to_check: list[str],
) -> None:
    """Full SCAN of all keys and values in Redis verifying zero secret leakage and no forbidden authority fields."""
    forbidden_authority_fields = {
        "token",
        "bearer_token",
        "auth_token",
        "api_key",
        "password",
        "plaintext_proof",
        "proof",
        "proof_pepper",
        "token_pepper",
        "runtime_credential_pepper",
        "assignment_proof_pepper",
    }
    cursor = 0
    while True:
        cursor, keys = await gateway._client.scan(cursor=cursor, count=100)
        for key in keys:
            key_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)
            for secret in secrets_to_check:
                assert secret not in key_str, f"Secret leaked in key name: {key_str}"

            ktype = await gateway._client.type(key)
            ktype_str = ktype.decode("utf-8") if isinstance(ktype, bytes) else str(ktype)

            def _inspect_data(data: Any, context: str) -> None:
                if isinstance(data, dict):
                    for k, v in data.items():
                        k_str = str(k).lower().strip()
                        assert k_str not in forbidden_authority_fields, (
                            f"Forbidden authority field '{k}' in {context}"
                        )
                        for secret in secrets_to_check:
                            assert secret not in str(k), (
                                f"Secret leaked in dict key '{k}' of {context}"
                            )
                        _inspect_data(v, f"{context} -> {k}")
                elif isinstance(data, list):
                    for idx, it in enumerate(data):
                        _inspect_data(it, f"{context}[{idx}]")
                elif isinstance(data, str):
                    for secret in secrets_to_check:
                        assert secret not in data, f"Secret leaked in value of {context}"
                    if data.startswith("{") and data.endswith("}"):
                        try:
                            parsed = json.loads(data)
                            if isinstance(parsed, dict):
                                _inspect_data(parsed, f"{context} (parsed JSON)")
                        except (json.JSONDecodeError, TypeError):
                            pass

            if ktype_str == "string":
                val = await gateway._client.get(key)
                val_str = (
                    val.decode("utf-8", errors="replace") if isinstance(val, bytes) else str(val)
                )
                _inspect_data(val_str, f"string key '{key_str}'")
            elif ktype_str == "hash":
                hvals = await gateway._client.hgetall(key)
                for hk, hv in hvals.items():
                    hk_str = (
                        hk.decode("utf-8", errors="replace") if isinstance(hk, bytes) else str(hk)
                    )
                    hv_str = (
                        hv.decode("utf-8", errors="replace") if isinstance(hv, bytes) else str(hv)
                    )
                    _inspect_data(hk_str, f"hash field '{hk_str}' of key '{key_str}'")
                    _inspect_data(hv_str, f"hash value of field '{hk_str}' in key '{key_str}'")
            elif ktype_str == "list":
                items = await gateway._client.lrange(key, 0, -1)
                for it in items:
                    it_str = (
                        it.decode("utf-8", errors="replace") if isinstance(it, bytes) else str(it)
                    )
                    _inspect_data(it_str, f"list key '{key_str}'")
            elif ktype_str == "set":
                members = await gateway._client.smembers(key)
                for mem in members:
                    mem_str = (
                        mem.decode("utf-8", errors="replace")
                        if isinstance(mem, bytes)
                        else str(mem)
                    )
                    _inspect_data(mem_str, f"set key '{key_str}'")
            elif ktype_str == "zset":
                zitems = await gateway._client.zrange(key, 0, -1)
                for zit in zitems:
                    zit_str = (
                        zit.decode("utf-8", errors="replace")
                        if isinstance(zit, bytes)
                        else str(zit)
                    )
                    _inspect_data(zit_str, f"zset key '{key_str}'")
            elif ktype_str == "stream":
                entries = await gateway._client.xrange(key)
                for _eid, fields in entries:
                    for fk, fv in fields.items():
                        fk_str = (
                            fk.decode("utf-8", errors="replace")
                            if isinstance(fk, bytes)
                            else str(fk)
                        )
                        fv_str = (
                            fv.decode("utf-8", errors="replace")
                            if isinstance(fv, bytes)
                            else str(fv)
                        )
                        _inspect_data(fk_str, f"stream field '{fk_str}' of key '{key_str}'")
                        _inspect_data(
                            fv_str, f"stream value of field '{fk_str}' in key '{key_str}'"
                        )

        if cursor == 0:
            break


@pytest.mark.anyio
@pytest.mark.redis
async def test_redis_e2e_research_to_analysis_completed_child_cross_instance(
    redis_url: str,
    clean_gateway: RedisGateway,
) -> None:
    settings = _build_test_settings(redis_url)
    app = create_app(settings=settings, gateway=clean_gateway)
    transport = httpx.ASGITransport(app=app)

    research_client_1 = OrchestratorClient("http://test", RESEARCH_1_TOKEN, transport=transport)
    research_client_2 = OrchestratorClient("http://test", RESEARCH_2_TOKEN, transport=transport)
    analysis_client_1 = OrchestratorClient("http://test", ANALYSIS_1_TOKEN, transport=transport)

    parent_id = f"task-parent-{uuid4().hex[:8]}"
    await _register_root_research_task(
        clean_gateway, parent_id, "Investigate topological insulator edge states"
    )

    # 1. research-agent-1 claims parent initial turn
    req_id_1 = f"req-{uuid4().hex[:8]}"
    claim_1, proof_1 = await research_client_1.claim_task(
        parent_id, req_id_1, "research", "initial"
    )
    assert claim_1["task_id"] == parent_id
    assert claim_1["fencing_epoch"] == 1
    assert await clean_gateway.get_status(parent_id) == "working"

    # Exact claim replay returns identical assignment, proof, and unadvanced fence
    claim_replay, proof_replay = await research_client_1.claim_task(
        parent_id, req_id_1, "research", "initial"
    )
    assert claim_replay["assignment_id"] == claim_1["assignment_id"]
    assert proof_replay == proof_1
    assert await clean_gateway.get_fencing_epoch(parent_id) == 1

    await research_client_1.acknowledge_assignment(parent_id, claim_1["assignment_id"], proof_1)

    # 2. Stage sealed artifact via POST /api/tasks/{parent_id}/artifacts
    art_id = f"art-{uuid4().hex[:8]}"
    art_content = {"material": "Topological Dirac cone observed at Fermi level in Bi2Se3."}
    stage_resp = await research_client_1.stage_artifact(
        task_id=parent_id,
        assignment_id=claim_1["assignment_id"],
        assignment_proof=proof_1,
        artifact_id=art_id,
        artifact_type="research_material",
        content=art_content,
    )
    assert stage_resp["artifact"]["artifact_id"] == art_id
    assert stage_resp["artifact"]["sealed"] is True

    # Publish turn 1 inference telemetry to real Redis stream
    turn1_telemetry = InferenceExecutionTelemetry(
        provider_name="gemini",
        attempt_index=0,
        latency_ms=150.0,
        status="succeeded",
        turn_id=claim_1["turn_id"],
        assignment_id=claim_1["assignment_id"],
        execution_kind="initial",
    )
    event_turn1 = AgentEvent.create(
        agent_id="research-agent-1",
        state="working",
        detail="Primary research execution",
        task_id=parent_id,
        turn_id=claim_1["turn_id"],
        assignment_id=claim_1["assignment_id"],
        execution_kind="initial",
        inference_telemetry=turn1_telemetry,
    )
    await clean_gateway.publish(event_turn1)

    # 3. Propose delegation to Analysis via POST /api/tasks/{parent_id}/delegations
    prop_id = f"prop-{uuid4().hex[:8]}"
    del_resp = await research_client_1.propose_delegation(
        task_id=parent_id,
        assignment_id=claim_1["assignment_id"],
        assignment_proof=proof_1,
        proposal_id=prop_id,
        child_intent="Analyze bandgap opening under magnetic doping",
        artifact_reference={
            "artifact_id": art_id,
            "producing_task_id": parent_id,
            "artifact_version": 1,
        },
        target_capability="analysis",
    )
    child_id = del_resp["child_task_id"]

    # Exact proposal replay returns identical child_task_id, unchanged continuation & fence, no duplicate hint
    del_replay = await research_client_1.propose_delegation(
        task_id=parent_id,
        assignment_id=claim_1["assignment_id"],
        assignment_proof=proof_1,
        proposal_id=prop_id,
        child_intent="Analyze bandgap opening under magnetic doping",
        artifact_reference={
            "artifact_id": art_id,
            "producing_task_id": parent_id,
            "artifact_version": 1,
        },
        target_capability="analysis",
    )
    assert del_replay["child_task_id"] == child_id
    assert del_replay["child_claim_deadline"] == del_resp["child_claim_deadline"]
    assert await clean_gateway.get_children(parent_id) == [child_id]
    assert (await clean_gateway.get_continuation(parent_id, 1)) is not None
    assert await clean_gateway.get_fencing_epoch(parent_id) == 1
    assert await clean_gateway._client.llen("agent.hints.analysis") == 1

    # Conflicting proposal replay (same proposal ID, conflicting child_intent) fails closed with 403
    with pytest.raises(CompletionSubmissionError) as exc_info:
        await research_client_1.propose_delegation(
            task_id=parent_id,
            assignment_id=claim_1["assignment_id"],
            assignment_proof=proof_1,
            proposal_id=prop_id,
            child_intent="Conflicting child intent for replay test",
            artifact_reference={
                "artifact_id": art_id,
                "producing_task_id": parent_id,
                "artifact_version": 1,
            },
            target_capability="analysis",
        )
    assert exc_info.value.status_code == 403
    assert await clean_gateway.get_children(parent_id) == [child_id]
    assert await clean_gateway.get_fencing_epoch(parent_id) == 1
    assert await clean_gateway._client.llen("agent.hints.analysis") == 1

    # 4. Assert parent yielded on real Redis: status waiting_for_children and NO active claim!
    assert await clean_gateway.get_status(parent_id) == "waiting_for_children"
    assert (await clean_gateway.get_active_claim(parent_id)) is None, (
        "Parent must have NO active claim"
    )

    # 5. Assert parent executor-slot invariant on real Redis:
    # research-agent-1 can claim and finish an unrelated task while parent is waiting!
    unrelated_id = f"task-unrelated-{uuid4().hex[:8]}"
    await _register_root_research_task(clean_gateway, unrelated_id, "Unrelated study")
    unrelated_claim, unrelated_proof = await research_client_1.claim_task(
        unrelated_id, f"req-{uuid4().hex[:8]}", "research", "initial"
    )
    await research_client_1.acknowledge_assignment(
        unrelated_id, unrelated_claim["assignment_id"], unrelated_proof
    )
    unrelated_cmd = CompletionCommand(
        completion_command_id=f"comp-{uuid4().hex[:8]}",
        assignment_id=unrelated_claim["assignment_id"],
        execution_kind="initial",
        completed_result=TaskResult(task_id=unrelated_id, output="Unrelated completed output"),
    )
    await research_client_1.submit_completion(unrelated_id, unrelated_cmd, unrelated_proof)
    assert await clean_gateway.get_status(unrelated_id) == "completed"
    assert await clean_gateway.get_status(parent_id) == "waiting_for_children"

    # 6. Analysis agent consumes wake-up hint from agent.hints.analysis on real Redis
    raw_analysis_hint = await clean_gateway._client.rpop("agent.hints.analysis")
    assert raw_analysis_hint is not None
    hint_dict = json.loads(raw_analysis_hint)
    assert hint_dict["task_id"] == child_id
    assert hint_dict["capability"] == "analysis"
    assert hint_dict["execution_kind"] == "delegated_child"

    # 7. Analysis agent claims and acknowledges child task on real Redis
    child_claim, child_proof = await analysis_client_1.claim_task(
        child_id, f"req-{uuid4().hex[:8]}", "analysis", "delegated_child"
    )
    assert child_claim["task_id"] == child_id
    assert child_claim["fencing_epoch"] == 1
    await analysis_client_1.acknowledge_assignment(
        child_id, child_claim["assignment_id"], child_proof
    )

    # 8. Analysis agent reads child TaskInput via GET /api/tasks/{child_id}/input on real Redis
    child_task_input = await analysis_client_1.get_child_task_input(
        child_id, child_claim["assignment_id"], child_proof
    )
    assert child_task_input.task_id == child_id
    assert "Analyze bandgap opening" in child_task_input.prompt

    # 9. Analysis agent completes child turn on real Redis
    child_completion_cmd = CompletionCommand(
        completion_command_id=f"comp-{uuid4().hex[:8]}",
        assignment_id=child_claim["assignment_id"],
        execution_kind="delegated_child",
        completed_result=TaskResult(
            task_id=child_id,
            output="Analysis findings: Cr-doping induces 23 meV magnetic exchange gap.",
        ),
    )
    child_outcome_resp = await analysis_client_1.submit_completion(
        child_id, child_completion_cmd, child_proof
    )
    assert child_outcome_resp["outcome"]["terminal_status"] == "completed"

    # Exact child completion replay returns same outcome, marks replayed=True, no duplicate hint
    child_comp_replay = await analysis_client_1.submit_completion(
        child_id, child_completion_cmd, child_proof
    )
    assert child_comp_replay["outcome"]["outcome_id"] == child_outcome_resp["outcome"]["outcome_id"]
    assert child_comp_replay["replayed"] is True
    assert await clean_gateway._client.llen("agent.hints.research") == 1
    assert await clean_gateway.get_status(parent_id) == "resumable"

    # Child completed -> parent is now resumable on real Redis!
    assert await clean_gateway.get_status(child_id) == "completed"
    assert await clean_gateway.get_status(parent_id) == "resumable"

    # 10. Parent wake-up hint emitted to agent.hints.research on real Redis
    raw_research_hint = await clean_gateway._client.rpop("agent.hints.research")
    assert raw_research_hint is not None
    res_hint_dict = json.loads(raw_research_hint)
    assert res_hint_dict["task_id"] == parent_id
    assert res_hint_dict["capability"] == "research"
    assert res_hint_dict["execution_kind"] == "resumed_continuation"
    assert res_hint_dict["expected_continuation_version"] == 1

    # 11. Cross-instance Research resumption by research-agent-2 on real Redis
    resumed_claim, resumed_proof = await research_client_2.claim_task(
        parent_id,
        f"req-resumed-{uuid4().hex[:8]}",
        "research",
        "resumed_continuation",
        expected_continuation_version=1,
    )
    assert resumed_claim["task_id"] == parent_id
    assert resumed_claim["fencing_epoch"] == 2
    assert resumed_claim["continuation_version"] == 1

    await research_client_2.acknowledge_assignment(
        parent_id, resumed_claim["assignment_id"], resumed_proof
    )

    # Prove stale completion rejection on real Redis at the detached-proof authorization boundary (stale completion, not stale acknowledgement):
    # research-agent-1 attempts stale completion submission after research-agent-2 owns the resumed epoch.
    # Rejected at the detached-proof authorization boundary (exact HTTP 403); assignment existence is not disclosed.
    hints_before = await clean_gateway._client.llen("agent.hints.research")
    stale_comp_cmd = CompletionCommand(
        completion_command_id=f"comp-stale-{uuid4().hex[:8]}",
        assignment_id=claim_1["assignment_id"],
        execution_kind="initial",
        completed_result=TaskResult(task_id=parent_id, output="Stale initial completion attempt on real Redis"),
    )
    with pytest.raises(CompletionSubmissionError) as exc_info:
        await research_client_1.submit_completion(parent_id, stale_comp_cmd, proof_1)
    assert exc_info.value.status_code == 403

    # Assert exactly:
    active_claim = await clean_gateway.get_active_claim(parent_id)
    assert active_claim is not None
    assert active_claim.assignment_id == resumed_claim["assignment_id"]
    assert active_claim.runtime_principal_id == "principal-research-2"
    assert active_claim.bound_agent_id == "research-agent-2"
    assert (await clean_gateway.get_fencing_epoch(parent_id)) == 2
    assert (await clean_gateway.get_status(parent_id)) == "working"
    assert (await clean_gateway.get_terminal_outcome(parent_id)) is None
    assert (await clean_gateway.get_canonical_result(parent_id)) is None
    assert (await clean_gateway._client.llen("agent.hints.research")) == hints_before
    assert (await clean_gateway.get_continuation(parent_id, 2)) is None

    # Publish turn 2 inference telemetry to real Redis stream
    turn2_telemetry = InferenceExecutionTelemetry(
        provider_name="gemini",
        attempt_index=0,  # attempt_index resets to 0 independently for resumed turn!
        latency_ms=180.0,
        status="succeeded",
        turn_id=resumed_claim["turn_id"],
        assignment_id=resumed_claim["assignment_id"],
        execution_kind="resumed_continuation",
        continuation_version=1,
    )
    event_turn2 = AgentEvent.create(
        agent_id="research-agent-2",
        state="working",
        detail="Resumed research synthesis",
        task_id=parent_id,
        turn_id=resumed_claim["turn_id"],
        assignment_id=resumed_claim["assignment_id"],
        execution_kind="resumed_continuation",
        continuation_version=1,
        inference_telemetry=turn2_telemetry,
    )
    await clean_gateway.publish(event_turn2)

    # 12. research-agent-2 reads continuation context on real Redis
    context = await research_client_2.get_resumed_continuation_context(
        parent_id, resumed_claim["assignment_id"], resumed_proof
    )
    assert context.parent_task_input.task_id == parent_id
    assert context.stage_artifact.artifact_id == art_id
    assert context.child_outcome.terminal_status == "completed"
    assert context.child_result is not None
    assert "23 meV magnetic exchange gap" in context.child_result.output

    # 13. research-agent-2 submits fenced completion with combined synthesis
    combined_report = (
        f"Primary Research Evidence:\n{context.stage_artifact.content['material']}\n\n"
        f"Specialist Analysis Findings:\n{context.child_result.output}"
    )
    resumed_completion_cmd = CompletionCommand(
        completion_command_id=f"comp-{uuid4().hex[:8]}",
        assignment_id=resumed_claim["assignment_id"],
        execution_kind="resumed_continuation",
        continuation_version=1,
        completed_result=TaskResult(task_id=parent_id, output=combined_report),
    )
    parent_outcome_resp = await research_client_2.submit_completion(
        parent_id, resumed_completion_cmd, resumed_proof
    )
    assert parent_outcome_resp["outcome"]["terminal_status"] == "completed"
    assert await clean_gateway.get_status(parent_id) == "completed"

    # Exact parent completion replay returns same terminal outcome, marks replayed=True, no duplicate projection
    parent_comp_replay = await research_client_2.submit_completion(
        parent_id, resumed_completion_cmd, resumed_proof
    )
    assert (
        parent_comp_replay["outcome"]["outcome_id"] == parent_outcome_resp["outcome"]["outcome_id"]
    )
    assert parent_comp_replay["replayed"] is True
    assert await clean_gateway.get_status(parent_id) == "completed"

    parent_final_result = await clean_gateway.get_canonical_result(parent_id)
    assert parent_final_result is not None
    assert "Bi2Se3" in parent_final_result.output
    assert "23 meV magnetic exchange gap" in parent_final_result.output

    # Reference-safe retention after parent completion
    parent_input_rec = await clean_gateway.get_task_input(parent_id, 1)
    assert parent_input_rec is not None
    assert "topological insulator" in parent_input_rec.prompt.lower()

    sealed_art = await clean_gateway.get_stage_artifact(parent_id, art_id, 1)
    assert sealed_art is not None
    assert sealed_art.sealed is True
    assert "Bi2Se3" in sealed_art.content["material"]

    cont_rec = await clean_gateway.get_continuation(parent_id, 1)
    assert cont_rec is not None
    assert cont_rec.artifact_references[0].artifact_id == art_id
    assert cont_rec.artifact_references[0].producing_task_id == parent_id
    assert cont_rec.artifact_references[0].artifact_version == 1

    # Real-Redis turn-aware telemetry validation across turns
    telemetry_history = await clean_gateway.task_inference_telemetry(parent_id)
    assert telemetry_history is not None
    assert telemetry_history["history_status"] == "complete"
    assert telemetry_history["final_status"] == "succeeded"
    turns = telemetry_history["turns"]
    assert len(turns) == 2

    # Turn 1: initial turn, attempt 0
    assert turns[0]["turn_id"] == claim_1["turn_id"]
    assert turns[0]["assignment_id"] == claim_1["assignment_id"]
    assert turns[0]["execution_kind"] == "initial"
    assert turns[0]["continuation_version"] is None
    assert len(turns[0]["attempts"]) == 1
    assert turns[0]["attempts"][0]["attempt_index"] == 0

    # Turn 2: resumed continuation, attempt index resets independently to 0
    assert turns[1]["turn_id"] == resumed_claim["turn_id"]
    assert turns[1]["assignment_id"] == resumed_claim["assignment_id"]
    assert turns[1]["execution_kind"] == "resumed_continuation"
    assert turns[1]["continuation_version"] == 1
    assert len(turns[1]["attempts"]) == 1
    assert turns[1]["attempts"][0]["attempt_index"] == 0

    # Post-completion full-database leakage scan across all keys, hashes, lists, streams
    all_secrets = [
        RESEARCH_1_TOKEN,
        RESEARCH_2_TOKEN,
        ANALYSIS_1_TOKEN,
        WORKER_1_TOKEN,
        PEPPER,
        PROOF_PEPPER,
        proof_1,
        child_proof,
        resumed_proof,
    ]
    await assert_database_zero_secret_leakage(clean_gateway, all_secrets)


@pytest.mark.anyio
@pytest.mark.redis
async def test_redis_e2e_research_to_analysis_failed_child_resumption(
    redis_url: str,
    clean_gateway: RedisGateway,
) -> None:
    settings = _build_test_settings(redis_url)
    app = create_app(settings=settings, gateway=clean_gateway)
    transport = httpx.ASGITransport(app=app)

    research_client_1 = OrchestratorClient("http://test", RESEARCH_1_TOKEN, transport=transport)
    research_client_2 = OrchestratorClient("http://test", RESEARCH_2_TOKEN, transport=transport)
    analysis_client_1 = OrchestratorClient("http://test", ANALYSIS_1_TOKEN, transport=transport)

    parent_id = f"task-parent-fail-{uuid4().hex[:8]}"
    await _register_root_research_task(
        clean_gateway, parent_id, "Analyze perovskite solar degradation"
    )

    # Initial research claim & delegation
    claim_1, proof_1 = await research_client_1.claim_task(
        parent_id, f"req-{uuid4().hex[:8]}", "research", "initial"
    )
    await research_client_1.acknowledge_assignment(parent_id, claim_1["assignment_id"], proof_1)

    art_id = f"art-{uuid4().hex[:8]}"
    await research_client_1.stage_artifact(
        task_id=parent_id,
        assignment_id=claim_1["assignment_id"],
        assignment_proof=proof_1,
        artifact_id=art_id,
        artifact_type="research_material",
        content={"material": "Perovskite crystal structure degrades at 85C under moisture."},
    )

    del_resp = await research_client_1.propose_delegation(
        task_id=parent_id,
        assignment_id=claim_1["assignment_id"],
        assignment_proof=proof_1,
        proposal_id=f"prop-{uuid4().hex[:8]}",
        child_intent="Analyze halide migration activation energy",
        artifact_reference={
            "artifact_id": art_id,
            "producing_task_id": parent_id,
            "artifact_version": 1,
        },
        target_capability="analysis",
    )
    child_id = del_resp["child_task_id"]
    assert await clean_gateway.get_status(parent_id) == "waiting_for_children"
    assert (await clean_gateway.get_active_claim(parent_id)) is None, (
        "Parent must have NO active claim"
    )

    # Analysis agent claims child and submits EXECUTION_FAILED
    child_claim, child_proof = await analysis_client_1.claim_task(
        child_id, f"req-{uuid4().hex[:8]}", "analysis", "delegated_child"
    )
    await analysis_client_1.acknowledge_assignment(
        child_id, child_claim["assignment_id"], child_proof
    )

    fail_cmd = CompletionCommand(
        completion_command_id=f"comp-{uuid4().hex[:8]}",
        assignment_id=child_claim["assignment_id"],
        execution_kind="delegated_child",
        failure_code="EXECUTION_FAILED",
    )
    fail_resp = await analysis_client_1.submit_completion(child_id, fail_cmd, child_proof)
    assert fail_resp["outcome"]["terminal_status"] == "failed"

    # Child failed -> parent is resumable on real Redis
    assert await clean_gateway.get_status(child_id) == "failed"
    assert await clean_gateway.get_status(parent_id) == "resumable"

    # research-agent-2 resumes parent turn on real Redis
    resumed_claim, resumed_proof = await research_client_2.claim_task(
        parent_id,
        f"req-resumed-{uuid4().hex[:8]}",
        "research",
        "resumed_continuation",
        expected_continuation_version=1,
    )
    await research_client_2.acknowledge_assignment(
        parent_id, resumed_claim["assignment_id"], resumed_proof
    )

    context = await research_client_2.get_resumed_continuation_context(
        parent_id, resumed_claim["assignment_id"], resumed_proof
    )
    assert context.child_outcome.terminal_status == "failed"

    # Synthesize degraded final report based on primary sealed evidence
    degraded_report = (
        f"Primary Research Evidence:\n{context.stage_artifact.content['material']}\n\n"
        "[Note: Specialist analysis was unavailable (failed); synthesizing based on primary sealed research evidence.]"
    )
    resumed_cmd = CompletionCommand(
        completion_command_id=f"comp-{uuid4().hex[:8]}",
        assignment_id=resumed_claim["assignment_id"],
        execution_kind="resumed_continuation",
        continuation_version=1,
        completed_result=TaskResult(task_id=parent_id, output=degraded_report),
    )
    await research_client_2.submit_completion(parent_id, resumed_cmd, resumed_proof)

    assert await clean_gateway.get_status(parent_id) == "completed"
    final_res = await clean_gateway.get_canonical_result(parent_id)
    assert final_res is not None
    assert "Specialist analysis was unavailable (failed)" in final_res.output
    assert "Perovskite crystal structure" in final_res.output

    # Reference-safe retention & verify no child result fabricated for failed child
    assert (await clean_gateway.get_canonical_result(child_id)) is None
    parent_input_rec = await clean_gateway.get_task_input(parent_id, 1)
    assert parent_input_rec is not None
    sealed_art = await clean_gateway.get_stage_artifact(parent_id, art_id, 1)
    assert sealed_art is not None
    assert sealed_art.sealed is True
    cont_rec = await clean_gateway.get_continuation(parent_id, 1)
    assert cont_rec is not None
    assert cont_rec.artifact_references[0].artifact_id == art_id


@pytest.mark.anyio
@pytest.mark.redis
async def test_redis_e2e_research_to_analysis_claim_timed_out_child_resumption(
    redis_url: str,
    clean_gateway: RedisGateway,
) -> None:
    settings = _build_test_settings(redis_url)
    app = create_app(settings=settings, gateway=clean_gateway)
    transport = httpx.ASGITransport(app=app)

    research_client_1 = OrchestratorClient("http://test", RESEARCH_1_TOKEN, transport=transport)
    research_client_2 = OrchestratorClient("http://test", RESEARCH_2_TOKEN, transport=transport)

    parent_id = f"task-parent-timeout-{uuid4().hex[:8]}"
    await _register_root_research_task(
        clean_gateway, parent_id, "Study twisted graphene moiré patterns"
    )

    # Initial research claim & delegation
    claim_1, proof_1 = await research_client_1.claim_task(
        parent_id, f"req-{uuid4().hex[:8]}", "research", "initial"
    )
    await research_client_1.acknowledge_assignment(parent_id, claim_1["assignment_id"], proof_1)

    art_id = f"art-{uuid4().hex[:8]}"
    await research_client_1.stage_artifact(
        task_id=parent_id,
        assignment_id=claim_1["assignment_id"],
        assignment_proof=proof_1,
        artifact_id=art_id,
        artifact_type="research_material",
        content={"material": "Magic angle 1.08 degrees produces flat bands near Fermi energy."},
    )

    del_resp = await research_client_1.propose_delegation(
        task_id=parent_id,
        assignment_id=claim_1["assignment_id"],
        assignment_proof=proof_1,
        proposal_id=f"prop-{uuid4().hex[:8]}",
        child_intent="Calculate Chern numbers of flat bands",
        artifact_reference={
            "artifact_id": art_id,
            "producing_task_id": parent_id,
            "artifact_version": 1,
        },
        target_capability="analysis",
    )
    child_id = del_resp["child_task_id"]
    child_deadline = del_resp["child_claim_deadline"]
    assert await clean_gateway.get_status(parent_id) == "waiting_for_children"
    assert (await clean_gateway.get_active_claim(parent_id)) is None, (
        "Parent must have NO active claim"
    )
    assert await clean_gateway.get_status(child_id) == "pending"

    # Advance clock past child_claim_deadline on real Redis
    past_deadline_dt = datetime.fromisoformat(child_deadline) + timedelta(seconds=10)
    past_deadline_iso = normalize_canonical_utc_timestamp(past_deadline_dt)

    eval_expiry = EvaluateExpiry(task_store=clean_gateway)
    operator = RuntimePrincipal(
        principal_id="principal-recovery-1",
        bound_agent_id="operator-1",
        authorized_capabilities=("recovery_operator",),
        active=True,
    )
    expiry_result = await eval_expiry.execute(operator, child_id, now=past_deadline_iso)
    assert expiry_result.action == "claim_timed_out"
    assert expiry_result.outcome is not None
    assert expiry_result.outcome.terminal_status == "claim_timed_out"

    # Real Redis state assertion: child is claim_timed_out, parent is resumable!
    assert await clean_gateway.get_status(child_id) == "claim_timed_out"
    assert await clean_gateway.get_status(parent_id) == "resumable"

    # research-agent-2 resumes parent turn on real Redis
    resumed_claim, resumed_proof = await research_client_2.claim_task(
        parent_id,
        f"req-resumed-{uuid4().hex[:8]}",
        "research",
        "resumed_continuation",
        expected_continuation_version=1,
    )
    await research_client_2.acknowledge_assignment(
        parent_id, resumed_claim["assignment_id"], resumed_proof
    )

    context = await research_client_2.get_resumed_continuation_context(
        parent_id, resumed_claim["assignment_id"], resumed_proof
    )
    assert context.child_outcome.terminal_status == "claim_timed_out"

    degraded_report = (
        f"Primary Research Evidence:\n{context.stage_artifact.content['material']}\n\n"
        "[Note: Specialist analysis was unavailable (claim_timed_out); synthesizing based on primary sealed research evidence.]"
    )
    resumed_cmd = CompletionCommand(
        completion_command_id=f"comp-{uuid4().hex[:8]}",
        assignment_id=resumed_claim["assignment_id"],
        execution_kind="resumed_continuation",
        continuation_version=1,
        completed_result=TaskResult(task_id=parent_id, output=degraded_report),
    )
    await research_client_2.submit_completion(parent_id, resumed_cmd, resumed_proof)

    assert await clean_gateway.get_status(parent_id) == "completed"
    final_res = await clean_gateway.get_canonical_result(parent_id)
    assert final_res is not None
    assert "Specialist analysis was unavailable (claim_timed_out)" in final_res.output
    assert "Magic angle 1.08 degrees" in final_res.output

    # Reference-safe retention & verify no child result fabricated for claim_timed_out child
    assert (await clean_gateway.get_canonical_result(child_id)) is None
    parent_input_rec = await clean_gateway.get_task_input(parent_id, 1)
    assert parent_input_rec is not None
    sealed_art = await clean_gateway.get_stage_artifact(parent_id, art_id, 1)
    assert sealed_art is not None
    assert sealed_art.sealed is True
    cont_rec = await clean_gateway.get_continuation(parent_id, 1)
    assert cont_rec is not None
    assert cont_rec.artifact_references[0].artifact_id == art_id


@pytest.mark.anyio
@pytest.mark.redis
async def test_redis_delegation_invariants_and_security(
    redis_url: str,
    clean_gateway: RedisGateway,
) -> None:
    settings = _build_test_settings(redis_url)
    app = create_app(settings=settings, gateway=clean_gateway)
    transport = httpx.ASGITransport(app=app)

    research_client_1 = OrchestratorClient("http://test", RESEARCH_1_TOKEN, transport=transport)
    parent_id = f"task-invariants-{uuid4().hex[:8]}"
    await _register_root_research_task(clean_gateway, parent_id, "Invariants verification")

    claim, proof = await research_client_1.claim_task(
        parent_id, f"req-{uuid4().hex[:8]}", "research", "initial"
    )
    await research_client_1.acknowledge_assignment(parent_id, claim["assignment_id"], proof)

    # 1. Artifact staging requires X-Assignment-Proof header
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw_http:
        valid_art_body = {
            "assignment_id": claim["assignment_id"],
            "artifact_id": "art-inv-1",
            "artifact_type": "research_material",
            "content": {"material": "Evidence"},
            "artifact_version": 1,
            "payload_schema_version": 1,
        }
        # Missing proof header -> 401
        resp = await raw_http.post(
            f"/api/tasks/{parent_id}/artifacts",
            json=valid_art_body,
            headers={"Authorization": f"Bearer {RESEARCH_1_TOKEN}"},
        )
        assert resp.status_code == 401

        # Extra assignment_proof in body -> 422 (extra='forbid')
        art_body_extra = dict(valid_art_body)
        art_body_extra["assignment_proof"] = proof
        resp = await raw_http.post(
            f"/api/tasks/{parent_id}/artifacts",
            json=art_body_extra,
            headers={"Authorization": f"Bearer {RESEARCH_1_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert resp.status_code == 422

    # Stage valid artifact
    await research_client_1.stage_artifact(
        task_id=parent_id,
        assignment_id=claim["assignment_id"],
        assignment_proof=proof,
        artifact_id="art-inv-1",
        artifact_type="research_material",
        content={"material": "Evidence"},
    )

    # Propose initial delegation to child
    del_resp = await research_client_1.propose_delegation(
        task_id=parent_id,
        assignment_id=claim["assignment_id"],
        assignment_proof=proof,
        proposal_id=f"prop-{uuid4().hex[:8]}",
        child_intent="Initial child intent",
        artifact_reference={
            "artifact_id": "art-inv-1",
            "producing_task_id": parent_id,
            "artifact_version": 1,
        },
        target_capability="analysis",
    )
    child_id = del_resp["child_task_id"]

    # 2. Reject second delegation wave on real Redis (parent status is waiting_for_children)
    with pytest.raises(CompletionSubmissionError) as exc_info:
        await research_client_1.propose_delegation(
            task_id=parent_id,
            assignment_id=claim["assignment_id"],
            assignment_proof=proof,
            proposal_id=f"prop-second-{uuid4().hex[:8]}",
            child_intent="Second wave intent",
            artifact_reference={
                "artifact_id": "art-inv-1",
                "producing_task_id": parent_id,
                "artifact_version": 1,
            },
            target_capability="analysis",
        )
    assert exc_info.value.status_code in (403, 409)

    # 3. Non-root delegation containment (ADR-0006 Items 8, 19; ADR-0007 Decision 8)
    # Under platform authority rules, delegated children receive no delegation grant.
    # Therefore, non-root tasks can never lawfully propose delegation.
    analysis_client = OrchestratorClient("http://test", ANALYSIS_1_TOKEN, transport=transport)
    child_claim, child_proof = await analysis_client.claim_task(
        child_id, f"req-{uuid4().hex[:8]}", "analysis", "delegated_child"
    )
    assert child_claim.get("delegation_grant_id") is None
    await analysis_client.acknowledge_assignment(
        child_id, child_claim["assignment_id"], child_proof
    )

    # 3a. Reject delegation from analysis child (capability mismatch) -> 403
    with pytest.raises(CompletionSubmissionError) as exc_info:
        await analysis_client.propose_delegation(
            task_id=child_id,
            assignment_id=child_claim["assignment_id"],
            assignment_proof=child_proof,
            proposal_id=f"prop-depth2-{uuid4().hex[:8]}",
            child_intent="Depth 2 child intent",
            artifact_reference={
                "artifact_id": "art-inv-1",
                "producing_task_id": parent_id,
                "artifact_version": 1,
            },
            target_capability="analysis",
        )
    assert exc_info.value.status_code == 403

    # 3b. Authenticated Research-principal attempt to delegate from non-root child task
    # Delegated children receive no delegation grant (ADR-0006 Items 8, 19; ADR-0007 Decision 8).
    # Non-root delegation therefore fails at the missing-grant boundary (HTTP 409).
    child_research_id = f"task-child-res-{uuid4().hex[:8]}"
    now_iso = normalize_canonical_utc_timestamp(datetime.now(UTC))
    child_research_req = TaskRequested(
        task_id=child_research_id,
        prompt="Sub-research study",
        created_at=now_iso,
        root_task_id=parent_id,
        parent_task_id=parent_id,
        version=2,
    )
    await clean_gateway.register_child_task(
        child_research_req,
        initial_status="pending",
        required_capability="research",
    )
    await clean_gateway.save_task_input(
        TaskInput(
            task_id=child_research_id,
            prompt="Sub-research study",
            created_at=now_iso,
            input_version=1,
        )
    )

    # Claim and acknowledge as research-agent-1 (authenticated Research principal)
    child_res_claim, child_res_proof = await research_client_1.claim_task(
        child_research_id,
        f"req-cres-{uuid4().hex[:8]}",
        "research",
        "delegated_child",
    )
    # Contract assertion: delegated child claim carries NO delegation grant
    assert child_res_claim.get("delegation_grant_id") is None
    await research_client_1.acknowledge_assignment(
        child_research_id,
        child_res_claim["assignment_id"],
        child_res_proof,
    )

    # Attempt delegation proposal from non-root child with authenticated Research principal
    # Fails closed at the missing-grant boundary: active assignment has no delegation grant (HTTP 409)
    with pytest.raises(CompletionSubmissionError) as exc_info:
        await research_client_1.propose_delegation(
            task_id=child_research_id,
            assignment_id=child_res_claim["assignment_id"],
            assignment_proof=child_res_proof,
            proposal_id=f"prop-depth2-research-{uuid4().hex[:8]}",
            child_intent="Grandchild analysis attempt",
            artifact_reference={
                "artifact_id": "art-inv-1",
                "producing_task_id": parent_id,
                "artifact_version": 1,
            },
            target_capability="analysis",
        )
    assert exc_info.value.status_code == 409
    # Assert zero grandchild, continuation, hint, fence, or wave mutation occurs
    assert await clean_gateway.get_children(child_research_id) == []
    assert (await clean_gateway.get_continuation(child_research_id, 1)) is None
    assert await clean_gateway.get_fencing_epoch(child_research_id) == 1
    assert (
        await clean_gateway._client.llen("agent.hints.analysis") == 1
    )  # Unchanged, no duplicate hint

    # 4. Reject unauthorized capability (Worker token proposing delegation) -> 403
    worker_client = OrchestratorClient("http://test", WORKER_1_TOKEN, transport=transport)
    with pytest.raises(CompletionSubmissionError) as exc_info:
        await worker_client.propose_delegation(
            task_id=parent_id,
            assignment_id=claim["assignment_id"],
            assignment_proof=proof,
            proposal_id=f"prop-unauth-{uuid4().hex[:8]}",
            child_intent="Unauthorized intent",
            artifact_reference={
                "artifact_id": "art-inv-1",
                "producing_task_id": parent_id,
                "artifact_version": 1,
            },
            target_capability="analysis",
        )
    assert exc_info.value.status_code == 403


@pytest.mark.anyio
@pytest.mark.redis
async def test_redis_zero_secret_leakage_full_db_scan(
    redis_url: str,
    clean_gateway: RedisGateway,
) -> None:
    settings = _build_test_settings(redis_url)
    app = create_app(settings=settings, gateway=clean_gateway)
    transport = httpx.ASGITransport(app=app)

    research_client_1 = OrchestratorClient("http://test", RESEARCH_1_TOKEN, transport=transport)
    parent_id = f"task-secrets-scan-{uuid4().hex[:8]}"
    await _register_root_research_task(clean_gateway, parent_id, "Secrets scan verification")

    claim, proof = await research_client_1.claim_task(
        parent_id, f"req-{uuid4().hex[:8]}", "research", "initial"
    )
    await research_client_1.acknowledge_assignment(parent_id, claim["assignment_id"], proof)

    await research_client_1.stage_artifact(
        task_id=parent_id,
        assignment_id=claim["assignment_id"],
        assignment_proof=proof,
        artifact_id="art-secrets-1",
        artifact_type="research_material",
        content={"material": "No secrets in database scan"},
    )
    await research_client_1.propose_delegation(
        task_id=parent_id,
        assignment_id=claim["assignment_id"],
        assignment_proof=proof,
        proposal_id=f"prop-sec-{uuid4().hex[:8]}",
        child_intent="Secrets check child intent",
        artifact_reference={
            "artifact_id": "art-secrets-1",
            "producing_task_id": parent_id,
            "artifact_version": 1,
        },
        target_capability="analysis",
    )

    secrets_to_check = [
        RESEARCH_1_TOKEN,
        RESEARCH_2_TOKEN,
        ANALYSIS_1_TOKEN,
        WORKER_1_TOKEN,
        PEPPER,
        PROOF_PEPPER,
        proof,
    ]

    await assert_database_zero_secret_leakage(clean_gateway, secrets_to_check)
