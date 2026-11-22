import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from unittest.mock import call as mock_call

import pytest
import redis.asyncio as redis
from agent_contracts import (
    AgentCapabilityMetadata,
    InferenceExecutionTelemetry,
    TaskRequested,
    TaskResult,
)
from analysis_agent.infrastructure.redis_bus import RedisBus as AnalysisRedisBus
from orchestrator.domain.capability import AgentCapability
from orchestrator.infrastructure.redis_adapter import RedisGateway
from research_agent.infrastructure.redis_bus import RedisBus as ResearchRedisBus
from worker.infrastructure.redis_bus import RedisBus


@pytest.mark.anyio
async def test_redis_gateway_routes_general_capability() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.rpush = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=None)
    gateway._client.pipeline = MagicMock(return_value=mock_pipe)
    task = TaskRequested.create("general task prompt")

    await gateway.enqueue(task, capability=AgentCapability.GENERAL)

    mock_pipe.rpush.assert_called_once_with("agent.tasks.general", task.to_json())
    mock_pipe.set.assert_called_once()
    mock_pipe.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_redis_gateway_routes_research_capability() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.rpush = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=None)
    gateway._client.pipeline = MagicMock(return_value=mock_pipe)
    task = TaskRequested.create("research task prompt")

    await gateway.enqueue(task, capability=AgentCapability.RESEARCH)

    mock_pipe.rpush.assert_called_once_with("agent.tasks.research", task.to_json())
    mock_pipe.set.assert_called_once()
    mock_pipe.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_redis_gateway_defaults_to_general() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.rpush = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=None)
    gateway._client.pipeline = MagicMock(return_value=mock_pipe)
    task = TaskRequested.create("default task prompt")

    await gateway.enqueue(task)

    mock_pipe.rpush.assert_called_once_with("agent.tasks.general", task.to_json())
    mock_pipe.set.assert_called_once()
    mock_pipe.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_redis_gateway_invalid_capability_raises_value_error() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.rpush = AsyncMock()
    task = TaskRequested.create("bad capability prompt")

    with pytest.raises(ValueError, match="Unknown capability"):
        await gateway.enqueue(task, capability="unsupported_capability")  # type: ignore[arg-type]

    gateway._client.rpush.assert_not_awaited()


def test_redis_bus_defaults_to_general_queue() -> None:
    bus = RedisBus("redis://localhost:6379/0")
    assert bus.queue_name == "agent.tasks.general"


def test_redis_bus_custom_queue_configuration() -> None:
    bus = RedisBus("redis://localhost:6379/0", queue_name="agent.tasks.research")
    assert bus.queue_name == "agent.tasks.research"


def test_analysis_redis_bus_defaults_to_analysis_queue() -> None:
    bus = AnalysisRedisBus("redis://localhost:6379/0")
    assert bus.queue_name == "agent.tasks.analysis"


@pytest.mark.anyio
async def test_analysis_redis_bus_consumes_from_configured_queue() -> None:
    bus = AnalysisRedisBus("redis://localhost:6379/0", queue_name="agent.tasks.analysis")
    task = TaskRequested.create("Analyze supplied material")
    bus.client.blpop = AsyncMock(return_value=("agent.tasks.analysis", task.to_json()))

    consumed = await bus.next_task()

    assert consumed.task_id == task.task_id
    assert consumed.prompt == "Analyze supplied material"
    bus.client.blpop.assert_awaited_once_with("agent.tasks.analysis", timeout=2)


@pytest.mark.anyio
async def test_analysis_redis_bus_registers_metadata_without_direct_result_write() -> None:
    bus = AnalysisRedisBus("redis://localhost:6379/0")
    bus.client.sadd = AsyncMock()
    bus.client.hset = AsyncMock()
    bus.client.set = AsyncMock()
    metadata = AgentCapabilityMetadata(
        agent_id="analysis-agent-1",
        agent_type="analysis-agent",
        capabilities=["analysis"],
        providers={"analysis_engine": "deterministic"},
        dispatch_target="agent.tasks.analysis",
    )

    await bus.register(metadata)

    bus.client.sadd.assert_awaited_once_with("agent.ids", "analysis-agent-1")
    assert bus.client.hset.await_args.args[0] == "agent.state:analysis-agent-1"
    assert bus.client.hset.await_args.kwargs["mapping"]["capabilities"] == '["analysis"]'
    assert (
        bus.client.hset.await_args.kwargs["mapping"]["dispatch_target"]
        == "agent.tasks.analysis"
    )
    # Canonical completion flows through the Orchestrator boundary; the bus
    # exposes no direct result-write seam.
    assert not hasattr(bus, "save")
    bus.client.set.assert_not_awaited()


@pytest.mark.anyio
async def test_redis_bus_consumes_from_default_queue() -> None:
    bus = RedisBus("redis://localhost:6379/0")
    task = TaskRequested.create("general worker prompt")
    bus.client.blpop = AsyncMock(return_value=("agent.tasks.general", task.to_json()))

    result = await bus.next_task()

    assert result.task_id == task.task_id
    assert result.prompt == "general worker prompt"
    bus.client.blpop.assert_awaited_once_with("agent.tasks.general", timeout=2)


@pytest.mark.anyio
async def test_redis_bus_consumes_from_configured_queue() -> None:
    bus = RedisBus("redis://localhost:6379/0", queue_name="agent.tasks.research")
    task = TaskRequested.create("research worker prompt")
    bus.client.blpop = AsyncMock(return_value=("agent.tasks.research", task.to_json()))

    result = await bus.next_task()

    assert result.task_id == task.task_id
    assert result.prompt == "research worker prompt"
    bus.client.blpop.assert_awaited_once_with("agent.tasks.research", timeout=2)


@pytest.mark.anyio
async def test_run_worker_uses_task_queue_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_TOKEN", "test-token")
    monkeypatch.setenv("TASK_QUEUE", "agent.tasks.research")
    monkeypatch.setenv("AGENT_ID", "research-worker-1")

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from worker.main import run_worker

        with pytest.raises(asyncio.CancelledError):
            await run_worker()

        mock_bus_cls.assert_called_once_with(
            "redis://localhost:6379/0",
            queue_name="agent.tasks.research",
        )


@pytest.mark.anyio
async def test_run_worker_defaults_to_general_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_TOKEN", "test-token")
    monkeypatch.delenv("TASK_QUEUE", raising=False)
    monkeypatch.setenv("AGENT_ID", "worker-1")

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from worker.main import run_worker

        with pytest.raises(asyncio.CancelledError):
            await run_worker()

        mock_bus_cls.assert_called_once_with(
            "redis://localhost:6379/0",
            queue_name="agent.tasks.general",
        )


@pytest.mark.anyio
async def test_redis_gateway_get_result_completed() -> None:
    from agent_contracts import (
        CompletionCommand,
        build_terminal_outcome,
    )

    gateway = RedisGateway("redis://localhost:6379/0")
    command = CompletionCommand(
        completion_command_id="cmd-res-1",
        assignment_id="assign-res-1",
        execution_kind="initial",
        completed_result=TaskResult(task_id="task-res-1", output="Found 3 articles"),
    )
    outcome, result = build_terminal_outcome(
        command=command,
        task_id="task-res-1",
        runtime_principal_id="principal-1",
        bound_agent_id="agent-1",
        completed_at="2026-11-20T00:00:00.000000Z",
    )
    assert result is not None
    gateway._client.get = AsyncMock(
        side_effect=[outcome.to_json(), result.to_json()]
    )

    res = await gateway.get_result("task-res-1")

    assert res == result.to_dict()
    assert gateway._client.get.await_args_list == [
        mock_call("task.outcome:task-res-1"),
        mock_call("task.result:task-res-1"),
    ]


@pytest.mark.anyio
async def test_redis_gateway_get_result_pending() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    # Outcome, then legacy result, then status, then claim_deadline
    gateway._client.get = AsyncMock(
        side_effect=[None, None, json.dumps({"task_id": "task-res-2", "status": "pending"}), None]
    )

    res = await gateway.get_result("task-res-2")

    assert res == {
        "task_id": "task-res-2",
        "status": "pending",
        "output": None,
    }


@pytest.mark.anyio
async def test_redis_gateway_get_result_failed_outcome_has_no_result() -> None:
    from agent_contracts import (
        CompletionCommand,
        build_terminal_outcome,
    )

    gateway = RedisGateway("redis://localhost:6379/0")
    command = CompletionCommand(
        completion_command_id="cmd-res-fail",
        assignment_id="assign-res-fail",
        execution_kind="initial",
        failure_code="EXECUTION_FAILED",
    )
    outcome, result = build_terminal_outcome(
        command=command,
        task_id="task-res-fail",
        runtime_principal_id="principal-1",
        bound_agent_id="agent-1",
        completed_at="2026-11-20T00:00:00.000000Z",
    )
    assert result is None
    gateway._client.get = AsyncMock(side_effect=[outcome.to_json(), None])

    res = await gateway.get_result("task-res-fail")

    assert res == {
        "task_id": "task-res-fail",
        "status": "failed",
        "output": None,
    }


@pytest.mark.anyio
async def test_redis_gateway_get_result_legacy_result_remains_readable() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    legacy_data = {
        "task_id": "task-legacy-1",
        "status": "completed",
        "output": "Legacy direct-write output",
        "acknowledged": True,
    }
    gateway._client.get = AsyncMock(
        side_effect=[None, json.dumps(legacy_data)]
    )

    res = await gateway.get_result("task-legacy-1")

    assert res == legacy_data


@pytest.mark.anyio
async def test_redis_gateway_get_result_unknown() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.get = AsyncMock(return_value=None)
    gateway.events = AsyncMock(return_value=[])

    res = await gateway.get_result("unknown-task-id")

    assert res is None


def _completed_outcome_and_result(task_id: str = "task-fc-1") -> tuple[str, str]:
    from agent_contracts import (
        CompletionCommand,
        TaskResult,
        build_terminal_outcome,
    )

    command = CompletionCommand(
        completion_command_id="cmd-fc-1",
        assignment_id="assign-fc-1",
        execution_kind="initial",
        completed_result=TaskResult(task_id=task_id, output="canonical output"),
    )
    outcome, result = build_terminal_outcome(
        command=command,
        task_id=task_id,
        runtime_principal_id="principal-1",
        bound_agent_id="agent-1",
        completed_at="2026-11-20T00:00:00.000000Z",
    )
    assert result is not None
    return outcome.to_json(), result.to_json()


@pytest.mark.anyio
async def test_redis_gateway_get_result_completed_missing_result_fails_closed() -> None:
    from orchestrator.domain.task import CorruptAuthoritativeRecordError

    gateway = RedisGateway("redis://localhost:6379/0")
    outcome_json, _ = _completed_outcome_and_result()
    gateway._client.get = AsyncMock(side_effect=[outcome_json, None])

    with pytest.raises(CorruptAuthoritativeRecordError, match="refusing to fabricate output"):
        await gateway.get_result("task-fc-1")


@pytest.mark.anyio
async def test_redis_gateway_get_result_completed_corrupt_result_fails_closed() -> None:
    from orchestrator.domain.task import CorruptAuthoritativeRecordError

    gateway = RedisGateway("redis://localhost:6379/0")
    outcome_json, _ = _completed_outcome_and_result()
    gateway._client.get = AsyncMock(side_effect=[outcome_json, "not-json{{{"])

    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt canonical result"):
        await gateway.get_result("task-fc-1")


@pytest.mark.anyio
async def test_redis_gateway_get_result_completed_wrong_task_result_fails_closed() -> None:
    from agent_contracts import TaskResult
    from orchestrator.domain.task import CorruptAuthoritativeRecordError

    gateway = RedisGateway("redis://localhost:6379/0")
    outcome_json, _ = _completed_outcome_and_result()
    other = TaskResult(task_id="task-elsewhere", output="canonical output")
    gateway._client.get = AsyncMock(side_effect=[outcome_json, other.to_json()])

    with pytest.raises(CorruptAuthoritativeRecordError, match="not bound to"):
        await gateway.get_result("task-fc-1")


@pytest.mark.anyio
async def test_redis_gateway_get_result_completed_non_completed_result_fails_closed() -> None:
    from agent_contracts import TaskResult
    from orchestrator.domain.task import CorruptAuthoritativeRecordError

    gateway = RedisGateway("redis://localhost:6379/0")
    outcome_json, _ = _completed_outcome_and_result()
    non_terminal = TaskResult(task_id="task-fc-1", output="canonical output", status="failed")
    gateway._client.get = AsyncMock(side_effect=[outcome_json, non_terminal.to_json()])

    with pytest.raises(CorruptAuthoritativeRecordError, match="not bound to"):
        await gateway.get_result("task-fc-1")


@pytest.mark.anyio
async def test_redis_gateway_get_result_failed_unexpected_result_fails_closed() -> None:
    from orchestrator.domain.task import CorruptAuthoritativeRecordError

    gateway = RedisGateway("redis://localhost:6379/0")
    from agent_contracts import CompletionCommand, build_terminal_outcome

    command = CompletionCommand(
        completion_command_id="cmd-fc-2",
        assignment_id="assign-fc-2",
        execution_kind="initial",
        failure_code="EXECUTION_FAILED",
    )
    outcome, result = build_terminal_outcome(
        command=command,
        task_id="task-fc-2",
        runtime_principal_id="principal-1",
        bound_agent_id="agent-1",
        completed_at="2026-11-20T00:00:00.000000Z",
    )
    assert result is None
    gateway._client.get = AsyncMock(
        side_effect=[outcome.to_json(), '{"task_id":"task-fc-2"}']
    )

    with pytest.raises(CorruptAuthoritativeRecordError, match="Unexpected canonical result"):
        await gateway.get_result("task-fc-2")


@pytest.mark.anyio
async def test_redis_gateway_get_result_corrupt_outcome_fails_closed() -> None:
    from orchestrator.domain.task import CorruptAuthoritativeRecordError

    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.get = AsyncMock(side_effect=["not-json{{{"])

    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt terminal outcome"):
        await gateway.get_result("task-fc-1")


@pytest.mark.anyio
async def test_redis_gateway_get_result_mismatched_outcome_task_fails_closed() -> None:
    from orchestrator.domain.task import CorruptAuthoritativeRecordError

    gateway = RedisGateway("redis://localhost:6379/0")
    outcome_json, _ = _completed_outcome_and_result(task_id="task-other")
    gateway._client.get = AsyncMock(side_effect=[outcome_json])

    with pytest.raises(CorruptAuthoritativeRecordError, match="not bound to the requested task"):
        await gateway.get_result("task-fc-1")


@pytest.mark.anyio
async def test_redis_gateway_get_result_legacy_wrong_task_fails_closed() -> None:
    from orchestrator.domain.task import CorruptAuthoritativeRecordError

    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.get = AsyncMock(
        side_effect=[None, json.dumps({"task_id": "task-elsewhere", "output": "x"})]
    )

    with pytest.raises(CorruptAuthoritativeRecordError, match="not bound to the requested task"):
        await gateway.get_result("task-legacy-9")


@pytest.mark.anyio
async def test_research_redis_bus_exposes_no_direct_result_write() -> None:
    bus = ResearchRedisBus("redis://localhost:6379/0")
    bus.client.set = AsyncMock()

    # Canonical completion flows through the Orchestrator boundary; the bus
    # exposes no direct result-write seam.
    assert not hasattr(bus, "save")
    bus.client.set.assert_not_awaited()


@pytest.mark.anyio
async def test_redis_gateway_agents_parses_capability_metadata() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.smembers = AsyncMock(return_value={"worker-1", "research-agent-1"})  # type: ignore[method-assign]

    async def mock_hgetall(key: str) -> dict[str, str]:
        if "research" in key:
            return {
                "agent_id": "research-agent-1",
                "agent_type": "research-agent",
                "capabilities": '["research"]',
                "providers": '{"research_executor": "tavily", "inference_provider": "cloudflare"}',
                "research_executor": "tavily",
                "inference_provider": "cloudflare",
                "state": "idle",
                "detail": "Waiting for work",
            }
        return {
            "agent_id": "worker-1",
            "agent_type": "worker",
            "capabilities": '["general"]',
            "providers": "{}",
            "state": "working",
            "detail": "Processing task",
        }

    gateway._client.hgetall = AsyncMock(side_effect=mock_hgetall)  # type: ignore[method-assign]

    agents = await gateway.agents()
    assert len(agents) == 2

    # Sorted by agent_id: research-agent-1, worker-1
    research_agent = agents[0]
    worker = agents[1]

    assert research_agent["agent_id"] == "research-agent-1"
    assert research_agent["agent_type"] == "research-agent"
    assert research_agent["capabilities"] == ["research"]
    assert research_agent["research_executor"] == "tavily"
    assert research_agent["inference_provider"] == "cloudflare"
    assert research_agent["providers"] == {
        "research_executor": "tavily",
        "inference_provider": "cloudflare",
    }
    assert research_agent["state"] == "idle"

    assert worker["agent_id"] == "worker-1"
    assert worker["agent_type"] == "worker"
    assert worker["capabilities"] == ["general"]
    assert worker["providers"] == {}
    assert worker["state"] == "working"


@pytest.mark.anyio
async def test_redis_gateway_agents_legacy_fallback() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.smembers = AsyncMock(return_value={"legacy-agent"})  # type: ignore[method-assign]
    gateway._client.hgetall = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "agent_id": "legacy-agent",
            "state": "idle",
            "detail": "Legacy worker",
        }
    )

    agents = await gateway.agents()
    assert len(agents) == 1
    assert agents[0]["agent_id"] == "legacy-agent"
    assert agents[0]["capabilities"] == []
    assert agents[0]["providers"] == {}
    assert agents[0]["state"] == "idle"


# ==============================================================================
# RedisGateway Task Inference Telemetry Query Tests (Issue #18 Increment 1)
# ==============================================================================


@pytest.mark.anyio
async def test_redis_gateway_task_inference_telemetry_newest_successful() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    telem_cf_failed = InferenceExecutionTelemetry(
        provider_name="cloudflare",
        attempt_index=0,
        latency_ms=120.0,
        status="fallback_eligible_failure",
        failure_reason="provider_unavailable_error",
    )
    telem_gemini_success = InferenceExecutionTelemetry(
        provider_name="gemini",
        attempt_index=1,
        latency_ms=85.5,
        status="succeeded",
        model_name="gemini-2.5-flash",
        finish_reason="stop",
    )

    # xrevrange returns newest entries first
    gateway._client.get = AsyncMock(return_value=None)  # type: ignore[method-assign]
    gateway._client.xrevrange = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            (
                "1000-2",
                {
                    "task_id": "task-test-1",
                    "inference_telemetry": json.dumps(telem_gemini_success.to_dict()),
                },
            ),
            (
                "1000-1",
                {
                    "task_id": "task-test-1",
                    "inference_telemetry": json.dumps(telem_cf_failed.to_dict()),
                },
            ),
        ]
    )

    result = await gateway.task_inference_telemetry("task-test-1")
    assert result is not None
    assert result["task_id"] == "task-test-1"
    assert result["inference_telemetry"]["provider_name"] == "gemini"
    assert result["inference_telemetry"]["status"] == "succeeded"
    assert result["inference_telemetry"]["attempt_index"] == 1
    assert result["inference_telemetry"]["latency_ms"] == 85.5
    gateway._client.xrevrange.assert_awaited_once_with("agent.events")


@pytest.mark.anyio
async def test_redis_gateway_task_inference_telemetry_surfaces_failed_attempts() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.get = AsyncMock(return_value=None)  # type: ignore[method-assign]
    telem_failed = InferenceExecutionTelemetry(
        provider_name="cloudflare",
        attempt_index=0,
        latency_ms=120.0,
        status="fallback_eligible_failure",
        failure_reason="provider_unavailable_error",
    )
    gateway._client.xrevrange = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            (
                "1000-2",
                {
                    "task_id": "task-test-failed",
                    "state": "error",
                },
            ),
            (
                "1000-1",
                {
                    "task_id": "task-test-failed",
                    "inference_telemetry": json.dumps(telem_failed.to_dict()),
                },
            ),
        ]
    )

    result = await gateway.task_inference_telemetry("task-test-failed")
    assert result is not None
    assert result["task_id"] == "task-test-failed"
    assert result["inference_telemetry"] is None
    assert result["final_status"] == "all_providers_failed"
    assert result["history_status"] == "complete"
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["provider_name"] == "cloudflare"
    assert result["attempts"][0]["status"] == "fallback_eligible_failure"
    assert result["attempts"][0]["failure_reason"] == "provider_unavailable_error"


@pytest.mark.anyio
async def test_redis_gateway_task_inference_telemetry_ignores_other_tasks() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.get = AsyncMock(return_value=None)  # type: ignore[method-assign]
    telem_success = InferenceExecutionTelemetry(
        provider_name="gemini",
        attempt_index=0,
        latency_ms=90.0,
        status="succeeded",
        model_name="gemini-2.5-flash",
    )
    gateway._client.xrevrange = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            (
                "1000-1",
                {
                    "task_id": "task-other",
                    "inference_telemetry": json.dumps(telem_success.to_dict()),
                },
            )
        ]
    )

    result = await gateway.task_inference_telemetry("task-target")
    assert result is None


@pytest.mark.anyio
async def test_redis_gateway_task_inference_telemetry_handles_malformed_and_invalid() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.get = AsyncMock(return_value=None)  # type: ignore[method-assign]
    gateway._client.xrevrange = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            # Malformed JSON
            (
                "1000-3",
                {
                    "task_id": "task-malformed",
                    "inference_telemetry": "{invalid json",
                },
            ),
            # Non-dict JSON (e.g. JSON array or int)
            (
                "1000-2",
                {
                    "task_id": "task-malformed",
                    "inference_telemetry": json.dumps(["not", "an", "object"]),
                },
            ),
            # Dict missing required fields according to InferenceExecutionTelemetry
            (
                "1000-1",
                {
                    "task_id": "task-malformed",
                    "inference_telemetry": json.dumps({"incomplete": "dict"}),
                },
            ),
        ]
    )

    result = await gateway.task_inference_telemetry("task-malformed")
    assert result is not None
    assert result["task_id"] == "task-malformed"
    assert result["inference_telemetry"] is None
    assert result["history_status"] == "partial"
    assert result["attempts"] == []


@pytest.mark.anyio
async def test_redis_gateway_task_inference_telemetry_returns_none_when_empty() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.get = AsyncMock(return_value=None)  # type: ignore[method-assign]
    gateway._client.xrevrange = AsyncMock(return_value=[])  # type: ignore[method-assign]

    result = await gateway.task_inference_telemetry("task-any")
    assert result is None


@pytest.mark.anyio
async def test_redis_gateway_event_stream_read_failure_raises_telemetry_storage_error() -> None:
    from orchestrator.application.ports import TelemetryStorageError

    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.get = AsyncMock(return_value=None)  # type: ignore[method-assign]
    gateway._client.xrevrange = AsyncMock(side_effect=redis.ConnectionError("Connection refused"))  # type: ignore[method-assign]

    with pytest.raises(TelemetryStorageError, match="Event stream storage unavailable"):
        await gateway.task_inference_telemetry("task-stream-fail")


@pytest.mark.anyio
async def test_redis_gateway_durable_key_read_failure_raises_telemetry_storage_error() -> None:
    from orchestrator.application.ports import TelemetryStorageError

    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.get = AsyncMock(side_effect=redis.RedisError("Redis cluster error"))  # type: ignore[method-assign]

    with pytest.raises(TelemetryStorageError, match="Task durable state storage unavailable"):
        await gateway.task_inference_telemetry("task-durable-fail")


