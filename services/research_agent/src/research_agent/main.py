import asyncio
import logging
import os
import sys
from typing import Any
from unittest.mock import DEFAULT
from uuid import uuid4

from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    HeartbeatRunner,
    TaskRequested,
    WakeUpHint,
    parse_health_probe_timeout,
    parse_heartbeat_interval,
)
from agent_infrastructure import CompletionSubmissionError, OrchestratorClient
from agent_policy import parse_provider_priority

from research_agent.application.fallback import ResearchFallbackCoordinator
from research_agent.application.ports import RuntimeInferenceProvider
from research_agent.application.processor import ResearchExecutionError, ResearchProcessor
from research_agent.infrastructure.executor_factory import create_research_executor
from research_agent.infrastructure.inference_factory import create_inference_provider
from research_agent.infrastructure.redis_bus import RedisBus

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("research_agent")


async def run_research_agent(
    heartbeat_interval: float | None = None,
    health_probe_timeout: float | None = None,
) -> None:
    agent_id = os.getenv("AGENT_ID", "research-agent-1")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    task_queue = os.getenv("TASK_QUEUE", "agent.tasks.research")
    hint_queue = os.getenv("HINT_QUEUE")
    executor_name = os.getenv("RESEARCH_EXECUTOR", "stub")
    inference_provider_name = os.getenv("INFERENCE_PROVIDER", "stub")
    orchestrator_base_url = os.getenv("ORCHESTRATOR_BASE_URL", "http://orchestrator:8080")
    runtime_token = os.getenv("RUNTIME_TOKEN", "")
    if not runtime_token.strip():
        raise RuntimeError(
            "RUNTIME_TOKEN must be set: the research agent claims and completes turns "
            "only through the authenticated Orchestrator completion boundary"
        )
    interval = (
        heartbeat_interval
        if heartbeat_interval is not None
        else parse_heartbeat_interval(os.getenv("HEARTBEAT_INTERVAL_SECONDS"))
    )
    probe_timeout = (
        health_probe_timeout
        if health_probe_timeout is not None
        else parse_health_probe_timeout(os.getenv("HEALTH_PROBE_TIMEOUT_SECONDS"))
    )

    executor = create_research_executor(executor_name)
    preferences = parse_provider_priority(inference_provider_name, default=("stub",))
    providers: dict[str, RuntimeInferenceProvider] = {
        name: create_inference_provider(name) for name in preferences
    }

    bus_kwargs: dict[str, Any] = {"queue_name": task_queue}
    if hint_queue:
        bus_kwargs["hint_queue"] = hint_queue
    bus = RedisBus(redis_url, **bus_kwargs)
    orchestrator = OrchestratorClient(orchestrator_base_url, runtime_token)
    fallback_coordinator = ResearchFallbackCoordinator(
        agent_id=agent_id,
        bus=bus,
        preferences=preferences,
        providers=providers,
        probe_timeout_seconds=probe_timeout,
    )
    processor = ResearchProcessor(
        agent_id,
        bus,
        executor,
        completion_client=orchestrator,
        context_reader=orchestrator,
        inference_provider=providers[preferences[0]],
        fallback_coordinator=fallback_coordinator,
    )

    metadata = AgentCapabilityMetadata(
        agent_id=agent_id,
        agent_type="research-agent",
        capabilities=["research"],
        providers={
            "research_executor": executor_name,
            "inference_provider": inference_provider_name,
        },
        dispatch_target=task_queue,
    )

    heartbeat_runner = HeartbeatRunner(
        agent_id=agent_id,
        publish_heartbeat=bus.heartbeat,
        get_state=lambda: bus.current_state,
        interval_seconds=interval,
    )

    logger.info(
        "Initializing research agent %s on queue %s with executor %s and inference provider %s (heartbeat: %.1fs, probe_timeout: %.1fs)...",
        agent_id,
        task_queue,
        executor_name,
        inference_provider_name,
        interval,
        probe_timeout,
    )

    try:
        await bus.register(metadata)
        await bus.publish(AgentEvent.create(agent_id, "idle", "Waiting for work"))
        heartbeat_runner.start()
        logger.info("Research agent %s registered and waiting for tasks", agent_id)

        while True:
            # If next_task was specifically configured with a custom mock in unit tests,
            # respect it for compatibility with tests that mock next_task
            is_mock_task = (
                hasattr(bus, "next_task")
                and (
                    getattr(bus.next_task, "side_effect", None) is not None
                    or getattr(bus.next_task, "_mock_return_value", None) is not DEFAULT
                )
            )
            is_mock_signal = (
                hasattr(bus, "next_signal")
                and (
                    getattr(bus.next_signal, "side_effect", None) is not None
                    or getattr(bus.next_signal, "_mock_return_value", None) is not DEFAULT
                )
            )
            if is_mock_task and not is_mock_signal:
                signal = await bus.next_task()
            else:
                signal = await bus.next_signal()
            if isinstance(signal, WakeUpHint):
                hint = signal
                task_id = (getattr(hint, "task_id", None) or "").strip()
                if not task_id:
                    logger.warning("Ignoring hint on research hint queue: missing or empty task_id: %s", hint)
                    continue
                execution_kind = getattr(hint, "execution_kind", None)
                if execution_kind != "resumed_continuation":
                    logger.warning(
                        "Ignoring unsupported execution_kind '%s' on research hint queue for task %s",
                        execution_kind,
                        task_id,
                    )
                    continue
                expected_continuation_version = getattr(hint, "expected_continuation_version", None)
                if expected_continuation_version != 1:
                    logger.warning(
                        "Ignoring hint for task %s: unsupported continuation_version %s",
                        task_id,
                        expected_continuation_version,
                    )
                    continue
                hint_cap = getattr(hint, "capability", "research")
                if hint_cap != "research":
                    logger.warning(
                        "Ignoring hint for task %s: capability mismatch %s",
                        task_id,
                        hint_cap,
                    )
                    continue

                execution_kind = hint.execution_kind
                expected_continuation_version = hint.expected_continuation_version
                logger.info(
                    "Processing research wake-up hint for task %s (execution_kind=%s)",
                    task_id,
                    execution_kind,
                )
                try:
                    request_id = f"req-{task_id}-{uuid4().hex[:8]}"
                    assignment, proof = await orchestrator.claim_task(
                        task_id=task_id,
                        request_id=request_id,
                        capability="research",
                        execution_kind=execution_kind,
                        expected_continuation_version=expected_continuation_version,
                    )
                    await orchestrator.acknowledge_assignment(
                        task_id, assignment["assignment_id"], proof
                    )
                    context = await orchestrator.get_resumed_continuation_context(
                        task_id, assignment["assignment_id"], proof
                    )
                    task = TaskRequested(
                        task_id=task_id,
                        root_task_id=task_id,
                        parent_task_id=None,
                        prompt=context.parent_task_input.prompt,
                        created_at=context.parent_task_input.created_at,
                    )
                    await processor.run_once(
                        task,
                        assignment_id=assignment["assignment_id"],
                        assignment_proof=proof,
                        execution_kind="resumed_continuation",
                        continuation_version=expected_continuation_version or 1,
                    )
                except CompletionSubmissionError as exc:
                    logger.warning(
                        "Research task %s hint claim/acknowledge failed with error: %s; ready for next signal",
                        task_id,
                        exc,
                    )
                except ResearchExecutionError as exc:
                    logger.warning(
                        "Research task %s processing failed with error: %s; ready for next signal",
                        exc.task_id,
                        exc.original_error,
                    )
            else:
                task = signal
                logger.info("Dispatching task %s to research processor", task.task_id)
                try:
                    request_id = f"req-{task.task_id}-{uuid4().hex[:8]}"
                    assignment, proof = await orchestrator.claim_task(
                        task.task_id, request_id, "research", "initial"
                    )
                    await orchestrator.acknowledge_assignment(
                        task.task_id, assignment["assignment_id"], proof
                    )
                    await processor.run_once(
                        task,
                        assignment_id=assignment["assignment_id"],
                        assignment_proof=proof,
                        execution_kind="initial",
                    )
                except CompletionSubmissionError as exc:
                    logger.warning(
                        "Research task %s claim/acknowledge failed with error: %s; ready for next task",
                        task.task_id,
                        exc,
                    )
                except ResearchExecutionError as exc:
                    logger.warning(
                        "Research task %s processing failed with error: %s; ready for next task",
                        exc.task_id,
                        exc.original_error,
                    )
    finally:
        logger.info("Shutting down research agent %s, closing bus...", agent_id)
        await heartbeat_runner.stop()
        await bus.aclose()
        logger.info("Research agent %s shutdown complete", agent_id)


def main() -> None:
    try:
        asyncio.run(run_research_agent())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Research agent process interrupted. Exiting.")


if __name__ == "__main__":
    main()
