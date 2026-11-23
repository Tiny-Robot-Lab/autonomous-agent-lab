import asyncio
import logging
import os
import sys
from unittest.mock import DEFAULT
from uuid import uuid4

from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    HeartbeatRunner,
    TaskRequested,
    WakeUpHint,
    parse_heartbeat_interval,
)
from agent_infrastructure import CompletionSubmissionError, OrchestratorClient

from analysis_agent.application.processor import AnalysisExecutionError, AnalysisProcessor
from analysis_agent.infrastructure.deterministic_analyzer import (
    DeterministicMaterialAnalyzer,
)
from analysis_agent.infrastructure.redis_bus import RedisBus

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("analysis_agent")


async def run_analysis_agent(heartbeat_interval: float | None = None) -> None:
    agent_id = os.getenv("AGENT_ID", "analysis-agent-1")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    task_queue = os.getenv("TASK_QUEUE", "agent.tasks.analysis")
    hint_queue = os.getenv("HINT_QUEUE")
    orchestrator_base_url = os.getenv("ORCHESTRATOR_BASE_URL", "http://orchestrator:8080")
    runtime_token = os.getenv("RUNTIME_TOKEN", "")
    if not runtime_token.strip():
        raise RuntimeError(
            "RUNTIME_TOKEN must be set: the analysis agent claims and completes turns "
            "only through the authenticated Orchestrator completion boundary"
        )
    interval = (
        heartbeat_interval
        if heartbeat_interval is not None
        else parse_heartbeat_interval(os.getenv("HEARTBEAT_INTERVAL_SECONDS"))
    )

    bus_kwargs: dict[str, str] = {"queue_name": task_queue}
    if hint_queue:
        bus_kwargs["hint_queue"] = hint_queue
    bus = RedisBus(redis_url, **bus_kwargs)
    orchestrator = OrchestratorClient(orchestrator_base_url, runtime_token)
    processor = AnalysisProcessor(
        agent_id=agent_id,
        bus=bus,
        analyzer=DeterministicMaterialAnalyzer(),
        completion_client=orchestrator,
    )
    metadata = AgentCapabilityMetadata(
        agent_id=agent_id,
        agent_type="analysis-agent",
        capabilities=["analysis"],
        providers={"analysis_engine": "deterministic"},
        dispatch_target=task_queue,
    )
    heartbeat_runner = HeartbeatRunner(
        agent_id=agent_id,
        publish_heartbeat=bus.heartbeat,
        get_state=lambda: bus.current_state,
        interval_seconds=interval,
    )

    logger.info(
        "Initializing analysis agent %s on queue %s%s (heartbeat: %.1fs)...",
        agent_id,
        task_queue,
        f" and hint queue {hint_queue}" if hint_queue else "",
        interval,
    )

    try:
        await bus.register(metadata)
        await bus.publish(AgentEvent.create(agent_id, "idle", "Waiting for work"))
        heartbeat_runner.start()
        logger.info("Analysis agent %s registered and waiting for tasks", agent_id)

        while True:
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
                    logger.warning(
                        "Ignoring hint on analysis hint queue: missing or empty task_id: %s",
                        hint,
                    )
                    continue
                execution_kind = getattr(hint, "execution_kind", None)
                if execution_kind != "delegated_child":
                    logger.warning(
                        "Ignoring unsupported execution_kind '%s' on analysis hint queue for task %s",
                        execution_kind,
                        task_id,
                    )
                    continue
                hint_cap = getattr(hint, "capability", "analysis")
                if hint_cap != "analysis":
                    logger.warning(
                        "Ignoring hint for task %s: capability mismatch %s",
                        task_id,
                        hint_cap,
                    )
                    continue

                logger.info("Processing analysis wake-up hint for task %s", task_id)
                try:
                    request_id = f"req-{task_id}-{uuid4().hex[:8]}"
                    assignment, proof = await orchestrator.claim_task(
                        task_id=task_id,
                        request_id=request_id,
                        capability="analysis",
                        execution_kind="delegated_child",
                    )
                    await orchestrator.acknowledge_assignment(
                        task_id, assignment["assignment_id"], proof
                    )
                    child_task_input = await orchestrator.get_child_task_input(
                        task_id, assignment["assignment_id"], proof
                    )
                    task = TaskRequested(
                        task_id=task_id,
                        root_task_id=assignment["root_task_id"],
                        parent_task_id=assignment.get("parent_task_id"),
                        prompt=child_task_input.prompt,
                        created_at=child_task_input.created_at,
                    )
                    await processor.run_once(
                        task,
                        assignment_id=assignment["assignment_id"],
                        assignment_proof=proof,
                        execution_kind="delegated_child",
                    )
                except CompletionSubmissionError as exc:
                    logger.warning(
                        "Analysis task %s hint claim/acknowledge failed: %s; ready for next signal",
                        task_id,
                        exc,
                    )
                except AnalysisExecutionError as exc:
                    logger.warning(
                        "Analysis task %s processing failed: %s; ready for next signal",
                        task_id,
                        exc.original_error,
                    )
            else:
                task = signal
                logger.info("Dispatching task %s to analysis processor", task.task_id)
                try:
                    request_id = f"req-{task.task_id}-{uuid4().hex[:8]}"
                    assignment, proof = await orchestrator.claim_task(
                        task.task_id, request_id, "analysis", "initial"
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
                        "Analysis task %s claim/acknowledge failed with error: %s; ready for next task",
                        task.task_id,
                        exc,
                    )
                except AnalysisExecutionError as exc:
                    logger.warning(
                        "Analysis task %s processing failed with error: %s; ready for next task",
                        task.task_id,
                        exc.original_error,
                    )
    finally:
        logger.info("Shutting down analysis agent %s, closing bus...", agent_id)
        await heartbeat_runner.stop()
        await bus.aclose()
        logger.info("Analysis agent %s shutdown complete", agent_id)


def main() -> None:
    try:
        asyncio.run(run_analysis_agent())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Analysis agent process interrupted. Exiting.")


if __name__ == "__main__":
    main()
