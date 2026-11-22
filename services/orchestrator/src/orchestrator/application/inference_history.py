"""Task-scoped inference fallback attempt history projection (Issue #189, ADR 0005)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent_contracts import FallbackTransition, InferenceExecutionTelemetry


@dataclass(frozen=True)
class TaskEventObservation:
    """Internal normalized observation from the event stream for task projection.

    Security Boundary:
        This internal type strictly excludes AgentEvent.detail to prevent prompt,
        output, or raw exception leakage. Only validated structured telemetry
        and lifecycle signals are observed.
    """

    event_id: str
    task_id: str
    stream_position: tuple[int, int] = (0, 0)
    stream_order: int = 0
    state: str | None = None
    inference_telemetry: InferenceExecutionTelemetry | None = None
    fallback_transition: FallbackTransition | None = None
    is_terminal_lifecycle: bool = False
    is_terminal_error: bool = False
    is_malformed: bool = False
    turn_id: str | None = None
    assignment_id: str | None = None
    execution_kind: str | None = None
    continuation_version: int | None = None


@dataclass(frozen=True)
class TaskInferenceHistory:
    """Canonical task-scoped inference execution and fallback history."""

    task_id: str
    history_status: str
    final_status: str | None
    inference_telemetry: dict[str, Any] | None
    attempts: list[dict[str, Any]]
    fallback_transitions: list[dict[str, Any]]
    turns: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "task_id": self.task_id,
            "history_status": self.history_status,
            "final_status": self.final_status,
            "inference_telemetry": self.inference_telemetry,
            "attempts": self.attempts,
            "fallback_transitions": self.fallback_transitions,
        }
        if self.turns is not None:
            data["turns"] = self.turns
        return data


def _are_telemetries_equivalent(
    a: InferenceExecutionTelemetry, b: InferenceExecutionTelemetry
) -> bool:
    """Check if two attempt records are semantically equivalent for deduplication."""
    return (
        a.attempt_index == b.attempt_index
        and a.provider_name == b.provider_name
        and a.status == b.status
        and a.latency_ms == b.latency_ms
        and a.model_name == b.model_name
        and a.failure_reason == b.failure_reason
        and a.status_code == b.status_code
        and a.finish_reason == b.finish_reason
        and a.input_tokens == b.input_tokens
        and a.output_tokens == b.output_tokens
        and a.total_tokens == b.total_tokens
        and getattr(a, "turn_id", None) == getattr(b, "turn_id", None)
        and getattr(a, "assignment_id", None) == getattr(b, "assignment_id", None)
        and getattr(a, "execution_kind", None) == getattr(b, "execution_kind", None)
        and getattr(a, "continuation_version", None) == getattr(b, "continuation_version", None)
    )


def _are_transitions_equivalent(a: FallbackTransition, b: FallbackTransition) -> bool:
    """Check if two fallback transitions are semantically equivalent for deduplication."""
    return (
        a.attempt_index == b.attempt_index
        and a.failed_provider == b.failed_provider
        and a.target_provider == b.target_provider
        and a.reason == b.reason
        and getattr(a, "turn_id", None) == getattr(b, "turn_id", None)
        and getattr(a, "assignment_id", None) == getattr(b, "assignment_id", None)
        and getattr(a, "execution_kind", None) == getattr(b, "execution_kind", None)
        and getattr(a, "continuation_version", None) == getattr(b, "continuation_version", None)
    )


def project_task_inference_history(
    task_id: str,
    observations: Sequence[TaskEventObservation],
    task_exists_durable: bool = False,
    task_completed: bool = False,
    task_failed: bool = False,
) -> dict[str, Any] | None:
    """Project task-scoped inference history from normalized stream observations and durable evidence.

    Security Boundary:
        This projection strictly receives normalized TaskEventObservation instances
        without AgentEvent.detail to prevent prompt, output, or raw exception leakage.
        It performs pure projection logic without raw Redis or JSON decoding.

    Returns:
        A dictionary representation of TaskInferenceHistory, or None if the
        task ID is completely unknown / non-existent.
    """
    clean_task_id = str(task_id).strip()
    if not clean_task_id:
        return None

    matching_obs = [obs for obs in observations if obs.task_id == clean_task_id]
    has_matching_events = bool(matching_obs)

    task_exists = has_matching_events or task_exists_durable
    if not task_exists:
        return None

    raw_attempts: list[tuple[TaskEventObservation, InferenceExecutionTelemetry]] = []
    raw_transitions: list[tuple[TaskEventObservation, FallbackTransition]] = []
    terminal_error_event = False
    has_malformed_event = False

    for obs in matching_obs:
        if obs.is_malformed:
            has_malformed_event = True
        if obs.is_terminal_error:
            terminal_error_event = True
        if obs.inference_telemetry is not None:
            raw_attempts.append((obs, obs.inference_telemetry))
        if obs.fallback_transition is not None:
            raw_transitions.append((obs, obs.fallback_transition))

    has_terminal_failure = bool(terminal_error_event or task_failed)
    has_terminal_completion = bool(task_completed)

    # Zero-attempt scenario
    if not raw_attempts and not raw_transitions:
        if has_matching_events:
            # Retained matching task events exist, but zero valid inference attempts recorded.
            # Terminal failure takes precedence over any idle/completion signal.
            final_status: str | None = None
            if has_terminal_failure:
                final_status = "failed"
            elif has_terminal_completion:
                final_status = "succeeded"
            else:
                # Idle alone without durable completed result only indicates agent availability
                final_status = None

            status = "partial" if has_malformed_event else "no_inference_recorded"
            return TaskInferenceHistory(
                task_id=clean_task_id,
                history_status=status,
                final_status=final_status,
                inference_telemetry=None,
                attempts=[],
                fallback_transitions=[],
                turns=[],
            ).to_dict()
        elif task_exists_durable:
            # Zero matching events in stream.
            # Only terminal durable completion/failure evidence supports aged_out.
            # Routing-only or pending status without retained events must not claim aged_out.
            if has_terminal_completion or has_terminal_failure:
                final_status = "succeeded" if has_terminal_completion else "failed"
                history_status = "aged_out"
            else:
                final_status = None
                history_status = "no_inference_recorded"

            return TaskInferenceHistory(
                task_id=clean_task_id,
                history_status=history_status,
                final_status=final_status,
                inference_telemetry=None,
                attempts=[],
                fallback_transitions=[],
                turns=[],
            ).to_dict()
        else:
            return None

    # Deduplicate and sort attempts and transitions by turn
    def _turn_key(obs: TaskEventObservation, telem_or_trans: Any) -> tuple[str | None, str | None]:
        t_id = getattr(telem_or_trans, "turn_id", None) or obs.turn_id
        a_id = getattr(telem_or_trans, "assignment_id", None) or obs.assignment_id
        return (t_id, a_id)

    turn_keys: list[tuple[str | None, str | None]] = []
    attempts_by_turn: dict[
        tuple[str | None, str | None],
        list[tuple[TaskEventObservation, InferenceExecutionTelemetry]],
    ] = {}
    for obs, telem in raw_attempts:
        tk = _turn_key(obs, telem)
        if tk not in attempts_by_turn:
            turn_keys.append(tk)
            attempts_by_turn[tk] = []
        attempts_by_turn[tk].append((obs, telem))

    transitions_by_turn: dict[
        tuple[str | None, str | None],
        list[tuple[TaskEventObservation, FallbackTransition]],
    ] = {}
    for obs, trans in raw_transitions:
        tk = _turn_key(obs, trans)
        if tk not in attempts_by_turn and tk not in transitions_by_turn:
            turn_keys.append(tk)
        transitions_by_turn.setdefault(tk, []).append((obs, trans))

    if not turn_keys:
        turn_keys.append((None, None))

    has_conflict = False
    has_gap = False
    final_attempts: list[InferenceExecutionTelemetry] = []
    final_transitions: list[FallbackTransition] = []
    turns_projection: list[dict[str, Any]] = []

    for tk in turn_keys:
        turn_raw_attempts = attempts_by_turn.get(tk, [])
        turn_raw_trans = transitions_by_turn.get(tk, [])

        turn_attempts_by_index: dict[
            int, list[tuple[TaskEventObservation, InferenceExecutionTelemetry]]
        ] = {}
        for obs, telem in turn_raw_attempts:
            turn_attempts_by_index.setdefault(telem.attempt_index, []).append((obs, telem))

        turn_deduped_attempts: list[tuple[TaskEventObservation, InferenceExecutionTelemetry]] = []
        for records in turn_attempts_by_index.values():
            unique_records: list[tuple[TaskEventObservation, InferenceExecutionTelemetry]] = []
            for obs, cand in records:
                if not any(_are_telemetries_equivalent(cand, u[1]) for u in unique_records):
                    unique_records.append((obs, cand))

            if len(unique_records) > 1:
                has_conflict = True
                turn_deduped_attempts.extend(unique_records)
            elif unique_records:
                turn_deduped_attempts.append(unique_records[0])

        turn_deduped_attempts.sort(
            key=lambda item: (
                item[1].attempt_index,
                item[0].stream_position,
                item[0].stream_order,
            )
        )
        turn_final_attempts = [item[1] for item in turn_deduped_attempts]

        turn_transitions_by_index: dict[
            int, list[tuple[TaskEventObservation, FallbackTransition]]
        ] = {}
        for obs, trans in turn_raw_trans:
            turn_transitions_by_index.setdefault(trans.attempt_index, []).append((obs, trans))

        turn_deduped_transitions: list[tuple[TaskEventObservation, FallbackTransition]] = []
        for records in turn_transitions_by_index.values():
            unique_records: list[tuple[TaskEventObservation, FallbackTransition]] = []
            for obs, cand in records:
                if not any(_are_transitions_equivalent(cand, u[1]) for u in unique_records):
                    unique_records.append((obs, cand))

            if len(unique_records) > 1:
                has_conflict = True
                turn_deduped_transitions.extend(unique_records)
            elif unique_records:
                turn_deduped_transitions.append(unique_records[0])

        turn_deduped_transitions.sort(
            key=lambda item: (
                item[1].attempt_index,
                item[0].stream_position,
                item[0].stream_order,
            )
        )
        turn_final_transitions = [item[1] for item in turn_deduped_transitions]

        # Evaluate completeness and gaps within this turn
        if not turn_final_attempts and (turn_final_transitions or len(turn_keys) == 1):
            has_gap = True
        elif turn_final_attempts:
            unique_indices = sorted({a.attempt_index for a in turn_final_attempts})
            if unique_indices[0] != 0 or unique_indices != list(range(len(unique_indices))):
                has_gap = True

        for trans in turn_final_transitions:
            source_attempt = next(
                (a for a in turn_final_attempts if a.attempt_index == trans.attempt_index), None
            )
            target_attempt = next(
                (a for a in turn_final_attempts if a.attempt_index == trans.attempt_index + 1), None
            )

            if source_attempt is None or source_attempt.provider_name != trans.failed_provider:
                has_gap = True
            if target_attempt is None or target_attempt.provider_name != trans.target_provider:
                has_gap = True

        final_attempts.extend(turn_final_attempts)
        final_transitions.extend(turn_final_transitions)

        ekind = None
        cver = None
        for a in turn_final_attempts:
            if getattr(a, "execution_kind", None):
                ekind = a.execution_kind
            if getattr(a, "continuation_version", None) is not None:
                cver = a.continuation_version

        turns_projection.append(
            {
                "turn_id": tk[0],
                "assignment_id": tk[1],
                "execution_kind": ekind,
                "continuation_version": cver,
                "attempts": [a.to_dict() for a in turn_final_attempts],
                "fallback_transitions": [t.to_dict() for t in turn_final_transitions],
            }
        )

    # Evaluate inference outcome and winning telemetry
    succeeded_attempts = [a for a in final_attempts if a.status == "succeeded"]
    winning_telemetry: InferenceExecutionTelemetry | None = None

    if succeeded_attempts:
        winning_telemetry = succeeded_attempts[-1]
        # Terminal failure takes precedence over successful inference if the task execution failed
        if has_terminal_failure:
            final_status = "failed"
        else:
            final_status = "succeeded"
    else:
        winning_telemetry = None
        all_attempts_failed = bool(final_attempts) and all(
            a.status in ("fallback_eligible_failure", "failed") for a in final_attempts
        )
        if has_terminal_failure and all_attempts_failed:
            if all(a.status == "fallback_eligible_failure" for a in final_attempts):
                final_status = "all_providers_failed"
            else:
                final_status = "failed"
        elif has_terminal_failure:
            final_status = "failed"
        else:
            # Without terminal failure evidence, do not infer all_providers_failed
            final_status = None

    # Terminal lifecycle confirmation
    if final_status == "succeeded":
        is_terminal_confirmed = has_terminal_completion and not has_terminal_failure
    elif final_status in ("all_providers_failed", "failed"):
        is_terminal_confirmed = has_terminal_failure
    else:
        is_terminal_confirmed = False

    if has_malformed_event or has_conflict or has_gap or not is_terminal_confirmed:
        history_status = "partial"
    else:
        history_status = "complete"

    return TaskInferenceHistory(
        task_id=clean_task_id,
        history_status=history_status,
        final_status=final_status,
        inference_telemetry=winning_telemetry.to_dict() if winning_telemetry else None,
        attempts=[a.to_dict() for a in final_attempts],
        fallback_transitions=[t.to_dict() for t in final_transitions],
        turns=turns_projection,
    ).to_dict()
