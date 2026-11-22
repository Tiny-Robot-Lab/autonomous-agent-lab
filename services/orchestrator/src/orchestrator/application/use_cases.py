import hmac
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from agent_contracts import (
    ActiveClaim,
    AgentEvent,
    ArtifactReference,
    AssignmentYieldRecord,
    ClaimTaskResponse,
    CompletionCommand,
    CompletionResponse,
    DelegationGrant,
    DelegationProposalRecord,
    DelegationProposalRequest,
    DelegationProposalResponse,
    RoutingDecision,
    RoutingProvenance,
    StageArtifact,
    TaskAssignment,
    TaskContinuation,
    TaskInput,
    TaskInputReference,
    TaskRequested,
    TaskResult,
    WakeUpHint,
    compute_proposal_fingerprint,
    normalize_canonical_utc_timestamp,
)

from orchestrator.domain.advisor import (
    AdvisorError,
    AdvisorRecommendation,
    AdvisorTimeoutError,
    AdvisorUnavailableError,
)
from orchestrator.domain.assessment import RoutingAssessment, RoutingClarity
from orchestrator.domain.capability import AgentCapability, CapabilityUnavailableError
from orchestrator.domain.principal import (
    RuntimePrincipal,
    compute_assignment_verifier_digest,
    derive_assignment_proof,
    derive_proof_subkeys,
    encode_canonical_proof_identity,
    parse_master_secret,
    verify_assignment_proof,
)
from orchestrator.domain.router import DeterministicCapabilityRouter
from orchestrator.domain.selection import (
    CandidateSelectionPolicy,
    DeterministicSelectionPolicy,
)
from orchestrator.domain.task import (
    CorruptAuthoritativeRecordError,
    DelegationDepthExceededError,
    ParentTaskNotFoundError,
    Task,
)

from .ports import (
    AssessingCapabilityRouter,
    AuthoritativeTaskStore,
    CandidateFinder,
    CapabilityMismatchError,
    CapabilityRouter,
    ClaimDeliveryLeaseExpiredError,
    ClaimNotFoundError,
    CompletionReplayAuthorizationError,
    DelegationReplayAuthorizationError,
    EventPublisher,
    InvalidClaimProofError,
    RoutingAdvisor,
    RoutingRecorder,
    TaskAlreadyClaimedError,
    TaskDispatcher,
    TaskNotClaimableError,
    TaskQueue,
    TurnCompletionStore,
)


class SubmitTask:
    def __init__(
        self,
        task_store: AuthoritativeTaskStore,
        queue: TaskQueue | TaskDispatcher | None = None,
        router: CapabilityRouter | None = None,
        candidate_finder: CandidateFinder | None = None,
        events: EventPublisher | None = None,
        selection_policy: CandidateSelectionPolicy | None = None,
        dispatcher: TaskDispatcher | None = None,
        routing_recorder: RoutingRecorder | None = None,
        advisor: RoutingAdvisor | None = None,
    ) -> None:
        if task_store is None:
            raise TypeError("SubmitTask requires an AuthoritativeTaskStore")
        if not isinstance(task_store, AuthoritativeTaskStore):
            raise TypeError(
                f"task_store must implement AuthoritativeTaskStore, got {type(task_store).__name__}"
            )
        self._task_store = task_store
        self._dispatcher: TaskDispatcher | None = dispatcher or (
            queue if isinstance(queue, TaskDispatcher) else None
        )
        self._queue: TaskQueue | None = queue if isinstance(queue, TaskQueue) else None
        self._router = router or DeterministicCapabilityRouter()
        self._candidate_finder = candidate_finder
        self._events = events
        self._selection_policy = selection_policy or DeterministicSelectionPolicy()
        self._routing_recorder: RoutingRecorder | None = routing_recorder
        self._advisor = advisor

    async def execute(
        self,
        prompt: str,
        capability: AgentCapability | None = None,
    ) -> TaskRequested:
        task = Task(prompt=prompt)
        is_explicit_override = capability is not None

        if is_explicit_override:
            resolved_capability = capability
            provenance = RoutingProvenance(
                is_explicit_override=True,
                final_selection_reason="Explicit capability override provided by caller",
            )
            if resolved_capability == AgentCapability.GENERAL:
                prompt_classification = resolved_capability.value
            elif hasattr(self._router, "detect_intent") and (intent := self._router.detect_intent(task.prompt)) is not None:
                prompt_classification = intent.value
            else:
                prompt_classification = resolved_capability.value
        elif isinstance(self._router, AssessingCapabilityRouter):
            res = self._router.assess(task.prompt)
            if not isinstance(res, RoutingAssessment):
                raise TypeError(
                    f"Router {type(self._router).__name__}.assess() must return RoutingAssessment, got {type(res).__name__}"
                )

            signal_types = tuple(s.signal_type.value for s in res.signals)
            if res.clarity == RoutingClarity.CLEAR:
                resolved_capability = res.provisional_capability
                provenance = RoutingProvenance(
                    is_explicit_override=False,
                    assessment_clarity=res.clarity.value,
                    provisional_capability=res.provisional_capability.value,
                    advisor_consulted=False,
                    advisor_verdict=None,
                    fallback_used=False,
                    final_selection_reason=res.reason,
                    deterministic_assessment_reason=res.reason,
                    deterministic_signal_types=signal_types,
                    advisor_reason="",
                )
            else:
                # RoutingClarity.AMBIGUOUS
                if self._advisor is not None:
                    try:
                        recommendation = await self._advisor.advise(task.prompt, res)
                        if not isinstance(recommendation, AdvisorRecommendation):
                            raise TypeError(
                                f"Advisor {type(self._advisor).__name__}.advise() must return AdvisorRecommendation, got {type(recommendation).__name__}"
                            )

                        if recommendation.recommended_capability == res.provisional_capability:
                            resolved_capability = res.provisional_capability
                            verdict = "agreed"
                        else:
                            resolved_capability = recommendation.recommended_capability
                            verdict = "changed"

                        provenance = RoutingProvenance(
                            is_explicit_override=False,
                            assessment_clarity=res.clarity.value,
                            provisional_capability=res.provisional_capability.value,
                            advisor_consulted=True,
                            advisor_verdict=verdict,
                            fallback_used=False,
                            final_selection_reason=recommendation.reason,
                            deterministic_assessment_reason=res.reason,
                            deterministic_signal_types=signal_types,
                            advisor_reason=recommendation.reason,
                        )
                    except (AdvisorError, TimeoutError) as exc:
                        resolved_capability = res.provisional_capability
                        if isinstance(exc, AdvisorUnavailableError):
                            verdict = "unavailable"
                        elif isinstance(exc, (AdvisorTimeoutError, TimeoutError)):
                            verdict = "timeout"
                        else:
                            verdict = "failed"

                        provenance = RoutingProvenance(
                            is_explicit_override=False,
                            assessment_clarity=res.clarity.value,
                            provisional_capability=res.provisional_capability.value,
                            advisor_consulted=True,
                            advisor_verdict=verdict,
                            fallback_used=True,
                            final_selection_reason=f"Advisor {verdict} ({exc}); falling back to deterministic provisional capability",
                            deterministic_assessment_reason=res.reason,
                            deterministic_signal_types=signal_types,
                            advisor_reason="",
                        )
                else:
                    resolved_capability = res.provisional_capability
                    provenance = RoutingProvenance(
                        is_explicit_override=False,
                        assessment_clarity=res.clarity.value,
                        provisional_capability=res.provisional_capability.value,
                        advisor_consulted=False,
                        advisor_verdict=None,
                        fallback_used=False,
                        final_selection_reason=f"No advisor configured for ambiguous assessment; using deterministic provisional capability ({res.reason})",
                        deterministic_assessment_reason=res.reason,
                        deterministic_signal_types=signal_types,
                        advisor_reason="",
                    )

            if resolved_capability == AgentCapability.GENERAL:
                prompt_classification = resolved_capability.value
            elif hasattr(self._router, "detect_intent") and (intent := self._router.detect_intent(task.prompt)) is not None:
                prompt_classification = intent.value
            else:
                prompt_classification = resolved_capability.value
        else:
            resolved_capability = self._router.route(task.prompt)
            provenance = RoutingProvenance(
                is_explicit_override=False,
                provisional_capability=resolved_capability.value,
                advisor_consulted=False,
                advisor_verdict=None,
                fallback_used=False,
                final_selection_reason="Deterministic route-only capability determination",
            )
            if resolved_capability == AgentCapability.GENERAL:
                prompt_classification = resolved_capability.value
            elif hasattr(self._router, "detect_intent") and (intent := self._router.detect_intent(task.prompt)) is not None:
                prompt_classification = intent.value
            else:
                prompt_classification = resolved_capability.value

        if self._candidate_finder is not None:
            candidates = await self._candidate_finder.execute(resolved_capability)
            if not candidates:
                rejection_task_id = str(uuid4())
                override_provenance = " (explicit capability override)" if is_explicit_override else ""
                rejection_message = (
                    f"No eligible agent is currently available for capability '{resolved_capability}'{override_provenance}"
                )

                decision = RoutingDecision(
                    task_id=rejection_task_id,
                    prompt_classification=prompt_classification,
                    required_capabilities=[resolved_capability.value],
                    candidates_evaluated=[],
                    selected_agent_id=None,
                    dispatch_target=None,
                    rationale=rejection_message,
                    provenance=provenance.to_dict(),
                )

                if self._routing_recorder is not None:
                    await self._routing_recorder.record_routing_decision(decision)

                if self._events is not None:
                    event = AgentEvent.create(
                        agent_id="orchestrator",
                        state="error",
                        detail=rejection_message,
                        task_id=rejection_task_id,
                        routing_decision=decision.to_dict(),
                    )
                    await self._events.publish(event)

                raise CapabilityUnavailableError(
                    capability=resolved_capability,
                    message=rejection_message,
                    task_id=rejection_task_id,
                )

            selected_candidate = self._selection_policy.select(candidates)
            target = selected_candidate.dispatch_target

            if not target or not isinstance(target, str) or not target.strip():
                raise ValueError(
                    f"Selected candidate '{selected_candidate.agent_id}' does not have a valid dispatch target advertised"
                )

            clean_target = target.strip()
            message = TaskRequested.create(task.prompt)

            evaluated_candidates = [
                {
                    "agent_id": c.agent_id,
                    "agent_type": c.agent_type,
                    "capabilities": list(c.capabilities),
                    "execution_state": (
                        c.execution_state.value
                        if hasattr(c.execution_state, "value")
                        else str(c.execution_state)
                    ),
                    "availability_state": (
                        c.availability.state.value
                        if hasattr(c.availability.state, "value")
                        else str(c.availability.state)
                    ),
                    "dispatch_target": c.dispatch_target,
                    "is_eligible": c.availability.is_eligible,
                }
                for c in candidates
            ]

            policy_name = getattr(
                self._selection_policy, "__class__", type(self._selection_policy)
            ).__name__
            override_provenance = " (explicit capability override)" if is_explicit_override else ""
            rationale = (
                f"Selected agent '{selected_candidate.agent_id}' targeting '{clean_target}' "
                f"from {len(candidates)} candidate(s) using {policy_name} for capability '{resolved_capability.value}'"
                f"{override_provenance}"
            )

            decision = RoutingDecision(
                task_id=message.task_id,
                prompt_classification=prompt_classification,
                required_capabilities=[resolved_capability.value],
                candidates_evaluated=evaluated_candidates,
                selected_agent_id=selected_candidate.agent_id,
                dispatch_target=clean_target,
                rationale=rationale,
                provenance=provenance.to_dict(),
            )

            if self._routing_recorder is not None:
                await self._routing_recorder.record_routing_decision(decision)

            resolved_cap_str = (
                resolved_capability.value
                if hasattr(resolved_capability, "value")
                else str(resolved_capability)
            )
            await self._task_store.register_root_task(
                message,
                initial_status="pending_dispatch",
                required_capability=resolved_cap_str,
            )

            if self._dispatcher is not None:
                try:
                    await self._dispatcher.dispatch(
                        message,
                        clean_target,
                        capability=resolved_cap_str,
                    )
                    return message
                except Exception:
                    await self._task_store.update_status(message.task_id, "dispatch_failed")
                    raise

            # Preserve backward compatibility only where required for older isolated test doubles
            # that implement TaskQueue but not TaskDispatcher
            if self._queue is not None:
                try:
                    await self._queue.enqueue(message, resolved_capability, target=clean_target)
                except Exception:
                    await self._task_store.update_status(message.task_id, "dispatch_failed")
                    raise
                await self._task_store.update_status(message.task_id, "pending")
                return message

            raise RuntimeError("No task dispatcher or queue configured for dispatch")

        # Non-dynamic fallback path (candidate_finder is None) - for backward compatibility with older isolated tests
        if self._queue is not None:
            message = TaskRequested.create(task.prompt)
            rationale = (
                f"Static configuration routed to capability '{resolved_capability.value}'"
            )
            decision = RoutingDecision(
                task_id=message.task_id,
                prompt_classification=prompt_classification,
                required_capabilities=[resolved_capability.value],
                candidates_evaluated=[],
                selected_agent_id=None,
                dispatch_target=None,
                rationale=rationale,
                provenance=provenance.to_dict(),
            )

            if self._routing_recorder is not None:
                await self._routing_recorder.record_routing_decision(decision)

            resolved_cap_str = (
                resolved_capability.value
                if hasattr(resolved_capability, "value")
                else str(resolved_capability)
            )
            await self._task_store.register_root_task(
                message,
                initial_status="pending_dispatch",
                required_capability=resolved_cap_str,
            )

            try:
                await self._queue.enqueue(message, resolved_capability)
            except Exception:
                await self._task_store.update_status(message.task_id, "dispatch_failed")
                raise
            await self._task_store.update_status(message.task_id, "pending")
            return message

        raise RuntimeError("No task queue configured for legacy enqueue")


class RegisterChildTask:
    """Internal platform-controlled primitive to create and register a child task.

    Enforces platform governance required by ADR-0006:
    - Verifies authoritative parent existence;
    - Derives root_task_id from the authoritative parent;
    - Enforces depth-1 policy (parent must be a root task);
    - Atomically commits the child existence consistency boundary (durable status,
      forward lineage, and parent-to-child membership);
    - Remains non-executable: does not route, enqueue, or dispatch.
    """

    def __init__(self, task_store: AuthoritativeTaskStore) -> None:
        if task_store is None:
            raise TypeError("RegisterChildTask requires an AuthoritativeTaskStore")
        if not isinstance(task_store, AuthoritativeTaskStore):
            raise TypeError(
                f"task_store must implement AuthoritativeTaskStore, got {type(task_store).__name__}"
            )
        self._task_store = task_store

    async def execute(
        self,
        parent_task_id: str,
        prompt: str,
    ) -> TaskRequested:
        if not isinstance(parent_task_id, str) or not parent_task_id.strip():
            raise ValueError("parent_task_id must be a non-empty string")
        clean_parent_task_id = parent_task_id.strip()

        task = Task(prompt=prompt)
        clean_prompt = task.prompt

        # 1. Authoritative parent existence check
        parent_lineage = await self._task_store.get_lineage(clean_parent_task_id)
        if parent_lineage is None:
            raise ParentTaskNotFoundError(
                f"Authoritative parent task not found: {clean_parent_task_id}"
            )

        # 2. Depth-1 policy enforcement (parent itself must be a root)
        if parent_lineage.parent_task_id is not None:
            raise DelegationDepthExceededError(
                f"Delegation depth exceeded: parent task '{clean_parent_task_id}' is already a child task"
            )

        # 3. Platform-derived root inheritance (never caller input)
        child_root_task_id = parent_lineage.root_task_id

        # 4. Mint child identity under trusted platform control
        child_task_id = str(uuid4())

        # 5. Construct valid TaskRequested v2 child
        child_task = TaskRequested(
            task_id=child_task_id,
            prompt=clean_prompt,
            created_at=datetime.now(UTC).isoformat(),
            root_task_id=child_root_task_id,
            parent_task_id=clean_parent_task_id,
            version=2,
        )

        # 6. Commit child existence consistency boundary (durable task state + forward lineage + parent membership)
        await self._task_store.register_child_task(
            child_task,
            initial_status="pending_dispatch",
            required_capability="analysis",
        )

        return child_task


class UnauthorizedPrincipalError(Exception):
    """Raised when an unauthenticated, invalid, or inactive principal attempts an operation."""


class ClaimTask:
    """Authenticated platform pull-claim use case for task turns.

    Governed by ADR-0007 (Decision 7) and Issue #220:
    - Atomically verifies task claimability, capability compatibility, and absence of active claim;
    - Transitions task to 'working';
    - Mints TaskAssignment, fencing epoch, delivery lease, and detached assignment proof;
    - Emits narrow one-turn research -> analysis grant for eligible Research root claims;
    - Idempotent for repeated requests with same (principal_id, request_id).
    """

    def __init__(
        self,
        task_store: AuthoritativeTaskStore,
        assignment_proof_pepper: str,
        delivery_lease_seconds: float = 30.0,
        clock: Any = None,
    ) -> None:
        if task_store is None:
            raise TypeError("ClaimTask requires an AuthoritativeTaskStore")
        self._task_store = task_store
        self._master_secret = parse_master_secret(assignment_proof_pepper)
        self._token_key, self._verifier_key = derive_proof_subkeys(self._master_secret)
        self._delivery_lease_seconds = float(delivery_lease_seconds)
        self._clock = clock or (lambda: datetime.now(UTC))

    async def execute(
        self,
        principal: RuntimePrincipal,
        task_id: str,
        request_id: str,
        capability: str,
        execution_kind: str = "initial",
        expected_continuation_version: int | None = None,
    ) -> ClaimTaskResponse:
        if not principal or not principal.active:
            raise UnauthorizedPrincipalError("Principal is invalid or inactive")

        clean_cap = capability.strip()
        if not principal.has_capability(clean_cap):
            raise CapabilityMismatchError(
                f"Principal '{principal.principal_id}' is not authorized for capability '{clean_cap}'"
            )

        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            raise ValueError("task_id must be a non-empty string")

        clean_request_id = str(request_id).strip()
        if not clean_request_id:
            raise ValueError("request_id must be a non-empty string")

        clean_kind = execution_kind.strip()
        if clean_kind not in ("initial", "delegated_child", "resumed_continuation"):
            raise ValueError(f"Invalid execution_kind: '{clean_kind}'")

        if clean_kind in ("initial", "delegated_child"):
            if expected_continuation_version is not None:
                raise ValueError(
                    f"expected_continuation_version must not be provided for {clean_kind} execution_kind"
                )
            continuation_version = None
            prior_claimable_status = "pending"
        elif clean_kind == "resumed_continuation":
            if expected_continuation_version is None or expected_continuation_version < 1:
                raise ValueError(
                    "expected_continuation_version must be an integer >= 1 for resumed_continuation"
                )
            continuation_version = expected_continuation_version
            prior_claimable_status = "resumable"

        # 1. Authoritative lineage lookup
        lineage = await self._task_store.get_lineage(clean_task_id)
        if lineage is None:
            raise TaskNotClaimableError(f"Task '{clean_task_id}' not found")

        # 2. Execution kind validation
        if clean_kind == "initial" and not lineage.is_root:
            raise TaskNotClaimableError(f"Task '{clean_task_id}' is not a root task for initial claim")
        if clean_kind == "delegated_child" and lineage.is_root:
            raise TaskNotClaimableError(f"Task '{clean_task_id}' is a root task, not a delegated child")
        if clean_kind == "resumed_continuation" and not lineage.is_root:
            raise TaskNotClaimableError(f"Task '{clean_task_id}' is not a root task for resumption")

        # 3. Authoritative required capability verification
        req_cap = await self._task_store.get_required_capability(clean_task_id)
        if req_cap is not None and req_cap.required_capability != clean_cap:
            raise CapabilityMismatchError(
                f"Task '{clean_task_id}' requires capability '{req_cap.required_capability}', but requested '{clean_cap}'"
            )

        # 4. Authoritative TaskInput loading
        task_input = await self._task_store.get_task_input(clean_task_id, 1)
        if task_input is None:
            raise TaskNotClaimableError(f"Task '{clean_task_id}' has no authoritative input record")

        # 5. Continuation verification (for resumed_continuation)
        if clean_kind == "resumed_continuation":
            assert continuation_version is not None
            continuation = await self._task_store.get_continuation(clean_task_id, continuation_version)
            if continuation is None:
                raise TaskNotClaimableError(
                    f"Continuation version {continuation_version} not found for task '{clean_task_id}'"
                )
            if continuation.parent_task_id != clean_task_id:
                raise TaskNotClaimableError("Continuation task mismatch")
            if continuation.continuation_capability != clean_cap:
                raise CapabilityMismatchError(
                    f"Continuation requires capability '{continuation.continuation_capability}', but requested '{clean_cap}'"
                )
            pinned_artifacts = tuple(dict.fromkeys(task_input.artifact_references + continuation.artifact_references))
        else:
            pinned_artifacts = task_input.artifact_references

        now_dt = self._clock()
        now_iso = normalize_canonical_utc_timestamp(now_dt)
        expires_at = normalize_canonical_utc_timestamp(
            now_dt + timedelta(seconds=self._delivery_lease_seconds)
        )

        # 6. Check current fencing epoch
        cur_epoch = await self._task_store.get_fencing_epoch(clean_task_id)
        candidate_epoch = cur_epoch + 1

        # 7. Check for eligible root grant
        grant: DelegationGrant | None = None
        assignment_id = f"assign-{uuid4()}"
        turn_id = f"turn-{uuid4()}"
        if clean_kind == "initial" and clean_cap == "research" and lineage.is_root:
            root_authority = await self._task_store.get_root_authority(clean_task_id)
            if root_authority and ("research", "analysis") in root_authority.allowed_delegation_edges:
                grant_id = f"grant-{uuid4()}"
                grant = DelegationGrant(
                    grant_id=grant_id,
                    root_task_id=clean_task_id,
                    issued_to_assignment_id=assignment_id,
                    issued_to_principal_id=principal.principal_id,
                    fencing_epoch=candidate_epoch,
                    turn_id=turn_id,
                    issued_at=now_iso,
                )

        claim_id = f"claim-{uuid4()}"

        # 8. Build candidate canonical proof identity & derive proof + verifier digest
        canonical_identity = encode_canonical_proof_identity(
            assignment_id=assignment_id,
            claim_id=claim_id,
            task_id=clean_task_id,
            root_task_id=lineage.root_task_id,
            turn_id=turn_id,
            runtime_principal_id=principal.principal_id,
            bound_agent_id=principal.bound_agent_id,
            assigned_capability=clean_cap,
            execution_kind=clean_kind,
            continuation_version=continuation_version,
            fencing_epoch=candidate_epoch,
            request_id=clean_request_id,
            delivery_expires_at=expires_at,
            delegation_grant_id=grant.grant_id if grant else None,
            prior_claimable_status=prior_claimable_status,
        )
        proof = derive_assignment_proof(self._token_key, canonical_identity)
        verifier_digest = compute_assignment_verifier_digest(self._verifier_key, proof)

        candidate_claim = ActiveClaim(
            task_id=clean_task_id,
            assignment_id=assignment_id,
            claim_id=claim_id,
            root_task_id=lineage.root_task_id,
            turn_id=turn_id,
            runtime_principal_id=principal.principal_id,
            bound_agent_id=principal.bound_agent_id,
            assigned_capability=clean_cap,
            execution_kind=clean_kind,
            continuation_version=continuation_version,
            fencing_epoch=candidate_epoch,
            proof_digest=verifier_digest,
            request_id=clean_request_id,
            claimed_at=now_iso,
            delivery_expires_at=expires_at,
            delegation_grant_id=grant.grant_id if grant else None,
            prior_claimable_status=prior_claimable_status,
        )

        active_claim, is_replay = await self._task_store.atomic_claim_task(
            task_id=clean_task_id,
            claim=candidate_claim,
            grant=grant,
            now=now_iso,
            expected_continuation_version=continuation_version if continuation_version is not None else -1,
            is_root=lineage.is_root,
        )

        if is_replay:
            if active_claim.state != "claimed_unacknowledged":
                raise TaskAlreadyClaimedError(f"Task '{clean_task_id}' assignment is already acknowledged")
            if active_claim.delivery_expires_at <= now_iso:
                raise ClaimDeliveryLeaseExpiredError(
                    f"Delivery lease for assignment '{active_claim.assignment_id}' has expired"
                )

            replay_identity = encode_canonical_proof_identity(
                assignment_id=active_claim.assignment_id,
                claim_id=active_claim.claim_id,
                task_id=active_claim.task_id,
                root_task_id=active_claim.root_task_id,
                turn_id=active_claim.turn_id,
                runtime_principal_id=active_claim.runtime_principal_id,
                bound_agent_id=active_claim.bound_agent_id,
                assigned_capability=active_claim.assigned_capability,
                execution_kind=active_claim.execution_kind,
                continuation_version=active_claim.continuation_version,
                fencing_epoch=active_claim.fencing_epoch,
                request_id=active_claim.request_id,
                delivery_expires_at=active_claim.delivery_expires_at,
                delegation_grant_id=active_claim.delegation_grant_id,
                prior_claimable_status=active_claim.prior_claimable_status,
            )
            proof = derive_assignment_proof(self._token_key, replay_identity)
            computed_verifier = compute_assignment_verifier_digest(self._verifier_key, proof)
            if not hmac.compare_digest(computed_verifier, active_claim.proof_digest):
                raise CorruptAuthoritativeRecordError("Replayed claim verifier digest mismatch")

        task_input_ref = TaskInputReference(
            task_id=clean_task_id,
            input_version=task_input.input_version,
        )

        assignment = TaskAssignment(
            assignment_id=active_claim.assignment_id,
            claim_id=active_claim.claim_id,
            task_id=active_claim.task_id,
            root_task_id=active_claim.root_task_id,
            turn_id=active_claim.turn_id,
            runtime_principal_id=active_claim.runtime_principal_id,
            bound_agent_id=active_claim.bound_agent_id,
            execution_kind=active_claim.execution_kind,
            assigned_capability=active_claim.assigned_capability,
            fencing_epoch=active_claim.fencing_epoch,
            claimed_at=active_claim.claimed_at,
            delivery_expires_at=active_claim.delivery_expires_at,
            task_input_reference=task_input_ref,
            continuation_version=active_claim.continuation_version,
            pinned_artifact_references=pinned_artifacts,
            delegation_grant_id=active_claim.delegation_grant_id,
        )

        return ClaimTaskResponse(
            assignment=assignment,
            assignment_proof=proof,
        )


class AcknowledgeAssignment:
    """Authenticated assignment acknowledgement use case.

    Governed by ADR-0007 (Decision 7) and Issue #220:
    - Transitions an active claim from 'claimed_unacknowledged' to 'acknowledged';
    - Verifies principal identity, assignment ID, and detached proof digest;
    - Idempotent on repeated calls for already-acknowledged assignments;
    - Rejects expired delivery leases.
    """

    def __init__(
        self,
        task_store: AuthoritativeTaskStore,
        assignment_proof_pepper: str,
        clock: Any = None,
    ) -> None:
        if task_store is None:
            raise TypeError("AcknowledgeAssignment requires an AuthoritativeTaskStore")
        self._task_store = task_store
        self._master_secret = parse_master_secret(assignment_proof_pepper)
        self._token_key, self._verifier_key = derive_proof_subkeys(self._master_secret)
        self._clock = clock or (lambda: datetime.now(UTC))

    async def execute(
        self,
        principal: RuntimePrincipal,
        task_id: str,
        assignment_id: str,
        assignment_proof: str,
    ) -> ActiveClaim:
        if not principal or not principal.active:
            raise UnauthorizedPrincipalError("Principal is invalid or inactive")

        clean_task_id = str(task_id).strip()
        clean_assign_id = str(assignment_id).strip()
        if not clean_task_id or not clean_assign_id:
            raise ValueError("task_id and assignment_id must be non-empty strings")

        claim = await self._task_store.get_active_claim(clean_task_id)
        if claim is None or claim.assignment_id != clean_assign_id:
            raise ClaimNotFoundError(
                f"Active claim for assignment '{clean_assign_id}' on task '{clean_task_id}' not found"
            )

        if claim.runtime_principal_id != principal.principal_id:
            raise ClaimNotFoundError(
                f"Assignment '{clean_assign_id}' is not owned by principal '{principal.principal_id}'"
            )

        if claim.bound_agent_id != principal.bound_agent_id:
            raise ClaimNotFoundError(
                f"Assignment '{clean_assign_id}' is bound to agent '{claim.bound_agent_id}', not '{principal.bound_agent_id}'"
            )

        if not principal.has_capability(claim.assigned_capability):
            raise CapabilityMismatchError(
                f"Principal '{principal.principal_id}' lacks capability '{claim.assigned_capability}'"
            )

        cur_epoch = await self._task_store.get_fencing_epoch(clean_task_id)
        if claim.fencing_epoch != cur_epoch:
            raise TaskAlreadyClaimedError(
                f"Fencing epoch mismatch for assignment '{clean_assign_id}': claim={claim.fencing_epoch}, current={cur_epoch}"
            )

        if not verify_assignment_proof(self._verifier_key, assignment_proof, claim.proof_digest):
            raise InvalidClaimProofError("Assignment proof verification failed")

        now_iso = normalize_canonical_utc_timestamp(self._clock())
        return await self._task_store.acknowledge_claim(
            task_id=clean_task_id,
            assignment_id=clean_assign_id,
            principal_id=principal.principal_id,
            acknowledged_at=now_iso,
        )


class StageArtifactUseCase:
    """Authenticated artifact staging and sealing use case.

    Governed by ADR-0007 (Decisions 8, 10) and Issue #220:
    - Verifies active assignment, principal, and detached proof;
    - Requires assignment to be in 'acknowledged' claim state;
    - Platform derives and stamps provenance from the active assignment;
    - Caller-asserted provenance or identity is strictly ignored;
    - Only the currently assigned task may receive staged artifacts;
    - Seals and persists immutable StageArtifact via AuthoritativeTaskStore.
    """

    def __init__(
        self,
        task_store: AuthoritativeTaskStore,
        assignment_proof_pepper: str,
        clock: Any = None,
    ) -> None:
        if task_store is None:
            raise TypeError("StageArtifactUseCase requires an AuthoritativeTaskStore")
        self._task_store = task_store
        self._master_secret = parse_master_secret(assignment_proof_pepper)
        self._token_key, self._verifier_key = derive_proof_subkeys(self._master_secret)
        self._clock = clock or (lambda: datetime.now(UTC))

    async def execute(
        self,
        principal: RuntimePrincipal,
        task_id: str,
        assignment_id: str,
        assignment_proof: str,
        artifact_id: str,
        artifact_type: str,
        content: Mapping[str, Any],
        artifact_version: int = 1,
        payload_schema_version: int = 1,
    ) -> StageArtifact:
        if not principal or not principal.active:
            raise UnauthorizedPrincipalError("Principal is invalid or inactive")

        clean_task_id = str(task_id).strip()
        clean_assign_id = str(assignment_id).strip()
        if not clean_task_id or not clean_assign_id:
            raise ValueError("task_id and assignment_id must be non-empty strings")

        claim = await self._task_store.get_active_claim(clean_task_id)
        if claim is None or claim.assignment_id != clean_assign_id:
            raise ClaimNotFoundError(
                f"Active claim for assignment '{clean_assign_id}' on task '{clean_task_id}' not found"
            )

        if claim.runtime_principal_id != principal.principal_id:
            raise ClaimNotFoundError(
                f"Assignment '{clean_assign_id}' is not owned by principal '{principal.principal_id}'"
            )

        if claim.bound_agent_id != principal.bound_agent_id:
            raise ClaimNotFoundError(
                f"Assignment '{clean_assign_id}' is bound to agent '{claim.bound_agent_id}', not '{principal.bound_agent_id}'"
            )

        if not principal.has_capability(claim.assigned_capability):
            raise CapabilityMismatchError(
                f"Principal '{principal.principal_id}' lacks capability '{claim.assigned_capability}'"
            )

        cur_epoch = await self._task_store.get_fencing_epoch(clean_task_id)
        if claim.fencing_epoch != cur_epoch:
            raise TaskAlreadyClaimedError(
                f"Fencing epoch mismatch for assignment '{clean_assign_id}': claim={claim.fencing_epoch}, current={cur_epoch}"
            )

        if claim.state != "acknowledged":
            raise TaskNotClaimableError("Assignment must be acknowledged before staging artifacts")

        if not verify_assignment_proof(self._verifier_key, assignment_proof, claim.proof_digest):
            raise InvalidClaimProofError("Assignment proof verification failed")

        now_iso = normalize_canonical_utc_timestamp(self._clock())

        # Platform-stamped provenance
        provenance = {
            "assignment_id": claim.assignment_id,
            "bound_agent_id": claim.bound_agent_id,
            "fencing_epoch": claim.fencing_epoch,
            "producing_task_id": claim.task_id,
            "runtime_principal_id": claim.runtime_principal_id,
            "stamped_at": now_iso,
            "turn_id": claim.turn_id,
        }

        artifact = StageArtifact(
            artifact_id=artifact_id,
            producing_task_id=claim.task_id,
            artifact_type=artifact_type,
            content=content,
            created_at=now_iso,
            producing_turn_id=claim.turn_id,
            artifact_version=artifact_version,
            payload_schema_version=payload_schema_version,
            sealed=True,
            metadata=provenance,
        )

        await self._task_store.save_stage_artifact(
            artifact,
            expected_assignment_id=claim.assignment_id,
            expected_principal_id=principal.principal_id,
            expected_fencing_epoch=claim.fencing_epoch,
        )
        return artifact


class AcceptDelegationProposal:
    """Authenticated Research -> Analysis delegation acceptance use case.

    Governed by ADR-0006 and ADR-0007:
    - Verifies active Research turn, assignment, and detached proof;
    - Strictly scoped to Research -> Analysis delegation;
    - Computes proposal fingerprint for idempotent replay comparison;
    - Invokes AuthoritativeTaskStore.accept_delegation_proposal atomically;
    - Commits child task in 'pending' status with claim deadline;
    - Emits initial non-secret Analysis wake-up hint;
    - Retires active parent claim with durable yield evidence;
    - Supports idempotent replay without duplicate hints or mutations.
    """

    def __init__(
        self,
        task_store: AuthoritativeTaskStore,
        assignment_proof_pepper: str,
        child_claim_deadline_seconds: float = 120.0,
        clock: Any = None,
    ) -> None:
        if task_store is None:
            raise TypeError("AcceptDelegationProposal requires an AuthoritativeTaskStore")
        self._task_store = task_store
        self._master_secret = parse_master_secret(assignment_proof_pepper)
        self._token_key, self._verifier_key = derive_proof_subkeys(self._master_secret)
        self._child_claim_deadline_seconds = float(child_claim_deadline_seconds)
        self._clock = clock or (lambda: datetime.now(UTC))

    async def execute(
        self,
        principal: RuntimePrincipal,
        parent_task_id: str,
        proposal_id: str,
        assignment_id: str,
        assignment_proof: str,
        child_intent: str,
        artifact_reference: ArtifactReference,
        target_capability: str = "analysis",
        parent_input_version: int = 1,
        control_fields: Mapping[str, Any] | None = None,
    ) -> DelegationProposalResponse:
        if not principal or not principal.active:
            raise UnauthorizedPrincipalError("Principal is invalid or inactive")

        if not principal.has_capability("research"):
            raise CapabilityMismatchError(
                f"Principal '{principal.principal_id}' lacks capability 'research' required to propose delegation"
            )

        clean_parent = str(parent_task_id).strip()
        clean_proposal_id = str(proposal_id).strip()
        clean_assign_id = str(assignment_id).strip()
        clean_target_cap = str(target_capability).strip()
        clean_proof = str(assignment_proof).strip()

        if not clean_parent or not clean_proposal_id or not clean_assign_id:
            raise ValueError("parent_task_id, proposal_id, and assignment_id must be non-empty strings")
        if not clean_proof:
            raise ValueError("assignment_proof must be a non-empty string")

        if clean_target_cap != "analysis":
            raise ValueError(
                f"Unsupported target_capability '{clean_target_cap}'. Strictly 'analysis' is supported."
            )

        # Validate caller proposal fields via contract
        proposal_req = DelegationProposalRequest(
            proposal_id=clean_proposal_id,
            child_intent=child_intent,
            artifact_reference=artifact_reference,
            target_capability=clean_target_cap,
            parent_input_version=parent_input_version,
            control_fields=control_fields or {},
        )

        fp = compute_proposal_fingerprint(
            parent_task_id=clean_parent,
            proposal_id=clean_proposal_id,
            assignment_id=clean_assign_id,
            target_capability=clean_target_cap,
            child_intent=proposal_req.child_intent,
            artifact_reference=proposal_req.artifact_reference,
            parent_input_version=proposal_req.parent_input_version,
            control_fields=proposal_req.control_fields,
        )

        claim = await self._task_store.get_active_claim(clean_parent)
        now_dt = self._clock()
        now_iso = normalize_canonical_utc_timestamp(now_dt)

        if claim is not None:
            # Active claim present: first acceptance (or active replay)
            if claim.assignment_id != clean_assign_id:
                raise ClaimNotFoundError(
                    f"Active claim for assignment '{clean_assign_id}' on task '{clean_parent}' not found"
                )
            if claim.runtime_principal_id != principal.principal_id:
                raise ClaimNotFoundError(
                    f"Assignment '{clean_assign_id}' is not owned by principal '{principal.principal_id}'"
                )
            if claim.bound_agent_id != principal.bound_agent_id:
                raise ClaimNotFoundError(
                    f"Assignment '{clean_assign_id}' is bound to agent '{claim.bound_agent_id}', not '{principal.bound_agent_id}'"
                )
            if claim.assigned_capability != "research":
                raise CapabilityMismatchError(
                    f"Principal '{principal.principal_id}' lacks capability '{claim.assigned_capability}'"
                )
            if claim.state != "acknowledged":
                raise TaskNotClaimableError("Assignment must be acknowledged before proposing delegation")

            cur_epoch = await self._task_store.get_fencing_epoch(clean_parent)
            if claim.fencing_epoch != cur_epoch:
                raise TaskAlreadyClaimedError(
                    f"Fencing epoch mismatch for assignment '{clean_assign_id}': claim={claim.fencing_epoch}, current={cur_epoch}"
                )

            if not verify_assignment_proof(self._verifier_key, clean_proof, claim.proof_digest):
                raise InvalidClaimProofError("Assignment proof verification failed")

            if not claim.delegation_grant_id:
                raise TaskNotClaimableError("Active assignment has no delegation grant")

            child_task_id = f"task-analysis-{uuid4()}"
            delegation_decision_id = f"dec-{uuid4()}"
            deadline_dt = now_dt + timedelta(seconds=self._child_claim_deadline_seconds)
            deadline_iso = normalize_canonical_utc_timestamp(deadline_dt)

            candidate_proposal = DelegationProposalRecord(
                parent_task_id=clean_parent,
                proposal_id=clean_proposal_id,
                child_task_id=child_task_id,
                delegation_decision_id=delegation_decision_id,
                assignment_id=clean_assign_id,
                runtime_principal_id=principal.principal_id,
                bound_agent_id=principal.bound_agent_id,
                proof_digest=claim.proof_digest,
                proposal_fingerprint=fp,
                delegation_grant_id=claim.delegation_grant_id,
                child_claim_deadline=deadline_iso,
                accepted_at=now_iso,
                target_capability="analysis",
                continuation_version=1,
            )
            child_input = TaskInput(
                task_id=child_task_id,
                prompt=proposal_req.child_intent,
                created_at=now_iso,
                input_version=1,
                artifact_references=(proposal_req.artifact_reference,),
            )
            continuation = TaskContinuation(
                parent_task_id=clean_parent,
                child_task_id=child_task_id,
                created_at=now_iso,
                updated_at=now_iso,
                input_version=proposal_req.parent_input_version,
                continuation_capability="research",
                stage="incorporate_analysis_child",
                control_fields=proposal_req.control_fields,
                artifact_references=(proposal_req.artifact_reference,),
                continuation_version=1,
            )
            yield_rec = AssignmentYieldRecord(
                assignment_id=clean_assign_id,
                parent_task_id=clean_parent,
                principal_id=principal.principal_id,
                proposal_id=clean_proposal_id,
                child_task_id=child_task_id,
                delegation_decision_id=delegation_decision_id,
                continuation_version=1,
                yielded_at=now_iso,
            )
            wake_up_hint = WakeUpHint(
                hint_id=f"hint-{uuid4()}",
                capability="analysis",
                created_at=now_iso,
                execution_kind="delegated_child",
                task_id=child_task_id,
            )

            accepted_record, _is_replay = await self._task_store.accept_delegation_proposal(
                parent_task_id=clean_parent,
                proposal=candidate_proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=wake_up_hint,
                expected_fencing_epoch=claim.fencing_epoch,
                now=now_iso,
            )
        else:
            # Active claim retired: read-only replay check
            record = await self._task_store.get_delegation_proposal(clean_parent, clean_proposal_id)
            if record is None:
                raise DelegationReplayAuthorizationError("Delegation replay authorization failed")

            # Strict replay invariant verification against authoritative accepted record
            if (
                record.parent_task_id != clean_parent
                or record.proposal_id != clean_proposal_id
                or record.assignment_id != clean_assign_id
                or record.runtime_principal_id != principal.principal_id
                or record.bound_agent_id != principal.bound_agent_id
                or record.proposal_fingerprint != fp
                or not verify_assignment_proof(self._verifier_key, clean_proof, record.proof_digest)
            ):
                raise DelegationReplayAuthorizationError("Delegation replay authorization failed")

            accepted_record = record

        return DelegationProposalResponse(
            parent_task_id=accepted_record.parent_task_id,
            proposal_id=accepted_record.proposal_id,
            child_task_id=accepted_record.child_task_id,
            delegation_decision_id=accepted_record.delegation_decision_id,
            child_claim_deadline=accepted_record.child_claim_deadline,
            accepted_at=accepted_record.accepted_at,
            target_capability=accepted_record.target_capability,
            continuation_version=accepted_record.continuation_version,
            status="accepted",
        )


class CompleteTurn:
    """Authenticated universal fenced turn-completion use case.

    Governed by ADR-0006, ADR-0007 (Decision 11), and Issue #222:
    - Always requires a Bearer-authenticated active RuntimePrincipal and a
      detached X-Assignment-Proof binding the live claim's proof digest;
    - First completion requires the current active acknowledged claim, the
      current nonterminal task state, and no conflicting canonical records;
    - Replay after claim retirement authorizes read-only against the persisted
      completion record, outcome, and stored proof digest, and normalizes every
      failure to one generic authorization error that reveals nothing;
    - Commits through the single-node TurnCompletionStore Lua boundary (all
      expected validation rejections occur read-only with zero mutation; no
      rollback is claimed for unexpected command errors, host failures,
      out-of-memory conditions, or persistence failures); never writes
      canonical result or outcome keys directly and never persists or logs the
      plaintext proof.
    """

    def __init__(
        self,
        task_store: AuthoritativeTaskStore,
        completion_store: TurnCompletionStore,
        assignment_proof_pepper: str,
        clock: Any = None,
    ) -> None:
        if task_store is None:
            raise TypeError("CompleteTurn requires an AuthoritativeTaskStore")
        if completion_store is None:
            raise TypeError("CompleteTurn requires a TurnCompletionStore")
        if not isinstance(completion_store, TurnCompletionStore):
            raise TypeError(
                "completion_store must implement TurnCompletionStore, "
                f"got {type(completion_store).__name__}"
            )
        self._task_store = task_store
        self._completion_store = completion_store
        self._master_secret = parse_master_secret(assignment_proof_pepper)
        self._token_key, self._verifier_key = derive_proof_subkeys(self._master_secret)
        self._clock = clock or (lambda: datetime.now(UTC))

    async def execute(
        self,
        principal: RuntimePrincipal,
        task_id: str,
        command_payload: dict[str, Any],
        assignment_proof: str,
    ) -> CompletionResponse:
        if not principal or not principal.active:
            raise UnauthorizedPrincipalError("Principal is invalid or inactive")

        clean_task = str(task_id).strip()
        if not clean_task:
            raise ValueError("task_id must be a non-empty string")

        clean_proof = str(assignment_proof).strip()
        if not clean_proof:
            raise ValueError("assignment_proof must be a non-empty string")

        command = CompletionCommand.from_dict(command_payload)

        if command.completed_result is not None:
            if command.completed_result.task_id != clean_task:
                raise ValueError(
                    f"Completed TaskResult task_id '{command.completed_result.task_id}' "
                    f"does not match route task '{clean_task}'"
                )
            if command.completed_result.status != "completed":
                raise ValueError(
                    "Completed TaskResult status must be 'completed' "
                    f"for route task '{clean_task}'"
                )

        claim = await self._task_store.get_active_claim(clean_task)

        if claim is not None and claim.assignment_id == command.assignment_id:
            return await self._execute_first_completion(
                principal=principal,
                task_id=clean_task,
                command=command,
                proof=clean_proof,
                claim=claim,
            )
        return await self._execute_retired_replay(
            principal=principal,
            task_id=clean_task,
            command=command,
            proof=clean_proof,
        )

    async def _execute_first_completion(
        self,
        principal: RuntimePrincipal,
        task_id: str,
        command: CompletionCommand,
        proof: str,
        claim: ActiveClaim,
    ) -> CompletionResponse:
        if claim.runtime_principal_id != principal.principal_id:
            raise ClaimNotFoundError(
                f"Assignment '{command.assignment_id}' is not owned by principal '{principal.principal_id}'"
            )
        if claim.bound_agent_id != principal.bound_agent_id:
            raise ClaimNotFoundError(
                f"Assignment '{command.assignment_id}' is bound to agent '{claim.bound_agent_id}', "
                f"not '{principal.bound_agent_id}'"
            )
        if not principal.has_capability(claim.assigned_capability):
            raise CapabilityMismatchError(
                f"Principal '{principal.principal_id}' lacks capability '{claim.assigned_capability}'"
            )
        if command.execution_kind != claim.execution_kind:
            raise CapabilityMismatchError(
                f"Completion execution_kind '{command.execution_kind}' does not match "
                f"claimed execution_kind '{claim.execution_kind}'"
            )
        if command.continuation_version != claim.continuation_version:
            raise TaskNotClaimableError(
                f"Completion continuation version '{command.continuation_version}' does not match "
                f"claimed continuation version '{claim.continuation_version}'"
            )
        if claim.state != "acknowledged":
            raise TaskNotClaimableError("Assignment must be acknowledged before completing the turn")

        cur_epoch = await self._task_store.get_fencing_epoch(task_id)
        if claim.fencing_epoch != cur_epoch:
            raise TaskAlreadyClaimedError(
                f"Fencing epoch mismatch for assignment '{command.assignment_id}': "
                f"claim={claim.fencing_epoch}, current={cur_epoch}"
            )

        if not verify_assignment_proof(self._verifier_key, proof, claim.proof_digest):
            raise InvalidClaimProofError("Assignment proof verification failed")

        lineage = await self._task_store.get_lineage(task_id)
        if lineage is None:
            raise CorruptAuthoritativeRecordError(
                f"Active claim exists for task '{task_id}' with no authoritative lineage"
            )

        now_iso = normalize_canonical_utc_timestamp(self._clock())
        outcome, result, replayed = await self._completion_store.complete_turn(
            task_id=task_id,
            command=command,
            proof_digest=claim.proof_digest,
            fencing_epoch=claim.fencing_epoch,
            parent_task_id=lineage.parent_task_id,
            root_task_id=lineage.root_task_id,
            completed_at=now_iso,
            runtime_principal_id=claim.runtime_principal_id,
            bound_agent_id=claim.bound_agent_id,
        )
        return CompletionResponse(outcome=outcome, replayed=replayed, result=result)

    async def _execute_retired_replay(
        self,
        principal: RuntimePrincipal,
        task_id: str,
        command: CompletionCommand,
        proof: str,
    ) -> CompletionResponse:
        record = await self._completion_store.get_completion_record(
            task_id, command.completion_command_id
        )
        if record is None:
            # Uniform failure: a missing record is indistinguishable from a
            # failed authorization so retired replays reveal no existence,
            # task-state, or command information.
            raise CompletionReplayAuthorizationError("Completion replay authorization failed")

        authorized = (
            record.task_id == task_id
            and record.assignment_id == command.assignment_id
            and record.runtime_principal_id == principal.principal_id
            and record.bound_agent_id == principal.bound_agent_id
            and record.execution_kind == command.execution_kind
            and record.continuation_version == command.continuation_version
            and record.semantic_fingerprint == command.semantic_fingerprint()
            and verify_assignment_proof(self._verifier_key, proof, record.proof_digest)
        )
        if not authorized:
            raise CompletionReplayAuthorizationError("Completion replay authorization failed")

        outcome = await self._completion_store.get_terminal_outcome(task_id)
        if outcome is None or outcome.outcome_id != record.outcome_id:
            raise CorruptAuthoritativeRecordError(
                f"Canonical terminal outcome missing or inconsistent for replayed task '{task_id}'"
            )
        if outcome.task_id != task_id:
            raise CorruptAuthoritativeRecordError(
                f"Canonical terminal outcome task mismatch for replayed task '{task_id}'"
            )

        result: TaskResult | None = None
        if outcome.terminal_status == "completed":
            result = await self._completion_store.get_canonical_result(task_id)
            if result is None:
                raise CorruptAuthoritativeRecordError(
                    f"Canonical result missing for completed replayed task '{task_id}'"
                )
            if result.task_id != task_id or result.status != "completed":
                raise CorruptAuthoritativeRecordError(
                    f"Canonical result inconsistent for completed replayed task '{task_id}'"
                )
        else:
            unexpected = await self._completion_store.get_canonical_result(task_id)
            if unexpected is not None:
                raise CorruptAuthoritativeRecordError(
                    f"Unexpected canonical result for failed replayed task '{task_id}'"
                )

        return CompletionResponse(outcome=outcome, replayed=True, result=result)
