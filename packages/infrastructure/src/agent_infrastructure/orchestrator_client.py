"""Authenticated Orchestrator API client for assignment-backed runtimes.

Provider-neutral boundary (Issue #222) used by the General Worker, Research
Agent, and Analysis Agent to claim acknowledged turns and submit fenced turn
completions. The detached assignment proof travels only in the
X-Assignment-Proof header; it is never placed in JSON bodies, logs, or events.
"""

from __future__ import annotations

from typing import Any

import httpx
from agent_contracts import (
    ChildTaskExecutionMaterial,
    CompletionCommand,
    ResumedContinuationContext,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskResult,
    TerminalOutcome,
)


class CompletionSubmissionError(RuntimeError):
    """Raised when the Orchestrator completion API rejects or drops a turn completion."""

    def __init__(self, task_id: str, status_code: int | None = None) -> None:
        self.task_id = task_id
        self.status_code = status_code
        # Retain only bounded metadata (task_id, status_code). Never store or
        # include response bodies, arbitrary server detail, credentials, proofs,
        # or outcome data.
        message = f"Completion submission for task '{task_id}' failed"
        if status_code is not None:
            message += f" (HTTP {status_code})"
        super().__init__(message)


class OrchestratorClient:
    """Minimal httpx client for the authenticated claim/acknowledge/complete flow."""

    def __init__(
        self,
        base_url: str,
        runtime_token: str,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        clean_base = str(base_url).strip().rstrip("/")
        if not clean_base:
            raise ValueError("base_url must be a non-empty string")
        clean_token = str(runtime_token).strip()
        if not clean_token:
            raise ValueError("runtime_token must be a non-empty string")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._base_url = clean_base
        self._token = clean_token
        self._timeout = float(timeout_seconds)
        self._transport = transport
        self._cached_contexts: dict[str, ResumedContinuationContext] = {}
        self._cached_inputs: dict[str, TaskInput] = {}
        self._cached_artifacts: dict[tuple[str, str, int], StageArtifact] = {}

    def _headers(self, proof: str | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._token}"}
        if proof is not None:
            headers["X-Assignment-Proof"] = proof
        return headers

    async def claim_task(
        self,
        task_id: str,
        request_id: str,
        capability: str,
        execution_kind: str = "initial",
        expected_continuation_version: int | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Claim a task turn; returns (assignment dict, detached proof)."""
        payload: dict[str, Any] = {
            "task_id": task_id,
            "request_id": request_id,
            "capability": capability,
            "execution_kind": execution_kind,
        }
        if expected_continuation_version is not None:
            payload["expected_continuation_version"] = expected_continuation_version
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.post(
                    f"{self._base_url}/api/tasks/claim",
                    json=payload,
                    headers=self._headers(),
                )
        except httpx.HTTPError as exc:
            raise CompletionSubmissionError(task_id, None) from exc
        if response.status_code != 200:
            raise CompletionSubmissionError(task_id, response.status_code)
        body = response.json()
        return body["assignment"], body["assignment_proof"]

    async def acknowledge_assignment(
        self,
        task_id: str,
        assignment_id: str,
        assignment_proof: str,
    ) -> dict[str, Any]:
        """Acknowledge assignment delivery; returns the acknowledgement body."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.post(
                    f"{self._base_url}/api/assignments/{assignment_id}/acknowledge",
                    json={"task_id": task_id, "assignment_proof": assignment_proof},
                    headers=self._headers(),
                )
        except httpx.HTTPError as exc:
            raise CompletionSubmissionError(task_id, None) from exc
        if response.status_code != 200:
            raise CompletionSubmissionError(task_id, response.status_code)
        return response.json()

    async def submit_completion(
        self,
        task_id: str,
        command: CompletionCommand,
        assignment_proof: str,
    ) -> dict[str, Any]:
        """Submit a fenced turn completion; returns the parsed response body.

        The 200 response carries the canonical outcome, the replayed flag, and
        (for completed turns) the canonical result. Any non-2xx status raises
        CompletionSubmissionError carrying only the status code.
        """
        if not isinstance(command, CompletionCommand):
            raise TypeError(f"command must be a CompletionCommand, got {type(command).__name__}")
        clean_proof = str(assignment_proof).strip()
        if not clean_proof:
            raise ValueError("assignment_proof must be a non-empty string")
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.post(
                    f"{self._base_url}/api/tasks/{task_id}/completion",
                    json=command.to_dict(),
                    headers=self._headers(clean_proof),
                )
        except httpx.HTTPError as exc:
            raise CompletionSubmissionError(task_id, None) from exc
        if response.status_code != 200:
            raise CompletionSubmissionError(task_id, response.status_code)
        return response.json()

    async def stage_artifact(
        self,
        task_id: str,
        assignment_id: str,
        assignment_proof: str,
        artifact_id: str,
        artifact_type: str,
        content: dict[str, Any],
        artifact_version: int = 1,
        payload_schema_version: int = 1,
    ) -> dict[str, Any]:
        """Stage a sealed parent artifact via POST /api/tasks/{task_id}/artifacts."""
        clean_task = str(task_id).strip()
        clean_proof = str(assignment_proof).strip()
        payload = {
            "assignment_id": str(assignment_id).strip(),
            "artifact_id": str(artifact_id).strip(),
            "artifact_type": str(artifact_type).strip(),
            "content": content,
            "artifact_version": artifact_version,
            "payload_schema_version": payload_schema_version,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.post(
                    f"{self._base_url}/api/tasks/{clean_task}/artifacts",
                    json=payload,
                    headers=self._headers(clean_proof),
                )
        except httpx.HTTPError as exc:
            raise CompletionSubmissionError(clean_task, None) from exc
        if response.status_code != 200:
            raise CompletionSubmissionError(clean_task, response.status_code)
        return response.json()

    async def propose_delegation(
        self,
        task_id: str,
        assignment_id: str,
        assignment_proof: str,
        proposal_id: str,
        child_intent: str,
        artifact_reference: dict[str, Any],
        target_capability: str = "analysis",
        parent_input_version: int = 1,
        control_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Propose delegation to Analysis via POST /api/tasks/{task_id}/delegations."""
        clean_task = str(task_id).strip()
        clean_proof = str(assignment_proof).strip()
        payload = {
            "proposal_id": str(proposal_id).strip(),
            "assignment_id": str(assignment_id).strip(),
            "child_intent": str(child_intent).strip(),
            "artifact_reference": artifact_reference,
            "target_capability": target_capability,
            "parent_input_version": parent_input_version,
            "control_fields": control_fields or {},
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.post(
                    f"{self._base_url}/api/tasks/{clean_task}/delegations",
                    json=payload,
                    headers=self._headers(clean_proof),
                )
        except httpx.HTTPError as exc:
            raise CompletionSubmissionError(clean_task, None) from exc
        if response.status_code != 200:
            raise CompletionSubmissionError(clean_task, response.status_code)
        return response.json()

    async def get_child_task_input(
        self,
        task_id: str,
        assignment_id: str,
        assignment_proof: str,
    ) -> ChildTaskExecutionMaterial:
        """Fetch authoritative child/assignment execution material via GET /api/tasks/{task_id}/input."""
        clean_task = str(task_id).strip()
        clean_assignment = str(assignment_id).strip()
        clean_proof = str(assignment_proof).strip()
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.get(
                    f"{self._base_url}/api/tasks/{clean_task}/input",
                    params={"assignment_id": clean_assignment},
                    headers=self._headers(clean_proof),
                )
        except httpx.HTTPError as exc:
            raise CompletionSubmissionError(clean_task, None) from exc
        if response.status_code != 200:
            raise CompletionSubmissionError(clean_task, response.status_code)
        body = response.json()
        material = ChildTaskExecutionMaterial.from_dict(body)
        self._cached_inputs[clean_task] = material.task_input
        for art in material.artifacts:
            self._cached_artifacts[(art.producing_task_id, art.artifact_id, art.artifact_version)] = art
            self._cached_artifacts[(clean_task, art.artifact_id, art.artifact_version)] = art
        return material

    get_assignment_input = get_child_task_input

    async def get_resumed_continuation_context(
        self,
        task_id: str,
        assignment_id: str,
        assignment_proof: str,
    ) -> ResumedContinuationContext:
        """Fetch and cache governed continuation context for a resumed parent turn."""
        clean_task = str(task_id).strip()
        clean_assignment = str(assignment_id).strip()
        clean_proof = str(assignment_proof).strip()
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                response = await client.get(
                    f"{self._base_url}/api/tasks/{clean_task}/continuation-context",
                    params={"assignment_id": clean_assignment},
                    headers=self._headers(clean_proof),
                )
        except httpx.HTTPError as exc:
            raise CompletionSubmissionError(clean_task, None) from exc
        if response.status_code != 200:
            raise CompletionSubmissionError(clean_task, response.status_code)
        ctx = ResumedContinuationContext.from_dict(response.json())
        self._cached_contexts[clean_task] = ctx
        return ctx

    async def get_task_input(self, task_id: str, version: int = 1) -> TaskInput | None:
        clean_task = str(task_id).strip()
        if clean_task in self._cached_inputs:
            inp = self._cached_inputs[clean_task]
            if inp.input_version == version:
                return inp
        ctx = self._cached_contexts.get(clean_task)
        if ctx is not None and ctx.parent_task_input.input_version == version:
            return ctx.parent_task_input
        return None

    async def get_continuation(self, task_id: str, version: int = 1) -> TaskContinuation | None:
        clean_task = str(task_id).strip()
        ctx = self._cached_contexts.get(clean_task)
        if ctx is not None and ctx.continuation.continuation_version == version:
            return ctx.continuation
        return None

    async def get_stage_artifact(
        self, producing_task_id: str, artifact_id: str, artifact_version: int = 1
    ) -> StageArtifact | None:
        clean_prod = str(producing_task_id).strip()
        clean_art = str(artifact_id).strip()
        cached = self._cached_artifacts.get((clean_prod, clean_art, artifact_version))
        if cached is not None:
            return cached
        for ctx in self._cached_contexts.values():
            art = ctx.stage_artifact
            if (
                art.producing_task_id == clean_prod
                and art.artifact_id == clean_art
                and art.artifact_version == artifact_version
            ):
                return art
        return None

    async def get_terminal_outcome(self, task_id: str) -> TerminalOutcome | None:
        clean_task = str(task_id).strip()
        for ctx in self._cached_contexts.values():
            if ctx.child_outcome.task_id == clean_task:
                return ctx.child_outcome
        return None

    async def get_canonical_result(self, task_id: str) -> TaskResult | None:
        clean_task = str(task_id).strip()
        for ctx in self._cached_contexts.values():
            if ctx.child_result is not None and ctx.child_result.task_id == clean_task:
                return ctx.child_result
        return None
