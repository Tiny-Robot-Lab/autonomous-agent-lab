"""Layer 2 deterministic end-to-end collaboration demo test for Research -> 3-child Analysis wave.

Validates Increment 1 of the Collaboration Demo before Issue #35:
1. Root Research parent task created with prompt:
   "Propose a practical deployment plan for a small self-hosted learning service."
2. Research agent (instance 1) executes initial turn with RESEARCH_DELEGATION_MODE="delegate_analysis_wave":
   - stages sealed StageArtifact ("research_material")
   - proposes deterministic 3-child wave with barrier policy WAIT_ALL:
     * Security analysis
     * Reliability analysis
     * Operations and capacity analysis
   - parent yields turn, freeing executor slot (status becomes waiting_for_children)
3. 3 child wake-up hints published to agent.hints.analysis.
4. Analysis agent executes production loop across the 3 delegated child hints:
   - claims, acknowledges, fetches execution material, executes analyzer, submits completion.
   - all 3 children reach status completed.
5. On the 3rd child completion, barrier policy WAIT_ALL resolves:
   - authoritative wave resolution becomes SATISFIED.
   - parent status transitions to resumable.
   - wake-up hint emitted for research with execution_kind="resumed_continuation"
     and expected_continuation_version=2.
6. Research agent (instance 2) executes resumed v2 continuation turn:
   - claims resumed continuation with expected_continuation_version=2 under epoch 2.
   - acknowledges assignment.
   - fetches governed ResumedContinuationContext v2 (GET /api/tasks/{parent_id}/continuation-context).
   - verifies released continuation and authoritative wave resolution SATISFIED.
   - does NOT fetch child results, does NOT inspect child output, does NOT synthesize or aggregate.
   - completes parent turn with deterministic lifecycle validation acknowledging that structured
     child-result aggregation remains Issue #35.
7. Parent task reaches completed status with canonical result verified.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from agent_contracts import (
    RootAuthorityRecord,
    normalize_canonical_utc_timestamp,
)
from agent_infrastructure import OrchestratorClient
from analysis_agent.main import run_analysis_agent
from orchestrator.config import Settings
from orchestrator.domain.principal import compute_token_digest
from orchestrator.presentation.api import create_app
from research_agent.main import run_research_agent

from tests.test_authoritative_task_store import (
    InMemoryAuthoritativeTaskStore,
    MockAtomicDispatcher,
)

PEPPER = "test_e2e_wave_pepper_12345678901234567890"
PROOF_PEPPER = "test_e2e_wave_proof_pepper_098765432109876543210987654321"
RESEARCH_1_TOKEN = "secret-token-e2e-research-1"
RESEARCH_2_TOKEN = "secret-token-e2e-research-2"
ANALYSIS_1_TOKEN = "secret-token-e2e-analysis-1"

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
}


class MockEventBus:
    def __init__(self, signals: list[Any] | None = None) -> None:
        self.published: list[Any] = []
        self.current_state: str = "idle"
        self._signals = list(signals or [])

    async def publish(self, event: Any) -> None:
        self.published.append(event)
        if hasattr(event, "state"):
            self.current_state = event.state

    async def register(self, metadata: Any) -> None:
        pass

    async def heartbeat(self, heartbeat: Any) -> None:
        pass

    async def next_signal(self) -> Any:
        if self._signals:
            return self._signals.pop(0)
        await asyncio.sleep(0.01)
        raise TimeoutError("No more signals in mock event bus")

    async def aclose(self) -> None:
        pass


def _build_test_app(store: InMemoryAuthoritativeTaskStore) -> tuple[Any, Settings]:
    settings = Settings(
        redis_url="redis://fake:6379/15",
        app_env="test",
        task_delivery_mode="authenticated_claim",
        claim_delivery_lease_seconds=30.0,
        child_claim_deadline_seconds=120.0,
        runtime_credential_pepper=PEPPER,
        assignment_proof_pepper=PROOF_PEPPER,
        runtime_principals=PRINCIPALS_CONFIG,
    )
    app = create_app(
        settings=settings,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
        result_reader=store,
    )
    return app, settings


@pytest.mark.anyio
async def test_e2e_production_loop_full_delegation_wave_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app, _ = _build_test_app(store)
    transport = httpx.ASGITransport(app=app)

    now_iso = normalize_canonical_utc_timestamp(datetime.now(UTC))
    demo_prompt = "Propose a practical deployment plan for a small self-hosted learning service."

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        # 1. Register root parent task via POST /api/tasks
        resp = await http_client.post(
            "/api/tasks",
            json={"prompt": demo_prompt, "capability": "research"},
        )
        assert resp.status_code == 202
        body = resp.json()
        parent_id = body["task_id"]

        # Grant RootAuthorityRecord supporting wave baseline
        store.authorities[parent_id] = RootAuthorityRecord.create_wave_baseline(
            root_task_id=parent_id,
            max_children_per_wave=4,
            created_at=now_iso,
        )

        # 2. Production initial Research loop on research-agent-1
        # Consumes initial hint from store, claims parent, runs ResearchProcessor with delegate_analysis_wave
        assert len(store.hints) == 1
        initial_hint = store.hints[0]
        assert initial_hint.task_id == parent_id
        assert initial_hint.capability == "research"
        assert initial_hint.execution_kind == "initial"

        monkeypatch.setenv("AGENT_ID", "research-agent-1")
        monkeypatch.setenv("RUNTIME_TOKEN", RESEARCH_1_TOKEN)
        monkeypatch.setenv("RESEARCH_DELEGATION_MODE", "delegate_analysis_wave")
        monkeypatch.setenv("RESEARCH_EXECUTOR", "stub")
        monkeypatch.setenv("REDIS_AUTH_MODE", "unauthenticated_local_test")
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.delenv("REDIS_USERNAME", raising=False)
        monkeypatch.delenv("REDIS_PASSWORD_FILE", raising=False)

        bus_1 = MockEventBus([initial_hint])
        with (
            patch("research_agent.main.RedisBus", return_value=bus_1),
            patch(
                "research_agent.main.OrchestratorClient",
                side_effect=lambda url, token, **kw: OrchestratorClient(url, token, transport=transport, **kw),
            ),
        ):
            await run_research_agent(max_signals=1)

        # 3. Verify turn yielded, executor slot freed, parent in waiting_for_children
        assert store.statuses[parent_id] == "waiting_for_children"
        assert (await store.get_active_claim(parent_id)) is None

        # Verify sealed research artifact staged
        staged_artifacts = [
            art for (t_id, _, _), art in store.artifacts.items() if t_id == parent_id
        ]
        assert len(staged_artifacts) == 1
        sealed_art = staged_artifacts[0]
        assert sealed_art.sealed is True
        assert sealed_art.artifact_type == "research_material"

        # 4. Verify 3 child hints emitted on agent.hints.analysis
        analysis_hints = [h for h in store.hints if h.capability == "analysis" and h.execution_kind == "delegated_child"]
        assert len(analysis_hints) == 3
        child_ids = [h.task_id for h in analysis_hints]
        assert len(set(child_ids)) == 3
        for c_id in child_ids:
            assert store.statuses[c_id] == "pending"

        # 5. Analysis agent runs production loop on the 3 delegated children sequentially
        monkeypatch.setenv("AGENT_ID", "analysis-agent-1")
        monkeypatch.setenv("RUNTIME_TOKEN", ANALYSIS_1_TOKEN)

        bus_analysis = MockEventBus(list(analysis_hints))
        with (
            patch("analysis_agent.main.RedisBus", return_value=bus_analysis),
            patch(
                "analysis_agent.main.OrchestratorClient",
                side_effect=lambda url, token, **kw: OrchestratorClient(url, token, transport=transport, **kw),
            ),
        ):
            await run_analysis_agent(max_signals=3)

        # Verify all 3 children completed
        for c_id in child_ids:
            assert store.statuses[c_id] == "completed"
            assert (await store.get_active_claim(c_id)) is None
            outcome = await store.get_terminal_outcome(c_id)
            assert outcome is not None
            assert outcome.terminal_status == "completed"
            child_res = await store.get_canonical_result(c_id)
            assert child_res is not None

        # 6. Verify wave barrier resolution and parent resumable
        assert store.statuses[parent_id] == "resumable"
        wave_resumption_hints = [
            h for h in store.hints
            if h.capability == "research" and h.execution_kind == "resumed_continuation"
        ]
        assert len(wave_resumption_hints) == 1
        resumed_hint = wave_resumption_hints[0]
        assert resumed_hint.task_id == parent_id
        assert resumed_hint.execution_kind == "resumed_continuation"
        assert resumed_hint.expected_continuation_version == 2

        # 7. Resumed Research production loop on distinct instance research-agent-2
        monkeypatch.setenv("AGENT_ID", "research-agent-2")
        monkeypatch.setenv("RUNTIME_TOKEN", RESEARCH_2_TOKEN)
        # Mode can be default complete_locally; resumption mode is governed by continuation version
        monkeypatch.setenv("RESEARCH_DELEGATION_MODE", "complete_locally")

        bus_2 = MockEventBus([resumed_hint])
        with (
            patch("research_agent.main.RedisBus", return_value=bus_2),
            patch(
                "research_agent.main.OrchestratorClient",
                side_effect=lambda url, token, **kw: OrchestratorClient(url, token, transport=transport, **kw),
            ),
        ):
            await run_research_agent(max_signals=1)

        # 8. Verify parent completion under epoch 2
        assert store.statuses[parent_id] == "completed"
        assert (await store.get_fencing_epoch(parent_id)) == 2
        assert (await store.get_active_claim(parent_id)) is None

        parent_outcome = await store.get_terminal_outcome(parent_id)
        assert parent_outcome is not None
        assert parent_outcome.terminal_status == "completed"
        assert parent_outcome.bound_agent_id == "research-agent-2"

        parent_result = await store.get_canonical_result(parent_id)
        assert parent_result is not None
        assert "Authoritative wave resolution: SATISFIED across 3 analysis children (barrier_policy=WAIT_ALL)" in parent_result.output
        assert "Structured child-result aggregation remains Issue #35" in parent_result.output

        # Query canonical result via GET /api/tasks/{parent_id}/result endpoint
        res_resp = await http_client.get(f"/api/tasks/{parent_id}/result")
        assert res_resp.status_code == 200
        res_data = res_resp.json()
        assert res_data["status"] == "completed"
        assert "Authoritative wave resolution: SATISFIED across 3 analysis children (barrier_policy=WAIT_ALL)" in res_data["output"]
        assert "Structured child-result aggregation remains Issue #35" in res_data["output"]

        # 9. Verify zero secret leakage across results and events
        for event in bus_1.published + bus_2.published + bus_analysis.published:
            event_str = str(event.__dict__ if hasattr(event, "__dict__") else event)
            assert RESEARCH_1_TOKEN not in event_str
            assert RESEARCH_2_TOKEN not in event_str
            assert ANALYSIS_1_TOKEN not in event_str
            assert PEPPER not in event_str
            assert PROOF_PEPPER not in event_str
