"""Bounded Research delegation coordinator and decision seam.

Governed by ADR-0006, ADR-0007, and Issue #224:
- Injectable production seam for turn-boundary delegation decisions;
- Decision domain: 'complete_locally' (production default) or 'delegate_analysis';
- No speculative autonomous planning or keyword heuristics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_contracts import TaskRequested


class ResearchDelegationCoordinator:
    """Bounded Research delegation decision coordinator.

    Production default: complete_locally.
    Injectable for collaborative workflows: delegate_analysis.
    """

    ALLOWED_DECISIONS = frozenset({"complete_locally", "delegate_analysis"})

    def __init__(self, mode: str = "complete_locally") -> None:
        clean_mode = str(mode).strip()
        if clean_mode not in self.ALLOWED_DECISIONS:
            raise ValueError(
                f"Invalid delegation mode '{mode}'. Allowed: {sorted(self.ALLOWED_DECISIONS)}"
            )
        self._mode = clean_mode

    @property
    def mode(self) -> str:
        return self._mode

    def decide(self, task: TaskRequested, execution_kind: str = "initial") -> str:
        """Return the delegation decision for a task turn.

        Resumed turns or non-initial execution kinds cannot delegate (no second wave).
        """
        if execution_kind != "initial":
            return "complete_locally"
        return self._mode
