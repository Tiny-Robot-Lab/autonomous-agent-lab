from dataclasses import dataclass
from typing import Any


class ParentTaskNotFoundError(Exception):
    """Raised when an authoritative parent task does not exist."""


class DelegationDepthExceededError(Exception):
    """Raised when delegation depth policy is violated (e.g. grandchild creation under depth-1)."""


class ConflictingTaskInputError(Exception):
    """Raised when attempting to overwrite an existing TaskInput version with conflicting data."""


class ConflictingStageArtifactError(Exception):
    """Raised when attempting to overwrite an existing StageArtifact version with conflicting data."""


class ConflictingContinuationError(Exception):
    """Raised when attempting to overwrite an existing TaskContinuation version with conflicting data."""


class ConflictingRootAuthorityError(Exception):
    """Raised when attempting to overwrite an existing RootAuthorityRecord with conflicting policy data."""


class ConflictingAuthorityGrantError(Exception):
    """Raised when attempting to overwrite an existing DelegationGrant with conflicting grant data."""


class ConflictingRootTaskRegistrationError(Exception):
    """Raised when attempting to register a root task with conflicting or partial state."""


class NonContiguousVersionError(Exception):
    """Raised when a versioned record write skips a contiguous sequence number (N > 1 without N - 1)."""


class CorruptAuthoritativeRecordError(Exception):
    """Raised when an authoritative record in storage is present but malformed or invalid."""


@dataclass(frozen=True)
class Task:
    prompt: str

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError("Prompt must be a string")
        clean = self.prompt.strip()
        if not clean:
            raise ValueError("Prompt must not be empty")
        if len(self.prompt) > 2000 or len(clean) > 2000:
            raise ValueError("Prompt must not exceed 2000 characters")
        object.__setattr__(self, "prompt", clean)


@dataclass(frozen=True)
class TaskLineage:
    """Authoritative forward lineage record for a task.

    Invariants governed by ADR-0006:
    - task_id is globally unique;
    - root_task_id is required and non-empty;
    - root tasks have parent_task_id is None and root_task_id == task_id;
    - child tasks have parent_task_id is not None, parent_task_id != task_id, and root_task_id != task_id.
    """

    task_id: str
    root_task_id: str
    parent_task_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        object.__setattr__(self, "task_id", self.task_id.strip())

        if not isinstance(self.root_task_id, str) or not self.root_task_id.strip():
            raise ValueError("root_task_id must be a non-empty string")
        object.__setattr__(self, "root_task_id", self.root_task_id.strip())

        if self.parent_task_id is not None:
            if not isinstance(self.parent_task_id, str) or not self.parent_task_id.strip():
                raise ValueError("parent_task_id must be a non-empty string or None")
            object.__setattr__(self, "parent_task_id", self.parent_task_id.strip())

        if self.parent_task_id is None:
            if self.root_task_id != self.task_id:
                raise ValueError("root tasks must have root_task_id equal to task_id")
        else:
            if self.parent_task_id == self.task_id:
                raise ValueError("parent_task_id must not equal task_id")
            if self.root_task_id == self.task_id:
                raise ValueError("child tasks must not have root_task_id equal to task_id")

    @property
    def is_root(self) -> bool:
        return self.parent_task_id is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "root_task_id": self.root_task_id,
            "parent_task_id": self.parent_task_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskLineage":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        return cls(
            task_id=data["task_id"],
            root_task_id=data["root_task_id"],
            parent_task_id=data.get("parent_task_id"),
        )
