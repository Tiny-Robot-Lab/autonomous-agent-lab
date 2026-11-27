"""AST-based dependency-boundary tests for Run 002 offline shadow routing."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ORCHESTRATOR_DIR = _REPO_ROOT / "services" / "orchestrator"
_RUN_002_DIR = _REPO_ROOT / "evaluation" / "routing" / "run_002"

PROHIBITED_ORCHESTRATOR_IMPORTS: frozenset[str] = frozenset({
    "setfit",
    "torch",
    "transformers",
    "sentence_transformers",
    "sklearn",
    "evaluation.routing",
})

PROHIBITED_INCREMENT_1_IMPORTS: frozenset[str] = frozenset({
    "setfit",
    "torch",
    "transformers",
    "sentence_transformers",
    "sklearn",
    "subprocess",
    "socket",
    "http",
    "urllib",
    "requests",
    "httpx",
    "orchestrator.application",
    "orchestrator.infrastructure",
    "orchestrator.presentation",
})


def _extract_imported_modules(source_text: str) -> list[str]:
    tree = ast.parse(source_text)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                imported.append(module)
            for alias in node.names:
                combined = f"{module}.{alias.name}" if module else alias.name
                imported.append(combined)
    return imported


def _find_forbidden_imports(
    source_text: str,
    forbidden: frozenset[str],
) -> list[str]:
    violations: list[str] = []
    for name in _extract_imported_modules(source_text):
        for target in forbidden:
            if name == target or name.startswith(f"{target}."):
                violations.append(name)
    return violations


def test_services_orchestrator_has_no_ml_or_evaluation_routing_imports() -> None:
    py_files = sorted(_ORCHESTRATOR_DIR.rglob("*.py"))
    assert py_files, "Expected Python files under services/orchestrator/"

    all_violations: dict[str, list[str]] = {}
    for py_file in py_files:
        violations = _find_forbidden_imports(
            py_file.read_text(encoding="utf-8"),
            PROHIBITED_ORCHESTRATOR_IMPORTS,
        )
        if violations:
            all_violations[str(py_file.relative_to(_REPO_ROOT))] = violations

    assert not all_violations, f"Prohibited imports found in services/orchestrator: {all_violations}"


def test_run_002_increment_1_has_no_ml_subprocess_network_or_production_app_imports() -> None:
    py_files = sorted(_RUN_002_DIR.rglob("*.py"))
    assert py_files, "Expected Python files under evaluation/routing/run_002/"

    all_violations: dict[str, list[str]] = {}
    for py_file in py_files:
        violations = _find_forbidden_imports(
            py_file.read_text(encoding="utf-8"),
            PROHIBITED_INCREMENT_1_IMPORTS,
        )
        if violations:
            all_violations[str(py_file.relative_to(_REPO_ROOT))] = violations

    assert not all_violations, f"Prohibited Increment 1 imports found: {all_violations}"


def test_dependency_boundary_checker_catches_real_imports_and_ignores_comments_and_docstrings() -> None:
    harmless_source = '''"""Module discussing setfit, torch, transformers, sentence_transformers, sklearn, and evaluation.routing."""

# Comment mentioning torch and evaluation.routing should not trigger.
DESCRIPTION = "evaluation.routing and sklearn are mentioned in text only"
'''
    assert _find_forbidden_imports(harmless_source, PROHIBITED_ORCHESTRATOR_IMPORTS) == []

    for bad_import in (
        "import setfit\n",
        "import torch.nn as nn\n",
        "from transformers import AutoModel\n",
        "from sentence_transformers import SentenceTransformer\n",
        "import sklearn\n",
        "import evaluation.routing\n",
        "from evaluation import routing\n",
        "from evaluation.routing.run_002 import canonicalize_prompt\n",
    ):
        assert _find_forbidden_imports(bad_import, PROHIBITED_ORCHESTRATOR_IMPORTS), (
            f"Expected checker to catch: {bad_import!r}"
        )
