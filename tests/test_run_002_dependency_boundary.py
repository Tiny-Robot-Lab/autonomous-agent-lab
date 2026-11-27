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

PROHIBITED_ML_IMPORTS: frozenset[str] = frozenset({
    "setfit",
    "torch",
    "transformers",
    "sentence_transformers",
    "sklearn",
    "joblib",
    "pickle",
})

PROHIBITED_NETWORK_IMPORTS: frozenset[str] = frozenset({
    "socket",
    "http",
    "urllib",
    "requests",
    "httpx",
})

PROHIBITED_PRODUCTION_APP_IMPORTS: frozenset[str] = frozenset({
    "orchestrator.application",
    "orchestrator.infrastructure",
    "orchestrator.presentation",
})

_SUBPROCESS_ADAPTER_REL = "evaluation/routing/run_002/subprocess_worker.py"
_WORKER_ENTRYPOINT_REL = "evaluation/routing/run_002/worker_entrypoint.py"


def _extract_imported_modules_from_nodes(nodes: list[ast.stmt] | ast.AST) -> list[str]:
    imported: list[str] = []
    iterable = ast.walk(nodes) if isinstance(nodes, ast.AST) else nodes
    for node in iterable:
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
    *,
    top_level_only: bool = False,
) -> list[str]:
    tree = ast.parse(source_text)
    nodes: list[ast.stmt] | ast.AST = tree.body if top_level_only else tree
    violations: list[str] = []
    for name in _extract_imported_modules_from_nodes(nodes):
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


def test_run_002_parent_and_worker_dependency_isolation() -> None:
    py_files = sorted(_RUN_002_DIR.rglob("*.py"))
    assert py_files, "Expected Python files under evaluation/routing/run_002/"

    violations: dict[str, list[str]] = {}
    for py_file in py_files:
        rel_path = str(py_file.relative_to(_REPO_ROOT))
        source = py_file.read_text(encoding="utf-8")

        forbidden_for_file = set(PROHIBITED_PRODUCTION_APP_IMPORTS)
        if rel_path != _WORKER_ENTRYPOINT_REL:
            forbidden_for_file |= PROHIBITED_ML_IMPORTS | PROHIBITED_NETWORK_IMPORTS
        if rel_path != _SUBPROCESS_ADAPTER_REL:
            forbidden_for_file.add("subprocess")

        found = _find_forbidden_imports(source, frozenset(forbidden_for_file))
        if rel_path == _WORKER_ENTRYPOINT_REL:
            top_level_ml = _find_forbidden_imports(
                source,
                PROHIBITED_ML_IMPORTS,
                top_level_only=True,
            )
            found.extend(f"top-level:{m}" for m in top_level_ml)

        if found:
            violations[rel_path] = found

    assert not violations, f"Prohibited Run 002 Increment 2 imports found: {violations}"


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
