#!/usr/bin/env python3
"""Code-grounded semantic completeness gate for Redis replay authority.

Derives the exact production command/family projection from first principles:

1. discovers every production transaction/pipeline and Lua execution site;
2. associates each mutating call with that call's actual key argument;
3. resolves Lua ``KEYS[n]`` positions through every production invocation to
   concrete canonical key families (never Lua variable names);
4. classifies worst-case Lua framing from reachable mutation counts;
5. parses the disabled-default ACL rule into an exact selector model; and
6. proves bidirectional exact equivalence between production and rule.

``replay_projection.json`` is reviewable evidence of this derivation, not the
proof itself. Artifact equality only detects code/policy drift; the semantic
proof is re-established from production code on every run. A green gate that
relies on snapshot agreement alone would be circular and is rejected here.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "services"
DEFAULT_GATEWAY = "services/orchestrator/src/orchestrator/infrastructure/redis_adapter.py"
DEFAULT_TEMPLATE = REPO_ROOT / "infrastructure" / "redis" / "users.acl.template"
DEFAULT_ENTRYPOINT = REPO_ROOT / "infrastructure" / "redis" / "redis-acl-entrypoint.sh"
DEFAULT_ARTIFACT = REPO_ROOT / "infrastructure" / "redis" / "replay_projection.json"

AUDIT_VERSION = 3

# redis-py client method name -> Redis command for mutating operations that can
# appear inside a transactional pipeline and therefore propagate to AOF inside
# MULTI/EXEC framing.
PIPELINE_MUTATING_METHODS = {
    "set": "SET",
    "rpush": "RPUSH",
    "sadd": "SADD",
    "delete": "DEL",
    "xadd": "XADD",
    "hset": "HSET",
    "lpush": "LPUSH",
    "lpop": "LPOP",
    "unlink": "UNLINK",
    "expire": "EXPIRE",
    "pexpire": "PEXPIRE",
}

# redis-py methods that are provably read-only; any other method call on a
# transactional pipeline object fails the gate closed.
PIPELINE_READ_ONLY_METHODS = {
    "get",
    "exists",
    "type",
    "smembers",
    "scard",
    "sismember",
    "xrevrange",
    "hgetall",
}

# Pipeline control operations that neither read nor write keys.
PIPELINE_CONTROL_METHODS = {"execute", "reset"}

# Lua commands that never propagate a mutation to AOF.
LUA_READ_COMMANDS = {
    "EXISTS",
    "GET",
    "TYPE",
    "SISMEMBER",
    "SCARD",
    "SMEMBERS",
    "HGETALL",
}

# Mutating Lua commands the replay envelope may legitimately contain. Any
# other mutating command inside a qualifying script fails the gate so TTL,
# UNLINK, and similar authority can only arrive through a deliberate,
# ADR-governed policy update.
LUA_ENVELOPE_COMMANDS = {"SET", "DEL", "SADD", "RPUSH", "XADD"}

EVAL_METHODS = {"eval", "evalsha"}
SCRIPT_METHODS = {"register_script", "script"}

# Bare tokens that must never appear in the disabled-default rule. Each entry
# carries its leading space so ``-@all`` cannot match ``+@all``.
FORBIDDEN_DEFAULT_TOKENS = (
    " ~*",
    " +@all",
    " +@write",
    " +@connection",
    " +@scripting",
    " +@admin",
    " +@dangerous",
    " +multi",
    " +exec",
    " +eval",
    " +unlink",
    " user default on",
    " nopass",
)

SELECT_COMMANDS = {"SET", "DEL", "SADD", "RPUSH", "XADD"}
SELECTOR_FAMILY_RE = re.compile(r"^task\.[A-Za-z0-9_]+:\*$|^agent\.[A-Za-z0-9_.]+$")

# Names that always denote the authoritative dispatch capability inside key
# expressions; they expand over the governed contract rather than to {*}.
CAPABILITY_NAMES = {
    "capability",
    "canonical_cap",
    "capability_name",
    "resolved_cap_str",
}


class ReplayAuditError(RuntimeError):
    """Gate failure: production, rule, and projection disagree or are unsafe."""


@dataclass
class LuaMutation:
    command: str
    key_expr: str
    keys_index: int | None
    families: tuple[str, ...]
    conditional: bool


@dataclass
class LuaClassification:
    name: str
    mutations: list[LuaMutation] = field(default_factory=list)
    worst_case_mutations: int = 0
    conditional_sites: int = 0
    has_mutating_loop: bool = False
    has_dynamic_mutation: bool = False
    invoked_in_outer_transaction: bool = False
    classification: str = "unclassified"
    invocation_families: dict[int, list[str]] = field(default_factory=dict)

    @property
    def pairs(self) -> set[tuple[str, str]]:
        return {
            (mutation.command, family)
            for mutation in self.mutations
            for family in mutation.families
        }


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def canonical_family(rendered_key: str) -> str:
    """Map a rendered key template to its canonical replay family.

    ``{*}`` marks a runtime-varying segment. ``task.*`` keys collapse to
    ``task.<second-segment>:*``; ``agent.*`` keys are exact queue/stream
    names and must contain no variable segment.
    """
    if rendered_key.startswith("task."):
        head = rendered_key.split(":")[0]
        parts = head.split(".")
        if len(parts) != 2 or not parts[1] or "{*}" in parts[1] or "*" in parts[1]:
            raise ReplayAuditError(
                f"production key does not resolve to a canonical family: {rendered_key!r}"
            )
        return f"task.{parts[1]}:*"
    if rendered_key.startswith("agent."):
        if "{*}" in rendered_key or "*" in rendered_key:
            raise ReplayAuditError(
                f"agent key must be an exact queue or stream name: {rendered_key!r}"
            )
        return rendered_key
    raise ReplayAuditError(
        f"production key is outside the governed replay namespaces: {rendered_key!r}"
    )


class _KeyEvaluator:
    """Resolve Redis key argument expressions to rendered key templates.

    Understands string constants, f-strings, local name assignments (unions
    over every static assignment, so if/else variants are all covered), and
    the governed capability-to-destination contract for parameter-derived
    indices and interpolations. Anything else fails closed.
    """

    def __init__(
        self,
        assignments: dict[str, list[ast.AST]],
        contract_map: dict[str, str],
        hint_contract_map: dict[str, str] | None = None,
    ) -> None:
        self._assignments = assignments
        self._contract_map = contract_map
        self._hint_map = hint_contract_map if hint_contract_map is not None else {}

    def render(self, node: ast.AST | None) -> list[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.JoinedStr):
            return self._render_joined(node, set())
        if isinstance(node, ast.Name):
            return self._render_name(node.id, set())
        if isinstance(node, ast.Subscript):
            return self._render_subscription(node, set())
        raise ReplayAuditError(
            "transactional write is missing a statically resolvable key argument"
        )

    def _render_joined(self, node: ast.JoinedStr, seen: set[str]) -> list[str]:
        pieces: list[list[str]] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                pieces.append([value.value])
            elif isinstance(value, ast.FormattedValue):
                pieces.append(self._render_interpolation(value.value, seen))
            else:
                raise ReplayAuditError("unsupported f-string segment in key argument")
        rendered = [""]
        for options in pieces:
            rendered = [prefix + option for prefix in rendered for option in options]
        return rendered

    def _render_interpolation(self, node: ast.AST | None, seen: set[str]) -> list[str]:
        if isinstance(node, ast.Name) and node.id in CAPABILITY_NAMES:
            return sorted(self._contract_map)
        return ["{*}"]

    def _render_name(self, name: str, seen: set[str]) -> list[str]:
        if name in seen:
            raise ReplayAuditError(f"circular key alias in production code: {name!r}")
        if name not in self._assignments:
            raise ReplayAuditError(f"transactional key alias has no static assignment: {name!r}")
        seen = seen | {name}
        rendered: list[str] = []
        for assignment in self._assignments[name]:
            if isinstance(assignment, ast.Subscript):
                rendered.extend(self._render_subscription(assignment, seen))
            elif isinstance(assignment, ast.Constant) and isinstance(assignment.value, str):
                rendered.append(assignment.value)
            elif isinstance(assignment, ast.JoinedStr):
                rendered.extend(self._render_joined(assignment, seen))
            elif isinstance(assignment, ast.Name):
                rendered.extend(self._render_name(assignment.id, seen))
            else:
                raise ReplayAuditError(f"key assignment is not a static string: {name!r}")
        if not rendered:
            raise ReplayAuditError(f"key alias resolves to nothing: {name!r}")
        return rendered

    def _render_subscription(self, node: ast.Subscript, seen: set[str]) -> list[str]:
        value = node.value
        is_queue_contract = (
            isinstance(value, ast.Attribute) and value.attr == "_capability_queue_contract"
        ) or (
            isinstance(value, ast.Name)
            and value.id
            in {
                "LOCAL_COMPOSE_CAPABILITY_QUEUE_MAP",
                "CAPABILITY_QUEUE_MAP",
            }
        )
        is_hint_contract = (
            isinstance(value, ast.Attribute) and value.attr == "_capability_hint_contract"
        ) or (
            isinstance(value, ast.Name)
            and value.id
            in {
                "LOCAL_COMPOSE_CAPABILITY_HINT_MAP",
                "CAPABILITY_HINT_MAP",
            }
        )
        if not (is_queue_contract or is_hint_contract):
            raise ReplayAuditError("key subscript is outside the governed contract")
        active_map = self._hint_map if is_hint_contract else self._contract_map
        if is_hint_contract and not active_map:
            raise ReplayAuditError("governed hint contract is not loaded")
        literal = _literal_string(node.slice)
        if literal is not None:
            if literal not in active_map:
                raise ReplayAuditError(f"capability {literal!r} is not in the governed contract")
            return [active_map[literal]]
        return sorted(set(active_map.values()))


def _load_contract_map(
    gateway_path: Path,
    contract_name: str = "LOCAL_COMPOSE_CAPABILITY_QUEUE_MAP",
) -> dict[str, str]:
    """Load a governed local capability-to-destination contract from gateway code."""
    tree = ast.parse(gateway_path.read_text(encoding="utf-8"))
    for node in tree.body:
        target: ast.AST | None = None
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if (
            isinstance(target, ast.Name)
            and target.id == contract_name
            and isinstance(value, ast.Dict)
        ):
            contract: dict[str, str] = {}
            for key_node, value_node in zip(node.value.keys, node.value.values):
                key = _literal_string(key_node)
                value = _literal_string(value_node)
                if key is None or value is None:
                    raise ReplayAuditError("governed contract is not a static map")
                contract[key] = value
            if not contract:
                raise ReplayAuditError("governed contract must be finite and non-empty")
            return contract
    raise ReplayAuditError("governed capability contract not found in gateway code")


def _function_assignments(func: ast.AsyncFunctionDef | ast.FunctionDef) -> dict[str, list[ast.AST]]:
    """Collect every static value assignment per local name in a function."""
    assignments: dict[str, list[ast.AST]] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                assignments.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            assignments.setdefault(node.target.id, []).append(node.value)
    return assignments


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _is_pipeline_constructor(call: ast.Call) -> bool | None:
    """None when not a pipeline constructor, else whether it is transactional."""
    if _call_name(call) != "pipeline":
        return None
    for keyword in call.keywords:
        if keyword.arg == "transaction":
            if isinstance(keyword.value, ast.Constant):
                return bool(keyword.value.value)
            raise ReplayAuditError("pipeline transaction flag is not static")
    return True


@dataclass
class _PipelineSite:
    file: str
    function: str
    lineno: int
    transactional: bool
    writes: list[tuple[str, str]] = field(default_factory=list)


def _discover_pipeline_sites(
    source_root: Path,
    gateway_relpath: str,
    contract_map: dict[str, str],
) -> tuple[list[_PipelineSite], list[str]]:
    """Find every pipeline construction; fail on any outside the gateway."""
    sites: list[_PipelineSite] = []
    outside: list[str] = []
    gateway_norm = gateway_relpath.replace("\\", "/")
    for path in sorted(source_root.rglob("*.py")):
        rel = path.relative_to(source_root.parent).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            raise ReplayAuditError(f"cannot parse production file {rel}: {exc}") from exc
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            transactional = _is_pipeline_constructor(node)
            if transactional is None:
                continue
            if node.func.value is not None and not isinstance(
                node.func.value, (ast.Attribute, ast.Name, ast.Call)
            ):
                raise ReplayAuditError(f"unrecognized pipeline receiver in {rel}")
            if rel != gateway_norm:
                outside.append(f"transactional pipeline: {rel}")
                continue
            sites.append(
                _PipelineSite(
                    file=rel, function="", lineno=node.lineno, transactional=transactional
                )
            )
    if outside:
        raise ReplayAuditError(
            "Redis transaction completeness gate failed: " + "; ".join(sorted(set(outside)))
        )
    return sites, []


def _pipeline_writes(
    gateway_path: Path,
    contract_map: dict[str, str],
    hint_contract_map: dict[str, str] | None = None,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], list[dict[str, object]]]:
    """Associate each transactional pipeline mutation with its actual key family."""
    tree = ast.parse(gateway_path.read_text(encoding="utf-8"))
    qualifying: set[tuple[str, str]] = set()
    ordinary: set[tuple[str, str]] = set()
    site_evidence: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        outside = _pipeline_constructors_outside_async_with(node)
        if outside:
            raise ReplayAuditError(
                f"pipeline acquired outside 'async with ... as pipe' in {node.name} "
                f"at lines {sorted(outside)}; the audit cannot observe its writes"
            )
        assignments = _function_assignments(node)
        evaluator = _KeyEvaluator(assignments, contract_map, hint_contract_map)
        for child in ast.walk(node):
            if not isinstance(child, ast.AsyncWith):
                continue
            pipe_vars: set[str] = set()
            is_transactional = False
            for item in child.items:
                call = item.context_expr
                if not isinstance(call, ast.Call):
                    continue
                transactional = _is_pipeline_constructor(call)
                if transactional is None:
                    continue
                is_transactional = bool(transactional)
                if isinstance(item.optional_vars, ast.Name):
                    pipe_vars.add(item.optional_vars.id)
            if not pipe_vars:
                continue
            for call in ast.walk(child):
                if not isinstance(call, ast.Call):
                    continue
                if not (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id in pipe_vars
                ):
                    continue
                method = call.func.attr
                if method in PIPELINE_READ_ONLY_METHODS | PIPELINE_CONTROL_METHODS:
                    continue
                if method not in PIPELINE_MUTATING_METHODS:
                    raise ReplayAuditError(
                        f"unclassified method {method!r} on a transactional pipeline "
                        f"in {node.name}; extend the audit before proceeding"
                    )
                command = PIPELINE_MUTATING_METHODS[method]
                for rendered in evaluator.render(call.args[0] if call.args else None):
                    pair = (command, canonical_family(rendered))
                    if is_transactional:
                        qualifying.add(pair)
                    else:
                        ordinary.add(pair)
                    site_evidence.append(
                        {
                            "function": node.name,
                            "transactional": is_transactional,
                            "command": command,
                            "key": rendered,
                            "family": pair[1],
                        }
                    )
    return qualifying, ordinary, site_evidence


def _strip_lua_comments(source: str) -> str:
    """Remove Lua line comments while preserving string contents."""
    result: list[str] = []
    i = 0
    n = len(source)
    in_string: str | None = None
    while i < n:
        char = source[i]
        if in_string:
            result.append(char)
            if char == "\\" and i + 1 < n:
                result.append(source[i + 1])
                i += 2
                continue
            if char == in_string:
                in_string = None
            i += 1
            continue
        if char in ("'", '"'):
            in_string = char
            result.append(char)
            i += 1
            continue
        if char == "[" and source.startswith("[", i) and _long_bracket_open(source, i):
            level = _long_bracket_level(source, i)
            result.append(source[i : i + 2 + level])
            i += 2 + level
            closing = f"]{('=' * level)}]"
            end = source.find(closing, i)
            if end == -1:
                result.append(source[i:])
                break
            result.append(source[i : end + len(closing)])
            i = end + len(closing)
            continue
        if char == "-" and source.startswith("--", i):
            j = i + 2
            if j < n and source[j] == "[" and _long_bracket_open(source, j):
                level = _long_bracket_level(source, j)
                closing = f"]{('=' * level)}]"
                end = source.find(closing, j + 1 + level)
                if end == -1:
                    raise ReplayAuditError(
                        "Lua source has an unterminated long-bracket comment "
                        "the audit cannot classify"
                    )
                i = end + len(closing)
                continue
            while i < n and source[i] != "\n":
                i += 1
            continue
        result.append(char)
        i += 1
    return "".join(result)


def _long_bracket_open(source: str, i: int) -> bool:
    j = i + 1
    while j < len(source) and source[j] == "=":
        j += 1
    return j < len(source) and source[j] == "["


def _long_bracket_level(source: str, i: int) -> int:
    level = 0
    j = i + 1
    while j < len(source) and source[j] == "=":
        level += 1
        j += 1
    return level


@dataclass
class _LuaCallSite:
    command: str | None
    key_expr: str | None
    dynamic: bool
    start: int
    end: int


def _find_redis_calls(clean: str) -> list[_LuaCallSite]:
    """Locate every redis.call/redis.pcall with its command and key argument."""
    sites: list[_LuaCallSite] = []
    pattern = re.compile(r"redis\.(call|pcall)\s*\(")
    for match in pattern.finditer(clean):
        depth = 1
        i = match.end()
        in_string: str | None = None
        while i < len(clean) and depth > 0:
            char = clean[i]
            if in_string:
                if char == "\\":
                    i += 1
                elif char == in_string:
                    in_string = None
            elif char in ("'", '"'):
                in_string = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            i += 1
        arg_text = clean[match.end() : i - 1]
        parts = _split_top_level_args(arg_text)
        first = parts[0].strip() if parts else ""
        literal = re.fullmatch(r"""['"]([A-Za-z]+)['"]""", first)
        if literal:
            command: str | None = literal.group(1).upper()
            key_expr: str | None = parts[1].strip() if len(parts) > 1 else None
            sites.append(
                _LuaCallSite(
                    command=command,
                    key_expr=key_expr,
                    dynamic=False,
                    start=match.start(),
                    end=i,
                )
            )
            continue
        unpack = re.fullmatch(r"unpack\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", first)
        if unpack:
            sites.append(
                _LuaCallSite(
                    command=None,
                    key_expr=None,
                    dynamic=False,
                    start=match.start(),
                    end=i,
                )
            )
            sites[-1].key_expr = f"unpack:{unpack.group(1)}"
            continue
        sites.append(
            _LuaCallSite(command=None, key_expr=None, dynamic=True, start=match.start(), end=i)
        )
    return sites


def _split_top_level_args(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    in_string: str | None = None
    i = 0
    while i < len(text):
        char = text[i]
        if in_string:
            current.append(char)
            if char == "\\" and i + 1 < len(text):
                current.append(text[i + 1])
                i += 1
            elif char == in_string:
                in_string = None
        elif char in ("'", '"'):
            in_string = char
            current.append(char)
        elif char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        i += 1
    parts.append("".join(current))
    return parts


def _block_spans(clean: str, opener: str, closer: str = "end") -> list[tuple[int, int]]:
    """Find matching opener/closer spans, honouring nesting of all Lua blocks."""
    tokens = re.compile(
        r"\b(for|while|if|function|do|repeat|end|until)\b|'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\""
    )
    stack: list[tuple[str, int]] = []
    spans: list[tuple[int, int]] = []
    for match in tokens.finditer(clean):
        token = match.group(0)
        if token.startswith(("'", '"')):
            continue
        if token in ("for", "while", "if", "function"):
            stack.append((token, match.start()))
        elif token == "do" and (not stack or stack[-1][0] not in ("for", "while")):
            stack.append(("do", match.start()))
        elif token == "repeat":
            stack.append(("repeat", match.start()))
        elif token == "until":
            if stack and stack[-1][0] == "repeat":
                kind, start = stack.pop()
                spans.append((start, match.end()))
        elif token == "end":
            if not stack:
                continue
            kind, start = stack.pop()
            if kind == opener or (opener == "loop" and kind in ("for", "while", "do")):
                spans.append((start, match.end()))
    return spans


def _conditional_spans(clean: str) -> list[tuple[int, int]]:
    return _block_spans(clean, "if")


def _resolve_unpack_tables(clean: str) -> dict[str, tuple[str, str]]:
    """Resolve ``local T = {'CMD', KEYEXPR, ...}`` argument tables."""
    tables: dict[str, tuple[str, str]] = {}
    for match in re.finditer(r"local\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{([^}]*)\}", clean):
        name = match.group(1)
        parts = _split_top_level_args(match.group(2))
        if len(parts) < 2:
            continue
        literal = re.fullmatch(r"""['"]([A-Za-z]+)['"]""", parts[0].strip())
        if literal:
            tables[name] = (literal.group(1).upper(), parts[1].strip())
    return tables


def _lua_key_aliases(clean: str) -> dict[str, int]:
    """Map every ``local NAME = KEYS[n]`` alias to its key position."""
    aliases: dict[str, int] = {}
    for match in re.finditer(r"local\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*KEYS\[(\d+)\]", clean):
        aliases[match.group(1)] = int(match.group(2))
    return aliases


_FLOW_BLOCK_KEYWORDS = {"if", "for", "while", "function", "do"}


@dataclass
class _FlowToken:
    kind: str
    start: int
    mutating: bool = False


def _mask_lua_strings(clean: str) -> str:
    """Blank out quoted spans so keywords inside strings cannot parse as code."""
    out = list(clean)
    i = 0
    while i < len(clean):
        quote = clean[i]
        if quote not in ("'", '"'):
            i += 1
            continue
        out[i] = " "
        i += 1
        while i < len(clean) and clean[i] != quote:
            if clean[i] == "\\" and i + 1 < len(clean):
                out[i] = " "
                i += 1
            if clean[i] != "\n":
                out[i] = " "
            i += 1
        if i < len(clean):
            out[i] = " "
            i += 1
    return "".join(out)


def _tokenize_lua_flow(masked: str, mutating_starts: set[int]) -> list[_FlowToken]:
    """Tokenize control-flow keywords, labels, and redis call sites in order."""
    tokens: list[_FlowToken] = []
    pattern = re.compile(
        r"::[A-Za-z_][A-Za-z0-9_]*::"
        r"|\b(?:if|then|elseif|else|end|for|while|do|repeat|until|return|function|break|goto)\b"
        r"|redis\.(?:call|pcall)\s*\("
    )
    for match in pattern.finditer(masked):
        text = match.group(0)
        if text.startswith("::"):
            tokens.append(_FlowToken("label", match.start()))
        elif text.startswith("redis"):
            tokens.append(_FlowToken("call", match.start(), match.start() in mutating_starts))
        else:
            tokens.append(_FlowToken(text, match.start()))
    return tokens


def _worst_case_mutations(name: str, clean: str, mutating_starts: set[int]) -> tuple[int, bool]:
    """Compute the worst-case per-path mutation count by control-flow analysis.

    Sequential statements accumulate. Each ``if`` contributes the maximum of
    its reachable branches (including the implicit zero-mutation branch when
    no ``else`` exists). Early ``return`` terminates its path. Mutating loops
    force multi-effect treatment. Anything unclassifiable fails closed.
    """
    tokens = _tokenize_lua_flow(_mask_lua_strings(clean), mutating_starts)
    pos = 0
    has_mutating_loop = False

    def peek() -> _FlowToken | None:
        return tokens[pos] if pos < len(tokens) else None

    def skip_to_keyword(targets: set[str], what: str) -> None:
        nonlocal pos
        while True:
            token = peek()
            if token is None:
                raise ReplayAuditError(f"Lua script {name} ends inside an unclosed {what}")
            if token.kind in targets:
                pos += 1
                return
            if token.kind in _FLOW_BLOCK_KEYWORDS or token.kind in {
                "then",
                "elseif",
                "else",
                "end",
                "repeat",
                "until",
                "return",
                "break",
                "goto",
                "label",
                "function",
            }:
                raise ReplayAuditError(f"Lua script {name} has an unclassifiable {what}")
            if token.kind == "call" and token.mutating:
                raise ReplayAuditError(
                    f"Lua script {name} mutates inside a {what}, which the audit cannot bound"
                )
            pos += 1

    def skip_function_body() -> None:
        nonlocal pos
        depth = 1
        while True:
            token = peek()
            if token is None:
                raise ReplayAuditError(f"Lua script {name} ends inside an unclosed function")
            if token.kind in _FLOW_BLOCK_KEYWORDS:
                depth += 1
            elif token.kind == "end":
                depth -= 1
                if depth == 0:
                    pos += 1
                    return
            elif token.kind in ("repeat", "goto", "label"):
                raise ReplayAuditError(
                    f"Lua script {name} uses control flow the audit cannot classify"
                )
            elif token.kind == "call" and token.mutating:
                raise ReplayAuditError(
                    f"Lua script {name} mutates inside a function body, which the audit "
                    "cannot bound to a caller path"
                )
            pos += 1

    def block_has_mutation(stmts: list[tuple]) -> bool:
        for stmt in stmts:
            if stmt[0] == "mut":
                return True
            if stmt[0] == "if" and any(block_has_mutation(branch) for branch in stmt[1]):
                return True
            if stmt[0] == "loop" and block_has_mutation(stmt[1]):
                return True
        return False

    def branch_info(stmts: list[tuple]) -> tuple[int, bool]:
        """Max mutations on any path through stmts, and whether any path falls through.

        Statements after an ``if`` are reachable only via its falling-through
        branches, so each branch contributes its own worst case to terminated
        paths while only falling branches feed the continuing accumulator.
        """
        best = 0
        acc = 0
        alive = True
        for stmt in stmts:
            if not alive:
                break
            if stmt[0] == "return":
                best = max(best, acc)
                alive = False
            elif stmt[0] == "mut":
                acc += 1
            elif stmt[0] == "if":
                infos = [branch_info(branch) for branch in stmt[1]]
                for branch_worst, _falls in infos:
                    best = max(best, acc + branch_worst)
                falling = [branch_worst for branch_worst, falls in infos if falls]
                if not falling:
                    alive = False
                else:
                    acc += max(falling)
            elif stmt[0] == "loop":
                acc += branch_info(stmt[1])[0]
        if alive:
            best = max(best, acc)
        return best, alive

    def parse_block(stop: set[str], loop_depth: int = 0) -> list[tuple]:
        nonlocal pos, has_mutating_loop
        stmts: list[tuple] = []
        while True:
            token = peek()
            if token is None:
                if stop:
                    raise ReplayAuditError(f"Lua script {name} ends inside an unclosed block")
                return stmts
            if token.kind in stop:
                return stmts
            if token.kind == "call":
                if token.mutating:
                    stmts.append(("mut",))
                pos += 1
            elif token.kind == "return":
                stmts.append(("return",))
                pos += 1
            elif token.kind == "break":
                # A break outside any loop is invalid Lua the audit cannot
                # bound; a break inside a loop keeps the conservative
                # whole-body worst case, so mutation bounds cannot shrink.
                if loop_depth == 0:
                    raise ReplayAuditError(
                        f"Lua script {name} uses break outside a loop, "
                        "which the audit cannot classify"
                    )
                stmts.append(("break",))
                pos += 1
            elif token.kind == "if":
                pos += 1
                skip_to_keyword({"then"}, "if condition")
                branches = [parse_block({"elseif", "else", "end"}, loop_depth)]
                while peek() is not None and peek().kind == "elseif":  # type: ignore[union-attr]
                    pos += 1
                    skip_to_keyword({"then"}, "elseif condition")
                    branches.append(parse_block({"elseif", "else", "end"}, loop_depth))
                if peek() is not None and peek().kind == "else":  # type: ignore[union-attr]
                    pos += 1
                    branches.append(parse_block({"end"}, loop_depth))
                else:
                    branches.append([])
                if peek() is None or peek().kind != "end":  # type: ignore[union-attr]
                    raise ReplayAuditError(f"Lua script {name} has an if without a closing end")
                pos += 1
                stmts.append(("if", branches))
            elif token.kind in ("for", "while"):
                pos += 1
                skip_to_keyword({"do"}, "loop header")
                body = parse_block({"end"}, loop_depth + 1)
                if peek() is None or peek().kind != "end":  # type: ignore[union-attr]
                    raise ReplayAuditError(f"Lua script {name} has a loop without a closing end")
                pos += 1
                if block_has_mutation(body):
                    has_mutating_loop = True
                    stmts.append(("loop", body, True))
                else:
                    stmts.append(("loop", body, False))
            elif token.kind == "do":
                pos += 1
                inner = parse_block({"end"}, loop_depth)
                if peek() is None or peek().kind != "end":  # type: ignore[union-attr]
                    raise ReplayAuditError(
                        f"Lua script {name} has a do block without a closing end"
                    )
                pos += 1
                stmts.extend(inner)
            elif token.kind == "function":
                pos += 1
                skip_function_body()
            elif token.kind in (
                "repeat",
                "goto",
                "label",
                "until",
                "then",
                "elseif",
                "else",
                "end",
            ):
                raise ReplayAuditError(
                    f"Lua script {name} uses control flow the audit cannot classify: {token.kind}"
                )
            else:  # pragma: no cover - defensive
                raise ReplayAuditError(
                    f"Lua script {name} uses control flow the audit cannot classify"
                )

    statements = parse_block(set())
    if pos != len(tokens):
        raise ReplayAuditError(f"Lua script {name} has trailing tokens the audit cannot classify")
    worst_count, _falls_through = branch_info(statements)
    return worst_count, has_mutating_loop


def _lua_key_position(key_expr: str, aliases: dict[str, int]) -> int | str | None:
    """Resolve a Lua key argument to a KEYS position, literal, or dynamic tag."""
    expression = key_expr.strip()
    direct = re.fullmatch(r"KEYS\[(\d+)\]", expression)
    if direct:
        return int(direct.group(1))
    if expression in aliases:
        return aliases[expression]
    literal = re.fullmatch(r"""['"]([^'"]+)['"]""", expression)
    if literal:
        return literal.group(1)
    return None


def classify_lua_script(
    name: str,
    source: str,
    invocation_families: dict[int, list[str]] | None = None,
    invoked_in_outer_transaction: bool = False,
) -> LuaClassification:
    """Classify one Lua script's worst-case propagated framing.

    Control-flow analysis computes the worst-case per-path mutation count:
    sequential statements accumulate, each ``if`` contributes the maximum of
    its reachable branches (including the implicit zero branch without
    ``else``), and early ``return`` terminates its path. Single-effect
    requires at most one mutation on every reachable path, no mutating loop,
    no dynamic or unbounded command construction, and no outer-transaction
    invocation. Anything else is multi-effect (qualifying) or fail-closed.
    The command/family pair union always covers every reachable mutation,
    even when the per-path worst case stays at one.
    """
    clean = _strip_lua_comments(source)
    if re.search(r"\[(=*)\[", clean):
        raise ReplayAuditError(
            f"Lua script {name} uses a long-bracket string form the audit cannot classify"
        )
    if re.search(r"\brepeat\b", clean):
        raise ReplayAuditError(
            f"Lua script {name} uses repeat/until framing the audit cannot classify"
        )
    aliases = _lua_key_aliases(clean)
    unpack_tables = _resolve_unpack_tables(clean)
    conditional_spans = _conditional_spans(clean)
    mutating_starts: set[int] = set()
    families = invocation_families or {}
    classification = LuaClassification(
        name=name,
        invoked_in_outer_transaction=invoked_in_outer_transaction,
        invocation_families={k: sorted(v) for k, v in families.items()},
    )
    for site in _find_redis_calls(clean):
        command = site.command
        key_expr = site.key_expr
        if site.dynamic:
            classification.has_dynamic_mutation = True
            classification.mutations.append(
                LuaMutation(
                    command="DYNAMIC",
                    key_expr="",
                    keys_index=None,
                    families=(),
                    conditional=any(start <= site.start <= end for start, end in conditional_spans),
                )
            )
            continue
        if key_expr is not None and key_expr.startswith("unpack:"):
            table = key_expr.split(":", 1)[1]
            if table not in unpack_tables:
                raise ReplayAuditError(
                    f"Lua script {name} unpacks an unresolvable argument table: {table}"
                )
            command, key_expr = unpack_tables[table]
        if command is None:
            classification.has_dynamic_mutation = True
            continue
        if command in LUA_READ_COMMANDS:
            continue
        if command not in LUA_ENVELOPE_COMMANDS:
            raise ReplayAuditError(
                f"Lua script {name} issues {command}, which has no replay justification"
            )
        position = _lua_key_position(key_expr or "", aliases)
        if position is None or isinstance(position, int) and position < 1:
            raise ReplayAuditError(
                f"Lua script {name} mutates through an unresolvable key: {key_expr!r}"
            )
        if isinstance(position, int):
            resolved = families.get(position, [])
            if not resolved:
                raise ReplayAuditError(
                    f"Lua script {name} mutates KEYS[{position}] with no production "
                    "invocation family"
                )
            pair_families = tuple(sorted(set(resolved)))
        else:
            pair_families = (canonical_family(position),)
        mutating_starts.add(site.start)
        classification.mutations.append(
            LuaMutation(
                command=command,
                key_expr=key_expr or "",
                keys_index=position if isinstance(position, int) else None,
                families=pair_families,
                conditional=any(start <= site.start <= end for start, end in conditional_spans),
            )
        )
    worst_case, flow_has_mutating_loop = _worst_case_mutations(name, clean, mutating_starts)
    if flow_has_mutating_loop:
        classification.has_mutating_loop = True
    classification.worst_case_mutations = worst_case
    classification.conditional_sites = sum(
        1 for mutation in classification.mutations if mutation.conditional
    )
    if classification.has_dynamic_mutation:
        raise ReplayAuditError(
            f"Lua script {name} builds a dynamic mutating command the audit cannot bound"
        )
    if (
        not classification.has_mutating_loop
        and not invoked_in_outer_transaction
        and classification.worst_case_mutations <= 1
    ):
        classification.classification = "single-effect"
    else:
        classification.classification = "multi-effect"
    return classification


@dataclass
class _EvalSite:
    file: str
    function: str
    script: str
    lineno: int
    key_families: dict[int, list[str]]
    in_outer_transaction: bool


def _discover_eval_sites(
    source_root: Path,
    gateway_relpath: str,
    gateway_path: Path,
    contract_map: dict[str, str],
    hint_contract_map: dict[str, str] | None = None,
) -> tuple[list[_EvalSite], dict[str, str]]:
    """Find every Lua execution site; fail on any outside the gateway.

    Returns the sites with per-position resolved key families, plus every
    module-level Lua script constant defined in the gateway.
    """
    tree = ast.parse(gateway_path.read_text(encoding="utf-8"))
    lua_sources: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and ("redis.call" in node.value.value or "redis.pcall" in node.value.value)
        ):
            lua_sources[node.targets[0].id] = node.value.value

    gateway_norm = gateway_relpath.replace("\\", "/")
    sites: list[_EvalSite] = []
    outside: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        rel = path.relative_to(source_root.parent).as_posix()
        try:
            file_tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            raise ReplayAuditError(f"cannot parse production file {rel}: {exc}") from exc
        for node in ast.walk(file_tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name in SCRIPT_METHODS:
                outside.append(f"Lua script loading ({name}): {rel}")
            elif name in EVAL_METHODS and rel != gateway_norm:
                outside.append(f"Lua execution: {rel}")
    if outside:
        raise ReplayAuditError(
            "Redis Lua completeness gate failed: " + "; ".join(sorted(set(outside)))
        )
    raw_multi = _find_raw_multi_exec(source_root, gateway_relpath)
    if raw_multi:
        raise ReplayAuditError(
            "raw MULTI/EXEC pathway requires audit support: " + "; ".join(raw_multi)
        )

    gateway_tree = ast.parse(gateway_path.read_text(encoding="utf-8"))
    for func in ast.walk(gateway_tree):
        if not isinstance(func, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        assignments = _function_assignments(func)
        evaluator = _KeyEvaluator(assignments, contract_map, hint_contract_map)
        pipeline_vars = _pipeline_vars_in_func(func)
        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node) not in EVAL_METHODS:
                continue
            if not node.args:
                raise ReplayAuditError(f"Lua eval without script in {func.name}")
            script_node = node.args[0]
            if not isinstance(script_node, ast.Name) or script_node.id not in lua_sources:
                raise ReplayAuditError(
                    f"Lua eval with a non-static script in {func.name}; the audit cannot bound it"
                )
            script_name = script_node.id
            key_families = _resolve_eval_keys(node, func, assignments, evaluator)
            in_outer = _inside_pipeline(node, func, pipeline_vars)
            sites.append(
                _EvalSite(
                    file=gateway_norm,
                    function=func.name,
                    script=script_name,
                    lineno=node.lineno,
                    key_families=key_families,
                    in_outer_transaction=in_outer,
                )
            )
    return sites, lua_sources


def _find_raw_multi_exec(source_root: Path, gateway_relpath: str) -> list[str]:
    hits: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        rel = path.relative_to(source_root.parent).as_posix()
        try:
            file_tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(file_tree):
            if isinstance(node, ast.Call) and _call_name(node) in {"multi", "exec"}:
                hits.append(f"{rel}:{node.lineno}")
    return hits


def _find_raw_execute_command(source_root: Path) -> list[str]:
    """Flag every raw execute_command: the audit has no model for it.

    Static MULTI/EXEC (any case) would bypass the framing analysis, a
    dynamic command cannot be bounded, and even any other static command
    would be silently omitted from the write evidence. All three fail
    closed until the audit deliberately models the call.
    """
    hits: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        rel = path.relative_to(source_root.parent).as_posix()
        try:
            file_tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(file_tree):
            if not (isinstance(node, ast.Call) and _call_name(node) == "execute_command"):
                continue
            first = _literal_string(node.args[0]) if node.args else None
            if first is not None and first.upper() in {"MULTI", "EXEC"}:
                hits.append(f"{rel}:{node.lineno} raw {first.upper()} framing")
            elif first is None:
                hits.append(f"{rel}:{node.lineno} dynamic execute_command")
            else:
                hits.append(f"{rel}:{node.lineno} execute_command without audit model")
    return hits


def _pipeline_constructors_outside_async_with(
    func: ast.AsyncFunctionDef | ast.FunctionDef,
) -> list[int]:
    """Line numbers of pipeline() calls not acquired via ``async with``.

    The write association below only observes the supported
    ``async with ... pipeline(...) as pipe`` shape; any other acquisition
    would silently omit its writes, so it fails closed here instead.
    """
    managed: set[int] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.AsyncWith):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    managed.add(id(item.context_expr))
    outside: list[int] = []
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and _is_pipeline_constructor(node) is not None
            and id(node) not in managed
        ):
            outside.append(node.lineno)
    return outside


def _pipeline_vars_in_func(func: ast.AST) -> set[str]:
    variables: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.AsyncWith):
            for item in node.items:
                if (
                    isinstance(item.context_expr, ast.Call)
                    and _is_pipeline_constructor(item.context_expr) is not None
                    and isinstance(item.optional_vars, ast.Name)
                ):
                    variables.add(item.optional_vars.id)
    return variables


def _inside_pipeline(node: ast.AST, func: ast.AST, pipeline_vars: set[str]) -> bool:
    for parent in ast.walk(func):
        if not isinstance(parent, ast.AsyncWith):
            continue
        for item in parent.items:
            if not (
                isinstance(item.context_expr, ast.Call)
                and _is_pipeline_constructor(item.context_expr)
                and isinstance(item.optional_vars, ast.Name)
            ):
                continue
            for call in ast.walk(parent):
                if call is node:
                    return True
    return False


def _resolve_eval_keys(
    node: ast.Call,
    func: ast.AST,
    assignments: dict[str, list[ast.AST]],
    evaluator: _KeyEvaluator,
) -> dict[int, list[str]]:
    """Resolve KEYS positions to canonical families for one eval call."""
    if len(node.args) < 2:
        raise ReplayAuditError(f"Lua eval without a key count in {node.lineno}")
    count_node = node.args[1]
    key_exprs: list[ast.AST] = []
    if isinstance(count_node, ast.Constant) and isinstance(count_node.value, int):
        expanded: list[ast.AST] = []
        for arg in node.args[2:]:
            if isinstance(arg, ast.Starred):
                if not isinstance(arg.value, ast.Name):
                    raise ReplayAuditError(
                        f"Lua eval spreads a non-static key list in {node.lineno}"
                    )
                expanded.extend(_expand_key_list(arg.value.id, func, assignments))
            else:
                expanded.append(arg)
        if len(expanded) < count_node.value:
            raise ReplayAuditError(f"Lua eval passes fewer keys than declared in {node.lineno}")
        key_exprs.extend(expanded[: count_node.value])
    elif (
        isinstance(count_node, ast.Call)
        and _call_name(count_node) == "len"
        and count_node.args
        and isinstance(count_node.args[0], ast.Name)
    ):
        # numkeys == len(keys): the starred list supplies every key and all
        # remaining positional arguments are ARGV by construction.
        key_exprs.extend(_expand_key_list(count_node.args[0].id, func, assignments))
    else:
        raise ReplayAuditError(f"Lua eval key count is not statically resolvable at {node.lineno}")
    families: dict[int, list[str]] = {}
    for position, expr in enumerate(key_exprs, start=1):
        rendered = evaluator.render(expr)
        families[position] = sorted({canonical_family(key) for key in rendered})
    return families


def _expand_key_list(
    name: str, func: ast.AST, assignments: dict[str, list[ast.AST]]
) -> list[ast.AST]:
    """Expand ``*keys`` to base elements plus every conditional append."""
    base: list[ast.AST] = []
    if name not in assignments:
        raise ReplayAuditError(f"Lua key list {name!r} has no static assignment")
    for assignment in assignments[name]:
        if isinstance(assignment, ast.List):
            if base:
                raise ReplayAuditError(f"Lua key list {name!r} has competing base assignments")
            base = list(assignment.elts)
    appends: list[tuple[int, ast.AST]] = []
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == name
            and node.args
        ):
            appends.append((node.lineno, node.args[0]))
    appends.sort(key=lambda item: item[0])
    return base + [expr for _, expr in appends]


def _ordinary_direct_writes(
    gateway_path: Path,
    contract_map: dict[str, str],
    hint_contract_map: dict[str, str] | None = None,
) -> tuple[set[tuple[str, str]], list[dict[str, object]]]:
    """Enumerate ordinary (non-transactional, non-Lua) gateway writes as evidence."""
    tree = ast.parse(gateway_path.read_text(encoding="utf-8"))
    ordinary: set[tuple[str, str]] = set()
    dynamic_sites: list[dict[str, object]] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        assignments = _function_assignments(func)
        evaluator = _KeyEvaluator(assignments, contract_map, hint_contract_map)
        pipeline_vars = _pipeline_vars_in_func(func)
        for node in ast.walk(func):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr
            if method in EVAL_METHODS or method == "pipeline":
                continue
            if method not in PIPELINE_MUTATING_METHODS:
                continue
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and receiver.id in pipeline_vars:
                continue
            if not (isinstance(receiver, ast.Attribute) and receiver.attr in {"_client", "client"}):
                continue
            try:
                rendered = evaluator.render(node.args[0] if node.args else None)
            except ReplayAuditError:
                dynamic_sites.append({"function": func.name, "method": method})
                continue
            for key in rendered:
                ordinary.add((PIPELINE_MUTATING_METHODS[method], canonical_family(key)))
    return ordinary, dynamic_sites


@dataclass
class DefaultRuleModel:
    line: str
    selectors: dict[str, set[str]]
    has_select: bool


def parse_default_rule(line: str) -> DefaultRuleModel:
    """Parse the disabled-default rule into an exact selector model.

    Redis ACL commands and categories are case-insensitive, so every
    semantic comparison folds the input to lowercase first. Selector mode
    (``%W~``) and key-family spelling stay case-sensitive: family case is
    semantically relevant, diagnostics reproduce the original text, and the
    canonical-family check requires the original content.
    """
    tokens = line.split()
    lowered = [token.lower() for token in tokens]
    if len(tokens) < 7 or lowered[0] != "user" or lowered[1] != "default":
        raise ReplayAuditError("disabled-default rule has an unexpected shape")
    if lowered[2] != "off":
        raise ReplayAuditError("disabled-default user must remain off")
    for required in ("resetpass", "resetkeys", "resetchannels", "-@all"):
        if required not in lowered:
            raise ReplayAuditError(f"disabled-default rule is missing required token: {required}")
    folded_line = " " + " ".join(lowered) + " "
    for forbidden in FORBIDDEN_DEFAULT_TOKENS:
        if forbidden in folded_line:
            raise ReplayAuditError(
                f"disabled-default rule carries prohibited authority: {forbidden.strip()}"
            )
    if "task.routing:" in folded_line:
        raise ReplayAuditError("disabled-default rule must not carry task.routing selectors")
    if re.search(r"(^|\s)>[^\s]*(\s|$)", line):
        raise ReplayAuditError("disabled-default rule carries a plaintext password rule")
    if re.search(r"(^|\s)#[0-9a-fA-F]{64}(\s|$)", line):
        raise ReplayAuditError("disabled-default rule carries a committed password hash")
    selectors: dict[str, set[str]] = {}
    has_select = False
    canonical_commands = {command.lower(): command for command in SELECT_COMMANDS}
    structural = {"user", "default", "off", "resetpass", "resetkeys", "resetchannels", "-@all"}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        fold = lowered[index]
        if fold == "+select":
            has_select = True
        elif fold in structural:
            pass
        elif fold.startswith("(+"):
            command = canonical_commands.get(fold[2:])
            if command is None:
                raise ReplayAuditError(
                    f"disabled-default selector group is unknown or unsupported: {token}"
                )
            families: set[str] = set()
            index += 1
            while index < len(tokens) and not tokens[index].endswith(")"):
                family_token = tokens[index]
                if not family_token.startswith("%W~"):
                    raise ReplayAuditError(
                        f"disabled-default selector for {command} is not write-only: {family_token}"
                    )
                family = family_token[len("%W~") :]
                if not SELECTOR_FAMILY_RE.fullmatch(family):
                    raise ReplayAuditError(
                        f"disabled-default selector family is not canonical: {family}"
                    )
                families.add(family)
                index += 1
            if index >= len(tokens):
                raise ReplayAuditError(
                    f"disabled-default selector group for {command} is unterminated"
                )
            last = tokens[index]
            if not last.endswith(")"):
                raise ReplayAuditError(
                    f"disabled-default selector group for {command} is malformed"
                )
            if not last[:-1].startswith("%W~"):
                raise ReplayAuditError(
                    f"disabled-default selector for {command} is not write-only: {last[:-1]}"
                )
            family = last[len("%W~") : -1]
            if not SELECTOR_FAMILY_RE.fullmatch(family):
                raise ReplayAuditError(
                    f"disabled-default selector family is not canonical: {family}"
                )
            families.add(family)
            selectors.setdefault(command.upper(), set()).update(families)
        else:
            raise ReplayAuditError(
                f"disabled-default rule carries an unclassifiable token: {token}"
            )
        index += 1
    for command, families in selectors.items():
        if command == "DEL" and families - {"task.claim:*"}:
            raise ReplayAuditError("disabled-default DEL authority is bounded to task.claim:* only")
        for family in families:
            if not SELECTOR_FAMILY_RE.fullmatch(family):
                raise ReplayAuditError(
                    f"disabled-default selector family is not canonical: {family}"
                )
    return DefaultRuleModel(line=line, selectors=selectors, has_select=has_select)


def _read_default_line(template_path: Path) -> str:
    lines = [
        line
        for line in template_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("user default ")
    ]
    if len(lines) != 1:
        raise ReplayAuditError("ACL template must contain exactly one default-user rule")
    return lines[0]


def _read_entrypoint_canonical(entrypoint_path: Path) -> str:
    text = entrypoint_path.read_text(encoding="utf-8")
    match = re.search(r"canonical_default='([^']*)'", text)
    if not match:
        raise ReplayAuditError("entrypoint canonical default rule not found")
    return match.group(1)


def derive_projection(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    gateway_relpath: str = DEFAULT_GATEWAY,
    template_path: Path = DEFAULT_TEMPLATE,
    entrypoint_path: Path = DEFAULT_ENTRYPOINT,
) -> dict[str, object]:
    """Derive and prove the replay projection from production code."""
    gateway_path = source_root.parent / gateway_relpath
    contract_map = _load_contract_map(gateway_path)
    hint_contract_map = _load_contract_map(
        gateway_path, contract_name="LOCAL_COMPOSE_CAPABILITY_HINT_MAP"
    )

    _discover_pipeline_sites(source_root, gateway_relpath, contract_map)
    raw_execute = _find_raw_execute_command(source_root)
    if raw_execute:
        raise ReplayAuditError(
            "raw execute_command pathway requires audit support: "
            + "; ".join(sorted(set(raw_execute)))
        )
    pipeline_qualifying, pipeline_ordinary, pipeline_evidence = _pipeline_writes(
        gateway_path, contract_map, hint_contract_map
    )

    eval_sites, lua_sources = _discover_eval_sites(
        source_root, gateway_relpath, gateway_path, contract_map, hint_contract_map
    )
    families_by_script: dict[str, dict[int, list[str]]] = {}
    outer_by_script: dict[str, bool] = {}
    for site in eval_sites:
        merged = families_by_script.setdefault(site.script, {})
        for position, site_families in site.key_families.items():
            merged.setdefault(position, [])
            for family in site_families:
                if family not in merged[position]:
                    merged[position].append(family)
        outer_by_script[site.script] = (
            outer_by_script.get(site.script, False) or site.in_outer_transaction
        )

    classifications: dict[str, LuaClassification] = {}
    for script_name, source in lua_sources.items():
        invoked = any(site.script == script_name for site in eval_sites)
        if not invoked:
            raise ReplayAuditError(
                f"Lua script {script_name} has no production invocation to bind KEYS"
            )
        classifications[script_name] = classify_lua_script(
            script_name,
            source,
            invocation_families=families_by_script.get(script_name, {}),
            invoked_in_outer_transaction=outer_by_script.get(script_name, False),
        )

    lua_qualifying: set[tuple[str, str]] = set()
    lua_excluded: list[dict[str, object]] = []
    for script_name in sorted(classifications):
        classification = classifications[script_name]
        if classification.classification == "multi-effect":
            lua_qualifying.update(classification.pairs)
        else:
            for command, family in sorted(classification.pairs):
                lua_excluded.append(
                    {
                        "pair": [command, family],
                        "reason": "single-effect Lua",
                        "script": script_name,
                    }
                )

    ordinary_direct, ordinary_dynamic = _ordinary_direct_writes(
        gateway_path, contract_map, hint_contract_map
    )
    if ordinary_dynamic:
        raise ReplayAuditError(
            "unresolved dynamic production ordinary writes remain: "
            + "; ".join(
                f"{site['function']}.{site['method']}" for site in sorted(
                    ordinary_dynamic,
                    key=lambda entry: (entry["function"], entry["method"]),
                )
            )
        )
    for command, family in sorted(ordinary_direct):
        lua_excluded.append({"pair": [command, family], "reason": "ordinary direct write"})
    for command, family in sorted(pipeline_ordinary):
        lua_excluded.append({"pair": [command, family], "reason": "non-transactional pipeline"})

    required = pipeline_qualifying | lua_qualifying

    default_line = _read_default_line(template_path)
    entrypoint_canonical = _read_entrypoint_canonical(entrypoint_path)
    if default_line != entrypoint_canonical:
        raise ReplayAuditError("template default rule and entrypoint canonical rule disagree")
    rule_model = parse_default_rule(default_line)
    rule_pairs = {
        (command, family)
        for command, families in rule_model.selectors.items()
        for family in families
    }

    missing = sorted(required - rule_pairs)
    if missing:
        raise ReplayAuditError(
            "production requires replay selectors the rule does not grant: "
            + ", ".join(f"{command} {family}" for command, family in missing)
        )
    extra = sorted(rule_pairs - required)
    if extra:
        raise ReplayAuditError(
            "replay rule grants selectors with no qualifying production "
            "justification: " + ", ".join(f"{command} {family}" for command, family in extra)
        )
    if (pipeline_qualifying or lua_qualifying) and not rule_model.has_select:
        raise ReplayAuditError("replay rule is missing the separately justified +select")
    if not (pipeline_qualifying or lua_qualifying) and rule_model.has_select:
        raise ReplayAuditError("+select has no qualifying production justification")

    return {
        "audit_version": AUDIT_VERSION,
        "adr_reference": "docs/adr/0009-disabled-default-user-write-only-aof-replay-authority.md",
        "governed_destination_contract_source": DEFAULT_GATEWAY,
        "default_rule": default_line,
        "rule_model": {
            "selectors": {
                command: sorted(families)
                for command, families in sorted(rule_model.selectors.items())
            },
            "has_select": rule_model.has_select,
        },
        "governed_contract": dict(sorted(contract_map.items())),
        "governed_hint_contract": dict(sorted(hint_contract_map.items())),
        "transactional_pipelines": sorted(
            {f"{site['function']}" for site in pipeline_evidence if site["transactional"]}
        ),
        "pipeline_writes": sorted(
            {
                f"{site['command']}:{site['family']}"
                for site in pipeline_evidence
                if site["transactional"]
            }
        ),
        "lua_scripts": {
            script_name: {
                "classification": classifications[script_name].classification,
                "path_sensitive": True,
                "worst_case_mutations": classifications[script_name].worst_case_mutations,
                "conditional_sites": classifications[script_name].conditional_sites,
                "has_mutating_loop": classifications[script_name].has_mutating_loop,
                "has_dynamic_mutation": classifications[script_name].has_dynamic_mutation,
                "invoked_in_outer_transaction": classifications[
                    script_name
                ].invoked_in_outer_transaction,
                "mutations": [
                    {
                        "command": mutation.command,
                        "key": mutation.key_expr,
                        "keys_index": mutation.keys_index,
                        "families": sorted(mutation.families),
                        "conditional": mutation.conditional,
                    }
                    for mutation in classifications[script_name].mutations
                ],
                "invocation_families": {
                    str(position): families
                    for position, families in sorted(
                        classifications[script_name].invocation_families.items()
                    )
                },
            }
            for script_name in sorted(classifications)
        },
        "qualifying_pairs": sorted(f"{command}:{family}" for command, family in required),
        "excluded_pairs": sorted(lua_excluded, key=lambda entry: (entry["reason"], entry["pair"])),
        "ordinary_dynamic_sites": sorted(
            ordinary_dynamic, key=lambda entry: (entry["function"], entry["method"])
        ),
        "select_justification": (
            "server-generated SELECT during AOF replay of qualifying "
            "transactional and multi-effect Lua writes"
        ),
        "copies_agree": {
            "template": True,
            "entrypoint_canonical": True,
        },
        "notes": (
            "Semantic proof is re-derived from production code on every run; "
            "artifact equality only detects drift. worst_case_mutations is the "
            "path-sensitive maximum: sequential statements accumulate, each if "
            "contributes its largest reachable branch (implicit zero branch "
            "without else), and early return terminates its path. mutations is "
            "the pair union over every reachable site, so a script can remain "
            "single-effect per path while justifying several command/family pairs."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    unknown = [arg for arg in args if arg != "--regenerate-projection"]
    if unknown:
        print(
            f"redis replay audit failed: unknown arguments: {' '.join(unknown)}",
            file=sys.stderr,
        )
        return 1
    regenerate = "--regenerate-projection" in args
    try:
        projection = derive_projection()
    except (OSError, SyntaxError, ReplayAuditError) as exc:
        print(f"redis replay audit failed: {exc}", file=sys.stderr)
        return 1

    if regenerate:
        DEFAULT_ARTIFACT.write_text(
            json.dumps(projection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("Regenerated replay projection artifact")
    elif DEFAULT_ARTIFACT.exists():
        try:
            previous = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"redis replay audit failed: {exc}", file=sys.stderr)
            return 1
        if previous != projection:
            print(
                "redis replay audit failed: replay artifact differs from the "
                "audited production derivation",
                file=sys.stderr,
            )
            return 1
    else:
        print(
            "redis replay audit failed: replay projection artifact is missing; "
            "regenerate it explicitly with --regenerate-projection",
            file=sys.stderr,
        )
        return 1

    print("Redis replay audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
