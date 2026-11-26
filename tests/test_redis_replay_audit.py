"""Adversarial tests for the code-grounded Redis replay completeness gate.

Each test proves the gate fails closed on one concrete attack or drift
shape. Fixture gateways, rules, and entrypoint snippets live in temporary
directories; production files are never mutated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_redis_replay import (
    ReplayAuditError,
    canonical_family,
    classify_lua_script,
    derive_projection,
    parse_default_rule,
)

CONTRACT = (
    'LOCAL_COMPOSE_CAPABILITY_QUEUE_MAP = {"general": "agent.tasks.general"}\n'
    'LOCAL_COMPOSE_CAPABILITY_HINT_MAP = {"general": "agent.hints.general"}\n'
)

SINGLE_EFFECT_LUA = """local target_key = KEYS[1]
if redis.call('EXISTS', target_key) == 0 then
    redis.call('SET', target_key, ARGV[1])
    return 'OK'
end
return 'IDEMPOTENT'
"""

BASE_RULE = (
    "user default off resetpass resetkeys resetchannels -@all +select (+set %W~task.status:*)"
)

DISPATCH_PIPELINE = """
async def dispatch(self):
    target_key = f"task.status:{task_id}"
    async with self._client.pipeline(transaction=True) as pipe:
        pipe.set(target_key, payload)
        await pipe.execute()
"""

EVAL_SINGLE = """
async def save(self):
    target_key = f"task.status:{task_id}"
    await self._client.eval(SAVE_LUA, 1, target_key, payload)
"""


def _write_tree(
    tmp_path: Path,
    gateway_src: str,
    rule: str = BASE_RULE,
    canonical: str | None = None,
    extra_sources: dict[str, str] | None = None,
) -> dict[str, Path]:
    source_root = tmp_path / "svc"
    gateway_dir = source_root / "gw"
    gateway_dir.mkdir(parents=True)
    gateway_path = gateway_dir / "gw.py"
    gateway_path.write_text(CONTRACT + gateway_src, encoding="utf-8")
    for name, src in (extra_sources or {}).items():
        extra_path = source_root / name
        extra_path.parent.mkdir(parents=True, exist_ok=True)
        extra_path.write_text(src, encoding="utf-8")
    template_path = tmp_path / "users.acl.template"
    template_path.write_text(rule + "\n", encoding="utf-8")
    entrypoint_path = tmp_path / "entrypoint.sh"
    entrypoint_path.write_text(
        f"canonical_default='{rule if canonical is None else canonical}'\n",
        encoding="utf-8",
    )
    return {
        "source_root": source_root,
        "template": template_path,
        "entrypoint": entrypoint_path,
    }


def _derive(paths: dict[str, Path]) -> dict[str, object]:
    return derive_projection(
        source_root=paths["source_root"],
        gateway_relpath="svc/gw/gw.py",
        template_path=paths["template"],
        entrypoint_path=paths["entrypoint"],
    )


def test_production_repo_projection_is_self_consistent() -> None:
    projection = derive_projection()
    assert projection["audit_version"] == 3
    assert (
        projection["adr_reference"]
        == "docs/adr/0009-disabled-default-user-write-only-aof-replay-authority.md"
    )
    assert (
        projection["governed_destination_contract_source"]
        == "services/orchestrator/src/orchestrator/infrastructure/redis_adapter.py"
    )
    pairs = {tuple(entry.split(":", 1)) for entry in projection["qualifying_pairs"]}
    rule_pairs = {
        (command, family)
        for command, families in projection["rule_model"]["selectors"].items()
        for family in families
    }
    assert pairs == rule_pairs
    assert projection["rule_model"]["has_select"] is True
    for script in projection["lua_scripts"].values():
        for mutation in script["mutations"]:
            for family in mutation["families"]:
                assert family.startswith(("task.", "agent."))


def test_missing_selector_fails(tmp_path: Path) -> None:
    paths = _write_tree(
        tmp_path,
        DISPATCH_PIPELINE,
        rule="user default off resetpass resetkeys resetchannels -@all",
    )
    with pytest.raises(ReplayAuditError, match="does not grant"):
        _derive(paths)


def test_extra_selector_fails(tmp_path: Path) -> None:
    paths = _write_tree(
        tmp_path,
        "async def idle(self):\n    pass\n",
        rule=BASE_RULE + " (+sadd %W~task.children:*)",
    )
    with pytest.raises(ReplayAuditError, match="no qualifying production"):
        _derive(paths)


def test_wrong_command_for_correct_family_fails(tmp_path: Path) -> None:
    paths = _write_tree(
        tmp_path,
        DISPATCH_PIPELINE,
        rule=(
            "user default off resetpass resetkeys resetchannels -@all +select "
            "(+sadd %W~task.status:*)"
        ),
    )
    with pytest.raises(ReplayAuditError):
        _derive(paths)


def test_wrong_family_for_correct_command_fails(tmp_path: Path) -> None:
    paths = _write_tree(
        tmp_path,
        DISPATCH_PIPELINE,
        rule=(
            "user default off resetpass resetkeys resetchannels -@all +select "
            "(+set %W~task.bogus:*)"
        ),
    )
    with pytest.raises(ReplayAuditError):
        _derive(paths)


def test_read_selector_instead_of_write_only_fails() -> None:
    with pytest.raises(ReplayAuditError, match="not write-only"):
        parse_default_rule(
            "user default off resetpass resetkeys resetchannels -@all +select "
            "(+set %R~task.status:*)"
        )


def test_no_cross_product_of_command_and_key_variables() -> None:
    lua = """local status_key = KEYS[1]
local children_key = KEYS[2]
redis.call('SET', status_key, ARGV[1])
redis.call('SADD', children_key, ARGV[2])
return 'OK'
"""
    classification = classify_lua_script(
        "CROSS",
        lua,
        invocation_families={1: ["task.status:*"], 2: ["task.children:*"]},
    )
    assert classification.classification == "multi-effect"
    assert classification.pairs == {
        ("SET", "task.status:*"),
        ("SADD", "task.children:*"),
    }


def test_symbolic_variable_families_are_rejected() -> None:
    with pytest.raises(ReplayAuditError, match="no production invocation family"):
        classify_lua_script(
            "UNBOUND",
            "redis.call('SET', KEYS[2], ARGV[1])\nreturn 'OK'\n",
            invocation_families={1: ["task.status:*"]},
        )


def test_dynamic_xadd_unpack_is_captured() -> None:
    transition = """local claim_key = KEYS[2]
local status_key = KEYS[1]
local events_key = KEYS[7]
redis.call('DEL', claim_key, ARGV[1])
redis.call('SET', status_key, ARGV[2])
if ARGV[3] ~= '' then
    local args = {'XADD', events_key, 'MAXLEN', '~', '500', '*'}
    for k, v in pairs(ARGV) do
        table.insert(args, tostring(k))
    end
    redis.call(unpack(args))
end
return 'OK'
"""
    classification = classify_lua_script(
        "TRANSITION",
        transition,
        invocation_families={
            1: ["task.status:*"],
            2: ["task.claim:*"],
            7: ["agent.events"],
        },
    )
    assert classification.classification == "multi-effect"
    assert ("XADD", "agent.events") in classification.pairs
    assert ("DEL", "task.claim:*") in classification.pairs
    assert ("SET", "task.status:*") in classification.pairs


def test_single_effect_with_branches_and_read_only_loop() -> None:
    lua = """local target_key = KEYS[1]
local fence_key = KEYS[2]
if redis.call('EXISTS', target_key) == 0 then
    if redis.call('EXISTS', fence_key) == 0 then
        return 'FENCE_NOT_FOUND'
    end
    for _, k in ipairs({target_key, fence_key}) do
        local t = redis.call('TYPE', k)['ok']
        if t ~= 'none' and t ~= 'string' then
            return 'WRONG_TYPE'
        end
    end
    redis.call('SET', target_key, ARGV[1])
    return 'OK'
else
    local existing = redis.call('GET', target_key)
    if existing == ARGV[1] then
        return 'IDEMPOTENT'
    end
    return 'CONFLICT'
end
"""
    classification = classify_lua_script(
        "SINGLE", lua, invocation_families={1: ["task.input:*"], 2: ["task.fence:*"]}
    )
    assert classification.classification == "single-effect"
    assert classification.worst_case_mutations == 1
    assert classification.has_mutating_loop is False
    assert classification.pairs == {("SET", "task.input:*")}


def test_mutually_exclusive_branches_count_max_not_sum() -> None:
    lua = """local first_key = KEYS[1]
local second_key = KEYS[2]
if ARGV[1] == 'a' then
    redis.call('SET', first_key, ARGV[2])
else
    redis.call('SET', second_key, ARGV[2])
end
return 'OK'
"""
    classification = classify_lua_script(
        "BRANCHY",
        lua,
        invocation_families={1: ["task.status:*"], 2: ["task.fence:*"]},
    )
    assert classification.classification == "single-effect"
    assert classification.worst_case_mutations == 1
    assert classification.pairs == {
        ("SET", "task.status:*"),
        ("SET", "task.fence:*"),
    }


def test_branch_with_two_sequential_writes_is_multi_effect() -> None:
    lua = """local first_key = KEYS[1]
local second_key = KEYS[2]
if ARGV[1] == 'a' then
    redis.call('SET', first_key, ARGV[2])
    redis.call('SET', second_key, ARGV[3])
else
    return 'SKIP'
end
return 'OK'
"""
    classification = classify_lua_script(
        "TWO_IN_BRANCH",
        lua,
        invocation_families={1: ["task.status:*"], 2: ["task.fence:*"]},
    )
    assert classification.classification == "multi-effect"
    assert classification.worst_case_mutations == 2


def test_nested_branch_maximum() -> None:
    lua = """local first_key = KEYS[1]
local second_key = KEYS[2]
if ARGV[1] == 'a' then
    if ARGV[2] == 'b' then
        redis.call('SET', first_key, ARGV[3])
        redis.call('SET', second_key, ARGV[4])
    end
else
    redis.call('SET', first_key, ARGV[3])
end
return 'OK'
"""
    classification = classify_lua_script(
        "NESTED",
        lua,
        invocation_families={1: ["task.status:*"], 2: ["task.fence:*"]},
    )
    assert classification.classification == "multi-effect"
    assert classification.worst_case_mutations == 2


def test_nested_single_writes_stay_single_effect() -> None:
    lua = """local first_key = KEYS[1]
local second_key = KEYS[2]
if ARGV[1] == 'a' then
    if ARGV[2] == 'b' then
        redis.call('SET', first_key, ARGV[3])
    end
else
    redis.call('SET', second_key, ARGV[3])
end
return 'OK'
"""
    classification = classify_lua_script(
        "NESTED_SINGLE",
        lua,
        invocation_families={1: ["task.status:*"], 2: ["task.fence:*"]},
    )
    assert classification.classification == "single-effect"
    assert classification.worst_case_mutations == 1
    assert classification.pairs == {
        ("SET", "task.status:*"),
        ("SET", "task.fence:*"),
    }


def test_no_else_means_implicit_zero_branch() -> None:
    lua = """local target_key = KEYS[1]
if ARGV[1] == 'a' then
    redis.call('SET', target_key, ARGV[2])
end
return 'OK'
"""
    classification = classify_lua_script("NO_ELSE", lua, invocation_families={1: ["task.status:*"]})
    assert classification.classification == "single-effect"
    assert classification.worst_case_mutations == 1


def test_early_return_before_later_write() -> None:
    lua = """local target_key = KEYS[1]
redis.call('SET', target_key, ARGV[1])
return 'OK'
redis.call('SET', target_key, ARGV[2])
"""
    classification = classify_lua_script(
        "EARLY_RETURN", lua, invocation_families={1: ["task.status:*"]}
    )
    assert classification.classification == "single-effect"
    assert classification.worst_case_mutations == 1


def test_early_return_within_one_branch() -> None:
    lua = """local first_key = KEYS[1]
local second_key = KEYS[2]
if ARGV[1] == 'a' then
    redis.call('SET', first_key, ARGV[2])
    return 'ONE'
else
    redis.call('SET', second_key, ARGV[2])
end
return 'OK'
"""
    classification = classify_lua_script(
        "BRANCH_RETURN",
        lua,
        invocation_families={1: ["task.status:*"], 2: ["task.fence:*"]},
    )
    assert classification.classification == "single-effect"
    assert classification.worst_case_mutations == 1


def test_sequential_writes_outside_branches_accumulate() -> None:
    lua = """local first_key = KEYS[1]
local second_key = KEYS[2]
redis.call('SET', first_key, ARGV[1])
redis.call('SET', second_key, ARGV[2])
return 'OK'
"""
    classification = classify_lua_script(
        "SEQUENTIAL",
        lua,
        invocation_families={1: ["task.status:*"], 2: ["task.fence:*"]},
    )
    assert classification.classification == "multi-effect"
    assert classification.worst_case_mutations == 2


def test_read_only_nested_loops_do_not_count() -> None:
    lua = """local target_key = KEYS[1]
local fence_key = KEYS[2]
for _, outer in ipairs({target_key}) do
    for _, k in ipairs({outer, fence_key}) do
        if redis.call('TYPE', k)['ok'] ~= 'string' then
            return 'WRONG_TYPE'
        end
    end
end
redis.call('SET', target_key, ARGV[1])
return 'OK'
"""
    classification = classify_lua_script(
        "NESTED_READ_LOOP",
        lua,
        invocation_families={1: ["task.status:*"], 2: ["task.fence:*"]},
    )
    assert classification.classification == "single-effect"
    assert classification.worst_case_mutations == 1
    assert classification.has_mutating_loop is False


def test_goto_fails_closed() -> None:
    with pytest.raises(ReplayAuditError, match="cannot classify"):
        classify_lua_script(
            "GOTO",
            "redis.call('SET', KEYS[1], ARGV[1])\n"
            "goto done\n"
            "redis.call('SET', KEYS[1], ARGV[2])\n"
            "::done::\nreturn 'OK'\n",
            invocation_families={1: ["task.status:*"]},
        )


def test_repeat_loop_fails_closed() -> None:
    with pytest.raises(ReplayAuditError, match="repeat"):
        classify_lua_script(
            "REPEAT",
            "local i = 0\nrepeat\nredis.call('SET', KEYS[1], ARGV[1])\ni = i + 1\nuntil i > 1\nreturn 'OK'\n",
            invocation_families={1: ["task.status:*"]},
        )


def test_redis_pcall_dynamic_fails_closed() -> None:
    with pytest.raises(ReplayAuditError, match="cannot bound"):
        classify_lua_script(
            "PCALL_DYNAMIC",
            "redis.pcall(ARGV[1], KEYS[1], ARGV[2])\nreturn 'OK'\n",
            invocation_families={1: ["task.status:*"]},
        )


def test_function_body_mutation_fails_closed() -> None:
    lua = """local target_key = KEYS[1]
local function commit(value)
    redis.call('SET', target_key, value)
end
commit(ARGV[1])
return 'OK'
"""
    with pytest.raises(ReplayAuditError, match="function body"):
        classify_lua_script("FUNC_MUT", lua, invocation_families={1: ["task.status:*"]})


def test_pair_union_covers_every_branch_despite_single_path() -> None:
    lua = """local first_key = KEYS[1]
local second_key = KEYS[2]
local third_key = KEYS[3]
if ARGV[1] == 'a' then
    redis.call('SADD', first_key, ARGV[2])
elseif ARGV[1] == 'b' then
    redis.call('DEL', second_key)
else
    redis.call('SET', third_key, ARGV[2])
end
return 'OK'
"""
    classification = classify_lua_script(
        "UNION",
        lua,
        invocation_families={
            1: ["task.children:*"],
            2: ["task.claim:*"],
            3: ["task.status:*"],
        },
    )
    assert classification.classification == "single-effect"
    assert classification.worst_case_mutations == 1
    assert classification.pairs == {
        ("SADD", "task.children:*"),
        ("DEL", "task.claim:*"),
        ("SET", "task.status:*"),
    }


def test_mutating_loop_is_multi_effect() -> None:
    lua = """local index_key = KEYS[1]
for i, member in ipairs(ARGV) do
    redis.call('SADD', index_key, member)
end
return 'OK'
"""
    classification = classify_lua_script("LOOPY", lua, invocation_families={1: ["task.children:*"]})
    assert classification.classification == "multi-effect"
    assert classification.has_mutating_loop is True


def test_dynamic_command_fails_closed() -> None:
    with pytest.raises(ReplayAuditError, match="cannot bound"):
        classify_lua_script(
            "DYNAMIC",
            "redis.call(ARGV[1], KEYS[1], ARGV[2])\nreturn 'OK'\n",
            invocation_families={1: ["task.status:*"]},
        )


def test_lua_in_outer_transaction_is_multi_effect() -> None:
    classification = classify_lua_script(
        "OUTER",
        SINGLE_EFFECT_LUA,
        invocation_families={1: ["task.status:*"]},
        invoked_in_outer_transaction=True,
    )
    assert classification.classification == "multi-effect"
    assert classification.invoked_in_outer_transaction is True


def test_bare_pipeline_defaults_to_transactional(tmp_path: Path) -> None:
    gateway = """
async def dispatch(self):
    target_key = f"task.status:{task_id}"
    async with self._client.pipeline() as pipe:
        pipe.set(target_key, payload)
        await pipe.execute()
"""
    paths = _write_tree(tmp_path, gateway)
    projection = _derive(paths)
    assert "SET:task.status:*" in projection["qualifying_pairs"]
    assert "SET:task.status:*" in projection["pipeline_writes"]


def test_transaction_false_pipeline_is_ordinary(tmp_path: Path) -> None:
    gateway = """
async def status(self):
    target_key = f"task.status:{task_id}"
    async with self._client.pipeline(transaction=False) as pipe:
        pipe.set(target_key, payload)
        await pipe.execute()
"""
    paths = _write_tree(
        tmp_path,
        gateway,
        rule="user default off resetpass resetkeys resetchannels -@all",
    )
    projection = _derive(paths)
    assert projection["qualifying_pairs"] == []
    assert projection["excluded_pairs"] == [
        {"pair": ["SET", "task.status:*"], "reason": "non-transactional pipeline"}
    ]


def test_production_site_outside_gateway_fails(tmp_path: Path) -> None:
    paths = _write_tree(
        tmp_path,
        "async def idle(self):\n    pass\n",
        rule="user default off resetpass resetkeys resetchannels -@all",
        extra_sources={
            "other.py": (
                "async def rogue(self):\n"
                "    async with self._client.pipeline() as pipe:\n"
                "        pipe.set('task.status:x', 'y')\n"
                "        await pipe.execute()\n"
            )
        },
    )
    with pytest.raises(ReplayAuditError, match="completeness gate"):
        _derive(paths)


def test_lua_execution_outside_gateway_fails(tmp_path: Path) -> None:
    paths = _write_tree(
        tmp_path,
        "async def idle(self):\n    pass\n",
        rule="user default off resetpass resetkeys resetchannels -@all",
        extra_sources={"other.py": "async def rogue(self):\n    await c.eval('return 1', 0)\n"},
    )
    with pytest.raises(ReplayAuditError, match="Lua"):
        _derive(paths)


def test_routing_selector_reintroduced_fails() -> None:
    with pytest.raises(ReplayAuditError, match="routing"):
        parse_default_rule(BASE_RULE + " (+set %W~task.routing:*)")


def test_template_entrypoint_mismatch_fails(tmp_path: Path) -> None:
    paths = _write_tree(
        tmp_path,
        DISPATCH_PIPELINE,
        canonical="user default off resetpass resetkeys resetchannels -@all",
    )
    with pytest.raises(ReplayAuditError, match="disagree"):
        _derive(paths)


def test_artifact_drift_is_detected(tmp_path: Path) -> None:
    import json

    paths = _write_tree(tmp_path, DISPATCH_PIPELINE)
    first = _derive(paths)
    assert "SET:task.status:*" in first["qualifying_pairs"]
    artifact_path = tmp_path / "replay_projection.json"
    artifact_path.write_text(json.dumps(first, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Unchanged inputs re-derive byte-identically: no drift.
    assert _derive(paths) == json.loads(artifact_path.read_text(encoding="utf-8"))
    # Any rule change breaks the derivation, so a stale artifact can never
    # pass as current evidence.
    drifted_rule = (
        "user default off resetpass resetkeys resetchannels -@all +select "
        "(+set %W~task.status:*) (+sadd %W~task.children:*)"
    )
    paths["template"].write_text(drifted_rule + "\n", encoding="utf-8")
    paths["entrypoint"].write_text(f"canonical_default='{drifted_rule}'\n", encoding="utf-8")
    with pytest.raises(ReplayAuditError, match="no qualifying production"):
        _derive(paths)


@pytest.mark.parametrize(
    "token",
    [
        " +@all",
        " +@write",
        " +@connection",
        " +@scripting",
        " +@admin",
        " +@dangerous",
        " +multi",
        " +exec",
        " +eval",
        " user default on",
        " nopass",
        " ~*",
    ],
)
def test_forbidden_authority_fails(token: str) -> None:
    with pytest.raises(ReplayAuditError):
        parse_default_rule(BASE_RULE + token)


def test_del_bounded_to_claim_only() -> None:
    with pytest.raises(ReplayAuditError, match="bounded"):
        parse_default_rule(BASE_RULE + " (+del %W~task.status:*)")


def test_retained_selector_without_qualifying_justification_fails(
    tmp_path: Path,
) -> None:
    gateway = "SAVE_LUA = '''" + SINGLE_EFFECT_LUA + "'''\n" + EVAL_SINGLE
    paths = _write_tree(tmp_path, gateway)
    with pytest.raises(ReplayAuditError, match="no qualifying production"):
        _derive(paths)


def test_canonical_family_rejects_symbolic_and_foreign_keys() -> None:
    assert canonical_family("task.status:abc") == "task.status:*"
    assert canonical_family("task.artifact:a:b:c") == "task.artifact:*"
    assert canonical_family("agent.tasks.general") == "agent.tasks.general"
    with pytest.raises(ReplayAuditError):
        canonical_family("status_key")
    with pytest.raises(ReplayAuditError):
        canonical_family("other.status:x")
    with pytest.raises(ReplayAuditError):
        canonical_family("agent.tasks.{*}")


@pytest.mark.parametrize(
    "token",
    [
        " (+MULTI %W~task.status:*)",
        " (+MuLtI %W~task.status:*)",
        " (+EXEC %W~task.status:*)",
        " (+ExEc %W~task.status:*)",
        " (+EVAL %W~task.status:*)",
        " (+eVaL %W~task.status:*)",
        " (+UNLINK %W~task.status:*)",
        " (+uNlInK %W~task.status:*)",
        " +MULTI",
        " +MuLtI",
        " +EXEC",
        " +EVAL",
        " +UNLINK",
        " +@ALL",
        " +@All",
        " +@WRITE",
        " +@Write",
        " +@CONNECTION",
        " +@Connection",
        " +@SCRIPTING",
        " +@ADMIN",
        " +@DANGEROUS",
        " +@Dangerous",
    ],
)
def test_uppercase_and_mixed_case_forbidden_authority_fails(token: str) -> None:
    with pytest.raises(ReplayAuditError):
        parse_default_rule(BASE_RULE + token)


def test_uppercase_default_user_shape_fails() -> None:
    with pytest.raises(ReplayAuditError):
        parse_default_rule(
            "USER DEFAULT ON RESETPASS RESETKEYS RESETCHANNELS -@all +select "
            "(+set %W~task.status:*)"
        )


@pytest.mark.parametrize("token", [" NOPASS", " Nopass", " NoPass"])
def test_mixed_case_nopass_fails(token: str) -> None:
    with pytest.raises(ReplayAuditError):
        parse_default_rule(BASE_RULE + token)


@pytest.mark.parametrize(
    "group",
    [
        "(+SET %W~task.status:*)",
        "(+Set %W~task.status:*)",
        "(+DEL %W~task.claim:*)",
        "(+Del %W~task.claim:*)",
        "(+SADD %W~task.children:*)",
        "(+Sadd %W~task.children:*)",
        "(+RPUSH %W~agent.tasks.general)",
        "(+Rpush %W~agent.tasks.general)",
        "(+XADD %W~agent.events)",
        "(+Xadd %W~agent.events)",
    ],
)
def test_supported_group_case_variants_parse_identically(group: str) -> None:
    opener, _, rest = group.partition(" ")
    assert opener.startswith("(+")
    lowered = parse_default_rule(BASE_RULE + " " + opener.lower() + " " + rest)
    folded = parse_default_rule(BASE_RULE + " " + group)
    assert folded.selectors == lowered.selectors


@pytest.mark.parametrize(
    "token",
    [
        " (+set %W~TASK.ROUTING:*)",
        " (+SET %W~task.routing:*)",
        " (+Set %W~Task.Routing:*)",
    ],
)
def test_task_routing_case_variants_fail(token: str) -> None:
    with pytest.raises(ReplayAuditError, match="task.routing"):
        parse_default_rule(BASE_RULE + token)


@pytest.mark.parametrize(
    "token",
    [
        " (+blpop %W~agent.tasks.general)",
        " (+BLPOP %W~agent.tasks.general)",
        " (+hset %W~agent.state:worker-1)",
        " (+HSET %W~agent.state:worker-1)",
        " (+get %W~task.status:*)",
        " (+GET %W~task.status:*)",
        " (+multi %W~task.status:*)",
        " (+MULTI %W~task.status:*)",
    ],
)
def test_unknown_group_command_fails(token: str) -> None:
    with pytest.raises(ReplayAuditError, match="unknown or unsupported"):
        parse_default_rule(BASE_RULE + token)


@pytest.mark.parametrize(
    "token",
    [
        " (+set %R~task.status:*)",
        " (+SET %R~task.status:*)",
        " (+set ~task.status:*)",
        " (+SET ~task.status:*)",
        " (+set %W~task.status:*",
        " (set %W~task.status:*)",
        " (+set %W~task.status:*) trailing-junk",
        " (+set %W~task.status:*) +ping",
        " +PING",
        " stray-token",
    ],
)
def test_malformed_unsupported_and_stray_tokens_fail(token: str) -> None:
    with pytest.raises(ReplayAuditError):
        parse_default_rule(BASE_RULE + token)


def test_structural_capitalization_is_semantically_equivalent() -> None:
    folded = parse_default_rule(
        "USER DEFAULT OFF RESETPASS RESETKEYS RESETCHANNELS -@ALL +SELECT "
        "(+SET %W~task.status:*)"
    )
    assert folded.selectors == {"SET": {"task.status:*"}}
    assert folded.has_select is True


def test_unresolved_dynamic_ordinary_write_fails_the_gate(tmp_path: Path) -> None:
    gateway_src = (
        CONTRACT
        + "async def dispatch_hint(self):\n"
        + "    await self._client.rpush(unresolved_target, payload)\n"
    )
    paths = _write_tree(tmp_path, gateway_src)
    with pytest.raises(ReplayAuditError, match="unresolved dynamic production ordinary writes"):
        _derive(paths)


def test_governed_dispatch_hint_records_exact_ordinary_pairs(tmp_path: Path) -> None:
    gateway_src = (
        CONTRACT
        + DISPATCH_PIPELINE
        + "async def dispatch_hint(self, hint, target=None, capability=None):\n"
        + "    approved_target = self._capability_hint_contract[capability]\n"
        + "    if target is None:\n"
        + "        resolved_target = approved_target\n"
        + "    else:\n"
        + "        if target != approved_target:\n"
        + "            raise ValueError('Hint target does not match authoritative capability')\n"
        + "        resolved_target = approved_target\n"
        + "    await self._client.rpush(resolved_target, payload)\n"
    )
    paths = _write_tree(tmp_path, gateway_src)
    projection = _derive(paths)
    assert projection["ordinary_dynamic_sites"] == []
    assert ["RPUSH", "agent.hints.general"] in [
        entry["pair"]
        for entry in projection["excluded_pairs"]
        if entry["reason"] == "ordinary direct write"
    ]


def test_missing_projection_artifact_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.audit_redis_replay as audit_module

    monkeypatch.setattr(
        audit_module, "DEFAULT_ARTIFACT", tmp_path / "replay_projection.json"
    )
    assert audit_module.main([]) == 1


def test_explicit_regeneration_flag_rewrites_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import scripts.audit_redis_replay as audit_module

    monkeypatch.setattr(
        audit_module, "DEFAULT_ARTIFACT", tmp_path / "replay_projection.json"
    )
    assert audit_module.main(["--regenerate-projection"]) == 0
    regenerated = json.loads((tmp_path / "replay_projection.json").read_text(encoding="utf-8"))
    assert regenerated["audit_version"] == 3
    assert audit_module.main([]) == 0


def test_pipeline_outside_async_with_fails_closed(tmp_path: Path) -> None:
    gateway_src = (
        CONTRACT
        + "async def legacy(self):\n"
        + "    pipe = self._client.pipeline(transaction=True)\n"
        + "    pipe.set(target_key, payload)\n"
        + "    await pipe.execute()\n"
    )
    paths = _write_tree(tmp_path, gateway_src)
    with pytest.raises(ReplayAuditError, match="outside 'async with"):
        _derive(paths)


@pytest.mark.parametrize("command", ["MULTI", "multi", "eXeC", "EXEC"])
def test_raw_framing_execute_command_fails_closed(
    tmp_path: Path, command: str
) -> None:
    gateway_src = (
        CONTRACT
        + "async def raw(self):\n"
        + f'    await self._client.execute_command("{command}")\n'
    )
    paths = _write_tree(tmp_path, gateway_src)
    with pytest.raises(ReplayAuditError, match="execute_command"):
        _derive(paths)


def test_dynamic_execute_command_fails_closed(tmp_path: Path) -> None:
    gateway_src = (
        CONTRACT
        + "async def raw(self, command):\n"
        + "    await self._client.execute_command(command)\n"
    )
    paths = _write_tree(tmp_path, gateway_src)
    with pytest.raises(ReplayAuditError, match="execute_command"):
        _derive(paths)


def test_static_non_framing_execute_command_has_no_audit_model(
    tmp_path: Path,
) -> None:
    gateway_src = (
        CONTRACT
        + "async def raw(self):\n"
        + '    await self._client.execute_command("SET", target_key, payload)\n'
    )
    paths = _write_tree(tmp_path, gateway_src)
    with pytest.raises(ReplayAuditError, match="execute_command"):
        _derive(paths)


def test_break_outside_loop_fails_closed() -> None:
    lua = """local target_key = KEYS[1]
if redis.call('EXISTS', target_key) == 0 then
    redis.call('SET', target_key, ARGV[1])
    break
end
return 'OK'
"""
    with pytest.raises(ReplayAuditError, match="break outside a loop"):
        classify_lua_script(
            "BREAKY", lua, invocation_families={1: ["task.status:*"]}
        )


def test_break_inside_mutating_loop_stays_multi_effect() -> None:
    lua = """local target_key = KEYS[1]
for i, key in ipairs(ARGV) do
    redis.call('SET', target_key, key)
    break
end
return 'OK'
"""
    classification = classify_lua_script(
        "BREAK_LOOP", lua, invocation_families={1: ["task.status:*"]}
    )
    assert classification.classification == "multi-effect"
    assert classification.has_mutating_loop is True


def test_block_comment_hiding_fake_mutation_is_ignored() -> None:
    lua = """local target_key = KEYS[1]
--[[ redis.call('SADD', target_key, ARGV[1]) ]]
if redis.call('EXISTS', target_key) == 0 then
    redis.call('SET', target_key, ARGV[1])
    return 'OK'
end
return 'IDEMPOTENT'
"""
    classification = classify_lua_script(
        "BLOCK_COMMENT", lua, invocation_families={1: ["task.input:*"]}
    )
    assert classification.classification == "single-effect"
    assert classification.pairs == {("SET", "task.input:*")}


def test_levelled_block_comment_is_stripped() -> None:
    lua = """local target_key = KEYS[1]
-- hard comment --[==[ not a comment ]==] still comment
--[==[ redis.call('DEL', target_key) ]==]
redis.call('SET', target_key, ARGV[1])
return 'OK'
"""
    classification = classify_lua_script(
        "LEVELLED", lua, invocation_families={1: ["task.input:*"]}
    )
    assert classification.pairs == {("SET", "task.input:*")}


def test_unterminated_block_comment_fails_closed() -> None:
    lua = """local target_key = KEYS[1]
--[[ redis.call('SET', target_key, ARGV[1])
return 'OK'
"""
    with pytest.raises(ReplayAuditError, match="unterminated long-bracket"):
        classify_lua_script(
            "UNTERMINATED", lua, invocation_families={1: ["task.input:*"]}
        )


def test_long_bracket_string_fails_closed() -> None:
    lua = """local target_key = KEYS[1]
local label = [[report]]
redis.call('SET', target_key, label)
return 'OK'
"""
    with pytest.raises(ReplayAuditError, match="long-bracket string"):
        classify_lua_script(
            "LONG_STRING", lua, invocation_families={1: ["task.input:*"]}
        )
