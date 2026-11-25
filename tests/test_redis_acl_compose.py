"""Static Compose and ACL policy checks for the local Redis boundary."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "compose.yaml"
ACL_TEMPLATE = REPO_ROOT / "infrastructure" / "redis" / "users.acl.template"
REDIS_ENTRYPOINT = REPO_ROOT / "infrastructure" / "redis" / "redis-acl-entrypoint.sh"
HEALTHCHECK = REPO_ROOT / "infrastructure" / "redis" / "redis-healthcheck.sh"

ACL_IDENTITIES = (
    "orchestrator",
    "worker-1",
    "research-agent-1",
    "research-agent-2",
    "analysis-agent-1",
    "healthcheck",
)
EXPECTED_USERS = {
    "orchestrator": "redis_orchestrator",
    "worker-1": "redis_worker_1",
    "research-agent-1": "redis_research_agent_1",
    "research-agent-2": "redis_research_agent_2",
    "analysis-agent-1": "redis_analysis_agent_1",
}
ORCHESTRATOR_TASK_PATTERNS = {
    "~task.registration:*",
    "~task.status:*",
    "~task.lineage:*",
    "~task.input:*",
    "~task.authority:*",
    "~task.authority_grant:*",
    "~task.required_capability:*",
    "~task.children:*",
    "~task.artifact:*",
    "~task.artifacts:*",
    "~task.continuation:*",
    "~task.continuation_latest:*",
    "~task.claim:*",
    "~task.claim_deadline:*",
    "~task.claim_expiry:*",
    "~task.fence:*",
    "~task.delegation_proposal:*",
    "~task.consumed_waves:*",
    "~task.consumed_grants:*",
    "~task.delegation_decisions:*",
    "~task.assignment_yield:*",
    "~task.completion:*",
    "~task.outcome:*",
    "~task.result:*",
    "~task.assignment_completion:*",
    "~task.child_completion:*",
    "~task.recovery_decision:*",
    "~task.routing:*",
}
ORCHESTRATOR_COMMANDS = {
    "+eval",
    "+get",
    "+set",
    "+exists",
    "+del",
    "+sadd",
    "+smembers",
    "+scard",
    "+sismember",
    "+rpush",
    "+type",
    "+xadd",
    "+xrevrange",
    "+hgetall",
    "+multi",
    "+exec",
    "+discard",
}


def _service_block(compose: str, service: str) -> str:
    headers = list(re.finditer(r"^  ([a-z0-9-]+):\n", compose, re.MULTILINE))
    for index, match in enumerate(headers):
        if match.group(1) == service:
            end = headers[index + 1].start() if index + 1 < len(headers) else len(compose)
            return compose[match.start() : end]
    raise AssertionError(f"Compose service not found: {service}")


def _service_secret_sources(block: str) -> set[str]:
    return set(re.findall(r"^\s+- source: ([a-z0-9_-]+)$", block, re.MULTILINE))


def _environment_value(block: str, name: str) -> str:
    match = re.search(
        rf"^      {re.escape(name)}: ([^\n]+)$",
        block,
        re.MULTILINE,
    )
    assert match is not None, f"Missing Compose environment value: {name}"
    return match.group(1).strip()


def _acl_user_lines() -> dict[str, str]:
    lines = ACL_TEMPLATE.read_text().splitlines()
    return {
        match.group(1): line
        for line in lines
        if (match := re.match(r"user ([^ ]+) ", line)) is not None
    }


def test_compose_pins_redis_to_an_explicit_supported_patch() -> None:
    compose = COMPOSE_FILE.read_text()
    redis_block = _service_block(compose, "redis")
    match = re.search(
        r"^    image: redis:(\d+)\.(\d+)\.(\d+)(?:-alpine)?$",
        redis_block,
        re.MULTILINE,
    )

    assert match is not None
    major, minor = int(match.group(1)), int(match.group(2))
    assert major > 7 or (major == 7 and minor >= 2)


def test_acl_template_is_default_deny_and_contains_only_reviewed_identities() -> None:
    template = ACL_TEMPLATE.read_text()
    lines = template.splitlines()
    users = _acl_user_lines()

    assert lines.count("user default off resetpass resetkeys resetchannels -@all") == 1
    assert sum(line.startswith("user default ") for line in lines) == 1
    assert set(users) == {"default", *EXPECTED_USERS, "healthcheck"}
    assert all(
        "resetkeys resetchannels -@all" in line
        for identity, line in users.items()
        if identity != "default"
    )
    assert not re.search(r"(^|\s)>[^\s]+", template)
    assert not re.search(r"#[0-9a-f]{64}", template)
    assert not re.search(r"^user\s+(test|admin)\b", template, re.MULTILINE)

    for identity in ACL_IDENTITIES:
        marker = f"@@{identity.upper().replace('-', '_')}_SHA256@@"
        assert template.count(marker) == 1


def test_runtime_acl_selectors_and_orchestrator_policy_match_audit() -> None:
    users = _acl_user_lines()
    forbidden = (
        "~*",
        "+@all",
        "+@connection",
        "+@scripting",
        "+@admin",
        "+@dangerous",
        "task.*",
    )
    assert not any(token in ACL_TEMPLATE.read_text() for token in forbidden)

    for identity, (queue, hint_queue) in {
        "worker-1": ("agent.tasks.general", "agent.hints.general"),
        "research-agent-1": ("agent.tasks.research", "agent.hints.research"),
        "research-agent-2": ("agent.tasks.research", "agent.hints.research"),
        "analysis-agent-1": ("agent.tasks.analysis", "agent.hints.analysis"),
    }.items():
        line = users[identity]
        assert f"(+blpop ~{queue} ~{hint_queue})" in line
        assert f"(+hset ~agent.state:{identity})" in line
        assert "(+sadd ~agent.ids)" in line
        assert "(+xadd ~agent.events)" in line
        assert "~task." not in line
        assert line.count("(+") == 4

    orchestrator = users["orchestrator"]
    assert ORCHESTRATOR_COMMANDS <= set(orchestrator.split())
    assert ORCHESTRATOR_TASK_PATTERNS <= set(orchestrator.split())
    assert "%R~agent.ids" in orchestrator
    assert "%R~agent.state:*" in orchestrator
    assert "(+rpush ~agent.tasks.general ~agent.tasks.research ~agent.tasks.analysis)" in orchestrator
    healthcheck = users["healthcheck"]
    assert healthcheck.endswith("resetkeys resetchannels -@all +ping")
    assert "~" not in healthcheck and "(+" not in healthcheck


def test_compose_isolates_secrets_and_keeps_urls_credential_free() -> None:
    compose = COMPOSE_FILE.read_text()
    services = {
        "orchestrator": "orchestrator",
        "worker": "worker-1",
        "research-agent": "research-agent-1",
        "research-agent-2": "research-agent-2",
        "analysis-agent": "analysis-agent-1",
    }
    for service, identity in services.items():
        block = _service_block(compose, service)
        assert _environment_value(block, "REDIS_URL") == "redis://redis:6379/0"
        assert _environment_value(block, "REDIS_AUTH_MODE") == "required"
        assert _environment_value(block, "REDIS_USERNAME") == identity
        assert _environment_value(block, "REDIS_PASSWORD_FILE") == "/run/redis-auth/password"
        assert _service_secret_sources(block) == {EXPECTED_USERS[identity]}

    redis_block = _service_block(compose, "redis")
    assert _service_secret_sources(redis_block) == {
        *EXPECTED_USERS.values(),
        "redis_healthcheck",
    }
    assert "redis://redis:6379/0" in compose
    assert not re.search(r"redis://[^/\s]*@", compose)
    assert "REDIS_URL:" not in redis_block
    assert ".secrets/redis/*" in (REPO_ROOT / ".gitignore").read_text()
    assert _environment_value(_service_block(compose, "worker"), "HINT_QUEUE") == (
        "agent.hints.general"
    )

    top_level_secrets = compose.split("\nsecrets:\n", maxsplit=1)[1]
    source_files = dict(
        re.findall(r"^  ([a-z0-9_-]+):\n    file: ([^\n]+)$", top_level_secrets, re.MULTILINE)
    )
    assert source_files == {
        "redis_orchestrator": ".secrets/redis/orchestrator",
        "redis_worker_1": ".secrets/redis/worker-1",
        "redis_research_agent_1": ".secrets/redis/research-agent-1",
        "redis_research_agent_2": ".secrets/redis/research-agent-2",
        "redis_analysis_agent_1": ".secrets/redis/analysis-agent-1",
        "redis_healthcheck": ".secrets/redis/healthcheck",
    }


def test_compose_keeps_tmpfs_acl_separate_from_persistent_aof_and_waits_for_health() -> None:
    compose = COMPOSE_FILE.read_text()
    redis_block = _service_block(compose, "redis")
    assert "/run/redis:rw,noexec,nosuid,nodev,size=16m,mode=0750" in redis_block
    assert "redis_data:/data" in redis_block
    assert "condition: service_healthy" in compose
    assert '"/usr/local/bin/redis-healthcheck.sh"' in redis_block
    assert (
        "./infrastructure/redis/redis-mountinfo.sh:"
        "/usr/local/bin/redis-mountinfo.sh:ro"
    ) in redis_block
    assert "redis_data:" in compose

    entrypoint = REDIS_ENTRYPOINT.read_text()
    assert 'runtime_dir=${REDIS_ACL_RUNTIME_DIR:-/run/redis}' in entrypoint
    assert "redis_mountinfo_is_tmpfs /proc/self/mountinfo \"$runtime_dir\"" in entrypoint
    assert 'chown root:redis "$acl_temporary"' in entrypoint
    assert 'chmod 0440 "$acl_temporary"' in entrypoint
    assert 'mv -f "$acl_temporary" "$runtime_dir/users.acl"' in entrypoint
    assert "chown -R redis:redis /data" not in entrypoint
    assert "sed " not in entrypoint
    assert "/data/users.acl" not in entrypoint


def test_healthcheck_is_ping_only_and_uses_scoped_secret_environment() -> None:
    source = HEALTHCHECK.read_text()
    healthcheck_acl = _acl_user_lines()["healthcheck"]

    assert healthcheck_acl.endswith("resetkeys resetchannels -@all +ping")
    assert 'REDISCLI_AUTH="$password" REDISCLI_HISTFILE=/dev/null' in source
    assert "unset REDISCLI_AUTH password" in source
    assert "redis-cli --raw --user healthcheck ping" in source
    assert "cmp -s -" in source
    assert "-a" not in source
    assert "redis://" not in source


def test_live_acl_harness_is_isolated_and_uses_real_template_without_ports_or_host_secrets() -> None:
    compose = COMPOSE_FILE.read_text()
    template = ACL_TEMPLATE.read_text()
    harness = (REPO_ROOT / "scripts" / "redis_acl_live_harness.py").read_text()
    live_suite = (REPO_ROOT / "tests" / "test_redis_acl_live.py").read_text()

    # Normal deployment artifacts contain zero test-admin identity or secret references
    assert "fixture-admin" not in compose
    assert "fixture-admin" not in template
    assert not re.search(r"redis_(?:test|admin)", compose)

    # Live harness pins exact Redis image, uses real bootstrap files, and avoids real .secrets/redis
    assert 'REDIS_IMAGE: Final[str] = "redis:7.4.12-alpine"' in harness
    assert "redis:7-alpine" not in harness
    assert "redis:7-alpine" not in live_suite
    assert "users.acl.template" in harness
    assert "redis-acl-entrypoint.sh" in harness
    assert ".secrets/redis" not in harness
    assert ".secrets/redis" not in live_suite

    # Zero host-published ports in live Docker harness
    assert '"-p"' not in harness
    assert '"--publish"' not in harness
    assert "--internal" in harness
    assert "cluster.cleanup()" in harness
    assert "cluster.cleanup()" in live_suite
