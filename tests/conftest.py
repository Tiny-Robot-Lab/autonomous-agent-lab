from __future__ import annotations

import shutil
import subprocess

import pytest

_PROVIDER_ENVIRONMENT_VARIABLES = (
    "RESEARCH_EXECUTOR",
    "TAVILY_API_KEY",
    "TAVILY_MCP_URL",
    "INFERENCE_PROVIDER",
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_AI_API_TOKEN",
    "CLOUDFLARE_AI_MODEL",
    "CLOUDFLARE_AI_MAX_TOKENS",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_BASE_URL",
    "GEMINI_MAX_TOKENS",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("external providers")
    group.addoption(
        "--run-external-providers",
        action="store_true",
        default=False,
        help="allow tests marked external_provider to execute",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("run_external_providers"):
        return

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        if item.get_closest_marker("external_provider") is None:
            selected.append(item)
        else:
            deselected.append(item)

    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)


@pytest.fixture(autouse=True)
def isolate_provider_environment(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if request.node.get_closest_marker("external_provider") is not None:
        return

    for variable in _PROVIDER_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def provider_environment_variables() -> tuple[str, ...]:
    return _PROVIDER_ENVIRONMENT_VARIABLES


@pytest.fixture(scope="session")
def redis_test_environment():
    if shutil.which("docker") is None:
        pytest.skip("Redis-marked tests require Docker; Docker is not installed")
    docker_info = subprocess.run(
        ["docker", "info"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if docker_info.returncode != 0:
        pytest.skip("Redis-marked tests require an available Docker daemon")

    from scripts.redis_acl_live_harness import LiveRedisAclCluster
    from scripts.redis_test_support import register_cluster, unregister_cluster

    cluster = LiveRedisAclCluster.start(
        build_service_images=False,
        publish_host_port=True,
    )
    register_cluster(cluster)
    try:
        yield cluster
    finally:
        unregister_cluster(cluster)
        cluster.cleanup()


@pytest.fixture(autouse=True)
def require_disposable_redis_for_marked_tests(
    request: pytest.FixtureRequest,
) -> None:
    if request.node.get_closest_marker("redis") is not None:
        request.getfixturevalue("redis_test_environment")


@pytest.fixture
def redis_url(redis_test_environment) -> str:
    return redis_test_environment.redis_url


@pytest.fixture
async def clean_gateway(redis_test_environment):
    from scripts.redis_test_support import make_test_gateway

    gateway = make_test_gateway(redis_test_environment.redis_url)
    try:
        await gateway._client.ping()
        await gateway._client.flushdb()
        yield gateway
    finally:
        try:
            await gateway._client.flushdb()
        finally:
            await gateway.aclose()
