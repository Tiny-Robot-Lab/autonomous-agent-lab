"""Authenticated host-side clients for disposable Redis integration tests."""

from __future__ import annotations

from typing import Any

import redis.asyncio as redis
from orchestrator.infrastructure.redis_adapter import RedisGateway

from scripts.redis_acl_live_harness import (
    TEST_ADMIN_IDENTITY,
    LiveRedisAclCluster,
)

_ACTIVE_CLUSTER: LiveRedisAclCluster | None = None


class RedisTestGateway(RedisGateway):
    """Keep application calls on Orchestrator auth and test inspection on fixture-admin."""

    def __init__(
        self,
        gateway: RedisGateway,
        fixture_admin_client: redis.Redis,
    ) -> None:
        object.__setattr__(self, "_gateway", gateway)
        object.__setattr__(self, "_client", fixture_admin_client)

    def __getattribute__(self, name: str) -> Any:
        if name in {
            "_gateway",
            "_client",
            "production_client",
            "aclose",
            "__dict__",
            "__class__",
        }:
            return object.__getattribute__(self, name)
        gateway = object.__getattribute__(self, "_gateway")
        try:
            return getattr(gateway, name)
        except AttributeError:
            return object.__getattribute__(self, name)

    @property
    def production_client(self) -> redis.Redis:
        return self._gateway._client

    async def aclose(self) -> None:
        try:
            await self._gateway.aclose()
        finally:
            await self._client.aclose()


def register_cluster(cluster: LiveRedisAclCluster) -> None:
    global _ACTIVE_CLUSTER
    if _ACTIVE_CLUSTER is not None:
        raise RuntimeError("A Redis integration cluster is already registered")
    _ACTIVE_CLUSTER = cluster


def unregister_cluster(cluster: LiveRedisAclCluster) -> None:
    global _ACTIVE_CLUSTER
    if _ACTIVE_CLUSTER is cluster:
        _ACTIVE_CLUSTER = None


def _cluster_for(url: str) -> LiveRedisAclCluster:
    if _ACTIVE_CLUSTER is None or _ACTIVE_CLUSTER.redis_url != url:
        raise RuntimeError("Redis test clients require the active disposable ACL cluster")
    return _ACTIVE_CLUSTER


def fixture_admin_client(
    url: str,
    *,
    decode_responses: bool = True,
) -> redis.Redis:
    cluster = _cluster_for(url)
    return redis.from_url(
        url,
        decode_responses=decode_responses,
        protocol=2,
        driver_info=None,
        username=TEST_ADMIN_IDENTITY,
        password=cluster.admin_secret,
    )


def make_test_gateway(
    url: str,
    *,
    delivery_mode: str = "legacy_push",
) -> RedisTestGateway:
    cluster = _cluster_for(url)
    gateway = RedisGateway(
        url,
        delivery_mode=delivery_mode,
        username="orchestrator",
        password=cluster.secrets_map["orchestrator"],
    )
    return RedisTestGateway(gateway, fixture_admin_client(url))


def make_test_bus(
    bus_type: type[Any],
    url: str,
    identity: str,
    *,
    queue_name: str | None = None,
    hint_queue: str | None = None,
) -> Any:
    cluster = _cluster_for(url)
    if identity not in cluster.secrets_map:
        raise ValueError(f"Unknown production Redis identity: {identity}")
    options: dict[str, Any] = {
        "username": identity,
        "password": cluster.secrets_map[identity],
    }
    if queue_name is not None:
        options["queue_name"] = queue_name
    if hint_queue is not None:
        options["hint_queue"] = hint_queue
    return bus_type(url, **options)
