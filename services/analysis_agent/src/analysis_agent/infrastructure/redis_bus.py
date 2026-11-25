import asyncio
import json
import logging
from collections.abc import Callable, Coroutine

import redis.asyncio as redis
from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    AgentHeartbeat,
    RedisConnectionConfig,
    TaskRequested,
    WakeUpHint,
    resolve_client_connection_kwargs,
)

logger = logging.getLogger(__name__)


async def retry_with_backoff[T](
    action: Callable[[], Coroutine[object, object, T]],
    max_retries: int = 5,
    initial_delay: float = 0.5,
    max_delay: float = 5.0,
    backoff_factor: float = 2.0,
) -> T:
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return await action()
        except (redis.ConnectionError, redis.TimeoutError, OSError) as exc:
            if attempt == max_retries:
                logger.error("Redis operation failed after %d attempts: %s", max_retries, exc)
                raise
            logger.warning(
                "Redis connection failed (attempt %d/%d). Retrying in %.1fs: %s",
                attempt,
                max_retries,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)
    raise RuntimeError("Retry loop exited unexpectedly")


class RedisBus:
    def __init__(
        self,
        url: str,
        queue_name: str = "agent.tasks.analysis",
        hint_queue: str = "agent.hints.analysis",
        *,
        connection_config: RedisConnectionConfig | None = None,
        username: str | None = None,
        password: str | None = None,
        allow_unauthenticated_local_test: bool = False,
    ) -> None:
        resolved_url, client_kwargs = resolve_client_connection_kwargs(
            url,
            connection_config=connection_config,
            username=username,
            password=password,
            allow_unauthenticated_local_test=allow_unauthenticated_local_test,
        )
        self.client = redis.from_url(resolved_url, **client_kwargs)
        self.queue_name = queue_name
        self.hint_queue = hint_queue
        self.current_state = "idle"
        self._prefer_hints: bool = True

    async def next_signal(self) -> TaskRequested | WakeUpHint:
        async def fetch():
            queues = (
                [self.hint_queue, self.queue_name]
                if self._prefer_hints
                else [self.queue_name, self.hint_queue]
            )
            return await self.client.blpop(queues, timeout=2)

        while True:
            result = await retry_with_backoff(fetch)
            if result is not None:
                source_queue, value = result
                self._prefer_hints = not self._prefer_hints
                if source_queue == self.hint_queue:
                    return WakeUpHint.from_json(value)
                return TaskRequested.from_json(value)

    async def next_task(self) -> TaskRequested:
        async def fetch() -> tuple[str, str] | None:
            return await self.client.blpop(self.queue_name, timeout=2)

        while True:
            result = await retry_with_backoff(fetch)
            if result is not None:
                _, value = result
                return TaskRequested.from_json(value)

    async def register(self, metadata: AgentCapabilityMetadata) -> None:
        mapping = {
            "agent_id": metadata.agent_id,
            "agent_type": metadata.agent_type,
            "capabilities": json.dumps(metadata.capabilities),
            "providers": json.dumps(metadata.providers),
            "version": str(metadata.version),
        }
        if metadata.dispatch_target:
            mapping["dispatch_target"] = metadata.dispatch_target
        for key, value in metadata.providers.items():
            mapping[key] = str(value)

        async def write() -> None:
            await self.client.sadd("agent.ids", metadata.agent_id)
            await self.client.hset(f"agent.state:{metadata.agent_id}", mapping=mapping)

        await retry_with_backoff(write)

    async def publish(self, event: AgentEvent) -> None:
        if event.state is not None:
            self.current_state = event.state
        mapping = {
            key: ""
            if value is None
            else json.dumps(value)
            if isinstance(value, (dict, list))
            else str(value)
            for key, value in event.to_dict().items()
        }
        mapping["updated_at"] = event.occurred_at

        async def write() -> None:
            await self.client.sadd("agent.ids", event.agent_id)
            if event.state is not None:
                await self.client.hset(f"agent.state:{event.agent_id}", mapping=mapping)
            await self.client.xadd("agent.events", mapping, maxlen=500, approximate=True)

        await retry_with_backoff(write)

    async def heartbeat(self, heartbeat: AgentHeartbeat) -> None:
        mapping = {"updated_at": heartbeat.occurred_at}

        async def write() -> None:
            await self.client.sadd("agent.ids", heartbeat.agent_id)
            await self.client.hset(f"agent.state:{heartbeat.agent_id}", mapping=mapping)

        await retry_with_backoff(write)

    async def aclose(self) -> None:
        await self.client.aclose()
