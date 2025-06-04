from typing import Optional, Any, Dict
import json
from redis.asyncio import Redis, ConnectionPool
from ..config import settings
from langgraph.checkpoint.redis import AsyncRedisSaver

class RedisService:
    def __init__(self):
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[Redis] = None
        self.checkpointer: Optional[AsyncRedisSaver] = None

    async def initialize(self):
        """Initialize Redis connection pool and client."""
        if not self.pool:
            self.pool = ConnectionPool.from_url(
                settings.get_redis_url(),
                decode_responses=True
            )
            self.client = Redis(connection_pool=self.pool)
            self.checkpointer = AsyncRedisSaver.from_conn_string(
                settings.get_redis_url()
            )

    async def close(self):
        """Close Redis connections."""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()

    async def save_checkpoint(self, key: str, state: Dict[str, Any]):
        """Save a checkpoint to Redis."""
        if not self.client:
            await self.initialize()
        await self.client.set(
            f"checkpoint:{key}",
            json.dumps(state),
            ex=3600  # Expire after 1 hour
        )

    async def load_checkpoint(self, key: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint from Redis."""
        if not self.client:
            await self.initialize()
        data = await self.client.get(f"checkpoint:{key}")
        return json.loads(data) if data else None

    async def delete_checkpoint(self, key: str):
        """Delete a checkpoint from Redis."""
        if not self.client:
            await self.initialize()
        await self.client.delete(f"checkpoint:{key}")

    def get_checkpointer(self) -> AsyncRedisSaver:
        """Get the LangGraph Redis checkpointer."""
        if not self.checkpointer:
            self.checkpointer = AsyncRedisSaver.from_conn_string(
                settings.get_redis_url()
            )
        return self.checkpointer 