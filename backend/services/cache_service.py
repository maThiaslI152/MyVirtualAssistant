from typing import Optional, Any, Dict, List, Union
import json
from datetime import datetime, timedelta
import hashlib
from redis.asyncio import Redis
from config import settings
import logging
import redis
from redis.exceptions import RedisError
import pickle
import zlib
from functools import wraps
import time
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.redis_client = self._init_redis()
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_items, daemon=True)
        self.stats_thread = threading.Thread(target=self._report_stats, daemon=True)
        self.cleanup_thread.start()
        self.stats_thread.start()
        
        # Initialize cache warming
        self.warmup_executor = ThreadPoolExecutor(
            max_workers=settings.CACHE_WARMUP_WORKERS
        )
    
    def _init_redis(self) -> redis.Redis:
        """Initialize Redis with optimal settings."""
        try:
            client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Set optimal Redis configuration
            client.config_set('maxmemory', '1gb')
            client.config_set('maxmemory-policy', 'allkeys-lru')
            client.config_set('appendonly', 'yes')
            client.config_set('appendfsync', 'everysec')
            
            return client
        except redis.RedisError as e:
            logger.error(f"Failed to initialize Redis: {e}")
            return None

    def _cleanup_expired_items(self):
        """Background task to clean up expired items from memory cache."""
        while True:
            try:
                current_time = datetime.now()
                expired_keys = [
                    key for key, data in self.memory_cache.items()
                    if data['expires_at'] and data['expires_at'] < current_time
                ]
                for key in expired_keys:
                    del self.memory_cache[key]
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
                time.sleep(60)

    def _report_stats(self):
        """Background task to report cache statistics."""
        while True:
            try:
                memory_size = len(pickle.dumps(self.memory_cache))
                memory_items = len(self.memory_cache)
                redis_info = self.redis_client.info() if self.redis_client else {}
                
                logger.info(f"Cache Stats - Memory: {memory_items} items, {memory_size/1024/1024:.2f}MB")
                if redis_info:
                    logger.info(f"Redis Stats - Used Memory: {redis_info.get('used_memory_human', 'N/A')}")
                time.sleep(300)  # Report every 5 minutes
            except Exception as e:
                logger.error(f"Error in stats thread: {e}")
                time.sleep(300)

    def _compress(self, data: Any) -> bytes:
        """Compress data if it exceeds the threshold."""
        serialized = pickle.dumps(data)
        if len(serialized) > settings.CACHE_COMPRESSION_THRESHOLD:
            return zlib.compress(serialized)
        return serialized

    def _decompress(self, data: bytes) -> Any:
        """Decompress data if it's compressed."""
        try:
            return pickle.loads(zlib.decompress(data))
        except:
            return pickle.loads(data)

    def get(self, key: str, level: str = "memory") -> Optional[Any]:
        """Get a value from the cache."""
        try:
            if level == "memory":
                if key in self.memory_cache:
                    data = self.memory_cache[key]
                    if data['expires_at'] and data['expires_at'] < datetime.now():
                        del self.memory_cache[key]
                        return None
                    return data['value']
            elif level == "redis" and self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    return self._decompress(data)
            return None
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None, level: str = "memory") -> bool:
        """Set a value in the cache."""
        try:
            ttl = ttl or settings.CACHE_TTL
            expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
            
            if level == "memory":
                # Check memory limit
                if len(pickle.dumps(self.memory_cache)) > settings.CACHE_MAX_MEMORY:
                    # Remove oldest items
                    sorted_items = sorted(
                        self.memory_cache.items(),
                        key=lambda x: x[1]['created_at']
                    )
                    while len(pickle.dumps(self.memory_cache)) > settings.CACHE_MAX_MEMORY:
                        if not sorted_items:
                            break
                        del self.memory_cache[sorted_items.pop(0)[0]]
                
                self.memory_cache[key] = {
                    'value': value,
                    'created_at': datetime.now(),
                    'expires_at': expires_at
                }
            elif level == "redis" and self.redis_client:
                compressed_data = self._compress(value)
                self.redis_client.setex(key, ttl, compressed_data)
            return True
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False

    def delete(self, key: str, level: str = "memory") -> bool:
        """Delete a value from the cache."""
        try:
            if level == "memory":
                self.memory_cache.pop(key, None)
            elif level == "redis" and self.redis_client:
                self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False

    def clear(self, level: str = "memory") -> bool:
        """Clear all values from the cache."""
        try:
            if level == "memory":
                self.memory_cache.clear()
            elif level == "redis" and self.redis_client:
                self.redis_client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def warmup(self, items: List[Dict[str, Any]]) -> None:
        """Preload multiple items into the cache in parallel."""
        def _warmup_item(item: Dict[str, Any]) -> None:
            try:
                self.set(
                    item['key'],
                    item['value'],
                    ttl=item.get('ttl'),
                    level=item.get('level', 'memory')
                )
            except Exception as e:
                logger.error(f"Error warming up cache item {item.get('key')}: {e}")

        futures = []
        for item in items:
            futures.append(self.warmup_executor.submit(_warmup_item, item))
        concurrent.futures.wait(futures)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            memory_stats = {
                'items': len(self.memory_cache),
                'size': len(pickle.dumps(self.memory_cache)),
                'hits': sum(1 for data in self.memory_cache.values() if data.get('hits', 0) > 0)
            }
            
            redis_stats = {}
            if self.redis_client:
                info = self.redis_client.info()
                redis_stats = {
                    'used_memory': info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients'),
                    'total_keys': info.get('db0', {}).get('keys', 0)
                }
            
            return {
                'memory': memory_stats,
                'redis': redis_stats,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

    def cache_decorator(self, ttl: Optional[int] = None):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # If not in cache, call function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl)
                return result
            return wrapper
        return decorator

    async def cache_search_results(
        self,
        query: str,
        results: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Cache search results."""
        key = self._generate_key("search", query)
        await self.set(key, results, ttl)

    async def get_cached_search_results(
        self,
        query: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached search results."""
        key = self._generate_key("search", query)
        return await self.get(key)

    async def cache_embeddings(
        self,
        text: str,
        embedding: list[float],
        ttl: Optional[int] = None
    ):
        """Cache text embeddings."""
        key = self._generate_key("embedding", text)
        await self.set(key, embedding, ttl)

    async def get_cached_embedding(
        self,
        text: str
    ) -> Optional[list[float]]:
        """Get cached text embedding."""
        key = self._generate_key("embedding", text)
        return await self.get(key)

    def _generate_key(self, key: str) -> str:
        """Generate a cache key with namespace."""
        return f"cache:{hashlib.md5(key.encode()).hexdigest()}" 