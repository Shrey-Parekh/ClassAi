"""
Multi-tier caching system for RAG pipeline.

Implements:
- Memory cache (LRU, fastest)
- Disk cache (persistent, medium speed)
- Optional Redis cache (distributed, fast)
"""

import hashlib
import json
import logging
from typing import Any, Optional, Dict
from functools import lru_cache
import diskcache
from pathlib import Path


class CacheManager:
    """
    Multi-tier cache with automatic fallback.
    
    Tier 1: Memory (LRU) - fastest, limited size
    Tier 2: Disk (diskcache) - persistent, larger
    Tier 3: Redis (optional) - distributed
    """
    
    def __init__(
        self,
        disk_cache_dir: str = "./cache",
        disk_size_limit: int = 1024 * 1024 * 1024,  # 1GB
        enable_redis: bool = False,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0
    ):
        """
        Initialize cache manager.
        
        Args:
            disk_cache_dir: Directory for disk cache
            disk_size_limit: Max disk cache size in bytes
            enable_redis: Whether to use Redis
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database number
        """
        self.logger = logging.getLogger(__name__)
        
        # Disk cache (always enabled)
        Path(disk_cache_dir).mkdir(parents=True, exist_ok=True)
        self.disk_cache = diskcache.Cache(disk_cache_dir, size_limit=disk_size_limit)
        self.logger.info(f"✓ Disk cache initialized: {disk_cache_dir}")
        
        # Redis cache (optional)
        self.redis_client = None
        if enable_redis:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=True
                )
                self.redis_client.ping()
                self.logger.info(f"✓ Redis cache connected: {redis_host}:{redis_port}")
            except Exception as e:
                self.logger.warning(f"Redis unavailable, using disk cache only: {e}")
                self.redis_client = None
    
    def _make_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        
        hash_obj = hashlib.sha256(content.encode())
        return f"{prefix}:{hash_obj.hexdigest()[:16]}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (checks all tiers).
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        # Try Redis first (fastest for distributed)
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    self.logger.debug(f"Cache hit (Redis): {key}")
                    return json.loads(value)
            except Exception as e:
                self.logger.debug(f"Redis get failed: {e}")
        
        # Try disk cache
        try:
            value = self.disk_cache.get(key)
            if value is not None:
                self.logger.debug(f"Cache hit (Disk): {key}")
                return value
        except Exception as e:
            self.logger.debug(f"Disk cache get failed: {e}")
        
        self.logger.debug(f"Cache miss: {key}")
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache (all tiers).
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = no expiry)
        
        Returns:
            True if successful
        """
        success = False
        
        # Set in Redis
        if self.redis_client:
            try:
                serialized = json.dumps(value)
                if ttl:
                    self.redis_client.setex(key, ttl, serialized)
                else:
                    self.redis_client.set(key, serialized)
                success = True
            except Exception as e:
                self.logger.debug(f"Redis set failed: {e}")
        
        # Set in disk cache
        try:
            if ttl:
                self.disk_cache.set(key, value, expire=ttl)
            else:
                self.disk_cache.set(key, value)
            success = True
        except Exception as e:
            self.logger.debug(f"Disk cache set failed: {e}")
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete key from all cache tiers."""
        success = False
        
        if self.redis_client:
            try:
                self.redis_client.delete(key)
                success = True
            except Exception as e:
                self.logger.debug(f"Redis delete failed: {e}")
        
        try:
            self.disk_cache.delete(key)
            success = True
        except Exception as e:
            self.logger.debug(f"Disk cache delete failed: {e}")
        
        return success
    
    def clear(self) -> bool:
        """Clear all caches."""
        success = False
        
        if self.redis_client:
            try:
                self.redis_client.flushdb()
                success = True
            except Exception as e:
                self.logger.debug(f"Redis clear failed: {e}")
        
        try:
            self.disk_cache.clear()
            success = True
        except Exception as e:
            self.logger.debug(f"Disk cache clear failed: {e}")
        
        return success
    
    def get_embedding(self, text: str, model: str) -> Optional[list]:
        """Get cached embedding."""
        key = self._make_key(f"emb:{model}", text)
        return self.get(key)
    
    def set_embedding(self, text: str, model: str, embedding: list, ttl: int = 86400) -> bool:
        """Cache embedding (24h default TTL)."""
        key = self._make_key(f"emb:{model}", text)
        return self.set(key, embedding, ttl)
    
    def get_query_result(self, query: str, top_k: int) -> Optional[Dict]:
        """Get cached query result."""
        key = self._make_key("query", f"{query}:{top_k}")
        return self.get(key)
    
    def set_query_result(self, query: str, top_k: int, result: Dict, ttl: int = 3600) -> bool:
        """Cache query result (1h default TTL)."""
        key = self._make_key("query", f"{query}:{top_k}")
        return self.set(key, result, ttl)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "disk": {
                "size": self.disk_cache.volume(),
                "count": len(self.disk_cache)
            }
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info("stats")
                stats["redis"] = {
                    "keys": self.redis_client.dbsize(),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0)
                }
            except Exception as e:
                self.logger.debug(f"Redis stats failed: {e}")
        
        return stats
