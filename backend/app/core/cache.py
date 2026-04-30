"""
Redis Caching Service for Performance Optimization
Caches entity extraction results to improve response times
"""

import logging
import os
import json
import hashlib
from typing import Optional, Dict, Any
from datetime import timedelta

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # Default 1 hour

class CacheService:
    """
    Redis-based caching service for entity extraction results.
    Improves performance by caching repeated extractions.
    """
    
    def __init__(self):
        self.enabled = os.getenv("ENABLE_CACHE", "true").lower() == "true"
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis server"""
        if not self.enabled:
            logger.info("Caching disabled")
            return
        
        try:
            import redis
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except ImportError:
            logger.warning("Redis package not installed. Install with: pip install redis")
            self.enabled = False
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
            self.enabled = False
    
    def _generate_cache_key(self, text: str, model_version: str = "v1") -> str:
        """
        Generate a unique cache key for text extraction.
        
        Args:
            text: Clinical text to extract
            model_version: Model version identifier
            
        Returns:
            Cache key
        """
        # Create hash of text for cache key
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"ner:{model_version}:{text_hash}"
    
    def get(self, text: str, model_version: str = "v1") -> Optional[Dict[str, Any]]:
        """
        Get cached extraction result.
        
        Args:
            text: Clinical text
            model_version: Model version identifier
            
        Returns:
            Cached result or None if not found
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key(text, model_version)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                result = json.loads(cached_data)
                logger.info(f"Cache hit for key: {cache_key}")
                return result
            
            logger.info(f"Cache miss for key: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, text: str, result: Dict[str, Any], model_version: str = "v1", ttl: int = CACHE_TTL) -> bool:
        """
        Cache extraction result.
        
        Args:
            text: Clinical text
            result: Extraction result to cache
            model_version: Model version identifier
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key(text, model_version)
            cached_data = json.dumps(result)
            
            self.redis_client.setex(cache_key, ttl, cached_data)
            logger.info(f"Cached result for key: {cache_key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, text: str, model_version: str = "v1") -> bool:
        """
        Delete cached result.
        
        Args:
            text: Clinical text
            model_version: Model version identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key(text, model_version)
            self.redis_client.delete(cache_key)
            logger.info(f"Deleted cache for key: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear_all(self) -> bool:
        """
        Clear all cached results.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            # Delete all keys with 'ner:' prefix
            keys = self.redis_client.keys("ner:*")
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cached results")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        if not self.enabled or not self.redis_client:
            return {"enabled": False}
        
        try:
            info = self.redis_client.info("stats")
            keyspace = self.redis_client.info("keyspace")
            
            return {
                "enabled": True,
                "total_keys": keyspace.get("db0", {}).get("keys", 0),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
            }
            
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"enabled": True, "error": str(e)}
    
    def is_available(self) -> bool:
        """Check if cache service is available"""
        return self.enabled and self.redis_client is not None


# Global singleton instance
cache_service = CacheService()
