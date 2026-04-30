"""
API Rate Limiting Module
Implements rate limiting for API endpoints to prevent abuse
"""

import logging
import os
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer
import redis
import time

logger = logging.getLogger(__name__)

# Rate limiting configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Rate limits (requests per minute)
DEFAULT_RATE_LIMIT = int(os.getenv("DEFAULT_RATE_LIMIT", 60))  # 60 requests per minute
AUTHENTICATED_RATE_LIMIT = int(os.getenv("AUTHENTICATED_RATE_LIMIT", 120))  # 120 requests per minute

class RateLimiter:
    """
    Redis-based rate limiter for API endpoints.
    Uses sliding window algorithm for accurate rate limiting.
    """
    
    def __init__(self):
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis server"""
        try:
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
            logger.info(f"Rate limiter connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis for rate limiting: {e}. Rate limiting disabled.")
            self.redis_client = None
    
    def is_allowed(
        self,
        key: str,
        limit: int = DEFAULT_RATE_LIMIT,
        window: int = 60
    ) -> bool:
        """
        Check if request is allowed based on rate limit.
        
        Args:
            key: Unique identifier for rate limiting (e.g., user ID or IP)
            limit: Maximum requests allowed in window
            window: Time window in seconds
            
        Returns:
            True if request is allowed, False otherwise
        """
        if not self.redis_client:
            # If Redis is unavailable, allow all requests (fail open)
            return True
        
        try:
            current_time = int(time.time())
            window_start = current_time - window
            
            # Remove old entries outside the window
            self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            current_count = self.redis_client.zcard(key)
            
            if current_count >= limit:
                logger.warning(f"Rate limit exceeded for key: {key} (count: {current_count}, limit: {limit})")
                return False
            
            # Add current request
            self.redis_client.zadd(key, {str(current_time): current_time})
            self.redis_client.expire(key, window)
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}. Allowing request.")
            return True
    
    def get_remaining(self, key: str, limit: int = DEFAULT_RATE_LIMIT, window: int = 60) -> int:
        """
        Get remaining requests for a key.
        
        Args:
            key: Unique identifier for rate limiting
            limit: Maximum requests allowed in window
            window: Time window in seconds
            
        Returns:
            Number of remaining requests
        """
        if not self.redis_client:
            return limit
        
        try:
            current_time = int(time.time())
            window_start = current_time - window
            
            # Remove old entries outside the window
            self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            current_count = self.redis_client.zcard(key)
            
            return max(0, limit - current_count)
            
        except Exception as e:
            logger.error(f"Error getting remaining requests: {e}")
            return limit


# Global singleton instance
rate_limiter = RateLimiter()

async def check_rate_limit(
    request: Request,
    limit: int = DEFAULT_RATE_LIMIT,
    window: int = 60
):
    """
    FastAPI dependency for rate limiting.
    
    Args:
        request: FastAPI request object
        limit: Maximum requests allowed in window
        window: Time window in seconds
        
    Raises:
        HTTPException if rate limit is exceeded
    """
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Get user ID if authenticated
    user_id = None
    if hasattr(request.state, "user"):
        user_id = request.state.user.username
    
    # Use user ID if authenticated, otherwise use IP
    rate_limit_key = f"rate_limit:{user_id}" if user_id else f"rate_limit:ip:{client_ip}"
    
    # Use higher limit for authenticated users
    actual_limit = AUTHENTICATED_RATE_LIMIT if user_id else limit
    
    if not rate_limiter.is_allowed(rate_limit_key, actual_limit, window):
        remaining = rate_limiter.get_remaining(rate_limit_key, actual_limit, window)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "limit": actual_limit,
                "remaining": remaining,
                "window": window
            },
            headers={
                "X-RateLimit-Limit": str(actual_limit),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int(time.time()) + window)
            }
        )
    
    # Add rate limit headers
    remaining = rate_limiter.get_remaining(rate_limit_key, actual_limit, window)
    return {
        "X-RateLimit-Limit": str(actual_limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(int(time.time()) + window)
    }
