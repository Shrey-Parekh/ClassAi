"""
Rate limiter with sliding window algorithm.

In-memory implementation, no Redis required for internal use.
"""

import time
from collections import defaultdict, deque
from typing import Dict, Tuple
import asyncio


class RateLimiter:
    """
    Sliding window rate limiter.
    
    Tracks requests per IP; timestamps older than the window are dropped.
    """
    
    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
        # Store timestamps per IP: {ip: deque([timestamp1, timestamp2, ...])}
        # ponytail: one entry per IP is never evicted; fine for local use,
        # evict idle IPs here if this is ever exposed publicly.
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, client_ip: str) -> Tuple[bool, int]:
        """
        Check if request is allowed for this IP.
        
        Args:
            client_ip: Client IP address
        
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        async with self.lock:
            now = time.time()
            cutoff = now - self.window_seconds
            
            # Get request history for this IP
            request_times = self.requests[client_ip]
            
            # Remove old requests outside window
            while request_times and request_times[0] < cutoff:
                request_times.popleft()
            
            # Check if under limit
            if len(request_times) < self.max_requests:
                request_times.append(now)
                return True, 0
            
            # Calculate retry-after (time until oldest request expires)
            oldest_request = request_times[0]
            retry_after = int(oldest_request + self.window_seconds - now) + 1
            
            return False, retry_after
