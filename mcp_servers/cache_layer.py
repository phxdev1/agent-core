#!/usr/bin/env python3
"""
Universal MCP Cache Layer
Automatically caches all MCP server responses to reduce API usage
Works transparently with any MCP server
"""

import json
import hashlib
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import sys
import redis
from datetime import datetime, timedelta
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import config
from utils.redis_logger import get_redis_logger

logger = get_redis_logger(__name__)


class MCPCacheLayer:
    """
    Universal caching layer for all MCP server calls
    Uses Redis for distributed caching with fallback to local file cache
    """
    
    def __init__(self, 
                 cache_ttl: int = 3600,  # 1 hour default TTL
                 redis_prefix: str = "mcp_cache:",
                 local_cache_dir: Optional[Path] = None):
        """
        Initialize the cache layer
        
        Args:
            cache_ttl: Time to live for cache entries in seconds
            redis_prefix: Prefix for Redis keys
            local_cache_dir: Directory for local file cache fallback
        """
        self.cache_ttl = cache_ttl
        self.redis_prefix = redis_prefix
        self.local_cache_dir = local_cache_dir or Path.home() / ".agent_core" / "mcp_cache"
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "api_calls_saved": 0,
            "cache_size": 0
        }
        
        # Initialize Redis connection
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(
                host=config['redis']['host'],
                port=config['redis']['port'],
                username=config['redis'].get('username', 'default'),
                password=config['redis']['password'],
                db=config['redis']['db'],
                decode_responses=False  # We'll handle encoding/decoding
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis cache unavailable, using local cache: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for the request
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool
            
        Returns:
            Unique cache key
        """
        # Create a deterministic string from the request
        key_data = {
            "server": server_name,
            "tool": tool_name,
            "args": arguments
        }
        
        # Sort keys for consistency
        key_string = json.dumps(key_data, sort_keys=True)
        
        # Generate hash for compact storage
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"{self.redis_prefix}{server_name}:{tool_name}:{key_hash}"
    
    async def get_cached_response(self, 
                                  server_name: str, 
                                  tool_name: str, 
                                  arguments: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached response if available
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool
            
        Returns:
            Cached response or None if not found
        """
        cache_key = self._generate_cache_key(server_name, tool_name, arguments)
        
        # Try Redis first
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    self.stats["hits"] += 1
                    self.stats["api_calls_saved"] += 1
                    
                    # Deserialize the cached response
                    response = pickle.loads(cached_data)
                    
                    logger.debug(f"Cache hit for {server_name}:{tool_name}")
                    return response
                    
            except Exception as e:
                logger.warning(f"Redis cache read error: {e}")
                self.stats["errors"] += 1
        
        # Fallback to local file cache
        local_cache_file = self.local_cache_dir / f"{cache_key.replace(':', '_')}.json"
        if local_cache_file.exists():
            try:
                # Check if cache is still valid
                file_age = time.time() - local_cache_file.stat().st_mtime
                if file_age < self.cache_ttl:
                    with open(local_cache_file, 'r', encoding='utf-8') as f:
                        response = json.load(f)
                    
                    self.stats["hits"] += 1
                    self.stats["api_calls_saved"] += 1
                    
                    logger.debug(f"Local cache hit for {server_name}:{tool_name}")
                    return response
                else:
                    # Cache expired, delete it
                    local_cache_file.unlink()
                    
            except Exception as e:
                logger.warning(f"Local cache read error: {e}")
                self.stats["errors"] += 1
        
        self.stats["misses"] += 1
        return None
    
    async def cache_response(self,
                           server_name: str,
                           tool_name: str,
                           arguments: Dict[str, Any],
                           response: Any) -> None:
        """
        Cache a response
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool
            response: Response to cache
        """
        cache_key = self._generate_cache_key(server_name, tool_name, arguments)
        
        # Store in Redis if available
        if self.redis_client:
            try:
                # Serialize the response
                serialized = pickle.dumps(response)
                
                # Store with TTL
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    serialized
                )
                
                # Update stats
                self.stats["cache_size"] = self.redis_client.dbsize()
                
                logger.debug(f"Cached response for {server_name}:{tool_name} in Redis")
                
            except Exception as e:
                logger.warning(f"Redis cache write error: {e}")
                self.stats["errors"] += 1
        
        # Also store in local file cache as backup
        try:
            local_cache_file = self.local_cache_dir / f"{cache_key.replace(':', '_')}.json"
            
            # Convert response to JSON-serializable format
            if hasattr(response, '__dict__'):
                response_data = response.__dict__
            elif isinstance(response, (list, dict, str, int, float, bool, type(None))):
                response_data = response
            else:
                response_data = str(response)
            
            with open(local_cache_file, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, default=str)
            
            logger.debug(f"Cached response for {server_name}:{tool_name} locally")
            
        except Exception as e:
            logger.warning(f"Local cache write error: {e}")
            self.stats["errors"] += 1
    
    def clear_cache(self, server_name: Optional[str] = None, tool_name: Optional[str] = None) -> int:
        """
        Clear cache entries
        
        Args:
            server_name: Clear only for specific server (optional)
            tool_name: Clear only for specific tool (optional)
            
        Returns:
            Number of entries cleared
        """
        cleared = 0
        
        # Clear from Redis
        if self.redis_client:
            try:
                pattern = f"{self.redis_prefix}"
                if server_name:
                    pattern += f"{server_name}:"
                    if tool_name:
                        pattern += f"{tool_name}:"
                pattern += "*"
                
                keys = list(self.redis_client.scan_iter(match=pattern))
                if keys:
                    cleared += self.redis_client.delete(*keys)
                    
            except Exception as e:
                logger.error(f"Redis cache clear error: {e}")
        
        # Clear from local cache
        pattern = ""
        if server_name:
            pattern += f"{server_name}_"
            if tool_name:
                pattern += f"{tool_name}_"
        
        for cache_file in self.local_cache_dir.glob(f"{self.redis_prefix.replace(':', '_')}{pattern}*.json"):
            try:
                cache_file.unlink()
                cleared += 1
            except Exception as e:
                logger.error(f"Local cache clear error: {e}")
        
        logger.info(f"Cleared {cleared} cache entries")
        return cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": f"{hit_rate:.1f}%",
            "total_requests": total_requests,
            "cache_backend": "Redis" if self.redis_client else "Local File"
        }
    
    def set_ttl_for_tool(self, tool_name: str, ttl: int) -> None:
        """
        Set custom TTL for specific tools
        Useful for tools with different freshness requirements
        
        Args:
            tool_name: Name of the tool
            ttl: Time to live in seconds
        """
        if not hasattr(self, 'custom_ttls'):
            self.custom_ttls = {}
        self.custom_ttls[tool_name] = ttl
        logger.info(f"Set TTL for {tool_name} to {ttl} seconds")
    
    def get_ttl_for_tool(self, tool_name: str) -> int:
        """Get TTL for a specific tool"""
        if hasattr(self, 'custom_ttls') and tool_name in self.custom_ttls:
            return self.custom_ttls[tool_name]
        return self.cache_ttl


class CachedMCPClient:
    """
    Wrapper for MCP clients that automatically adds caching
    """
    
    def __init__(self, mcp_client: Any, cache_layer: MCPCacheLayer, server_name: str):
        """
        Initialize cached MCP client
        
        Args:
            mcp_client: The original MCP client
            cache_layer: The cache layer to use
            server_name: Name of the MCP server
        """
        self.client = mcp_client
        self.cache = cache_layer
        self.server_name = server_name
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool with automatic caching
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            Tool response (from cache or fresh call)
        """
        # Check cache first
        cached_response = await self.cache.get_cached_response(
            self.server_name, 
            tool_name, 
            arguments
        )
        
        if cached_response is not None:
            logger.info(f"Using cached response for {self.server_name}:{tool_name}")
            return cached_response
        
        # Make the actual API call
        logger.info(f"Making API call to {self.server_name}:{tool_name}")
        response = await self.client.call_tool(tool_name, arguments)
        
        # Cache the response
        await self.cache.cache_response(
            self.server_name,
            tool_name,
            arguments,
            response
        )
        
        return response


# Global cache instance
_cache_layer = None

def get_cache_layer() -> MCPCacheLayer:
    """Get or create the global cache layer"""
    global _cache_layer
    if _cache_layer is None:
        _cache_layer = MCPCacheLayer(
            cache_ttl=3600,  # 1 hour default
            redis_prefix="mcp_cache:"
        )
        
        # Set custom TTLs for different tools
        _cache_layer.set_ttl_for_tool("web_search", 1800)  # 30 minutes for web search
        _cache_layer.set_ttl_for_tool("news_search", 900)   # 15 minutes for news
        _cache_layer.set_ttl_for_tool("shopping_search", 3600)  # 1 hour for shopping
        _cache_layer.set_ttl_for_tool("scholar_search", 7200)  # 2 hours for academic
        _cache_layer.set_ttl_for_tool("ai_overview", 3600)  # 1 hour for AI overviews
        
    return _cache_layer


def wrap_mcp_client(mcp_client: Any, server_name: str) -> CachedMCPClient:
    """
    Wrap an MCP client with caching
    
    Args:
        mcp_client: The MCP client to wrap
        server_name: Name of the MCP server
        
    Returns:
        Cached MCP client
    """
    cache_layer = get_cache_layer()
    return CachedMCPClient(mcp_client, cache_layer, server_name)