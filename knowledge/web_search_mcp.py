#!/usr/bin/env python3
"""
Web Search via SERPAPI MCP
Provides web search capabilities through MCP protocol
"""

import json
import logging
import asyncio
import subprocess
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys
import httpx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import config
from utils.redis_logger import get_redis_logger
sys.path.insert(0, str(Path(__file__).parent.parent / "mcp_servers"))
from cache_layer import get_cache_layer

logger = get_redis_logger(__name__)


class WebSearchMCP:
    """Web search using SERPAPI through MCP protocol"""
    
    def __init__(self):
        self.server_process = None
        self.port = 5174  # Different port from Wikipedia MCP
        self.base_url = f"http://localhost:{self.port}"
        self.api_key = config.get('serpapi_key')
        
        if not self.api_key:
            logger.warning("SERPAPI_KEY not configured")
    
    async def start_server(self):
        """Start the SERPAPI MCP server"""
        if not self.api_key:
            logger.error("Cannot start SERPAPI server without API key")
            return False
        
        try:
            # Start the MCP server
            server_path = Path(__file__).parent.parent / "mcp_servers" / "serpapi_server.py"
            self.server_process = subprocess.Popen(
                ["python", str(server_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "SERPAPI_KEY": self.api_key}
            )
            
            # Give it time to start
            await asyncio.sleep(2)
            logger.info("SERPAPI MCP server started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start SERPAPI MCP server: {e}")
            return False
    
    async def stop_server(self):
        """Stop the SERPAPI MCP server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None
            logger.info("SERPAPI MCP server stopped")
    
    async def web_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web using Google
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/call-tool",
                    json={
                        "method": "tools/call",
                        "params": {
                            "name": "web_search",
                            "arguments": {
                                "query": query,
                                "num_results": num_results
                            }
                        }
                    },
                    timeout=15.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("content", [])
                    if content and isinstance(content, list):
                        # Parse the JSON response
                        return json.loads(content[0].get("text", "[]"))
                    return []
                else:
                    logger.warning(f"Web search failed: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []
    
    async def news_search(self, query: str, time_range: str = "week") -> List[Dict[str, Any]]:
        """
        Search for news articles
        
        Args:
            query: Search query
            time_range: Time range ('hour', 'day', 'week', 'month', 'year')
            
        Returns:
            List of news articles
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/call-tool",
                    json={
                        "method": "tools/call",
                        "params": {
                            "name": "news_search",
                            "arguments": {
                                "query": query,
                                "time_range": time_range
                            }
                        }
                    },
                    timeout=15.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("content", [])
                    if content and isinstance(content, list):
                        return json.loads(content[0].get("text", "[]"))
                    return []
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Error in news search: {e}")
            return []
    
    async def scholar_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for academic papers
        
        Args:
            query: Search query
            num_results: Number of results
            
        Returns:
            List of academic papers
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/call-tool",
                    json={
                        "method": "tools/call",
                        "params": {
                            "name": "scholar_search",
                            "arguments": {
                                "query": query,
                                "num_results": num_results
                            }
                        }
                    },
                    timeout=15.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("content", [])
                    if content and isinstance(content, list):
                        return json.loads(content[0].get("text", "[]"))
                    return []
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Error in scholar search: {e}")
            return []


class DirectWebSearch:
    """Direct web search using SERPAPI (fallback when MCP unavailable)"""
    
    def __init__(self):
        self.api_key = config.get('serpapi_key')
        self.cache = get_cache_layer()
        
    async def web_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Direct web search using SERPAPI with caching"""
        if not self.api_key:
            logger.warning("SERPAPI_KEY not configured")
            return []
        
        # Check cache first
        cached_response = await self.cache.get_cached_response(
            "serpapi_direct",
            "web_search",
            {"query": query, "num_results": num_results}
        )
        
        if cached_response is not None:
            logger.info(f"Using cached web search for: {query}")
            return cached_response
        
        try:
            from serpapi import GoogleSearch
            
            search = GoogleSearch({
                "q": query,
                "api_key": self.api_key,
                "num": num_results
            })
            
            results = search.get_dict()
            organic_results = results.get("organic_results", [])
            
            formatted_results = []
            for result in organic_results[:num_results]:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "position": result.get("position", 0)
                })
            
            # Include answer box if available
            answer_box = results.get("answer_box", {})
            if answer_box:
                formatted_results.insert(0, {
                    "type": "answer_box",
                    "answer": answer_box.get("answer", answer_box.get("snippet", "")),
                    "source": answer_box.get("link", "")
                })
            
            # Cache the response
            await self.cache.cache_response(
                "serpapi_direct",
                "web_search", 
                {"query": query, "num_results": num_results},
                formatted_results
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Direct web search error: {e}")
            return []
    
    async def news_search(self, query: str, time_range: str = "week") -> List[Dict[str, Any]]:
        """Direct news search with caching"""
        if not self.api_key:
            return []
        
        # Check cache first
        cached_response = await self.cache.get_cached_response(
            "serpapi_direct",
            "news_search",
            {"query": query, "time_range": time_range}
        )
        
        if cached_response is not None:
            logger.info(f"Using cached news search for: {query}")
            return cached_response
        
        try:
            from serpapi import GoogleSearch
            
            time_map = {
                "hour": "qdr:h",
                "day": "qdr:d",
                "week": "qdr:w",
                "month": "qdr:m",
                "year": "qdr:y"
            }
            
            search = GoogleSearch({
                "q": query,
                "api_key": self.api_key,
                "tbm": "nws",
                "tbs": time_map.get(time_range, "qdr:w")
            })
            
            results = search.get_dict()
            news_results = results.get("news_results", [])
            
            formatted_results = []
            for article in news_results[:10]:
                formatted_results.append({
                    "title": article.get("title", ""),
                    "link": article.get("link", ""),
                    "source": article.get("source", ""),
                    "date": article.get("date", ""),
                    "snippet": article.get("snippet", "")
                })
            
            # Cache the response
            await self.cache.cache_response(
                "serpapi_direct",
                "news_search",
                {"query": query, "time_range": time_range},
                formatted_results
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Direct news search error: {e}")
            return []


# Singleton instance
_web_search = None

async def get_web_search() -> DirectWebSearch:
    """Get or create web search instance (using direct API for now)"""
    global _web_search
    if _web_search is None:
        _web_search = DirectWebSearch()
    return _web_search