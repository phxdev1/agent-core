#!/usr/bin/env python3
"""
Tavily AI Search MCP Server
Provides AI-optimized search capabilities through MCP
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import Server, Tool
from mcp.types import TextContent
from utils.redis_logger import get_redis_logger

logger = get_redis_logger(__name__)

# Create MCP server instance
server = Server("tavily-search")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available Tavily tools"""
    return [
        Tool(
            name="tavily_search",
            description="AI-optimized search with real-time web results",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "search_depth": {
                        "type": "string",
                        "description": "Search depth: 'basic' or 'advanced'",
                        "default": "basic"
                    },
                    "include_answer": {
                        "type": "boolean",
                        "description": "Include AI-generated answer",
                        "default": True
                    },
                    "include_raw_content": {
                        "type": "boolean",
                        "description": "Include raw page content",
                        "default": False
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="tavily_qna",
            description="Direct question answering with sources",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer"
                    },
                    "search_depth": {
                        "type": "string",
                        "description": "Search depth: 'basic' or 'advanced'",
                        "default": "advanced"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="tavily_context",
            description="Get comprehensive context about a topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to get context about"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens in context",
                        "default": 4000
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="tavily_extract",
            description="Extract specific information from URLs",
            inputSchema={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "description": "List of URLs to extract from",
                        "items": {"type": "string"}
                    },
                    "prompt": {
                        "type": "string",
                        "description": "What information to extract"
                    }
                },
                "required": ["urls", "prompt"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute a Tavily tool"""
    api_key = os.getenv('TAVILY_API_KEY', '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')
    
    if not api_key:
        return [TextContent(
            type="text",
            text="Tavily API key not configured. Please set TAVILY_API_KEY."
        )]
    
    try:
        if name == "tavily_search":
            return await tavily_search(api_key, arguments)
        elif name == "tavily_qna":
            return await tavily_qna(api_key, arguments)
        elif name == "tavily_context":
            return await tavily_context(api_key, arguments)
        elif name == "tavily_extract":
            return await tavily_extract(api_key, arguments)
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def tavily_search(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Perform AI-optimized search using Tavily"""
    query = arguments.get("query", "")
    search_depth = arguments.get("search_depth", "basic")
    include_answer = arguments.get("include_answer", True)
    include_raw_content = arguments.get("include_raw_content", False)
    max_results = arguments.get("max_results", 5)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                headers={
                    "Content-Type": "application/json"
                },
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": search_depth,
                    "include_answer": include_answer,
                    "include_raw_content": include_raw_content,
                    "max_results": max_results,
                    "include_domains": [],
                    "exclude_domains": []
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Format results
                results = {
                    "query": query,
                    "answer": data.get("answer", "") if include_answer else None,
                    "results": []
                }
                
                for result in data.get("results", [])[:max_results]:
                    formatted_result = {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "score": result.get("score", 0),
                        "published_date": result.get("published_date", "")
                    }
                    
                    if include_raw_content:
                        formatted_result["raw_content"] = result.get("raw_content", "")
                    
                    results["results"].append(formatted_result)
                
                # Add query understanding
                results["query_understanding"] = {
                    "search_depth": search_depth,
                    "total_results": len(data.get("results", [])),
                    "response_time": data.get("response_time", 0)
                }
                
                return [TextContent(
                    type="text",
                    text=json.dumps(results, indent=2)
                )]
            else:
                logger.error(f"Tavily API error: {response.status_code}")
                return [TextContent(
                    type="text",
                    text=f"API error: {response.status_code}"
                )]
                
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return [TextContent(
            type="text",
            text=f"Search error: {str(e)}"
        )]


async def tavily_qna(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Direct question answering using Tavily"""
    question = arguments.get("question", "")
    search_depth = arguments.get("search_depth", "advanced")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/qna",
                headers={
                    "Content-Type": "application/json"
                },
                json={
                    "api_key": api_key,
                    "query": question,
                    "search_depth": search_depth,
                    "include_citations": True,
                    "max_results": 10
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                result = {
                    "question": question,
                    "answer": data.get("answer", ""),
                    "confidence": data.get("confidence_score", 0),
                    "sources": []
                }
                
                # Extract sources
                for source in data.get("sources", []):
                    result["sources"].append({
                        "title": source.get("title", ""),
                        "url": source.get("url", ""),
                        "snippet": source.get("snippet", "")
                    })
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"API error: {response.status_code}"
                )]
                
    except Exception as e:
        logger.error(f"Tavily QnA error: {e}")
        return [TextContent(
            type="text",
            text=f"QnA error: {str(e)}"
        )]


async def tavily_context(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Get comprehensive context about a topic"""
    topic = arguments.get("topic", "")
    max_tokens = arguments.get("max_tokens", 4000)
    
    try:
        async with httpx.AsyncClient() as client:
            # Use search with advanced depth to get comprehensive context
            response = await client.post(
                "https://api.tavily.com/search",
                headers={
                    "Content-Type": "application/json"
                },
                json={
                    "api_key": api_key,
                    "query": f"comprehensive overview of {topic}",
                    "search_depth": "advanced",
                    "include_answer": True,
                    "include_raw_content": True,
                    "max_results": 10,
                    "max_tokens": max_tokens
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Build comprehensive context
                context = {
                    "topic": topic,
                    "overview": data.get("answer", ""),
                    "key_points": [],
                    "sources": [],
                    "related_topics": []
                }
                
                # Extract key points from results
                for result in data.get("results", []):
                    if result.get("content"):
                        # Extract first few sentences as key points
                        sentences = result["content"].split(". ")[:2]
                        key_point = ". ".join(sentences)
                        if key_point and key_point not in context["key_points"]:
                            context["key_points"].append(key_point)
                    
                    # Add source
                    context["sources"].append({
                        "title": result.get("title", ""),
                        "url": result.get("url", "")
                    })
                
                # Extract related topics from search
                if "related_searches" in data:
                    context["related_topics"] = data["related_searches"][:5]
                
                return [TextContent(
                    type="text",
                    text=json.dumps(context, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"API error: {response.status_code}"
                )]
                
    except Exception as e:
        logger.error(f"Tavily context error: {e}")
        return [TextContent(
            type="text",
            text=f"Context error: {str(e)}"
        )]


async def tavily_extract(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Extract specific information from URLs"""
    urls = arguments.get("urls", [])
    prompt = arguments.get("prompt", "")
    
    if not urls:
        return [TextContent(
            type="text",
            text="Please provide URLs to extract from"
        )]
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/extract",
                headers={
                    "Content-Type": "application/json"
                },
                json={
                    "api_key": api_key,
                    "urls": urls[:5],  # Limit to 5 URLs
                    "prompt": prompt
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                result = {
                    "prompt": prompt,
                    "extractions": []
                }
                
                for extraction in data.get("results", []):
                    result["extractions"].append({
                        "url": extraction.get("url", ""),
                        "extracted_content": extraction.get("extracted_content", ""),
                        "success": extraction.get("success", False)
                    })
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"API error: {response.status_code}"
                )]
                
    except Exception as e:
        logger.error(f"Tavily extract error: {e}")
        return [TextContent(
            type="text",
            text=f"Extract error: {str(e)}"
        )]


async def main():
    """Run the MCP server"""
    from mcp.server.stdio import stdio_server
    
    logger.info("Starting Tavily MCP server...")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())