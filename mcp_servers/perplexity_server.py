#!/usr/bin/env python3
"""
Perplexity AI MCP Server
Provides AI-powered search and research capabilities through MCP
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
server = Server("perplexity-ai")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available Perplexity tools"""
    return [
        Tool(
            name="perplexity_search",
            description="AI-powered search with citations and comprehensive answers",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question"
                    },
                    "search_domain": {
                        "type": "string", 
                        "description": "Domain to focus search: 'general', 'academic', 'news', 'reddit', 'youtube'",
                        "default": "general"
                    },
                    "recency": {
                        "type": "string",
                        "description": "Time filter: 'day', 'week', 'month', 'year', 'all'",
                        "default": "all"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="perplexity_research",
            description="Deep research on a topic with multiple perspectives",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to research"
                    },
                    "depth": {
                        "type": "string",
                        "description": "Research depth: 'quick', 'standard', 'comprehensive'",
                        "default": "standard"
                    },
                    "citations": {
                        "type": "boolean",
                        "description": "Include source citations",
                        "default": True
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="perplexity_compare",
            description="Compare multiple items, concepts, or options",
            inputSchema={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "List of items to compare",
                        "items": {"type": "string"}
                    },
                    "criteria": {
                        "type": "array",
                        "description": "Comparison criteria (optional)",
                        "items": {"type": "string"}
                    }
                },
                "required": ["items"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute a Perplexity tool"""
    api_key = os.getenv('PERPLEXITY_API_KEY', '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')
    
    if not api_key:
        return [TextContent(
            type="text",
            text="Perplexity API key not configured. Please set PERPLEXITY_API_KEY."
        )]
    
    try:
        if name == "perplexity_search":
            return await perplexity_search(api_key, arguments)
        elif name == "perplexity_research":
            return await perplexity_research(api_key, arguments)
        elif name == "perplexity_compare":
            return await perplexity_compare(api_key, arguments)
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


async def perplexity_search(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Perform AI-powered search using Perplexity"""
    query = arguments.get("query", "")
    search_domain = arguments.get("search_domain", "general")
    recency = arguments.get("recency", "all")
    
    # Map domain to Perplexity model
    model_map = {
        "general": "llama-3.1-sonar-large-128k-online",
        "academic": "llama-3.1-sonar-large-128k-online",
        "news": "llama-3.1-sonar-large-128k-online",
        "reddit": "llama-3.1-sonar-large-128k-online",
        "youtube": "llama-3.1-sonar-large-128k-online"
    }
    
    # Add domain prefix to query if needed
    if search_domain != "general":
        query = f"{search_domain}: {query}"
    
    # Add recency filter
    if recency != "all":
        query = f"{query} (past {recency})"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_map.get(search_domain, "llama-3.1-sonar-large-128k-online"),
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful search assistant. Provide comprehensive answers with sources."
                        },
                        {
                            "role": "user",
                            "content": query
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2048,
                    "return_citations": True,
                    "search_domain_filter": [search_domain] if search_domain != "general" else None,
                    "search_recency_filter": recency if recency != "all" else None
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract the response and citations
                answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                citations = data.get("citations", [])
                
                # Format the response
                result = {
                    "answer": answer,
                    "citations": citations,
                    "model": model_map.get(search_domain),
                    "search_domain": search_domain,
                    "recency": recency
                }
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
            else:
                logger.error(f"Perplexity API error: {response.status_code}")
                return [TextContent(
                    type="text",
                    text=f"API error: {response.status_code}"
                )]
                
    except Exception as e:
        logger.error(f"Perplexity search error: {e}")
        return [TextContent(
            type="text",
            text=f"Search error: {str(e)}"
        )]


async def perplexity_research(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Conduct deep research on a topic"""
    topic = arguments.get("topic", "")
    depth = arguments.get("depth", "standard")
    include_citations = arguments.get("citations", True)
    
    # Adjust query based on depth
    queries = []
    if depth == "quick":
        queries = [f"What is {topic}?"]
    elif depth == "comprehensive":
        queries = [
            f"What is {topic}? Provide a detailed explanation.",
            f"What are the key benefits and advantages of {topic}?",
            f"What are the challenges and limitations of {topic}?",
            f"What are recent developments and future trends in {topic}?",
            f"How does {topic} compare to alternatives?"
        ]
    else:  # standard
        queries = [
            f"Explain {topic} in detail.",
            f"What are the pros and cons of {topic}?",
            f"What are the latest developments in {topic}?"
        ]
    
    research_results = []
    
    try:
        async with httpx.AsyncClient() as client:
            for query in queries:
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.1-sonar-large-128k-online",
                        "messages": [
                            {
                                "role": "user",
                                "content": query
                            }
                        ],
                        "temperature": 0.3,
                        "max_tokens": 1024,
                        "return_citations": include_citations
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    citations = data.get("citations", []) if include_citations else []
                    
                    research_results.append({
                        "question": query,
                        "answer": answer,
                        "citations": citations
                    })
                
                # Small delay between requests
                await asyncio.sleep(0.5)
        
        # Compile research
        compiled_research = {
            "topic": topic,
            "depth": depth,
            "sections": research_results,
            "summary": f"Comprehensive research on {topic} covering {len(research_results)} aspects."
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(compiled_research, indent=2)
        )]
        
    except Exception as e:
        logger.error(f"Perplexity research error: {e}")
        return [TextContent(
            type="text",
            text=f"Research error: {str(e)}"
        )]


async def perplexity_compare(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Compare multiple items using Perplexity AI"""
    items = arguments.get("items", [])
    criteria = arguments.get("criteria", [])
    
    if len(items) < 2:
        return [TextContent(
            type="text",
            text="Please provide at least 2 items to compare"
        )]
    
    # Build comparison query
    comparison_query = f"Compare {' vs '.join(items)}"
    if criteria:
        comparison_query += f" based on: {', '.join(criteria)}"
    else:
        comparison_query += " across all relevant dimensions"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-sonar-large-128k-online",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert analyst. Provide detailed, balanced comparisons with pros and cons for each option."
                        },
                        {
                            "role": "user",
                            "content": comparison_query
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2048,
                    "return_citations": True
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                citations = data.get("citations", [])
                
                result = {
                    "items": items,
                    "criteria": criteria if criteria else "all aspects",
                    "comparison": answer,
                    "citations": citations
                }
                
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
        logger.error(f"Perplexity compare error: {e}")
        return [TextContent(
            type="text",
            text=f"Comparison error: {str(e)}"
        )]


async def main():
    """Run the MCP server"""
    from mcp.server.stdio import stdio_server
    
    logger.info("Starting Perplexity MCP server...")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())