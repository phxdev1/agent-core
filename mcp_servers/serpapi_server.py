#!/usr/bin/env python3
"""
SERPAPI MCP Server
Provides web search capabilities through the Model Context Protocol
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import Server, Tool
from mcp.types import TextContent, ImageContent, EmbeddedResource
from serpapi import GoogleSearch
from utils.config_loader import config
from utils.redis_logger import get_redis_logger

logger = get_redis_logger(__name__)

# Create MCP server instance
server = Server("serpapi-search")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available search tools"""
    return [
        Tool(
            name="web_search",
            description="Search the web using Google via SERPAPI",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="news_search",
            description="Search for recent news articles",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The news search query"
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Time range: 'hour', 'day', 'week', 'month', 'year'",
                        "default": "week"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="image_search",
            description="Search for images on the web",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The image search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="scholar_search",
            description="Search academic papers and scholarly articles",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The academic search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="local_search",
            description="Search for local businesses and places",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (e.g., 'coffee shops')"
                    },
                    "location": {
                        "type": "string",
                        "description": "Location to search in (e.g., 'New York, NY')"
                    }
                },
                "required": ["query", "location"]
            }
        ),
        Tool(
            name="events_search",
            description="Search for upcoming events and activities",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Type of event or activity"
                    },
                    "location": {
                        "type": "string",
                        "description": "Location for events (e.g., 'San Francisco, CA')"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="maps_search",
            description="Search for directions and location information",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Place name or address to search for"
                    },
                    "search_type": {
                        "type": "string",
                        "description": "Type: 'place', 'directions', 'distance'",
                        "default": "place"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="shopping_search",
            description="Search for products on Amazon, Walmart, or Home Depot",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Product to search for"
                    },
                    "store": {
                        "type": "string",
                        "description": "Store to search: 'amazon', 'walmart', 'home_depot'",
                        "default": "amazon"
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Sort by: 'relevance', 'price_low', 'price_high', 'rating'",
                        "default": "relevance"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="ai_overview",
            description="Get an AI-generated overview/summary of a topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to get an overview about"
                    },
                    "detail_level": {
                        "type": "string",
                        "description": "Level of detail: 'brief', 'standard', 'detailed'",
                        "default": "standard"
                    }
                },
                "required": ["topic"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute a search tool"""
    api_key = config.get('serpapi_key')
    if not api_key:
        return [TextContent(
            type="text",
            text="SERPAPI key not configured. Please set SERPAPI_KEY in your environment."
        )]
    
    try:
        if name == "web_search":
            return await web_search(api_key, arguments)
        elif name == "news_search":
            return await news_search(api_key, arguments)
        elif name == "image_search":
            return await image_search(api_key, arguments)
        elif name == "scholar_search":
            return await scholar_search(api_key, arguments)
        elif name == "local_search":
            return await local_search(api_key, arguments)
        elif name == "events_search":
            return await events_search(api_key, arguments)
        elif name == "maps_search":
            return await maps_search(api_key, arguments)
        elif name == "shopping_search":
            return await shopping_search(api_key, arguments)
        elif name == "ai_overview":
            return await ai_overview(api_key, arguments)
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


async def web_search(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Perform a web search"""
    query = arguments.get("query", "")
    num_results = arguments.get("num_results", 5)
    
    search = GoogleSearch({
        "q": query,
        "api_key": api_key,
        "num": num_results
    })
    
    results = search.get_dict()
    
    # Format organic results
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
    
    # Include knowledge graph if available
    knowledge_graph = results.get("knowledge_graph", {})
    if knowledge_graph:
        formatted_results.insert(0, {
            "type": "knowledge_graph",
            "title": knowledge_graph.get("title", ""),
            "description": knowledge_graph.get("description", ""),
            "source": knowledge_graph.get("source", {}).get("link", "")
        })
    
    return [TextContent(
        type="text",
        text=json.dumps(formatted_results, indent=2)
    )]


async def news_search(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Search for news articles"""
    query = arguments.get("query", "")
    time_range = arguments.get("time_range", "week")
    
    # Map time range to SERPAPI parameter
    time_map = {
        "hour": "qdr:h",
        "day": "qdr:d",
        "week": "qdr:w",
        "month": "qdr:m",
        "year": "qdr:y"
    }
    
    search = GoogleSearch({
        "q": query,
        "api_key": api_key,
        "tbm": "nws",  # News search
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
    
    return [TextContent(
        type="text",
        text=json.dumps(formatted_results, indent=2)
    )]


async def image_search(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Search for images"""
    query = arguments.get("query", "")
    num_results = arguments.get("num_results", 5)
    
    search = GoogleSearch({
        "q": query,
        "api_key": api_key,
        "tbm": "isch",  # Image search
        "num": num_results
    })
    
    results = search.get_dict()
    images = results.get("images_results", [])
    
    formatted_results = []
    for image in images[:num_results]:
        formatted_results.append({
            "title": image.get("title", ""),
            "link": image.get("link", ""),
            "source": image.get("source", ""),
            "thumbnail": image.get("thumbnail", ""),
            "original": image.get("original", "")
        })
    
    return [TextContent(
        type="text",
        text=json.dumps(formatted_results, indent=2)
    )]


async def scholar_search(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Search for academic papers"""
    query = arguments.get("query", "")
    num_results = arguments.get("num_results", 5)
    
    from serpapi import GoogleScholarSearch
    
    search = GoogleScholarSearch({
        "q": query,
        "api_key": api_key,
        "num": num_results
    })
    
    results = search.get_dict()
    organic_results = results.get("organic_results", [])
    
    formatted_results = []
    for paper in organic_results[:num_results]:
        formatted_results.append({
            "title": paper.get("title", ""),
            "link": paper.get("link", ""),
            "snippet": paper.get("snippet", ""),
            "authors": paper.get("publication_info", {}).get("authors", []),
            "cited_by": paper.get("inline_links", {}).get("cited_by", {}).get("total", 0),
            "year": paper.get("publication_info", {}).get("summary", "")
        })
    
    return [TextContent(
        type="text",
        text=json.dumps(formatted_results, indent=2)
    )]


async def local_search(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Search for local businesses"""
    query = arguments.get("query", "")
    location = arguments.get("location", "")
    
    search = GoogleSearch({
        "q": f"{query} near {location}",
        "api_key": api_key,
        "tbm": "lcl"  # Local search
    })
    
    results = search.get_dict()
    local_results = results.get("local_results", [])
    
    formatted_results = []
    for place in local_results[:10]:
        formatted_results.append({
            "title": place.get("title", ""),
            "address": place.get("address", ""),
            "rating": place.get("rating", ""),
            "reviews": place.get("reviews", 0),
            "type": place.get("type", ""),
            "hours": place.get("hours", ""),
            "phone": place.get("phone", "")
        })
    
    return [TextContent(
        type="text",
        text=json.dumps(formatted_results, indent=2)
    )]


async def events_search(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Search for events"""
    query = arguments.get("query", "")
    location = arguments.get("location", "")
    
    search_query = f"{query} events"
    if location:
        search_query += f" in {location}"
    
    search = GoogleSearch({
        "q": search_query,
        "api_key": api_key,
        "tbm": "evts"  # Events search (if supported)
    })
    
    results = search.get_dict()
    
    # Try to extract event information from regular search
    organic_results = results.get("organic_results", [])
    events_results = results.get("events_results", organic_results)
    
    formatted_results = []
    for event in events_results[:10]:
        formatted_results.append({
            "title": event.get("title", ""),
            "date": event.get("date", event.get("snippet", "")),
            "location": event.get("address", event.get("location", "")),
            "link": event.get("link", ""),
            "description": event.get("snippet", "")
        })
    
    return [TextContent(
        type="text",
        text=json.dumps(formatted_results, indent=2)
    )]


async def maps_search(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Search for maps and location information"""
    query = arguments.get("query", "")
    search_type = arguments.get("search_type", "place")
    
    # Use Google Maps via SERPAPI
    from serpapi import GoogleMapsSearch
    
    search = GoogleMapsSearch({
        "q": query,
        "api_key": api_key,
        "type": "search"
    })
    
    results = search.get_dict()
    local_results = results.get("local_results", [])
    
    formatted_results = []
    for place in local_results[:5]:
        result = {
            "name": place.get("title", ""),
            "address": place.get("address", ""),
            "gps_coordinates": place.get("gps_coordinates", {}),
            "rating": place.get("rating", ""),
            "reviews": place.get("reviews", ""),
            "type": place.get("type", ""),
            "hours": place.get("hours", ""),
            "phone": place.get("phone", "")
        }
        
        # Add distance if available
        if search_type == "distance" and place.get("distance"):
            result["distance"] = place.get("distance")
        
        formatted_results.append(result)
    
    # Add place info if available
    place_results = results.get("place_results", {})
    if place_results:
        formatted_results.insert(0, {
            "type": "place_details",
            "name": place_results.get("title", ""),
            "address": place_results.get("address", ""),
            "gps_coordinates": place_results.get("gps_coordinates", {}),
            "description": place_results.get("description", "")
        })
    
    return [TextContent(
        type="text",
        text=json.dumps(formatted_results, indent=2)
    )]


async def shopping_search(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Search for products on shopping sites"""
    query = arguments.get("query", "")
    store = arguments.get("store", "amazon").lower()
    sort_by = arguments.get("sort_by", "relevance")
    
    try:
        if store == "amazon":
            from serpapi import AmazonSearch
            
            params = {
                "query": query,
                "api_key": api_key
            }
            
            # Add sorting
            if sort_by == "price_low":
                params["s"] = "price-asc-rank"
            elif sort_by == "price_high":
                params["s"] = "price-desc-rank"
            elif sort_by == "rating":
                params["s"] = "review-rank"
            
            search = AmazonSearch(params)
            
        elif store == "walmart":
            from serpapi import WalmartSearch
            
            params = {
                "query": query,
                "api_key": api_key
            }
            
            # Add sorting
            if sort_by == "price_low":
                params["sort"] = "price_low"
            elif sort_by == "price_high":
                params["sort"] = "price_high"
            elif sort_by == "rating":
                params["sort"] = "best_seller"
            
            search = WalmartSearch(params)
            
        elif store == "home_depot":
            from serpapi import HomeDepotSearch
            
            params = {
                "q": query,
                "api_key": api_key
            }
            
            # Add sorting  
            if sort_by == "price_low":
                params["ps"] = "1"  # Price low to high
            elif sort_by == "price_high":
                params["ps"] = "2"  # Price high to low
            elif sort_by == "rating":
                params["ps"] = "3"  # Top rated
            
            search = HomeDepotSearch(params)
            
        else:
            return [TextContent(
                type="text",
                text=f"Unknown store: {store}. Supported: amazon, walmart, home_depot"
            )]
        
        results = search.get_dict()
        
        # Extract product results
        if store == "amazon":
            products = results.get("organic_results", [])
        elif store == "walmart":
            products = results.get("organic_results", [])
        else:  # home_depot
            products = results.get("products", [])
        
        formatted_results = []
        for product in products[:10]:
            result = {
                "title": product.get("title", ""),
                "price": product.get("price", product.get("primary_offer", {}).get("offer_price", "")),
                "rating": product.get("rating", ""),
                "reviews_count": product.get("reviews_count", product.get("ratings_total", "")),
                "link": product.get("link", product.get("product_link", "")),
                "image": product.get("thumbnail", product.get("image", "")),
                "availability": product.get("availability", {}).get("raw", "In stock"),
                "store": store.title()
            }
            
            # Add store-specific fields
            if store == "amazon":
                result["prime"] = product.get("is_prime", False)
                result["asin"] = product.get("asin", "")
            elif store == "walmart":
                result["seller"] = product.get("seller_name", "Walmart")
                result["two_day_shipping"] = product.get("two_day_shipping", False)
            elif store == "home_depot":
                result["model"] = product.get("model", "")
                result["brand"] = product.get("brand", "")
            
            # Include full description if available
            result["description"] = product.get("snippet", product.get("description", ""))
            
            formatted_results.append(result)
        
        return [TextContent(
            type="text",
            text=json.dumps(formatted_results, indent=2)
        )]
        
    except ImportError as e:
        return [TextContent(
            type="text",
            text=f"Store API not available: {e}. Using general search instead."
        )]
    except Exception as e:
        logger.error(f"Shopping search error: {e}")
        # Fallback to general search
        search = GoogleSearch({
            "q": f"{query} site:{store}.com",
            "api_key": api_key,
            "num": 10
        })
        
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        
        formatted_results = []
        for result in organic_results[:10]:
            formatted_results.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "store": store.title()
            })
        
        return [TextContent(
            type="text",
            text=json.dumps(formatted_results, indent=2)
        )]


async def ai_overview(api_key: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Get AI-generated overview of a topic"""
    topic = arguments.get("topic", "")
    detail_level = arguments.get("detail_level", "standard")
    
    # Search for comprehensive information
    search = GoogleSearch({
        "q": topic,
        "api_key": api_key,
        "num": 10
    })
    
    results = search.get_dict()
    
    # Collect various information sources
    overview_data = {
        "topic": topic,
        "summary": "",
        "key_facts": [],
        "sources": []
    }
    
    # Extract answer box
    answer_box = results.get("answer_box", {})
    if answer_box:
        overview_data["summary"] = answer_box.get("answer", answer_box.get("snippet", ""))
        overview_data["sources"].append(answer_box.get("link", ""))
    
    # Extract knowledge graph
    knowledge_graph = results.get("knowledge_graph", {})
    if knowledge_graph:
        if not overview_data["summary"]:
            overview_data["summary"] = knowledge_graph.get("description", "")
        
        # Extract key facts from knowledge graph
        for key, value in knowledge_graph.items():
            if key not in ["title", "description", "source", "see_results_about"]:
                if isinstance(value, (str, int, float)):
                    overview_data["key_facts"].append(f"{key}: {value}")
    
    # Extract key information from top results
    organic_results = results.get("organic_results", [])
    for result in organic_results[:3]:
        snippet = result.get("snippet", "")
        if snippet and len(snippet) > 50:
            if detail_level in ["standard", "detailed"]:
                overview_data["key_facts"].append(snippet)
        overview_data["sources"].append(result.get("link", ""))
    
    # Add related searches for context
    if detail_level == "detailed":
        related_searches = results.get("related_searches", [])
        if related_searches:
            overview_data["related_topics"] = [rs.get("query", "") for rs in related_searches[:5]]
    
    # Format based on detail level
    if detail_level == "brief":
        # Just summary and top 3 facts
        overview_data["key_facts"] = overview_data["key_facts"][:3]
    elif detail_level == "detailed":
        # Include everything
        pass
    else:  # standard
        # Summary and top 5 facts
        overview_data["key_facts"] = overview_data["key_facts"][:5]
    
    return [TextContent(
        type="text",
        text=json.dumps(overview_data, indent=2)
    )]


async def main():
    """Run the MCP server"""
    from mcp.server.stdio import stdio_server
    
    logger.info("Starting SERPAPI MCP server...")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())