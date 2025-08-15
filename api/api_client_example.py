#!/usr/bin/env python3
"""
Example client for Agent Core API
Shows how to interact with the API using different authentication methods
"""

import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any


class AgentCoreClient:
    """Client for Agent Core API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session_id = None
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def chat(self, message: str, session_id: Optional[str] = None, agent_type: str = "natural") -> Dict[str, Any]:
        """Send a chat message"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "message": message,
                "agent_type": agent_type
            }
            if session_id or self.session_id:
                payload["session_id"] = session_id or self.session_id
            
            async with session.post(
                f"{self.base_url}/chat",
                headers=self.headers,
                json=payload
            ) as response:
                result = await response.json()
                
                # Store session ID for continuity
                if "session_id" in result:
                    self.session_id = result["session_id"]
                
                return result
    
    async def research(self, topic: str, max_results: int = 10) -> Dict[str, Any]:
        """Research a topic"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/research",
                headers=self.headers,
                json={
                    "topic": topic,
                    "max_results": max_results,
                    "sources": ["arxiv", "scholar", "web"]
                }
            ) as response:
                return await response.json()
    
    async def list_sessions(self) -> list:
        """List all active sessions"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/sessions",
                headers=self.headers
            ) as response:
                return await response.json()
    
    async def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session"""
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{self.base_url}/sessions/{session_id}",
                headers=self.headers
            ) as response:
                return await response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                return await response.json()
    
    async def openai_compatible_chat(self, messages: list, model: str = "agent-core") -> Dict[str, Any]:
        """OpenAI-compatible chat completions"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/completions",
                headers=self.headers,
                json={
                    "model": model,
                    "messages": messages,
                    "user": self.session_id or "default"
                }
            ) as response:
                return await response.json()


class WebSocketClient:
    """WebSocket client for real-time chat"""
    
    def __init__(self, base_url: str = "http://localhost:8000", session_id: str = "default"):
        self.base_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.session_id = session_id
    
    async def connect_and_chat(self):
        """Connect to WebSocket and start chat"""
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"{self.base_url}/ws/{self.session_id}") as ws:
                print(f"Connected to WebSocket (session: {self.session_id})")
                
                # Start listener task
                listener_task = asyncio.create_task(self.listen_messages(ws))
                
                # Send messages
                try:
                    while True:
                        message = input("\nYou: ")
                        if message.lower() in ['exit', 'quit']:
                            break
                        
                        await ws.send_json({
                            "message": message,
                            "agent_type": "natural"
                        })
                
                except KeyboardInterrupt:
                    pass
                finally:
                    listener_task.cancel()
                    await ws.close()
    
    async def listen_messages(self, ws):
        """Listen for messages from server"""
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    print(f"\nAgent: {data.get('response', 'No response')}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"WebSocket error: {ws.exception()}")
        except asyncio.CancelledError:
            pass


# Example usage
async def main():
    """Example usage of the API client"""
    
    # Initialize client with API key
    # In production, use: api_key = "sk-user-..."
    client = AgentCoreClient(api_key="test-api-key")
    
    print("Agent Core API Client Examples")
    print("=" * 50)
    
    # 1. Health check
    print("\n1. Health Check:")
    health = await client.health_check()
    print(f"   Status: {health.get('status')}")
    print(f"   Version: {health.get('version')}")
    
    # 2. Simple chat
    print("\n2. Simple Chat:")
    response = await client.chat("Hello! How are you today?")
    print(f"   Response: {response.get('response')}")
    print(f"   Session ID: {response.get('session_id')}")
    
    # 3. Continued conversation
    print("\n3. Continued Conversation:")
    response = await client.chat("What did I just ask you?")
    print(f"   Response: {response.get('response')}")
    
    # 4. Research (if API key is valid)
    try:
        print("\n4. Research:")
        research = await client.research("quantum computing", max_results=3)
        print(f"   Topic: {research.get('topic')}")
        print(f"   Results: {len(research.get('results', []))} items")
    except Exception as e:
        print(f"   Research requires valid API key: {e}")
    
    # 5. OpenAI-compatible format
    print("\n5. OpenAI-Compatible Chat:")
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = await client.openai_compatible_chat(messages)
    if "choices" in response:
        print(f"   Response: {response['choices'][0]['message']['content']}")
    
    # 6. List sessions
    try:
        print("\n6. Active Sessions:")
        sessions = await client.list_sessions()
        print(f"   Total sessions: {len(sessions)}")
    except Exception as e:
        print(f"   Session listing requires valid API key: {e}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")


async def websocket_example():
    """Example WebSocket usage"""
    print("\nWebSocket Chat Example")
    print("=" * 50)
    print("Connecting to WebSocket...")
    
    ws_client = WebSocketClient(session_id="demo-session")
    await ws_client.connect_and_chat()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "websocket":
        # Run WebSocket example
        asyncio.run(websocket_example())
    else:
        # Run REST API examples
        asyncio.run(main())