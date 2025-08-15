#!/usr/bin/env python3
"""
FastAPI application for Agent Core
Provides OpenAPI-compatible REST API and WebSocket endpoints
"""

import os
import asyncio
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Set up environment
os.environ.setdefault('REDIS_HOST', 'localhost')

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our agents
from agents.natural_agent import NaturalAgent
from agents.simple_agent import SimpleAgent
from agents.fast_agent_chat import FastChatAgent
from agents.react_orchestrator import get_orchestrator
from utils.config_loader import config

# Security
security = HTTPBearer()

# Store active sessions
active_sessions: Dict[str, Any] = {}
active_agents: Dict[str, Any] = {}


# Pydantic models for API
class ChatMessage(BaseModel):
    """Chat message model"""
    message: str = Field(..., description="User message to send to agent")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    agent_type: Optional[str] = Field("orchestrator", description="Agent type: orchestrator, natural, simple, or fast")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello, how are you?",
                "session_id": "user-123-session-456",
                "agent_type": "natural"
            }
        }


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="Agent's response")
    session_id: str = Field(..., description="Session ID for conversation continuity")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "I'm doing well, thank you! How can I help you today?",
                "session_id": "user-123-session-456",
                "timestamp": "2024-01-15T10:30:00",
                "metadata": {"agent_type": "natural", "response_time_ms": 250}
            }
        }


class SessionInfo(BaseModel):
    """Session information model"""
    session_id: str
    agent_type: str
    created_at: datetime
    last_activity: datetime
    message_count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user-123-session-456",
                "agent_type": "natural",
                "created_at": "2024-01-15T10:00:00",
                "last_activity": "2024-01-15T10:30:00",
                "message_count": 5
            }
        }


class ResearchRequest(BaseModel):
    """Research request model"""
    topic: str = Field(..., description="Topic to research")
    max_results: Optional[int] = Field(10, description="Maximum number of results")
    sources: Optional[List[str]] = Field(["arxiv", "scholar", "web"], description="Sources to search")
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "quantum computing applications",
                "max_results": 10,
                "sources": ["arxiv", "scholar"]
            }
        }


class HealthStatus(BaseModel):
    """Health check status"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Uptime in seconds")
    active_sessions: int = Field(..., description="Number of active sessions")


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    print("Starting Agent Core API...")
    app.state.start_time = datetime.now()
    
    # Initialize default agent
    try:
        app.state.default_agent = NaturalAgent()
        print("Default agent initialized")
    except Exception as e:
        print(f"Warning: Could not initialize default agent: {e}")
        app.state.default_agent = None
    
    yield
    
    # Shutdown
    print("Shutting down Agent Core API...")
    # Clean up sessions
    active_sessions.clear()
    active_agents.clear()


# Create FastAPI app with OpenAPI documentation
app = FastAPI(
    title="Agent Core API",
    description="OpenAPI-compatible API for AI Agent with natural conversation, research, and memory capabilities",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "http://localhost:3000", "http://127.0.0.1:8001", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify bearer token"""
    token = credentials.credentials
    
    # Simple token validation - replace with proper auth in production
    if not token or len(token) < 10:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token


# Helper functions
async def get_or_create_agent(session_id: str, agent_type: str = "orchestrator") -> Any:
    """Get existing agent or create new one for session"""
    if session_id not in active_agents:
        if agent_type == "orchestrator":
            agent = await get_orchestrator()
        elif agent_type == "simple":
            agent = SimpleAgent()
        elif agent_type == "fast":
            agent = FastChatAgent()
        else:
            agent = NaturalAgent()
        
        active_agents[session_id] = agent
        active_sessions[session_id] = {
            "agent_type": agent_type,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "message_count": 0
        }
    
    return active_agents[session_id]


# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Agent Core API",
        "version": "1.0.0",
        "documentation": "/docs",
        "openapi_spec": "/openapi.json"
    }


@app.get("/health", response_model=HealthStatus, tags=["General"])
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - app.state.start_time).total_seconds()
    
    return HealthStatus(
        status="healthy",
        version="1.0.0",
        uptime=uptime,
        active_sessions=len(active_sessions)
    )


@app.post("/chat/public", response_model=ChatResponse, tags=["Chat"])
async def chat_public(request: ChatMessage):
    """
    Public chat endpoint (no authentication required).
    For web interface and testing.
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get or create agent (defaults to orchestrator)
        agent = await get_or_create_agent(session_id, request.agent_type or "orchestrator")
        
        # Process message
        start_time = datetime.now()
        response = await agent.process_message(request.message)
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update session
        if session_id in active_sessions:
            active_sessions[session_id]["last_activity"] = datetime.now()
            active_sessions[session_id]["message_count"] += 1
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            metadata={
                "agent_type": request.agent_type or "orchestrator",
                "response_time_ms": round(response_time, 2)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatMessage,
    token: str = Depends(verify_token)
):
    """
    Send a message to the AI agent and receive a response.
    
    The agent maintains conversation context within a session.
    Supports different agent types: natural, simple, or fast.
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get or create agent
        agent = await get_or_create_agent(session_id, request.agent_type)
        
        # Process message
        start_time = datetime.now()
        response = await agent.process_message(request.message)
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update session
        if session_id in active_sessions:
            active_sessions[session_id]["last_activity"] = datetime.now()
            active_sessions[session_id]["message_count"] += 1
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            metadata={
                "agent_type": request.agent_type,
                "response_time_ms": round(response_time, 2)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions", response_model=List[SessionInfo], tags=["Sessions"])
async def list_sessions(token: str = Depends(verify_token)):
    """List all active sessions"""
    sessions = []
    for sid, info in active_sessions.items():
        sessions.append(SessionInfo(
            session_id=sid,
            agent_type=info["agent_type"],
            created_at=info["created_at"],
            last_activity=info["last_activity"],
            message_count=info["message_count"]
        ))
    return sessions


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def delete_session(
    session_id: str,
    token: str = Depends(verify_token)
):
    """Delete a specific session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del active_sessions[session_id]
    if session_id in active_agents:
        del active_agents[session_id]
    
    return {"message": "Session deleted successfully"}


@app.post("/research", tags=["Research"])
async def research(
    request: ResearchRequest,
    token: str = Depends(verify_token)
):
    """
    Initiate research on a specific topic.
    
    This endpoint triggers background research using multiple sources.
    """
    try:
        # Create a research agent
        from knowledge.research_system import ResearchSystem
        research_system = ResearchSystem()
        
        # Perform research
        results = await research_system.research(
            request.topic,
            max_results=request.max_results
        )
        
        return {
            "topic": request.topic,
            "results": results,
            "sources_searched": request.sources,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat with streaming responses.
    
    Send messages as JSON: {"message": "your message", "agent_type": "natural"}
    """
    await websocket.accept()
    
    # Get or create agent
    agent = get_or_create_agent(session_id, "natural")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            if not message:
                continue
            
            # Process message
            response = await agent.process_message(message)
            
            # Send response
            await websocket.send_json({
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            })
            
            # Update session
            if session_id in active_sessions:
                active_sessions[session_id]["last_activity"] = datetime.now()
                active_sessions[session_id]["message_count"] += 1
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


@app.post("/completions", tags=["OpenAI Compatible"])
async def openai_compatible_completions(
    request: Dict[str, Any],
    token: str = Depends(verify_token)
):
    """
    OpenAI-compatible completions endpoint.
    
    Accepts the same format as OpenAI's chat completions API.
    """
    try:
        # Extract messages
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="Messages required")
        
        # Get last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Process with agent
        session_id = request.get("user", str(uuid.uuid4()))
        agent = get_or_create_agent(session_id, "natural")
        response = await agent.process_message(user_message)
        
        # Return OpenAI-compatible response
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "agent-core-natural",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(user_message.split()) + len(response.split())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Main entry point
if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )