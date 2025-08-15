#!/usr/bin/env python3
"""
Simple FastAPI for testing deployment
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional

# Simple FastAPI app
app = FastAPI(
    title="Agent Core API",
    description="Simple API for testing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Agent Core API",
        "version": "1.0.0",
        "status": "running",
        "message": "Simple API is working!"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "agent-core-api"}

@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {
        "name": "Agent Core API", 
        "version": "1.0.0",
        "documentation": "/docs",
        "openapi_spec": "/openapi.json"
    }

@app.get("/api/health")
async def api_health():
    """API health check endpoint"""
    return {"status": "healthy", "service": "agent-core-api"}

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    session_id: Optional[str] = None
    agent_type: Optional[str] = "orchestrator"

@app.post("/api/chat/public")
async def chat_public_api(request: ChatRequest):
    """Public chat endpoint for testing (with /api prefix)"""
    # Simple echo response for now
    return {
        "response": f"Hello! You said: {request.message}. I'm a simple test API - the full agent system will be integrated soon.",
        "session_id": request.session_id or "test-session",
        "metadata": {
            "agent_type": request.agent_type,
            "response_time_ms": 100
        }
    }

@app.post("/chat/public")
async def chat_public(request: ChatRequest):
    """Public chat endpoint for testing (without /api prefix)"""
    # Simple echo response for now
    return {
        "response": f"Hello! You said: {request.message}. I'm a simple test API - the full agent system will be integrated soon.",
        "session_id": request.session_id or "test-session",
        "metadata": {
            "agent_type": request.agent_type,
            "response_time_ms": 100
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    # Listen on all interfaces including IPv6 for Fly.io internal networking
    uvicorn.run(app, host="::", port=port, log_level="info")