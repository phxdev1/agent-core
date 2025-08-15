#!/usr/bin/env python3
"""
Minimal API with basic agent functionality
"""

import os
import asyncio
import uuid
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI
app = FastAPI(
    title="Agent Core API",
    description="Minimal AI Agent API",
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

# Simple in-memory session storage
sessions = {}

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    session_id: Optional[str] = None
    agent_type: Optional[str] = "simple"

class SimpleAgent:
    """Very simple agent that provides basic responses"""
    
    def __init__(self):
        self.responses = {
            "hello": "Hello! I'm Agent Core, your AI assistant. How can I help you today?",
            "hi": "Hi there! What can I do for you?",
            "help": "I can help you with various tasks. Try asking me questions or giving me commands!",
            "how are you": "I'm functioning well, thank you! Ready to assist you.",
            "what can you do": "I can answer questions, help with tasks, and have conversations. I'm constantly learning!",
            "bye": "Goodbye! Feel free to come back anytime you need assistance.",
        }
    
    async def process_message(self, message: str, session_id: str) -> str:
        """Process a message and return a response"""
        # Store message in session
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append({"user": message, "timestamp": datetime.now()})
        
        # Simple keyword matching
        message_lower = message.lower()
        
        # Check for exact matches
        for key, response in self.responses.items():
            if key in message_lower:
                sessions[session_id].append({"assistant": response, "timestamp": datetime.now()})
                return response
        
        # Default response with echo
        response = f"I understand you said: '{message}'. I'm a minimal agent system. The full AI capabilities will be integrated soon!"
        sessions[session_id].append({"assistant": response, "timestamp": datetime.now()})
        return response

# Initialize agent
agent = SimpleAgent()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Agent Core API",
        "version": "1.0.0",
        "status": "running",
        "type": "minimal"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "agent-core-api", "type": "minimal"}

@app.get("/api/health")
async def api_health():
    """API health check endpoint"""
    return {"status": "healthy", "service": "agent-core-api", "type": "minimal"}

@app.post("/chat/public")
async def chat_public(request: ChatRequest):
    """Public chat endpoint"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        response = await agent.process_message(request.message, session_id)
        
        return {
            "response": response,
            "session_id": session_id,
            "metadata": {
                "agent_type": "simple",
                "response_time_ms": 50
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/public")
async def api_chat_public(request: ChatRequest):
    """API chat endpoint (with /api prefix)"""
    return await chat_public(request)

@app.get("/sessions")
async def get_sessions():
    """Get active sessions count"""
    return {
        "active_sessions": len(sessions),
        "total_messages": sum(len(s) for s in sessions.values())
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    # Listen on all interfaces including IPv6
    uvicorn.run(app, host="::", port=port, log_level="info")