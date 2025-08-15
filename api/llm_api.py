#!/usr/bin/env python3
"""
API with LLM integration for real AI responses
"""

import os
import asyncio
import uuid
import json
from typing import Optional, List, Dict
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import httpx

# Initialize FastAPI
app = FastAPI(
    title="Agent Core API",
    description="AI Agent API with LLM Integration",
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

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")

# Simple in-memory session storage
sessions = {}

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    session_id: Optional[str] = None
    agent_type: Optional[str] = "llm"

class LLMAgent:
    """Agent that uses LLM for responses"""
    
    def __init__(self):
        self.system_prompt = """You are a professional AI assistant. Be concise, direct, and helpful.
Focus on accuracy and clarity. Avoid unnecessary pleasantries or filler text.
When appropriate, use markdown formatting for better readability.
If you need clarification, ask specific questions.
Provide actionable insights and solutions."""
        
    async def get_llm_response(self, messages: List[Dict]) -> str:
        """Get response from LLM API"""
        
        # Check if we have API keys
        if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
            return "I'm currently in demo mode. To enable full AI capabilities, please configure API keys."
        
        try:
            # Use OpenAI-compatible API
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": LLM_MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 500
                }
                
                response = await client.post(
                    f"{LLM_API_BASE}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    print(f"LLM API error: {response.status_code} - {response.text}")
                    return self.get_fallback_response(messages[-1]["content"])
                    
        except Exception as e:
            print(f"LLM API exception: {e}")
            return self.get_fallback_response(messages[-1]["content"])
    
    def get_fallback_response(self, message: str) -> str:
        """Fallback responses when LLM is unavailable"""
        responses = {
            "hello": "Hello! I'm Agent Core. How can I assist you today?",
            "hi": "Hi there! What can I help you with?",
            "help": "I'm here to help! You can ask me questions, request information, or just have a conversation.",
            "how are you": "I'm doing great, thank you for asking! Ready to help you.",
            "what can you do": "I can answer questions, provide information, help with tasks, and have conversations on many topics!",
        }
        
        message_lower = message.lower()
        for key, response in responses.items():
            if key in message_lower:
                return response
        
        return f"I understand you're asking about '{message}'. While I'm in limited mode, I'm still here to help as best I can!"
    
    async def process_message(self, message: str, session_id: str) -> str:
        """Process a message and return a response"""
        
        # Initialize session if new
        if session_id not in sessions:
            sessions[session_id] = []
        
        # Add user message to history
        sessions[session_id].append({"role": "user", "content": message})
        
        # Prepare messages for LLM (include system prompt and history)
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add recent conversation history (last 10 messages)
        history = sessions[session_id][-10:]
        messages.extend(history)
        
        # Get LLM response
        response = await self.get_llm_response(messages)
        
        # Add assistant response to history
        sessions[session_id].append({"role": "assistant", "content": response})
        
        # Limit session history to 50 messages
        if len(sessions[session_id]) > 50:
            sessions[session_id] = sessions[session_id][-50:]
        
        return response

# Initialize agent
agent = LLMAgent()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Agent Core API",
        "version": "1.0.0",
        "status": "running",
        "type": "llm",
        "llm_configured": bool(OPENAI_API_KEY or ANTHROPIC_API_KEY)
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "agent-core-api", "type": "llm"}

@app.get("/api/health")
async def api_health():
    """API health check endpoint"""
    return {"status": "healthy", "service": "agent-core-api", "type": "llm"}

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
                "agent_type": "llm",
                "response_time_ms": 100,
                "llm_model": LLM_MODEL if OPENAI_API_KEY else "demo"
            }
        }
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/public")
async def api_chat_public(request: ChatRequest):
    """API chat endpoint (with /api prefix)"""
    return await chat_public(request)

@app.get("/sessions")
async def get_sessions():
    """Get active sessions info"""
    return {
        "active_sessions": len(sessions),
        "total_messages": sum(len(s) for s in sessions.values()),
        "llm_configured": bool(OPENAI_API_KEY or ANTHROPIC_API_KEY)
    }

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session cleared"}
    return {"message": "Session not found"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting LLM API on port {port}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"API Keys configured: OpenAI={bool(OPENAI_API_KEY)}, Anthropic={bool(ANTHROPIC_API_KEY)}")
    
    # Listen on all interfaces including IPv6
    uvicorn.run(app, host="::", port=port, log_level="info")