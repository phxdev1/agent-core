#!/usr/bin/env python3
"""
Quick working API for immediate deployment
"""

import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Agent Core API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    agent_type: Optional[str] = "llm"

@app.get("/")
async def root():
    return {"name": "Agent Core API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/chat/public")
async def chat_public(request: ChatRequest):
    # For now, just echo with a friendly response
    responses = {
        "hello": "Hello! I'm Agent Core, your AI assistant. How can I help you today?",
        "hi": "Hi there! What can I do for you?",
        "help": "I can answer questions, provide information, and assist with various tasks!",
        "how are you": "I'm doing great, thank you! Ready to help.",
    }
    
    message_lower = request.message.lower()
    
    # Check for keyword matches
    for key, response in responses.items():
        if key in message_lower:
            return {
                "response": response,
                "session_id": request.session_id or str(uuid.uuid4()),
                "metadata": {"agent_type": "quick"}
            }
    
    # Default response
    return {
        "response": f"I understand you're asking about '{request.message}'. I'm here to help! What specific information are you looking for?",
        "session_id": request.session_id or str(uuid.uuid4()),
        "metadata": {"agent_type": "quick"}
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)