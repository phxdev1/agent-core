#!/usr/bin/env python3
"""
Simple web server for the Agent Core web interface
Serves the HTML file and handles CORS
"""

import os
import sys
from pathlib import Path
import asyncio
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# FastAPI imports
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create FastAPI app
app = FastAPI(title="Agent Core Web", version="1.0.0")

# Configure CORS - allow the web interface to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the main page
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main chat interface"""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "web"}

# Serve any additional static files (CSS, JS, images)
web_dir = Path(__file__).parent
if (web_dir / "static").exists():
    app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")

def main():
    """Run the web server"""
    print("=" * 60)
    print("Agent Core Web Interface")
    print("=" * 60)
    print("\nStarting web server...")
    print(f"Web interface: http://localhost:8001")
    print(f"API should be running on: http://localhost:8000")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # Different port from API
        log_level="info"
    )

if __name__ == "__main__":
    main()