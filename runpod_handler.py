#!/usr/bin/env python3
"""
RunPod Serverless Handler for Agent Core
"""

import os
import json
import time
import asyncio
import subprocess
from typing import Dict, Any

import runpod

# Start Redis locally if not running
def ensure_redis():
    """Ensure Redis is running locally"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
    except:
        # Start Redis in background
        subprocess.Popen(['redis-server', '--daemonize', 'yes', '--dir', '/data'])
        time.sleep(2)

# Initialize Redis on container start
ensure_redis()

# Import agent after Redis is ready
from agent_chat import ChatAgent
from config_loader import config

# Global agent instance
agent = None

def initialize_agent():
    """Initialize the agent once"""
    global agent
    if agent is None:
        agent = ChatAgent()
    return agent

async def process_message(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Process a message through the agent"""
    
    # Get inputs
    message = job_input.get('message', '')
    session_id = job_input.get('session_id', 'default')
    mode = job_input.get('mode', 'chat')
    context_window = job_input.get('context_window', 4)
    
    if not message:
        return {
            'error': 'No message provided',
            'status': 'failed'
        }
    
    # Initialize agent
    agent = initialize_agent()
    
    # Set session
    if session_id != agent.session_id:
        agent.session_id = session_id
        agent.memory.create_session(session_id)
    
    # Track processing time
    start_time = time.time()
    
    try:
        # Process message
        response = await agent.process_message(message)
        
        # Calculate metadata
        processing_time = time.time() - start_time
        
        # Get conversation stats
        stats = agent.memory.get_statistics()
        
        # Build output
        output = {
            'response': response,
            'session_id': session_id,
            'metadata': {
                'processing_time': processing_time,
                'tokens_used': len(response.split()),  # Rough estimate
                'confidence': 0.95,  # Could be calculated based on context
                'memory_sessions': stats.get('active_sessions', 1),
                'context_depth': context_window
            },
            'status': 'success'
        }
        
        # Add sources if research mode
        if mode == 'research' and hasattr(agent, 'last_sources'):
            output['metadata']['sources'] = agent.last_sources
        
        return output
        
    except Exception as e:
        return {
            'error': str(e),
            'status': 'failed',
            'processing_time': time.time() - start_time
        }

def handler(job):
    """RunPod handler function"""
    
    job_input = job.get('input', {})
    
    # Run async function
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_message(job_input))
        loop.close()
        return result
    except Exception as e:
        return {
            'error': f'Handler error: {str(e)}',
            'status': 'failed'
        }

# Health check endpoint
def health_check():
    """Health check for the service"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        
        # Check if agent can be initialized
        agent = initialize_agent()
        
        return {
            'status': 'healthy',
            'redis': 'connected',
            'agent': 'ready',
            'model': config.get('model', 'unknown')
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }

# RunPod serverless start
if __name__ == '__main__':
    runpod.serverless.start({
        'handler': handler,
        'health_check': health_check
    })