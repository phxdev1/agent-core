#!/usr/bin/env python3
"""
Test main.py natural mode
"""

import os
import sys
import asyncio

# Set no Redis to avoid dependencies
os.environ['REDIS_HOST'] = ''

# Import main
from main import get_agent

print("Testing main.py agent loading...")

try:
    agent = get_agent('natural')
    print("[OK] Natural agent loaded from main.py")
    print(f"  Agent type: {type(agent).__name__}")
    print(f"  Model: {agent.model}")
    
    # Test a simple message
    async def test():
        response = await agent.process_message("Hello, just testing!")
        print(f"[OK] Got response: {response[:100]}...")
    
    asyncio.run(test())
    
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()