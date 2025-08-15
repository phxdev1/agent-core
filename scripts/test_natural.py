#!/usr/bin/env python3
"""
Test Natural Agent without interactive input
"""

import os
import sys
import asyncio
from pathlib import Path

# Set up path
sys.path.insert(0, str(Path(__file__).parent))

# Set no Redis to avoid those dependencies
os.environ['REDIS_HOST'] = ''

# Test imports
print("Testing imports...")

try:
    from agents.natural_agent import NaturalAgent
    print("[OK] Natural agent imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import natural agent: {e}")
    
try:
    from run_simple_natural import MinimalNaturalAgent
    print("[OK] Minimal agent imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import minimal agent: {e}")

# Test agent initialization
print("\nTesting agent initialization...")

try:
    agent = MinimalNaturalAgent()
    print("[OK] Minimal agent initialized")
    print(f"  Model: {agent.model}")
    print(f"  API key set: {bool(agent.api_key)}")
except Exception as e:
    print(f"[FAIL] Failed to initialize: {e}")

# Test a simple call
async def test_call():
    """Test a simple LLM call"""
    print("\nTesting LLM call...")
    agent = MinimalNaturalAgent()
    
    if not agent.api_key:
        print("[FAIL] No API key set - skipping LLM test")
        return
        
    response = await agent.call_llm("Hello! Just testing if you're working.")
    print(f"[OK] Got response: {response[:100]}...")

# Run test
if __name__ == "__main__":
    asyncio.run(test_call())