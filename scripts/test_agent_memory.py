#!/usr/bin/env python3
"""
Test that the natural agent uses real memory without fabrication
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.natural_agent import NaturalAgent

async def test_agent_memory():
    """Test agent memory handling"""
    print("Testing Natural Agent Memory System")
    print("=" * 50)
    
    # Create agent
    agent = NaturalAgent()
    
    # Test 1: Ask about previous conversations (should not fabricate)
    print("\nTest 1: Asking about previous conversations")
    response1 = await agent.process_message("What did we talk about last time?")
    print(f"User: What did we talk about last time?")
    print(f"Agent: {response1}")
    
    # Test 2: Have a real conversation
    print("\n\nTest 2: Having a real conversation")
    response2 = await agent.process_message("Hi! My favorite color is blue and I like Python programming.")
    print(f"User: Hi! My favorite color is blue and I like Python programming.")
    print(f"Agent: {response2}")
    
    # Test 3: Reference the real conversation
    print("\n\nTest 3: Referencing real conversation")
    response3 = await agent.process_message("What did I just tell you about?")
    print(f"User: What did I just tell you about?")
    print(f"Agent: {response3}")
    
    # Test 4: Check memory persistence (create new agent instance)
    print("\n\nTest 4: Memory persistence with new agent instance")
    agent2 = NaturalAgent()
    response4 = await agent2.process_message("What do you know about me?")
    print(f"User: What do you know about me?")
    print(f"Agent: {response4}")
    
    print("\n" + "=" * 50)
    print("Memory Test Complete")
    print("\nKey observations:")
    print("- Agent should NOT fabricate past conversations")
    print("- Agent should remember real conversations within session")
    print("- Agent should load previous context when reinitialized")

if __name__ == "__main__":
    asyncio.run(test_agent_memory())