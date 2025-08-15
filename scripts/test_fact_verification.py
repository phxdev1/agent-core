#!/usr/bin/env python3
"""
Test the fact verification system
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.natural_agent import NaturalAgent

async def test_fact_verification():
    """Test fact verification functionality"""
    print("Testing Fact Verification System")
    print("=" * 50)
    
    # Create agent
    agent = NaturalAgent()
    
    # Test 1: Verify a true fact
    print("\nTest 1: Verifying a true fact")
    response1 = await agent.process_message("Fact check: The Eiffel Tower was built in 1889")
    print(f"User: Fact check: The Eiffel Tower was built in 1889")
    print(f"Agent: {response1}")
    
    # Test 2: Verify a false fact
    print("\n\nTest 2: Verifying a false fact")
    response2 = await agent.process_message("Is it true that the Great Wall of China was built in 1950")
    print(f"User: Is it true that the Great Wall of China was built in 1950")
    print(f"Agent: {response2}")
    
    # Test 3: Have a conversation with factual claims
    print("\n\nTest 3: Conversation with automatic fact checking")
    response3 = await agent.process_message("Tell me about when the Internet was invented")
    print(f"User: Tell me about when the Internet was invented")
    print(f"Agent: {response3}")
    
    # Test 4: Verify a complex claim
    print("\n\nTest 4: Complex claim verification")
    response4 = await agent.process_message("Verify: Albert Einstein won the Nobel Prize in Physics in 1921")
    print(f"User: Verify: Albert Einstein won the Nobel Prize in Physics in 1921")
    print(f"Agent: {response4}")
    
    print("\n" + "=" * 50)
    print("Fact Verification Test Complete")
    print("\nKey features tested:")
    print("- Direct fact checking with 'Fact check:' command")
    print("- Question-based verification with 'Is it true that'")
    print("- Automatic fact verification in responses")
    print("- Complex claim verification")

if __name__ == "__main__":
    asyncio.run(test_fact_verification())