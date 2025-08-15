#!/usr/bin/env python3
"""
Runner for Natural Agent
Run this from the project root
"""

import asyncio
import sys

from agents.natural_agent import NaturalAgent, ConversationalAgent


def main():
    """Run the natural agent"""
    # Check for personality argument
    personality = "balanced"
    if len(sys.argv) > 1:
        personality = sys.argv[1]
    
    print(f"\nStarting Natural Agent (personality: {personality})")
    
    agent = ConversationalAgent(personality_type=personality)
    asyncio.run(agent.chat())


if __name__ == "__main__":
    main()