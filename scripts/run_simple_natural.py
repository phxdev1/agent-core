#!/usr/bin/env python3
"""
Simple runner for Natural Agent without complex dependencies
"""

import os
import sys
import asyncio
from pathlib import Path

# Set up path
sys.path.insert(0, str(Path(__file__).parent))

# Set no Redis to avoid those dependencies
os.environ['REDIS_HOST'] = ''

# Import only what we need
import aiohttp
from typing import Optional, Dict, Any, List


class MinimalNaturalAgent:
    """Minimal natural agent for testing"""
    
    def __init__(self):
        # Get API key from environment or .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except:
            pass
            
        self.api_key = os.getenv('OPENROUTER_API_KEY', '')
        if not self.api_key:
            print("Warning: OPENROUTER_API_KEY not set")
            
        self.model = 'mistralai/mistral-medium-3.1'
        self.conversation_history = []
        
        self.system_prompt = """You're having a natural conversation. Be helpful but conversational.
Don't announce actions, just respond naturally. Be concise."""
    
    async def call_llm(self, message: str) -> str:
        """Call the LLM"""
        if not self.api_key:
            return "API key not configured. Please set OPENROUTER_API_KEY."
            
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add recent history
        messages.extend(self.conversation_history[-4:])
        messages.append({"role": "user", "content": message})
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.8,
                        "max_tokens": 1024
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        error = await response.text()
                        return f"API error: {error[:100]}"
        except Exception as e:
            return f"Connection error: {str(e)}"
    
    async def chat(self):
        """Simple chat loop"""
        print("\n" + "="*50)
        print("Minimal Natural Agent Chat")
        print("Type 'bye' to exit")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = input("> ").strip()
                
                if user_input.lower() in ['bye', 'exit', 'quit']:
                    print("\nGoodbye!")
                    break
                    
                if not user_input:
                    continue
                
                # Add to history
                self.conversation_history.append({"role": "user", "content": user_input})
                
                # Get response
                response = await self.call_llm(user_input)
                print(f"\n{response}\n")
                
                # Add response to history
                self.conversation_history.append({"role": "assistant", "content": response})
                
                # Keep history reasonable
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def main():
    """Run the minimal agent"""
    agent = MinimalNaturalAgent()
    asyncio.run(agent.chat())


if __name__ == "__main__":
    main()