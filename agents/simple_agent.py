#!/usr/bin/env python3
"""
Simplified Agent Interface
Reduced complexity with lazy loading and core functionality only
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
import aiohttp

from utils.config_loader import config
from core.unified_memory_system import UnifiedMemorySystem, StepType


class SimpleAgent:
    """Simplified conversational agent with lazy loading"""
    
    def __init__(self):
        self.api_key = config['openrouter_api_key']
        self.model = config.get('model', 'mistralai/mistral-medium-3.1')
        
        # Core system only - lazy load others as needed
        self._memory = None
        self._research = None
        self._search = None
        
        # Simple session management
        self.session_id = "main"
        self.system_prompt = self._get_default_prompt()
    
    @property
    def memory(self):
        """Lazy load memory system"""
        if self._memory is None:
            self._memory = UnifiedMemorySystem()
            self._memory.create_session(self.session_id)
        return self._memory
    
    @property
    def research(self):
        """Lazy load research system"""
        if self._research is None:
            from knowledge.research_system import ResearchSystem
            self._research = ResearchSystem()
        return self._research
    
    @property 
    def search(self):
        """Lazy load search system"""
        if self._search is None:
            from knowledge.hybrid_search_system import SimpleSearchSystem
            self._search = SimpleSearchSystem()
        return self._search
    
    def _get_default_prompt(self) -> str:
        """Get default system prompt"""
        return """You are a helpful AI assistant with access to memory and research capabilities.
Be concise, practical, and helpful. Focus on solving problems efficiently."""
    
    async def call_llm(self, message: str, context: List[Dict] = None) -> str:
        """Call the LLM with context"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add context if provided
        if context:
            for item in context:
                messages.append(item)
        
        messages.append({"role": "user", "content": message})
        
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
                    "max_tokens": config['max_tokens'],
                    "temperature": config['temperature']
                }
            ) as response:
                result = await response.json()
                return result['choices'][0]['message']['content']
    
    async def process_message(self, user_input: str) -> str:
        """Process a user message and return response"""
        # Store user input
        self.memory.add_step(
            self.session_id,
            StepType.USER_INPUT,
            user_input
        )
        
        # Get recent context
        context = self.memory.get_conversation_context(self.session_id, limit=5)
        
        # Check if research is needed
        if any(keyword in user_input.lower() for keyword in ['research', 'search', 'find out', 'look up']):
            # Use research system
            research_result = await self.research.research(user_input)
            context.append({"role": "system", "content": f"Research results: {research_result}"})
        
        # Generate response
        response = await self.call_llm(user_input, context)
        
        # Store response
        self.memory.add_step(
            self.session_id,
            StepType.AGENT_RESPONSE,
            response
        )
        
        return response
    
    async def chat(self):
        """Interactive chat loop"""
        print("Simple Agent Chat (type 'exit' to quit)")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                response = await self.process_message(user_input)
                print(f"\nAgent: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point"""
    agent = SimpleAgent()
    asyncio.run(agent.chat())


if __name__ == "__main__":
    main()