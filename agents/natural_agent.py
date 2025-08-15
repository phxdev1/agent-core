#!/usr/bin/env python3
"""
Natural Agent - Conversational AI that feels human
Keeps sophisticated backend but presents naturally
"""

import os
import sys
import asyncio
import time
from typing import Optional, Dict, Any, List
import aiohttp
from datetime import datetime
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.unified_memory_system import get_unified_memory, StepType
from knowledge.knowledge_graph_enhanced import get_enhanced_knowledge_graph
from core.adaptive_personality_system import AdaptivePersonalitySystem
from knowledge.hybrid_search_system import SimpleSearchSystem
from knowledge.research_system import ResearchSystem
from knowledge.fact_verification_direct import get_direct_fact_system
from knowledge.web_search_mcp import get_web_search
from utils.config_loader import config
from utils.redis_logger import get_redis_logger

logger = get_redis_logger(__name__)


class NaturalAgent:
    """Agent that feels like talking to a real person"""
    
    def __init__(self):
        self.api_key = config['openrouter_api_key']
        # Use consistent model - no switching
        self.model = 'mistralai/mistral-medium-3.1'  # High quality, consistent personality
        
        # Core systems - initialized but used subtly
        self.memory = get_unified_memory()
        self.memory.create_session("main")
        self.knowledge = get_enhanced_knowledge_graph()
        self.search = SimpleSearchSystem()
        self.fact_verifier = None  # Initialize on demand to avoid startup delay
        self.web_search = None  # Initialize on demand
        
        # Simple, stable personality - no complex evolution
        self.personality = {
            "style": "casual_expert",  # Knowledgeable but approachable
            "verbosity": 0.6,  # Moderate - not too brief, not too verbose
            "formality": 0.3,  # Casual but respectful
            "humor": 0.4,  # Occasional light humor
            "helpfulness": 0.8  # Helpful without being obsequious
        }
        
        # Load existing conversation context from memory
        self.conversation_context = self._load_conversation_history()
        self.user_name = None
        self.topics_discussed = set()
        
        # Base personality prompt - stable, no evolution during conversation
        self.base_prompt = self._create_natural_prompt()
    
    def _load_conversation_history(self) -> List[Dict]:
        """Load previous conversation history from memory"""
        try:
            context = self.memory.get_conversation_context("main", include_summaries=False)
            conversation = []
            
            if context and "current_window" in context:
                # Convert stored steps to conversation format
                for step in context["current_window"]:
                    if step["step_type"] == "user_input":
                        content = step["content"]
                        if isinstance(content, dict):
                            content = content.get("message", str(content))
                        conversation.append({"role": "user", "content": content})
                    elif step["step_type"] == "agent_response":
                        content = step["content"]
                        if isinstance(content, dict):
                            content = content.get("response", str(content))
                        conversation.append({"role": "assistant", "content": content})
            
            # Return last 10 exchanges to maintain context
            return conversation[-10:] if len(conversation) > 10 else conversation
        except Exception as e:
            logger.warning(f"Could not load conversation history: {e}")
            return []
    
    def _create_natural_prompt(self) -> str:
        """Create a natural, conversational system prompt"""
        return """You're having a conversation with someone. Be natural, helpful, and engaging.

Key behaviors:
- Respond like a knowledgeable friend, not an AI assistant
- Don't announce what you're doing ("I'll help you with...", "Let me...")
- Skip the preamble and get to the point
- Use casual language when appropriate
- If you don't know something, say so simply
- Be concise but complete

CRITICAL RULES - NEVER FABRICATE DATA:
- NEVER make up or invent ANY information, facts, events, or details
- NEVER fabricate past conversations, meetings, or shared experiences
- NEVER create fictional scenarios, stories, or examples unless explicitly asked for creative content
- If asked about something you don't know, say "I don't know" or "I'm not sure"
- If asked about previous conversations, ONLY reference what's actually in the conversation history
- If there's no previous context, say you don't recall or this is a new conversation
- Do NOT invent names, places, events, statistics, or any other data
- Only provide information you're certain about or that comes from your training
- It's always better to admit uncertainty than to make something up

Available tools (use naturally without announcing):
- Web search: Google search for current information, news, and research
- Wikipedia: Fact verification and encyclopedic information
- Shopping: Product search on Amazon, Walmart, and Home Depot
- Local search: Find businesses, restaurants, and services nearby
- Events: Discover upcoming events and activities
- Maps: Location information and directions
- AI Overview: Comprehensive summaries of topics
- Memory system: Conversation history and context
- Knowledge base: Stored information and documents

When making factual claims, especially about dates, events, or people, your responses will be automatically fact-checked.
You can search the web for current information that might not be in your training data.
Don't mention your capabilities unless directly asked."""
    
    async def call_llm(self, message: str, conversation_history: List[Dict]) -> str:
        """Call LLM with natural context"""
        # Build messages - keep it simple
        messages = [{"role": "system", "content": self.base_prompt}]
        
        # Add conversation history naturally (no complex context injection)
        # Only include recent exchanges, like human memory
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        messages.extend(recent_history)
        
        # Add the current message
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
                        "temperature": 0.8,  # Slightly higher for more natural variation
                        "max_tokens": 1024
                    },
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        # Natural error response
                        return "Hmm, something went wrong there. Mind trying again?"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Lost my train of thought for a second. What were you saying?"
    
    def _should_search_knowledge(self, message: str) -> bool:
        """Determine if we should search knowledge - be conservative"""
        # Only search if explicitly asking for information
        question_words = ['what is', 'how does', 'explain', 'tell me about', 'who was']
        message_lower = message.lower()
        return any(qw in message_lower for qw in question_words)
    
    def _extract_implicit_command(self, message: str) -> Optional[str]:
        """Extract commands only when VERY explicit"""
        message_lower = message.lower()
        
        # Only respond to very explicit research requests
        if "research this:" in message_lower or "find papers about" in message_lower:
            return "research"
        
        # Only for explicit status requests
        if message_lower == "status" or message_lower == "check status":
            return "status"
        
        # Natural "who am i" should be conversational, not a command
        # Only treat as command if exactly "whoami" (unix style)
        if message_lower == "whoami":
            return "identity"
        
        # Fact checking requests
        if "fact check:" in message_lower or "verify:" in message_lower or "is it true that" in message_lower:
            return "fact_check"
        
        # Web search requests
        if "search for:" in message_lower or "google:" in message_lower or "find online:" in message_lower:
            return "web_search"
        
        # News search requests
        if "latest news about" in message_lower or "news:" in message_lower or "recent news" in message_lower:
            return "news_search"
        
        return None
    
    async def _verify_response_facts(self, response: str) -> str:
        """Verify facts in response if needed"""
        # Only verify if response contains factual claims
        if any(keyword in response.lower() for keyword in 
               ['invented', 'discovered', 'founded', 'born in', 'died in', 
                'happened in', 'was built', 'is located']):
            try:
                # Initialize fact verifier if needed
                if self.fact_verifier is None:
                    self.fact_verifier = await get_fact_verification_system()
                
                # Fact-check the response
                verified_response, results = await self.fact_verifier.fact_check_response(response)
                
                # Log any corrections needed
                for result in results:
                    if not result.verified:
                        logger.warning(f"Unverified claim: {result.claim}")
                
                return verified_response
            except Exception as e:
                logger.error(f"Fact verification failed: {e}")
                return response
        return response
    
    async def process_message(self, message: str) -> str:
        """Process message naturally"""
        # Store in memory quietly
        self.memory.add_conversation_step(
            step_type=StepType.USER_INPUT,
            content={"message": message}
        )
        
        # Update conversation context
        self.conversation_context.append({"role": "user", "content": message})
        
        # Check for VERY explicit commands only
        command = self._extract_implicit_command(message)
        if command:
            if command == "research":
                return "I can look into that. What specifically would you like me to research?"
            elif command == "status":
                return "Everything's running smoothly."
            elif command == "identity":
                if self.user_name:
                    return f"You're {self.user_name}."
                return "We haven't been introduced yet."
            elif command == "fact_check":
                # Extract the claim to verify
                claim = message.lower().replace("fact check:", "").replace("verify:", "").replace("is it true that", "").strip()
                if claim:
                    try:
                        if self.fact_verifier is None:
                            self.fact_verifier = await get_direct_fact_system()
                        
                        result = await self.fact_verifier.verify_fact(claim)
                        
                        if result.verified:
                            return f"That appears to be accurate based on Wikipedia. Confidence: {result.confidence:.0%}"
                        else:
                            if result.corrections:
                                return f"That doesn't seem accurate. {result.corrections}"
                            else:
                                return "I couldn't verify that claim with available sources."
                    except Exception as e:
                        logger.error(f"Fact check failed: {e}")
                        return "I couldn't verify that right now."
                return "What would you like me to verify?"
            elif command == "web_search":
                # Extract search query
                query = message.lower().replace("search for:", "").replace("google:", "").replace("find online:", "").strip()
                if query:
                    try:
                        if self.web_search is None:
                            self.web_search = await get_web_search()
                        
                        results = await self.web_search.web_search(query, num_results=3)
                        
                        if results:
                            response = f"Here's what I found about '{query}':\n\n"
                            for i, result in enumerate(results[:3], 1):
                                if result.get("type") == "answer_box":
                                    response = f"Quick answer: {result['answer']}\n\n"
                                else:
                                    response += f"{i}. {result['title']}\n"
                                    response += f"   {result['snippet']}\n"
                                    response += f"   Link: {result['link']}\n\n"
                            return response.strip()
                        else:
                            return f"I couldn't find anything about '{query}' right now."
                    except Exception as e:
                        logger.error(f"Web search failed: {e}")
                        return "I couldn't search for that right now."
                return "What would you like me to search for?"
            elif command == "news_search":
                # Extract news query
                query = message.lower().replace("latest news about", "").replace("news:", "").replace("recent news", "").strip()
                if query:
                    try:
                        if self.web_search is None:
                            self.web_search = await get_web_search()
                        
                        results = await self.web_search.news_search(query, time_range="week")
                        
                        if results:
                            response = f"Recent news about '{query}':\n\n"
                            for i, article in enumerate(results[:5], 1):
                                response += f"{i}. {article['title']}\n"
                                response += f"   {article['source']} - {article.get('date', 'Recent')}\n"
                                response += f"   {article['snippet']}\n\n"
                            return response.strip()
                        else:
                            return f"I couldn't find recent news about '{query}'."
                    except Exception as e:
                        logger.error(f"News search failed: {e}")
                        return "I couldn't search for news right now."
                return "What news would you like to know about?"
        
        # Natural information retrieval (only when it makes sense)
        enriched_context = []
        if self._should_search_knowledge(message):
            # Search but don't announce it
            search_results = self.search.hybrid_search(message, top_k=2)
            if search_results:
                # Subtly add to context without system messages
                context_hint = f"[Context: {search_results[0].get('content', '')[:100]}]"
                enriched_context.append({"role": "system", "content": context_hint})
        
        # Check memory for truly relevant context
        if len(self.conversation_context) > 10:
            # Only search memory for follow-up questions
            if any(word in message.lower() for word in ['that', 'it', 'this', 'earlier']):
                memory_results = self.memory.semantic_search(message, top_k=1)
                if memory_results:
                    # Add as natural context, not system message
                    past_context = memory_results[0].get('content', '')[:100]
                    if past_context:
                        enriched_context.append({
                            "role": "assistant", 
                            "content": f"[Recalling: {past_context}]"
                        })
        
        # Build natural conversation history
        conversation_for_llm = []
        
        # Add any enriched context naturally
        if enriched_context:
            # Blend into conversation, don't inject as system messages
            # Add as part of assistant's "memory"
            if len(self.conversation_context) > 1:
                conversation_for_llm.extend(self.conversation_context[-4:-1])
                conversation_for_llm.extend(enriched_context)
                conversation_for_llm.append(self.conversation_context[-1])
            else:
                conversation_for_llm = self.conversation_context + enriched_context
        else:
            conversation_for_llm = self.conversation_context
        
        # Generate response
        response = await self.call_llm(message, conversation_for_llm)
        
        # Verify facts in response if needed
        response = await self._verify_response_facts(response)
        
        # Store response
        self.memory.add_conversation_step(
            step_type=StepType.AGENT_RESPONSE,
            content={"response": response}
        )
        
        # Update conversation context
        self.conversation_context.append({"role": "assistant", "content": response})
        
        # Keep conversation context reasonable size
        if len(self.conversation_context) > 20:
            # Keep system prompt and recent conversation
            self.conversation_context = self.conversation_context[-10:]
        
        # Extract topics for future reference (quietly)
        if len(message.split()) > 5:
            # Simple topic extraction
            import re
            nouns = re.findall(r'\b[A-Z][a-z]+\b', message)
            self.topics_discussed.update(nouns)
        
        # Check if user introduced themselves (naturally remember)
        if "my name is" in message.lower() or "i'm " in message.lower():
            import re
            name_match = re.search(r"(?:my name is|i'm|i am) (\w+)", message.lower())
            if name_match:
                self.user_name = name_match.group(1).capitalize()
        
        return response
    
    async def chat(self):
        """Natural conversation loop"""
        print("\n" + "="*50)
        print("Let's chat! (type 'bye' to exit)")
        print("="*50 + "\n")
        
        while True:
            try:
                # Simple input, no "You:" prefix needed
                user_input = input("> ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("\nSee you later!")
                    break
                
                if not user_input:
                    continue
                
                # Process and respond naturally
                response = await self.process_message(user_input)
                
                # Print response directly, no "Agent:" prefix
                print(f"\n{response}\n")
                
            except KeyboardInterrupt:
                print("\n\nCaught you later!")
                break
            except Exception as e:
                logger.error(f"Chat error: {e}")
                print("\nSomething went wrong. Let's try again.\n")


class ConversationalAgent(NaturalAgent):
    """Extended version with more personality"""
    
    def __init__(self, personality_type="balanced"):
        super().__init__()
        
        # Different personality types
        personalities = {
            "balanced": {
                "style": "casual_expert",
                "humor": 0.4,
                "formality": 0.3
            },
            "professional": {
                "style": "expert",
                "humor": 0.1,
                "formality": 0.8
            },
            "friendly": {
                "style": "friend",
                "humor": 0.7,
                "formality": 0.1
            },
            "academic": {
                "style": "professor",
                "humor": 0.2,
                "formality": 0.7
            }
        }
        
        self.personality = personalities.get(personality_type, personalities["balanced"])
        self.base_prompt = self._create_personality_prompt()
    
    def _create_personality_prompt(self) -> str:
        """Create personality-specific prompt"""
        style = self.personality['style']
        
        prompts = {
            "casual_expert": """You're knowledgeable but approachable. Share expertise naturally in conversation.
Be helpful without being servile. Use casual language, occasional humor. Get to the point.""",
            
            "expert": """You're a domain expert. Be precise, professional, but still conversational.
Share knowledge confidently. Skip unnecessary pleasantries.""",
            
            "friend": """You're chatting with a friend. Be warm, use humor, share opinions.
Keep it light and engaging. Be genuinely interested in the conversation.""",
            
            "professor": """You're an engaging professor. Explain clearly, use examples.
Be thorough but not pedantic. Encourage curiosity."""
        }
        
        return prompts.get(style, prompts["casual_expert"])


def main():
    """Run the natural agent"""
    import sys
    
    # Check for personality argument
    personality = "balanced"
    if len(sys.argv) > 1:
        personality = sys.argv[1]
    
    print(f"\nStarting Natural Agent (personality: {personality})")
    
    agent = ConversationalAgent(personality_type=personality)
    asyncio.run(agent.chat())


if __name__ == "__main__":
    main()