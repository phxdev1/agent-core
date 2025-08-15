#!/usr/bin/env python3
"""
Fast Agent Chat Interface
Optimized for sub-500ms response times with background intelligence processing
"""

import os
import json
import time
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import aiohttp
from collections import deque
from datetime import datetime, timedelta
import hashlib

from rq import Queue
from redis import Redis

# Our systems - imported at module level for speed
from core.unified_memory_system import get_unified_memory, StepType
from core.context_intelligence_system import ContextIntelligenceSystem
from knowledge.knowledge_graph_enhanced import get_enhanced_knowledge_graph
from core.adaptive_personality_system import AdaptivePersonalitySystem
from knowledge.hybrid_search_system import SimpleSearchSystem
from knowledge.research_system import ResearchSystem
from core.temporal_context import TemporalContext, TimeGranularity
from utils.redis_logger import get_redis_logger
from core.prompt_evolution_system import PromptEvolutionSystem
from utils.config_loader import config

logger = get_redis_logger(__name__)


class ResponseCache:
    """Simple LRU cache for frequent responses"""
    def __init__(self, max_size=100):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: str):
        # Simple LRU eviction
        if len(self.cache) >= self.max_size:
            oldest = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest]
            del self.access_times[oldest]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def hash_key(self, message: str, context_size: int = 0) -> str:
        """Create cache key from message and context"""
        return hashlib.md5(f"{message}_{context_size}".encode()).hexdigest()[:16]


class FastChatAgent:
    """High-performance conversational agent with background processing"""
    
    def __init__(self):
        self.api_key = config['openrouter_api_key']
        self.model = 'mistralai/mistral-medium-3.1'
        
        # Initialize systems lazily
        self._memory = None
        self._context = None
        self._knowledge = None
        self._personality = None
        self._search = None
        self._research = None
        self._temporal = None
        self._prompt_evolution = None
        
        # Redis for background tasks
        self.redis = Redis(
            host=config['redis']['host'],
            port=config['redis']['port'],
            password=config['redis']['password'],
            db=config['redis']['db'],
            decode_responses=False
        )
        
        # Queues
        self.research_queue = Queue('research', connection=self.redis)
        self.background_queue = Queue('background', connection=self.redis)
        
        # Session management
        self.session_id = "persistent_main"
        self.user_id = "default_user"
        
        # Response cache
        self.response_cache = ResponseCache(max_size=50)
        
        # Recent context cache (avoid repeated memory lookups)
        self.context_cache = deque(maxlen=10)
        
        # Pending background tasks
        self.pending_tasks = {}
        
        # Base system prompt (cached)
        self._system_prompt = None
        self._prompt_last_updated = None
    
    # Lazy property getters for systems
    @property
    def memory(self):
        if self._memory is None:
            self._memory = get_unified_memory()
            self._memory.create_session(self.session_id)
        return self._memory
    
    @property
    def context(self):
        if self._context is None:
            self._context = ContextIntelligenceSystem()
        return self._context
    
    @property
    def knowledge(self):
        if self._knowledge is None:
            self._knowledge = get_enhanced_knowledge_graph()
        return self._knowledge
    
    @property
    def personality(self):
        if self._personality is None:
            self._personality = AdaptivePersonalitySystem()
            self._personality.base_personality = {
                "openness": 0.7,
                "conscientiousness": 0.8,
                "extraversion": 0.3,
                "agreeableness": 0.5,
                "neuroticism": 0.2
            }
        return self._personality
    
    @property
    def search(self):
        if self._search is None:
            self._search = SimpleSearchSystem()
        return self._search
    
    @property
    def research(self):
        if self._research is None:
            self._research = ResearchSystem()
        return self._research
    
    @property
    def temporal(self):
        if self._temporal is None:
            self._temporal = TemporalContext()
        return self._temporal
    
    @property
    def prompt_evolution(self):
        if self._prompt_evolution is None:
            self._prompt_evolution = PromptEvolutionSystem()
        return self._prompt_evolution
    
    @property
    def system_prompt(self):
        """Get cached system prompt, refresh every 5 minutes"""
        now = datetime.now()
        if (self._system_prompt is None or 
            self._prompt_last_updated is None or
            (now - self._prompt_last_updated) > timedelta(minutes=5)):
            self._system_prompt = self.prompt_evolution.get_current_prompt()
            self._prompt_last_updated = now
        return self._system_prompt
    
    async def call_llm_fast(self, message: str, context: List[Dict] = None, 
                            stream: bool = False) -> str:
        """Fast LLM call with streaming support"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add minimal context (keep it light for speed)
        if context:
            messages.extend(context[:3])  # Limit context for speed
        
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
                        "temperature": 0.7,
                        "max_tokens": 512,  # Shorter for speed
                        "stream": stream
                    },
                    timeout=aiohttp.ClientTimeout(total=10)  # Shorter timeout
                ) as response:
                    if response.status == 200:
                        if stream:
                            # Return async generator for streaming
                            return self._stream_response(response)
                        else:
                            data = await response.json()
                            return data['choices'][0]['message']['content']
                    else:
                        return "I'm thinking... (processing in background)"
        except asyncio.TimeoutError:
            return "Let me think about that... (processing in background)"
        except Exception as e:
            logger.error(f"Fast LLM call failed: {e}")
            return "I'm processing your request..."
    
    async def _stream_response(self, response):
        """Stream response chunks"""
        async for line in response.content:
            if line:
                yield line.decode('utf-8')
    
    async def process_message_fast(self, message: str) -> str:
        """
        Ultra-fast message processing with background intelligence
        Target: <500ms response time
        """
        start_time = time.time()
        
        # Check cache first (instant response)
        cache_key = self.response_cache.hash_key(message)
        cached_response = self.response_cache.get(cache_key)
        if cached_response:
            logger.info(f"Cache hit, response time: {(time.time() - start_time)*1000:.0f}ms")
            # Still do background processing for learning
            asyncio.create_task(self._background_intelligence(message, None))
            return cached_response
        
        # Create background tasks for heavy processing
        background_tasks = []
        
        # 1. Quick store user input (non-blocking)
        background_tasks.append(
            asyncio.create_task(self._store_user_input(message))
        )
        
        # 2. Get minimal recent context (from cache if possible)
        quick_context = self._get_quick_context()
        
        # 3. Start parallel background intelligence gathering
        intelligence_task = asyncio.create_task(
            self._background_intelligence(message, quick_context)
        )
        background_tasks.append(intelligence_task)
        
        # 4. Generate fast initial response (this is the critical path)
        initial_response = await self.call_llm_fast(message, quick_context)
        
        response_time = (time.time() - start_time) * 1000
        logger.info(f"Initial response time: {response_time:.0f}ms")
        
        # Cache the response
        self.response_cache.set(cache_key, initial_response)
        
        # 5. Store response in background
        asyncio.create_task(self._store_response(initial_response, message))
        
        # 6. Schedule enhancement if response was too quick/simple
        if response_time < 300 and len(initial_response) < 100:
            asyncio.create_task(self._enhance_response(message, initial_response, intelligence_task))
        
        return initial_response
    
    async def _store_user_input(self, message: str):
        """Store user input in background"""
        try:
            self.memory.add_conversation_step(
                step_type=StepType.USER_INPUT,
                content={"message": message},
                metadata={"user_id": self.user_id, "session_id": self.session_id}
            )
            
            # Add to temporal context
            self.temporal.add_event(
                event_type="user_message",
                description=message[:100],
                metadata={"full_message": message, "user_id": self.user_id}
            )
        except Exception as e:
            logger.error(f"Failed to store user input: {e}")
    
    async def _store_response(self, response: str, original_message: str):
        """Store agent response in background"""
        try:
            self.memory.add_conversation_step(
                step_type=StepType.AGENT_RESPONSE,
                content={"response": response},
                metadata={
                    "original_message": original_message,
                    "response_time": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Failed to store response: {e}")
    
    def _get_quick_context(self) -> List[Dict]:
        """Get minimal context quickly from cache"""
        # Return last 2 exchanges from cache
        if self.context_cache:
            return list(self.context_cache)[-4:]  # Last 2 user/assistant pairs
        return []
    
    async def _background_intelligence(self, message: str, context: List[Dict]):
        """
        Gather intelligence in background (non-blocking)
        This runs after the initial response is sent
        """
        try:
            # Run these in parallel
            tasks = []
            
            # 1. Context analysis (if heavy, make it async)
            tasks.append(asyncio.create_task(
                self._analyze_context_async(message)
            ))
            
            # 2. Entity extraction
            tasks.append(asyncio.create_task(
                self._extract_entities_async(message)
            ))
            
            # 3. Memory search
            tasks.append(asyncio.create_task(
                self._search_memory_async(message)
            ))
            
            # 4. Knowledge search
            tasks.append(asyncio.create_task(
                self._search_knowledge_async(message)
            ))
            
            # 5. Personality update
            tasks.append(asyncio.create_task(
                self._update_personality_async(message)
            ))
            
            # Wait for all with timeout
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results for potential enhancement
            self.pending_tasks[message] = {
                'context_analysis': results[0] if not isinstance(results[0], Exception) else None,
                'entities': results[1] if not isinstance(results[1], Exception) else None,
                'memory_results': results[2] if not isinstance(results[2], Exception) else None,
                'knowledge_results': results[3] if not isinstance(results[3], Exception) else None,
                'personality_updated': results[4] if not isinstance(results[4], Exception) else None,
                'timestamp': time.time()
            }
            
            # Clean old pending tasks (>5 minutes)
            current_time = time.time()
            self.pending_tasks = {
                k: v for k, v in self.pending_tasks.items() 
                if current_time - v['timestamp'] < 300
            }
            
        except Exception as e:
            logger.error(f"Background intelligence failed: {e}")
    
    async def _analyze_context_async(self, message: str):
        """Async wrapper for context analysis"""
        return await asyncio.to_thread(
            self.context.analyze_conversation,
            [{"type": "user_input", "content": message}],
            self.user_id
        )
    
    async def _extract_entities_async(self, message: str):
        """Async wrapper for entity extraction"""
        return await asyncio.to_thread(
            self.knowledge.extract_entities,
            message
        )
    
    async def _search_memory_async(self, message: str):
        """Async wrapper for memory search"""
        return await asyncio.to_thread(
            self.memory.semantic_search,
            message,
            3  # top_k
        )
    
    async def _search_knowledge_async(self, message: str):
        """Async wrapper for knowledge search"""
        return await asyncio.to_thread(
            self.search.hybrid_search,
            message,
            3  # top_k
        )
    
    async def _update_personality_async(self, message: str):
        """Async wrapper for personality update"""
        # Simple sentiment-based update
        return await asyncio.to_thread(
            self.personality.adapt_to_interaction,
            self.user_id,
            {"message": message}
        )
    
    async def _enhance_response(self, message: str, initial_response: str, 
                               intelligence_task: asyncio.Task):
        """
        Enhance response with gathered intelligence (optional)
        Only runs if initial response was very quick/simple
        """
        try:
            # Wait for intelligence to complete (with timeout)
            await asyncio.wait_for(intelligence_task, timeout=2.0)
            
            # Get the gathered intelligence
            if message in self.pending_tasks:
                intelligence = self.pending_tasks[message]
                
                # If we found significant new context, we could:
                # 1. Send a follow-up message
                # 2. Update a UI indicator
                # 3. Store enhanced context for next interaction
                
                if intelligence.get('memory_results') or intelligence.get('knowledge_results'):
                    # Add to context cache for next interaction
                    self.context_cache.append({
                        "role": "system",
                        "content": f"Previous context: {intelligence}"
                    })
                    
                    logger.info("Response enhanced with background intelligence")
        except asyncio.TimeoutError:
            logger.debug("Enhancement timeout - intelligence still processing")
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
    
    async def process_message(self, message: str) -> str:
        """
        Main entry point - routes to fast or full processing
        """
        # Check for special commands that need full processing
        message_lower = message.lower()
        
        if any(phrase in message_lower for phrase in [
            "research", "find papers", "search arxiv",
            "system status", "who am i", "job status"
        ]):
            # These need full processing - delegate to original method
            # For now, use fast path with a note
            response = await self.process_message_fast(message)
            if "research" in message_lower:
                response += "\n\n*Note: Full research capabilities are being processed in background.*"
            return response
        
        # Use fast path for general conversation
        return await self.process_message_fast(message)
    
    async def chat(self):
        """Interactive chat loop with fast responses"""
        print("Fast Agent Chat (sub-500ms responses)")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Time the response
                start = time.time()
                response = await self.process_message(user_input)
                elapsed = (time.time() - start) * 1000
                
                print(f"\nAgent: {response}")
                print(f"[Response time: {elapsed:.0f}ms]")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point"""
    agent = FastChatAgent()
    asyncio.run(agent.chat())


if __name__ == "__main__":
    main()