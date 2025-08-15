#!/usr/bin/env python3
"""
Agent Chat Interface
Natural conversation with the agent
"""

import os
import sys
import json
import time
import asyncio
import re
import socket
from typing import Optional, Dict, Any, List
from pathlib import Path
import aiohttp

# No hardcoded paths needed

from rq import Queue
from redis import Redis
from rq.job import Job

# Our systems
from core.unified_memory_system import get_unified_memory, StepType
from core.context_intelligence_system import ContextIntelligenceSystem
from knowledge.knowledge_graph_enhanced import get_enhanced_knowledge_graph
from core.adaptive_personality_system import AdaptivePersonalitySystem
from knowledge.hybrid_search_system import SimpleSearchSystem
from knowledge.research_system import ResearchSystem
from core.temporal_context import TemporalContext, TimeGranularity
from utils.redis_logger import get_redis_logger
from core.prompt_evolution_system import PromptEvolutionSystem

logger = get_redis_logger(__name__)

# Load config
from utils.config_loader import config

# Redis connection from config
redis_conn = Redis(
    host=config['redis']['host'],
    port=config['redis']['port'],
    password=config['redis']['password'],
    db=config['redis']['db'],
    decode_responses=False
)


class ChatAgent:
    """Conversational agent interface"""
    
    def __init__(self):
        self.api_key = config['openrouter_api_key']
        self.model = 'mistralai/mistral-medium-3.1'  # Consistent high-quality model
        
        # Initialize ALL our sophisticated systems
        self.memory = get_unified_memory()  # Tape-based memory with sliding windows
        self.context = ContextIntelligenceSystem()  # Deep psychological understanding
        self.knowledge = get_enhanced_knowledge_graph()  # Entity/relationship tracking
        self.personality = AdaptivePersonalitySystem()  # Adaptive personality
        self.search = SimpleSearchSystem()  # Hybrid BM25 + vector search
        self.research = ResearchSystem()  # Full research capabilities
        self.temporal = TemporalContext()  # Temporal awareness and scheduling
        self.prompt_evolution = PromptEvolutionSystem()  # Dynamic prompt management
        self.redis = redis_conn
        
        # Queues for background tasks
        self.research_queue = Queue('research', connection=redis_conn)
        self.default_queue = Queue('default', connection=redis_conn)
        
        # Use persistent session and load user profile
        self.session_id = "persistent_main"  # Use a persistent session ID
        self.memory.create_session(self.session_id)  # Will reuse if exists
        self.user_id = "default_user"
        
        # Simple user profile in Redis
        self.user_profile_key = f"user:profile:{self.user_id}"
        self.user_profile = self._load_user_profile()
        
        # Set personality base
        self.personality.base_personality = {
            "openness": 0.7,  # Creative but practical
            "conscientiousness": 0.8,  # Reliable, organized
            "extraversion": 0.3,  # Introverted, focused
            "agreeableness": 0.5,  # Balanced, not a pushover
            "neuroticism": 0.2  # Stable, unflappable
        }
        self.personality.current_state = self.personality.base_personality.copy()
        
        # Load dynamic system prompt from Redis
        self.system_prompt = self.prompt_evolution.get_current_prompt()
    
    async def call_llm(self, message: str, context: List[Dict] = None) -> str:
        """Call the LLM with context"""
        # Refresh system prompt from Redis (in case it was updated)
        self.system_prompt = self.prompt_evolution.get_current_prompt()
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add context from memory
        if context:
            for msg in context:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Call OpenRouter
        total_tokens = sum(len(m.get("content", "")) for m in messages)
        logger.info(f"Calling LLM with {len(messages)} messages (~{total_tokens} chars)")
        # Force IPv4 due to IPv6 connectivity issues
        connector = aiohttp.TCPConnector(family=socket.AF_INET)
        async with aiohttp.ClientSession(connector=connector) as session:
            try:
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
                        "max_tokens": 2048
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    logger.info(f"LLM response status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        error = await response.text()
                        logger.error(f"LLM error: {error}")
                        return "I encountered an error processing that request."
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return "I'm having trouble connecting to my language model."
    
    async def process_message(self, message: str) -> str:
        """Process user message and generate response"""
        # Add temporal event
        self.temporal.add_event(
            event_type="user_message",
            description=message[:100],
            metadata={"full_message": message, "user_id": self.user_id}
        )
        
        # Store in memory with full tape tracking
        self.memory.add_conversation_step(
            step_type=StepType.USER_INPUT,
            content={"message": message},
            metadata={"user_id": self.user_id, "session_id": self.session_id}
        )
        
        # Analyze context and psychological state
        recent_history = self.memory.current_tape.steps if hasattr(self.memory, 'current_tape') else []
        recent_history = recent_history[-10:] if len(recent_history) > 10 else recent_history
        
        # Context analysis is not async
        context_analysis = self.context.analyze_conversation(
            [{"type": s.step_type.value if hasattr(s.step_type, 'value') else s.step_type, 
              "content": s.content} for s in recent_history],
            self.user_id
        )
        
        # Extract entities for knowledge graph
        entities = self.knowledge.extract_entities(message)
        for entity in entities:
            self.knowledge.add_entity(entity.get('name', entity.get('text', '')), entity['type'])
        
        # Update personality based on interaction
        if hasattr(context_analysis, 'rapport_level'):
            self.personality.update_rapport(self.user_id, context_analysis.rapport_level)
        if hasattr(context_analysis, 'big5_profile') and context_analysis.big5_profile:
            self.personality.update_user_traits(self.user_id, context_analysis.big5_profile)
        
        # Check for commands
        message_lower = message.lower()
        
        # Only catch VERY specific command patterns
        # Research request - must be explicit about wanting research
        if any(phrase in message_lower for phrase in ["research", "find papers", "search arxiv", "look up papers"]):
            return await self.handle_research(message)
        
        # System monitoring - explicit system queries
        elif any(phrase in message_lower for phrase in ["system status", "cpu usage", "memory usage", "temperature", "disk usage"]):
            return self.handle_monitoring()
        
        # User identity question - "Who am I?" (keep this special handler)
        elif "who am i" in message_lower:
            return self.handle_identity_question()
        
        # Task/job status - explicit job check
        elif ("job" in message_lower and "status" in message_lower) or "check jobs" in message_lower:
            return self.handle_status()
        
        # Prompt management commands
        elif "prompt" in message_lower and any(cmd in message_lower for cmd in ["show", "evolve", "history", "rollback", "metrics"]):
            return await self.handle_prompt_management(message)
        
        # General conversation
        else:
            # Search memory for relevant past conversations
            memory_results = self.memory.semantic_search(message, top_k=3)
            
            # Search knowledge base for relevant information
            knowledge_results = self.search.hybrid_search(message, top_k=3)
            
            # Build enhanced context
            context = []
            
            # Add context reminder if we have conversation history
            if len(recent_history) > 1:
                context.append({
                    "role": "system",
                    "content": "Maintain context from the ongoing conversation. Reference previous topics naturally."
                })
            
            # Add recent conversation - exclude the current message (already added)
            # Only include last 2 exchanges (4 messages) to avoid token limits
            history_to_include = recent_history[-5:-1] if len(recent_history) > 1 else []
            for step in history_to_include:
                if step.step_type == StepType.USER_INPUT:
                    context.append({"role": "user", "content": step.content.get('message', '')})
                elif step.step_type == StepType.AGENT_RESPONSE:
                    context.append({"role": "assistant", "content": step.content.get('response', '')})
            
            # Add relevant memory if found
            if memory_results and len(memory_results) > 0:
                # Handle different memory result formats
                first_result = memory_results[0]
                if isinstance(first_result, dict):
                    content = first_result.get('content', str(first_result))[:200]
                else:
                    content = str(first_result)[:200]
                context.append({
                    "role": "system",
                    "content": f"Relevant memory: {content}"
                })
            
            # Add knowledge if found
            if knowledge_results:
                doc = knowledge_results[0]
                context.append({
                    "role": "system", 
                    "content": f"Knowledge base: {doc.get('title', 'Unknown')} - {doc.get('content', '')[:200]}"
                })
            
            # Generate response with full context
            response = await self.call_llm(message, context)
            
            # Track prompt performance
            success = len(response) > 0 and "error" not in response.lower()
            self.prompt_evolution.update_metrics("conversation", success=success)
            
            # Check for confusion indicators
            if any(phrase in response.lower() for phrase in ["i don't understand", "could you clarify", "not sure what you mean"]):
                self.prompt_evolution.update_metrics("confusion", success=False)
            
            # Store response with metadata
            self.memory.add_conversation_step(
                step_type=StepType.AGENT_RESPONSE,
                content={"response": response},
                metadata={
                    "context_depth": len(context),
                    "memory_used": bool(memory_results),
                    "knowledge_used": bool(knowledge_results),
                    "emotional_trajectory": context_analysis.emotional_trajectory,
                    "prompt_version": self.prompt_evolution.redis.hget(
                        self.prompt_evolution.CURRENT_PROMPT_KEY, "id"
                    )
                }
            )
            
            # Update user profile
            self._update_user_profile(message, entities)
            
            return response
    
    async def handle_research(self, message: str) -> str:
        """Handle research requests"""
        # Extract topic using LLM
        topic = await self.extract_topic(message)
        
        # Submit research job
        from rq_worker_system import research_task
        job = self.research_queue.enqueue(
            research_task,
            topic=topic,
            max_documents=10,
            job_timeout='30m'
        )
        
        # Store job reference
        self.memory.add_conversation_step(
            step_type=StepType.CONTEXT_RETRIEVED,  # Using closest matching type
            content={
                "topic": topic,
                "job_id": job.id
            }
        )
        
        response = f"*adjusts my metaphorical research glasses*\n\n"
        response += f"I'm researching '{topic}' for you. This involves:\n"
        response += f"- Searching ArXiv for academic papers\n"
        response += f"- Querying Google Scholar\n"
        response += f"- Extracting and analyzing PDFs\n\n"
        response += f"This will take a few minutes. I'll let you know when I have results.\n"
        response += f"In the meantime, feel free to ask me other questions."
        
        return response
    
    async def handle_query(self, message: str) -> str:
        """Handle knowledge queries"""
        # Query knowledge base
        result = await self.research.query_knowledge(message, use_rag=True)
        
        if result['documents']:
            response = f"Based on my knowledge base:\n\n"
            
            # Synthesize answer
            for i, doc in enumerate(result['documents'][:3], 1):
                response += f"{i}. **{doc.title}**\n"
                if doc.abstract:
                    response += f"   {doc.abstract[:150]}...\n\n"
            
            if result.get('synthesized_answer'):
                response += f"\n**Summary:** {result['synthesized_answer']}"
        else:
            response = "I don't have specific information about that in my knowledge base. "
            response += "Would you like me to research it for you?"
        
        return response
    
    def handle_monitoring(self) -> str:
        """Handle system monitoring requests"""
        import psutil
        
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        response = "*checks the system diagnostics*\n\n"
        response += f"The Pi's doing alright:\n"
        response += f"- CPU: {cpu:.1f}% (not breaking a sweat)\n"
        response += f"- Memory: {mem.percent:.1f}% used ({mem.used/1024**3:.1f}GB of {mem.total/1024**3:.1f}GB)\n"
        response += f"- Disk: {disk.percent:.1f}% used ({disk.used/1024**3:.1f}GB of {disk.total/1024**3:.1f}GB)\n"
        
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", 'r') as f:
                temp = float(f.read().strip()) / 1000
                response += f"- Temperature: {temp:.1f}°C "
                if temp > 70:
                    response += "(running a bit warm)"
                elif temp < 50:
                    response += "(nice and cool)"
                else:
                    response += "(just right)"
        except:
            pass
        
        return response
    
    def handle_identity_question(self) -> str:
        """Handle 'Who am I?' question"""
        # Get user profile
        profile = self.user_profile
        interaction_count = int(profile.get("interaction_count", 0))
        topics = profile.get("topics_of_interest", "").split(",") if profile.get("topics_of_interest") else []
        
        # Get temporal context
        recent_events = self.temporal.get_events_by_granularity(
            TimeGranularity.TODAY
        )
        
        # Build response
        response = "*looks at my notes*\n\n"
        
        if interaction_count == 0:
            response += "Well, this is our first conversation, so I don't know much about you yet. "
            response += "But I'm good at picking things up as we go.\n\n"
        else:
            response += f"Based on our {interaction_count} interactions, here's what I've gathered:\n\n"
            
            if topics:
                response += f"You're interested in: {', '.join(topics[-5:])}\n"
            
            # Check recent interactions
            today_count = len([e for e in recent_events if e.event_type == "user_message"])
            if today_count > 0:
                response += f"We've talked {today_count} times today.\n"
            
            # Get last interaction time
            last_seen = profile.get("last_seen")
            if last_seen:
                from datetime import datetime
                last_time = datetime.fromtimestamp(float(last_seen))
                if (datetime.now() - last_time).days > 1:
                    response += f"Last time we talked was {(datetime.now() - last_time).days} days ago.\n"
            
            response += "\nYour communication style seems direct and technical, which I appreciate."
        
        response += "\n\nBut more importantly - you're someone with questions that need answers. "
        response += "That's what I'm here for."
        
        return response
    
    async def handle_prompt_management(self, message: str) -> str:
        """Handle prompt management commands"""
        message_lower = message.lower()
        
        if "show" in message_lower or "current" in message_lower:
            # Show current prompt
            prompt = self.prompt_evolution.get_current_prompt()
            prompt_data = self.prompt_evolution.redis.hgetall(
                self.prompt_evolution.CURRENT_PROMPT_KEY
            )
            
            response = "*pulls up the workshop manual*\n\n"
            response += f"Current System Prompt (v{prompt_data.get('version', 1)}):\n"
            response += f"ID: {prompt_data.get('id', 'unknown')}\n"
            response += f"Updated: {prompt_data.get('updated_at', 'unknown')}\n\n"
            response += "--- PROMPT ---\n"
            response += prompt[:1000] + "...\n"
            response += "--- END ---\n"
            
        elif "metrics" in message_lower:
            # Show prompt performance metrics
            metrics = self.prompt_evolution.get_prompt_metrics()
            
            response = "*checks the performance gauges*\n\n"
            response += "Prompt Performance Metrics:\n"
            response += f"- Total interactions: {metrics.get('interaction_count', 0)}\n"
            response += f"- Success rate: {metrics.get('success_rate', 0):.1%}\n"
            response += f"- Confusion events: {metrics.get('type_confusion', 0)}\n"
            response += f"- Research tasks: {metrics.get('type_research', 0)}\n"
            response += f"- Conversations: {metrics.get('type_conversation', 0)}\n"
            
        elif "history" in message_lower:
            # Show prompt history
            history = self.prompt_evolution.get_prompt_history(5)
            
            response = "*flips through the version logs*\n\n"
            response += "Recent Prompt Versions:\n"
            for i, version in enumerate(history, 1):
                response += f"{i}. v{version.get('version', '?')} "
                response += f"(ID: {version.get('id', 'unknown')[:8]})\n"
                response += f"   Source: {version.get('source', 'unknown')}\n"
                response += f"   Date: {version.get('updated_at', 'unknown')}\n"
            
        elif "evolve" in message_lower:
            # Trigger prompt evolution
            response = "*considers the workshop improvements*\n\n"
            
            # Gather feedback
            feedback = {
                "user_satisfaction": 0.8,  # Could be calculated from user signals
                "recent_failures": []
            }
            
            result = self.prompt_evolution.evolve_prompt(feedback)
            if result:
                response += f"Evolved prompt to new version (ID: {result})\n"
                response += "Changes will take effect on next interaction."
            else:
                response += "No evolution needed at this time.\n"
                metrics = self.prompt_evolution.get_prompt_metrics()
                response += f"Current success rate: {metrics.get('success_rate', 0):.1%}"
        
        elif "rollback" in message_lower:
            # Extract version ID if provided
            import re
            match = re.search(r'rollback\s+to\s+(\w+)', message_lower)
            if match:
                version_id = match.group(1)
                if self.prompt_evolution.rollback_prompt(version_id):
                    response = f"*dusts off the old manual*\n\nRolled back to version {version_id}."
                    # Reload prompt for next interaction
                    self.system_prompt = self.prompt_evolution.get_current_prompt()
                else:
                    response = f"Cannot rollback to {version_id} - version not found."
            else:
                response = "Specify a version ID to rollback to (e.g., 'rollback to abc123')"
        
        else:
            response = "Available prompt commands: show, metrics, history, evolve, rollback"
        
        return response
    
    def handle_status(self) -> str:
        """Check status of background jobs"""
        response = "Let me check on running tasks...\n\n"
        
        # Check recent research jobs
        recent_jobs = self.memory.search_steps(
            query="research_started",
            limit=3
        )
        
        if recent_jobs:
            for job_info in recent_jobs:
                job_id = job_info['content'].get('job_id')
                topic = job_info['content'].get('topic')
                
                try:
                    job = Job.fetch(job_id, connection=self.redis)
                    status = job.get_status()
                    
                    response += f"**{topic}**: "
                    if job.is_finished:
                        result = job.result
                        if result:
                            response += f"✓ Complete - found {result.get('total_documents', 0)} documents\n"
                    elif job.is_failed:
                        response += f"✗ Failed\n"
                    else:
                        progress = job.meta.get('progress', 0)
                        response += f"⏳ In progress ({progress}%)\n"
                except:
                    pass
        else:
            response += "No recent background tasks."
        
        return response
    
    async def extract_topic(self, message: str) -> str:
        """Extract research topic from message using LLM for cleanup"""
        # First pass: basic extraction
        message_lower = message.lower()
        
        # Simple pattern to get the general topic area
        patterns = [
            r"research\s+(.+)",
            r"look\s+up\s+(.+)",
            r"find\s+(?:papers\s+)?(?:on|about)\s+(.+)",
            r"search\s+(?:for\s+)?(.+)",
            r"(?:learn|know)\s+about\s+(.+)",
            r"about\s+(.+)"
        ]
        
        raw_topic = None
        for pattern in patterns:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                raw_topic = match.group(1).strip()
                break
        
        if not raw_topic:
            # Fallback
            words_to_remove = ["research", "look", "up", "find", "papers", "search", 
                              "please", "can", "you", "for", "me", "about", "on", "the"]
            words = message_lower.split()
            remaining = [w for w in words if w not in words_to_remove]
            raw_topic = " ".join(remaining) if remaining else "general topic"
        
        # Use LLM to clean up the topic
        cleanup_prompt = f"""Extract just the research topic from this phrase. 
Remove any conversational elements like "for me", "please", etc.
Return ONLY the clean topic, nothing else.

Input: {raw_topic}
Clean topic:"""
        
        try:
            clean_topic = await self.call_llm(cleanup_prompt, [])
            # Final cleanup - remove quotes if LLM added them
            clean_topic = clean_topic.strip().strip('"\'').strip()
            return clean_topic if clean_topic else raw_topic
        except:
            # If LLM fails, use the raw topic with basic cleanup
            raw_topic = re.sub(r'\s+(for\s+me|please|thanks?|now|today)$', '', raw_topic)
            return raw_topic.strip()
    
    def _load_user_profile(self) -> Dict:
        """Load user profile from Redis"""
        try:
            profile_data = self.redis.hgetall(self.user_profile_key)
            if profile_data:
                # Decode bytes to string
                return {k.decode(): v.decode() if isinstance(v, bytes) else v 
                       for k, v in profile_data.items()}
            else:
                # Create default profile
                default_profile = {
                    "communication_style": "direct",
                    "technical_level": "high",
                    "topics_of_interest": "",
                    "interaction_count": "0",
                    "last_seen": str(time.time())
                }
                self.redis.hset(self.user_profile_key, mapping=default_profile)
                return default_profile
        except Exception as e:
            logger.error(f"Error loading user profile: {e}")
            return {"communication_style": "direct", "technical_level": "high"}
    
    def _update_user_profile(self, message: str, entities: List):
        """Update user profile with interaction data"""
        try:
            # Update interaction count
            self.redis.hincrby(self.user_profile_key, "interaction_count", 1)
            
            # Update last seen
            self.redis.hset(self.user_profile_key, "last_seen", str(time.time()))
            
            # Add topics if entities found
            if entities:
                topics = [e.get('name', e.get('text', '')) for e in entities if e.get('name') or e.get('text')]
                existing = self.redis.hget(self.user_profile_key, "topics_of_interest")
                if existing:
                    existing = existing.decode() if isinstance(existing, bytes) else existing
                    topics_list = existing.split(",") if existing else []
                    topics_list.extend(topics)
                    # Keep last 20 topics
                    topics_list = topics_list[-20:]
                else:
                    topics_list = topics
                
                self.redis.hset(self.user_profile_key, "topics_of_interest", ",".join(topics_list))
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
    
    async def generate_welcome_message(self) -> str:
        """Generate dynamic welcome message using LLM based on user history"""
        try:
            # Check user profile
            interaction_count = self.redis.hget(self.user_profile_key, "interaction_count")
            interaction_count = int(interaction_count) if interaction_count else 0
            
            last_seen = self.redis.hget(self.user_profile_key, "last_seen")
            topics = self.redis.hget(self.user_profile_key, "topics_of_interest")
            
            # Check recent temporal events (last 24 hours)
            end_time = time.time()
            start_time = end_time - (24 * 3600)
            recent_events = self.temporal.get_events_by_time_range(start_time, end_time)
            
            # Build context for LLM
            context_info = {
                "interaction_count": interaction_count,
                "time_since_last": None,
                "topics_discussed": [],
                "recent_activities": []
            }
            
            if last_seen:
                time_diff = time.time() - float(last_seen)
                if time_diff < 3600:
                    context_info["time_since_last"] = "less than an hour ago"
                elif time_diff < 86400:
                    context_info["time_since_last"] = f"{int(time_diff/3600)} hours ago"
                else:
                    days = int(time_diff / 86400)
                    context_info["time_since_last"] = f"{days} days ago"
            
            if topics:
                topics_str = topics.decode() if isinstance(topics, bytes) else topics
                context_info["topics_discussed"] = topics_str.split(",")[:5] if topics_str else []
            
            if recent_events:
                for event in recent_events[-3:]:
                    context_info["recent_activities"].append(event.event_type)
            
            # Generate welcome via LLM
            prompt = f"""Generate a welcome message as Nick Offerman-inspired personality (without woodworking).
            
Context:
- User interactions: {interaction_count}
- Last seen: {context_info['time_since_last'] or 'first time'}
- Previous topics: {', '.join(context_info['topics_discussed']) if context_info['topics_discussed'] else 'none'}
- My recent activities: {', '.join(context_info['recent_activities']) if context_info['recent_activities'] else 'standard operations'}

Generate an appropriate greeting that:
- Is in character (dry, competent, direct)
- References the context naturally
- Includes an action (*looks up*, *nods*, etc)
- Is 1-3 sentences max
- Ends with asking what they need"""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            welcome = await self.call_llm(prompt, [])
            return welcome.strip()
            
        except Exception as e:
            logger.error(f"Error generating welcome: {e}")
            # Fallback to default
            return """*looks up from a terminal window*

Name's Nick. Well, not really, but I've been told I have his demeanor.
I run on this Raspberry Pi, handling research, system management, that sort of thing.
Ask me whatever you need."""
    
    async def chat_loop(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("AGENT CHAT INTERFACE")
        print("="*60)
        print("Multi-line: ''' or \"\"\" for quotes, \\ for continuation")
        print("Commands: exit | prompt show/metrics/history")
        print("="*60)
        
        # Generate dynamic welcome
        welcome = await self.generate_welcome_message()
        print(f"\n{welcome}\n")
        
        while True:
            try:
                # Get user input - supports multi-line
                import sys
                print("You: ", end="", flush=True)
                lines = []
                
                if sys.stdin.isatty():
                    # Interactive terminal
                    first_line = input().strip()
                    
                    if first_line.startswith('"""') or first_line.startswith("'''"):
                        # Multi-line mode with triple quotes
                        quote_type = first_line[:3]
                        if first_line.endswith(quote_type) and len(first_line) > 6:
                            # Single line with triple quotes on both ends
                            lines.append(first_line[3:-3])
                        else:
                            lines.append(first_line[3:] if len(first_line) > 3 else '')
                            while True:
                                line = input()
                                if line.endswith(quote_type):
                                    if len(line) > 3:
                                        lines.append(line[:-3])
                                    break
                                lines.append(line)
                    elif first_line.endswith('\\'):
                        # Multi-line mode with backslash continuation
                        lines.append(first_line[:-1])
                        while True:
                            line = input()
                            if not line.endswith('\\'):
                                lines.append(line)
                                break
                            lines.append(line[:-1])
                    else:
                        # Single line input
                        lines.append(first_line)
                else:
                    # Piped input - read all lines
                    try:
                        for line in sys.stdin:
                            stripped = line.rstrip('\n')
                            if stripped.lower() in ['exit', 'quit', 'bye']:
                                if lines:  # Process previous lines first
                                    break
                                lines.append(stripped)  # Add exit command
                                break
                            lines.append(stripped)
                    except EOFError:
                        pass
                
                user_input = '\n'.join(lines).strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    # Generate dynamic goodbye using LLM
                    interaction_count = self.redis.hget(self.user_profile_key, "interaction_count")
                    interaction_count = int(interaction_count) if interaction_count else 0
                    
                    # Get recent conversation summary
                    recent_steps = self.memory.current_tape.steps[-5:] if hasattr(self.memory, 'current_tape') else []
                    topics_discussed = []
                    for step in recent_steps:
                        if step.step_type == StepType.USER_INPUT:
                            msg = step.content.get('message', '')
                            if msg and msg.lower() not in ['exit', 'quit', 'bye']:
                                topics_discussed.append(msg[:50])
                    
                    prompt = f"""Generate a goodbye message as Nick Offerman-inspired personality.
                    
Context:
- Total interactions: {interaction_count}
- Topics from this session: {', '.join(topics_discussed[:3]) if topics_discussed else 'general chat'}

Generate a brief goodbye that:
- Is in character (dry, direct, competent)
- References something from the conversation if relevant
- Includes an action if appropriate (*nods*, *goes back to terminal*, etc)
- Is 1 sentence
- Sounds natural, not forced"""

                    try:
                        goodbye = await self.call_llm(prompt, [])
                        print(f"\nAgent: {goodbye.strip()}")
                    except:
                        # Fallback
                        print("\nAgent: Alright. I'll be here.")
                    break
                
                # Process and respond
                print("\nAgent: ", end="", flush=True)
                response = await self.process_message(user_input)
                
                # Print response with typing effect (optional)
                for char in response:
                    print(char, end="", flush=True)
                    time.sleep(0.01)  # Typing effect
                print("\n")
                
            except (EOFError, KeyboardInterrupt):
                # Handle EOF (no input) and Ctrl+C gracefully
                print("\n\nAgent: Alright, we'll continue this later.")
                break
            except Exception as e:
                # Only log unexpected errors
                if "EOF" not in str(e):
                    logger.error(f"Chat error: {e}")
                    print(f"\nAgent: I encountered an error: {e}\n")
                else:
                    # Silent EOF handling for piped input
                    break


async def main():
    """Main entry point"""
    agent = ChatAgent()
    await agent.chat_loop()


if __name__ == "__main__":
    asyncio.run(main())