#!/usr/bin/env python3
"""
Unified Memory System
Consolidates all memory operations including Redis, conversation tracking, tape management,
sliding windows, and archival into a single cohesive system.
"""

import json
import logging
import redis
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class StepType(Enum):
    """Types of conversation steps"""
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    CONTEXT_RETRIEVED = "context_retrieved"
    MEMORY_STORED = "memory_stored"
    OPTION_PRESENTED = "option_presented"
    OPTION_SELECTED = "option_selected"
    COMMAND_PREPARED = "command_prepared"
    COMMAND_EXECUTED = "command_executed"
    ERROR_OCCURRED = "error_occurred"
    PERSONALITY_ADJUSTED = "personality_adjusted"


@dataclass
class ConversationStep:
    """Single step in conversation tape"""
    step_type: StepType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        return {
            "step_type": self.step_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


@dataclass
class MemoryWindow:
    """Sliding window of conversation with summaries"""
    raw_steps: deque  # Recent detailed steps
    summaries: List[Dict[str, Any]]  # Hierarchical summaries
    overflow_buffer: List[ConversationStep]  # Steps awaiting summarization
    total_steps: int = 0
    
    def should_summarize(self, threshold: int = 50) -> bool:
        return len(self.overflow_buffer) >= threshold


class UnifiedMemorySystem:
    """
    Unified system for all memory operations
    Combines Redis persistence, conversation tracking, tape management,
    sliding windows, and archival
    """
    
    def __init__(self, 
                 redis_config: Optional[Dict[str, Any]] = None,
                 window_size: int = 20,
                 summary_chunk_size: int = 50,
                 archive_after_days: int = 30):
        
        # Redis configuration - hardcoded credentials
        # Ignore any passed config for security
        self.redis_config = {
            'host': 'redis-11364.c24.us-east-mz-1.ec2.redns.redis-cloud.com',
            'port': 11364,
            'username': 'default',
            'password': 'UQHtexzcYtFCoG7R55InSvmdYHn8fvcf',
            'db': 0,
            'decode_responses': False
        }
        
        # Initialize Redis connection pool with retry logic
        self.redis_client = None
        self.pool = None
        
        for attempt in range(3):  # Try 3 times
            try:
                self.pool = redis.ConnectionPool(**self.redis_config)
                self.redis_client = redis.Redis(connection_pool=self.pool)
                # Test connection
                self.redis_client.ping()
                logger.info("Redis connection established")
                break
            except Exception as e:
                if attempt < 2:
                    logger.info(f"Redis connection attempt {attempt + 1} failed, retrying...")
                    import time
                    time.sleep(1)  # Wait 1 second before retry
                else:
                    logger.warning(f"Redis connection failed after 3 attempts: {e}. Using in-memory storage only.")
                    self.redis_client = None
                    self.pool = None
        
        # Memory configuration
        self.window_size = window_size
        self.summary_chunk_size = summary_chunk_size
        self.archive_after_days = archive_after_days
        
        # Session management
        self.sessions: Dict[str, MemoryWindow] = {}
        self.active_session: Optional[str] = None
        
        # Key prefixes for Redis
        self.CONVERSATION_PREFIX = "conversation:"
        self.TAPE_PREFIX = "tape:"
        self.VECTOR_PREFIX = "vector:"
        self.ARCHIVE_PREFIX = "archive:"
        self.SUMMARY_PREFIX = "summary:"
        self.CONTEXT_PREFIX = "context:"
        self.USER_PREFIX = "user:"
        
        # BM25 search components
        self.bm25_k1 = 1.2
        self.bm25_b = 0.75
        self.document_lengths = {}
        self.average_doc_length = 0
        self.doc_count = 0
        
        logger.info("Unified Memory System initialized")
    
    @property
    def current_tape(self):
        """Get the current session's tape/window"""
        if self.active_session and self.active_session in self.sessions:
            window = self.sessions[self.active_session]
            # Create a tape-like object with steps
            class Tape:
                def __init__(self, steps):
                    self.steps = steps
            return Tape(list(window.raw_steps))
        return None
    
    def create_session(self, session_id: str) -> str:
        """Create or resume a conversation session"""
        if session_id not in self.sessions:
            # Try to load existing session from Redis first
            loaded = False
            if self.redis_client:
                tape_key = f"{self.TAPE_PREFIX}{session_id}:steps"
                existing_steps = self.redis_client.lrange(tape_key, -self.window_size, -1)
                
                if existing_steps:
                    # Load existing session
                    window = MemoryWindow(
                        raw_steps=deque(maxlen=self.window_size),
                        summaries=[],
                        overflow_buffer=[],
                        total_steps=len(existing_steps)
                    )
                    
                    # Parse and load recent steps
                    for step_data in existing_steps:
                        try:
                            step_dict = json.loads(step_data)
                            step = ConversationStep(
                                step_type=StepType(step_dict['step_type']),
                                content=step_dict['content'],
                                metadata=step_dict.get('metadata', {}),
                                timestamp=step_dict.get('timestamp', '')
                            )
                            window.raw_steps.append(step)
                        except Exception as e:
                            logger.warning(f"Failed to load step: {e}")
                    
                    self.sessions[session_id] = window
                    loaded = True
                    logger.info(f"Resumed session {session_id} with {len(window.raw_steps)} recent steps")
            
            if not loaded:
                # Create new session
                self.sessions[session_id] = MemoryWindow(
                    raw_steps=deque(maxlen=self.window_size),
                    summaries=[],
                    overflow_buffer=[],
                    total_steps=0
                )
                
                # Store session metadata in Redis if available
                if self.redis_client:
                    session_key = f"{self.CONVERSATION_PREFIX}{session_id}:metadata"
                    self.redis_client.hset(session_key, mapping={
                        "created_at": datetime.now().isoformat(),
                        "last_active": datetime.now().isoformat(),
                        "step_count": 0
                    })
                    
                    # Add to all sessions set
                    self.redis_client.sadd("all_sessions", session_id)
                
                logger.info(f"Created new session {session_id}")
            
            self.active_session = session_id
        else:
            # Session already loaded in memory
            self.active_session = session_id
            logger.info(f"Using existing session {session_id}")
            
        return session_id
    
    def add_conversation_step(self, 
                            step_type: StepType, 
                            content: Any,
                            metadata: Dict[str, Any] = None,
                            session_id: Optional[str] = None) -> ConversationStep:
        """Add a step to the conversation tape"""
        session_id = session_id or self.active_session
        if not session_id:
            raise ValueError("No active session")
        
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        # Create step
        step = ConversationStep(
            step_type=step_type,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.now().isoformat()
        )
        
        # Add to memory window
        window = self.sessions[session_id]
        window.total_steps += 1
        
        # Handle window overflow
        if len(window.raw_steps) >= self.window_size:
            window.overflow_buffer.append(window.raw_steps[0])
            
            # Check if we need to summarize
            if window.should_summarize(self.summary_chunk_size):
                self._create_summary(session_id)
        
        # Add to current window
        window.raw_steps.append(step)
        
        # Persist to Redis tape if available
        if self.redis_client:
            tape_key = f"{self.TAPE_PREFIX}{session_id}:steps"
            self.redis_client.rpush(tape_key, json.dumps(step.to_dict()))
            self.redis_client.expire(tape_key, 86400 * self.archive_after_days)
            
            # Update session metadata
            session_key = f"{self.CONVERSATION_PREFIX}{session_id}:metadata"
            self.redis_client.hset(session_key, "last_active", datetime.now().isoformat())
            self.redis_client.hincrby(session_key, "step_count", 1)
        
        # Update BM25 index for search
        if step_type in [StepType.USER_INPUT, StepType.AGENT_RESPONSE]:
            self._update_bm25_index(session_id, step)
        
        return step
    
    def _create_summary(self, session_id: str):
        """Create intelligent summary of overflow buffer"""
        window = self.sessions[session_id]
        if not window.overflow_buffer:
            return
        
        # Analyze steps for summary
        summary_data = self._analyze_steps_for_summary(window.overflow_buffer)
        
        # Create summary object
        summary = {
            "timestamp": datetime.now().isoformat(),
            "period": f"{window.overflow_buffer[0].timestamp} to {window.overflow_buffer[-1].timestamp}",
            "step_count": len(window.overflow_buffer),
            "themes": summary_data["themes"],
            "entities": summary_data["entities"],
            "key_exchanges": summary_data["key_exchanges"],
            "emotional_state": summary_data["emotional_state"],
            "summary_text": summary_data["text"]
        }
        
        # Add to summaries
        window.summaries.append(summary)
        
        # Store in Redis
        summary_key = f"{self.SUMMARY_PREFIX}{session_id}:{len(window.summaries)}"
        self.redis_client.set(summary_key, json.dumps(summary))
        self.redis_client.expire(summary_key, 86400 * self.archive_after_days)
        
        # Clear overflow buffer
        window.overflow_buffer = []
        
        # Consolidate if too many summaries
        if len(window.summaries) > 5:
            self._consolidate_summaries(session_id)
    
    def _analyze_steps_for_summary(self, steps: List[ConversationStep]) -> Dict[str, Any]:
        """Analyze steps to create rich summary"""
        themes = set()
        entities = set()
        key_exchanges = []
        emotions = defaultdict(int)
        
        for i, step in enumerate(steps):
            content = str(step.content).lower()
            
            # Extract themes (simple keyword extraction)
            words = content.split()
            themes.update([w for w in words if len(w) > 5])
            
            # Extract entities from metadata
            if "entities" in step.metadata:
                entities.update(step.metadata["entities"])
            
            # Track key exchanges
            if step.step_type == StepType.USER_INPUT:
                # Look for response
                for j in range(i+1, min(i+3, len(steps))):
                    if steps[j].step_type == StepType.AGENT_RESPONSE:
                        key_exchanges.append({
                            "user": step.content[:100],
                            "agent": steps[j].content[:100]
                        })
                        break
            
            # Simple emotion detection
            emotion_keywords = {
                "positive": ["good", "great", "thanks", "perfect", "excellent"],
                "negative": ["bad", "wrong", "error", "problem", "issue"],
                "neutral": ["okay", "fine", "alright", "sure"]
            }
            
            for emotion, keywords in emotion_keywords.items():
                if any(kw in content for kw in keywords):
                    emotions[emotion] += 1
        
        # Determine dominant emotion
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"
        
        # Create summary text
        summary_text = f"Conversation covering {len(steps)} steps, "
        summary_text += f"discussing {', '.join(list(themes)[:3])} "
        summary_text += f"with {dominant_emotion} tone."
        
        return {
            "themes": list(themes)[:10],
            "entities": list(entities)[:10],
            "key_exchanges": key_exchanges[:5],
            "emotional_state": dominant_emotion,
            "text": summary_text
        }
    
    def _consolidate_summaries(self, session_id: str):
        """Consolidate multiple summaries into higher-level summary"""
        window = self.sessions[session_id]
        if len(window.summaries) < 2:
            return
        
        # Take half of summaries to consolidate
        to_consolidate = window.summaries[:len(window.summaries)//2]
        remaining = window.summaries[len(window.summaries)//2:]
        
        # Merge themes and entities
        all_themes = set()
        all_entities = set()
        total_steps = 0
        
        for summary in to_consolidate:
            all_themes.update(summary.get("themes", []))
            all_entities.update(summary.get("entities", []))
            total_steps += summary.get("step_count", 0)
        
        # Create meta-summary
        meta_summary = {
            "timestamp": datetime.now().isoformat(),
            "type": "consolidated",
            "period": f"{to_consolidate[0]['period'].split(' to ')[0]} to {to_consolidate[-1]['period'].split(' to ')[1]}",
            "step_count": total_steps,
            "summary_count": len(to_consolidate),
            "themes": list(all_themes)[:20],
            "entities": list(all_entities)[:20],
            "summary_text": f"Consolidated {len(to_consolidate)} segments covering {total_steps} steps"
        }
        
        # Replace with consolidated version
        window.summaries = [meta_summary] + remaining
    
    def get_conversation_context(self, 
                                session_id: Optional[str] = None,
                                include_summaries: bool = True) -> Dict[str, Any]:
        """Get current conversation context"""
        session_id = session_id or self.active_session
        if not session_id or session_id not in self.sessions:
            return {}
        
        window = self.sessions[session_id]
        
        context = {
            "session_id": session_id,
            "total_steps": window.total_steps,
            "current_window": [step.to_dict() for step in window.raw_steps]
        }
        
        if include_summaries and window.summaries:
            context["summaries"] = window.summaries[-3:]  # Last 3 summaries
        
        return context
    
    def semantic_search(self, 
                       query: str,
                       session_id: Optional[str] = None,
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """Search conversation history semantically"""
        results = []
        
        # BM25 search across all conversations or specific session
        if session_id:
            sessions = [session_id]
        else:
            sessions = list(self.sessions.keys())
        
        for sid in sessions:
            # Search in current window
            if sid in self.sessions:
                window = self.sessions[sid]
                for step in window.raw_steps:
                    if self._bm25_score(query, str(step.content)) > 0.5:
                        results.append({
                            "session_id": sid,
                            "step": step.to_dict(),
                            "score": self._bm25_score(query, str(step.content))
                        })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _bm25_score(self, query: str, document: str) -> float:
        """Calculate BM25 relevance score"""
        query_terms = query.lower().split()
        doc_terms = document.lower().split()
        
        score = 0.0
        doc_length = len(doc_terms)
        
        for term in query_terms:
            tf = doc_terms.count(term)
            if tf > 0:
                # Simplified BM25 without IDF for now
                numerator = tf * (self.bm25_k1 + 1)
                denominator = tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_length / max(self.average_doc_length, 1))
                score += numerator / denominator
        
        return score
    
    def _update_bm25_index(self, session_id: str, step: ConversationStep):
        """Update BM25 search index"""
        doc_id = f"{session_id}:{step.timestamp}"
        doc_length = len(str(step.content).split())
        
        self.document_lengths[doc_id] = doc_length
        self.doc_count += 1
        self.average_doc_length = sum(self.document_lengths.values()) / max(self.doc_count, 1)
    
    def archive_session(self, session_id: str) -> bool:
        """Archive a session for long-term storage"""
        if session_id not in self.sessions:
            return False
        
        try:
            window = self.sessions[session_id]
            
            # Create comprehensive archive
            archive = {
                "session_id": session_id,
                "archived_at": datetime.now().isoformat(),
                "total_steps": window.total_steps,
                "summaries": window.summaries,
                "final_window": [step.to_dict() for step in window.raw_steps],
                "metadata": self._get_session_metadata(session_id)
            }
            
            # Store in Redis with no expiration
            archive_key = f"{self.ARCHIVE_PREFIX}{session_id}"
            self.redis_client.set(archive_key, json.dumps(archive))
            
            # Add to archive index
            self.redis_client.sadd("archived_sessions", session_id)
            
            # Clean up active session data
            self._cleanup_session(session_id)
            
            logger.info(f"Archived session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive session {session_id}: {e}")
            return False
    
    def _get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """Get session metadata from Redis"""
        if not self.redis_client:
            return {}
        try:
            session_key = f"{self.CONVERSATION_PREFIX}{session_id}:metadata"
            metadata = self.redis_client.hgetall(session_key)
            return {k.decode(): v.decode() for k, v in metadata.items()} if metadata else {}
        except Exception:
            return {}
    
    def _cleanup_session(self, session_id: str):
        """Clean up session data after archival"""
        # Remove from active sessions
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        # Remove from active sessions set
        self.redis_client.srem("all_sessions", session_id)
        
        # Clean up Redis keys (they'll expire anyway)
        patterns = [
            f"{self.CONVERSATION_PREFIX}{session_id}:*",
            f"{self.TAPE_PREFIX}{session_id}:*",
            f"{self.SUMMARY_PREFIX}{session_id}:*"
        ]
        
        for pattern in patterns:
            for key in self.redis_client.scan_iter(match=pattern):
                self.redis_client.delete(key)
    
    def auto_archive_old_sessions(self, days: Optional[int] = None):
        """Automatically archive old sessions"""
        days = days or self.archive_after_days
        cutoff = datetime.now() - timedelta(days=days)
        
        archived_count = 0
        
        for session_id in list(self.sessions.keys()):
            metadata = self._get_session_metadata(session_id)
            if metadata.get("last_active"):
                last_active = datetime.fromisoformat(metadata["last_active"])
                if last_active < cutoff:
                    if self.archive_session(session_id):
                        archived_count += 1
        
        logger.info(f"Auto-archived {archived_count} sessions")
        return archived_count
    
    def format_context_for_llm(self, 
                              session_id: Optional[str] = None,
                              max_tokens: int = 2000) -> str:
        """Format conversation context for LLM consumption"""
        session_id = session_id or self.active_session
        if not session_id:
            return ""
        
        context = self.get_conversation_context(session_id)
        parts = []
        
        # Add summaries
        if "summaries" in context:
            parts.append("=== Previous Context ===")
            for summary in context["summaries"]:
                parts.append(summary.get("summary_text", ""))
        
        # Add recent conversation
        parts.append("\n=== Recent Conversation ===")
        for step in context.get("current_window", [])[-10:]:
            if step["step_type"] == "user_input":
                parts.append(f"User: {step['content']}")
            elif step["step_type"] == "agent_response":
                parts.append(f"Agent: {str(step['content'])[:200]}")
        
        # Truncate if needed
        result = "\n".join(parts)
        if len(result) > max_tokens * 4:  # Rough estimate
            result = result[-(max_tokens * 4):]
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        stats = {
            "active_sessions": len(self.sessions),
            "current_session": self.active_session,
            "redis_connected": False
        }
        
        if self.redis_client:
            try:
                stats["total_sessions"] = self.redis_client.scard("all_sessions")
                stats["archived_sessions"] = self.redis_client.scard("archived_sessions")
                stats["redis_connected"] = self.redis_client.ping()
            except:
                stats["total_sessions"] = len(self.sessions)
                stats["archived_sessions"] = 0
        else:
            stats["total_sessions"] = len(self.sessions)
            stats["archived_sessions"] = 0
        
        return stats


# Singleton instance
_memory_instance = None

def get_unified_memory() -> UnifiedMemorySystem:
    """Get or create the unified memory system singleton"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = UnifiedMemorySystem()
    return _memory_instance