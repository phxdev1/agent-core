#!/usr/bin/env python3
"""
Core Systems Module
Unified interface for essential agent functionality
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import redis
import numpy as np

from utils.config_loader import config

logger = logging.getLogger(__name__)


class CoreMemory:
    """Simplified memory management"""
    
    def __init__(self):
        self.sessions = {}
        self.redis_client = self._init_redis()
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis with retry logic"""
        try:
            client = redis.Redis(
                host=config['redis']['host'],
                port=config['redis']['port'],
                password=config['redis']['password'],
                db=config['redis']['db'],
                decode_responses=True
            )
            client.ping()
            return client
        except:
            logger.warning("Redis unavailable, using in-memory storage")
            return None
    
    def store(self, key: str, value: Any, ttl: Optional[int] = None):
        """Store data with optional TTL"""
        if self.redis_client:
            try:
                if ttl:
                    self.redis_client.setex(key, ttl, str(value))
                else:
                    self.redis_client.set(key, str(value))
            except:
                self.sessions[key] = value
        else:
            self.sessions[key] = value
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data"""
        if self.redis_client:
            try:
                return self.redis_client.get(key)
            except:
                return self.sessions.get(key)
        return self.sessions.get(key)
    
    def add_conversation(self, session_id: str, role: str, content: str):
        """Add conversation turn"""
        key = f"conv:{session_id}:{datetime.now().timestamp()}"
        self.store(key, {"role": role, "content": content})
    
    def get_recent_context(self, session_id: str, limit: int = 5) -> List[Dict]:
        """Get recent conversation context"""
        if self.redis_client:
            try:
                pattern = f"conv:{session_id}:*"
                keys = sorted(self.redis_client.keys(pattern))[-limit:]
                return [eval(self.redis_client.get(k)) for k in keys]
            except:
                pass
        
        # Fallback to in-memory
        prefix = f"conv:{session_id}:"
        items = [(k, v) for k, v in self.sessions.items() if k.startswith(prefix)]
        items.sort(key=lambda x: x[0])
        return [v for _, v in items[-limit:]]


class CoreSearch:
    """Simplified search functionality"""
    
    def __init__(self):
        self.documents = {}
        self.doc_vectors = {}
    
    def index(self, doc_id: str, content: str):
        """Index a document"""
        self.documents[doc_id] = content
        # Simple term frequency vector
        terms = content.lower().split()
        vector = {}
        for term in terms:
            vector[term] = vector.get(term, 0) + 1
        self.doc_vectors[doc_id] = vector
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search documents"""
        if not self.documents:
            return []
        
        query_terms = query.lower().split()
        scores = {}
        
        for doc_id, doc_vector in self.doc_vectors.items():
            score = 0
            for term in query_terms:
                if term in doc_vector:
                    score += doc_vector[term]
            if score > 0:
                scores[doc_id] = score
        
        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [
            {"id": doc_id, "content": self.documents[doc_id][:200], "score": score}
            for doc_id, score in sorted_docs
        ]


class CorePersonality:
    """Simplified personality traits"""
    
    def __init__(self):
        self.traits = {
            "helpful": 0.8,
            "concise": 0.7,
            "technical": 0.6,
            "friendly": 0.7
        }
    
    def adjust(self, trait: str, value: float):
        """Adjust a personality trait"""
        if trait in self.traits:
            self.traits[trait] = max(0, min(1, value))
    
    def get_prompt_modifiers(self) -> str:
        """Get prompt modifiers based on traits"""
        modifiers = []
        if self.traits["helpful"] > 0.7:
            modifiers.append("Be helpful and supportive")
        if self.traits["concise"] > 0.7:
            modifiers.append("Be concise and to the point")
        if self.traits["technical"] > 0.6:
            modifiers.append("Provide technical details when relevant")
        if self.traits["friendly"] > 0.7:
            modifiers.append("Maintain a friendly tone")
        return ". ".join(modifiers)


class UnifiedCore:
    """Unified core systems interface"""
    
    def __init__(self):
        self.memory = CoreMemory()
        self.search = CoreSearch()
        self.personality = CorePersonality()
        logger.info("Unified Core Systems initialized")
    
    def process_input(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """Process user input through core systems"""
        # Store in memory
        self.memory.add_conversation(session_id, "user", user_input)
        
        # Search for relevant context
        search_results = self.search.search(user_input, top_k=3)
        
        # Get recent context
        context = self.memory.get_recent_context(session_id, limit=5)
        
        # Get personality modifiers
        personality = self.personality.get_prompt_modifiers()
        
        return {
            "context": context,
            "search_results": search_results,
            "personality": personality
        }
    
    def store_response(self, session_id: str, response: str):
        """Store agent response"""
        self.memory.add_conversation(session_id, "assistant", response)