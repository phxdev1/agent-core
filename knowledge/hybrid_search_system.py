#!/usr/bin/env python3
"""
Lightweight Search System Stub
Provides basic search functionality without heavy ML dependencies
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SimpleSearchSystem:
    """
    Lightweight search system that provides basic functionality
    without requiring PyTorch or sentence-transformers
    """
    
    def __init__(self):
        self.documents = {}
        logger.info("Simple Search System initialized")
    
    def hybrid_search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Simple search stub that returns empty results
        Real implementation would use OpenAI embeddings or other lightweight methods
        """
        logger.debug(f"Search called with query: {query[:50]}...")
        return []
    
    def index_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Store document for future search"""
        self.documents[doc_id] = {
            'content': content,
            'metadata': metadata or {}
        }
        return True
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Alias for hybrid_search"""
        return self.hybrid_search(query, top_k)


# Removed unnecessary factory function - use SimpleSearchSystem() directly