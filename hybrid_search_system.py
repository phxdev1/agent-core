#!/usr/bin/env python3
"""
Hybrid Search System
Combines vector embeddings and BM25 lexical search for optimal retrieval
"""

import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class HybridSearchSystem:
    """
    Unified search system combining vector similarity and BM25 lexical search
    """
    
    def __init__(self, redis_client=None, embedding_model=None):
        self.redis_client = redis_client
        self.embedding_model = embedding_model
        
        # BM25 parameters
        self.bm25_k1 = 1.2
        self.bm25_b = 0.75
        
        # Document statistics for BM25
        self.doc_count = 0
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.term_doc_freqs = defaultdict(int)  # term -> document frequency
        
        # Vector index
        self.vectors = {}  # doc_id -> vector
        self.vector_dim = 384  # Default dimension for sentence-transformers
        
        # Document store
        self.documents = {}  # doc_id -> document content
        self.metadata = {}  # doc_id -> metadata
        
        logger.info("Hybrid Search System initialized")
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text"""
        if self.embedding_model:
            try:
                # Use the provided embedding model
                embedding = self.embedding_model.encode(text)
                return np.array(embedding)
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                return None
        else:
            # Fallback to simple hash-based pseudo-embedding
            return self._generate_hash_embedding(text)
    
    def _generate_hash_embedding(self, text: str) -> np.ndarray:
        """Generate a deterministic pseudo-embedding using hashing"""
        # Create multiple hash values to fill the vector
        embedding = []
        for i in range(self.vector_dim // 8):
            hash_obj = hashlib.sha256(f"{text}_{i}".encode())
            hash_bytes = hash_obj.digest()[:8]
            # Convert bytes to float between -1 and 1
            value = int.from_bytes(hash_bytes, 'little') / (2**63)
            embedding.append(value)
        
        # Pad if necessary
        while len(embedding) < self.vector_dim:
            embedding.append(0.0)
        
        return np.array(embedding[:self.vector_dim])
    
    def index_document(self, 
                      doc_id: str,
                      content: str,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Index a document for both vector and lexical search"""
        try:
            # Store document
            self.documents[doc_id] = content
            self.metadata[doc_id] = metadata or {}
            
            # Generate and store vector
            embedding = self.generate_embedding(content)
            if embedding is not None:
                self.vectors[doc_id] = embedding
                
                # Store in Redis if available
                if self.redis_client:
                    vector_key = f"vector:{doc_id}"
                    self.redis_client.set(vector_key, embedding.tobytes())
            
            # Update BM25 statistics
            self._update_bm25_stats(doc_id, content)
            
            # Store in Redis if available
            if self.redis_client:
                doc_key = f"document:{doc_id}"
                self.redis_client.hset(doc_key, mapping={
                    "content": content,
                    "metadata": json.dumps(metadata or {}),
                    "indexed_at": datetime.now().isoformat()
                })
            
            logger.debug(f"Indexed document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index document {doc_id}: {e}")
            return False
    
    def _update_bm25_stats(self, doc_id: str, content: str):
        """Update BM25 statistics for a document"""
        # Tokenize
        terms = self._tokenize(content)
        
        # Update document length
        self.doc_lengths[doc_id] = len(terms)
        self.doc_count += 1
        
        # Update average document length
        self.avg_doc_length = sum(self.doc_lengths.values()) / max(self.doc_count, 1)
        
        # Update term document frequencies
        unique_terms = set(terms)
        for term in unique_terms:
            self.term_doc_freqs[term] += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split
        text = text.lower()
        # Remove punctuation and split
        import re
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_id: str) -> float:
        """Calculate BM25 score for a document"""
        if doc_id not in self.documents:
            return 0.0
        
        doc_content = self.documents[doc_id]
        doc_terms = self._tokenize(doc_content)
        doc_length = len(doc_terms)
        
        score = 0.0
        
        for term in query_terms:
            # Term frequency in document
            tf = doc_terms.count(term)
            if tf == 0:
                continue
            
            # Inverse document frequency
            df = self.term_doc_freqs.get(term, 0)
            idf = np.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            
            # BM25 formula
            numerator = idf * tf * (self.bm25_k1 + 1)
            denominator = tf + self.bm25_k1 * (1 - self.bm25_b + 
                                              self.bm25_b * doc_length / max(self.avg_doc_length, 1))
            
            score += numerator / denominator
        
        return score
    
    def _calculate_vector_similarity(self, query_vector: np.ndarray, doc_vector: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        # Cosine similarity
        dot_product = np.dot(query_vector, doc_vector)
        norm_product = np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
        
        if norm_product == 0:
            return 0.0
        
        return dot_product / norm_product
    
    def hybrid_search(self,
                     query: str,
                     session_id: Optional[str] = None,
                     top_k: int = 10,
                     alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and BM25 scores
        
        Args:
            query: Search query
            session_id: Optional session ID to filter results
            top_k: Number of results to return
            alpha: Weight for vector similarity (1-alpha for BM25)
        """
        results = []
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        query_terms = self._tokenize(query)
        
        # Score all documents
        for doc_id, doc_content in self.documents.items():
            # Filter by session if specified
            if session_id and not doc_id.startswith(session_id):
                continue
            
            # Calculate vector similarity
            vector_score = 0.0
            if query_embedding is not None and doc_id in self.vectors:
                vector_score = self._calculate_vector_similarity(
                    query_embedding,
                    self.vectors[doc_id]
                )
            
            # Calculate BM25 score
            bm25_score = self._calculate_bm25_score(query_terms, doc_id)
            
            # Normalize scores (simple min-max normalization)
            # In production, you'd want more sophisticated normalization
            normalized_vector = (vector_score + 1) / 2  # Cosine similarity is [-1, 1]
            normalized_bm25 = min(bm25_score / 10, 1.0)  # Cap at 10 for normalization
            
            # Combine scores
            final_score = alpha * normalized_vector + (1 - alpha) * normalized_bm25
            
            if final_score > 0.1:  # Threshold
                results.append({
                    "doc_id": doc_id,
                    "content": doc_content[:200],  # Truncate for display
                    "metadata": self.metadata.get(doc_id, {}),
                    "score": final_score,
                    "vector_score": vector_score,
                    "bm25_score": bm25_score
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def vector_search(self, 
                     query: str,
                     top_k: int = 10,
                     threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Perform pure vector similarity search"""
        query_embedding = self.generate_embedding(query)
        if query_embedding is None:
            return []
        
        results = []
        
        for doc_id, doc_vector in self.vectors.items():
            similarity = self._calculate_vector_similarity(query_embedding, doc_vector)
            
            if similarity > threshold:
                results.append({
                    "doc_id": doc_id,
                    "content": self.documents.get(doc_id, "")[:200],
                    "metadata": self.metadata.get(doc_id, {}),
                    "score": similarity
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def lexical_search(self,
                      query: str,
                      top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform pure BM25 lexical search"""
        query_terms = self._tokenize(query)
        results = []
        
        for doc_id in self.documents:
            score = self._calculate_bm25_score(query_terms, doc_id)
            
            if score > 0:
                results.append({
                    "doc_id": doc_id,
                    "content": self.documents[doc_id][:200],
                    "metadata": self.metadata.get(doc_id, {}),
                    "score": score
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def find_similar(self,
                    doc_id: str,
                    top_k: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document"""
        if doc_id not in self.vectors:
            return []
        
        target_vector = self.vectors[doc_id]
        results = []
        
        for other_id, other_vector in self.vectors.items():
            if other_id == doc_id:
                continue
            
            similarity = self._calculate_vector_similarity(target_vector, other_vector)
            
            if similarity > 0.5:
                results.append({
                    "doc_id": other_id,
                    "content": self.documents.get(other_id, "")[:200],
                    "metadata": self.metadata.get(other_id, {}),
                    "score": similarity
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def update_document(self,
                       doc_id: str,
                       content: str,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing document"""
        # Remove old BM25 stats
        if doc_id in self.doc_lengths:
            self.doc_count -= 1
            old_terms = set(self._tokenize(self.documents.get(doc_id, "")))
            for term in old_terms:
                self.term_doc_freqs[term] = max(0, self.term_doc_freqs[term] - 1)
        
        # Re-index
        return self.index_document(doc_id, content, metadata)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the index"""
        try:
            # Remove from document store
            if doc_id in self.documents:
                # Update BM25 stats
                old_terms = set(self._tokenize(self.documents[doc_id]))
                for term in old_terms:
                    self.term_doc_freqs[term] = max(0, self.term_doc_freqs[term] - 1)
                
                del self.documents[doc_id]
                self.doc_count = max(0, self.doc_count - 1)
            
            # Remove vector
            if doc_id in self.vectors:
                del self.vectors[doc_id]
            
            # Remove metadata
            if doc_id in self.metadata:
                del self.metadata[doc_id]
            
            # Remove document length
            if doc_id in self.doc_lengths:
                del self.doc_lengths[doc_id]
                # Recalculate average
                if self.doc_count > 0:
                    self.avg_doc_length = sum(self.doc_lengths.values()) / self.doc_count
            
            # Remove from Redis if available
            if self.redis_client:
                self.redis_client.delete(f"document:{doc_id}")
                self.redis_client.delete(f"vector:{doc_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def clear_index(self):
        """Clear the entire search index"""
        self.documents.clear()
        self.vectors.clear()
        self.metadata.clear()
        self.doc_lengths.clear()
        self.term_doc_freqs.clear()
        self.doc_count = 0
        self.avg_doc_length = 0
        
        logger.info("Search index cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search system statistics"""
        return {
            "total_documents": len(self.documents),
            "total_vectors": len(self.vectors),
            "average_doc_length": self.avg_doc_length,
            "unique_terms": len(self.term_doc_freqs),
            "vector_dimension": self.vector_dim,
            "index_size_bytes": sum(
                len(content.encode()) for content in self.documents.values()
            )
        }
    
    def batch_index(self, documents: List[Dict[str, Any]]) -> int:
        """Index multiple documents at once"""
        success_count = 0
        
        for doc in documents:
            doc_id = doc.get("id", str(hash(doc.get("content", ""))))
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            if self.index_document(doc_id, content, metadata):
                success_count += 1
        
        logger.info(f"Batch indexed {success_count}/{len(documents)} documents")
        return success_count
    
    def export_index(self) -> Dict[str, Any]:
        """Export the search index"""
        return {
            "documents": self.documents,
            "metadata": self.metadata,
            "vectors": {k: v.tolist() for k, v in self.vectors.items()},
            "statistics": {
                "doc_count": self.doc_count,
                "avg_doc_length": self.avg_doc_length,
                "term_doc_freqs": dict(self.term_doc_freqs)
            },
            "exported_at": datetime.now().isoformat()
        }
    
    def import_index(self, data: Dict[str, Any]):
        """Import a search index"""
        self.clear_index()
        
        # Import documents
        self.documents = data.get("documents", {})
        self.metadata = data.get("metadata", {})
        
        # Import vectors
        for doc_id, vector_list in data.get("vectors", {}).items():
            self.vectors[doc_id] = np.array(vector_list)
        
        # Import statistics
        stats = data.get("statistics", {})
        self.doc_count = stats.get("doc_count", 0)
        self.avg_doc_length = stats.get("avg_doc_length", 0)
        self.term_doc_freqs = defaultdict(int, stats.get("term_doc_freqs", {}))
        
        # Rebuild doc lengths
        for doc_id, content in self.documents.items():
            self.doc_lengths[doc_id] = len(self._tokenize(content))
        
        logger.info(f"Imported index with {len(self.documents)} documents")


# Singleton instance
_search_instance = None

def get_search_system(redis_client=None, embedding_model=None) -> HybridSearchSystem:
    """Get or create the search system singleton"""
    global _search_instance
    if _search_instance is None:
        _search_instance = HybridSearchSystem(redis_client, embedding_model)
    return _search_instance