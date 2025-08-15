#!/usr/bin/env python3
"""
Fact Verification System using Wikipedia MCP
Verifies facts and provides accurate information from Wikipedia
"""

import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
import httpx
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.redis_logger import get_redis_logger

logger = get_redis_logger(__name__)


@dataclass
class FactCheckResult:
    """Result of a fact check"""
    claim: str
    verified: bool
    confidence: float
    evidence: List[str]
    sources: List[str]
    corrections: Optional[str] = None


class WikipediaMCPClient:
    """Client for Wikipedia MCP server"""
    
    def __init__(self, port: int = 5173):
        self.base_url = f"http://localhost:{port}"
        self.session = None
        self.server_process = None
        
    async def start_server(self):
        """Start the Wikipedia MCP server"""
        import subprocess
        try:
            # Start the server in the background
            self.server_process = subprocess.Popen(
                ["wikipedia-mcp"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Give it time to start
            await asyncio.sleep(2)
            logger.info("Wikipedia MCP server started")
        except Exception as e:
            logger.error(f"Failed to start Wikipedia MCP server: {e}")
            
    async def stop_server(self):
        """Stop the Wikipedia MCP server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None
            logger.info("Wikipedia MCP server stopped")
    
    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search Wikipedia for articles"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/call-tool",
                    json={
                        "method": "tools/call",
                        "params": {
                            "name": "search_wikipedia",
                            "arguments": {
                                "query": query,
                                "limit": limit
                            }
                        }
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("content", [])
                else:
                    logger.warning(f"Wikipedia search failed: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {e}")
            return []
    
    async def get_article(self, title: str) -> Optional[Dict[str, Any]]:
        """Get full article content"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/call-tool",
                    json={
                        "method": "tools/call",
                        "params": {
                            "name": "get_article",
                            "arguments": {
                                "title": title
                            }
                        }
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("content")
                else:
                    logger.warning(f"Failed to get article: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting article: {e}")
            return None
    
    async def get_summary(self, title: str) -> Optional[str]:
        """Get article summary"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/call-tool",
                    json={
                        "method": "tools/call",
                        "params": {
                            "name": "get_summary",
                            "arguments": {
                                "title": title
                            }
                        }
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("content", {}).get("summary")
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return None


class FactVerificationSystem:
    """System for verifying facts using Wikipedia and other sources"""
    
    def __init__(self):
        self.wikipedia_client = WikipediaMCPClient()
        self.fact_cache = {}  # Cache verified facts
        
    async def initialize(self):
        """Initialize the fact verification system"""
        await self.wikipedia_client.start_server()
        
    async def shutdown(self):
        """Shutdown the system"""
        await self.wikipedia_client.stop_server()
    
    async def verify_fact(self, claim: str) -> FactCheckResult:
        """
        Verify a factual claim using Wikipedia
        
        Args:
            claim: The claim to verify
            
        Returns:
            FactCheckResult with verification details
        """
        # Check cache first
        if claim in self.fact_cache:
            return self.fact_cache[claim]
        
        # Extract key terms from claim
        key_terms = self._extract_key_terms(claim)
        
        # Search Wikipedia for relevant articles
        search_results = await self.wikipedia_client.search(" ".join(key_terms), limit=3)
        
        if not search_results:
            return FactCheckResult(
                claim=claim,
                verified=False,
                confidence=0.0,
                evidence=[],
                sources=[],
                corrections="Could not find relevant information to verify this claim"
            )
        
        # Gather evidence from articles
        evidence = []
        sources = []
        
        for result in search_results:
            title = result.get("title", "")
            if title:
                # Get article summary for quick verification
                summary = await self.wikipedia_client.get_summary(title)
                if summary:
                    evidence.append(summary[:500])  # First 500 chars
                    sources.append(f"Wikipedia: {title}")
        
        # Analyze evidence to verify claim
        verification_result = self._analyze_evidence(claim, evidence)
        
        # Create result
        result = FactCheckResult(
            claim=claim,
            verified=verification_result["verified"],
            confidence=verification_result["confidence"],
            evidence=evidence[:3],  # Top 3 pieces of evidence
            sources=sources[:3],
            corrections=verification_result.get("corrections")
        )
        
        # Cache the result
        self.fact_cache[claim] = result
        
        return result
    
    async def get_accurate_info(self, topic: str) -> Dict[str, Any]:
        """
        Get accurate, verified information about a topic
        
        Args:
            topic: The topic to get information about
            
        Returns:
            Dictionary with verified information
        """
        # Search for the topic
        search_results = await self.wikipedia_client.search(topic, limit=1)
        
        if not search_results:
            return {
                "topic": topic,
                "found": False,
                "message": "No information found on this topic"
            }
        
        # Get the most relevant article
        title = search_results[0].get("title", "")
        article = await self.wikipedia_client.get_article(title)
        
        if not article:
            return {
                "topic": topic,
                "found": False,
                "message": "Could not retrieve article content"
            }
        
        # Extract key information
        return {
            "topic": topic,
            "found": True,
            "title": article.get("title", ""),
            "summary": article.get("extract", "")[:1000],  # First 1000 chars
            "url": article.get("url", ""),
            "categories": article.get("categories", [])[:5],
            "last_updated": article.get("timestamp", "")
        }
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for searching"""
        # Simple extraction - can be enhanced with NLP
        import re
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'was', 'were', 'been', 'be', 'have', 
                     'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                     'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
                     'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
                     'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
                     'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just'}
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out stop words and short words
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Also look for capitalized words (likely proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        key_terms.extend(proper_nouns)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        return unique_terms[:5]  # Top 5 terms
    
    def _analyze_evidence(self, claim: str, evidence: List[str]) -> Dict[str, Any]:
        """
        Analyze evidence to determine if claim is verified
        Simple heuristic-based analysis (can be enhanced with LLM)
        """
        if not evidence:
            return {
                "verified": False,
                "confidence": 0.0,
                "corrections": "No evidence found"
            }
        
        # Convert to lowercase for comparison
        claim_lower = claim.lower()
        evidence_text = " ".join(evidence).lower()
        
        # Extract key terms from claim
        claim_terms = self._extract_key_terms(claim)
        
        # Count how many key terms appear in evidence
        matches = 0
        for term in claim_terms:
            if term.lower() in evidence_text:
                matches += 1
        
        # Calculate confidence based on matches
        if claim_terms:
            confidence = matches / len(claim_terms)
        else:
            confidence = 0.0
        
        # Determine if verified (simple threshold)
        verified = confidence > 0.5
        
        # Generate corrections if not verified
        corrections = None
        if not verified and confidence > 0.2:
            corrections = "The claim could not be fully verified. Some related information was found but key details may be incorrect."
        elif not verified:
            corrections = "No supporting evidence found for this claim in Wikipedia."
        
        return {
            "verified": verified,
            "confidence": confidence,
            "corrections": corrections
        }
    
    async def fact_check_response(self, response: str) -> Tuple[str, List[FactCheckResult]]:
        """
        Fact-check an AI response and annotate with verifications
        
        Args:
            response: The AI response to fact-check
            
        Returns:
            Tuple of (annotated response, list of fact check results)
        """
        # Extract factual claims from response
        claims = self._extract_claims(response)
        
        # Verify each claim
        results = []
        for claim in claims:
            result = await self.verify_fact(claim)
            results.append(result)
        
        # Annotate response with fact-check markers
        annotated = response
        for result in results:
            if not result.verified and result.confidence < 0.3:
                # Mark unverified claims
                annotated = annotated.replace(
                    result.claim,
                    f"[UNVERIFIED: {result.claim}]"
                )
            elif result.corrections:
                # Add corrections
                annotated = annotated.replace(
                    result.claim,
                    f"{result.claim} [Note: {result.corrections}]"
                )
        
        return annotated, results
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Simple sentence-based extraction
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Filter for sentences that look like factual claims
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Skip very short sentences
                # Look for patterns that suggest factual claims
                if any(pattern in sentence.lower() for pattern in 
                      ['is ', 'was ', 'were ', 'are ', 'has ', 'have ', 'had ',
                       'invented', 'discovered', 'founded', 'created', 'built',
                       'born', 'died', 'happened', 'occurred']):
                    claims.append(sentence)
        
        return claims[:5]  # Limit to 5 claims for performance


# Singleton instance
_fact_system = None

async def get_fact_verification_system() -> FactVerificationSystem:
    """Get or create the fact verification system singleton"""
    global _fact_system
    if _fact_system is None:
        _fact_system = FactVerificationSystem()
        await _fact_system.initialize()
    return _fact_system