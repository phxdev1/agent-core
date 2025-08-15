#!/usr/bin/env python3
"""
Direct Fact Verification System using Wikipedia API
No MCP server needed - direct Wikipedia access
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
import wikipediaapi
from dataclasses import dataclass
import re

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


class DirectWikipediaClient:
    """Direct Wikipedia API client"""
    
    def __init__(self, language='en'):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='agent-core/1.0 (https://github.com/user/agent-core)',
            language=language
        )
        logger.info("Wikipedia client initialized")
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search Wikipedia for relevant pages"""
        try:
            # Wikipedia-API doesn't have built-in search, so we'll try direct page access
            # and related pages
            results = []
            
            # Try direct page access
            page = self.wiki.page(query)
            if page.exists():
                results.append({
                    "title": page.title,
                    "url": page.fullurl,
                    "summary": page.summary[:500] if page.summary else ""
                })
            
            # Try variations of the query
            query_variations = [
                query.replace(" ", "_"),
                query.title(),
                query.lower(),
                query.upper()
            ]
            
            for variation in query_variations[:limit]:
                if variation != query:
                    page = self.wiki.page(variation)
                    if page.exists() and not any(r["title"] == page.title for r in results):
                        results.append({
                            "title": page.title,
                            "url": page.fullurl,
                            "summary": page.summary[:500] if page.summary else ""
                        })
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {e}")
            return []
    
    def get_page(self, title: str) -> Optional[wikipediaapi.WikipediaPage]:
        """Get a Wikipedia page"""
        try:
            page = self.wiki.page(title)
            if page.exists():
                return page
            return None
        except Exception as e:
            logger.error(f"Error getting Wikipedia page: {e}")
            return None
    
    def get_summary(self, title: str) -> Optional[str]:
        """Get page summary"""
        page = self.get_page(title)
        if page:
            return page.summary
        return None


class DirectFactVerificationSystem:
    """Fact verification using direct Wikipedia API"""
    
    def __init__(self):
        self.wikipedia = DirectWikipediaClient()
        self.fact_cache = {}
    
    async def verify_fact(self, claim: str) -> FactCheckResult:
        """Verify a factual claim using Wikipedia"""
        # Check cache
        if claim in self.fact_cache:
            return self.fact_cache[claim]
        
        # Extract key terms
        key_terms = self._extract_key_terms(claim)
        
        # Search Wikipedia
        evidence = []
        sources = []
        
        for term in key_terms[:3]:  # Check top 3 terms
            results = self.wikipedia.search(term, limit=2)
            for result in results:
                if result["summary"]:
                    evidence.append(result["summary"])
                    sources.append(f"Wikipedia: {result['title']}")
        
        # Analyze evidence
        verification = self._analyze_evidence(claim, evidence)
        
        # Create result
        result = FactCheckResult(
            claim=claim,
            verified=verification["verified"],
            confidence=verification["confidence"],
            evidence=evidence[:3],
            sources=sources[:3],
            corrections=verification.get("corrections")
        )
        
        # Cache result
        self.fact_cache[claim] = result
        
        return result
    
    async def get_accurate_info(self, topic: str) -> Dict[str, Any]:
        """Get accurate information about a topic"""
        page = self.wikipedia.get_page(topic)
        
        if not page:
            # Try variations
            for variation in [topic.title(), topic.lower(), topic.replace(" ", "_")]:
                page = self.wikipedia.get_page(variation)
                if page:
                    break
        
        if page:
            return {
                "topic": topic,
                "found": True,
                "title": page.title,
                "summary": page.summary[:1000] if page.summary else "",
                "url": page.fullurl,
                "sections": list(page.sections.keys())[:10] if page.sections else []
            }
        
        return {
            "topic": topic,
            "found": False,
            "message": "No Wikipedia page found for this topic"
        }
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'was', 'were', 'been', 'be', 'have',
                     'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                     'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
                     'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
                     'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
                     'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
                     'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against',
                     'between', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'to', 'from', 'up', 'down', 'out', 'off',
                     'over', 'under', 'again', 'further', 'then', 'once', 'built',
                     'true', 'false', 'check', 'fact', 'verify', 'won', 'prize'}
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Filter and prioritize
        key_terms = []
        
        # First, add proper nouns (capitalized words)
        for word in words:
            if word[0].isupper() and word.lower() not in stop_words:
                key_terms.append(word)
        
        # Then add numbers (years, etc.)
        numbers = re.findall(r'\b\d{4}\b', text)  # Years
        key_terms.extend(numbers)
        
        # Add remaining significant words
        for word in words:
            if word.lower() not in stop_words and len(word) > 3 and word not in key_terms:
                key_terms.append(word)
        
        return key_terms[:10]
    
    def _analyze_evidence(self, claim: str, evidence: List[str]) -> Dict[str, Any]:
        """Analyze evidence to verify claim"""
        if not evidence:
            return {
                "verified": False,
                "confidence": 0.0,
                "corrections": "No evidence found on Wikipedia"
            }
        
        # Combine evidence
        evidence_text = " ".join(evidence).lower()
        claim_lower = claim.lower()
        
        # Extract key facts from claim
        key_terms = self._extract_key_terms(claim)
        
        # Look for specific patterns
        matches = 0
        total_terms = len(key_terms)
        
        for term in key_terms:
            if term.lower() in evidence_text:
                matches += 1
        
        # Calculate confidence
        confidence = matches / max(total_terms, 1)
        
        # Check for specific date/year mentions
        years_in_claim = re.findall(r'\b\d{4}\b', claim)
        years_in_evidence = re.findall(r'\b\d{4}\b', evidence_text)
        
        if years_in_claim:
            year_match = any(year in years_in_evidence for year in years_in_claim)
            if year_match:
                confidence = min(confidence + 0.3, 1.0)
            else:
                confidence = max(confidence - 0.3, 0.0)
        
        # Determine verification
        verified = confidence > 0.5
        
        # Generate corrections
        corrections = None
        if not verified:
            if confidence > 0.2:
                corrections = "Some related information found but key details could not be verified"
            else:
                corrections = "Could not find relevant information to verify this claim"
        
        return {
            "verified": verified,
            "confidence": confidence,
            "corrections": corrections
        }
    
    async def fact_check_response(self, response: str) -> Tuple[str, List[FactCheckResult]]:
        """Fact-check an AI response"""
        # Extract potential factual claims
        claims = self._extract_claims(response)
        
        # Verify each
        results = []
        for claim in claims:
            result = await self.verify_fact(claim)
            results.append(result)
        
        # Annotate response
        annotated = response
        for result in results:
            if not result.verified and result.confidence < 0.3:
                annotated = annotated.replace(
                    result.claim,
                    f"[UNVERIFIED: {result.claim}]"
                )
            elif result.corrections:
                annotated = annotated.replace(
                    result.claim,
                    f"{result.claim} [Note: {result.corrections}]"
                )
        
        return annotated, results
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Look for factual patterns
            if len(sentence) > 20:
                if any(keyword in sentence.lower() for keyword in
                      ['invented', 'discovered', 'founded', 'built', 'created',
                       'born', 'died', 'won', 'prize', 'year', '19', '20',
                       'first', 'last', 'largest', 'smallest', 'oldest']):
                    claims.append(sentence)
        
        return claims[:5]


# Singleton
_fact_system = None

async def get_direct_fact_system() -> DirectFactVerificationSystem:
    """Get or create fact verification system"""
    global _fact_system
    if _fact_system is None:
        _fact_system = DirectFactVerificationSystem()
    return _fact_system