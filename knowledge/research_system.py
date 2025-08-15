#!/usr/bin/env python3
"""
Research System for Knowledge Acquisition
Integrates SERPAPI, Arxiv, Google Scholar, and other sources to build RAG knowledge base
"""

import os
import json
import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import re
from dataclasses import dataclass
import arxiv
import requests
from serpapi import GoogleSearch

# Import our systems - use relative imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.unified_memory_system import UnifiedMemorySystem, StepType
from knowledge.hybrid_search_system import SimpleSearchSystem
from knowledge.knowledge_graph_enhanced import get_enhanced_knowledge_graph
from utils.redis_logger import get_redis_logger
from utils.pdf_extractor import PDFExtractor

logger = get_redis_logger(__name__)


@dataclass
class ResearchDocument:
    """Represents a research document"""
    doc_id: str
    title: str
    authors: List[str]
    abstract: str
    content: str
    source: str  # arxiv, scholar, web, etc.
    url: str
    published_date: Optional[str]
    citations: Optional[int]
    metadata: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.doc_id:
            # Generate ID from title and source
            self.doc_id = hashlib.md5(f"{self.title}_{self.source}".encode()).hexdigest()[:12]


class ResearchSystem:
    """System for conducting research and building knowledge base"""
    
    def __init__(self, serpapi_key: Optional[str] = None, llm_model: str = "mistralai/mistral-medium-3.1"):
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY")
        self.search_system = SimpleSearchSystem()
        self.knowledge_graph = get_enhanced_knowledge_graph()
        self.memory = get_unified_memory()
        self.pdf_extractor = PDFExtractor()
        
        # LLM configuration for PDF processing
        self.llm_model = llm_model  # Use mistralai/mistral-medium-3.1 for PDF summarization
        
        # Research configuration
        self.max_results_per_source = 10
        self.summary_max_length = 500
        self.extract_pdfs = True  # Enable full PDF extraction
        
        # Concurrency settings
        self.max_concurrent_papers = 5  # Process up to 5 papers simultaneously
        self.max_concurrent_sources = 3  # Query up to 3 sources simultaneously
        self.paper_semaphore = asyncio.Semaphore(self.max_concurrent_papers)
        self.source_semaphore = asyncio.Semaphore(self.max_concurrent_sources)
        
        logger.info(f"Research System initialized with LLM model: {self.llm_model}")
    
    async def research_topic(self, 
                            topic: str,
                            sources: List[str] = None,
                            max_documents: int = 20) -> Dict[str, Any]:
        """
        Research a topic across multiple sources
        
        Args:
            topic: Research topic/query
            sources: List of sources to use (arxiv, scholar, web, etc.)
            max_documents: Maximum documents to retrieve
        
        Returns:
            Research results with documents and synthesis
        """
        if sources is None:
            sources = ["arxiv", "scholar", "web"]
        
        logger.info(f"Starting research on topic: {topic}")
        
        # Store research session
        research_id = hashlib.md5(f"{topic}_{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        
        # Collect documents from each source in parallel
        source_tasks = []
        
        for source in sources:
            if source == "arxiv":
                source_tasks.append(self._search_source_with_semaphore(
                    "arxiv", 
                    self._search_arxiv(topic, max_results=min(10, max_documents))
                ))
            elif source == "scholar" and self.serpapi_key:
                source_tasks.append(self._search_source_with_semaphore(
                    "scholar",
                    self._search_google_scholar(topic, max_results=min(10, max_documents))
                ))
            elif source == "web" and self.serpapi_key:
                source_tasks.append(self._search_source_with_semaphore(
                    "web",
                    self._search_web(topic, max_results=min(10, max_documents))
                ))
        
        # Execute all source searches in parallel
        logger.info(f"Searching {len(source_tasks)} sources in parallel...")
        source_results = await asyncio.gather(*source_tasks, return_exceptions=True)
        
        # Process results
        all_documents = []
        for result in source_results:
            if isinstance(result, Exception):
                logger.error(f"Source search failed: {result}")
            elif isinstance(result, tuple):
                source_name, docs = result
                all_documents.extend(docs)
                logger.info(f"Retrieved {len(docs)} documents from {source_name}")
            elif isinstance(result, list):
                all_documents.extend(result)
        
        # Limit to max_documents
        all_documents = all_documents[:max_documents]
        
        # Index documents for semantic search
        indexed_count = self._index_documents(all_documents)
        
        # Extract knowledge and build graph
        knowledge_extracted = self._extract_knowledge(all_documents, topic)
        
        # Synthesize findings
        synthesis = self._synthesize_research(all_documents, topic)
        
        # Store research session in memory
        self.memory.add_conversation_step(
            StepType.MEMORY_STORED,
            {
                "research_id": research_id,
                "topic": topic,
                "documents_found": len(all_documents),
                "synthesis": synthesis
            },
            {"research_session": True}
        )
        
        return {
            "research_id": research_id,
            "topic": topic,
            "documents": all_documents,
            "total_documents": len(all_documents),
            "indexed": indexed_count,
            "knowledge_extracted": knowledge_extracted,
            "synthesis": synthesis,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _search_arxiv(self, query: str, max_results: int = 10) -> List[ResearchDocument]:
        """Search ArXiv for papers with parallel PDF extraction"""
        documents = []
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            # Collect all search results first
            search_results = list(search.results())
            logger.info(f"Found {len(search_results)} ArXiv papers for query: {query}")
            
            # Process papers in parallel if PDF extraction is enabled
            if self.extract_pdfs:
                # Create tasks for parallel PDF processing
                pdf_tasks = []
                for result in search_results:
                    pdf_tasks.append(self._process_arxiv_paper(result))
                
                # Process all papers concurrently (with controlled concurrency)
                documents = await self._process_papers_parallel(pdf_tasks)
                logger.info(f"Processed {len(documents)} papers with PDF extraction")
            else:
                # No PDF extraction - just use abstracts
                for result in search_results:
                    doc = ResearchDocument(
                        doc_id=result.entry_id.split('/')[-1],
                        title=result.title,
                        authors=[author.name for author in result.authors],
                        abstract=result.summary,
                        content=result.summary,
                        source="arxiv",
                        url=result.entry_id,
                        published_date=result.published.isoformat() if result.published else None,
                        citations=None,
                        metadata={
                            "categories": result.categories,
                            "primary_category": result.primary_category,
                            "comment": result.comment,
                            "journal_ref": result.journal_ref,
                            "pdf_extracted": False
                        }
                    )
                    documents.append(doc)
                
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
        
        return documents
    
    async def _search_source_with_semaphore(self, source_name: str, search_coro) -> Tuple[str, List[ResearchDocument]]:
        """Execute source search with semaphore for rate limiting"""
        async with self.source_semaphore:
            try:
                docs = await search_coro
                return (source_name, docs)
            except Exception as e:
                logger.error(f"Error searching {source_name}: {e}")
                return (source_name, [])
    
    async def _process_arxiv_paper(self, result) -> Optional[ResearchDocument]:
        """Process a single ArXiv paper with PDF extraction"""
        async with self.paper_semaphore:
            try:
                content = result.summary
                pdf_url = result.entry_id
                
                # Extract full PDF content
                logger.info(f"Processing: {result.title[:50]}...")
                pdf_data = await self.pdf_extractor.process_arxiv_paper(pdf_url)
                
                if pdf_data.get("success"):
                    content = pdf_data.get("text", result.summary)
                    logger.info(f"Extracted {pdf_data.get('word_count', 0)} words from {result.title[:30]}")
                    
                    metadata = {
                        "categories": result.categories,
                        "primary_category": result.primary_category,
                        "comment": result.comment,
                        "journal_ref": result.journal_ref,
                        "pdf_extracted": True,
                        "word_count": pdf_data.get("word_count", 0),
                        "sections": list(pdf_data.get("sections", {}).keys()),
                        "key_information": pdf_data.get("key_information", {}),
                        "llm_summary": pdf_data.get("llm_summary"),
                        "summary_model": pdf_data.get("summary_model")
                    }
                else:
                    logger.warning(f"PDF extraction failed for {result.title[:30]}")
                    metadata = {
                        "categories": result.categories,
                        "primary_category": result.primary_category,
                        "comment": result.comment,
                        "journal_ref": result.journal_ref,
                        "pdf_extracted": False
                    }
                
                return ResearchDocument(
                    doc_id=result.entry_id.split('/')[-1],
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    content=content,
                    source="arxiv",
                    url=result.entry_id,
                    published_date=result.published.isoformat() if result.published else None,
                    citations=None,
                    metadata=metadata
                )
                
            except Exception as e:
                logger.error(f"Error processing paper {result.title[:30]}: {e}")
                return None
    
    async def _process_papers_parallel(self, tasks: List) -> List[ResearchDocument]:
        """Process multiple papers in parallel with controlled concurrency"""
        logger.info(f"Processing {len(tasks)} papers in parallel (max {self.max_concurrent_papers} concurrent)")
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed papers
        documents = []
        failed_count = 0
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Paper processing failed: {result}")
                failed_count += 1
            elif result is not None:
                documents.append(result)
            else:
                failed_count += 1
        
        if failed_count > 0:
            logger.warning(f"{failed_count} papers failed to process")
        
        return documents
    
    async def _search_google_scholar(self, query: str, max_results: int = 10) -> List[ResearchDocument]:
        """Search Google Scholar using SERPAPI"""
        documents = []
        
        if not self.serpapi_key:
            logger.warning("SERPAPI key not configured, skipping Google Scholar")
            return documents
        
        try:
            search = GoogleSearch({
                "engine": "google_scholar",
                "q": query,
                "num": max_results,
                "api_key": self.serpapi_key
            })
            
            results = search.get_dict()
            
            for result in results.get("organic_results", [])[:max_results]:
                doc = ResearchDocument(
                    doc_id=result.get("result_id", hashlib.md5(result.get("title", "").encode()).hexdigest()[:12]),
                    title=result.get("title", ""),
                    authors=self._parse_authors(result.get("publication_info", {}).get("authors", [])),
                    abstract=result.get("snippet", ""),
                    content=result.get("snippet", ""),
                    source="scholar",
                    url=result.get("link", ""),
                    published_date=result.get("publication_info", {}).get("summary", ""),
                    citations=result.get("cited_by", {}).get("value"),
                    metadata={
                        "type": result.get("type", ""),
                        "inline_links": result.get("inline_links", {}),
                        "resources": result.get("resources", [])
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Google Scholar search failed: {e}")
        
        return documents
    
    async def _search_web(self, query: str, max_results: int = 10) -> List[ResearchDocument]:
        """Search web using SERPAPI"""
        documents = []
        
        if not self.serpapi_key:
            logger.warning("SERPAPI key not configured, skipping web search")
            return documents
        
        try:
            search = GoogleSearch({
                "q": query,
                "num": max_results,
                "api_key": self.serpapi_key
            })
            
            results = search.get_dict()
            
            for result in results.get("organic_results", [])[:max_results]:
                doc = ResearchDocument(
                    doc_id=hashlib.md5(result.get("link", "").encode()).hexdigest()[:12],
                    title=result.get("title", ""),
                    authors=[],
                    abstract=result.get("snippet", ""),
                    content=result.get("snippet", ""),
                    source="web",
                    url=result.get("link", ""),
                    published_date=result.get("date", ""),
                    citations=None,
                    metadata={
                        "position": result.get("position"),
                        "displayed_link": result.get("displayed_link"),
                        "source": result.get("source")
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Web search failed: {e}")
        
        return documents
    
    def _parse_authors(self, authors_data: Any) -> List[str]:
        """Parse authors from various formats"""
        if isinstance(authors_data, list):
            return [a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in authors_data]
        elif isinstance(authors_data, str):
            return [a.strip() for a in authors_data.split(",")]
        return []
    
    def _index_documents(self, documents: List[ResearchDocument]) -> int:
        """Index documents for semantic search"""
        indexed = 0
        
        for doc in documents:
            try:
                # Create searchable content
                content = f"{doc.title}\n{' '.join(doc.authors)}\n{doc.abstract}"
                
                # Index in search system
                success = self.search_system.index_document(
                    doc_id=doc.doc_id,
                    content=content,
                    metadata={
                        "title": doc.title,
                        "source": doc.source,
                        "url": doc.url,
                        "published": doc.published_date,
                        "citations": doc.citations,
                        "timestamp": doc.timestamp
                    }
                )
                
                if success:
                    indexed += 1
                    
            except Exception as e:
                logger.error(f"Failed to index document {doc.doc_id}: {e}")
        
        logger.info(f"Indexed {indexed}/{len(documents)} documents")
        return indexed
    
    def _extract_knowledge(self, documents: List[ResearchDocument], topic: str) -> Dict[str, Any]:
        """Extract knowledge and build knowledge graph"""
        entities_extracted = 0
        relationships_created = 0
        
        # Create topic entity
        topic_entity_id = self.knowledge_graph.add_entity(
            name=topic,
            entity_type="research_topic",
            properties={"timestamp": datetime.now().isoformat()}
        )
        
        for doc in documents:
            try:
                # Extract entities from document
                doc_text = f"{doc.title} {doc.abstract}"
                entities = self.knowledge_graph.extract_entities(doc_text)
                
                # Create document entity
                doc_entity_id = self.knowledge_graph.add_entity(
                    name=doc.title[:100],
                    entity_type="document",
                    properties={
                        "source": doc.source,
                        "url": doc.url,
                        "doc_id": doc.doc_id
                    }
                )
                
                # Link document to topic
                self.knowledge_graph.add_relationship(
                    doc_entity_id,
                    topic_entity_id,
                    "related_to"
                )
                relationships_created += 1
                
                # Add author entities
                for author in doc.authors:
                    author_id = self.knowledge_graph.add_entity(
                        name=author,
                        entity_type="person",
                        properties={"role": "author"}
                    )
                    self.knowledge_graph.add_relationship(
                        author_id,
                        doc_entity_id,
                        "authored"
                    )
                    relationships_created += 1
                    entities_extracted += 1
                
                # Add extracted entities
                for entity in entities[:10]:  # Limit per document
                    entity_id = self.knowledge_graph.add_entity(
                        entity["name"],
                        entity["type"],
                        entity.get("properties", {})
                    )
                    self.knowledge_graph.add_relationship(
                        entity_id,
                        doc_entity_id,
                        "mentioned_in"
                    )
                    relationships_created += 1
                    entities_extracted += 1
                    
            except Exception as e:
                logger.error(f"Failed to extract knowledge from {doc.doc_id}: {e}")
        
        return {
            "entities_extracted": entities_extracted,
            "relationships_created": relationships_created,
            "topic_entity_id": topic_entity_id
        }
    
    def _synthesize_research(self, documents: List[ResearchDocument], topic: str) -> Dict[str, Any]:
        """Synthesize research findings"""
        if not documents:
            return {"summary": "No documents found", "key_findings": []}
        
        # Group by source
        by_source = {}
        for doc in documents:
            if doc.source not in by_source:
                by_source[doc.source] = []
            by_source[doc.source].append(doc)
        
        # Find most cited
        cited_docs = [d for d in documents if d.citations]
        most_cited = sorted(cited_docs, key=lambda x: x.citations or 0, reverse=True)[:3]
        
        # Extract key themes (simple keyword extraction)
        all_text = " ".join([f"{d.title} {d.abstract}" for d in documents])
        words = re.findall(r'\b[a-z]+\b', all_text.lower())
        
        # Filter common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been'}
        significant_words = [w for w in words if w not in stop_words and len(w) > 4]
        
        # Count frequencies
        from collections import Counter
        word_freq = Counter(significant_words)
        key_themes = [word for word, _ in word_freq.most_common(10)]
        
        # Create synthesis
        synthesis = {
            "summary": f"Research on '{topic}' yielded {len(documents)} documents from {len(by_source)} sources",
            "sources_breakdown": {source: len(docs) for source, docs in by_source.items()},
            "key_themes": key_themes,
            "most_cited_papers": [
                {
                    "title": doc.title,
                    "citations": doc.citations,
                    "url": doc.url
                } for doc in most_cited
            ],
            "date_range": {
                "earliest": min([d.published_date for d in documents if d.published_date], default="N/A"),
                "latest": max([d.published_date for d in documents if d.published_date], default="N/A")
            },
            "total_authors": len(set([author for doc in documents for author in doc.authors]))
        }
        
        return synthesis
    
    async def query_knowledge(self, 
                             query: str,
                             use_rag: bool = True,
                             top_k: int = 5) -> Dict[str, Any]:
        """
        Query the knowledge base with RAG
        
        Args:
            query: Query string
            use_rag: Whether to use RAG retrieval
            top_k: Number of documents to retrieve
        
        Returns:
            Query results with relevant documents
        """
        results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "documents": [],
            "context": ""
        }
        
        if use_rag:
            # Perform hybrid search
            search_results = self.search_system.hybrid_search(
                query=query,
                top_k=top_k,
                alpha=0.6  # Balance between semantic and lexical
            )
            
            # Format results
            for result in search_results:
                results["documents"].append({
                    "doc_id": result["doc_id"],
                    "content": result["content"],
                    "score": result["score"],
                    "metadata": result.get("metadata", {})
                })
            
            # Build context for LLM
            if search_results:
                context_parts = []
                for i, result in enumerate(search_results[:3], 1):
                    metadata = result.get("metadata", {})
                    context_parts.append(
                        f"[{i}] {metadata.get('title', 'Document')} "
                        f"(Source: {metadata.get('source', 'unknown')})\n"
                        f"{result['content'][:200]}..."
                    )
                results["context"] = "\n\n".join(context_parts)
        
        # Also search knowledge graph
        kg_results = self.knowledge_graph.search_entities(query)
        results["knowledge_graph_hits"] = len(kg_results)
        
        return results
    
    def get_research_statistics(self) -> Dict[str, Any]:
        """Get statistics about research and knowledge base"""
        search_stats = self.search_system.get_statistics()
        kg_stats = self.knowledge_graph.get_statistics()
        
        return {
            "documents_indexed": search_stats.get("total_documents", 0),
            "vectors_stored": search_stats.get("total_vectors", 0),
            "unique_terms": search_stats.get("unique_terms", 0),
            "entities": kg_stats.get("total_entities", 0),
            "relationships": kg_stats.get("total_relationships", 0),
            "index_size_bytes": search_stats.get("index_size_bytes", 0),
            "last_updated": datetime.now().isoformat()
        }
    
    async def continuous_learning(self, 
                                 topics: List[str],
                                 interval_hours: int = 24,
                                 max_docs_per_topic: int = 10):
        """
        Continuously research topics and update knowledge base
        
        Args:
            topics: List of topics to research
            interval_hours: Hours between research cycles
            max_docs_per_topic: Maximum documents per topic
        """
        logger.info(f"Starting continuous learning for {len(topics)} topics")
        
        while True:
            for topic in topics:
                try:
                    logger.info(f"Researching: {topic}")
                    
                    # Research the topic
                    results = await self.research_topic(
                        topic=topic,
                        max_documents=max_docs_per_topic
                    )
                    
                    logger.info(f"Completed research on {topic}: "
                              f"{results['total_documents']} documents indexed")
                    
                    # Wait between topics to avoid rate limiting
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Error researching {topic}: {e}")
            
            # Wait for next cycle
            logger.info(f"Research cycle complete. Waiting {interval_hours} hours...")
            await asyncio.sleep(interval_hours * 3600)


# Singleton instance
_research_instance = None

def get_research_system(serpapi_key: Optional[str] = None) -> ResearchSystem:
    """Get or create the research system singleton"""
    global _research_instance
    if _research_instance is None:
        _research_instance = ResearchSystem(serpapi_key)
    return _research_instance


if __name__ == "__main__":
    import sys
    
    async def test_research():
        """Test the research system"""
        # Get SERPAPI key from environment or command line
        serpapi_key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("SERPAPI_KEY")
        
        research = ResearchSystem(serpapi_key)
        
        # Test research
        topic = "transformer neural networks attention mechanism"
        print(f"Researching: {topic}")
        
        results = await research.research_topic(
            topic=topic,
            sources=["arxiv"],  # Start with just arxiv (free)
            max_documents=5
        )
        
        print(f"\nFound {results['total_documents']} documents")
        print(f"Indexed: {results['indexed']}")
        print(f"Knowledge extracted: {results['knowledge_extracted']}")
        
        print("\nSynthesis:")
        print(json.dumps(results['synthesis'], indent=2))
        
        # Test RAG query
        query = "What is self-attention?"
        print(f"\nQuerying: {query}")
        
        query_results = await research.query_knowledge(query, use_rag=True)
        
        print(f"Found {len(query_results['documents'])} relevant documents")
        if query_results['context']:
            print("\nContext for LLM:")
            print(query_results['context'])
    
    asyncio.run(test_research())