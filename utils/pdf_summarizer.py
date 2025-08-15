#!/usr/bin/env python3
"""
PDF Summarization using Lower-Tier LLM
Uses mistralai/mistral-medium-3.1 for efficient PDF processing
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
import aiohttp
from .redis_logger import get_redis_logger
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import re
import hashlib
from dataclasses import dataclass

# Load environment variables
load_dotenv()

logger = get_redis_logger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    index: int
    start_pos: int
    end_pos: int
    token_count: int
    section_title: Optional[str] = None


class RecursiveTextChunker:
    """Recursively chunk text respecting natural boundaries"""
    
    def __init__(self, max_chunk_size: int = 3000, overlap_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # Hierarchy of separators (from most to least preferred)
        self.separators = [
            "\n\n\n",  # Multiple blank lines (section breaks)
            "\n\n",    # Paragraph breaks
            ". ",      # Sentence endings
            "! ",      # Exclamation endings
            "? ",      # Question endings
            "; ",      # Semicolon breaks
            ", ",      # Comma breaks
            " ",       # Word breaks
            ""         # Character level (last resort)
        ]
    
    def chunk_text(self, text: str, section_title: Optional[str] = None) -> List[TextChunk]:
        """
        Recursively chunk text into manageable pieces
        
        Args:
            text: Text to chunk
            section_title: Optional section title for context
        
        Returns:
            List of TextChunk objects
        """
        chunks = []
        current_position = 0
        chunk_index = 0
        
        # Clean text
        text = text.strip()
        
        while current_position < len(text):
            # Determine chunk end position
            chunk_end = min(current_position + self.max_chunk_size, len(text))
            
            # If not at the end, find a good break point
            if chunk_end < len(text):
                chunk_text = text[current_position:chunk_end]
                break_point = self._find_best_break(chunk_text)
                
                if break_point > 0:
                    chunk_end = current_position + break_point
            
            # Extract chunk
            chunk_content = text[current_position:chunk_end].strip()
            
            if chunk_content:
                chunk = TextChunk(
                    content=chunk_content,
                    index=chunk_index,
                    start_pos=current_position,
                    end_pos=chunk_end,
                    token_count=self._estimate_tokens(chunk_content),
                    section_title=section_title
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move position with overlap
            if chunk_end < len(text):
                # Look for overlap start at a natural boundary
                overlap_start = max(0, chunk_end - self.overlap_size)
                overlap_text = text[overlap_start:chunk_end]
                
                # Find a good starting point in the overlap
                for separator in self.separators[:3]:  # Only use major separators
                    sep_pos = overlap_text.rfind(separator)
                    if sep_pos > 0:
                        current_position = overlap_start + sep_pos + len(separator)
                        break
                else:
                    current_position = chunk_end
            else:
                current_position = chunk_end
        
        return chunks
    
    def _find_best_break(self, text: str) -> int:
        """
        Find the best position to break text
        
        Args:
            text: Text to find break in
        
        Returns:
            Best break position
        """
        # Try each separator in order of preference
        for separator in self.separators:
            if not separator:
                continue
                
            # Find last occurrence of separator
            position = text.rfind(separator)
            
            # Make sure we're not breaking too early (at least 50% of chunk)
            if position > len(text) * 0.5:
                return position + len(separator)
        
        # If no good break found, return full length
        return len(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: ~1 token per 4 characters
        return len(text) // 4


class PDFSummarizer:
    """Summarize PDF content using mistralai/mistral-medium-3.1 with parallel processing"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mistralai/mistral-medium-3.1"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("MISTRAL_API_KEY")
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Chunking settings
        self.chunker = RecursiveTextChunker(max_chunk_size=3000, overlap_size=200)
        self.max_concurrent_requests = 3
        
        # Summarization settings
        self.chunk_summary_length = 300  # words per chunk
        self.final_summary_length = 1000  # words for final summary
        
        # Cache for summaries
        self.summary_cache = {}
        
        # Rate limiting
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        logger.info(f"PDF Summarizer initialized with model: {self.model}")
    
    async def summarize_text(self, text: str, focus: Optional[str] = None) -> Dict[str, Any]:
        """
        Summarize text using chunking and parallel processing
        
        Args:
            text: Text to summarize
            focus: Optional focus area for summarization
        
        Returns:
            Summary with key points
        """
        if not self.api_key:
            logger.warning("No API key configured for summarization")
            return self._fallback_summary(text)
        
        # Check cache
        cache_key = hashlib.md5(f"{text[:500]}_{focus}".encode()).hexdigest()
        if cache_key in self.summary_cache:
            logger.info("Using cached summary")
            return self.summary_cache[cache_key]
        
        try:
            # Chunk the text
            chunks = self.chunker.chunk_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            if len(chunks) == 1:
                # Single chunk - direct summarization
                result = await self._summarize_chunk(chunks[0], focus)
            else:
                # Multiple chunks - parallel processing
                result = await self._summarize_parallel(chunks, focus)
            
            # Cache result
            self.summary_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return self._fallback_summary(text)
    
    async def _summarize_chunk(self, chunk: TextChunk, focus: Optional[str] = None) -> Dict[str, Any]:
        """Summarize a single chunk"""
        async with self.semaphore:
            # Prepare prompt
            context = f"Section: {chunk.section_title}" if chunk.section_title else "Document excerpt"
            
            if focus:
                prompt = f"""Summarize this {context} focusing on {focus}.
                
Extract key points about:
- Main ideas
- Important findings
- Methodologies mentioned
- Conclusions

Text:
{chunk.content}

Provide a concise summary (max {self.chunk_summary_length} words)."""
            else:
                prompt = f"""Summarize this {context}.

Extract:
- Main ideas
- Key findings
- Important details

Text:
{chunk.content}

Provide a concise summary (max {self.chunk_summary_length} words)."""

            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/rpi-agent",
                        "X-Title": "RPI Research Agent"
                    }
                    
                    payload = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a research paper summarizer. Be concise and accurate."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 500
                    }
                    
                    async with session.post(
                        self.api_url, 
                        headers=headers, 
                        json=payload, 
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return {
                                "success": True,
                                "summary": result['choices'][0]['message']['content'],
                                "chunk_index": chunk.index,
                                "tokens_used": result.get('usage', {})
                            }
                        else:
                            error_text = await response.text()
                            logger.error(f"API error for chunk {chunk.index}: {response.status}")
                            return {
                                "success": False,
                                "summary": chunk.content[:500],
                                "chunk_index": chunk.index,
                                "error": error_text
                            }
                            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for chunk {chunk.index}")
                return {
                    "success": False,
                    "summary": chunk.content[:500],
                    "chunk_index": chunk.index,
                    "error": "timeout"
                }
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.index}: {e}")
                return {
                    "success": False,
                    "summary": chunk.content[:500],
                    "chunk_index": chunk.index,
                    "error": str(e)
                }
    
    async def _summarize_parallel(self, chunks: List[TextChunk], focus: Optional[str] = None) -> Dict[str, Any]:
        """Summarize multiple chunks in parallel"""
        logger.info(f"Starting parallel summarization of {len(chunks)} chunks")
        
        # Create tasks for parallel processing
        tasks = [self._summarize_chunk(chunk, focus) for chunk in chunks]
        
        # Wait for all summaries
        chunk_summaries = await asyncio.gather(*tasks)
        
        # Filter successful summaries
        successful_summaries = [s for s in chunk_summaries if s.get("success")]
        failed_count = len(chunk_summaries) - len(successful_summaries)
        
        if failed_count > 0:
            logger.warning(f"{failed_count} chunks failed to summarize")
        
        if not successful_summaries:
            return self._fallback_summary("\n".join([c.content for c in chunks[:3]]))
        
        # Combine chunk summaries
        combined_text = "\n\n".join([
            f"[Part {s['chunk_index']+1}] {s['summary']}" 
            for s in sorted(successful_summaries, key=lambda x: x['chunk_index'])
        ])
        
        # Generate final summary from chunk summaries
        final_summary = await self._generate_final_summary(combined_text, focus)
        
        # Calculate total tokens
        total_tokens = sum(s.get('tokens_used', {}).get('total_tokens', 0) for s in successful_summaries)
        
        return {
            "success": True,
            "summary": final_summary,
            "model_used": self.model,
            "chunks_processed": len(successful_summaries),
            "chunks_failed": failed_count,
            "tokens_used": {"total_tokens": total_tokens},
            "chunk_summaries": [s['summary'] for s in sorted(successful_summaries, key=lambda x: x['chunk_index'])]
        }
    
    async def _generate_final_summary(self, combined_summaries: str, focus: Optional[str] = None) -> str:
        """Generate final summary from chunk summaries"""
        if not self.api_key:
            return combined_summaries[:self.final_summary_length * 5]
        
        prompt = f"""Based on these section summaries, create a comprehensive summary{f' focusing on {focus}' if focus else ''}.

Section Summaries:
{combined_summaries}

Create a coherent summary that:
1. Synthesizes the main points
2. Highlights key findings
3. Notes important methodologies
4. Captures conclusions

Maximum {self.final_summary_length} words."""

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a research synthesizer. Create coherent summaries from multiple sources."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 1500
                }
                
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        logger.error(f"Failed to generate final summary: {response.status}")
                        return combined_summaries[:self.final_summary_length * 5]
                        
        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            return combined_summaries[:self.final_summary_length * 5]
    
    def _fallback_summary(self, text: str) -> Dict[str, Any]:
        """Fallback summary when LLM is not available"""
        # Use recursive chunker for smart extraction
        chunks = self.chunker.chunk_text(text)
        
        summary_parts = []
        if chunks:
            # Take first chunk (usually abstract/intro)
            summary_parts.append(chunks[0].content[:500])
            
            # Take last chunk (usually conclusion)
            if len(chunks) > 1:
                summary_parts.append(chunks[-1].content[:500])
        
        return {
            "success": False,
            "summary": "\n\n".join(summary_parts) if summary_parts else text[:1000],
            "model_used": "fallback_extractive",
            "tokens_used": {}
        }
    
    async def summarize_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        """
        Summarize each section of a paper in parallel
        
        Args:
            sections: Dictionary of section_name -> section_text
        
        Returns:
            Dictionary of section_name -> summary
        """
        tasks = []
        section_names = []
        
        for section_name, section_text in sections.items():
            if section_text and len(section_text) > 100:
                # Create chunk for section
                chunk = TextChunk(
                    content=section_text[:self.chunker.max_chunk_size],
                    index=len(tasks),
                    start_pos=0,
                    end_pos=len(section_text),
                    token_count=len(section_text) // 4,
                    section_title=section_name
                )
                tasks.append(self._summarize_chunk(chunk, focus=f"the {section_name} section"))
                section_names.append(section_name)
        
        if not tasks:
            return sections
        
        # Process all sections in parallel
        results = await asyncio.gather(*tasks)
        
        # Map results back to section names
        summaries = {}
        for name, result in zip(section_names, results):
            if result.get("success"):
                summaries[name] = result["summary"]
            else:
                summaries[name] = sections[name][:200] + "..."
        
        # Add unchanged short sections
        for name, text in sections.items():
            if name not in summaries:
                summaries[name] = text
        
        return summaries


async def test_summarizer():
    """Test the PDF summarizer with chunking"""
    summarizer = PDFSummarizer()
    
    # Test with a longer text that needs chunking
    test_text = """
    Attention Is All You Need
    
    Abstract:
    The dominant sequence transduction models are based on complex recurrent or convolutional neural networks 
    that include an encoder and a decoder. The best performing models also connect the encoder and decoder 
    through an attention mechanism. We propose a new simple network architecture, the Transformer, based 
    solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
    
    Introduction:
    Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, 
    have been firmly established as state of the art approaches in sequence modeling and transduction problems 
    such as language modeling and machine translation. Numerous efforts have since continued to push the 
    boundaries of recurrent language models and encoder-decoder architectures.
    
    Recurrent models typically factor computation along the symbol positions of the input and output sequences. 
    Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, 
    as a function of the previous hidden state ht−1 and the input for position t. This inherently sequential 
    nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, 
    as memory constraints limit batching across examples.
    
    Attention mechanisms have become an integral part of compelling sequence modeling and transduction models 
    in various tasks, allowing modeling of dependencies without regard to their distance in the input or output 
    sequences. In all but a few cases, however, such attention mechanisms are used in conjunction with a 
    recurrent network.
    
    In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying 
    entirely on an attention mechanism to draw global dependencies between input and output. The Transformer 
    allows for significantly more parallelization and can reach a new state of the art in translation quality 
    after being trained for as little as twelve hours on eight P100 GPUs.
    """ * 3  # Repeat to make it longer
    
    print("Testing PDF Summarizer with Chunking...")
    print("-" * 40)
    
    # Test chunking
    chunker = RecursiveTextChunker(max_chunk_size=500)
    chunks = chunker.chunk_text(test_text)
    print(f"Text split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}: {chunk.token_count} tokens, {len(chunk.content)} chars")
        print(f"Preview: {chunk.content[:100]}...")
    
    print("\n" + "-" * 40)
    print("Testing parallel summarization...")
    
    result = await summarizer.summarize_text(test_text, focus="attention mechanisms")
    
    if result["success"]:
        print(f"✓ Summary generated using {result['model_used']}")
        print(f"✓ Processed {result.get('chunks_processed', 1)} chunks")
        if result.get('chunks_failed', 0) > 0:
            print(f"⚠ {result['chunks_failed']} chunks failed")
        print(f"\nFinal Summary:\n{result['summary']}")
    else:
        print(f"✗ Fallback summary used")
        print(f"\nSummary:\n{result['summary']}")


if __name__ == "__main__":
    asyncio.run(test_summarizer())