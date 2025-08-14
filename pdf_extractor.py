#!/usr/bin/env python3
"""
PDF Extraction for Research Papers
Downloads and extracts full text from ArXiv PDFs
"""

import os
import re
import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any
import hashlib
from pathlib import Path
import PyPDF2
import pdfplumber
from io import BytesIO

from redis_logger import get_redis_logger
from pdf_summarizer import PDFSummarizer

logger = get_redis_logger(__name__)


class PDFExtractor:
    """Extract text content from research PDFs"""
    
    def __init__(self, cache_dir: str = "/tmp/pdf_cache", use_llm_summary: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # ArXiv PDF URL pattern
        self.arxiv_pdf_pattern = r'arxiv\.org/abs/(\d+\.\d+)'
        
        # Initialize summarizer with mistral-medium-3.1
        self.use_llm_summary = use_llm_summary
        if self.use_llm_summary:
            self.summarizer = PDFSummarizer(model="mistralai/mistral-medium-3.1")
        
    async def download_pdf(self, url: str) -> Optional[bytes]:
        """Download PDF from URL"""
        # Convert ArXiv abstract URL to PDF URL
        if 'arxiv.org/abs/' in url:
            pdf_url = url.replace('/abs/', '/pdf/') + '.pdf'
        else:
            pdf_url = url
            
        # Check cache first
        url_hash = hashlib.md5(pdf_url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{url_hash}.pdf"
        
        if cache_file.exists():
            logger.info(f"Using cached PDF: {cache_file}")
            return cache_file.read_bytes()
        
        try:
            logger.info(f"Downloading PDF: {pdf_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # Cache the PDF
                        cache_file.write_bytes(content)
                        logger.info(f"Downloaded {len(content)} bytes, cached as {cache_file}")
                        
                        return content
                    else:
                        logger.error(f"Failed to download PDF: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None
    
    def extract_text_pypdf2(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text using PyPDF2"""
        try:
            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text.append(page.extract_text())
            
            full_text = "\n".join(text)
            return full_text
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return None
    
    def extract_text_pdfplumber(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text using pdfplumber (better for complex layouts)"""
        try:
            pdf_file = BytesIO(pdf_bytes)
            
            text = []
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            
            full_text = "\n".join(text)
            return full_text
            
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return None
    
    async def extract_from_url(self, url: str) -> Dict[str, Any]:
        """Download and extract text from PDF URL"""
        result = {
            "url": url,
            "success": False,
            "text": "",
            "page_count": 0,
            "word_count": 0,
            "extraction_method": None
        }
        
        # Download PDF
        pdf_bytes = await self.download_pdf(url)
        if not pdf_bytes:
            result["error"] = "Failed to download PDF"
            return result
        
        # Try pdfplumber first (usually better)
        text = self.extract_text_pdfplumber(pdf_bytes)
        extraction_method = "pdfplumber"
        
        # Fallback to PyPDF2 if pdfplumber fails
        if not text or len(text.strip()) < 100:
            text = self.extract_text_pypdf2(pdf_bytes)
            extraction_method = "pypdf2"
        
        if text and len(text.strip()) > 100:
            result["success"] = True
            result["text"] = text
            result["word_count"] = len(text.split())
            result["extraction_method"] = extraction_method
            
            # Count pages (approximate)
            result["page_count"] = text.count('\n\n') // 50  # Rough estimate
            
            logger.info(f"Extracted {result['word_count']} words using {extraction_method}")
        else:
            result["error"] = "Failed to extract meaningful text"
            
        return result
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract common paper sections"""
        sections = {
            "abstract": "",
            "introduction": "",
            "methodology": "",
            "results": "",
            "conclusion": "",
            "references": ""
        }
        
        # Common section headers
        section_patterns = {
            "abstract": r"(?i)(abstract|summary)\s*\n",
            "introduction": r"(?i)(1\s*\.?\s*introduction|introduction)\s*\n",
            "methodology": r"(?i)(method|methodology|approach|model)\s*\n",
            "results": r"(?i)(results?|experiments?|evaluation)\s*\n",
            "conclusion": r"(?i)(conclusion|summary|discussion)\s*\n",
            "references": r"(?i)(references|bibliography)\s*\n"
        }
        
        text_lower = text.lower()
        
        for section, pattern in section_patterns.items():
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            if matches:
                start = matches[0].end()
                
                # Find the next section start
                end = len(text)
                for other_section, other_pattern in section_patterns.items():
                    if other_section != section:
                        other_matches = list(re.finditer(other_pattern, text[start:], re.MULTILINE))
                        if other_matches:
                            end = min(end, start + other_matches[0].start())
                
                sections[section] = text[start:end].strip()[:5000]  # Limit section length
        
        return sections
    
    def extract_key_information(self, text: str) -> Dict[str, Any]:
        """Extract key information from paper text"""
        info = {
            "equations": [],
            "figures_mentioned": 0,
            "tables_mentioned": 0,
            "citations_count": 0,
            "code_snippets": [],
            "datasets": [],
            "metrics": []
        }
        
        # Extract equations (LaTeX format)
        equation_pattern = r'\$\$?[^$]+\$\$?'
        info["equations"] = re.findall(equation_pattern, text)[:10]
        
        # Count figure and table mentions
        info["figures_mentioned"] = len(re.findall(r'(?i)figure\s+\d+', text))
        info["tables_mentioned"] = len(re.findall(r'(?i)table\s+\d+', text))
        
        # Count citations (approximation)
        info["citations_count"] = len(re.findall(r'\[\d+\]', text))
        
        # Find dataset mentions
        dataset_patterns = [
            r'(?i)(mnist|cifar|imagenet|coco|glue|squad|wmt)',
            r'(?i)(dataset|corpus|benchmark)'
        ]
        for pattern in dataset_patterns:
            matches = re.findall(pattern, text)
            info["datasets"].extend(matches[:5])
        
        # Find metrics mentioned
        metric_patterns = [
            r'(?i)(accuracy|precision|recall|f1|bleu|rouge|perplexity)',
            r'\d+\.?\d*\s*%'
        ]
        for pattern in metric_patterns:
            matches = re.findall(pattern, text)
            info["metrics"].extend(matches[:10])
        
        # Look for code snippets
        code_pattern = r'```[^`]+```'
        info["code_snippets"] = re.findall(code_pattern, text)[:3]
        
        return info
    
    async def process_arxiv_paper(self, arxiv_url: str) -> Dict[str, Any]:
        """Full processing pipeline for an ArXiv paper"""
        logger.info(f"Processing ArXiv paper: {arxiv_url}")
        
        # Extract PDF content
        extraction = await self.extract_from_url(arxiv_url)
        
        if not extraction["success"]:
            return extraction
        
        # Extract sections
        sections = self.extract_sections(extraction["text"])
        extraction["sections"] = sections
        
        # Extract key information
        key_info = self.extract_key_information(extraction["text"])
        extraction["key_information"] = key_info
        
        # Generate LLM summary if enabled
        if self.use_llm_summary and hasattr(self, 'summarizer'):
            try:
                logger.info("Generating LLM summary using mistral-medium-3.1...")
                
                # Summarize the full text
                summary_result = await self.summarizer.summarize_text(
                    extraction["text"][:10000],  # Limit input size
                    focus="key contributions and methodology"
                )
                
                if summary_result["success"]:
                    extraction["llm_summary"] = summary_result["summary"]
                    extraction["summary_model"] = summary_result["model_used"]
                    logger.info(f"Generated summary using {summary_result['model_used']}")
                else:
                    logger.warning("LLM summary failed, using extractive summary")
                    
                # Also summarize individual sections if they exist
                if sections:
                    section_summaries = await self.summarizer.summarize_sections(sections)
                    extraction["section_summaries"] = section_summaries
                    
            except Exception as e:
                logger.error(f"Error generating LLM summary: {e}")
        
        # Create extractive summary as fallback or additional option
        summary_parts = []
        if sections["abstract"]:
            summary_parts.append(sections["abstract"][:500])
        if sections["introduction"]:
            summary_parts.append(sections["introduction"][:500])
        if sections["conclusion"]:
            summary_parts.append(sections["conclusion"][:500])
        
        extraction["extractive_summary"] = "\n\n".join(summary_parts)
        
        # Use LLM summary if available, otherwise extractive
        extraction["summary"] = extraction.get("llm_summary", extraction["extractive_summary"])
        
        return extraction


async def test_pdf_extraction():
    """Test PDF extraction with a real ArXiv paper"""
    extractor = PDFExtractor()
    
    # Test with "Attention Is All You Need" paper
    test_url = "https://arxiv.org/abs/1706.03762"
    
    print(f"Testing PDF extraction for: {test_url}")
    print("-" * 40)
    
    result = await extractor.process_arxiv_paper(test_url)
    
    if result["success"]:
        print(f"✓ Successfully extracted {result['word_count']} words")
        print(f"✓ Extraction method: {result['extraction_method']}")
        print(f"✓ Figures mentioned: {result['key_information']['figures_mentioned']}")
        print(f"✓ Tables mentioned: {result['key_information']['tables_mentioned']}")
        print(f"✓ Citations: {result['key_information']['citations_count']}")
        
        print("\nSections found:")
        for section, content in result["sections"].items():
            if content:
                print(f"  • {section}: {len(content)} chars")
        
        print("\nAbstract preview:")
        if result["sections"]["abstract"]:
            print(result["sections"]["abstract"][:300] + "...")
    else:
        print(f"✗ Extraction failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(test_pdf_extraction())