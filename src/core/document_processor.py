"""
Document Processor for OpenSCENARIO 2.0 RAG System
Handles PDF extraction, code example processing, and chunking with LangExtract.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import re

import fitz  # PyMuPDF
from langextract import LangExtract
import pandas as pd

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

class DocumentChunk:
    """Represents a processed document chunk."""
    
    def __init__(self, content: str, metadata: Dict[str, Any], chunk_id: str = None):
        self.content = content
        self.metadata = metadata
        self.chunk_id = chunk_id or self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for chunk based on content hash."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        return f"chunk_{content_hash[:12]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create chunk from dictionary."""
        return cls(
            content=data['content'],
            metadata=data['metadata'],
            chunk_id=data.get('chunk_id')
        )

class DocumentProcessor:
    """Processes OpenSCENARIO documentation and code examples."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.langextract = LangExtract()
        self.chunk_size = config.get('document_processing.chunk_size', 800)
        self.chunk_overlap = config.get('document_processing.chunk_overlap', 100)
        self.processed_dir = Path(config.get('data.processed_chunks_dir'))
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DocumentProcessor initialized with chunk_size={self.chunk_size}")
    
    def process_pdf_documentation(self, pdf_path: str) -> List[DocumentChunk]:
        """Process ASAM OpenSCENARIO PDF documentation.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of document chunks
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF documentation: {pdf_path}")
        
        # Check cache first
        cache_file = self.processed_dir / f"{pdf_path.stem}_chunks.json"
        if cache_file.exists():
            logger.info(f"Loading cached chunks from {cache_file}")
            return self._load_cached_chunks(cache_file)
        
        chunks = []
        
        try:
            # Extract text using LangExtract for better structure preservation
            extracted_data = self.langextract.extract(str(pdf_path))
            
            if extracted_data and 'content' in extracted_data:
                # Process structured content
                chunks.extend(self._process_structured_content(
                    extracted_data, 
                    source=pdf_path.name
                ))
            else:
                # Fallback to PyMuPDF
                logger.warning("LangExtract failed, falling back to PyMuPDF")
                chunks.extend(self._extract_with_pymupdf(pdf_path))
            
            # Cache processed chunks
            self._cache_chunks(chunks, cache_file)
            
            logger.info(f"Processed {len(chunks)} chunks from PDF")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            # Fallback to PyMuPDF
            return self._extract_with_pymupdf(pdf_path)
    
    def _process_structured_content(self, extracted_data: Dict, source: str) -> List[DocumentChunk]:
        """Process structured content from LangExtract."""
        chunks = []
        content = extracted_data.get('content', '')
        
        # Extract sections, tables, and code blocks
        sections = self._extract_sections(content)
        tables = extracted_data.get('tables', [])
        code_blocks = self._extract_code_blocks(content)
        
        # Process sections
        for section in sections:
            section_chunks = self._chunk_text(
                section['content'],
                metadata={
                    'source': source,
                    'type': 'section',
                    'title': section['title'],
                    'level': section['level']
                }
            )
            chunks.extend(section_chunks)
        
        # Process tables
        for i, table in enumerate(tables):
            table_content = self._format_table(table)
            chunk = DocumentChunk(
                content=table_content,
                metadata={
                    'source': source,
                    'type': 'table',
                    'table_id': i
                }
            )
            chunks.append(chunk)
        
        # Process code blocks
        for i, code_block in enumerate(code_blocks):
            chunk = DocumentChunk(
                content=code_block,
                metadata={
                    'source': source,
                    'type': 'code_block',
                    'code_id': i
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> List[DocumentChunk]:
        """Fallback PDF extraction using PyMuPDF."""
        chunks = []
        
        try:
            doc = fitz.open(str(pdf_path))
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                full_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
            
            doc.close()
            
            # Chunk the full text
            chunks = self._chunk_text(
                full_text,
                metadata={
                    'source': pdf_path.name,
                    'type': 'documentation',
                    'extraction_method': 'pymupdf'
                }
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return []
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections from content based on headings."""
        sections = []
        lines = content.split('\n')
        current_section = {'title': '', 'content': '', 'level': 0}
        
        for line in lines:
            # Detect headings (markdown style)
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if heading_match:
                # Save previous section
                if current_section['content'].strip():
                    sections.append(current_section.copy())
                
                # Start new section
                level = len(heading_match.group(1))
                title = heading_match.group(2)
                current_section = {
                    'title': title,
                    'content': '',
                    'level': level
                }
            else:
                current_section['content'] += line + '\n'
        
        # Add final section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract code blocks from content."""
        # Pattern for fenced code blocks
        code_pattern = r'```(?:\w+)?\n(.*?)\n```'
        code_blocks = re.findall(code_pattern, content, re.DOTALL)
        
        # Also look for OpenSCENARIO specific patterns
        osc_pattern = r'(?:scenario|action|condition|trigger)[\s\S]*?(?=\n\n|\n(?:scenario|action|condition|trigger)|\Z)'
        osc_blocks = re.findall(osc_pattern, content, re.IGNORECASE)
        
        return code_blocks + osc_blocks
    
    def _format_table(self, table: Dict) -> str:
        """Format table data as text."""
        if not table:
            return ""
        
        # Simple table formatting
        formatted = "Table:\n"
        if 'headers' in table and 'rows' in table:
            headers = table['headers']
            rows = table['rows']
            
            # Add headers
            formatted += " | ".join(headers) + "\n"
            formatted += "-" * (len(" | ".join(headers))) + "\n"
            
            # Add rows
            for row in rows:
                formatted += " | ".join(str(cell) for cell in row) + "\n"
        
        return formatted
    
    def process_code_examples(self, examples_dir: str) -> List[DocumentChunk]:
        """Process code examples directory.
        
        Args:
            examples_dir: Directory containing code examples
            
        Returns:
            List of document chunks with code examples
        """
        examples_dir = Path(examples_dir)
        if not examples_dir.exists():
            logger.warning(f"Code examples directory not found: {examples_dir}")
            return []
        
        logger.info(f"Processing code examples from: {examples_dir}")
        
        # Check cache
        cache_file = self.processed_dir / "code_examples_chunks.json"
        if cache_file.exists():
            logger.info(f"Loading cached code examples from {cache_file}")
            return self._load_cached_chunks(cache_file)
        
        chunks = []
        
        # Process different file types
        for file_path in examples_dir.rglob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() in ['.osc', '.txt', '.md', '.json']:
                    example_chunks = self._process_code_file(file_path)
                    chunks.extend(example_chunks)
        
        # Cache processed chunks
        self._cache_chunks(chunks, cache_file)
        
        logger.info(f"Processed {len(chunks)} code example chunks")
        return chunks
    
    def _process_code_file(self, file_path: Path) -> List[DocumentChunk]:
        """Process a single code example file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract tags from filename or content
            tags = self._extract_tags(file_path, content)
            
            # For JSON files, try to parse structured data
            if file_path.suffix.lower() == '.json':
                return self._process_json_example(content, file_path, tags)
            
            # For other files, create single chunk
            chunk = DocumentChunk(
                content=content,
                metadata={
                    'source': file_path.name,
                    'type': 'code_example',
                    'file_type': file_path.suffix.lower(),
                    'tags': tags,
                    'full_path': str(file_path)
                }
            )
            
            return [chunk]
            
        except Exception as e:
            logger.error(f"Error processing code file {file_path}: {e}")
            return []
    
    def _process_json_example(self, content: str, file_path: Path, tags: List[str]) -> List[DocumentChunk]:
        """Process JSON-formatted code example."""
        try:
            data = json.loads(content)
            chunks = []
            
            if isinstance(data, list):
                # Multiple examples in one file
                for i, example in enumerate(data):
                    chunk = self._create_example_chunk(example, file_path, tags, i)
                    if chunk:
                        chunks.append(chunk)
            else:
                # Single example
                chunk = self._create_example_chunk(data, file_path, tags)
                if chunk:
                    chunks.append(chunk)
            
            return chunks
            
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in {file_path}, treating as text")
            return []
    
    def _create_example_chunk(self, example: Dict, file_path: Path, tags: List[str], index: int = 0) -> Optional[DocumentChunk]:
        """Create chunk from structured example data."""
        if not isinstance(example, dict):
            return None
        
        # Build content from example structure
        content_parts = []
        
        if 'instruction' in example:
            content_parts.append(f"Instruction: {example['instruction']}")
        
        if 'input' in example:
            content_parts.append(f"Input: {example['input']}")
        
        if 'output' in example:
            content_parts.append(f"Output:\n{example['output']}")
        
        if 'description' in example:
            content_parts.append(f"Description: {example['description']}")
        
        content = "\n\n".join(content_parts)
        
        # Extract additional tags from example
        example_tags = tags.copy()
        if 'tags' in example:
            if isinstance(example['tags'], list):
                example_tags.extend(example['tags'])
            else:
                example_tags.append(str(example['tags']))
        
        return DocumentChunk(
            content=content,
            metadata={
                'source': file_path.name,
                'type': 'structured_example',
                'tags': example_tags,
                'example_index': index,
                'full_path': str(file_path)
            }
        )
    
    def _extract_tags(self, file_path: Path, content: str) -> List[str]:
        """Extract tags from filename and content."""
        tags = []
        
        # Extract from filename
        filename_tags = re.findall(r'([a-z_]+)', file_path.stem.lower())
        tags.extend(filename_tags)
        
        # Extract from content (look for common OpenSCENARIO patterns)
        osc_patterns = [
            r'cut.?in', r'lane.?change', r'overtaking', r'parallel.?drive',
            r'parking', r'intersection', r'traffic.?light', r'pedestrian',
            r'weather', r'road.?condition'
        ]
        
        for pattern in osc_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                tag = pattern.replace('.?', '_').replace('\\', '')
                tags.append(tag)
        
        # Remove duplicates and filter
        tags = list(set(tag for tag in tags if len(tag) > 2))
        
        return tags
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split text into chunks with overlap."""
        if not text.strip():
            return []
        
        chunks = []
        words = text.split()
        
        if len(words) <= self.chunk_size:
            # Text is small enough for single chunk
            chunk = DocumentChunk(content=text, metadata=metadata)
            return [chunk]
        
        # Split into overlapping chunks
        start = 0
        chunk_num = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_number'] = chunk_num
            chunk_metadata['total_chunks'] = (len(words) + self.chunk_size - 1) // self.chunk_size
            
            chunk = DocumentChunk(content=chunk_text, metadata=chunk_metadata)
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(words) else end
            chunk_num += 1
        
        return chunks
    
    def _cache_chunks(self, chunks: List[DocumentChunk], cache_file: Path):
        """Cache processed chunks to file."""
        try:
            chunk_data = [chunk.to_dict() for chunk in chunks]
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Cached {len(chunks)} chunks to {cache_file}")
        except Exception as e:
            logger.error(f"Error caching chunks: {e}")
    
    def _load_cached_chunks(self, cache_file: Path) -> List[DocumentChunk]:
        """Load cached chunks from file."""
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            chunks = [DocumentChunk.from_dict(data) for data in chunk_data]
            logger.info(f"Loaded {len(chunks)} cached chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading cached chunks: {e}")
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents."""
        cache_files = list(self.processed_dir.glob("*.json"))
        
        stats = {
            'cached_files': len(cache_files),
            'cache_directory': str(self.processed_dir),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        
        # Count total chunks across all cache files
        total_chunks = 0
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    total_chunks += len(data)
            except:
                continue
        
        stats['total_cached_chunks'] = total_chunks
        
        return stats
