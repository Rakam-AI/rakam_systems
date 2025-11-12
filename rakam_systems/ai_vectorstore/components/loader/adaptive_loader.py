"""
Adaptive data loader that automatically detects and processes different input types.

Supports:
- Plain text files (.txt)
- PDF documents (.pdf)
- Word documents (.docx, .doc)
- Markdown files (.md)
- JSON data (.json)
- CSV/TSV data (.csv, .tsv)
- HTML files (.html)
- Code files (.py, .js, .java, etc.)
- Raw text strings
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ai_core.interfaces.loader import Loader
from ai_vectorstore.components.chunker import TextChunker
from ai_vectorstore.core import Node, NodeMetadata, VSFile

logger = logging.getLogger(__name__)


class AdaptiveLoader(Loader):
    """
    Adaptive data loader that automatically detects input type and applies
    appropriate preprocessing strategy.
    
    This loader handles various input formats:
    - File paths (detects type by extension)
    - Raw text strings
    - Structured data (JSON, CSV)
    - Binary documents (PDF, DOCX)
    """
    
    # Supported file extensions and their types
    FILE_TYPE_MAP = {
        # Text files
        '.txt': 'text',
        '.text': 'text',
        
        # Markdown
        '.md': 'markdown',
        '.markdown': 'markdown',
        
        # Documents
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'doc',
        
        # Structured data
        '.json': 'json',
        '.csv': 'csv',
        '.tsv': 'tsv',
        
        # HTML
        '.html': 'html',
        '.htm': 'html',
        
        # Code files
        '.py': 'code',
        '.js': 'code',
        '.java': 'code',
        '.cpp': 'code',
        '.c': 'code',
        '.go': 'code',
        '.rs': 'code',
        '.ts': 'code',
        '.tsx': 'code',
        '.jsx': 'code',
    }
    
    def __init__(
        self,
        name: str = "adaptive_loader",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize adaptive loader.
        
        Args:
            name: Component name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        self._encoding = config.get('encoding', 'utf-8') if config else 'utf-8'
        chunk_size = config.get('chunk_size', 512) if config else 512
        chunk_overlap = config.get('chunk_overlap', 50) if config else 50
        self._chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def run(self, source: str) -> List[str]:
        """
        Load data from source.
        
        Args:
            source: File path or raw text
            
        Returns:
            List of text chunks
        """
        # Check if source is a file path
        if os.path.isfile(source):
            return self._load_from_file(source)
        else:
            # Treat as raw text
            return self._process_text(source, "raw_text")
    
    def load_as_nodes(
        self,
        source: Union[str, Path],
        source_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Node]:
        """
        Load data and return as Node objects with metadata.
        
        Args:
            source: File path or raw text
            source_id: Optional source identifier (defaults to file path)
            custom_metadata: Optional custom metadata to attach to nodes
            
        Returns:
            List of Node objects
        """
        # Load text chunks
        if isinstance(source, Path):
            source = str(source)
        
        chunks = self.run(source)
        
        # Determine source ID
        if source_id is None:
            if os.path.isfile(source):
                source_id = source
            else:
                source_id = "text_input"
        
        # Create nodes with metadata
        nodes = []
        for idx, chunk in enumerate(chunks):
            metadata = NodeMetadata(
                source_file_uuid=source_id,
                position=idx,
                custom=custom_metadata or {}
            )
            node = Node(content=chunk, metadata=metadata)
            nodes.append(node)
        
        return nodes
    
    def load_as_vsfile(
        self,
        file_path: Union[str, Path],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> VSFile:
        """
        Load a file and return as VSFile object.
        
        Args:
            file_path: Path to file
            custom_metadata: Optional custom metadata
            
        Returns:
            VSFile object with nodes
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Create VSFile
        vsfile = VSFile(file_path)
        
        # Load and create nodes
        nodes = self.load_as_nodes(file_path, str(vsfile.uuid), custom_metadata)
        vsfile.nodes = nodes
        vsfile.processed = True
        
        return vsfile
    
    def _load_from_file(self, file_path: str) -> List[str]:
        """Load content from a file based on its type."""
        path = Path(file_path)
        file_type = self._detect_file_type(path)
        
        logger.info(f"Loading file: {file_path} (type: {file_type})")
        
        if file_type == 'text':
            return self._load_text_file(path)
        elif file_type == 'markdown':
            return self._load_markdown_file(path)
        elif file_type == 'pdf':
            return self._load_pdf_file(path)
        elif file_type == 'docx':
            return self._load_docx_file(path)
        elif file_type == 'json':
            return self._load_json_file(path)
        elif file_type == 'csv':
            return self._load_csv_file(path)
        elif file_type == 'html':
            return self._load_html_file(path)
        elif file_type == 'code':
            return self._load_code_file(path)
        else:
            # Default: try to load as text
            logger.warning(f"Unknown file type, attempting to load as text: {file_path}")
            return self._load_text_file(path)
    
    def _detect_file_type(self, path: Path) -> str:
        """Detect file type based on extension."""
        suffix = path.suffix.lower()
        return self.FILE_TYPE_MAP.get(suffix, 'unknown')
    
    def _load_text_file(self, path: Path) -> List[str]:
        """Load plain text file."""
        try:
            with open(path, 'r', encoding=self._encoding) as f:
                content = f.read()
            return self._process_text(content, "text")
        except Exception as e:
            logger.error(f"Error loading text file {path}: {e}")
            raise
    
    def _load_markdown_file(self, path: Path) -> List[str]:
        """Load markdown file."""
        try:
            with open(path, 'r', encoding=self._encoding) as f:
                content = f.read()
            return self._process_markdown(content)
        except Exception as e:
            logger.error(f"Error loading markdown file {path}: {e}")
            raise
    
    def _load_pdf_file(self, path: Path) -> List[str]:
        """Load PDF file."""
        try:
            import pymupdf  # PyMuPDF
        except ImportError:
            logger.error("pymupdf is required for PDF support. Install with: pip install pymupdf")
            raise ImportError("pymupdf is required for PDF support")
        
        try:
            doc = pymupdf.open(path)
            text_chunks = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    # Process each page separately to maintain structure
                    chunks = self._chunker.chunk_text(text, f"page_{page_num + 1}")
                    text_chunks.extend(chunks)
            
            doc.close()
            return text_chunks
        except Exception as e:
            logger.error(f"Error loading PDF file {path}: {e}")
            raise
    
    def _load_docx_file(self, path: Path) -> List[str]:
        """Load DOCX file."""
        try:
            from docx import Document
        except ImportError:
            logger.error("python-docx is required for DOCX support. Install with: pip install python-docx")
            raise ImportError("python-docx is required for DOCX support")
        
        try:
            doc = Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = '\n\n'.join(paragraphs)
            return self._process_text(full_text, "document")
        except Exception as e:
            logger.error(f"Error loading DOCX file {path}: {e}")
            raise
    
    def _load_json_file(self, path: Path) -> List[str]:
        """Load JSON file."""
        try:
            with open(path, 'r', encoding=self._encoding) as f:
                data = json.load(f)
            return self._process_json(data)
        except Exception as e:
            logger.error(f"Error loading JSON file {path}: {e}")
            raise
    
    def _load_csv_file(self, path: Path) -> List[str]:
        """Load CSV file."""
        try:
            import csv
        except ImportError:
            logger.error("csv module is required")
            raise
        
        try:
            with open(path, 'r', encoding=self._encoding) as f:
                # Detect delimiter
                sample = f.read(1024)
                f.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(f, delimiter=delimiter)
                rows = list(reader)
            
            return self._process_csv(rows)
        except Exception as e:
            logger.error(f"Error loading CSV file {path}: {e}")
            raise
    
    def _load_html_file(self, path: Path) -> List[str]:
        """Load HTML file."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("beautifulsoup4 is required for HTML support. Install with: pip install beautifulsoup4")
            raise ImportError("beautifulsoup4 is required for HTML support")
        
        try:
            with open(path, 'r', encoding=self._encoding) as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return self._process_text(text, "html")
        except Exception as e:
            logger.error(f"Error loading HTML file {path}: {e}")
            raise
    
    def _load_code_file(self, path: Path) -> List[str]:
        """Load code file."""
        try:
            with open(path, 'r', encoding=self._encoding) as f:
                content = f.read()
            return self._process_code(content, path.suffix)
        except Exception as e:
            logger.error(f"Error loading code file {path}: {e}")
            raise
    
    def _process_text(self, text: str, source_type: str) -> List[str]:
        """Process plain text into chunks."""
        if not text or not text.strip():
            return []
        
        return self._chunker.chunk_text(text, source_type)
    
    def _process_markdown(self, text: str) -> List[str]:
        """Process markdown text."""
        # Split by headers to maintain structure
        chunks = []
        current_chunk = []
        
        for line in text.split('\n'):
            if line.startswith('#'):
                # New section, save previous chunk
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    if chunk_text.strip():
                        chunks.append(chunk_text)
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        # If no chunks created, chunk normally
        if not chunks:
            return self._chunker.chunk_text(text, "markdown")
        
        return chunks
    
    def _process_json(self, data: Any) -> List[str]:
        """Process JSON data."""
        if isinstance(data, dict):
            # Convert dict to readable text
            text = json.dumps(data, indent=2)
            return [text]
        elif isinstance(data, list):
            # Process each item separately
            return [json.dumps(item, indent=2) for item in data if item]
        else:
            return [str(data)]
    
    def _process_csv(self, rows: List[Dict[str, Any]]) -> List[str]:
        """Process CSV rows."""
        chunks = []
        for row in rows:
            # Convert each row to readable text
            text = ', '.join(f"{k}: {v}" for k, v in row.items() if v)
            if text.strip():
                chunks.append(text)
        return chunks
    
    def _process_code(self, code: str, extension: str) -> List[str]:
        """Process code files."""
        # For code, we want to preserve structure
        # Split by functions/classes or use simple chunking
        return self._chunker.chunk_text(code, f"code{extension}")


def create_adaptive_loader(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    encoding: str = 'utf-8'
) -> AdaptiveLoader:
    """
    Factory function to create an adaptive loader.
    
    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        encoding: Text encoding
        
    Returns:
        Configured adaptive loader
    """
    config = {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'encoding': encoding
    }
    
    return AdaptiveLoader(config=config)


__all__ = ["AdaptiveLoader", "create_adaptive_loader"]

