"""
Adaptive data loader that automatically detects and processes different input types.

Supports:
- Plain text files (.txt)
- PDF documents (.pdf)
- Word documents (.docx, .doc)
- ODT documents (.odt)
- Email files (.eml)
- Markdown files (.md)
- JSON data (.json)
- CSV/TSV/XLSX data (.csv, .tsv, .xlsx, .xls)
- HTML files (.html)
- Code files (.py, .js, .java, etc.)
- Raw text strings

This loader delegates to specialized loaders in the same folder for each file type.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rakam_systems_core.ai_utils import logging
from rakam_systems_core.ai_core.interfaces.loader import Loader
from rakam_systems_vectorstore.components.chunker import TextChunker
from rakam_systems_vectorstore.core import Node, NodeMetadata, VSFile

logger = logging.getLogger(__name__)


class AdaptiveLoader(Loader):
    """
    Adaptive data loader that automatically detects input type and applies
    appropriate preprocessing strategy by delegating to specialized loaders.

    This loader handles various input formats:
    - File paths (detects type by extension)
    - Raw text strings
    - Structured data (JSON, CSV, XLSX)
    - Binary documents (PDF, DOCX, ODT)
    - Email files (EML)
    - Code files
    - HTML files
    - Markdown files

    All 4 loader interface methods (load_as_text, load_as_chunks, load_as_nodes,
    load_as_vsfile) are supported and delegate to the appropriate specialized loader.
    """

    # Supported file extensions and their loader types
    FILE_TYPE_MAP = {
        # Text files
        '.txt': 'text',
        '.text': 'text',

        # Markdown
        '.md': 'md',
        '.markdown': 'md',

        # Documents
        '.pdf': 'pdf',
        '.docx': 'doc',
        '.doc': 'doc',
        '.odt': 'odt',

        # Email files
        '.eml': 'eml',
        '.msg': 'eml',

        # Structured/Tabular data
        '.json': 'json',
        '.csv': 'tabular',
        '.tsv': 'tabular',
        '.xlsx': 'tabular',
        '.xls': 'tabular',

        # HTML
        '.html': 'html',
        '.htm': 'html',
        '.xhtml': 'html',

        # Code files - comprehensive list from CodeLoader
        '.py': 'code',
        '.pyw': 'code',
        '.pyi': 'code',
        '.js': 'code',
        '.jsx': 'code',
        '.ts': 'code',
        '.tsx': 'code',
        '.mjs': 'code',
        '.cjs': 'code',
        '.java': 'code',
        '.kt': 'code',
        '.kts': 'code',
        '.c': 'code',
        '.h': 'code',
        '.cpp': 'code',
        '.cc': 'code',
        '.cxx': 'code',
        '.hpp': 'code',
        '.hxx': 'code',
        '.cs': 'code',
        '.go': 'code',
        '.rs': 'code',
        '.rb': 'code',
        '.rake': 'code',
        '.php': 'code',
        '.swift': 'code',
        '.scala': 'code',
        '.sh': 'code',
        '.bash': 'code',
        '.zsh': 'code',
        '.sql': 'code',
        '.yaml': 'code',
        '.yml': 'code',
        '.r': 'code',
        '.R': 'code',
        '.lua': 'code',
        '.pl': 'code',
        '.pm': 'code',
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
            config: Optional configuration with keys:
                - encoding: Text encoding (default: "utf-8")
                - chunk_size: Maximum tokens per chunk (default: 512)
                - chunk_overlap: Overlap between chunks in tokens (default: 50)

                Additional config options are passed through to specialized loaders.
        """
        super().__init__(name=name, config=config)
        self._config = config or {}
        self._encoding = self._config.get('encoding', 'utf-8')
        chunk_size = self._config.get('chunk_size', 512)
        chunk_overlap = self._config.get('chunk_overlap', 50)
        self._chunker = TextChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Cache for lazy-loaded specialized loaders
        self._loaders: Dict[str, Loader] = {}

    def _detect_file_type(self, path: Union[str, Path]) -> str:
        """Detect file type based on extension."""
        if isinstance(path, str):
            path = Path(path)
        suffix = path.suffix.lower()
        file_type = self.FILE_TYPE_MAP.get(suffix, 'text')

        # Log detection for debugging
        logger.debug(
            f"File type detection: {path.name} -> suffix='{suffix}' -> type='{file_type}'")

        # Safety check: ensure PDFs are never routed to doc loader
        if suffix == '.pdf' and file_type != 'pdf':
            logger.error(
                f"PDF file '{path.name}' was incorrectly mapped to type '{file_type}'. Forcing to 'pdf'.")
            file_type = 'pdf'

        return file_type

    def _get_loader(self, loader_type: str) -> Loader:
        """Get or create a cached loader instance."""
        if loader_type not in self._loaders:
            loader_config = {
                'encoding': self._encoding,
                'chunk_size': self._chunker._chunk_size,
                'chunk_overlap': self._chunker._chunk_overlap,
                **{k: v for k, v in self._config.items() if k not in ('encoding', 'chunk_size', 'chunk_overlap')}
            }

            if loader_type == 'md':
                from .md_loader import MdLoader
                self._loaders[loader_type] = MdLoader(config=loader_config)
            elif loader_type == 'pdf':
                from .pdf_loader_light import PdfLoaderLight
                self._loaders[loader_type] = PdfLoaderLight(
                    config=loader_config)
            elif loader_type == 'doc':
                from .doc_loader import DocLoader
                self._loaders[loader_type] = DocLoader(config=loader_config)
            elif loader_type == 'odt':
                from .odt_loader import OdtLoader
                self._loaders[loader_type] = OdtLoader(config=loader_config)
            elif loader_type == 'eml':
                from .eml_loader import EmlLoader
                self._loaders[loader_type] = EmlLoader(config=loader_config)
            elif loader_type == 'html':
                from .html_loader import HtmlLoader
                self._loaders[loader_type] = HtmlLoader(config=loader_config)
            elif loader_type == 'code':
                from .code_loader import CodeLoader
                self._loaders[loader_type] = CodeLoader(config=loader_config)
            elif loader_type == 'tabular':
                from .tabular_loader import TabularLoader
                self._loaders[loader_type] = TabularLoader(
                    config=loader_config)
            else:
                raise ValueError(f"Unknown loader type: {loader_type}")

        return self._loaders[loader_type]

    def _get_loader_for_file(self, source: Union[str, Path]) -> Optional[Loader]:
        """Get the appropriate loader for a file based on its type."""
        file_type = self._detect_file_type(source)

        # These types don't have specialized loaders
        if file_type in ('text', 'json'):
            return None

        return self._get_loader(file_type)

    # =========================================================================
    # Loader Interface Implementation
    # =========================================================================

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
            return self.load_as_chunks(source)
        else:
            # Treat as raw text
            return self._process_text(source)

    def load_as_text(self, source: Union[str, Path]) -> str:
        """
        Load document and return as a single text string.

        Detects the file type and delegates to the appropriate specialized loader.

        Args:
            source: Path to document file or raw text

        Returns:
            Full text content as a single string
        """
        if isinstance(source, Path):
            source = str(source)

        # Check if source is a file path
        if not os.path.isfile(source):
            # Treat as raw text
            return source

        file_type = self._detect_file_type(source)
        logger.info(f"Loading file as text: {source} (type: {file_type})")

        # Handle text files directly
        if file_type == 'text':
            return self._load_text_file_as_text(source)

        # Handle JSON files directly
        if file_type == 'json':
            return self._load_json_file_as_text(source)

        # Delegate to specialized loader
        loader = self._get_loader(file_type)
        return loader.load_as_text(source)

    def load_as_chunks(self, source: Union[str, Path]) -> List[str]:
        """
        Load document and return as a list of text chunks.

        Detects the file type and delegates to the appropriate specialized loader.

        Args:
            source: Path to document file or raw text

        Returns:
            List of text chunks
        """
        if isinstance(source, Path):
            source = str(source)

        # Check if source is a file path
        if not os.path.isfile(source):
            # Treat as raw text
            return self._process_text(source)

        file_type = self._detect_file_type(source)
        logger.info(f"Loading file as chunks: {source} (type: {file_type})")

        # Handle text files directly
        if file_type == 'text':
            return self._load_text_file_as_chunks(source)

        # Handle JSON files directly
        if file_type == 'json':
            return self._load_json_file_as_chunks(source)

        # Delegate to specialized loader
        loader = self._get_loader(file_type)
        return loader.load_as_chunks(source)

    def load_as_nodes(
        self,
        source: Union[str, Path],
        source_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Node]:
        """
        Load document and return as Node objects with metadata.

        Detects the file type and delegates to the appropriate specialized loader.

        Args:
            source: Path to document file or raw text
            source_id: Optional source identifier (defaults to file path)
            custom_metadata: Optional custom metadata to attach to nodes

        Returns:
            List of Node objects with text chunks and metadata
        """
        if isinstance(source, Path):
            source = str(source)

        # Check if source is a file path
        if not os.path.isfile(source):
            # Treat as raw text - create nodes manually
            chunks = self._process_text(source)
            return self._chunks_to_nodes(chunks, source_id or "text_input", custom_metadata)

        file_type = self._detect_file_type(source)
        source_path = Path(source)
        logger.info(
            f"Loading file as nodes: {source_path.name} (detected type: {file_type}, extension: {source_path.suffix})")

        # Handle text files directly
        if file_type == 'text':
            chunks = self._load_text_file_as_chunks(source)
            return self._chunks_to_nodes(chunks, source_id or source, custom_metadata)

        # Handle JSON files directly
        if file_type == 'json':
            chunks = self._load_json_file_as_chunks(source)
            return self._chunks_to_nodes(chunks, source_id or source, custom_metadata)

        # Delegate to specialized loader
        loader = self._get_loader(file_type)
        return loader.load_as_nodes(source, source_id, custom_metadata)

    def load_as_vsfile(
        self,
        file_path: Union[str, Path],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> VSFile:
        """
        Load document and return as VSFile object.

        Detects the file type and delegates to the appropriate specialized loader.

        Args:
            file_path: Path to document file
            custom_metadata: Optional custom metadata

        Returns:
            VSFile object with nodes
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = self._detect_file_type(file_path)
        logger.info(f"Loading file as VSFile: {file_path} (type: {file_type})")

        # Handle text files directly
        if file_type == 'text':
            vsfile = VSFile(file_path)
            chunks = self._load_text_file_as_chunks(file_path)
            vsfile.nodes = self._chunks_to_nodes(
                chunks, str(vsfile.uuid), custom_metadata)
            vsfile.processed = True
            return vsfile

        # Handle JSON files directly
        if file_type == 'json':
            vsfile = VSFile(file_path)
            chunks = self._load_json_file_as_chunks(file_path)
            vsfile.nodes = self._chunks_to_nodes(
                chunks, str(vsfile.uuid), custom_metadata)
            vsfile.processed = True
            return vsfile

        # Delegate to specialized loader
        loader = self._get_loader(file_type)
        return loader.load_as_vsfile(file_path, custom_metadata)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _chunks_to_nodes(
        self,
        chunks: List[str],
        source_id: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Node]:
        """Convert text chunks to Node objects."""
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

    def _process_text(self, text: str) -> List[str]:
        """Process plain text into chunks."""
        if not text or not text.strip():
            return []
        # chunk_text returns list of dicts with 'text' key
        chunk_dicts = self._chunker.chunk_text(text, "text")
        return [chunk['text'] for chunk in chunk_dicts]

    def _load_text_file_as_text(self, file_path: str) -> str:
        """Load plain text file and return as single string."""
        try:
            with open(file_path, 'r', encoding=self._encoding) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            raise

    def _load_text_file_as_chunks(self, file_path: str) -> List[str]:
        """Load plain text file and return as chunks."""
        content = self._load_text_file_as_text(file_path)
        return self._process_text(content)

    def _load_json_file_as_text(self, file_path: str) -> str:
        """Load JSON file and return as formatted string."""
        try:
            with open(file_path, 'r', encoding=self._encoding) as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            raise

    def _load_json_file_as_chunks(self, file_path: str) -> List[str]:
        """Load JSON file and return as chunks."""
        try:
            with open(file_path, 'r', encoding=self._encoding) as f:
                data = json.load(f)

            if isinstance(data, dict):
                # Return entire dict as single chunk
                return [json.dumps(data, indent=2)]
            elif isinstance(data, list):
                # Each item becomes a chunk
                return [json.dumps(item, indent=2) for item in data if item]
            else:
                return [str(data)]
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            raise


def create_adaptive_loader(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    encoding: str = 'utf-8',
    **kwargs
) -> AdaptiveLoader:
    """
    Factory function to create an adaptive loader.

    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        encoding: Text encoding
        **kwargs: Additional configuration options passed to specialized loaders

    Returns:
        Configured adaptive loader
    """
    config = {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'encoding': encoding,
        **kwargs
    }

    return AdaptiveLoader(config=config)


__all__ = ["AdaptiveLoader", "create_adaptive_loader"]
