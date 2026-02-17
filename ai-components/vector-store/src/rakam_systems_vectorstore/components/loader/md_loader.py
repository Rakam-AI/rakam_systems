"""
Markdown Loader for processing Markdown (.md) files.

This loader provides intelligent markdown processing with:
- Header-based section splitting
- Code block preservation
- Metadata extraction (frontmatter)
- Configurable chunking strategies
- Support for common markdown extensions
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rakam_systems_tools.utils import logging
from rakam_systems_core.interfaces.loader import Loader
from rakam_systems_vectorstore.components.chunker import AdvancedChunker
from rakam_systems_vectorstore.core import Node, NodeMetadata, VSFile

logger = logging.getLogger(__name__)


class MdLoader(Loader):
    """
    Markdown loader for processing .md files.

    This loader provides markdown processing with support for:
    - Header-based section splitting (preserves document structure)
    - Code block preservation (keeps code blocks intact)
    - YAML frontmatter extraction
    - Advanced text chunking
    - Configurable processing options

    The extracted content is chunked and returned as text or Node objects.
    """

    # Default configuration
    DEFAULT_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_CHUNK_SIZE = 2048
    DEFAULT_CHUNK_OVERLAP = 128

    # Regex patterns
    FRONTMATTER_PATTERN = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)

    def __init__(
        self,
        name: str = "md_loader",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Markdown loader.

        Args:
            name: Component name
            config: Optional configuration with keys:
                - embed_model_id: HuggingFace model ID for tokenization (default: "sentence-transformers/all-MiniLM-L6-v2")
                - chunk_size: Maximum tokens per chunk (default: 2048)
                - chunk_overlap: Overlap between chunks in tokens (default: 128)
                - min_sentences_per_chunk: Minimum sentences per chunk (default: 1)
                - tokenizer: Tokenizer for chunking (default: "character")
                - split_by_headers: Whether to split by headers (default: True)
                - preserve_code_blocks: Whether to keep code blocks intact (default: True)
                - extract_frontmatter: Whether to extract YAML frontmatter (default: True)
                - include_frontmatter_in_chunks: Whether to include frontmatter in chunks (default: False)
                - encoding: File encoding (default: "utf-8")
        """
        super().__init__(name=name, config=config)

        # Extract configuration
        config = config or {}
        self._encoding = config.get('encoding', 'utf-8')
        self._split_by_headers = config.get('split_by_headers', True)
        self._preserve_code_blocks = config.get('preserve_code_blocks', True)
        self._extract_frontmatter = config.get('extract_frontmatter', True)
        self._include_frontmatter_in_chunks = config.get(
            'include_frontmatter_in_chunks', False)

        # Chunking configuration
        self._chunk_size = config.get('chunk_size', self.DEFAULT_CHUNK_SIZE)
        self._chunk_overlap = config.get(
            'chunk_overlap', self.DEFAULT_CHUNK_OVERLAP)
        self._min_sentences_per_chunk = config.get(
            'min_sentences_per_chunk', 1)
        self._tokenizer = config.get('tokenizer', 'character')

        # Initialize advanced chunker
        embed_model_id = config.get(
            'embed_model_id', self.DEFAULT_EMBED_MODEL_ID)
        self._chunker = AdvancedChunker(
            embed_model_id=embed_model_id,
            strategy="default"
        )

        # Store last extraction info
        self._last_frontmatter = None
        self._last_headers = []

        logger.info(
            f"Initialized MdLoader with chunk_size={self._chunk_size}, chunk_overlap={self._chunk_overlap}")

    def run(self, source: str) -> List[str]:
        """
        Execute the primary operation for the component.

        This method satisfies the BaseComponent abstract method requirement
        and delegates to load_as_chunks.

        Args:
            source: Path to Markdown file

        Returns:
            List of text chunks extracted from the Markdown file
        """
        return self.load_as_chunks(source)

    def load_as_text(
        self,
        source: Union[str, Path],
    ) -> str:
        """
        Load Markdown file and return as a single text string.

        This method extracts all text from the Markdown file and returns it
        as a single string without chunking.

        Args:
            source: Path to Markdown file

        Returns:
            Full text content of the Markdown file as a single string

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source is not a Markdown file
            Exception: If file processing fails
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Validate file exists
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Validate file is a Markdown file
        if not self._is_md_file(source):
            raise ValueError(f"File is not a Markdown file: {source}")

        logger.info(f"Loading Markdown as text: {source}")
        start_time = time.time()

        try:
            # Read file content
            with open(source, 'r', encoding=self._encoding) as f:
                content = f.read()

            # Extract and optionally remove frontmatter
            content, frontmatter = self._process_frontmatter(content)
            self._last_frontmatter = frontmatter

            # Extract headers for metadata
            self._last_headers = self._extract_headers(content)

            elapsed = time.time() - start_time
            logger.info(
                f"Markdown loaded as text in {elapsed:.2f}s: {len(content)} characters")

            return content

        except Exception as e:
            logger.error(f"Error loading Markdown as text {source}: {e}")
            raise

    def load_as_chunks(
        self,
        source: Union[str, Path],
    ) -> List[str]:
        """
        Load Markdown file and return as a list of text chunks.

        This method extracts text from the Markdown file, processes it with
        the configured chunker, and returns a list of text chunks.

        Args:
            source: Path to Markdown file

        Returns:
            List of text chunks extracted from the Markdown file

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source is not a Markdown file
            Exception: If file processing fails
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Validate file exists
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Validate file is a Markdown file
        if not self._is_md_file(source):
            raise ValueError(f"File is not a Markdown file: {source}")

        logger.info(f"Loading Markdown file: {source}")
        start_time = time.time()

        try:
            # Read file content
            with open(source, 'r', encoding=self._encoding) as f:
                content = f.read()

            # Extract and optionally remove frontmatter
            content, frontmatter = self._process_frontmatter(content)
            self._last_frontmatter = frontmatter

            # Extract headers for metadata
            self._last_headers = self._extract_headers(content)

            # Chunk the content
            if self._split_by_headers:
                text_chunks = self._chunk_by_headers(content)
            else:
                text_chunks = self._chunk_text(content)

            # Optionally prepend frontmatter to first chunk
            if self._include_frontmatter_in_chunks and frontmatter and text_chunks:
                frontmatter_text = self._frontmatter_to_text(frontmatter)
                text_chunks[0] = frontmatter_text + "\n\n" + text_chunks[0]

            elapsed = time.time() - start_time
            logger.info(
                f"Markdown processed in {elapsed:.2f}s: {len(text_chunks)} chunks")

            return text_chunks

        except Exception as e:
            logger.error(f"Error processing Markdown {source}: {e}")
            raise

    def load_as_nodes(
        self,
        source: Union[str, Path],
        source_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Node]:
        """
        Load Markdown file and return as Node objects with metadata.

        Args:
            source: Path to Markdown file
            source_id: Optional source identifier (defaults to file path)
            custom_metadata: Optional custom metadata to attach to nodes

        Returns:
            List of Node objects with text chunks and metadata
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Load text chunks
        chunks = self.load_as_chunks(source)

        # Determine source ID
        if source_id is None:
            source_id = source

        # Build custom metadata with frontmatter if available
        node_custom_metadata = custom_metadata.copy() if custom_metadata else {}
        if self._last_frontmatter:
            node_custom_metadata['frontmatter'] = self._last_frontmatter
        if self._last_headers:
            node_custom_metadata['headers'] = self._last_headers

        # Create nodes with metadata
        nodes = []
        for idx, chunk in enumerate(chunks):
            metadata = NodeMetadata(
                source_file_uuid=source_id,
                position=idx,
                custom=node_custom_metadata
            )
            node = Node(content=chunk, metadata=metadata)
            nodes.append(node)

        logger.info(f"Created {len(nodes)} nodes from Markdown: {source}")
        return nodes

    def load_as_vsfile(
        self,
        file_path: Union[str, Path],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> VSFile:
        """
        Load Markdown file and return as VSFile object.

        Args:
            file_path: Path to Markdown file
            custom_metadata: Optional custom metadata

        Returns:
            VSFile object with nodes

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a Markdown file
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self._is_md_file(file_path):
            raise ValueError(f"File is not a Markdown file: {file_path}")

        # Create VSFile
        vsfile = VSFile(file_path)

        # Load and create nodes
        nodes = self.load_as_nodes(
            file_path, str(vsfile.uuid), custom_metadata)
        vsfile.nodes = nodes
        vsfile.processed = True

        logger.info(
            f"Created VSFile with {len(nodes)} nodes from: {file_path}")
        return vsfile

    def get_frontmatter(self) -> Optional[Dict[str, Any]]:
        """
        Get the frontmatter from the last processed file.

        Returns:
            Dictionary of frontmatter key-value pairs, or None if no frontmatter
        """
        return self._last_frontmatter

    def get_headers(self) -> List[Dict[str, Any]]:
        """
        Get the headers from the last processed file.

        Returns:
            List of header dictionaries with 'level' and 'text' keys
        """
        return self._last_headers

    def _is_md_file(self, file_path: str) -> bool:
        """
        Check if file is a Markdown file based on extension.

        Args:
            file_path: Path to file

        Returns:
            True if file is a Markdown file, False otherwise
        """
        path = Path(file_path)
        return path.suffix.lower() in ['.md', '.markdown', '.mdown', '.mkd', '.mkdn']

    def _process_frontmatter(self, content: str) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Extract and optionally remove YAML frontmatter from content.

        Args:
            content: Raw markdown content

        Returns:
            Tuple of (content without frontmatter, frontmatter dict or None)
        """
        if not self._extract_frontmatter:
            return content, None

        match = self.FRONTMATTER_PATTERN.match(content)
        if not match:
            return content, None

        try:
            import yaml
            frontmatter_text = match.group(1)
            frontmatter = yaml.safe_load(frontmatter_text)

            # Remove frontmatter from content
            content_without_frontmatter = content[match.end():]

            logger.debug(
                f"Extracted frontmatter with {len(frontmatter) if frontmatter else 0} keys")
            return content_without_frontmatter, frontmatter

        except ImportError:
            logger.warning(
                "PyYAML not installed. Frontmatter extraction disabled.")
            return content, None
        except Exception as e:
            logger.warning(f"Failed to parse frontmatter: {e}")
            return content, None

    def _frontmatter_to_text(self, frontmatter: Dict[str, Any]) -> str:
        """
        Convert frontmatter dictionary to readable text.

        Args:
            frontmatter: Frontmatter dictionary

        Returns:
            Human-readable text representation
        """
        if not frontmatter:
            return ""

        lines = ["--- Document Metadata ---"]
        for key, value in frontmatter.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            lines.append(f"{key}: {value}")
        lines.append("---")

        return "\n".join(lines)

    def _extract_headers(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract all headers from markdown content.

        Args:
            content: Markdown content

        Returns:
            List of header dictionaries with 'level' and 'text' keys
        """
        headers = []
        for match in self.HEADER_PATTERN.finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            headers.append({
                'level': level,
                'text': text
            })

        return headers

    def _chunk_by_headers(self, content: str) -> List[str]:
        """
        Split content by headers while preserving code blocks.

        Args:
            content: Markdown content

        Returns:
            List of text chunks split by headers
        """
        if not content or not content.strip():
            return []

        # Preserve code blocks by replacing them with placeholders
        code_blocks = []
        if self._preserve_code_blocks:
            def replace_code_block(match):
                code_blocks.append(match.group(0))
                return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

            content = self.CODE_BLOCK_PATTERN.sub(replace_code_block, content)

        # Split by headers
        chunks = []
        current_chunk = []
        current_header = None

        for line in content.split('\n'):
            header_match = self.HEADER_PATTERN.match(line)

            if header_match:
                # Save previous chunk if it has content
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)

                # Start new chunk with this header
                current_chunk = [line]
                current_header = header_match.group(2)
            else:
                current_chunk.append(line)

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        # Restore code blocks
        if self._preserve_code_blocks and code_blocks:
            restored_chunks = []
            for chunk in chunks:
                for i, code_block in enumerate(code_blocks):
                    chunk = chunk.replace(f"__CODE_BLOCK_{i}__", code_block)
                restored_chunks.append(chunk)
            chunks = restored_chunks

        # If chunks are too large, further split them
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self._chunk_size * 4:  # Rough character estimate
                sub_chunks = self._chunk_text(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        # If no chunks created, use standard chunking
        if not final_chunks:
            return self._chunk_text(content)

        return final_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using AdvancedChunker's chunk_text method.

        Args:
            text: Full text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        try:
            # Use AdvancedChunker's chunk_text method for plain text
            chunk_dicts = self._chunker.chunk_text(
                text=text,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                min_sentences_per_chunk=self._min_sentences_per_chunk,
                tokenizer=self._tokenizer
            )

            # Extract just the text from the chunk dictionaries
            text_chunks = [chunk_dict['text'] for chunk_dict in chunk_dicts]

            logger.debug(f"Chunked text into {len(text_chunks)} chunks")
            return text_chunks

        except Exception as e:
            logger.warning(f"Failed to chunk text with AdvancedChunker: {e}")
            # Fall back to returning the whole text as a single chunk
            logger.info("Falling back to single chunk")
            return [text]


def create_md_loader(
    chunk_size: int = 2048,
    chunk_overlap: int = 128,
    min_sentences_per_chunk: int = 1,
    tokenizer: str = "character",
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    split_by_headers: bool = True,
    preserve_code_blocks: bool = True,
    extract_frontmatter: bool = True,
    include_frontmatter_in_chunks: bool = False,
    encoding: str = "utf-8"
) -> MdLoader:
    """
    Factory function to create a Markdown loader.

    Args:
        chunk_size: Maximum tokens per chunk (default: 2048)
        chunk_overlap: Overlap between chunks in tokens (default: 128)
        min_sentences_per_chunk: Minimum sentences per chunk (default: 1)
        tokenizer: Tokenizer for chunking - "character", "gpt2", or HuggingFace model (default: "character")
        embed_model_id: HuggingFace model ID for tokenization (default: "sentence-transformers/all-MiniLM-L6-v2")
        split_by_headers: Whether to split content by headers (default: True)
        preserve_code_blocks: Whether to keep code blocks intact (default: True)
        extract_frontmatter: Whether to extract YAML frontmatter (default: True)
        include_frontmatter_in_chunks: Whether to include frontmatter in chunks (default: False)
        encoding: File encoding (default: "utf-8")

    Returns:
        Configured Markdown loader

    Example:
        >>> loader = create_md_loader(chunk_size=1024, chunk_overlap=64)
        >>> chunks = loader.run("docs/README.md")
        >>> print(f"Extracted {len(chunks)} chunks")

        >>> # Create loader without header splitting
        >>> loader = create_md_loader(split_by_headers=False)
        >>> chunks = loader.run("docs/README.md")

        >>> # Access frontmatter after loading
        >>> loader = create_md_loader()
        >>> chunks = loader.run("docs/article.md")
        >>> frontmatter = loader.get_frontmatter()
        >>> if frontmatter:
        ...     print(f"Title: {frontmatter.get('title')}")
    """
    config = {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'min_sentences_per_chunk': min_sentences_per_chunk,
        'tokenizer': tokenizer,
        'embed_model_id': embed_model_id,
        'split_by_headers': split_by_headers,
        'preserve_code_blocks': preserve_code_blocks,
        'extract_frontmatter': extract_frontmatter,
        'include_frontmatter_in_chunks': include_frontmatter_in_chunks,
        'encoding': encoding
    }

    return MdLoader(config=config)


__all__ = ["MdLoader", "create_md_loader"]
