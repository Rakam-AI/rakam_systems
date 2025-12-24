"""
HTML Loader for processing HTML files.

This loader handles HTML documents and provides:
- HTML parsing and text extraction
- Script/style tag removal
- Semantic structure preservation
- Meta tag extraction (title, description, etc.)
- Link and image reference extraction
- Table content extraction
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rakam_systems.core.ai_utils import logging
from rakam_systems.core.ai_core.interfaces.loader import Loader
from rakam_systems.vectorestore.components.chunker import TextChunker
from rakam_systems.vectorestore.core import Node, NodeMetadata, VSFile

logger = logging.getLogger(__name__)


class HtmlLoader(Loader):
    """
    HTML loader for processing HTML documents.

    This loader provides HTML file processing with support for:
    - HTML parsing and clean text extraction
    - Script/style tag removal
    - Semantic structure preservation (headings, paragraphs, lists)
    - Meta tag extraction
    - Configurable text chunking

    The extracted content is chunked and returned as text or Node objects.
    """

    # Default configuration
    DEFAULT_CHUNK_SIZE = 3000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_MIN_SENTENCES_PER_CHUNK = 5
    DEFAULT_TOKENIZER = "character"

    # Supported HTML file extensions
    SUPPORTED_EXTENSIONS = {'.html', '.htm', '.xhtml'}

    # Tags to remove entirely (content and all)
    REMOVE_TAGS = ['script', 'style', 'noscript', 'iframe', 'svg', 'canvas']

    # Tags that indicate section boundaries
    SECTION_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                    'section', 'article', 'header', 'footer', 'nav', 'aside']

    def __init__(
        self,
        name: str = "html_loader",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize HTML loader.

        Args:
            name: Component name
            config: Optional configuration with keys:
                - chunk_size: Maximum tokens per chunk (default: 3000)
                - chunk_overlap: Overlap between chunks in tokens (default: 200)
                - min_sentences_per_chunk: Minimum sentences per chunk (default: 5)
                - tokenizer: Tokenizer for chunking (default: "character")
                - extract_metadata: Whether to extract meta tags (default: True)
                - preserve_links: Whether to preserve link references (default: False)
                - preserve_structure: Whether to preserve HTML structure hints (default: True)
                - encoding: File encoding (default: "utf-8")
        """
        super().__init__(name=name, config=config)

        # Extract configuration
        config = config or {}
        self._chunk_size = config.get('chunk_size', self.DEFAULT_CHUNK_SIZE)
        self._chunk_overlap = config.get(
            'chunk_overlap', self.DEFAULT_CHUNK_OVERLAP)
        self._min_sentences_per_chunk = config.get(
            'min_sentences_per_chunk', self.DEFAULT_MIN_SENTENCES_PER_CHUNK)
        self._tokenizer = config.get('tokenizer', self.DEFAULT_TOKENIZER)
        self._extract_metadata = config.get('extract_metadata', True)
        self._preserve_links = config.get('preserve_links', False)
        self._preserve_structure = config.get('preserve_structure', True)
        self._encoding = config.get('encoding', 'utf-8')

        # Initialize text chunker
        self._chunker = TextChunker(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            min_sentences_per_chunk=self._min_sentences_per_chunk,
            tokenizer=self._tokenizer
        )

        logger.info(
            f"Initialized HtmlLoader with chunk_size={self._chunk_size}, chunk_overlap={self._chunk_overlap}")

    def run(self, source: str) -> List[str]:
        """
        Execute the primary operation for the component.

        This method satisfies the BaseComponent abstract method requirement
        and delegates to load_as_chunks.

        Args:
            source: Path to HTML file

        Returns:
            List of text chunks extracted from the HTML file
        """
        return self.load_as_chunks(source)

    def load_as_text(
        self,
        source: Union[str, Path],
    ) -> str:
        """
        Load HTML and return as a single text string.

        This method extracts all text from the HTML file and returns it as a single
        string without chunking. Useful when you need the full content.

        Args:
            source: Path to HTML file

        Returns:
            Full text content of the HTML as a single string

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source is not an HTML file
            Exception: If HTML processing fails
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Validate file exists
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Validate file is an HTML
        if not self._is_html_file(source):
            raise ValueError(
                f"File is not an HTML: {source}. Extension: {Path(source).suffix}")

        logger.info(f"Loading HTML as text: {source}")
        start_time = time.time()

        try:
            # Read and parse HTML
            with open(source, 'r', encoding=self._encoding, errors='replace') as f:
                html_content = f.read()

            # Extract text from HTML
            full_text = self._extract_text_from_html(html_content)

            elapsed = time.time() - start_time
            logger.info(
                f"HTML loaded as text in {elapsed:.2f}s: {len(full_text)} characters")

            return full_text

        except Exception as e:
            logger.error(f"Error loading HTML as text {source}: {e}")
            raise

    def load_as_chunks(
        self,
        source: Union[str, Path],
    ) -> List[str]:
        """
        Load HTML and return as a list of text chunks.

        This method extracts text from the HTML file, processes it with the configured
        chunker, and returns a list of text chunks.

        Args:
            source: Path to HTML file

        Returns:
            List of text chunks extracted from the HTML file

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source is not an HTML file
            Exception: If HTML processing fails
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Validate file exists
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Validate file is an HTML
        if not self._is_html_file(source):
            raise ValueError(
                f"File is not an HTML: {source}. Extension: {Path(source).suffix}")

        logger.info(f"Loading HTML file: {source}")
        start_time = time.time()

        try:
            # Read and parse HTML
            with open(source, 'r', encoding=self._encoding, errors='replace') as f:
                html_content = f.read()

            # Extract text from HTML
            full_text = self._extract_text_from_html(html_content)

            # Chunk the text using TextChunker
            text_chunks = self._chunk_text(full_text)

            elapsed = time.time() - start_time
            logger.info(
                f"HTML processed in {elapsed:.2f}s: {len(text_chunks)} chunks")

            return text_chunks

        except Exception as e:
            logger.error(f"Error processing HTML {source}: {e}")
            raise

    def load_as_nodes(
        self,
        source: Union[str, Path],
        source_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Node]:
        """
        Load HTML and return as Node objects with metadata.

        Args:
            source: Path to HTML file
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

        # Extract HTML metadata if enabled
        html_metadata = {}
        if self._extract_metadata:
            try:
                with open(source, 'r', encoding=self._encoding, errors='replace') as f:
                    html_content = f.read()
                html_metadata = self._extract_html_metadata(html_content)
            except Exception as e:
                logger.warning(f"Failed to extract HTML metadata: {e}")

        # Create nodes with metadata
        nodes = []
        for idx, chunk in enumerate(chunks):
            # Build custom metadata with HTML info
            node_custom = custom_metadata.copy() if custom_metadata else {}
            node_custom.update(html_metadata)
            node_custom['content_type'] = 'html'

            metadata = NodeMetadata(
                source_file_uuid=source_id,
                position=idx,
                custom=node_custom
            )
            node = Node(content=chunk, metadata=metadata)
            nodes.append(node)

        logger.info(f"Created {len(nodes)} nodes from HTML: {source}")
        return nodes

    def load_as_vsfile(
        self,
        file_path: Union[str, Path],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> VSFile:
        """
        Load HTML and return as VSFile object.

        Args:
            file_path: Path to HTML file
            custom_metadata: Optional custom metadata

        Returns:
            VSFile object with nodes

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not an HTML
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self._is_html_file(file_path):
            raise ValueError(f"File is not an HTML: {file_path}")

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

    def _is_html_file(self, file_path: str) -> bool:
        """
        Check if file is an HTML based on extension.

        Args:
            file_path: Path to file

        Returns:
            True if file is an HTML, False otherwise
        """
        path = Path(file_path)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _extract_text_from_html(self, html_content: str) -> str:
        """
        Extract text from HTML content.

        Args:
            html_content: Raw HTML content

        Returns:
            Extracted text content
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error(
                "beautifulsoup4 is required for HTML support. Install with: pip install beautifulsoup4")
            raise ImportError("beautifulsoup4 is required for HTML support")

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove unwanted tags
            for tag in self.REMOVE_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()

            # Extract text with structure preservation if enabled
            if self._preserve_structure:
                text = self._extract_with_structure(soup)
            else:
                text = self._extract_plain_text(soup)

            return text

        except Exception as e:
            logger.error(f"Failed to extract text from HTML: {e}")
            raise

    def _extract_with_structure(self, soup) -> str:
        """
        Extract text while preserving semantic structure.

        Args:
            soup: BeautifulSoup object

        Returns:
            Extracted text with structure hints
        """
        text_parts = []

        # Get title if present
        title = soup.find('title')
        if title and title.string:
            text_parts.append(f"Title: {title.string.strip()}")
            text_parts.append("")

        # Process body content
        body = soup.find('body') or soup

        for element in body.descendants:
            if element.name in self.SECTION_TAGS:
                # Add section header
                header_text = element.get_text(strip=True)
                if header_text:
                    # Add visual separator for headings
                    if element.name.startswith('h') and len(element.name) == 2:
                        level = int(element.name[1])
                        prefix = '#' * level
                        text_parts.append(f"\n{prefix} {header_text}")
                    else:
                        text_parts.append(f"\n[{element.name.upper()}]")
                        text_parts.append(header_text)

            elif element.name == 'p':
                para_text = element.get_text(strip=True)
                if para_text:
                    text_parts.append(para_text)
                    text_parts.append("")

            elif element.name in ['li']:
                li_text = element.get_text(strip=True)
                if li_text:
                    text_parts.append(f"• {li_text}")

            elif element.name == 'a' and self._preserve_links:
                link_text = element.get_text(strip=True)
                href = element.get('href', '')
                if link_text and href:
                    text_parts.append(f"[{link_text}]({href})")

            elif element.name == 'table':
                table_text = self._extract_table(element)
                if table_text:
                    text_parts.append(table_text)
                    text_parts.append("")

        # If no structured content found, fall back to plain text
        if not text_parts:
            return self._extract_plain_text(soup)

        # Clean and join text
        full_text = '\n'.join(text_parts)

        # Clean up excessive whitespace
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        full_text = re.sub(r' {2,}', ' ', full_text)

        return full_text.strip()

    def _extract_plain_text(self, soup) -> str:
        """
        Extract plain text from HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            Plain text content
        """
        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip()
                  for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text

    def _extract_table(self, table_element) -> str:
        """
        Extract text from table element.

        Args:
            table_element: BeautifulSoup table element

        Returns:
            Formatted table text
        """
        rows = []

        for row in table_element.find_all('tr'):
            cells = []
            for cell in row.find_all(['th', 'td']):
                cell_text = cell.get_text(strip=True)
                cells.append(cell_text)

            if cells:
                rows.append(' | '.join(cells))

        return '\n'.join(rows)

    def _extract_html_metadata(self, html_content: str) -> Dict[str, Any]:
        """
        Extract metadata from HTML (title, description, etc.).

        Args:
            html_content: Raw HTML content

        Returns:
            Dictionary of metadata
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {}

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            metadata = {}

            # Extract title
            title = soup.find('title')
            if title and title.string:
                metadata['title'] = title.string.strip()

            # Extract meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name', '').lower()
                content = meta.get('content', '')

                if name == 'description' and content:
                    metadata['description'] = content
                elif name == 'keywords' and content:
                    metadata['keywords'] = content
                elif name == 'author' and content:
                    metadata['author'] = content

            # Extract Open Graph metadata
            for meta in soup.find_all('meta', property=True):
                prop = meta.get('property', '').lower()
                content = meta.get('content', '')

                if prop.startswith('og:') and content:
                    key = prop.replace('og:', 'og_')
                    metadata[key] = content

            return metadata

        except Exception as e:
            logger.warning(f"Failed to extract HTML metadata: {e}")
            return {}

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using TextChunker.

        Args:
            text: Full text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        try:
            # Use TextChunker's chunk_text method
            chunk_dicts = self._chunker.chunk_text(text, context="html")

            # Extract just the text from the chunk dictionaries
            text_chunks = [chunk_dict['text'] for chunk_dict in chunk_dicts]

            logger.debug(f"Chunked HTML text into {len(text_chunks)} chunks")
            return text_chunks

        except Exception as e:
            logger.warning(f"Failed to chunk text with TextChunker: {e}")
            # Fall back to returning the whole text as a single chunk
            logger.info("Falling back to single chunk")
            return [text]


def create_html_loader(
    chunk_size: int = 3000,
    chunk_overlap: int = 200,
    min_sentences_per_chunk: int = 5,
    tokenizer: str = "character",
    extract_metadata: bool = True,
    preserve_links: bool = False,
    preserve_structure: bool = True,
    encoding: str = 'utf-8'
) -> HtmlLoader:
    """
    Factory function to create an HTML loader.

    Args:
        chunk_size: Maximum tokens per chunk (default: 3000)
        chunk_overlap: Overlap between chunks in tokens (default: 200)
        min_sentences_per_chunk: Minimum sentences per chunk (default: 5)
        tokenizer: Tokenizer for chunking - "character", "gpt2", or HuggingFace model (default: "character")
        extract_metadata: Whether to extract meta tags (default: True)
        preserve_links: Whether to preserve link references in output (default: False)
        preserve_structure: Whether to preserve HTML structure hints (default: True)
        encoding: File encoding (default: "utf-8")

    Returns:
        Configured HTML loader

    Example:
        >>> loader = create_html_loader(chunk_size=1024, chunk_overlap=64)
        >>> chunks = loader.run("page.html")
        >>> print(f"Extracted {len(chunks)} chunks")

        >>> # Create loader with link preservation
        >>> loader = create_html_loader(preserve_links=True)
        >>> chunks = loader.run("page.html")
    """
    config = {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'min_sentences_per_chunk': min_sentences_per_chunk,
        'tokenizer': tokenizer,
        'extract_metadata': extract_metadata,
        'preserve_links': preserve_links,
        'preserve_structure': preserve_structure,
        'encoding': encoding
    }

    return HtmlLoader(config=config)


__all__ = ["HtmlLoader", "create_html_loader"]
