"""
EML Loader for processing email files (.eml format).

This loader uses Python's email library to extract text content from EML files.
It supports:
- Email header extraction (From, To, Subject, Date)
- Plain text email body extraction
- HTML email body extraction with text conversion
- Multipart email parsing
- Text-based chunking using TextChunker

The extracted content is chunked and returned as text or Node objects.
"""

from __future__ import annotations

import email
import os
import time
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rakam_systems.core.ai_utils import logging
from rakam_systems.core.ai_core.interfaces.loader import Loader
from vectorestore.components.chunker import TextChunker
from vectorestore.core import Node, NodeMetadata, VSFile

logger = logging.getLogger(__name__)


class EmlLoader(Loader):
    """
    EML loader for processing email files.

    This loader provides EML file processing with support for:
    - Email header extraction (From, To, Subject, Date)
    - Plain text and HTML email body extraction
    - Multipart email parsing
    - Text-based chunking with configurable parameters

    The extracted content is chunked using TextChunker and returned as text or Node objects.
    """

    # Default configuration
    DEFAULT_CHUNK_SIZE = 3000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_MIN_SENTENCES_PER_CHUNK = 5
    DEFAULT_TOKENIZER = "character"

    def __init__(
        self,
        name: str = "eml_loader",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize EML loader.

        Args:
            name: Component name
            config: Optional configuration with keys:
                - chunk_size: Maximum tokens per chunk (default: 3000)
                - chunk_overlap: Overlap between chunks in tokens (default: 200)
                - min_sentences_per_chunk: Minimum sentences per chunk (default: 5)
                - tokenizer: Tokenizer for chunking (default: "character")
                - include_headers: Whether to include email headers in output (default: True)
                - extract_html: Whether to extract text from HTML parts (default: True)
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
        self._include_headers = config.get('include_headers', True)
        self._extract_html = config.get('extract_html', True)

        # Initialize text chunker
        self._chunker = TextChunker(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            min_sentences_per_chunk=self._min_sentences_per_chunk,
            tokenizer=self._tokenizer
        )

        logger.info(
            f"Initialized EmlLoader with chunk_size={self._chunk_size}, chunk_overlap={self._chunk_overlap}")

    def run(self, source: str) -> List[str]:
        """
        Execute the primary operation for the component.

        This method satisfies the BaseComponent abstract method requirement
        and delegates to load_as_chunks.

        Args:
            source: Path to EML file

        Returns:
            List of text chunks extracted from the EML file
        """
        return self.load_as_chunks(source)

    def load_as_text(
        self,
        source: Union[str, Path],
    ) -> str:
        """
        Load EML and return as a single text string.

        This method extracts all text from the EML file and returns it as a single
        string without chunking. Useful when you need the full email content.

        Args:
            source: Path to EML file

        Returns:
            Full text content of the EML as a single string

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source is not an EML file
            Exception: If EML processing fails
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Validate file exists
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Validate file is an EML
        if not self._is_eml_file(source):
            raise ValueError(
                f"File is not an EML: {source}. Extension: {Path(source).suffix}")

        logger.info(f"Loading EML as text: {source}")
        start_time = time.time()

        try:
            # Extract text from EML
            full_text = self._extract_text_from_eml(source)

            elapsed = time.time() - start_time
            logger.info(
                f"EML loaded as text in {elapsed:.2f}s: {len(full_text)} characters")

            return full_text

        except Exception as e:
            logger.error(f"Error loading EML as text {source}: {e}")
            raise

    def load_as_chunks(
        self,
        source: Union[str, Path],
    ) -> List[str]:
        """
        Load EML and return as a list of text chunks.

        This method extracts text from the EML file, processes it with the configured
        chunker, and returns a list of text chunks.

        Args:
            source: Path to EML file

        Returns:
            List of text chunks extracted from the EML file

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source is not an EML file
            Exception: If EML processing fails
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Validate file exists
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Validate file is an EML
        if not self._is_eml_file(source):
            raise ValueError(
                f"File is not an EML: {source}. Extension: {Path(source).suffix}")

        logger.info(f"Loading EML file: {source}")
        start_time = time.time()

        try:
            # Extract text from EML
            full_text = self._extract_text_from_eml(source)

            # Chunk the text using TextChunker
            text_chunks = self._chunk_text(full_text)

            elapsed = time.time() - start_time
            logger.info(
                f"EML processed in {elapsed:.2f}s: {len(text_chunks)} chunks")

            return text_chunks

        except Exception as e:
            logger.error(f"Error processing EML {source}: {e}")
            raise

    def load_as_nodes(
        self,
        source: Union[str, Path],
        source_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Node]:
        """
        Load EML and return as Node objects with metadata.

        Each EML file is loaded as a single node (one email = one node).

        Args:
            source: Path to EML file
            source_id: Optional source identifier (defaults to file path)
            custom_metadata: Optional custom metadata to attach to nodes

        Returns:
            List of Node objects (single node containing the full email)
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Load full email text (no chunking)
        full_text = self.load_as_text(source)

        # Determine source ID
        if source_id is None:
            source_id = source

        # Create single node with metadata
        metadata = NodeMetadata(
            source_file_uuid=source_id,
            position=0,
            custom=custom_metadata or {}
        )
        node = Node(content=full_text, metadata=metadata)

        logger.info(f"Created 1 node from EML: {source}")
        return [node]

    def load_as_vsfile(
        self,
        file_path: Union[str, Path],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> VSFile:
        """
        Load EML and return as VSFile object.

        Args:
            file_path: Path to EML file
            custom_metadata: Optional custom metadata

        Returns:
            VSFile object with nodes

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not an EML
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self._is_eml_file(file_path):
            raise ValueError(f"File is not an EML: {file_path}")

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

    def _is_eml_file(self, file_path: str) -> bool:
        """
        Check if file is an EML based on extension.

        Args:
            file_path: Path to file

        Returns:
            True if file is an EML, False otherwise
        """
        # Check extension
        path = Path(file_path)
        return path.suffix.lower() == '.eml'

    def _extract_text_from_eml(self, eml_path: str) -> str:
        """
        Extract text from EML file including headers and body.

        Args:
            eml_path: Path to EML file

        Returns:
            Extracted text content
        """
        try:
            # Parse the EML file
            with open(eml_path, 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)

            # Extract headers if enabled
            text_parts = []

            if self._include_headers:
                headers_text = self._extract_headers(msg)
                if headers_text:
                    text_parts.append(headers_text)

            # Extract body content
            body_text = self._extract_body(msg)
            if body_text:
                text_parts.append(body_text)

            # Combine all parts
            full_text = "\n\n".join(text_parts)

            logger.debug(f"Extracted {len(full_text)} characters from EML")
            return full_text

        except Exception as e:
            logger.error(f"Failed to extract text from EML: {e}")
            raise

    def _extract_headers(self, msg: email.message.EmailMessage) -> str:
        """
        Extract relevant email headers.

        Args:
            msg: Email message object

        Returns:
            Formatted header text
        """
        headers = []

        # Extract common headers
        if msg['Subject']:
            headers.append(f"Subject: {msg['Subject']}")

        if msg['From']:
            headers.append(f"From: {msg['From']}")

        if msg['To']:
            headers.append(f"To: {msg['To']}")

        if msg['Date']:
            headers.append(f"Date: {msg['Date']}")

        if msg['Cc']:
            headers.append(f"Cc: {msg['Cc']}")

        return "\n".join(headers)

    def _extract_body(self, msg: email.message.EmailMessage) -> str:
        """
        Extract email body content from plain text and/or HTML parts.

        Args:
            msg: Email message object

        Returns:
            Extracted body text
        """
        body_parts = []

        # Try to get plain text body first
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                # Extract plain text
                if content_type == "text/plain":
                    try:
                        text = part.get_content()
                        if text and text.strip():
                            body_parts.append(text.strip())
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract plain text part: {e}")

                # Extract HTML and convert to text if enabled
                elif content_type == "text/html" and self._extract_html:
                    try:
                        html = part.get_content()
                        text = self._html_to_text(html)
                        if text and text.strip():
                            body_parts.append(text.strip())
                    except Exception as e:
                        logger.warning(f"Failed to extract HTML part: {e}")
        else:
            # Single part message
            content_type = msg.get_content_type()

            if content_type == "text/plain":
                try:
                    text = msg.get_content()
                    if text and text.strip():
                        body_parts.append(text.strip())
                except Exception as e:
                    logger.warning(f"Failed to extract plain text: {e}")

            elif content_type == "text/html" and self._extract_html:
                try:
                    html = msg.get_content()
                    text = self._html_to_text(html)
                    if text and text.strip():
                        body_parts.append(text.strip())
                except Exception as e:
                    logger.warning(f"Failed to extract HTML: {e}")

        return "\n\n".join(body_parts)

    def _html_to_text(self, html: str) -> str:
        """
        Convert HTML to plain text.

        Args:
            html: HTML content

        Returns:
            Plain text extracted from HTML
        """
        try:
            from bs4 import BeautifulSoup

            # Use 'lxml' parser for better performance (falls back to html.parser if not available)
            try:
                soup = BeautifulSoup(html, 'lxml')
            except Exception:
                soup = BeautifulSoup(html, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text - use separator for better text extraction
            text = soup.get_text(separator=' ', strip=True)

            # Clean up excessive whitespace more efficiently
            import re
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n', text)

            return text.strip()

        except ImportError:
            logger.warning(
                "beautifulsoup4 not installed, returning HTML as-is")
            return html
        except Exception as e:
            logger.warning(f"Failed to convert HTML to text: {e}")
            return html

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
            chunk_dicts = self._chunker.chunk_text(text, context="eml")

            # Extract just the text from the chunk dictionaries
            text_chunks = [chunk_dict['text'] for chunk_dict in chunk_dicts]

            logger.info(f"Chunked EML text into {len(text_chunks)} chunks")
            return text_chunks

        except Exception as e:
            logger.warning(f"Failed to chunk text with TextChunker: {e}")
            # Fall back to returning the whole text as a single chunk
            logger.info("Falling back to single chunk")
            return [text]


def create_eml_loader(
    chunk_size: int = 3000,
    chunk_overlap: int = 200,
    min_sentences_per_chunk: int = 5,
    tokenizer: str = "character",
    include_headers: bool = True,
    extract_html: bool = True
) -> EmlLoader:
    """
    Factory function to create an EML loader.

    Args:
        chunk_size: Maximum tokens per chunk (default: 3000)
        chunk_overlap: Overlap between chunks in tokens (default: 200)
        min_sentences_per_chunk: Minimum sentences per chunk (default: 5)
        tokenizer: Tokenizer for chunking - "character", "gpt2", or HuggingFace model (default: "character")
        include_headers: Whether to include email headers in output (default: True)
        extract_html: Whether to extract text from HTML parts (default: True)

    Returns:
        Configured EML loader

    Example:
        >>> loader = create_eml_loader(chunk_size=1024, chunk_overlap=64)
        >>> chunks = loader.run("data/email.eml")
        >>> print(f"Extracted {len(chunks)} chunks")

        >>> # Create loader without headers
        >>> loader = create_eml_loader(include_headers=False)
        >>> chunks = loader.run("data/email.eml")
    """
    config = {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'min_sentences_per_chunk': min_sentences_per_chunk,
        'tokenizer': tokenizer,
        'include_headers': include_headers,
        'extract_html': extract_html
    }

    return EmlLoader(config=config)


__all__ = ["EmlLoader", "create_eml_loader"]
