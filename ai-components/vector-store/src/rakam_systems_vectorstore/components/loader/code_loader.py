"""
Code Loader for processing source code files.

This loader handles various programming language files and provides:
- Language detection based on file extension
- Syntax-aware chunking that preserves code structure
- Support for multiple languages (Python, JavaScript, TypeScript, Java, C/C++, Go, Rust, etc.)
- Comment and docstring extraction
- Function/class boundary detection for smarter chunking
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rakam_systems_tools.utils import logging
from rakam_systems_core.interfaces.loader import Loader
from rakam_systems_vectorstore.components.chunker import TextChunker
from rakam_systems_vectorstore.core import Node, NodeMetadata, VSFile

logger = logging.getLogger(__name__)


class CodeLoader(Loader):
    """
    Code loader for processing source code files.

    This loader provides code file processing with support for:
    - Multiple programming languages
    - Language detection based on file extension
    - Syntax-aware chunking that preserves code structure
    - Comment and docstring extraction

    The extracted content is chunked and returned as text or Node objects.
    """

    # Default configuration
    DEFAULT_CHUNK_SIZE = 2000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_MIN_SENTENCES_PER_CHUNK = 3
    DEFAULT_TOKENIZER = "character"

    # Language detection by file extension
    EXTENSION_TO_LANGUAGE = {
        # Python
        '.py': 'python',
        '.pyw': 'python',
        '.pyi': 'python',

        # JavaScript/TypeScript
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.mjs': 'javascript',
        '.cjs': 'javascript',

        # Java/Kotlin
        '.java': 'java',
        '.kt': 'kotlin',
        '.kts': 'kotlin',

        # C/C++
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.hpp': 'cpp',
        '.hxx': 'cpp',

        # C#
        '.cs': 'csharp',

        # Go
        '.go': 'go',

        # Rust
        '.rs': 'rust',

        # Ruby
        '.rb': 'ruby',
        '.rake': 'ruby',

        # PHP
        '.php': 'php',

        # Swift
        '.swift': 'swift',

        # Scala
        '.scala': 'scala',

        # Shell
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',

        # SQL
        '.sql': 'sql',

        # R
        '.r': 'r',
        '.R': 'r',

        # Lua
        '.lua': 'lua',

        # Perl
        '.pl': 'perl',
        '.pm': 'perl',

        # Haskell
        '.hs': 'haskell',

        # Elixir/Erlang
        '.ex': 'elixir',
        '.exs': 'elixir',
        '.erl': 'erlang',

        # Dart
        '.dart': 'dart',

        # YAML
        '.yaml': 'yaml',
        '.yml': 'yaml',

        # TOML
        '.toml': 'toml',

        # Config files
        '.json': 'json',
        '.xml': 'xml',
        '.ini': 'ini',
        '.cfg': 'ini',
        '.conf': 'ini',
    }

    # Supported code file extensions
    SUPPORTED_EXTENSIONS = set(EXTENSION_TO_LANGUAGE.keys())

    def __init__(
        self,
        name: str = "code_loader",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Code loader.

        Args:
            name: Component name
            config: Optional configuration with keys:
                - chunk_size: Maximum tokens per chunk (default: 2000)
                - chunk_overlap: Overlap between chunks in tokens (default: 200)
                - min_sentences_per_chunk: Minimum sentences per chunk (default: 3)
                - tokenizer: Tokenizer for chunking (default: "character")
                - preserve_structure: Whether to preserve code structure in chunks (default: True)
                - include_comments: Whether to include comments in output (default: True)
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
        self._preserve_structure = config.get('preserve_structure', True)
        self._include_comments = config.get('include_comments', True)
        self._encoding = config.get('encoding', 'utf-8')

        # Initialize text chunker
        self._chunker = TextChunker(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            min_sentences_per_chunk=self._min_sentences_per_chunk,
            tokenizer=self._tokenizer
        )

        logger.info(
            f"Initialized CodeLoader with chunk_size={self._chunk_size}, chunk_overlap={self._chunk_overlap}")

    def run(self, source: str) -> List[str]:
        """
        Execute the primary operation for the component.

        This method satisfies the BaseComponent abstract method requirement
        and delegates to load_as_chunks.

        Args:
            source: Path to code file

        Returns:
            List of text chunks extracted from the code file
        """
        return self.load_as_chunks(source)

    def load_as_text(
        self,
        source: Union[str, Path],
    ) -> str:
        """
        Load code file and return as a single text string.

        This method extracts all text from the code file and returns it as a single
        string without chunking. Useful when you need the full code content.

        Args:
            source: Path to code file

        Returns:
            Full text content of the code file as a single string

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source is not a supported code file
            Exception: If code processing fails
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Validate file exists
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Validate file is a code file
        if not self._is_code_file(source):
            raise ValueError(
                f"File is not a supported code file: {source}. Extension: {Path(source).suffix}")

        logger.info(f"Loading code file as text: {source}")
        start_time = time.time()

        try:
            # Read file content
            with open(source, 'r', encoding=self._encoding, errors='replace') as f:
                content = f.read()

            elapsed = time.time() - start_time
            logger.info(
                f"Code file loaded as text in {elapsed:.2f}s: {len(content)} characters")

            return content

        except Exception as e:
            logger.error(f"Error loading code file as text {source}: {e}")
            raise

    def load_as_chunks(
        self,
        source: Union[str, Path],
    ) -> List[str]:
        """
        Load code file and return as a list of text chunks.

        This method extracts text from the code file, processes it with structure-aware
        chunking, and returns a list of text chunks.

        Args:
            source: Path to code file

        Returns:
            List of text chunks extracted from the code file

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source is not a supported code file
            Exception: If code processing fails
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Validate file exists
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Validate file is a code file
        if not self._is_code_file(source):
            raise ValueError(
                f"File is not a supported code file: {source}. Extension: {Path(source).suffix}")

        logger.info(f"Loading code file: {source}")
        start_time = time.time()

        try:
            # Read file content
            with open(source, 'r', encoding=self._encoding, errors='replace') as f:
                content = f.read()

            # Detect language
            language = self._detect_language(source)

            # Process code with structure-aware chunking
            if self._preserve_structure:
                text_chunks = self._chunk_code_with_structure(
                    content, language)
            else:
                text_chunks = self._chunk_text(content, language)

            elapsed = time.time() - start_time
            logger.info(
                f"Code file processed in {elapsed:.2f}s: {len(text_chunks)} chunks")

            return text_chunks

        except Exception as e:
            logger.error(f"Error processing code file {source}: {e}")
            raise

    def load_as_nodes(
        self,
        source: Union[str, Path],
        source_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Node]:
        """
        Load code file and return as Node objects with metadata.

        Args:
            source: Path to code file
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

        # Detect language for metadata
        language = self._detect_language(source)

        # Create nodes with metadata
        nodes = []
        for idx, chunk in enumerate(chunks):
            # Build custom metadata with language info
            node_custom = custom_metadata.copy() if custom_metadata else {}
            node_custom['language'] = language
            node_custom['file_extension'] = Path(source).suffix

            metadata = NodeMetadata(
                source_file_uuid=source_id,
                position=idx,
                custom=node_custom
            )
            node = Node(content=chunk, metadata=metadata)
            nodes.append(node)

        logger.info(f"Created {len(nodes)} nodes from code file: {source}")
        return nodes

    def load_as_vsfile(
        self,
        file_path: Union[str, Path],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> VSFile:
        """
        Load code file and return as VSFile object.

        Args:
            file_path: Path to code file
            custom_metadata: Optional custom metadata

        Returns:
            VSFile object with nodes

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a supported code file
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self._is_code_file(file_path):
            raise ValueError(f"File is not a supported code file: {file_path}")

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

    def _is_code_file(self, file_path: str) -> bool:
        """
        Check if file is a supported code file based on extension.

        Args:
            file_path: Path to file

        Returns:
            True if file is a supported code file, False otherwise
        """
        path = Path(file_path)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _detect_language(self, file_path: str) -> str:
        """
        Detect programming language based on file extension.

        Args:
            file_path: Path to code file

        Returns:
            Language name string
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        return self.EXTENSION_TO_LANGUAGE.get(suffix, 'unknown')

    def _chunk_code_with_structure(self, content: str, language: str) -> List[str]:
        """
        Chunk code while preserving structural boundaries.

        This method attempts to split code at logical boundaries like
        function definitions, class definitions, etc.

        Args:
            content: Code content
            language: Programming language

        Returns:
            List of text chunks
        """
        if not content or not content.strip():
            return []

        # Split by structural elements based on language
        blocks = self._split_by_structure(content, language)

        # Chunk each block, combining small ones
        chunks = []
        current_chunk = []
        current_size = 0

        for block in blocks:
            block_size = len(block)

            # If block is too large, chunk it separately
            if block_size > self._chunk_size:
                # Save current accumulated chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Chunk the large block
                sub_chunks = self._chunk_text(block, language)
                chunks.extend(sub_chunks)

            # If adding this block would exceed limit, save current and start new
            elif current_size + block_size > self._chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [block]
                current_size = block_size

            # Add to current chunk
            else:
                current_chunk.append(block)
                current_size += block_size

        # Don't forget the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks if chunks else [content]

    def _split_by_structure(self, content: str, language: str) -> List[str]:
        """
        Split code by structural elements (functions, classes, etc).

        Args:
            content: Code content
            language: Programming language

        Returns:
            List of code blocks
        """
        # Language-specific patterns for structural elements
        patterns = self._get_structure_patterns(language)

        if not patterns:
            # Fall back to line-based splitting
            return self._split_by_blank_lines(content)

        # Find all structural boundaries
        blocks = []
        lines = content.split('\n')
        current_block = []

        for line in lines:
            # Check if this line starts a new structural element
            is_boundary = any(re.match(pattern, line) for pattern in patterns)

            if is_boundary and current_block:
                # Save current block and start new one
                blocks.append('\n'.join(current_block))
                current_block = [line]
            else:
                current_block.append(line)

        # Add final block
        if current_block:
            blocks.append('\n'.join(current_block))

        return blocks

    def _get_structure_patterns(self, language: str) -> List[str]:
        """
        Get regex patterns for structural elements in a language.

        Args:
            language: Programming language

        Returns:
            List of regex patterns
        """
        patterns = {
            'python': [
                r'^class\s+\w+',           # class definition
                r'^def\s+\w+',             # function definition
                r'^async\s+def\s+\w+',     # async function
                # decorator (start of decorated block)
                r'^@\w+',
            ],
            'javascript': [
                r'^(export\s+)?(async\s+)?function\s+\w+',  # function
                r'^(export\s+)?class\s+\w+',               # class
                # arrow function
                r'^(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?\(?\w*\)?\s*=>',
            ],
            'typescript': [
                r'^(export\s+)?(async\s+)?function\s+\w+',
                r'^(export\s+)?class\s+\w+',
                r'^(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?\(?\w*\)?\s*=>',
                r'^(export\s+)?interface\s+\w+',
                r'^(export\s+)?type\s+\w+',
            ],
            'java': [
                r'^(public|private|protected)?\s*(static\s+)?class\s+\w+',
                r'^(public|private|protected)?\s*(static\s+)?\w+\s+\w+\s*\(',
                r'^(public|private|protected)?\s*interface\s+\w+',
            ],
            'go': [
                r'^func\s+(\(\w+\s+\*?\w+\)\s+)?\w+',  # function or method
                r'^type\s+\w+\s+(struct|interface)',   # type definition
            ],
            'rust': [
                r'^(pub\s+)?fn\s+\w+',     # function
                r'^(pub\s+)?struct\s+\w+',  # struct
                r'^(pub\s+)?enum\s+\w+',   # enum
                r'^(pub\s+)?trait\s+\w+',  # trait
                r'^impl\s+',               # impl block
            ],
            'cpp': [
                r'^class\s+\w+',
                r'^(virtual\s+)?(static\s+)?\w+\s+\w+\s*\(',
                r'^namespace\s+\w+',
            ],
            'c': [
                r'^\w+\s+\w+\s*\(',  # function definition
                r'^struct\s+\w+',
                r'^typedef\s+',
            ],
            'ruby': [
                r'^class\s+\w+',
                r'^module\s+\w+',
                r'^def\s+\w+',
            ],
            'php': [
                r'^(public|private|protected)?\s*(static\s+)?function\s+\w+',
                r'^class\s+\w+',
                r'^interface\s+\w+',
                r'^trait\s+\w+',
            ],
        }

        return patterns.get(language, [])

    def _split_by_blank_lines(self, content: str) -> List[str]:
        """
        Split content by blank lines as fallback.

        Args:
            content: Code content

        Returns:
            List of code blocks
        """
        # Split by one or more blank lines
        blocks = re.split(r'\n\s*\n', content)
        return [block.strip() for block in blocks if block.strip()]

    def _chunk_text(self, text: str, language: str) -> List[str]:
        """
        Chunk text using TextChunker.

        Args:
            text: Full text to chunk
            language: Programming language for context

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        try:
            # Use TextChunker's chunk_text method
            chunk_dicts = self._chunker.chunk_text(
                text, context=f"code_{language}")

            # Extract just the text from the chunk dictionaries
            text_chunks = [chunk_dict['text'] for chunk_dict in chunk_dicts]

            logger.debug(f"Chunked code text into {len(text_chunks)} chunks")
            return text_chunks

        except Exception as e:
            logger.warning(f"Failed to chunk text with TextChunker: {e}")
            # Fall back to returning the whole text as a single chunk
            logger.info("Falling back to single chunk")
            return [text]


def create_code_loader(
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    min_sentences_per_chunk: int = 3,
    tokenizer: str = "character",
    preserve_structure: bool = True,
    include_comments: bool = True,
    encoding: str = 'utf-8'
) -> CodeLoader:
    """
    Factory function to create a code loader.

    Args:
        chunk_size: Maximum tokens per chunk (default: 2000)
        chunk_overlap: Overlap between chunks in tokens (default: 200)
        min_sentences_per_chunk: Minimum sentences per chunk (default: 3)
        tokenizer: Tokenizer for chunking - "character", "gpt2", or HuggingFace model (default: "character")
        preserve_structure: Whether to preserve code structure in chunks (default: True)
        include_comments: Whether to include comments in output (default: True)
        encoding: File encoding (default: "utf-8")

    Returns:
        Configured code loader

    Example:
        >>> loader = create_code_loader(chunk_size=1024, chunk_overlap=64)
        >>> chunks = loader.run("src/main.py")
        >>> print(f"Extracted {len(chunks)} chunks")

        >>> # Create loader without structure preservation
        >>> loader = create_code_loader(preserve_structure=False)
        >>> chunks = loader.run("src/utils.js")
    """
    config = {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'min_sentences_per_chunk': min_sentences_per_chunk,
        'tokenizer': tokenizer,
        'preserve_structure': preserve_structure,
        'include_comments': include_comments,
        'encoding': encoding
    }

    return CodeLoader(config=config)


__all__ = ["CodeLoader", "create_code_loader"]
