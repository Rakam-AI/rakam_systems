"""
PDF Loader using pymupdf4llm library for lightweight PDF processing.

This loader uses the pymupdf4llm library to extract text from PDF documents
and convert them to markdown format. It provides a lightweight alternative to
the Docling-based loader with:
- Fast text extraction with markdown formatting
- Table support
- Image references
- Configurable chunking

The loader stores extracted markdown in a scratch folder within the data directory.
"""

from __future__ import annotations

import mimetypes
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pymupdf4llm

from rakam_systems_tools.utils import logging
from rakam_systems_core.interfaces.loader import Loader
from rakam_systems_vectorstore.components.chunker import AdvancedChunker
from rakam_systems_vectorstore.core import Node, NodeMetadata, VSFile

logger = logging.getLogger(__name__)

# Global lock for pymupdf4llm operations (library is not thread-safe)
_pymupdf4llm_lock = threading.Lock()


class PdfLoaderLight(Loader):
    """
    Lightweight PDF loader using pymupdf4llm for document processing.

    This loader provides fast PDF processing with support for:
    - Text extraction with markdown formatting
    - Table extraction
    - Image references
    - Configurable chunking

    The extracted content is chunked and returned as text or Node objects.
    Markdown files can be saved to a scratch directory for reference.
    """

    # Default configuration
    DEFAULT_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_CHUNKER_STRATEGY = "markdown_tables"
    DEFAULT_MAX_TOKENS = 1024  # Larger chunks for better context
    DEFAULT_MIN_CHUNK_TOKENS = 50  # Minimum tokens for standalone chunks
    DEFAULT_PAGE_CHUNKS = False  # Whether to chunk by page
    DEFAULT_DPI = 150  # DPI for image extraction
    DEFAULT_IMAGE_PATH = "data/ingestion_image/"  # Default path for extracted images

    def __init__(
        self,
        name: str = "pdf_loader_light",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize PDF loader with pymupdf4llm.

        Args:
            name: Component name
            config: Optional configuration with keys:
                - embed_model_id: HuggingFace model ID for tokenization (default: "sentence-transformers/all-MiniLM-L6-v2")
                - chunker_strategy: Strategy for chunking ("default", "markdown_tables", "annotations", default: "markdown_tables")
                - save_markdown: Whether to save markdown files (default: True)
                - scratch_folder_name: Name of scratch folder (default: "scratch")
                - max_tokens: Maximum tokens per chunk (default: 1024)
                - merge_peers: Whether to merge adjacent small chunks (default: True)
                - min_chunk_tokens: Minimum tokens for standalone chunks (default: 50)
                - filter_toc: Whether to filter out Table of Contents entries (default: True)
                - page_chunks: Whether to chunk by page (default: False)
                - write_images: Whether to extract images (default: True)
                - image_path: Path to save images (default: None, uses INGESTION_IMAGE_PATH env var or "data/ingestion_image/")
                - image_format: Format for extracted images (default: "png")
                - dpi: DPI for image extraction (default: 150)
                - margins: Margins for text extraction (default: (0, 50, 0, 50))
        """
        super().__init__(name=name, config=config)

        # Extract configuration
        config = config or {}
        self._save_markdown = config.get('save_markdown', True)
        self._scratch_folder_name = config.get(
            'scratch_folder_name', 'scratch')
        self._page_chunks = config.get('page_chunks', self.DEFAULT_PAGE_CHUNKS)
        self._write_images = config.get('write_images', True)
        self._image_format = config.get('image_format', 'png')
        self._dpi = config.get('dpi', self.DEFAULT_DPI)
        self._margins = config.get('margins', (0, 50, 0, 50))

        # Get image path from config, env var, or use default
        self._image_path = config.get('image_path') or os.getenv(
            'INGESTION_IMAGE_PATH', self.DEFAULT_IMAGE_PATH)

        # Chunker configuration
        self._max_tokens = config.get('max_tokens', self.DEFAULT_MAX_TOKENS)
        self._merge_peers = config.get('merge_peers', True)
        self._min_chunk_tokens = config.get(
            'min_chunk_tokens', self.DEFAULT_MIN_CHUNK_TOKENS)
        self._filter_toc = config.get('filter_toc', True)

        # Initialize advanced chunker
        embed_model_id = config.get(
            'embed_model_id', self.DEFAULT_EMBED_MODEL_ID)
        chunker_strategy = config.get(
            'chunker_strategy', self.DEFAULT_CHUNKER_STRATEGY)
        self._chunker = AdvancedChunker(
            embed_model_id=embed_model_id,
            strategy=chunker_strategy,
            max_tokens=self._max_tokens,
            merge_peers=self._merge_peers,
            min_chunk_tokens=self._min_chunk_tokens,
            filter_toc=self._filter_toc,
        )

        # Store last conversion result
        self._last_markdown = None
        self._last_scratch_dir = None

        # Store image path mapping: {relative_path_in_markdown: absolute_path_on_disk}
        self._image_path_mapping: Dict[str, str] = {}

        logger.info(
            f"Initialized PdfLoaderLight with chunker_strategy={chunker_strategy}, page_chunks={self._page_chunks}, dpi={self._dpi}, image_path={self._image_path}")

    def run(self, source: str) -> List[str]:
        """
        Execute the primary operation for the component.

        This method satisfies the BaseComponent abstract method requirement
        and delegates to load_as_chunks.

        Args:
            source: Path to PDF file

        Returns:
            List of text chunks extracted from the PDF
        """
        return self.load_as_chunks(source)

    def load_as_nodes(
        self,
        source: Union[str, Path],
        source_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Node]:
        """
        Load PDF and return as Node objects with metadata.

        Args:
            source: Path to PDF file
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

        logger.info(f"Created {len(nodes)} nodes from PDF: {source}")
        return nodes

    def load_as_text(
        self,
        source: Union[str, Path],
    ) -> str:
        """
        Load PDF and return as a single text string.

        This method extracts all text from the PDF and returns it as a single
        string without chunking. Useful when you need the full document text.

        Args:
            source: Path to PDF file

        Returns:
            Full text content of the PDF as a single string

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source is not a PDF file
            Exception: If PDF processing fails
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Validate file exists
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Validate file is a PDF
        if not self._is_pdf_file(source):
            raise ValueError(
                f"File is not a PDF: {source}. MIME type: {mimetypes.guess_type(source)[0]}")

        logger.info(f"Loading PDF as text: {source}")
        start_time = time.time()

        try:
            # Get image extraction path
            image_path = self._get_image_path(
                source) if self._write_images else None

            # Convert PDF to markdown using pymupdf4llm
            # Use lock because pymupdf4llm is not thread-safe
            with _pymupdf4llm_lock:
                md_text = pymupdf4llm.to_markdown(
                    source,
                    page_chunks=False,  # Get full document
                    write_images=self._write_images,
                    image_path=image_path,
                    image_format=self._image_format,
                    dpi=self._dpi,
                    margins=self._margins,
                )

            # Build image path mapping
            if self._write_images and image_path:
                self._build_image_path_mapping(md_text, image_path)

            # Store for later use
            self._last_markdown = md_text
            scratch_dir = self._get_scratch_dir(source)
            self._last_scratch_dir = scratch_dir

            # Save markdown if enabled
            if self._save_markdown:
                self._save_markdown_file(source, md_text, scratch_dir)

            elapsed = time.time() - start_time
            logger.info(
                f"PDF loaded as text in {elapsed:.2f}s: {len(md_text)} characters")

            return md_text

        except Exception as e:
            logger.error(f"Error loading PDF as text {source}: {e}")
            raise

    def load_as_chunks(
        self,
        source: Union[str, Path],
    ) -> List[str]:
        """
        Load PDF and return as a list of text chunks.

        This method extracts text from the PDF, processes it with the configured
        chunker strategy, and returns a list of text chunks. Each chunk includes
        contextualization.

        Args:
            source: Path to PDF file

        Returns:
            List of text chunks extracted from the PDF

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source is not a PDF file
            Exception: If PDF processing fails
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Validate file exists
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Validate file is a PDF
        if not self._is_pdf_file(source):
            raise ValueError(
                f"File is not a PDF: {source}. MIME type: {mimetypes.guess_type(source)[0]}")

        logger.info(f"Loading PDF file: {source}")
        start_time = time.time()

        try:
            # Get image extraction path
            image_path = self._get_image_path(
                source) if self._write_images else None

            # Convert PDF to markdown using pymupdf4llm
            # Use lock because pymupdf4llm is not thread-safe
            with _pymupdf4llm_lock:
                md_result = pymupdf4llm.to_markdown(
                    source,
                    page_chunks=self._page_chunks,
                    write_images=self._write_images,
                    image_path=image_path,
                    image_format=self._image_format,
                    dpi=self._dpi,
                    margins=self._margins,
                )

            # Build image path mapping
            if self._write_images and image_path:
                if isinstance(md_result, list):
                    # Page chunks - combine all text for mapping
                    full_text = "\n\n".join([page['text']
                                            for page in md_result])
                    self._build_image_path_mapping(full_text, image_path)
                else:
                    self._build_image_path_mapping(md_result, image_path)

            # Store for later use
            self._last_markdown = md_result
            scratch_dir = self._get_scratch_dir(source)
            self._last_scratch_dir = scratch_dir

            # Save markdown if enabled
            if self._save_markdown:
                if self._page_chunks and isinstance(md_result, list):
                    # Save each page separately
                    for idx, page_md in enumerate(md_result):
                        self._save_markdown_file(
                            source, page_md['text'], scratch_dir, page_num=idx+1)
                    # Also save full document
                    full_text = "\n\n".join([page['text']
                                            for page in md_result])
                    self._save_markdown_file(source, full_text, scratch_dir)
                else:
                    self._save_markdown_file(source, md_result, scratch_dir)

            # Extract and chunk text
            text_chunks = self._extract_and_chunk_text(md_result)

            elapsed = time.time() - start_time
            logger.info(
                f"PDF processed in {elapsed:.2f}s: {len(text_chunks)} chunks")

            return text_chunks

        except Exception as e:
            logger.error(f"Error processing PDF {source}: {e}")
            raise

    def load_as_vsfile(
        self,
        file_path: Union[str, Path],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> VSFile:
        """
        Load PDF and return as VSFile object.

        Args:
            file_path: Path to PDF file
            custom_metadata: Optional custom metadata

        Returns:
            VSFile object with nodes

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a PDF
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self._is_pdf_file(file_path):
            raise ValueError(f"File is not a PDF: {file_path}")

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

    def _is_pdf_file(self, file_path: str) -> bool:
        """
        Check if file is a PDF based on extension and MIME type.

        Args:
            file_path: Path to file

        Returns:
            True if file is a PDF, False otherwise
        """
        # Check extension
        path = Path(file_path)
        if path.suffix.lower() != '.pdf':
            return False

        # Check MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type != 'application/pdf':
            return False

        return True

    def _get_image_path(self, source_path: str) -> str:
        """
        Get the path where images should be extracted.

        Uses the configured image path (from config, env var, or default).
        Creates a subdirectory based on the source PDF filename.

        Args:
            source_path: Path to source PDF file

        Returns:
            Absolute path to image extraction directory
        """
        source = Path(source_path)
        doc_filename = source.stem

        # Create base image path
        base_path = Path(self._image_path)

        # Create subdirectory for this document
        image_dir = base_path / doc_filename
        image_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Using image path: {image_dir}")
        return str(image_dir)

    def _build_image_path_mapping(self, markdown_text: str, image_path: str) -> None:
        """
        Build mapping between image references in markdown and actual file paths.

        Extracts image references from markdown (e.g., ![](image.png)) and maps them
        to their absolute paths on disk.

        Args:
            markdown_text: Markdown text containing image references
            image_path: Directory where images were extracted
        """
        # Clear previous mapping
        self._image_path_mapping.clear()

        # Find all image references in markdown: ![alt text](image_path)
        # Use a more robust pattern that handles parentheses in paths
        # Match everything between ![]( and the closing ) that comes before a newline or another ![
        image_pattern = r'!\[([^\]]*)\]\(([^)]+(?:\([^)]*\)[^)]*)*\.(?:png|jpg|jpeg|gif|bmp|svg|webp))\)'
        matches = re.findall(image_pattern, markdown_text, re.IGNORECASE)

        for alt_text, markdown_path in matches:
            # The markdown_path could be:
            # 1. Relative path: "image.png" or "subdir/image.png"
            # 2. Full path: "data/ingestion_image/doc/image.png"

            # Convert to Path for easier manipulation
            path_obj = Path(markdown_path)

            # Check if the path exists as-is (might be absolute or relative)
            if path_obj.exists():
                absolute_path = path_obj.resolve()
            else:
                # Try combining with image_path
                absolute_path = Path(image_path) / path_obj.name

            # Store mapping: markdown path -> absolute path on disk
            if absolute_path.exists():
                self._image_path_mapping[markdown_path] = str(absolute_path)
                logger.debug(
                    f"Mapped image: {markdown_path} -> {absolute_path}")
            else:
                logger.warning(f"Image file not found: {absolute_path}")

        logger.info(
            f"Built image path mapping with {len(self._image_path_mapping)} images")

    def get_image_path_mapping(self) -> Dict[str, str]:
        """
        Get the mapping of image paths from markdown to actual file paths.

        Returns:
            Dictionary mapping relative paths in markdown to absolute paths on disk
        """
        return self._image_path_mapping.copy()

    def get_image_absolute_path(self, markdown_image_path: str) -> Optional[str]:
        """
        Get the absolute file path for an image referenced in the markdown.

        Args:
            markdown_image_path: The relative path as it appears in the markdown

        Returns:
            Absolute path to the image file, or None if not found
        """
        return self._image_path_mapping.get(markdown_image_path)

    def _get_scratch_dir(self, source_path: str) -> Path:
        """
        Get scratch directory for storing extracted files.

        The scratch directory is created inside the data folder relative to the source file.

        Args:
            source_path: Path to source PDF file

        Returns:
            Path to scratch directory
        """
        source = Path(source_path)

        # Find data folder - assume it's a parent of the source or sibling
        if 'data' in source.parts:
            # Navigate to data folder
            data_folder = source
            while data_folder.name != 'data' and data_folder.parent != data_folder:
                data_folder = data_folder.parent
        else:
            # Use parent directory and create/use data folder
            data_folder = source.parent / 'data'

        # Create scratch directory inside data folder
        scratch_dir = data_folder / self._scratch_folder_name
        scratch_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Using scratch directory: {scratch_dir}")
        return scratch_dir

    def _save_markdown_file(
        self,
        source_path: str,
        markdown_text: str,
        scratch_dir: Path,
        page_num: Optional[int] = None
    ) -> None:
        """
        Save markdown text to file.

        Args:
            source_path: Path to source PDF
            markdown_text: Markdown text to save
            scratch_dir: Scratch directory path
            page_num: Optional page number for page-specific files
        """
        try:
            doc_filename = Path(source_path).stem

            if page_num is not None:
                md_filename = scratch_dir / \
                    f"{doc_filename}-page-{page_num}.md"
            else:
                md_filename = scratch_dir / f"{doc_filename}.md"

            md_filename.write_text(markdown_text, encoding='utf-8')
            logger.debug(f"Saved markdown to {md_filename}")

        except Exception as e:
            logger.warning(f"Failed to save markdown file: {e}")

    def _extract_and_chunk_text(self, md_result: Union[str, List[Dict[str, Any]]]) -> List[str]:
        """
        Extract text and chunk it using AdvancedChunker.

        Args:
            md_result: Markdown result from pymupdf4llm (string or list of page dicts)

        Returns:
            List of text chunks with contextualization
        """
        text_chunks = []

        try:
            # Handle page_chunks=True case (list of dicts)
            if isinstance(md_result, list):
                # Each item is a dict with 'text' and metadata
                for page_dict in md_result:
                    page_text = page_dict.get('text', '')
                    if page_text and page_text.strip():
                        # Chunk each page separately
                        page_chunks = self._chunker.run([page_text])
                        text_chunks.extend(page_chunks)
            else:
                # page_chunks=False case (single string)
                text_chunks = self._chunker.run([md_result])

        except Exception as e:
            logger.warning(
                f"Failed to chunk document with AdvancedChunker: {e}")
            # Fall back to simple text extraction if advanced chunking fails
            logger.info("Falling back to simple text extraction")
            if isinstance(md_result, list):
                text_chunks = [page_dict.get(
                    'text', '') for page_dict in md_result if page_dict.get('text', '').strip()]
            else:
                text_chunks = [md_result]

        return text_chunks


def create_pdf_loader_light(
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunker_strategy: str = "markdown_tables",
    save_markdown: bool = True,
    scratch_folder_name: str = 'scratch',
    max_tokens: int = 1024,
    merge_peers: bool = True,
    min_chunk_tokens: int = 50,
    filter_toc: bool = True,
    page_chunks: bool = False,
    write_images: bool = True,
    image_path: Optional[str] = None,
    image_format: str = 'png',
    dpi: int = 150,
    margins: tuple = (0, 50, 0, 50),
) -> PdfLoaderLight:
    """
    Factory function to create a lightweight PDF loader.

    Args:
        embed_model_id: HuggingFace model ID for tokenization
        chunker_strategy: Strategy for chunking:
            - "default": Default serialization
            - "markdown_tables": Markdown table formatting (recommended)
            - "annotations": Include picture annotations
        save_markdown: Whether to save markdown files
        scratch_folder_name: Name of scratch folder in data directory
        max_tokens: Maximum tokens per chunk (default: 1024). Larger values create
                   bigger, more contextual chunks. Recommended: 512-2048.
        merge_peers: Whether to merge adjacent small chunks with same metadata (default: True)
        min_chunk_tokens: Minimum tokens for a standalone chunk (default: 50).
                         Smaller chunks will be merged with neighbors.
        filter_toc: Whether to filter out Table of Contents entries (default: True).
                   TOC entries often create noisy, low-value chunks.
        page_chunks: Whether to chunk by page (default: False). If True, each page
                    is processed separately.
        write_images: Whether to extract images from PDF (default: True)
        image_path: Path to save extracted images (default: None). If None, uses 
                   INGESTION_IMAGE_PATH env var or "data/ingestion_image/".
        image_format: Format for extracted images (default: 'png'). Options: 'png', 'jpg', 'ppm', 'pnm'
        dpi: DPI for image extraction (default: 150). Higher values = better quality but larger files.
        margins: Margins for text extraction in points (left, top, right, bottom).
                Default: (0, 50, 0, 50) - excludes 50pt from top and bottom.

    Returns:
        Configured lightweight PDF loader

    Example:
        >>> # Basic usage with default settings
        >>> loader = create_pdf_loader_light()
        >>> chunks = loader.load_as_chunks("data/document.pdf")
        >>> print(f"Extracted {len(chunks)} chunks")

        >>> # Create loader with larger chunks and page-based chunking
        >>> loader = create_pdf_loader_light(
        ...     max_tokens=2048,
        ...     page_chunks=True,
        ...     filter_toc=True
        ... )
        >>> chunks = loader.load_as_chunks("data/document.pdf")

        >>> # Create loader with high-quality image extraction
        >>> loader = create_pdf_loader_light(
        ...     write_images=True,
        ...     dpi=300,
        ...     image_format='png'
        ... )
        >>> chunks = loader.load_as_chunks("data/document.pdf")

        >>> # Access image path mapping after loading
        >>> loader = create_pdf_loader_light(write_images=True)
        >>> chunks = loader.load_as_chunks("data/document.pdf")
        >>> image_mapping = loader.get_image_path_mapping()
        >>> # image_mapping = {'image-1.png': '/path/to/data/ingestion_image/document/image-1.png', ...}
        >>> # Get specific image path from markdown reference
        >>> abs_path = loader.get_image_absolute_path('image-1.png')
    """
    config = {
        'embed_model_id': embed_model_id,
        'chunker_strategy': chunker_strategy,
        'save_markdown': save_markdown,
        'scratch_folder_name': scratch_folder_name,
        'max_tokens': max_tokens,
        'merge_peers': merge_peers,
        'min_chunk_tokens': min_chunk_tokens,
        'filter_toc': filter_toc,
        'page_chunks': page_chunks,
        'write_images': write_images,
        'image_path': image_path,
        'image_format': image_format,
        'dpi': dpi,
        'margins': margins,
    }

    return PdfLoaderLight(config=config)


__all__ = ["PdfLoaderLight", "create_pdf_loader_light"]
