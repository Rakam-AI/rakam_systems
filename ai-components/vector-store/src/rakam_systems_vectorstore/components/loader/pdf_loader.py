"""
PDF Loader using Docling library for advanced PDF processing.

This loader uses the Docling library to extract text, images, tables, and figures
from PDF documents with high quality. It supports:
- Text extraction with layout preservation
- Image extraction (page images, figures, tables)
- Markdown export with embedded or referenced images
- Configurable image resolution

The loader stores extracted images and markdown in a scratch folder within the data directory.
"""

from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from rakam_systems_tools.utils import logging
from rakam_systems_core.interfaces.loader import Loader
from rakam_systems_vectorstore.components.chunker import AdvancedChunker
from rakam_systems_vectorstore.core import Node, NodeMetadata, VSFile

logger = logging.getLogger(__name__)


class PdfLoader(Loader):
    """
    PDF loader using Docling for advanced document processing.

    This loader provides high-quality PDF processing with support for:
    - Text extraction with layout preservation
    - Image extraction (pages, figures, tables)
    - Markdown export with images
    - Configurable processing options

    The extracted content is chunked and returned as text or Node objects.
    Images and markdown files are saved to a scratch directory for reference.
    """

    # Default configuration
    DEFAULT_IMAGE_SCALE = 2.0  # Scale=1 ~ 72 DPI, Scale=2 ~ 144 DPI
    DEFAULT_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_CHUNKER_STRATEGY = "markdown_tables"
    DEFAULT_MAX_TOKENS = 1024  # Larger chunks for better context
    DEFAULT_MIN_CHUNK_TOKENS = 50  # Minimum tokens for standalone chunks

    def __init__(
        self,
        name: str = "pdf_loader",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize PDF loader with Docling.

        Args:
            name: Component name
            config: Optional configuration with keys:
                - image_scale: Image resolution scale (default: 2.0)
                - generate_page_images: Whether to generate page images (default: True)
                - generate_picture_images: Whether to generate picture images (default: True)
                - embed_model_id: HuggingFace model ID for tokenization (default: "sentence-transformers/all-MiniLM-L6-v2")
                - chunker_strategy: Strategy for chunking ("default", "markdown_tables", "annotations", default: "markdown_tables")
                - save_images: Whether to save images to disk (default: True)
                - save_markdown: Whether to save markdown files (default: True)
                - scratch_folder_name: Name of scratch folder (default: "scratch")
                - include_images_in_chunks: Whether to include image references in text chunks (default: True)
                - max_tokens: Maximum tokens per chunk (default: 1024)
                - merge_peers: Whether to merge adjacent small chunks (default: True)
                - min_chunk_tokens: Minimum tokens for standalone chunks (default: 50)
                - filter_toc: Whether to filter out Table of Contents entries (default: True)
        """
        super().__init__(name=name, config=config)

        # Extract configuration
        config = config or {}
        self._image_scale = config.get('image_scale', self.DEFAULT_IMAGE_SCALE)
        self._generate_page_images = config.get('generate_page_images', True)
        self._generate_picture_images = config.get(
            'generate_picture_images', True)
        self._save_images = config.get('save_images', True)
        self._save_markdown = config.get('save_markdown', True)
        self._scratch_folder_name = config.get(
            'scratch_folder_name', 'scratch')
        self._include_images_in_chunks = config.get(
            'include_images_in_chunks', True)

        # Chunker configuration
        self._max_tokens = config.get('max_tokens', self.DEFAULT_MAX_TOKENS)
        self._merge_peers = config.get('merge_peers', True)
        self._min_chunk_tokens = config.get(
            'min_chunk_tokens', self.DEFAULT_MIN_CHUNK_TOKENS)
        self._filter_toc = config.get('filter_toc', True)

        # Initialize advanced chunker with improved settings
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

        # Initialize document converter with pipeline options
        self._doc_converter = self._create_converter()

        # Store conversion result for image tracking
        self._last_conv_res = None
        self._last_scratch_dir = None

        logger.info(
            f"Initialized PdfLoader with image_scale={self._image_scale}, chunker_strategy={chunker_strategy}, include_images_in_chunks={self._include_images_in_chunks}")

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

    def _create_converter(self) -> DocumentConverter:
        """Create and configure the Docling document converter."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = self._image_scale
        pipeline_options.generate_page_images = self._generate_page_images
        pipeline_options.generate_picture_images = self._generate_picture_images

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options)
            }
        )

        return doc_converter

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
            # Convert PDF document
            conv_res = self._doc_converter.convert(source)

            # Export the full document as markdown text
            full_text = conv_res.document.export_to_markdown()

            elapsed = time.time() - start_time
            logger.info(
                f"PDF loaded as text in {elapsed:.2f}s: {len(conv_res.document.pages)} pages, {len(full_text)} characters")

            return full_text

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
        contextualization and optionally image references.

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
            # Convert PDF document
            conv_res = self._doc_converter.convert(source)

            # Create scratch directory in data folder
            scratch_dir = self._get_scratch_dir(source)

            # Store for later use in image inclusion
            self._last_conv_res = conv_res
            self._last_scratch_dir = scratch_dir

            # Save images and tables if enabled
            if self._save_images:
                self._save_page_images(conv_res, scratch_dir)
                self._save_element_images(conv_res, scratch_dir)

            # Save markdown if enabled
            if self._save_markdown:
                self._save_markdown_files(conv_res, scratch_dir)

            # Extract text and chunk it
            text_chunks = self._extract_and_chunk_text(conv_res, scratch_dir)

            elapsed = time.time() - start_time
            logger.info(
                f"PDF processed in {elapsed:.2f}s: {len(conv_res.document.pages)} pages, {len(text_chunks)} chunks")

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

    def _save_page_images(self, conv_res, scratch_dir: Path) -> None:
        """Save page images to scratch directory."""
        doc_filename = conv_res.input.file.stem

        for page_no, page in conv_res.document.pages.items():
            if not hasattr(page, 'image') or page.image is None:
                continue

            page_image_filename = scratch_dir / \
                f"{doc_filename}-page-{page.page_no}.png"
            try:
                with page_image_filename.open("wb") as fp:
                    page.image.pil_image.save(fp, format="PNG")
                logger.debug(
                    f"Saved page {page.page_no} image to {page_image_filename}")
            except Exception as e:
                logger.warning(
                    f"Failed to save page {page.page_no} image: {e}")

    def _save_element_images(self, conv_res, scratch_dir: Path) -> None:
        """Save images of tables and figures to scratch directory."""
        doc_filename = conv_res.input.file.stem
        table_counter = 0
        picture_counter = 0

        for element, _level in conv_res.document.iterate_items():
            try:
                if isinstance(element, TableItem):
                    table_counter += 1
                    element_image_filename = (
                        scratch_dir /
                        f"{doc_filename}-table-{table_counter}.png"
                    )
                    with element_image_filename.open("wb") as fp:
                        element.get_image(conv_res.document).save(fp, "PNG")
                    logger.debug(
                        f"Saved table {table_counter} to {element_image_filename}")

                elif isinstance(element, PictureItem):
                    picture_counter += 1
                    element_image_filename = (
                        scratch_dir /
                        f"{doc_filename}-picture-{picture_counter}.png"
                    )
                    with element_image_filename.open("wb") as fp:
                        element.get_image(conv_res.document).save(fp, "PNG")
                    logger.debug(
                        f"Saved picture {picture_counter} to {element_image_filename}")

            except Exception as e:
                logger.warning(f"Failed to save element image: {e}")

        logger.info(
            f"Saved {table_counter} tables and {picture_counter} pictures")

    def _save_markdown_files(self, conv_res, scratch_dir: Path) -> None:
        """Save markdown files with images."""
        doc_filename = conv_res.input.file.stem

        try:
            # Save markdown with embedded images
            md_filename = scratch_dir / f"{doc_filename}-with-images.md"
            conv_res.document.save_as_markdown(
                md_filename, image_mode=ImageRefMode.EMBEDDED)
            logger.debug(
                f"Saved markdown with embedded images to {md_filename}")

            # Save markdown with referenced images
            md_filename = scratch_dir / f"{doc_filename}-with-image-refs.md"
            conv_res.document.save_as_markdown(
                md_filename, image_mode=ImageRefMode.REFERENCED)
            logger.debug(
                f"Saved markdown with image references to {md_filename}")

            # Save HTML with referenced images
            html_filename = scratch_dir / \
                f"{doc_filename}-with-image-refs.html"
            conv_res.document.save_as_html(
                html_filename, image_mode=ImageRefMode.REFERENCED)
            logger.debug(
                f"Saved HTML with image references to {html_filename}")

        except Exception as e:
            logger.warning(f"Failed to save markdown files: {e}")

    def _extract_and_chunk_text(self, conv_res, scratch_dir: Path) -> List[str]:
        """
        Extract text from conversion result and chunk it using AdvancedChunker.

        Args:
            conv_res: Docling conversion result
            scratch_dir: Path to scratch directory with images

        Returns:
            List of text chunks with contextualization and optional image references
        """
        text_chunks = []

        try:
            # Use AdvancedChunker to chunk the DoclingDocument directly
            # This provides better chunking with table support and contextualization
            chunk_count = 0
            for chunk in self._chunker.chunk_docling_document(conv_res.document):
                # Get contextualized text for each chunk
                ctx_text = self._chunker.contextualize(chunk=chunk)

                # If enabled, add image references to chunks
                if self._include_images_in_chunks:
                    logger.debug(
                        f"Processing chunk {chunk_count}: has meta={hasattr(chunk, 'meta')}")
                    if hasattr(chunk, 'meta'):
                        logger.debug(
                            f"  meta has doc_items={hasattr(chunk.meta, 'doc_items')}")
                        if hasattr(chunk.meta, 'doc_items'):
                            logger.debug(
                                f"  doc_items count={len(chunk.meta.doc_items)}")

                    ctx_text = self._add_images_to_chunk(
                        ctx_text, chunk, conv_res, scratch_dir)

                text_chunks.append(ctx_text)
                chunk_count += 1

        except Exception as e:
            logger.warning(
                f"Failed to chunk document with AdvancedChunker: {e}")
            # Fall back to simple text extraction if advanced chunking fails
            logger.info("Falling back to simple text extraction")
            text_chunks = self._extract_text_fallback(conv_res)

        return text_chunks

    def _add_images_to_chunk(self, chunk_text: str, chunk, conv_res, scratch_dir: Path) -> str:
        """
        Add image references to a text chunk based on text content matching.
        Images are added in the order they appear in the original document.

        This method:
        1. Gets the full markdown document text
        2. Finds images (![Image](...)) in the markdown
        3. Uses fuzzy matching to find which images belong to this chunk
        4. Appends image references to the chunk

        Args:
            chunk_text: The contextualized text of the chunk
            chunk: The chunk object from the chunker
            conv_res: Docling conversion result
            scratch_dir: Path to scratch directory with saved images

        Returns:
            Chunk text with appended image references in document order
        """
        doc_filename = conv_res.input.file.stem
        image_refs = []

        try:
            # Read the full document markdown text with image references from the saved file
            md_filename = scratch_dir / f"{doc_filename}-with-image-refs.md"
            if not md_filename.exists():
                logger.warning(
                    f"Markdown file with image references not found: {md_filename}")
                return chunk_text

            full_doc_text = md_filename.read_text()

            # Find all ![Image](...) markers in the markdown
            # Note: paths can contain parentheses, so we need to match until .png) or .jpg)
            import re
            image_pattern = r'!\[Image\]\((.+?\.(?:png|jpg|jpeg|gif|webp))\)'
            image_positions = []  # (position, img_path)

            for match in re.finditer(image_pattern, full_doc_text):
                img_pos = match.start()
                img_path_in_md = match.group(1)
                # Use the image path directly from markdown
                image_positions.append((img_pos, img_path_in_md))
                logger.debug(f"Found image in document at position {img_pos}")

            logger.debug(
                f"Found {len(image_positions)} images in full document (length: {len(full_doc_text)})")

            # Now find which images belong to this chunk
            # Strategy: Look for text snippets from the chunk in the full document
            # Split chunk into sentences/paragraphs for better matching
            chunk_lines = [line.strip() for line in chunk_text.split(
                '\n') if line.strip() and len(line.strip()) > 20]

            if chunk_lines:
                # Find the position range of this chunk in the full document
                # Try to match beginning and end of chunk
                # First 100 chars of first substantial line
                first_line = chunk_lines[0][:100]
                last_line = chunk_lines[-1][:100] if len(
                    chunk_lines) > 1 else first_line

                # Remove title markers that chunker might add
                first_line_clean = first_line.replace(
                    '## ', '').replace('# ', '').strip()

                chunk_start_pos = full_doc_text.find(first_line_clean)

                if chunk_start_pos == -1:
                    # Try with less text
                    first_line_clean = first_line_clean[:50]
                    chunk_start_pos = full_doc_text.find(first_line_clean)

                if chunk_start_pos != -1:
                    # Find chunk end - look for the last line
                    last_line_clean = last_line.replace(
                        '## ', '').replace('# ', '').strip()[:50]
                    chunk_end_search = full_doc_text.find(
                        last_line_clean, chunk_start_pos)

                    if chunk_end_search != -1:
                        chunk_end_pos = chunk_end_search + \
                            len(last_line_clean) + 500  # Add buffer
                    else:
                        chunk_end_pos = chunk_start_pos + \
                            len(chunk_text) + 500  # Estimate

                    logger.debug(
                        f"Chunk found at position {chunk_start_pos}-{chunk_end_pos}")

                    # Find images that fall within this chunk's range
                    for img_pos, img_path in sorted(image_positions):
                        if chunk_start_pos <= img_pos <= chunk_end_pos:
                            image_refs.append(f"\n![Image]({img_path})")
                            logger.debug(
                                f"Added image at position {img_pos} to chunk")
                else:
                    logger.debug(
                        f"Could not find chunk position in full document (tried: '{first_line_clean[:30]}...')")

            # Append image references to the chunk text
            if image_refs:
                chunk_text = chunk_text + "".join(image_refs)
                logger.info(
                    f"Added {len(image_refs)} image references to chunk")

        except Exception as e:
            logger.warning(f"Could not add images to chunk: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")

        return chunk_text

    def _extract_text_fallback(self, conv_res) -> List[str]:
        """
        Fallback method for text extraction if advanced chunking fails.

        Args:
            conv_res: Docling conversion result

        Returns:
            List of text chunks
        """
        text_parts = []

        # Extract text from each page
        for page_no, page in conv_res.document.pages.items():
            try:
                # Export page as markdown to preserve structure
                page_text = page.export_to_markdown()

                if page_text and page_text.strip():
                    text_parts.append(page_text)

            except Exception as e:
                logger.warning(
                    f"Failed to extract text from page {page_no}: {e}")

        # Join all text and use AdvancedChunker's raw text chunking
        full_text = "\n\n".join(text_parts)
        return self._chunker.run([full_text])


def create_pdf_loader(
    image_scale: float = 2.0,
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunker_strategy: str = "markdown_tables",
    save_images: bool = True,
    save_markdown: bool = True,
    scratch_folder_name: str = 'scratch',
    include_images_in_chunks: bool = True,
    max_tokens: int = 1024,
    merge_peers: bool = True,
    min_chunk_tokens: int = 50,
    filter_toc: bool = True,
) -> PdfLoader:
    """
    Factory function to create a PDF loader.

    Args:
        image_scale: Image resolution scale (1.0 ~ 72 DPI, 2.0 ~ 144 DPI)
        embed_model_id: HuggingFace model ID for tokenization
        chunker_strategy: Strategy for chunking:
            - "default": Default serialization
            - "markdown_tables": Markdown table formatting (recommended)
            - "annotations": Include picture annotations
            - "custom_placeholder": Custom image placeholders
        save_images: Whether to save extracted images
        save_markdown: Whether to save markdown files
        scratch_folder_name: Name of scratch folder in data directory
        include_images_in_chunks: Whether to include image references in text chunks (default: True)
        max_tokens: Maximum tokens per chunk (default: 1024). Larger values create
                   bigger, more contextual chunks. Recommended: 512-2048.
        merge_peers: Whether to merge adjacent small chunks with same metadata (default: True)
        min_chunk_tokens: Minimum tokens for a standalone chunk (default: 50).
                         Smaller chunks will be merged with neighbors.
        filter_toc: Whether to filter out Table of Contents entries (default: True).
                   TOC entries often create noisy, low-value chunks.

    Returns:
        Configured PDF loader

    Example:
        >>> # Basic usage with default settings
        >>> loader = create_pdf_loader()
        >>> chunks = loader.load_as_chunks("data/document.pdf")
        >>> print(f"Extracted {len(chunks)} chunks")

        >>> # Create loader with larger chunks and TOC filtering
        >>> loader = create_pdf_loader(
        ...     max_tokens=2048,
        ...     filter_toc=True,
        ...     min_chunk_tokens=100
        ... )
        >>> chunks = loader.load_as_chunks("data/document.pdf")

        >>> # Create loader without image references in chunks
        >>> loader = create_pdf_loader(include_images_in_chunks=False)
        >>> chunks = loader.load_as_chunks("data/document.pdf")
    """
    config = {
        'image_scale': image_scale,
        'embed_model_id': embed_model_id,
        'chunker_strategy': chunker_strategy,
        'save_images': save_images,
        'save_markdown': save_markdown,
        'scratch_folder_name': scratch_folder_name,
        'include_images_in_chunks': include_images_in_chunks,
        'generate_page_images': True,
        'generate_picture_images': True,
        'max_tokens': max_tokens,
        'merge_peers': merge_peers,
        'min_chunk_tokens': min_chunk_tokens,
        'filter_toc': filter_toc,
    }

    return PdfLoader(config=config)


__all__ = ["PdfLoader", "create_pdf_loader"]
