"""
ODT Loader using odfpy library for ODT document processing.

This loader uses the odfpy library to extract text and images from ODT documents.
It supports:
- Text extraction with paragraph preservation
- Image extraction from the ODT archive
- Configurable chunking of plain text

The loader stores extracted images in a scratch folder within the data directory.
"""

from __future__ import annotations

import mimetypes
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from odf import text, teletype, draw
from odf.opendocument import load as odf_load
from odf.element import Element

from rakam_systems.ai_utils import logging
from rakam_systems.ai_core.interfaces.loader import Loader
from rakam_systems.ai_vectorstore.components.chunker import AdvancedChunker
from rakam_systems.ai_vectorstore.core import Node, NodeMetadata, VSFile

logger = logging.getLogger(__name__)


class OdtLoader(Loader):
    """
    ODT loader using odfpy for document processing.

    This loader provides ODT processing with support for:
    - Text extraction with paragraph preservation
    - Image extraction from ODT archive
    - Advanced text chunking
    - Configurable processing options

    The extracted content is chunked and returned as text or Node objects.
    Images are saved to a scratch directory for reference.
    """

    # Default configuration
    DEFAULT_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_CHUNK_SIZE = 2048
    DEFAULT_CHUNK_OVERLAP = 128
    DEFAULT_IMAGE_PATH = "data/ingestion_image/"  # Default path for extracted images

    def __init__(
        self,
        name: str = "odt_loader",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ODT loader with odfpy.

        Args:
            name: Component name
            config: Optional configuration with keys:
                - embed_model_id: HuggingFace model ID for tokenization (default: "sentence-transformers/all-MiniLM-L6-v2")
                - chunk_size: Maximum tokens per chunk (default: 2048)
                - chunk_overlap: Overlap between chunks in tokens (default: 128)
                - min_sentences_per_chunk: Minimum sentences per chunk (default: 1)
                - tokenizer: Tokenizer for chunking (default: "character")
                - save_images: Whether to save images to disk (default: True)
                - image_path: Path to save images (default: None, uses INGESTION_IMAGE_PATH env var or "data/ingestion_image/")
                - scratch_folder_name: Name of scratch folder (default: "scratch")
                - include_images_in_text: Whether to add image references to text (default: True)
        """
        super().__init__(name=name, config=config)

        # Extract configuration
        config = config or {}
        self._save_images = config.get('save_images', True)
        self._image_path = config.get('image_path') or os.getenv(
            'INGESTION_IMAGE_PATH', self.DEFAULT_IMAGE_PATH)
        self._scratch_folder_name = config.get(
            'scratch_folder_name', 'scratch')
        self._include_images_in_text = config.get(
            'include_images_in_text', True)

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
            strategy="default"  # We'll use chunk_text() which doesn't need a specific strategy
        )

        # Store last extraction info for image tracking
        self._last_scratch_dir = None
        self._last_image_files = []
        self._image_path_mapping: Dict[str, str] = {}

        logger.info(
            f"Initialized OdtLoader with chunk_size={self._chunk_size}, chunk_overlap={self._chunk_overlap}, image_path={self._image_path}")

    def run(self, source: str) -> List[str]:
        """
        Execute the primary operation for the component.

        This method satisfies the BaseComponent abstract method requirement
        and delegates to load_as_chunks.

        Args:
            source: Path to ODT file

        Returns:
            List of text chunks extracted from the ODT
        """
        return self.load_as_chunks(source)

    def load_as_text(
        self,
        source: Union[str, Path],
    ) -> str:
        """
        Load ODT and return as a single text string.

        This method extracts all text from the ODT and returns it as a single
        string without chunking. Useful when you need the full document text.

        Args:
            source: Path to ODT file

        Returns:
            Full text content of the ODT as a single string

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source is not an ODT file
            Exception: If ODT processing fails
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Validate file exists
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Validate file is an ODT
        if not self._is_odt_file(source):
            raise ValueError(
                f"File is not an ODT: {source}. MIME type: {mimetypes.guess_type(source)[0]}")

        logger.info(f"Loading ODT as text: {source}")
        start_time = time.time()

        try:
            # Create scratch directory in data folder
            scratch_dir = self._get_scratch_dir(source)
            self._last_scratch_dir = scratch_dir

            # Extract images if enabled
            image_files = []
            if self._save_images:
                image_dir = self._get_image_path(source)
                image_files = self._extract_images(source, Path(image_dir))
                self._last_image_files = image_files
                logger.info(f"Extracted {len(image_files)} images from ODT")

            # Extract text from ODT with image positions if enabled
            if self._include_images_in_text and image_files:
                full_text = self._extract_text_with_image_positions(
                    source, image_files)
            else:
                full_text = self._extract_text(source)

            elapsed = time.time() - start_time
            logger.info(
                f"ODT loaded as text in {elapsed:.2f}s: {len(full_text)} characters")

            return full_text

        except Exception as e:
            logger.error(f"Error loading ODT as text {source}: {e}")
            raise

    def load_as_chunks(
        self,
        source: Union[str, Path],
    ) -> List[str]:
        """
        Load ODT and return as a list of text chunks.

        This method extracts text from the ODT, processes it with the configured
        chunker, and returns a list of text chunks. Each chunk optionally includes
        image references.

        Args:
            source: Path to ODT file

        Returns:
            List of text chunks extracted from the ODT

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If source is not an ODT file
            Exception: If ODT processing fails
        """
        # Convert Path to string
        if isinstance(source, Path):
            source = str(source)

        # Validate file exists
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")

        # Validate file is an ODT
        if not self._is_odt_file(source):
            raise ValueError(
                f"File is not an ODT: {source}. MIME type: {mimetypes.guess_type(source)[0]}")

        logger.info(f"Loading ODT file: {source}")
        start_time = time.time()

        try:
            # Create scratch directory in data folder
            scratch_dir = self._get_scratch_dir(source)
            self._last_scratch_dir = scratch_dir

            # Extract images if enabled
            image_files = []
            if self._save_images:
                image_dir = self._get_image_path(source)
                image_files = self._extract_images(source, Path(image_dir))
                self._last_image_files = image_files
                logger.info(f"Extracted {len(image_files)} images from ODT")

            # Extract text from ODT with image positions if enabled
            if self._include_images_in_text and image_files:
                full_text = self._extract_text_with_image_positions(
                    source, image_files)
            else:
                full_text = self._extract_text(source)

            # Chunk the text using AdvancedChunker's chunk_text method
            text_chunks = self._chunk_text(full_text)

            elapsed = time.time() - start_time
            logger.info(
                f"ODT processed in {elapsed:.2f}s: {len(text_chunks)} chunks")

            return text_chunks

        except Exception as e:
            logger.error(f"Error processing ODT {source}: {e}")
            raise

    def load_as_nodes(
        self,
        source: Union[str, Path],
        source_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Node]:
        """
        Load ODT and return as Node objects with metadata.

        Args:
            source: Path to ODT file
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

        logger.info(f"Created {len(nodes)} nodes from ODT: {source}")
        return nodes

    def load_as_vsfile(
        self,
        file_path: Union[str, Path],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> VSFile:
        """
        Load ODT and return as VSFile object.

        Args:
            file_path: Path to ODT file
            custom_metadata: Optional custom metadata

        Returns:
            VSFile object with nodes

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not an ODT
        """
        if isinstance(file_path, Path):
            file_path = str(file_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self._is_odt_file(file_path):
            raise ValueError(f"File is not an ODT: {file_path}")

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

    def _is_odt_file(self, file_path: str) -> bool:
        """
        Check if file is an ODT based on extension and MIME type.

        Args:
            file_path: Path to file

        Returns:
            True if file is an ODT, False otherwise
        """
        # Check extension
        path = Path(file_path)
        if path.suffix.lower() != '.odt':
            return False

        # Check MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type not in ['application/vnd.oasis.opendocument.text', None]:
            return False

        return True

    def _get_scratch_dir(self, source_path: str) -> Path:
        """
        Get scratch directory for storing extracted files.

        The scratch directory is created inside the data folder relative to the source file.

        Args:
            source_path: Path to source ODT file

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

    def _get_image_path(self, source_path: str) -> str:
        """
        Get the path where images should be extracted.

        Uses the configured image path (from config, env var, or default).
        Creates a subdirectory based on the source document filename.

        Args:
            source_path: Path to source ODT file

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

    def get_image_path_mapping(self) -> Dict[str, str]:
        """
        Get the mapping of image paths.

        Returns:
            Dictionary mapping image filenames to absolute paths on disk
        """
        return self._image_path_mapping.copy()

    def get_image_absolute_path(self, image_filename: str) -> Optional[str]:
        """
        Get the absolute file path for an image.

        Args:
            image_filename: The image filename

        Returns:
            Absolute path to the image file, or None if not found
        """
        return self._image_path_mapping.get(image_filename)

    def _extract_text(self, odt_path: str) -> str:
        """
        Extract text from ODT file using odfpy.

        Args:
            odt_path: Path to ODT file

        Returns:
            Extracted text content
        """
        try:
            textdoc = odf_load(odt_path)
            allparas = textdoc.getElementsByType(text.P)
            extracted_text = "\n".join(
                teletype.extractText(para) for para in allparas)

            logger.debug(
                f"Extracted {len(extracted_text)} characters from ODT")
            return extracted_text

        except Exception as e:
            logger.error(f"Failed to extract text from ODT: {e}")
            raise

    def _extract_text_with_image_positions(self, odt_path: str, image_files: List[str]) -> str:
        """
        Extract text from ODT file and insert image references at their correct positions.

        This method traverses the ODT document structure and tracks where images appear
        relative to text, inserting image markers at the appropriate positions.

        Args:
            odt_path: Path to ODT file
            image_files: List of extracted image file paths

        Returns:
            Extracted text with image references at correct positions
        """
        try:
            textdoc = odf_load(odt_path)

            # Create a mapping of image names in ODT to extracted file paths
            image_name_to_path = {}
            for img_path in image_files:
                img_name = Path(img_path).name
                image_name_to_path[img_name] = img_path

            # Get all body content elements (paragraphs, frames, etc.)
            body = textdoc.body
            text_parts = []
            processed_frames = set()  # Track processed frames to avoid duplicates

            # Recursively traverse document elements
            def traverse_element(element):
                """Recursively traverse ODT elements and extract text with image positions."""
                # Check if this is a paragraph
                if hasattr(element, 'qname') and element.qname == (text.TEXTNS, 'p'):
                    para_text = teletype.extractText(element)
                    if para_text.strip():
                        text_parts.append(para_text)

                    # Check for images within or after this paragraph
                    # Images are typically in draw:frame elements
                    for child in element.childNodes:
                        if hasattr(child, 'qname') and child.qname == (draw.DRAWNS, 'frame'):
                            # Use object id to track unique frames
                            frame_id = id(child)
                            if frame_id not in processed_frames:
                                # Found a frame (which may contain an image)
                                image_ref = self._extract_image_reference_from_frame(
                                    child, image_name_to_path)
                                if image_ref:
                                    text_parts.append(image_ref)
                                    processed_frames.add(frame_id)

                # Check if this is a frame element (can be at various levels)
                elif hasattr(element, 'qname') and element.qname == (draw.DRAWNS, 'frame'):
                    frame_id = id(element)
                    if frame_id not in processed_frames:
                        image_ref = self._extract_image_reference_from_frame(
                            element, image_name_to_path)
                        if image_ref:
                            text_parts.append(image_ref)
                            processed_frames.add(frame_id)

                # Recursively process child elements (but skip frames since we handle them above)
                if hasattr(element, 'childNodes'):
                    for child in element.childNodes:
                        if isinstance(child, Element):
                            # Don't recurse into frames we've already processed
                            if not (hasattr(child, 'qname') and child.qname == (draw.DRAWNS, 'frame') and id(child) in processed_frames):
                                traverse_element(child)

            # Start traversal from body
            traverse_element(body)

            # Join all text parts
            extracted_text = "\n".join(text_parts)

            logger.debug(
                f"Extracted {len(extracted_text)} characters from ODT with image positions")
            return extracted_text

        except Exception as e:
            logger.error(
                f"Failed to extract text with image positions from ODT: {e}")
            # Fall back to regular text extraction
            logger.warning(
                "Falling back to regular text extraction without image positioning")
            return self._extract_text(odt_path)

    def _extract_image_reference_from_frame(self, frame_element, image_name_to_path: Dict[str, str]) -> Optional[str]:
        """
        Extract image reference from a draw:frame element.

        Args:
            frame_element: ODT frame element that may contain an image
            image_name_to_path: Mapping of image names to their extracted file paths

        Returns:
            Image reference string or None if no image found
        """
        try:
            # Look for draw:image elements within the frame
            for child in frame_element.childNodes:
                if hasattr(child, 'qname') and child.qname == (draw.DRAWNS, 'image'):
                    # Get the image href
                    href = child.getAttribute('href')
                    if href:
                        # Extract image filename from href (e.g., "Pictures/image1.png" -> "image1.png")
                        img_filename = Path(href).name

                        # Look up the extracted file path
                        if img_filename in image_name_to_path:
                            img_path = image_name_to_path[img_filename]
                            return f"\n![Image]({img_path})"
                        else:
                            logger.debug(
                                f"Image {img_filename} referenced but not found in extracted files")

        except Exception as e:
            logger.debug(f"Error extracting image reference from frame: {e}")

        return None

    def _extract_images(self, odt_path: str, output_dir: Path) -> List[str]:
        """
        Extract all images from an ODT file.

        ODT files are ZIP archives with images stored in the Pictures/ directory.

        Args:
            odt_path: Path to the ODT file
            output_dir: Directory to save extracted images

        Returns:
            List of paths to extracted image files
        """
        odt_path = Path(odt_path)
        extracted_files = []

        # Clear previous mapping
        self._image_path_mapping.clear()

        try:
            # ODT files are ZIP archives
            with zipfile.ZipFile(odt_path, 'r') as zip_ref:
                # Images are typically stored in Pictures/ directory
                for file_info in zip_ref.filelist:
                    # Extract only image files
                    if file_info.filename.startswith('Pictures/') and not file_info.is_dir():
                        # Extract the file
                        filename = Path(file_info.filename).name
                        extracted_path = output_dir / filename
                        with zip_ref.open(file_info) as source, open(extracted_path, 'wb') as target:
                            target.write(source.read())
                        extracted_files.append(str(extracted_path))

                        # Build image path mapping
                        self._image_path_mapping[filename] = str(
                            extracted_path)
                        logger.debug(f"Extracted image: {extracted_path}")

            logger.info(f"Extracted {len(extracted_files)} images from ODT")
            logger.info(
                f"Built image path mapping with {len(self._image_path_mapping)} images")

        except Exception as e:
            logger.warning(f"Failed to extract images from ODT: {e}")

        return extracted_files

    def _add_image_references_to_text(self, text: str, image_files: List[str]) -> str:
        """
        Add image references to the extracted text.

        Args:
            text: Extracted text content
            image_files: List of extracted image file paths

        Returns:
            Text with appended image references
        """
        if not image_files:
            return text

        # Add image references at the end of the text
        image_refs = "\n\n--- Embedded Images ---\n"
        for img_path in image_files:
            img_name = Path(img_path).name
            image_refs += f"\n![Image]({img_path})"

        return text + image_refs

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using AdvancedChunker's chunk_text method.

        This method uses chunk_text() which is specifically designed for plain text strings,
        as opposed to run() which processes PDF parsing data types.

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

            logger.info(f"Chunked text into {len(text_chunks)} chunks")
            return text_chunks

        except Exception as e:
            logger.warning(f"Failed to chunk text with AdvancedChunker: {e}")
            # Fall back to returning the whole text as a single chunk
            logger.info("Falling back to single chunk")
            return [text]


def create_odt_loader(
    chunk_size: int = 2048,
    chunk_overlap: int = 128,
    min_sentences_per_chunk: int = 1,
    tokenizer: str = "character",
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    save_images: bool = True,
    scratch_folder_name: str = 'scratch',
    include_images_in_text: bool = True
) -> OdtLoader:
    """
    Factory function to create an ODT loader.

    Args:
        chunk_size: Maximum tokens per chunk (default: 2048)
        chunk_overlap: Overlap between chunks in tokens (default: 128)
        min_sentences_per_chunk: Minimum sentences per chunk (default: 1)
        tokenizer: Tokenizer for chunking - "character", "gpt2", or HuggingFace model (default: "character")
        embed_model_id: HuggingFace model ID for tokenization (default: "sentence-transformers/all-MiniLM-L6-v2")
        save_images: Whether to save extracted images (default: True)
        scratch_folder_name: Name of scratch folder in data directory (default: "scratch")
        include_images_in_text: Whether to include image references in text (default: True)

    Returns:
        Configured ODT loader

    Example:
        >>> loader = create_odt_loader(chunk_size=1024, chunk_overlap=64)
        >>> chunks = loader.run("data/document.odt")
        >>> print(f"Extracted {len(chunks)} chunks")

        >>> # Create loader without image references
        >>> loader = create_odt_loader(include_images_in_text=False)
        >>> chunks = loader.run("data/document.odt")
    """
    config = {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'min_sentences_per_chunk': min_sentences_per_chunk,
        'tokenizer': tokenizer,
        'embed_model_id': embed_model_id,
        'save_images': save_images,
        'scratch_folder_name': scratch_folder_name,
        'include_images_in_text': include_images_in_text
    }

    return OdtLoader(config=config)


__all__ = ["OdtLoader", "create_odt_loader"]
