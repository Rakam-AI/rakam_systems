"""
Advanced Chunker with Customizable Serialization

This module provides an advanced chunking system that allows customization of
serialization strategies for different document elements (tables, pictures, etc.)
during the chunking process.

Key Features:
- Hybrid chunking with customizable serialization
- Support for different table serialization formats (triplet, markdown, etc.)
- Configurable picture serialization with annotation support
- Token-aware chunking with contextual information
- Extensible serializer provider pattern

Usage Example:
    ```python
    from advanced_chunker import AdvancedChunker
    
    # Create chunker with markdown tables
    chunker = AdvancedChunker(strategy="markdown_tables")
    
    # Chunk documents
    documents = ["document text here"]
    chunks = chunker.run(documents)
    ```
"""

from __future__ import annotations
import re
from typing import Any, Iterable, List, Optional, Type
from abc import abstractmethod

from rakam_systems.core.ai_core.interfaces.chunker import Chunker

try:
    from chonkie import SentenceChunker
    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hierarchical_chunker import (
    DocChunk,
    DocMeta,
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownTableSerializer,
    MarkdownPictureSerializer,
    MarkdownParams,
)
from docling_core.types.doc.document import (
    DoclingDocument,
    PictureClassificationData,
    PictureDescriptionData,
    PictureMoleculeData,
    PictureItem,
)
from docling_core.types.doc.labels import DocItemLabel
from transformers import AutoTokenizer
from typing_extensions import override


class BaseSerializerProvider(ChunkingSerializerProvider):
    """Base class for serializer providers with common configuration."""

    def __init__(
        self,
        table_serializer: Optional[Any] = None,
        picture_serializer: Optional[Any] = None,
        params: Optional[MarkdownParams] = None,
    ):
        """
        Initialize the serializer provider.

        Args:
            table_serializer: Custom table serializer instance
            picture_serializer: Custom picture serializer instance
            params: Markdown serialization parameters
        """
        self.table_serializer = table_serializer
        self.picture_serializer = picture_serializer
        self.params = params or MarkdownParams()

    @abstractmethod
    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        """Get the configured serializer for the document."""
        pass


class DefaultSerializerProvider(BaseSerializerProvider):
    """Default serializer provider with standard settings."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        """Get default serializer."""
        kwargs = {"doc": doc, "params": self.params}
        if self.table_serializer:
            kwargs["table_serializer"] = self.table_serializer
        if self.picture_serializer:
            kwargs["picture_serializer"] = self.picture_serializer
        return ChunkingDocSerializer(**kwargs)


class MDTableSerializerProvider(BaseSerializerProvider):
    """
    Serializer provider that uses Markdown format for tables.

    This provider converts tables to Markdown format instead of the default
    triplet notation, making them more human-readable.
    """

    def __init__(self, params: Optional[MarkdownParams] = None):
        """Initialize with Markdown table serializer."""
        super().__init__(
            table_serializer=MarkdownTableSerializer(),
            params=params,
        )

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        """Get serializer with Markdown table formatting."""
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=self.table_serializer,
            params=self.params,
        )


class ImgPlaceholderSerializerProvider(BaseSerializerProvider):
    """
    Serializer provider with customizable image placeholder.

    This provider allows you to specify a custom placeholder text for images
    in the serialized output.
    """

    def __init__(self, image_placeholder: str = "<!-- image -->"):
        """
        Initialize with custom image placeholder.

        Args:
            image_placeholder: Text to use as placeholder for images
        """
        super().__init__(
            params=MarkdownParams(image_placeholder=image_placeholder)
        )

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        """Get serializer with custom image placeholder."""
        return ChunkingDocSerializer(doc=doc, params=self.params)


class AnnotationPictureSerializer(MarkdownPictureSerializer):
    """
    Picture serializer that leverages picture annotations.

    This serializer extracts and includes annotation information such as:
    - Picture classifications (predicted class)
    - Molecule data (SMILES notation)
    - Picture descriptions
    """

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        """
        Serialize picture with annotations.

        Args:
            item: Picture item to serialize
            doc_serializer: Document serializer instance
            doc: Parent document
            **kwargs: Additional serialization arguments

        Returns:
            Serialization result with annotation text
        """
        text_parts: list[str] = []

        # Extract annotations
        for annotation in item.annotations:
            if isinstance(annotation, PictureClassificationData):
                predicted_class = (
                    annotation.predicted_classes[0].class_name
                    if annotation.predicted_classes
                    else None
                )
                if predicted_class is not None:
                    text_parts.append(f"Picture type: {predicted_class}")

            elif isinstance(annotation, PictureMoleculeData):
                text_parts.append(f"SMILES: {annotation.smi}")

            elif isinstance(annotation, PictureDescriptionData):
                text_parts.append(f"Picture description: {annotation.text}")

        # Join and post-process
        text_res = "\n".join(text_parts)
        text_res = doc_serializer.post_process(text=text_res)
        return create_ser_result(text=text_res, span_source=item)


class ImgAnnotationSerializerProvider(BaseSerializerProvider):
    """
    Serializer provider that includes picture annotations in output.

    This provider uses the AnnotationPictureSerializer to include rich
    annotation data for pictures in the chunked output.
    """

    def __init__(self):
        """Initialize with annotation picture serializer."""
        super().__init__(picture_serializer=AnnotationPictureSerializer())

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        """Get serializer with picture annotation support."""
        return ChunkingDocSerializer(
            doc=doc,
            picture_serializer=self.picture_serializer,
        )


class AdvancedChunker(Chunker):
    """
    Advanced chunker with customizable serialization strategies.

    This class implements the Chunker interface and wraps the HybridChunker
    to provide customizable serialization strategies for various document elements.

    Attributes:
        tokenizer: Tokenizer instance for token counting
        hybrid_chunker: Underlying HybridChunker instance
        embed_model_id: Model ID for tokenization
        serializer_provider: Provider for custom serialization
        include_heading_markers: Whether to include markdown # markers in headings
        max_tokens: Maximum tokens per chunk
        merge_peers: Whether to merge adjacent small chunks
        min_chunk_tokens: Minimum tokens for a chunk to be kept standalone
    """

    # Default configuration for better chunking quality
    DEFAULT_MAX_TOKENS = 1024  # Larger chunks for better context
    DEFAULT_MERGE_PEERS = True  # Merge small adjacent chunks
    DEFAULT_MIN_CHUNK_TOKENS = 50  # Minimum tokens for standalone chunks

    def __init__(
        self,
        embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        tokenizer: Optional[BaseTokenizer] = None,
        serializer_provider: Optional[ChunkingSerializerProvider] = None,
        strategy: Optional[str] = None,
        name: str = "advanced_chunker",
        include_heading_markers: bool = True,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        merge_peers: bool = DEFAULT_MERGE_PEERS,
        min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
        filter_toc: bool = True,
        **chunker_kwargs,
    ):
        """
        Initialize the advanced chunker.

        Args:
            embed_model_id: HuggingFace model ID for tokenization
            tokenizer: Optional custom tokenizer (if not provided, will be created)
            serializer_provider: Custom serializer provider for document elements
            strategy: Pre-configured strategy name (default, markdown_tables, etc.)
            name: Component name
            include_heading_markers: If True, adds markdown # markers to headings 
                                     in contextualized output (default: True)
            max_tokens: Maximum tokens per chunk (default: 1024)
            merge_peers: If True, merges adjacent small chunks with same metadata (default: True)
            min_chunk_tokens: Minimum tokens for a chunk to be kept, smaller chunks 
                             will be merged with neighbors (default: 50)
            filter_toc: If True, filters out Table of Contents entries (default: True)
            **chunker_kwargs: Additional arguments for HybridChunker
        """
        super().__init__(name=name)

        self.embed_model_id = embed_model_id
        self.include_heading_markers = include_heading_markers
        self.max_tokens = max_tokens
        self.merge_peers = merge_peers
        self.min_chunk_tokens = min_chunk_tokens
        self.filter_toc = filter_toc

        # Handle strategy-based provider creation
        if strategy is not None and serializer_provider is None:
            serializer_provider = self._create_provider_from_strategy(
                strategy, **chunker_kwargs
            )

        self.serializer_provider = serializer_provider

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(embed_model_id)
            )
        else:
            self.tokenizer = tokenizer

        # Initialize chunker with improved settings
        chunker_config = {
            "tokenizer": self.tokenizer,
            "max_tokens": max_tokens,
            "merge_peers": merge_peers,
        }
        if self.serializer_provider is not None:
            chunker_config["serializer_provider"] = self.serializer_provider
        chunker_config.update(chunker_kwargs)

        self.hybrid_chunker = HybridChunker(**chunker_config)

    def _create_provider_from_strategy(
        self, strategy: str, **kwargs
    ) -> ChunkingSerializerProvider:
        """Create a serializer provider from a strategy name."""
        provider_map = {
            "default": DefaultSerializerProvider,
            "markdown_tables": MDTableSerializerProvider,
            "custom_placeholder": ImgPlaceholderSerializerProvider,
            "annotations": ImgAnnotationSerializerProvider,
        }

        if strategy not in provider_map:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available: {list(provider_map.keys())}"
            )

        provider_class = provider_map[strategy]

        # Filter kwargs for provider initialization
        import inspect
        provider_sig = inspect.signature(provider_class.__init__)
        provider_params = set(provider_sig.parameters.keys()) - {"self"}
        provider_kwargs = {k: v for k,
                           v in kwargs.items() if k in provider_params}

        return provider_class(**provider_kwargs)

    def run(self, documents: List[str]) -> List[str]:
        """
        Split documents into smaller chunks.

        This implementation expects documents to be already processed by Docling
        or similar tools. For raw text, it falls back to simple chunking.

        Args:
            documents: List of document strings to chunk

        Returns:
            List of chunk strings
        """
        chunks = []

        for doc_str in documents:
            # Try to parse as DoclingDocument JSON
            try:
                import json
                # Check if it's JSON format
                json.loads(doc_str)
                doc = DoclingDocument.model_validate_json(doc_str)
                # Use hybrid chunker for structured documents
                for chunk in self.hybrid_chunker.chunk(dl_doc=doc):
                    ctx_text = self.contextualize(chunk=chunk)
                    chunks.append(ctx_text)
            except (Exception,):
                # Fall back to simple text chunking for raw text using Chonkie
                chunk_results = self.chunk_text(doc_str)
                chunks.extend([chunk_info["text"]
                              for chunk_info in chunk_results])

        return chunks

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 2048,
        chunk_overlap: int = 128,
        min_sentences_per_chunk: int = 1,
        tokenizer: str = "character",
    ) -> List[dict[str, Any]]:
        """
        Chunk raw text using the Chonkie library's SentenceChunker.

        This method provides a simpler alternative to the Docling-based chunking
        for plain text documents. It uses sentence-based chunking with configurable
        token limits and overlap.

        Args:
            text: Raw text to chunk
            chunk_size: Maximum tokens per chunk (default: 2048)
            chunk_overlap: Overlap between consecutive chunks in tokens (default: 128)
            min_sentences_per_chunk: Minimum sentences per chunk (default: 1)
            tokenizer: Tokenizer to use - "character", "gpt2", or any HuggingFace tokenizer (default: "character")

        Returns:
            List of dictionaries with chunk information:
                - text: The chunk text
                - token_count: Number of tokens in the chunk
                - start_index: Starting character index in original text
                - end_index: Ending character index in original text

        Raises:
            ImportError: If chonkie is not installed

        Example:
            ```python
            chunker = AdvancedChunker()
            chunks = chunker.chunk_text(
                "Your long text here...",
                chunk_size=1024,
                chunk_overlap=64
            )

            for chunk_info in chunks:
                print(f"Text: {chunk_info['text']}")
                print(f"Tokens: {chunk_info['token_count']}")
            ```
        """
        if not CHONKIE_AVAILABLE:
            raise ImportError(
                "chonkie is not installed. Please install it with: "
                "pip install chonkie==1.4.2"
            )

        # Initialize the Chonkie SentenceChunker
        chonkie_chunker = SentenceChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_sentences_per_chunk=min_sentences_per_chunk,
        )

        # Chunk the text
        chunks = chonkie_chunker(text)

        # Convert Chonkie chunks to our format
        result = []
        for chunk in chunks:
            chunk_info = {
                "text": chunk.text,
                "token_count": chunk.token_count,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
            }
            result.append(chunk_info)

        return result

    def chunk_docling_document(
        self,
        dl_doc: DoclingDocument,
        post_process: bool = True
    ) -> Iterable[BaseChunk]:
        """
        Generate chunks from a Docling document.

        This is an advanced method for working directly with DoclingDocument objects.
        For the standard Chunker interface, use the run() method.

        Args:
            dl_doc: DoclingDocument to chunk
            post_process: If True, applies post-processing to filter TOC and merge
                         small chunks (default: True)

        Returns:
            Iterable of BaseChunk objects
        """
        chunks = list(self.hybrid_chunker.chunk(dl_doc=dl_doc))

        if post_process:
            chunks = self._post_process_chunks(chunks)

        return chunks

    def _post_process_chunks(self, chunks: List[BaseChunk]) -> List[BaseChunk]:
        """
        Post-process chunks to improve quality.

        This method:
        1. Filters out Table of Contents entries
        2. Merges image-only chunks with adjacent content
        3. Merges incomplete table fragments with adjacent content
        4. Merges very small chunks with their neighbors
        5. Removes duplicate heading-only chunks

        Args:
            chunks: List of chunks to process

        Returns:
            Processed list of chunks
        """
        if not chunks:
            return chunks

        # First pass: Filter TOC and mark chunks for processing
        filtered_chunks = []
        # Chunks waiting to be merged (images, table fragments)
        pending_merge_chunks = []

        for chunk in chunks:
            # Filter TOC entries
            if self.filter_toc and self._is_toc_chunk(chunk):
                continue

            # Check if chunk is image-only or incomplete table fragment
            should_merge = (
                self._is_image_only_chunk(chunk) or
                self._is_incomplete_table_fragment(chunk)
            )

            if should_merge:
                # Accumulate chunks to merge with next content chunk
                pending_merge_chunks.append(chunk)
                continue

            # If we have pending chunks to merge, merge them with this chunk
            if pending_merge_chunks:
                # Prepend all pending chunks to this chunk
                merge_texts = [
                    merge_chunk.text for merge_chunk in pending_merge_chunks]
                chunk.text = "\n".join(merge_texts) + "\n" + chunk.text
                pending_merge_chunks = []

            filtered_chunks.append(chunk)

        # If there are still pending chunks at the end, append to last chunk
        if pending_merge_chunks and filtered_chunks:
            last_chunk = filtered_chunks[-1]
            merge_texts = [
                merge_chunk.text for merge_chunk in pending_merge_chunks]
            last_chunk.text = last_chunk.text + "\n" + "\n".join(merge_texts)

        # Second pass: Merge small chunks
        processed = []

        for chunk in filtered_chunks:
            # Check if chunk is too small
            token_count = self.count_tokens(chunk.text)

            if token_count < self.min_chunk_tokens and processed:
                # Try to merge with previous chunk
                prev_chunk = processed[-1]
                merged_text = prev_chunk.text + "\n\n" + chunk.text
                merged_tokens = self.count_tokens(merged_text)

                # Only merge if it doesn't exceed max_tokens
                if merged_tokens <= self.max_tokens:
                    # Update the text while preserving the chunk structure
                    prev_chunk.text = merged_text
                    continue

            processed.append(chunk)

        return processed

    def _is_image_only_chunk(self, chunk: BaseChunk) -> bool:
        """
        Check if a chunk contains only image placeholders.

        Image-only chunks typically contain only:
        - <!-- image --> placeholders
        - Whitespace and newlines
        - No meaningful text content

        Args:
            chunk: Chunk to check

        Returns:
            True if chunk is image-only
        """
        text = chunk.text.strip()

        # Remove all image placeholders
        text_without_images = re.sub(
            r'<!--\s*image\s*-->', '', text, flags=re.IGNORECASE)
        text_without_images = text_without_images.strip()

        # If nothing remains after removing image placeholders, it's image-only
        if not text_without_images:
            return True

        # Also check for very short content that's just whitespace or punctuation
        # This catches cases where there might be a stray character
        if len(text_without_images) < 5 and not any(c.isalnum() for c in text_without_images):
            return True

        return False

    def _is_incomplete_table_fragment(self, chunk: BaseChunk) -> bool:
        """
        Check if a chunk contains an incomplete table fragment.

        Incomplete table fragments typically contain:
        - Only table separator lines (|---|---|)
        - Only table borders without content
        - Very short lines with mostly dashes and pipes
        - Single dash or pipe character lines

        Args:
            chunk: Chunk to check

        Returns:
            True if chunk is an incomplete table fragment
        """
        text = chunk.text.strip()

        # Remove heading markers to get the actual content
        lines = text.split('\n')
        content_lines = []

        for line in lines:
            # Skip heading lines (starting with #)
            stripped = line.strip()
            if not stripped.startswith('#'):
                content_lines.append(stripped)

        # If no content lines, not a table fragment
        if not content_lines:
            return False

        # Join content lines
        content = '\n'.join(content_lines).strip()

        # Check if it's only table separators (lines with |, -, and whitespace)
        # Pattern: lines containing mostly |, -, and spaces
        table_separator_pattern = r'^[\s\|\-]+$'

        # Check each content line
        separator_lines = 0
        total_content_lines = len(content_lines)

        for line in content_lines:
            if not line.strip():
                continue
            # Check if line is mostly table separators
            if re.match(table_separator_pattern, line):
                separator_lines += 1

        # If all non-empty lines are separators, it's an incomplete fragment
        if separator_lines > 0 and separator_lines == total_content_lines:
            return True

        # Check for very short content that's mostly punctuation
        # Remove all whitespace, pipes, and dashes
        content_cleaned = re.sub(r'[\s\|\-]', '', content)

        # If very little actual content remains (less than 10 chars),
        # and original has table markers, it's likely a fragment
        if len(content_cleaned) < 10 and ('|' in content or '---' in content):
            return True

        return False

    def _is_toc_chunk(self, chunk: BaseChunk) -> bool:
        """
        Check if a chunk is a Table of Contents entry.

        TOC entries typically:
        - Have "Table of Contents", "Table des matières", "Contents", "Sommaire" headings
        - Contain many dots (....) or dashes (----) as separators
        - Have page numbers at the end of lines

        Args:
            chunk: Chunk to check

        Returns:
            True if chunk appears to be a TOC entry
        """
        text = chunk.text.lower()

        # Check for TOC heading patterns
        toc_headings = [
            "table of contents",
            "table des matières",
            "contents",
            "sommaire",
            "índice",
            "inhaltsverzeichnis",
        ]

        # Get heading context if available
        doc_chunk = DocChunk.model_validate(chunk)
        headings = doc_chunk.meta.headings or []
        heading_text = " ".join(headings).lower()

        for toc_heading in toc_headings:
            if toc_heading in heading_text or toc_heading in text[:100]:
                # Additional check: TOC entries often have separator patterns
                # Like dots (....) or dashes (---) or pipe tables
                separator_count = (
                    text.count('....') +
                    text.count('----') +
                    text.count('|---')
                )

                # If has TOC heading and separator patterns, it's likely TOC
                if separator_count > 0:
                    return True

                # Also check for page number patterns at end of lines
                page_number_pattern = r'\d+\s*$|\d+\s*\|'
                lines = text.split('\n')
                page_number_lines = sum(
                    1 for line in lines
                    if re.search(page_number_pattern, line.strip())
                )

                # If most lines end with numbers, likely TOC
                if len(lines) > 1 and page_number_lines > len(lines) * 0.5:
                    return True

        return False

    def chunk_from_markdown_file(
        self,
        md_file_path: str,
        contextualize: bool = True,
    ) -> List[dict[str, Any]]:
        """
        Chunk content directly from a markdown file using Docling.

        This method uses the Docling DocumentConverter to convert the markdown file
        and then chunks it using the HybridChunker. It provides a convenient way
        to process markdown files without manual conversion.

        Args:
            md_file_path: Path to the markdown file to chunk
            contextualize: If True, applies contextualization to add hierarchical 
                         context from headings (default: True)

        Returns:
            List of dictionaries with chunk information:
                - text: The chunk text (contextualized if enabled)
                - num_tokens: Number of tokens in the chunk
                - doc_items: List of document item references
                - chunk: The BaseChunk object

        Raises:
            ImportError: If docling is not installed
            FileNotFoundError: If the markdown file doesn't exist

        Example:
            ```python
            from advanced_chunker import AdvancedChunker

            # Create chunker
            chunker = AdvancedChunker(strategy="markdown_tables")

            # Chunk from markdown file
            chunks = chunker.chunk_from_markdown_file(
                md_file_path="/path/to/document.md",
                contextualize=True
            )

            # Access chunk information
            for chunk_info in chunks:
                print(f"Text: {chunk_info['text']}")
                print(f"Tokens: {chunk_info['num_tokens']}")
            ```
        """
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            raise ImportError(
                "docling is not installed. Please install it with: "
                "pip install docling"
            )

        import os
        if not os.path.exists(md_file_path):
            raise FileNotFoundError(f"Markdown file not found: {md_file_path}")

        # Convert markdown file to DoclingDocument
        converter = DocumentConverter()
        result = converter.convert(source=md_file_path)
        dl_doc = result.document

        # Chunk the document
        chunks_list = []
        for chunk in self.hybrid_chunker.chunk(dl_doc=dl_doc):
            # Get contextualized text if requested
            if contextualize:
                chunk_text = self.contextualize(chunk=chunk)
            else:
                chunk_text = chunk.text

            # Get chunk information
            num_tokens = self.count_tokens(text=chunk_text)
            doc_chunk = DocChunk.model_validate(chunk)
            doc_items_refs = [it.self_ref for it in doc_chunk.meta.doc_items]

            chunk_info = {
                "text": chunk_text,
                "num_tokens": num_tokens,
                "doc_items": doc_items_refs,
                "chunk": chunk,
            }
            chunks_list.append(chunk_info)

        return chunks_list

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return self.tokenizer.count_tokens(text=text)

    def get_max_tokens(self) -> int:
        """
        Get maximum token limit for the tokenizer.

        Returns:
            Maximum number of tokens
        """
        return self.tokenizer.get_max_tokens()

    def contextualize(self, chunk: BaseChunk) -> str:
        """
        Contextualize a chunk by adding hierarchical context from headings.

        This method enriches the chunk text with context from parent headings
        and section titles, which improves RAG retrieval quality by providing
        more semantic context.

        If `include_heading_markers` is True, headings will be prefixed with
        markdown-style `#` markers based on their hierarchy level.

        Args:
            chunk: The chunk to contextualize

        Returns:
            Context-enriched text string

        Example:
            >>> for chunk in chunker.chunk(dl_doc=doc):
            ...     enriched_text = chunker.contextualize(chunk=chunk)
            ...     # Use enriched_text for embedding
        """
        if not self.include_heading_markers:
            return self.hybrid_chunker.contextualize(chunk=chunk)

        # Custom contextualization with markdown heading markers
        doc_chunk = DocChunk.model_validate(chunk)
        meta = doc_chunk.meta

        items = []

        # Add headings with markdown markers
        if meta.headings:
            for i, heading in enumerate(meta.headings):
                # Level starts at 1 for first heading, increases for nested
                level = i + 1
                items.append(f"{'#' * level} {heading}")

        # Add the chunk text
        items.append(chunk.text)

        return self.hybrid_chunker.delim.join(items)

    @staticmethod
    def find_nth_chunk_with_label(
        chunks: Iterable[BaseChunk],
        n: int,
        label: DocItemLabel,
    ) -> tuple[Optional[int], Optional[DocChunk]]:
        """
        Find the n-th chunk containing a specific document item label.

        Args:
            chunks: Iterable of chunks to search
            n: Zero-based index of the chunk to find
            label: Document item label to search for

        Returns:
            Tuple of (chunk_index, chunk) or (None, None) if not found
        """
        num_found = -1
        for i, chunk in enumerate(chunks):
            doc_chunk = DocChunk.model_validate(chunk)
            for it in doc_chunk.meta.doc_items:
                if it.label == label:
                    num_found += 1
                    if num_found == n:
                        return i, doc_chunk
        return None, None

    def get_chunk_info(self, chunk: BaseChunk) -> dict[str, Any]:
        """
        Get detailed information about a chunk.

        Args:
            chunk: Chunk to analyze

        Returns:
            Dictionary with chunk information including:
                - text: Contextualized text
                - num_tokens: Token count
                - doc_items: List of document item references
        """
        ctx_text = self.contextualize(chunk=chunk)
        num_tokens = self.count_tokens(text=ctx_text)
        doc_chunk = DocChunk.model_validate(chunk)
        doc_items_refs = [it.self_ref for it in doc_chunk.meta.doc_items]

        return {
            "text": ctx_text,
            "num_tokens": num_tokens,
            "doc_items": doc_items_refs,
            "chunk": doc_chunk,
        }


# Convenience function for quick setup
def create_chunker(
    strategy: str = "default",
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    include_heading_markers: bool = True,
    max_tokens: int = AdvancedChunker.DEFAULT_MAX_TOKENS,
    merge_peers: bool = AdvancedChunker.DEFAULT_MERGE_PEERS,
    min_chunk_tokens: int = AdvancedChunker.DEFAULT_MIN_CHUNK_TOKENS,
    filter_toc: bool = True,
    **kwargs,
) -> AdvancedChunker:
    """
    Create a pre-configured advanced chunker.

    Args:
        strategy: Chunking strategy to use:
            - "default": Default serialization
            - "markdown_tables": Markdown table formatting
            - "custom_placeholder": Custom image placeholder
            - "annotations": Include picture annotations
        embed_model_id: HuggingFace model ID for tokenization
        include_heading_markers: If True, adds markdown # markers to headings
                                 in contextualized output (default: True)
        max_tokens: Maximum tokens per chunk (default: 1024)
        merge_peers: If True, merges adjacent small chunks (default: True)
        min_chunk_tokens: Minimum tokens for standalone chunks (default: 50)
        filter_toc: If True, filters out Table of Contents entries (default: True)
        **kwargs: Additional arguments passed to strategy-specific providers

    Returns:
        Configured AdvancedChunker instance

    Example:
        ```python
        # Create chunker with markdown tables
        chunker = create_chunker(strategy="markdown_tables")

        # Create chunker with custom image placeholder
        chunker = create_chunker(
            strategy="custom_placeholder",
            image_placeholder="[IMAGE]"
        )

        # Create chunker with larger chunks and TOC filtering
        chunker = create_chunker(
            strategy="markdown_tables",
            max_tokens=2048,
            filter_toc=True
        )
        ```
    """
    provider_map = {
        "default": DefaultSerializerProvider,
        "markdown_tables": MDTableSerializerProvider,
        "custom_placeholder": ImgPlaceholderSerializerProvider,
        "annotations": ImgAnnotationSerializerProvider,
    }

    if strategy not in provider_map:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Available: {list(provider_map.keys())}"
        )

    provider_class = provider_map[strategy]

    # Filter kwargs for provider initialization
    import inspect
    provider_sig = inspect.signature(provider_class.__init__)
    provider_params = set(provider_sig.parameters.keys()) - {"self"}
    provider_kwargs = {k: v for k, v in kwargs.items() if k in provider_params}

    provider = provider_class(**provider_kwargs)

    return AdvancedChunker(
        embed_model_id=embed_model_id,
        serializer_provider=provider,
        include_heading_markers=include_heading_markers,
        max_tokens=max_tokens,
        merge_peers=merge_peers,
        min_chunk_tokens=min_chunk_tokens,
        filter_toc=filter_toc,
    )
