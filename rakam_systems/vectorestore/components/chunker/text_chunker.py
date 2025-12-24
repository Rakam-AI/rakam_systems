"""
Text chunking utilities for splitting text into smaller pieces with overlap.
"""

from __future__ import annotations

from typing import Any, List

from rakam_systems.core.ai_utils import logging
from rakam_systems.core.ai_core.interfaces.chunker import Chunker

try:
    from chonkie import SentenceChunker
    CHONKIE_AVAILABLE = True
except ImportError:
    CHONKIE_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextChunker(Chunker):
    """
    Text chunker that splits text into smaller pieces with overlap.

    This chunker uses Chonkie's SentenceChunker for sentence-based chunking
    with token-aware splitting and configurable overlap.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_sentences_per_chunk: int = 1,
        tokenizer: str = "character",
        name: str = "text_chunker"
    ):
        """
        Initialize text chunker.

        Args:
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            min_sentences_per_chunk: Minimum sentences per chunk (default: 1)
            tokenizer: Tokenizer to use - "character", "gpt2", or any HuggingFace tokenizer (default: "character")
            name: Component name
        """
        super().__init__(name=name)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_sentences_per_chunk = min_sentences_per_chunk
        self._tokenizer = tokenizer

    def run(self, documents: List[str]) -> List[str]:
        """
        Chunk a list of documents into smaller text pieces.

        Args:
            documents: List of text documents to chunk

        Returns:
            List of text chunks (just the text content)
        """
        all_chunks = []
        for doc_idx, document in enumerate(documents):
            chunk_results = self.chunk_text(document, context=f"doc_{doc_idx}")
            # Extract just the text from the chunk dictionaries
            chunks = [chunk_info["text"] for chunk_info in chunk_results]
            all_chunks.extend(chunks)
        return all_chunks

    def chunk_text(self, text: str, context: str = "") -> List[dict[str, Any]]:
        """
        Chunk text into smaller pieces with overlap using Chonkie's SentenceChunker.

        This method uses sentence-based chunking with configurable token limits and overlap,
        providing more intelligent chunking than simple character-based splitting.

        Args:
            text: Text to chunk
            context: Context label for logging (optional)

        Returns:
            List of dictionaries with chunk information:
                - text: The chunk text
                - token_count: Number of tokens in the chunk
                - start_index: Starting character index in original text
                - end_index: Ending character index in original text

        Raises:
            ImportError: If chonkie is not installed
        """
        if not text or not text.strip():
            return []

        if not CHONKIE_AVAILABLE:
            raise ImportError(
                "chonkie is not installed. Please install it with: "
                "pip install chonkie==1.4.2"
            )

        # Initialize the Chonkie SentenceChunker
        chonkie_chunker = SentenceChunker(
            tokenizer=self._tokenizer,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            min_sentences_per_chunk=self._min_sentences_per_chunk,
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

        logger.debug(
            f"Chunked {context}: {len(text)} chars -> {len(result)} chunks")
        return result


def create_text_chunker(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_sentences_per_chunk: int = 1,
    tokenizer: str = "character"
) -> TextChunker:
    """
    Factory function to create a text chunker.

    Args:
        chunk_size: Size of text chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        min_sentences_per_chunk: Minimum sentences per chunk (default: 1)
        tokenizer: Tokenizer to use - "character", "gpt2", or any HuggingFace tokenizer (default: "character")

    Returns:
        Configured text chunker
    """
    return TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_sentences_per_chunk=min_sentences_per_chunk,
        tokenizer=tokenizer
    )


__all__ = ["TextChunker", "create_text_chunker"]
