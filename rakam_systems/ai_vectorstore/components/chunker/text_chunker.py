"""
Text chunking utilities for splitting text into smaller pieces with overlap.
"""

from __future__ import annotations

import logging
from typing import List

from ai_core.interfaces.chunker import Chunker

logger = logging.getLogger(__name__)


class TextChunker(Chunker):
    """
    Text chunker that splits text into smaller pieces with overlap.
    
    This chunker uses character-based chunking with word boundary detection
    to ensure chunks break at natural points.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        name: str = "text_chunker"
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            name: Component name
        """
        super().__init__(name=name)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
    
    def run(self, documents: List[str]) -> List[str]:
        """
        Chunk a list of documents into smaller text pieces.
        
        Args:
            documents: List of text documents to chunk
            
        Returns:
            List of text chunks
        """
        all_chunks = []
        for doc_idx, document in enumerate(documents):
            chunks = self.chunk_text(document, context=f"doc_{doc_idx}")
            all_chunks.extend(chunks)
        return all_chunks
    
    def chunk_text(self, text: str, context: str = "") -> List[str]:
        """
        Chunk text into smaller pieces with overlap.
        
        Args:
            text: Text to chunk
            context: Context label for logging
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Simple character-based chunking with overlap
        chunks = []
        text_len = len(text)
        
        if text_len <= self._chunk_size:
            return [text]
        
        start = 0
        while start < text_len:
            end = start + self._chunk_size
            
            # Try to break at word boundary
            if end < text_len:
                # Look for space or newline
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in ' \n\t':
                        end = i
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self._chunk_overlap
            if start < 0:
                start = end
        
        logger.debug(f"Chunked {context}: {len(text)} chars -> {len(chunks)} chunks")
        return chunks


def create_text_chunker(
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> TextChunker:
    """
    Factory function to create a text chunker.
    
    Args:
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        Configured text chunker
    """
    return TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


__all__ = ["TextChunker", "create_text_chunker"]

