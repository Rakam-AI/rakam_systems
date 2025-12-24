from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ..base import BaseComponent

if TYPE_CHECKING:
    from rakam_systems.vectorestore import Node, VSFile


class Loader(BaseComponent, ABC):
    """
    Abstract base class for document loaders.
    
    This class provides a common interface for loading documents into different formats.
    Subclasses must implement the load_as_text, load_as_chunks, load_as_nodes, and load_as_vsfile methods.
    """
    
    @abstractmethod
    def run(self, source: str) -> List[str]:
        """Load raw documents from a source (path, URL, id, etc.)."""
        raise NotImplementedError
    
    @abstractmethod
    def load_as_text(self, source: Union[str, Path]) -> str:
        """
        Load document and return as a single text string.
        
        Args:
            source: Path to document file
            
        Returns:
            Full text content of the document as a single string
        """
        raise NotImplementedError
    
    @abstractmethod
    def load_as_chunks(self, source: Union[str, Path]) -> List[str]:
        """
        Load document and return as a list of text chunks.
        
        Args:
            source: Path to document file
            
        Returns:
            List of text chunks extracted from the document
        """
        raise NotImplementedError
    
    @abstractmethod
    def load_as_nodes(
        self,
        source: Union[str, Path],
        source_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List["Node"]:
        """
        Load document and return as Node objects with metadata.
        
        Args:
            source: Path to document file
            source_id: Optional source identifier (defaults to file path)
            custom_metadata: Optional custom metadata to attach to nodes
            
        Returns:
            List of Node objects with text chunks and metadata
        """
        raise NotImplementedError
    
    @abstractmethod
    def load_as_vsfile(
        self,
        file_path: Union[str, Path],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> "VSFile":
        """
        Load document and return as VSFile object.
        
        Args:
            file_path: Path to document file
            custom_metadata: Optional custom metadata
            
        Returns:
            VSFile object with nodes
        """
        raise NotImplementedError
