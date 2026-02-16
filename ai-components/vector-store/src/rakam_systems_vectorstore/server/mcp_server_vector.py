"""
Vector Store MCP Server

This module provides an MCP (Model Context Protocol) server for vector store operations.
It creates a unified interface for vector search, storage, and management operations
that can be easily integrated with AI agents.

Features:
- Vector search with semantic similarity
- Document storage and indexing
- Collection management and info
- Async/await support for concurrent operations

Example:
    >>> from rakam_systems_vectorstore.server.mcp_server_vector import run_vector_mcp
    >>> from rakam_systems_vectorstore.components.vectorstore.configurable_pg_vector_store import ConfigurablePgVectorStore
    >>> 
    >>> # Initialize vector store
    >>> vector_store = ConfigurablePgVectorStore(name="my_store")
    >>> vector_store.setup()
    >>> 
    >>> # Create MCP server with vector store tools
    >>> mcp_server = run_vector_mcp(vector_store)
    >>> 
    >>> # Use the server to route messages
    >>> result = await mcp_server.asend_message(
    ...     sender="agent",
    ...     receiver="vector_search",
    ...     message={'arguments': {'query': 'test', 'top_k': 5}}
    ... )
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

from rakam_systems_tools.utils import logging
from rakam_systems_core.mcp.mcp_server import MCPServer
from rakam_systems_core.interfaces import ToolComponent

logger = logging.getLogger(__name__)


class VectorSearchTool(ToolComponent):
    """Tool component for performing vector search operations."""

    def __init__(self, name: str, vector_store, config: Optional[Dict] = None):
        """
        Initialize vector search tool.

        Args:
            name: Tool name
            vector_store: ConfigurablePgVectorStore instance
            config: Optional configuration dictionary
        """
        super().__init__(name, config or {})
        self.vector_store = vector_store

    async def run(
        self,
        query: str,
        collection_name: str = "documents",
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search the vector store for relevant documents.

        Args:
            query: Search query text
            collection_name: Name of the collection to search
            top_k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            Dictionary with search results containing:
                - query: Original query
                - collection: Collection name
                - results_count: Number of results
                - results: List of result documents with content and metadata
        """
        from asgiref.sync import sync_to_async

        try:
            # Perform search using the vector store (wrap in sync_to_async)
            results, result_nodes = await sync_to_async(self.vector_store.search)(
                collection_name=collection_name,
                query=query,
                number=top_k,
                **kwargs
            )

            # Format results
            formatted_results = []
            for node in result_nodes:
                formatted_results.append({
                    'content': node.content,
                    'node_id': node.metadata.node_id,
                    'source_file': node.metadata.source_file_uuid,
                    'position': node.metadata.position,
                    'metadata': node.metadata.custom or {}
                })

            return {
                'query': query,
                'collection': collection_name,
                'results_count': len(formatted_results),
                'results': formatted_results
            }
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'query': query,
                'collection': collection_name
            }


class VectorStorageTool(ToolComponent):
    """Tool component for adding documents to vector store."""

    def __init__(self, name: str, vector_store, config: Optional[Dict] = None):
        """
        Initialize vector storage tool.

        Args:
            name: Tool name
            vector_store: ConfigurablePgVectorStore instance
            config: Optional configuration dictionary
        """
        super().__init__(name, config or {})
        self.vector_store = vector_store

    async def run(
        self,
        documents: List[str],
        collection_name: str = "documents",
        doc_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts to add
            collection_name: Target collection name
            doc_metadata: Optional metadata to attach to documents

        Returns:
            Dictionary with operation results containing:
                - success: Boolean indicating success
                - collection: Collection name
                - documents_added: Number of documents added
                - node_ids: List of created node IDs
        """
        from rakam_systems_vectorstore.core import Node, NodeMetadata
        from asgiref.sync import sync_to_async

        try:
            # Create nodes from documents
            nodes = []
            for idx, doc_text in enumerate(documents):
                node_metadata = NodeMetadata(
                    source_file_uuid="mcp_upload",
                    position=idx,
                    custom=doc_metadata or {}
                )
                node = Node(content=doc_text, metadata=node_metadata)
                nodes.append(node)

            # Add to vector store (wrap in sync_to_async)
            await sync_to_async(self.vector_store.create_collection_from_nodes)(
                collection_name, nodes
            )

            return {
                'success': True,
                'collection': collection_name,
                'documents_added': len(documents),
                'node_ids': [node.metadata.node_id for node in nodes]
            }
        except Exception as e:
            logger.error(f"Vector storage failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }


class VectorInfoTool(ToolComponent):
    """Tool component for getting vector store information."""

    def __init__(self, name: str, vector_store, config: Optional[Dict] = None):
        """
        Initialize vector info tool.

        Args:
            name: Tool name
            vector_store: ConfigurablePgVectorStore instance
            config: Optional configuration dictionary
        """
        super().__init__(name, config or {})
        self.vector_store = vector_store

    async def run(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about vector store collections.

        Args:
            collection_name: Specific collection name (optional)

        Returns:
            Dictionary with collection information:
                - If collection_name provided: info about that collection
                - If not provided: list of all collections
        """
        from asgiref.sync import sync_to_async

        try:
            if collection_name:
                # Get info for specific collection
                info = await sync_to_async(self.vector_store.get_collection_info)(
                    collection_name
                )
                return {
                    'collection_name': collection_name,
                    'node_count': info.get('node_count', 0),
                    'embedding_dim': info.get('embedding_dim', 0),
                }
            else:
                # List all collections
                collections = await sync_to_async(self.vector_store.list_collections)()
                return {
                    'total_collections': len(collections),
                    'collections': collections
                }
        except Exception as e:
            logger.error(
                f"Vector info retrieval failed: {str(e)}", exc_info=True)
            return {
                'error': str(e)
            }


def run_vector_mcp(
    vector_store,
    name: str = "vector_store_mcp",
    enable_logging: bool = False
) -> MCPServer:
    """
    Create and configure an MCP server with vector store tools.

    This function creates a fully configured MCP server with three main tools:
    - vector_search: Search documents using semantic similarity
    - vector_storage: Add documents to the vector store
    - vector_info: Get information about collections

    Args:
        vector_store: ConfigurablePgVectorStore instance (must be set up)
        name: Name for the MCP server (default: "vector_store_mcp")
        enable_logging: Whether to enable detailed MCP logging (default: False)

    Returns:
        MCPServer instance with registered vector store tools

    Example:
        >>> from rakam_systems_vectorstore.config import VectorStoreConfig, EmbeddingConfig
        >>> from rakam_systems_vectorstore.components.vectorstore.configurable_pg_vector_store import ConfigurablePgVectorStore
        >>> 
        >>> # Initialize vector store
        >>> config = VectorStoreConfig(
        ...     name="my_store",
        ...     embedding=EmbeddingConfig(
        ...         model_type="sentence_transformer",
        ...         model_name="Snowflake/snowflake-arctic-embed-m"
        ...     )
        ... )
        >>> vector_store = ConfigurablePgVectorStore(name="my_store", config=config)
        >>> vector_store.setup()
        >>> 
        >>> # Create MCP server
        >>> mcp_server = run_vector_mcp(vector_store)
        >>> 
        >>> # List registered tools
        >>> print(mcp_server.list_components())
        >>> # Output: ['vector_info', 'vector_search', 'vector_storage']
        >>> 
        >>> # Use the server
        >>> result = await mcp_server.asend_message(
        ...     sender="client",
        ...     receiver="vector_search",
        ...     message={'arguments': {'query': 'machine learning', 'top_k': 3}}
        ... )
    """
    logger.info(f"Creating vector store MCP server: {name}")

    # Create MCP server
    server = MCPServer(name=name, enable_logging=enable_logging)
    server.setup()

    # Create and register tool components
    search_tool = VectorSearchTool(
        name="vector_search",
        vector_store=vector_store
    )

    storage_tool = VectorStorageTool(
        name="vector_storage",
        vector_store=vector_store
    )

    info_tool = VectorInfoTool(
        name="vector_info",
        vector_store=vector_store
    )

    # Register all tools
    server.register_component(search_tool)
    server.register_component(storage_tool)
    server.register_component(info_tool)

    logger.info(
        f"Vector MCP server '{name}' ready with {len(server)} tool(s): "
        f"{server.list_components()}"
    )

    return server
