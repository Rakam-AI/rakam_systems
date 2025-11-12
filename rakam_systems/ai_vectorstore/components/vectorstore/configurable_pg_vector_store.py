"""
Configurable PostgreSQL Vector Store with enhanced features.

This module provides an enhanced, fully configurable PgVectorStore that:
- Supports configuration via YAML/JSON files or dictionaries
- Allows pluggable embedding models
- Provides update_vector capability
- Maintains clean separation from other components
- Supports all search configurations
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.db import connection, transaction

from rakam_systems.ai_core.interfaces.vectorstore import VectorStore
from rakam_systems.ai_vectorstore.components.embedding_model.configurable_embeddings import ConfigurableEmbeddings
from rakam_systems.ai_vectorstore.components.vectorstore.pg_models import Collection, NodeEntry
from rakam_systems.ai_vectorstore.config import VectorStoreConfig, load_config
from rakam_systems.ai_vectorstore.core import Node, NodeMetadata, VSFile

logger = logging.getLogger(__name__)


class ConfigurablePgVectorStore(VectorStore):
    """
    Enhanced PostgreSQL Vector Store with full configuration support.
    
    Features:
    - Configuration via YAML/JSON or dict
    - Pluggable embedding models
    - Configurable similarity metrics
    - Hybrid search with configurable weights
    - Update operations for vectors
    - Comprehensive metadata filtering
    """
    
    def __init__(
        self,
        name: str = "configurable_pg_vector_store",
        config: Optional[Union[VectorStoreConfig, Dict, str]] = None
    ):
        """
        Initialize configurable PostgreSQL vector store.
        
        Args:
            name: Component name
            config: Configuration (VectorStoreConfig object, dict, or path to config file)
        """
        # Load configuration
        if isinstance(config, VectorStoreConfig):
            self.vs_config = config
        elif isinstance(config, dict):
            self.vs_config = VectorStoreConfig.from_dict(config)
        elif isinstance(config, str):
            # Path to config file
            self.vs_config = load_config(config)
        else:
            # Use defaults
            self.vs_config = VectorStoreConfig()
        
        # Validate configuration
        self.vs_config.validate()
        
        # Initialize base component
        super().__init__(name=name, config=self.vs_config.to_dict())
        
        # Setup logging
        if self.vs_config.enable_logging:
            logging.basicConfig(level=self.vs_config.log_level)
        
        # Initialize embedding model
        self.embedding_model = ConfigurableEmbeddings(
            name=f"{name}_embeddings",
            config=self.vs_config.embedding
        )
        
        self.embedding_dim: Optional[int] = None
        
        logger.info(f"Initialized {name} with config: {self.vs_config.name}")
    
    def setup(self) -> None:
        """Initialize resources and connections."""
        logger.info("Setting up ConfigurablePgVectorStore...")
        
        # Ensure pgvector extension
        self._ensure_pgvector_extension()
        
        # Setup embedding model
        self.embedding_model.setup()
        self.embedding_dim = self.embedding_model.embedding_dimension
        
        logger.info(f"Vector store ready with embedding dimension: {self.embedding_dim}")
        super().setup()
    
    def _ensure_pgvector_extension(self) -> None:
        """Ensure pgvector extension is installed."""
        with connection.cursor() as cursor:
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("Ensured pgvector extension is installed")
            except Exception as e:
                logger.error(f"Failed to create pgvector extension: {e}")
                raise
    
    def get_or_create_collection(
        self,
        collection_name: str,
        embedding_dim: Optional[int] = None
    ) -> Collection:
        """
        Get or create a collection.
        
        Args:
            collection_name: Name of the collection
            embedding_dim: Embedding dimension (uses model dimension if not specified)
            
        Returns:
            Collection object
        """
        if embedding_dim is None:
            embedding_dim = self.embedding_dim
        
        collection, created = Collection.objects.get_or_create(
            name=collection_name,
            defaults={"embedding_dim": embedding_dim}
        )
        
        logger.info(
            f"{'Created new' if created else 'Using existing'} collection: {collection_name}"
        )
        return collection
    
    def _get_distance_operator(self, distance_type: Optional[str] = None) -> str:
        """Get SQL distance operator for the configured similarity metric."""
        if distance_type is None:
            distance_type = self.vs_config.search.similarity_metric
        
        operators = {
            "cosine": "<=>",
            "l2": "<->",
            "dot_product": "<#>",
            "dot": "<#>"
        }
        
        if distance_type not in operators:
            raise ValueError(f"Unsupported distance type: {distance_type}")
        
        return operators[distance_type]
    
    @lru_cache(maxsize=1000)
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a query (with caching)."""
        if not self.vs_config.enable_caching:
            # Don't use cache
            return np.array(self.embedding_model.encode_query(query), dtype=np.float32)
        
        embedding = self.embedding_model.encode_query(query)
        return np.array(embedding, dtype=np.float32)
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    @transaction.atomic
    def create_collection_from_nodes(
        self,
        collection_name: str,
        nodes: List[Node]
    ) -> None:
        """
        Create a collection from nodes.
        
        Args:
            collection_name: Name of collection
            nodes: List of Node objects
        """
        if not nodes:
            logger.warning(f"No nodes provided for collection '{collection_name}'")
            return
        
        logger.info(f"Creating collection '{collection_name}' with {len(nodes)} nodes")
        
        # Get or create collection
        collection = self.get_or_create_collection(collection_name)
        
        # Clear existing nodes
        NodeEntry.objects.filter(collection=collection).delete()
        
        # Generate embeddings
        texts = [node.content for node in nodes]
        embeddings = self.embedding_model.encode_documents(texts)
        
        # Create node entries
        node_entries = [
            NodeEntry(
                collection=collection,
                content=node.content,
                embedding=embeddings[i],
                source_file_uuid=node.metadata.source_file_uuid,
                position=node.metadata.position,
                custom_metadata=node.metadata.custom or {},
            )
            for i, node in enumerate(nodes)
        ]
        
        # Bulk insert
        created_entries = NodeEntry.objects.bulk_create(
            node_entries,
            batch_size=self.vs_config.index.batch_insert_size
        )
        
        # Update node IDs
        for i, node in enumerate(nodes):
            node.metadata.node_id = created_entries[i].node_id
        
        logger.info(f"Created collection '{collection_name}' with {len(created_entries)} nodes")
    
    @transaction.atomic
    def create_collection_from_files(
        self,
        collection_name: str,
        files: List[VSFile]
    ) -> None:
        """
        Create collection from VSFile objects.
        
        Args:
            collection_name: Name of collection
            files: List of VSFile objects
        """
        nodes = [node for file in files for node in file.nodes]
        self.create_collection_from_nodes(collection_name, nodes)
    
    @transaction.atomic
    def add_nodes(self, collection_name: str, nodes: List[Node]) -> None:
        """
        Add nodes to existing collection.
        
        Args:
            collection_name: Name of collection
            nodes: Nodes to add
        """
        if not nodes:
            logger.warning("No nodes to add")
            return
        
        logger.info(f"Adding {len(nodes)} nodes to collection '{collection_name}'")
        
        try:
            collection = Collection.objects.get(name=collection_name)
        except Collection.DoesNotExist:
            raise ValueError(f"Collection not found: {collection_name}")
        
        # Generate embeddings
        texts = [node.content for node in nodes]
        embeddings = self.embedding_model.encode_documents(texts)
        
        # Create entries
        node_entries = [
            NodeEntry(
                collection=collection,
                content=node.content,
                embedding=embeddings[i],
                source_file_uuid=node.metadata.source_file_uuid,
                position=node.metadata.position,
                custom_metadata=node.metadata.custom or {},
            )
            for i, node in enumerate(nodes)
        ]
        
        created_entries = NodeEntry.objects.bulk_create(
            node_entries,
            batch_size=self.vs_config.index.batch_insert_size
        )
        
        # Update node IDs
        for i, node in enumerate(nodes):
            node.metadata.node_id = created_entries[i].node_id
        
        logger.info(f"Added {len(created_entries)} nodes to '{collection_name}'")
    
    @transaction.atomic
    def update_vector(
        self,
        collection_name: str,
        node_id: int,
        new_content: Optional[str] = None,
        new_embedding: Optional[List[float]] = None,
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update a vector in the collection.
        
        Args:
            collection_name: Name of collection
            node_id: ID of node to update
            new_content: New content (will regenerate embedding if provided)
            new_embedding: New embedding vector (used if new_content not provided)
            new_metadata: New metadata to merge with existing
        """
        try:
            collection = Collection.objects.get(name=collection_name)
        except Collection.DoesNotExist:
            raise ValueError(f"Collection not found: {collection_name}")
        
        try:
            node_entry = NodeEntry.objects.get(collection=collection, node_id=node_id)
        except NodeEntry.DoesNotExist:
            raise ValueError(f"Node {node_id} not found in collection '{collection_name}'")
        
        # Update content and embedding
        if new_content is not None:
            node_entry.content = new_content
            # Generate new embedding
            embedding = self.embedding_model.encode_query(new_content)
            node_entry.embedding = embedding
            logger.info(f"Updated content and regenerated embedding for node {node_id}")
        elif new_embedding is not None:
            node_entry.embedding = new_embedding
            logger.info(f"Updated embedding for node {node_id}")
        
        # Update metadata
        if new_metadata is not None:
            # Merge with existing metadata
            current_metadata = node_entry.custom_metadata or {}
            current_metadata.update(new_metadata)
            node_entry.custom_metadata = current_metadata
            logger.info(f"Updated metadata for node {node_id}")
        
        node_entry.save()
        logger.info(f"Successfully updated node {node_id} in collection '{collection_name}'")
    
    @transaction.atomic
    def delete_nodes(self, collection_name: str, node_ids: List[int]) -> None:
        """
        Delete nodes from collection.
        
        Args:
            collection_name: Name of collection
            node_ids: List of node IDs to delete
        """
        if not node_ids:
            logger.warning("No node IDs to delete")
            return
        
        try:
            collection = Collection.objects.get(name=collection_name)
        except Collection.DoesNotExist:
            raise ValueError(f"Collection not found: {collection_name}")
        
        deleted_count, _ = NodeEntry.objects.filter(
            collection=collection,
            node_id__in=node_ids
        ).delete()
        
        logger.info(f"Deleted {deleted_count} nodes from collection '{collection_name}'")
    
    @transaction.atomic
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection and all its nodes.
        
        Args:
            collection_name: Name of collection to delete
        """
        try:
            collection = Collection.objects.get(name=collection_name)
            node_count = NodeEntry.objects.filter(collection=collection).count()
            collection.delete()
            logger.info(f"Deleted collection '{collection_name}' with {node_count} nodes")
        except Collection.DoesNotExist:
            raise ValueError(f"Collection not found: {collection_name}")
    
    def search(
        self,
        collection_name: str,
        query: str,
        distance_type: Optional[str] = None,
        number: Optional[int] = None,
        meta_data_filters: Optional[Dict[str, Any]] = None,
        hybrid_search: Optional[bool] = None
    ) -> Tuple[Dict, List[Node]]:
        """
        Search for similar vectors in collection.
        
        Args:
            collection_name: Name of collection to search
            query: Search query
            distance_type: Distance metric (uses config default if None)
            number: Number of results (uses config default if None)
            meta_data_filters: Metadata filters
            hybrid_search: Enable hybrid search (uses config default if None)
            
        Returns:
            Tuple of (results dict, list of Node objects)
        """
        # Use config defaults
        if distance_type is None:
            distance_type = self.vs_config.search.similarity_metric
        if number is None:
            number = self.vs_config.search.default_top_k
        if hybrid_search is None:
            hybrid_search = self.vs_config.search.enable_hybrid_search
        
        logger.info(f"Searching in '{collection_name}' for: '{query}'")
        
        try:
            collection = Collection.objects.get(name=collection_name)
        except Collection.DoesNotExist:
            logger.error(f"Collection not found: {collection_name}")
            raise ValueError(f"Collection not found: {collection_name}")
        
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        
        # Normalize if using cosine distance
        if distance_type == "cosine":
            query_embedding = self._normalize_embedding(query_embedding)
        
        # Build queryset
        queryset = NodeEntry.objects.filter(collection=collection)
        
        # Apply metadata filters
        if meta_data_filters:
            for key, value in meta_data_filters.items():
                queryset = queryset.filter(**{f"custom_metadata__{key}": value})
        
        # Determine search buffer
        search_buffer_factor = (
            self.vs_config.search.search_buffer_factor if hybrid_search else 1
        )
        limit = number * search_buffer_factor
        
        # Build SQL query
        distance_operator = self._get_distance_operator(distance_type)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        
        sql_query = f"""
            SELECT
                node_id,
                content,
                source_file_uuid,
                position,
                custom_metadata,
                embedding {distance_operator} %s::vector AS distance
            FROM
                {NodeEntry._meta.db_table}
            WHERE
                collection_id = %s
            ORDER BY
                distance
            LIMIT
                %s
        """
        
        # Execute query
        with connection.cursor() as cursor:
            cursor.execute(sql_query, [embedding_str, collection.id, limit])
            results = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
        
        # Process results
        valid_suggestions = {}
        suggested_nodes = []
        seen_texts = set()
        
        for row in results:
            result_dict = dict(zip(columns, row))
            node_id = result_dict["node_id"]
            content = result_dict["content"]
            distance = result_dict["distance"]
            
            if content not in seen_texts:
                seen_texts.add(content)
                
                custom_metadata = result_dict["custom_metadata"] or {}
                
                metadata = NodeMetadata(
                    source_file_uuid=result_dict["source_file_uuid"],
                    position=result_dict["position"],
                    custom=custom_metadata,
                )
                metadata.node_id = node_id
                
                node = Node(content=content, metadata=metadata)
                suggested_nodes.append(node)
                
                valid_suggestions[str(node_id)] = (
                    {
                        "node_id": node_id,
                        "source_file_uuid": result_dict["source_file_uuid"],
                        "position": result_dict["position"],
                        "custom": custom_metadata,
                    },
                    content,
                    float(distance),
                )
        
        # Apply hybrid search and re-ranking if enabled
        if hybrid_search and self.vs_config.search.rerank:
            valid_suggestions, suggested_nodes = self._rerank_results(
                query, list(valid_suggestions.values()), suggested_nodes, number
            )
        
        logger.info(f"Search returned {len(valid_suggestions)} results")
        return valid_suggestions, suggested_nodes
    
    def _rerank_results(
        self,
        query: str,
        results: List[Tuple[Dict, str, float]],
        suggested_nodes: List[Node],
        top_k: int,
    ) -> Tuple[Dict, List[Node]]:
        """Re-rank results using hybrid scoring."""
        logger.debug(f"Re-ranking {len(results)} results")
        
        # Get hybrid alpha from config
        alpha = self.vs_config.search.hybrid_alpha
        
        # Perform full-text search
        search_query = SearchQuery(query, config="english")
        node_ids = [int(res[0]["node_id"]) for res in results]
        
        queryset = NodeEntry.objects.filter(
            node_id__in=node_ids
        ).annotate(
            rank=SearchRank(SearchVector("content", config="english"), search_query)
        )
        
        # Combine scores
        reranked_results = []
        node_id_to_rank = {node.node_id: node.rank for node in queryset}
        
        for metadata, content, distance in results:
            node_id = metadata["node_id"]
            keyword_score = node_id_to_rank.get(node_id, 0.0)
            
            # Combined score: alpha * vector + (1-alpha) * keyword
            combined_score = alpha * (1 - distance) + (1 - alpha) * keyword_score
            reranked_results.append((metadata, content, combined_score))
        
        # Sort and take top_k
        reranked_results = sorted(reranked_results, key=lambda x: x[2], reverse=True)[:top_k]
        valid_suggestions = {str(res[0]["node_id"]): res for res in reranked_results}
        
        # Update node order
        node_id_order = [res[0]["node_id"] for res in reranked_results]
        updated_nodes = sorted(
            suggested_nodes,
            key=lambda node: (
                node_id_order.index(node.metadata.node_id)
                if node.metadata.node_id in node_id_order
                else len(node_id_order)
            ),
        )[:top_k]
        
        return valid_suggestions, updated_nodes
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        return list(Collection.objects.values_list("name", flat=True))
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        try:
            collection = Collection.objects.get(name=collection_name)
        except Collection.DoesNotExist:
            raise ValueError(f"Collection not found: {collection_name}")
        
        node_count = NodeEntry.objects.filter(collection=collection).count()
        return {
            "name": collection.name,
            "embedding_dim": collection.embedding_dim,
            "node_count": node_count,
            "created_at": collection.created_at,
            "updated_at": collection.updated_at,
        }
    
    # VectorStore interface methods
    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]) -> Any:
        """Add vectors with metadata (VectorStore interface)."""
        if not vectors or not metadatas:
            logger.warning("Empty vectors or metadatas")
            return []
        
        if len(vectors) != len(metadatas):
            raise ValueError("Number of vectors must match number of metadatas")
        
        collection_name = metadatas[0].get("collection_name", "default_collection")
        collection = self.get_or_create_collection(collection_name)
        
        node_entries = []
        for i, (vector, metadata) in enumerate(zip(vectors, metadatas)):
            node_entries.append(
                NodeEntry(
                    collection=collection,
                    content=metadata.get("content", ""),
                    embedding=vector,
                    source_file_uuid=metadata.get("source_file_uuid", ""),
                    position=metadata.get("position", i),
                    custom_metadata={
                        k: v
                        for k, v in metadata.items()
                        if k not in ["content", "source_file_uuid", "position", "collection_name"]
                    },
                )
            )
        
        created_entries = NodeEntry.objects.bulk_create(node_entries)
        return [entry.node_id for entry in created_entries]
    
    def query(self, vector: List[float], top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Query vector store (VectorStore interface)."""
        collection_name = kwargs.get("collection_name", "default_collection")
        distance_type = kwargs.get("distance_type", self.vs_config.search.similarity_metric)
        
        try:
            collection = Collection.objects.get(name=collection_name)
        except Collection.DoesNotExist:
            logger.warning(f"Collection '{collection_name}' not found")
            return []
        
        # Normalize query vector if needed
        query_embedding = np.array(vector, dtype=np.float32)
        if distance_type == "cosine":
            query_embedding = self._normalize_embedding(query_embedding)
        
        # Build and execute query
        distance_operator = self._get_distance_operator(distance_type)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        
        sql_query = f"""
            SELECT node_id, content, source_file_uuid, position, custom_metadata,
                   embedding {distance_operator} %s::vector AS distance
            FROM {NodeEntry._meta.db_table}
            WHERE collection_id = %s
            ORDER BY distance
            LIMIT %s
        """
        
        with connection.cursor() as cursor:
            cursor.execute(sql_query, [embedding_str, collection.id, top_k])
            results = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
        
        return [
            {
                "node_id": dict(zip(columns, row))["node_id"],
                "content": dict(zip(columns, row))["content"],
                "metadata": dict(zip(columns, row))["custom_metadata"] or {},
                "distance": float(dict(zip(columns, row))["distance"]),
            }
            for row in results
        ]
    
    def shutdown(self) -> None:
        """Shutdown and cleanup resources."""
        logger.info("Shutting down ConfigurablePgVectorStore")
        if self.embedding_model:
            self.embedding_model.shutdown()
        super().shutdown()


__all__ = ["ConfigurablePgVectorStore"]

