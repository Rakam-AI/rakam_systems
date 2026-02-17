"""
Configurable PostgreSQL Vector Store with enhanced features.

This module provides an enhanced, fully configurable PgVectorStore that:
- Supports configuration via YAML/JSON files or dictionaries
- Allows pluggable embedding models
- Provides update_vector capability
- Maintains clean separation from other components
- Supports all search configurations
- **Dimension-agnostic vector storage**: No need to recreate tables when switching models!

## Flexible Vector Storage

Vector columns are created WITHOUT dimension constraints, allowing you to:
✓ Switch between embedding models without altering database schema
✓ Store vectors of any dimension in the same table structure
✓ No automatic table recreation or data loss
✓ Simplified database management

## Multi-Model Support

By default (use_dimension_specific_tables=True), each embedding model automatically
gets its own dedicated tables based on the model name:

- 'all-MiniLM-L6-v2' → application_nodeentry_all_minilm_l6_v2
- 'multi-qa-mpnet-base-cos-v1' → application_nodeentry_multi_qa_mpnet_base_cos_v1
- 'text-embedding-ada-002' → application_nodeentry_text_embedding_ada_002

**Why model-specific tables?**

Even if two models have the same dimensions (e.g., both 384D), their vector spaces
are completely different! Mixing embeddings from different models would give
meaningless results.

Example:
    - Model A: 'all-MiniLM-L6-v2' (384D)
    - Model B: 'paraphrase-MiniLM-L3-v2' (384D)
    
These produce vectors in DIFFERENT semantic spaces. You cannot:
❌ Search Model A embeddings using Model B query vectors
❌ Store both in the same table and expect meaningful results

This allows you to:
✓ Use multiple models simultaneously (each in its own vector space)
✓ Prevent accidental mixing of incompatible vector spaces
✓ No manual table management needed

Example:
    # Safe by default - each model uses its own tables
    store_mini = ConfigurablePgVectorStore(
        config=config_minilm  # Uses all-MiniLM-L6-v2
    )
    
    store_mpnet = ConfigurablePgVectorStore(
        config=config_mpnet  # Uses multi-qa-mpnet-base-cos-v1
    )
    # Both can coexist without conflicts or vector space mixing!

For shared table behavior, set use_dimension_specific_tables=False.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.db import connection, transaction

from rakam_systems_tools.utils import logging
from rakam_systems_core.interfaces.vectorstore import VectorStore
from rakam_systems_vectorstore.components.embedding_model.configurable_embeddings import ConfigurableEmbeddings
from rakam_systems_vectorstore.components.vectorstore.pg_models import Collection, NodeEntry
from rakam_systems_vectorstore.config import VectorStoreConfig, load_config
from rakam_systems_vectorstore.core import Node, NodeMetadata, VSFile

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
        config: Optional[Union[VectorStoreConfig, Dict, str]] = None,
        auto_recreate_on_dimension_mismatch: bool = False,
        use_dimension_specific_tables: bool = True
    ):
        """
        Initialize configurable PostgreSQL vector store.

        Args:
            name: Component name
            config: Configuration (VectorStoreConfig object, dict, or path to config file)
            auto_recreate_on_dimension_mismatch: DEPRECATED - No longer used. Vector columns now
                                                 support any dimension without schema changes.
            use_dimension_specific_tables: If True, each embedding model gets its own dedicated tables
                                          based on the model name, preventing:
                                          - Mixing incompatible vector spaces
                                          - Meaningless search results from mixed embeddings
                                          DEFAULT: True (STRONGLY recommended)

        Important:
            Even models with the same dimensions produce vectors in different semantic spaces!
            For example, 'all-MiniLM-L6-v2' and 'paraphrase-MiniLM-L3-v2' are both 384D,
            but their vectors are NOT compatible. Always use model-specific tables.
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

        # Store configuration
        self.auto_recreate_on_dimension_mismatch = auto_recreate_on_dimension_mismatch
        self.use_dimension_specific_tables = use_dimension_specific_tables

        # Table names will be set after we know the embedding dimension
        self.table_collection = "application_collection"
        self.table_nodeentry = "application_nodeentry"

        # Initialize embedding model
        self.embedding_model = ConfigurableEmbeddings(
            name=f"{name}_embeddings",
            config=self.vs_config.embedding
        )

        self.embedding_dim: Optional[int] = None

        logger.info(f"Initialized {name} with config: {self.vs_config.name}")

    def setup(self) -> None:
        """Initialize resources and connections."""
        # Skip if already initialized
        if self.initialized:
            logger.debug(
                "ConfigurablePgVectorStore already initialized, skipping setup")
            return

        logger.info("Setting up ConfigurablePgVectorStore...")

        # Ensure pgvector extension
        self._ensure_pgvector_extension()

        # Setup embedding model (will skip if already initialized)
        self.embedding_model.setup()
        self.embedding_dim = self.embedding_model.embedding_dimension

        # Set table names based on model if using model-specific tables
        if self.use_dimension_specific_tables:
            # Create a safe table suffix from model name
            # Each model gets its own table because even same-dimension models
            # have different vector spaces!
            model_name = self.vs_config.embedding.model_name
            safe_model_name = self._sanitize_model_name(model_name)

            self.table_collection = f"application_collection_{safe_model_name}"
            self.table_nodeentry = f"application_nodeentry_{safe_model_name}"
            logger.info(
                f"Using model-specific tables for '{model_name}' ({self.embedding_dim}D): "
                f"collection={self.table_collection}, "
                f"nodeentry={self.table_nodeentry}"
            )

        # Ensure the required tables exist
        self._ensure_vector_dimension_compatibility()

        logger.info(
            f"Vector store ready with embedding dimension: {self.embedding_dim}")
        super().setup()

    def _sanitize_model_name(self, model_name: str) -> str:
        """
        Convert model name to a safe table suffix.

        Examples:
            'all-MiniLM-L6-v2' -> 'all_minilm_l6_v2'
            'sentence-transformers/multi-qa-mpnet-base-cos-v1' -> 'multi_qa_mpnet_base_cos_v1'
            'text-embedding-ada-002' -> 'text_embedding_ada_002'
        """
        import re

        # Remove common prefixes
        name = model_name.replace('sentence-transformers/', '')
        name = name.replace('models/', '')

        # Replace non-alphanumeric with underscore
        name = re.sub(r'[^a-zA-Z0-9]', '_', name)

        # Convert to lowercase
        name = name.lower()

        # Remove consecutive underscores
        name = re.sub(r'_+', '_', name)

        # Remove leading/trailing underscores
        name = name.strip('_')

        # Limit length (PostgreSQL identifier limit is 63 chars)
        if len(name) > 40:
            # Keep last 40 chars (usually has version info)
            name = name[-40:]
            name = name.lstrip('_')

        return name

    def _ensure_pgvector_extension(self) -> None:
        """Ensure pgvector extension is installed."""
        with connection.cursor() as cursor:
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("Ensured pgvector extension is installed")
            except Exception as e:
                logger.error(f"Failed to create pgvector extension: {e}")
                raise

    def _ensure_vector_dimension_compatibility(self) -> None:
        """
        Ensures that the required tables exist.

        Note: Vector columns are created without dimension constraints, allowing
        flexibility to store vectors of any dimension without needing to alter
        the database schema when switching embedding models.
        """
        with connection.cursor() as cursor:
            try:
                # First ensure the collection table exists
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_collection} (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) UNIQUE NOT NULL,
                        embedding_dim INTEGER NOT NULL DEFAULT 384,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Check if the nodeentry table exists
                cursor.execute(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{self.table_nodeentry}'
                    );
                """)
                table_exists = cursor.fetchone()[0]

                if not table_exists:
                    # Table doesn't exist, create it without dimension constraint
                    logger.info(
                        f"Creating new table '{self.table_nodeentry}' (supports any vector dimension)...")
                    cursor.execute(f"""
                        CREATE TABLE {self.table_nodeentry} (
                            node_id SERIAL PRIMARY KEY,
                            collection_id INTEGER NOT NULL REFERENCES {self.table_collection}(id) ON DELETE CASCADE,
                            content TEXT NOT NULL,
                            embedding vector,
                            source_file_uuid VARCHAR(255) NOT NULL,
                            position INTEGER,
                            custom_metadata JSONB DEFAULT '{{}}'::jsonb,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                    """)

                    # Create indexes
                    cursor.execute(f"""
                        CREATE INDEX {self.table_nodeentry}_source_idx 
                        ON {self.table_nodeentry}(source_file_uuid);
                    """)
                    cursor.execute(f"""
                        CREATE INDEX {self.table_nodeentry}_collect_idx 
                        ON {self.table_nodeentry}(collection_id, source_file_uuid);
                    """)

                    logger.info(
                        f"✓ Created table '{self.table_nodeentry}' (dimension-agnostic)")
                else:
                    logger.info(
                        f"✓ Table '{self.table_nodeentry}' already exists")

            except Exception as e:
                logger.error(f"Failed to ensure table exists: {e}")
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

        # Use raw SQL when custom table names are in use
        if self.use_dimension_specific_tables:
            with connection.cursor() as cursor:
                # Try to get existing collection
                cursor.execute(
                    f"""
                    SELECT id, name, embedding_dim, created_at, updated_at
                    FROM {self.table_collection}
                    WHERE name = %s
                    """,
                    [collection_name]
                )
                row = cursor.fetchone()

                if row:
                    # Collection exists - create a Collection object manually
                    collection = Collection(
                        id=row[0],
                        name=row[1],
                        embedding_dim=row[2],
                        created_at=row[3],
                        updated_at=row[4]
                    )
                    # Mark it as existing in DB
                    collection._state.adding = False
                    logger.info(
                        f"Using existing collection: {collection_name}")
                else:
                    # Create new collection
                    cursor.execute(
                        f"""
                        INSERT INTO {self.table_collection} (name, embedding_dim, created_at, updated_at)
                        VALUES (%s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        RETURNING id, name, embedding_dim, created_at, updated_at
                        """,
                        [collection_name, embedding_dim]
                    )
                    row = cursor.fetchone()
                    collection = Collection(
                        id=row[0],
                        name=row[1],
                        embedding_dim=row[2],
                        created_at=row[3],
                        updated_at=row[4]
                    )
                    collection._state.adding = False
                    logger.info(f"Created new collection: {collection_name}")

                return collection
        else:
            # Use Django ORM for standard tables
            collection, created = Collection.objects.get_or_create(
                name=collection_name,
                defaults={"embedding_dim": embedding_dim}
            )

            logger.info(
                f"{'Created new' if created else 'Using existing'} collection: {collection_name}"
            )
            return collection

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get an existing collection (raises ValueError if not found).

        Args:
            collection_name: Name of the collection

        Returns:
            Collection object

        Raises:
            ValueError: If collection does not exist
        """
        # Use raw SQL when custom table names are in use
        if self.use_dimension_specific_tables:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT id, name, embedding_dim, created_at, updated_at
                    FROM {self.table_collection}
                    WHERE name = %s
                    """,
                    [collection_name]
                )
                row = cursor.fetchone()

                if not row:
                    raise ValueError(
                        f"Collection not found: {collection_name}")

                # Create Collection object from row
                collection = Collection(
                    id=row[0],
                    name=row[1],
                    embedding_dim=row[2],
                    created_at=row[3],
                    updated_at=row[4]
                )
                collection._state.adding = False
                return collection
        else:
            try:
                return Collection.objects.get(name=collection_name)
            except Collection.DoesNotExist:
                raise ValueError(f"Collection not found: {collection_name}")

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
            logger.warning(
                f"No nodes provided for collection '{collection_name}'")
            return

        # Filter out nodes with None or empty content (these would cause embedding errors)
        original_count = len(nodes)
        nodes = [node for node in nodes if node.content is not None and str(
            node.content).strip()]

        if len(nodes) < original_count:
            logger.warning(
                f"Filtered out {original_count - len(nodes)} nodes with empty/None content")

        if not nodes:
            logger.warning(
                f"No valid nodes to add for collection '{collection_name}' after filtering")
            return

        logger.info(
            f"Creating collection '{collection_name}' with {len(nodes)} nodes")

        # Get or create collection
        collection = self.get_or_create_collection(collection_name)

        # Generate embeddings - ensure all content is string type
        texts = [str(node.content) for node in nodes]
        embeddings = self.embedding_model.encode_documents(texts)

        # Use raw SQL when custom table names are in use
        if self.use_dimension_specific_tables:
            import json
            with connection.cursor() as cursor:
                # Clear existing nodes
                cursor.execute(
                    f"DELETE FROM {self.table_nodeentry} WHERE collection_id = %s",
                    [collection.id]
                )

                # Insert nodes using raw SQL
                for i, node in enumerate(nodes):
                    # Convert custom_metadata dict to JSON string
                    custom_metadata_json = json.dumps(
                        node.metadata.custom or {})

                    cursor.execute(
                        f"""
                        INSERT INTO {self.table_nodeentry}
                        (collection_id, content, embedding, source_file_uuid, position, custom_metadata, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        RETURNING node_id
                        """,
                        [
                            collection.id,
                            node.content,
                            embeddings[i],
                            node.metadata.source_file_uuid,
                            node.metadata.position,
                            custom_metadata_json
                        ]
                    )
                    node_id = cursor.fetchone()[0]
                    node.metadata.node_id = node_id

            logger.info(
                f"Created collection '{collection_name}' with {len(nodes)} nodes")
        else:
            # Use Django ORM for standard tables
            # Clear existing nodes
            NodeEntry.objects.filter(collection=collection).delete()

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

            logger.info(
                f"Created collection '{collection_name}' with {len(created_entries)} nodes")

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

        # Filter out nodes with None or empty content (these would cause embedding errors)
        original_count = len(nodes)
        nodes = [node for node in nodes if node.content is not None and str(
            node.content).strip()]

        if len(nodes) < original_count:
            logger.warning(
                f"Filtered out {original_count - len(nodes)} nodes with empty/None content")

        if not nodes:
            logger.warning("No valid nodes to add after filtering")
            return

        logger.info(
            f"Adding {len(nodes)} nodes to collection '{collection_name}'")

        # Generate embeddings BEFORE starting the database transaction
        # This is critical for large datasets as embedding generation can take minutes,
        # which would otherwise cause the DB connection to timeout
        logger.info(
            f"Preparing {len(nodes)} nodes for embedding generation...")
        texts = [str(node.content) for node in nodes]
        total_texts = len(texts)
        logger.info(
            f"Starting embedding generation for {total_texts} texts (this may take a while)...")
        logger.info(
            f"Model: {self.embedding_model.model_name}, Batch size: {self.embedding_model.batch_size}")
        logger.info(
            f"Expected batches: {(total_texts + self.embedding_model.batch_size - 1) // self.embedding_model.batch_size}")
        embed_start_time = time.time()

        logger.info("Calling embedding model encode_documents()...")
        embeddings = self.embedding_model.encode_documents(texts)

        embed_elapsed = time.time() - embed_start_time
        avg_rate = total_texts / embed_elapsed if embed_elapsed > 0 else 0
        logger.info(
            f"✓ Embedding generation completed in {embed_elapsed:.1f}s ({avg_rate:.1f} texts/s)")
        logger.info(f"Generated {len(embeddings)} embeddings")

        # MEMORY OPTIMIZATION: Clear texts list after embedding generation
        # The texts are no longer needed as embeddings are already computed
        del texts
        import gc
        gc.collect()

        # Now perform the database operations within a transaction
        self._insert_nodes_with_embeddings(collection_name, nodes, embeddings)

        # MEMORY OPTIMIZATION: Clear embeddings after insertion
        del embeddings
        gc.collect()

    @transaction.atomic
    def _insert_nodes_with_embeddings(self, collection_name: str, nodes: List[Node], embeddings: List) -> None:
        """
        Insert nodes with pre-computed embeddings into the database.

        This is a separate method to allow embedding generation to happen
        outside the database transaction, preventing connection timeouts
        for large datasets.

        Args:
            collection_name: Name of collection
            nodes: Nodes to insert
            embeddings: Pre-computed embeddings for the nodes
        """
        collection = self.get_collection(collection_name)

        if self.use_dimension_specific_tables:
            # Use raw SQL with batch inserts when custom tables are in use
            import json
            from django.db import transaction

            batch_size = self.vs_config.index.batch_insert_size
            total_nodes = len(nodes)
            total_batches = (total_nodes + batch_size - 1) // batch_size

            logger.info(
                f"Starting batch insert: {total_nodes} nodes in {total_batches} batches (batch_size={batch_size})")
            insert_start_time = time.time()

            # Process in batches with individual transactions to prevent connection timeout
            # Each batch is committed separately to avoid long-running transactions
            for batch_idx, batch_start in enumerate(range(0, total_nodes, batch_size)):
                batch_start_time = time.time()
                batch_end = min(batch_start + batch_size, total_nodes)
                batch_nodes = nodes[batch_start:batch_end]
                batch_embeddings = embeddings[batch_start:batch_end]

                # Use atomic transaction for each batch
                with transaction.atomic():
                    with connection.cursor() as cursor:
                        # Build batch insert values
                        values_list = []
                        params = []
                        for i, node in enumerate(batch_nodes):
                            custom_metadata_json = json.dumps(
                                node.metadata.custom or {})
                            # Convert embedding to string format for pgvector: "[1.0, 2.0, ...]"
                            embedding_values = batch_embeddings[i].tolist() if hasattr(
                                batch_embeddings[i], 'tolist') else list(batch_embeddings[i])
                            embedding_str = "[" + ",".join(str(x)
                                                           for x in embedding_values) + "]"
                            # Convert any dict values to JSON strings for psycopg2 compatibility
                            source_file_uuid = json.dumps(node.metadata.source_file_uuid) if isinstance(
                                node.metadata.source_file_uuid, dict) else node.metadata.source_file_uuid
                            position = json.dumps(node.metadata.position) if isinstance(
                                node.metadata.position, dict) else node.metadata.position
                            content = json.dumps(node.content) if isinstance(
                                node.content, dict) else node.content

                            values_list.append(
                                "(%s, %s, %s::vector, %s, %s, %s::jsonb, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)")
                            params.extend([
                                collection.id,
                                content,
                                embedding_str,
                                source_file_uuid,
                                position,
                                custom_metadata_json
                            ])

                        # Execute batch insert
                        cursor.execute(
                            f"""
                            INSERT INTO {self.table_nodeentry}
                            (collection_id, content, embedding, source_file_uuid, position, custom_metadata, created_at, updated_at)
                            VALUES {", ".join(values_list)}
                            RETURNING node_id
                            """,
                            params
                        )

                        # Get returned node IDs and update nodes
                        node_ids = cursor.fetchall()
                        for i, node in enumerate(batch_nodes):
                            node.metadata.node_id = node_ids[i][0]

                # Transaction is committed here automatically when exiting the atomic() context
                batch_elapsed = time.time() - batch_start_time

                # Log progress for every batch or at milestones
                current_batch = batch_idx + 1
                if current_batch % 10 == 0 or current_batch == total_batches or batch_elapsed > 1.0:
                    total_elapsed = time.time() - insert_start_time
                    nodes_per_sec = batch_end / total_elapsed if total_elapsed > 0 else 0
                    eta_seconds = (total_nodes - batch_end) / \
                        nodes_per_sec if nodes_per_sec > 0 else 0
                    logger.info(
                        f"Insert progress: {batch_end}/{total_nodes} nodes "
                        f"({batch_end * 100 // total_nodes}%) | "
                        f"Batch {current_batch}/{total_batches} took {batch_elapsed:.2f}s | "
                        f"Rate: {nodes_per_sec:.0f} nodes/s | "
                        f"ETA: {eta_seconds:.0f}s"
                    )

            total_time = time.time() - insert_start_time
            logger.info(
                f"Completed inserting {total_nodes} nodes to '{collection_name}' in {total_time:.2f}s ({total_nodes/total_time:.0f} nodes/s)")
        else:
            # Create entries using ORM
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

            logger.info(
                f"Added {len(created_entries)} nodes to '{collection_name}'")

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
        collection = self.get_collection(collection_name)

        if self.use_dimension_specific_tables:
            # Use raw SQL when custom tables are in use
            import json
            with connection.cursor() as cursor:
                # First, get the current node
                cursor.execute(
                    f"""
                    SELECT content, embedding, custom_metadata
                    FROM {self.table_nodeentry}
                    WHERE collection_id = %s AND node_id = %s
                    """,
                    [collection.id, node_id]
                )
                row = cursor.fetchone()

                if not row:
                    raise ValueError(
                        f"Node {node_id} not found in collection '{collection_name}'")

                current_content, current_embedding, current_metadata = row

                # Parse JSON metadata if it's a string
                if isinstance(current_metadata, str):
                    current_metadata = json.loads(current_metadata)

                # Determine updates
                updated_content = current_content
                updated_embedding = current_embedding
                updated_metadata = current_metadata or {}

                if new_content is not None:
                    updated_content = new_content
                    updated_embedding = self.embedding_model.encode_query(
                        new_content)
                    logger.info(
                        f"Updated content and regenerated embedding for node {node_id}")
                elif new_embedding is not None:
                    updated_embedding = new_embedding
                    logger.info(f"Updated embedding for node {node_id}")

                if new_metadata is not None:
                    updated_metadata.update(new_metadata)
                    logger.info(f"Updated metadata for node {node_id}")

                # Update the node
                updated_metadata_json = json.dumps(updated_metadata)
                # Convert embedding to string format for pgvector: "[1.0, 2.0, ...]"
                if updated_embedding is not None:
                    embedding_values = updated_embedding.tolist() if hasattr(
                        updated_embedding, 'tolist') else list(updated_embedding)
                    embedding_str = "[" + ",".join(str(x)
                                                   for x in embedding_values) + "]"
                else:
                    embedding_str = None
                cursor.execute(
                    f"""
                    UPDATE {self.table_nodeentry}
                    SET content = %s, embedding = %s::vector, custom_metadata = %s::jsonb, updated_at = CURRENT_TIMESTAMP
                    WHERE collection_id = %s AND node_id = %s
                    """,
                    [updated_content, embedding_str,
                        updated_metadata_json, collection.id, node_id]
                )

            logger.info(
                f"Successfully updated node {node_id} in collection '{collection_name}'")
        else:
            try:
                node_entry = NodeEntry.objects.get(
                    collection=collection, node_id=node_id)
            except NodeEntry.DoesNotExist:
                raise ValueError(
                    f"Node {node_id} not found in collection '{collection_name}'")

            # Update content and embedding
            if new_content is not None:
                node_entry.content = new_content
                # Generate new embedding
                embedding = self.embedding_model.encode_query(new_content)
                node_entry.embedding = embedding
                logger.info(
                    f"Updated content and regenerated embedding for node {node_id}")
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
            logger.info(
                f"Successfully updated node {node_id} in collection '{collection_name}'")

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

        collection = self.get_collection(collection_name)

        if self.use_dimension_specific_tables:
            # Use raw SQL when custom tables are in use
            with connection.cursor() as cursor:
                placeholders = ','.join(['%s'] * len(node_ids))
                cursor.execute(
                    f"""
                    DELETE FROM {self.table_nodeentry}
                    WHERE collection_id = %s AND node_id IN ({placeholders})
                    """,
                    [collection.id] + list(node_ids)
                )
                deleted_count = cursor.rowcount
        else:
            deleted_count, _ = NodeEntry.objects.filter(
                collection=collection,
                node_id__in=node_ids
            ).delete()

        logger.info(
            f"Deleted {deleted_count} nodes from collection '{collection_name}'")

    @transaction.atomic
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection and all its nodes.

        Args:
            collection_name: Name of collection to delete
        """
        collection = self.get_collection(collection_name)

        if self.use_dimension_specific_tables:
            # Use raw SQL when custom tables are in use
            with connection.cursor() as cursor:
                # Get node count first
                cursor.execute(
                    f"SELECT COUNT(*) FROM {self.table_nodeentry} WHERE collection_id = %s",
                    [collection.id]
                )
                node_count = cursor.fetchone()[0]

                # Delete nodes (cascade should handle this, but be explicit)
                cursor.execute(
                    f"DELETE FROM {self.table_nodeentry} WHERE collection_id = %s",
                    [collection.id]
                )

                # Delete collection
                cursor.execute(
                    f"DELETE FROM {self.table_collection} WHERE id = %s",
                    [collection.id]
                )

            logger.info(
                f"Deleted collection '{collection_name}' with {node_count} nodes")
        else:
            node_count = NodeEntry.objects.filter(
                collection=collection).count()
            collection.delete()
            logger.info(
                f"Deleted collection '{collection_name}' with {node_count} nodes")

    def keyword_search(
        self,
        collection_name: str,
        query: str,
        number: Optional[int] = None,
        meta_data_filters: Optional[Dict[str, Any]] = None,
        min_rank: float = 0.0,
        ranking_algorithm: Optional[str] = None
    ) -> Tuple[Dict, List[Node]]:
        """
        Perform pure keyword-based full-text search in collection.

        Supports multiple ranking algorithms:
        - BM25: Okapi BM25 ranking function (default, state-of-the-art)
        - ts_rank: PostgreSQL's native full-text search ranking

        Args:
            collection_name: Name of collection to search
            query: Search query (keywords)
            number: Number of results to return (uses config default if None)
            meta_data_filters: Optional metadata filters to apply
            min_rank: Minimum rank threshold (0.0 to 1.0, default 0.0)
            ranking_algorithm: Ranking algorithm to use ('bm25' or 'ts_rank', uses config default if None)

        Returns:
            Tuple of (results dict, list of Node objects)

        Example:
            # Using BM25 (default)
            results, nodes = store.keyword_search(
                collection_name="my_docs",
                query="machine learning algorithms",
                number=10,
                min_rank=0.01
            )

            # Using ts_rank
            results, nodes = store.keyword_search(
                collection_name="my_docs",
                query="machine learning algorithms",
                number=10,
                ranking_algorithm="ts_rank"
            )
        """
        # Use config defaults
        if number is None:
            number = self.vs_config.search.default_top_k
        if ranking_algorithm is None:
            ranking_algorithm = self.vs_config.search.keyword_ranking_algorithm

        logger.info(
            f"Keyword search ({ranking_algorithm}) in '{collection_name}' for: '{query}'")

        # Get collection
        try:
            collection = self.get_collection(collection_name)
        except ValueError as e:
            logger.error(str(e))
            raise

        # Use correct table name
        table_name = self.table_nodeentry if self.use_dimension_specific_tables else NodeEntry._meta.db_table

        # Build WHERE clause with metadata filters
        where_conditions = ["collection_id = %s"]
        where_params = [collection.id]

        if meta_data_filters:
            for key, value in meta_data_filters.items():
                where_conditions.append(f"custom_metadata->>'{key}' = %s")
                where_params.append(str(value))

        where_clause = " AND ".join(where_conditions)

        # Build SQL query based on ranking algorithm
        if ranking_algorithm == "bm25":
            # BM25 uses where_clause twice (doc_stats and collection_stats)
            # Parameter order: where_params (doc_stats), where_params (collection_stats), query, min_rank, limit
            sql_query = self._build_bm25_query(
                table_name, where_clause, query, min_rank, number)
            query_params = where_params + \
                where_params + [query, min_rank, number]
        else:  # ts_rank
            # ts_rank uses where_clause once
            # Parameter order: query (SELECT), where_params (WHERE), query (AND), query (AND), min_rank, limit
            sql_query = self._build_tsrank_query(
                table_name, where_clause, query, min_rank, number)
            query_params = [query] + where_params + \
                [query, query, min_rank, number]

        # Execute query
        with connection.cursor() as cursor:
            cursor.execute(sql_query, query_params)
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
            rank = result_dict["rank"]

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
                    float(rank),
                )

        logger.info(
            f"Keyword search returned {len(valid_suggestions)} results")
        return valid_suggestions, suggested_nodes

    def _build_bm25_query(
        self,
        table_name: str,
        where_clause: str,
        query: str,
        min_rank: float,
        limit: int
    ) -> str:
        """
        Build BM25 ranking SQL query.

        BM25 (Best Matching 25) is a ranking function used by search engines.
        It's based on the probabilistic retrieval framework and considers:
        - Term frequency (TF): How often query terms appear in the document
        - Inverse document frequency (IDF): How rare the terms are across all documents
        - Document length normalization: Adjusts for document length

        Formula: BM25(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D| / avgdl))

        Where:
        - D: document
        - Q: query
        - qi: query term i
        - f(qi,D): frequency of qi in D
        - |D|: length of document D
        - avgdl: average document length in the collection
        - k1: term frequency saturation parameter (default: 1.5)
        - b: length normalization parameter (default: 0.75)
        """
        k1 = self.vs_config.search.bm25_k1
        b = self.vs_config.search.bm25_b

        return f"""
            WITH doc_stats AS (
                -- Calculate document statistics
                SELECT
                    node_id,
                    content,
                    source_file_uuid,
                    position,
                    custom_metadata,
                    LENGTH(content) AS doc_length,
                    to_tsvector('english', content) AS doc_vector
                FROM
                    {table_name}
                WHERE
                    {where_clause}
            ),
            collection_stats AS (
                -- Calculate collection-wide statistics
                SELECT
                    AVG(LENGTH(content)) AS avg_doc_length,
                    COUNT(*) AS total_docs
                FROM
                    {table_name}
                WHERE
                    {where_clause}
            ),
            query_terms AS (
                -- Extract query terms and calculate IDF
                SELECT
                    word,
                    -- IDF calculation: log((N - df + 0.5) / (df + 0.5) + 1)
                    -- where N is total docs and df is document frequency
                    LN(
                        (cs.total_docs - COUNT(DISTINCT ds.node_id) + 0.5) / 
                        (COUNT(DISTINCT ds.node_id) + 0.5) + 1
                    ) AS idf
                FROM
                    unnest(string_to_array(lower(%s), ' ')) AS word,
                    doc_stats ds,
                    collection_stats cs
                WHERE
                    ds.doc_vector @@ to_tsquery('english', word)
                GROUP BY
                    word, cs.total_docs
            ),
            bm25_scores AS (
                -- Calculate BM25 score for each document
                SELECT
                    ds.node_id,
                    ds.content,
                    ds.source_file_uuid,
                    ds.position,
                    ds.custom_metadata,
                    SUM(
                        qt.idf * 
                        (
                            -- Term frequency component
                            (ts_rank(ds.doc_vector, to_tsquery('english', qt.word)) * 1000 * ({k1} + 1)) /
                            (
                                ts_rank(ds.doc_vector, to_tsquery('english', qt.word)) * 1000 + 
                                {k1} * (1 - {b} + {b} * ds.doc_length / cs.avg_doc_length)
                            )
                        )
                    ) AS bm25_score
                FROM
                    doc_stats ds
                CROSS JOIN
                    collection_stats cs
                CROSS JOIN
                    query_terms qt
                WHERE
                    ds.doc_vector @@ to_tsquery('english', qt.word)
                GROUP BY
                    ds.node_id,
                    ds.content,
                    ds.source_file_uuid,
                    ds.position,
                    ds.custom_metadata
            )
            SELECT
                node_id,
                content,
                source_file_uuid,
                position,
                custom_metadata,
                COALESCE(bm25_score, 0.0) AS rank
            FROM
                bm25_scores
            WHERE
                COALESCE(bm25_score, 0.0) > %s
            ORDER BY
                rank DESC
            LIMIT
                %s
        """

    def _build_tsrank_query(
        self,
        table_name: str,
        where_clause: str,
        query: str,
        min_rank: float,
        limit: int
    ) -> str:
        """
        Build ts_rank SQL query.

        Uses PostgreSQL's native full-text search ranking function.
        """
        return f"""
            SELECT
                node_id,
                content,
                source_file_uuid,
                position,
                custom_metadata,
                ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) AS rank
            FROM
                {table_name}
            WHERE
                {where_clause}
                AND to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                AND ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) > %s
            ORDER BY
                rank DESC
            LIMIT
                %s
        """

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

        # Get collection
        try:
            collection = self.get_collection(collection_name)
        except ValueError as e:
            logger.error(str(e))
            raise

        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Normalize if using cosine distance
        if distance_type == "cosine":
            query_embedding = self._normalize_embedding(query_embedding)

        # Determine search buffer
        search_buffer_factor = (
            self.vs_config.search.search_buffer_factor if hybrid_search else 1
        )
        limit = number * search_buffer_factor

        # Build SQL query
        distance_operator = self._get_distance_operator(distance_type)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Use correct table name
        table_name = self.table_nodeentry if self.use_dimension_specific_tables else NodeEntry._meta.db_table

        # Build WHERE clause with metadata filters
        where_conditions = ["collection_id = %s"]
        query_params = [embedding_str, collection.id]

        if meta_data_filters:
            for key, value in meta_data_filters.items():
                where_conditions.append(f"custom_metadata->>'{key}' = %s")
                query_params.append(str(value))

        where_clause = " AND ".join(where_conditions)
        query_params.append(limit)

        sql_query = f"""
            SELECT
                node_id,
                content,
                source_file_uuid,
                position,
                custom_metadata,
                embedding {distance_operator} %s::vector AS distance
            FROM
                {table_name}
            WHERE
                {where_clause}
            ORDER BY
                distance
            LIMIT
                %s
        """

        # Execute query
        with connection.cursor() as cursor:
            cursor.execute(sql_query, query_params)
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
        if hybrid_search and self.vs_config.search.rerank and valid_suggestions:
            valid_suggestions, suggested_nodes = self._rerank_results(
                query, list(valid_suggestions.values()
                            ), suggested_nodes, number
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
        node_ids = [int(res[0]["node_id"]) for res in results]

        if self.use_dimension_specific_tables:
            # Use raw SQL for full-text search with custom tables
            with connection.cursor() as cursor:
                placeholders = ','.join(['%s'] * len(node_ids))
                cursor.execute(
                    f"""
                    SELECT node_id,
                           ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) as rank
                    FROM {self.table_nodeentry}
                    WHERE node_id IN ({placeholders})
                    """,
                    [query] + node_ids
                )
                node_id_to_rank = {row[0]: row[1] for row in cursor.fetchall()}
        else:
            search_query = SearchQuery(query, config="english")
            queryset = NodeEntry.objects.filter(
                node_id__in=node_ids
            ).annotate(
                rank=SearchRank(SearchVector(
                    "content", config="english"), search_query)
            )
            node_id_to_rank = {node.node_id: node.rank for node in queryset}

        # Combine scores
        reranked_results = []

        for metadata, content, distance in results:
            node_id = metadata["node_id"]
            keyword_score = node_id_to_rank.get(node_id, 0.0)

            # Combined score: alpha * vector + (1-alpha) * keyword
            combined_score = alpha * (1 - distance) + \
                (1 - alpha) * keyword_score
            reranked_results.append((metadata, content, combined_score))

        # Sort and take top_k
        reranked_results = sorted(
            reranked_results, key=lambda x: x[2], reverse=True)[:top_k]
        valid_suggestions = {
            str(res[0]["node_id"]): res for res in reranked_results}

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
        if self.use_dimension_specific_tables:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT name FROM {self.table_collection} ORDER BY name")
                return [row[0] for row in cursor.fetchall()]
        else:
            return list(Collection.objects.values_list("name", flat=True))

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        collection = self.get_collection(collection_name)

        if self.use_dimension_specific_tables:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {self.table_nodeentry} WHERE collection_id = %s",
                    [collection.id]
                )
                node_count = cursor.fetchone()[0]
        else:
            node_count = NodeEntry.objects.filter(
                collection=collection).count()
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
            raise ValueError(
                "Number of vectors must match number of metadatas")

        collection_name = metadatas[0].get(
            "collection_name", "default_collection")
        collection = self.get_or_create_collection(collection_name)

        if self.use_dimension_specific_tables:
            # Use raw SQL when custom tables are in use
            import json
            node_ids = []
            with connection.cursor() as cursor:
                for i, (vector, metadata) in enumerate(zip(vectors, metadatas)):
                    custom_metadata = {
                        k: v
                        for k, v in metadata.items()
                        if k not in ["content", "source_file_uuid", "position", "collection_name"]
                    }
                    custom_metadata_json = json.dumps(custom_metadata)

                    cursor.execute(
                        f"""
                        INSERT INTO {self.table_nodeentry}
                        (collection_id, content, embedding, source_file_uuid, position, custom_metadata, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        RETURNING node_id
                        """,
                        [
                            collection.id,
                            metadata.get("content", ""),
                            vector,
                            metadata.get("source_file_uuid", ""),
                            metadata.get("position", i),
                            custom_metadata_json
                        ]
                    )
                    node_id = cursor.fetchone()[0]
                    node_ids.append(node_id)
            return node_ids
        else:
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
        distance_type = kwargs.get(
            "distance_type", self.vs_config.search.similarity_metric)

        try:
            collection = self.get_collection(collection_name)
        except ValueError:
            logger.warning(f"Collection '{collection_name}' not found")
            return []

        # Normalize query vector if needed
        query_embedding = np.array(vector, dtype=np.float32)
        if distance_type == "cosine":
            query_embedding = self._normalize_embedding(query_embedding)

        # Build and execute query
        distance_operator = self._get_distance_operator(distance_type)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Use correct table name
        table_name = self.table_nodeentry if self.use_dimension_specific_tables else NodeEntry._meta.db_table

        sql_query = f"""
            SELECT node_id, content, source_file_uuid, position, custom_metadata,
                   embedding {distance_operator} %s::vector AS distance
            FROM {table_name}
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
