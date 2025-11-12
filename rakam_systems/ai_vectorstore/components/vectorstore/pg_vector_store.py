import logging
import os
import time
from functools import lru_cache
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import dotenv
import numpy as np
from django.contrib.postgres.search import SearchQuery
from django.contrib.postgres.search import SearchRank
from django.contrib.postgres.search import SearchVector
from django.db import connection
from django.db import transaction
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ai_core.interfaces.vectorstore import VectorStore
from ai_vectorstore.components.vectorstore.pg_models import Collection
from ai_vectorstore.components.vectorstore.pg_models import NodeEntry
from ai_vectorstore.core import Node
from ai_vectorstore.core import NodeMetadata
from ai_vectorstore.core import VSFile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


class PgVectorStore(VectorStore):
    """
    A class for managing collection-based vector stores using pgvector and Django ORM.
    Enhanced for better semantic search performance with hybrid search, re-ranking, and caching.
    """

    def __init__(
        self,
        name: str = "pg_vector_store",
        config=None,
        embedding_model: str = "Snowflake/snowflake-arctic-embed-m",
        use_embedding_api: bool = False,
        api_model: str = "text-embedding-3-small",
    ) -> None:
        """
        Initializes the PgVectorStore with the specified embedding model.

        :param name: Name of the vector store component.
        :param config: Configuration object.
        :param embedding_model: Pre-trained SentenceTransformer model name.
        :param use_embedding_api: Whether to use OpenAI's embedding API instead of local model.
        :param api_model: OpenAI API model to use for embeddings if use_embedding_api is True.
        """
        super().__init__(name=name, config=config)
        self._ensure_pgvector_extension()
        self.use_embedding_api = use_embedding_api

        if self.use_embedding_api:
            self.client = OpenAI(api_key=api_key)
            self.api_model = api_model
            sample_embedding = self._get_api_embedding("Sample text")
            self.embedding_dim = len(sample_embedding)
        else:
            self.embedding_model = SentenceTransformer(
                embedding_model, trust_remote_code=True
            )
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        logger.info(
            f"Initialized PgVectorStore with embedding dimension: {self.embedding_dim}"
        )

    def _ensure_pgvector_extension(self) -> None:
        """
        Ensures that the pgvector extension is installed in the PostgreSQL database.
        """
        with connection.cursor() as cursor:
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("Ensured pgvector extension is installed")
            except Exception as e:
                logger.error(f"Failed to create pgvector extension: {e}")
                raise

    def _get_api_embedding(self, text: str) -> List[float]:
        """
        Gets embedding from OpenAI API.

        :param text: Text to embed
        :return: Embedding vector
        """
        try:
            response = self.client.embeddings.create(input=[text], model=self.api_model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to get API embedding: {e}")
            raise

    @lru_cache(maxsize=1000)
    def predict_embeddings(self, query: str) -> np.ndarray:
        """
        Predicts embeddings for a given query using the embedding model.
        Caches results to reduce redundant computations.

        :param query: Query string to encode.
        :return: Normalized embedding vector for the query.
        """
        logger.debug(f"Predicting embeddings for query: {query}")
        start_time = time.time()

        if self.use_embedding_api:
            query_embedding = self._get_api_embedding(query)
            query_embedding = np.array(query_embedding, dtype="float32")
        else:
            query_embedding = self.embedding_model.encode(query)
            query_embedding = np.array(query_embedding, dtype="float32")

        # Normalize embedding for cosine similarity
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        else:
            logger.warning(f"Zero norm encountered for query: {query}")

        logger.debug(
            f"Embedding generation took {time.time() - start_time:.2f} seconds"
        )
        return query_embedding

    def get_embeddings(
        self, sentences: List[str], parallel: bool = True, batch_size: int = 8
    ) -> np.ndarray:
        """
        Generates embeddings for a list of sentences with normalization.

        :param sentences: List of sentences to encode.
        :param parallel: Whether to use parallel processing (default is True).
        :param batch_size: Batch size for processing (default is 8).
        :return: Normalized embedding vectors for the sentences.
        """
        logger.info(f"Generating embeddings for {len(sentences)} sentences")
        start = time.time()

        if self.use_embedding_api:
            all_embeddings = []
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                response = self.client.embeddings.create(
                    input=batch, model=self.api_model
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            embeddings = np.array(all_embeddings, dtype="float32")
        else:
            if parallel:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                pool = self.embedding_model.start_multi_process_pool(
                    target_devices=["cpu"] * 5
                )
                embeddings = self.embedding_model.encode_multi_process(
                    sentences, pool, batch_size=batch_size
                )
                self.embedding_model.stop_multi_process_pool(pool)
            else:
                os.environ["TOKENIZERS_PARALLELISM"] = "true"
                embeddings = self.embedding_model.encode(
                    sentences,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                )
                embeddings = embeddings.cpu().detach().numpy()

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms

        logger.info(
            f"Time taken to encode {len(sentences)} items: {time.time() - start:.2f} seconds"
        )
        return embeddings

    def get_or_create_collection(self, collection_name: str) -> Collection:
        """
        Gets or creates a collection with the specified name.

        :param collection_name: Name of the collection.
        :return: Collection object.
        """
        collection, created = Collection.objects.get_or_create(
            name=collection_name, defaults={"embedding_dim": self.embedding_dim}
        )
        logger.info(
            f"{'Created new' if created else 'Using existing'} collection: {collection_name}"
        )
        return collection

    def _rerank_results(
        self,
        query: str,
        results: List[Tuple[Dict, str, float]],
        suggested_nodes: List[Node],
        top_k: int,
    ) -> Tuple[Dict, List[Node]]:
        """
        Re-ranks search results using a combination of vector similarity and keyword relevance.

        :param query: The search query.
        :param results: Initial search results (metadata, content, distance).
        :param suggested_nodes: List of Node objects.
        :param top_k: Number of results to return after re-ranking.
        :return: Tuple of re-ranked results dictionary and updated suggested_nodes.
        """
        logger.debug(f"Re-ranking {len(results)} results for query: {query}")

        # Perform full-text search to get keyword relevance scores
        search_query = SearchQuery(query, config="english")
        queryset = NodeEntry.objects.filter(
            collection__name="document_collection",
            node_id__in=[int(res[0]["node_id"]) for res in results],
        ).annotate(
            rank=SearchRank(SearchVector("content", config="english"), search_query)
        )

        # Combine vector distance and keyword rank
        reranked_results = []
        node_id_to_rank = {node.node_id: node.rank for node in queryset}
        for metadata, content, distance in results:
            node_id = metadata["node_id"]
            keyword_score = node_id_to_rank.get(node_id, 0.0)
            # Combine scores (adjust weights as needed)
            combined_score = 0.7 * (1 - distance) + 0.3 * keyword_score
            reranked_results.append((metadata, content, combined_score))

        # Sort by combined score and take top_k
        reranked_results = sorted(reranked_results, key=lambda x: x[2], reverse=True)[
            :top_k
        ]
        valid_suggestions = {str(res[0]["node_id"]): res for res in reranked_results}

        # Update suggested_nodes to match re-ranked order
        node_id_order = [res[0]["node_id"] for res in reranked_results]
        updated_nodes = sorted(
            suggested_nodes,
            key=lambda node: node_id_order.index(node.metadata.node_id)
            if node.metadata.node_id in node_id_order
            else len(node_id_order),
        )[:top_k]

        logger.debug(f"Re-ranked to {len(valid_suggestions)} results")
        return valid_suggestions, updated_nodes

    def search(
        self,
        collection_name: str,
        query: str,
        distance_type: str = "cosine",
        number: int = 5,
        meta_data_filters: Optional[Dict[str, Any]] = None,
        hybrid_search: bool = True,
    ) -> Tuple[Dict, List[Node]]:
        """
        Retrieve relevant documents from the vector store using hybrid search and re-ranking.

        :param collection_name: Name of the collection to search.
        :param query: Search query.
        :param distance_type: Distance metric ("cosine", "l2", "dot").
        :param number: Number of results to return.
        :param meta_data_filters: Dictionary of metadata filters (e.g., {"is_validated": True}).
        :param hybrid_search: Whether to use hybrid search combining vector and keyword search.
        :return: Tuple of search results (dictionary) and suggested nodes.
        """
        logger.info(f"Searching in collection: {collection_name} for query: '{query}'")

        try:
            collection = Collection.objects.get(name=collection_name)
        except Collection.DoesNotExist:
            logger.error(f"No collection found with name: {collection_name}")
            raise ValueError(f"No collection found with name: {collection_name}")

        # Generate query embedding
        query_embedding = self.predict_embeddings(query)

        # Build base queryset
        queryset = NodeEntry.objects.filter(collection=collection)

        # Apply metadata filters
        if meta_data_filters:
            for key, value in meta_data_filters.items():
                queryset = queryset.filter(**{f"custom_metadata__{key}": value})

        # Construct SQL query for vector search
        if distance_type == "cosine":
            distance_operator = "<=>"
        elif distance_type == "l2":
            distance_operator = "<->"
        elif distance_type == "dot":
            distance_operator = "<#>"
        else:
            logger.error(f"Unsupported distance type: {distance_type}")
            raise ValueError(f"Unsupported distance type: {distance_type}")

        # Request more results for hybrid search and re-ranking
        search_buffer_factor = 2 if hybrid_search else 1
        limit = number * search_buffer_factor
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

        # Execute vector search
        with connection.cursor() as cursor:
            cursor.execute(sql_query, [embedding_str, collection.id, limit])
            results = cursor.fetchall()
            columns = [col[0] for col in cursor.description]

        # Process vector search results
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
                if isinstance(custom_metadata, str):
                    try:
                        import json

                        custom_metadata = json.loads(custom_metadata)
                    except (json.JSONDecodeError, TypeError):
                        custom_metadata = {}

                metadata = NodeMetadata(
                    source_file_uuid=result_dict["source_file_uuid"],
                    position=result_dict["position"],
                    custom=custom_metadata,
                )
                metadata.node_id = node_id
                node = Node(content=content, metadata=metadata)
                node.embedding = result_dict.get("embedding")
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

        # Perform hybrid search and re-ranking if enabled
        if hybrid_search:
            valid_suggestions, suggested_nodes = self._rerank_results(
                query, list(valid_suggestions.values()), suggested_nodes, number
            )

        logger.info(f"Search returned {len(valid_suggestions)} results")
        return valid_suggestions, suggested_nodes

    @transaction.atomic
    def create_collection_from_files(
        self, collection_name: str, files: List[VSFile]
    ) -> None:
        """
        Creates a collection from a list of VSFile objects.

        :param collection_name: Name of the collection to create.
        :param files: List of VSFile objects containing nodes.
        """
        logger.info(f"Creating collection: {collection_name} from files")
        nodes = [node for file in files for node in file.nodes]
        self.create_collection_from_nodes(collection_name, nodes)

    @transaction.atomic
    def create_collection_from_nodes(
        self, collection_name: str, nodes: List[Node]
    ) -> None:
        """
        Creates a collection from a list of nodes.

        :param collection_name: Name of the collection to create.
        :param nodes: List of Node objects.
        """
        if not nodes:
            logger.warning(
                f"Cannot create collection '{collection_name}' because nodes list is empty"
            )
            return

        logger.info(f"Creating collection: {collection_name} with {len(nodes)} nodes")
        collection = self.get_or_create_collection(collection_name)
        NodeEntry.objects.filter(collection=collection).delete()

        text_chunks = [node.content for node in nodes]
        embeddings = self.get_embeddings(text_chunks, parallel=False)

        node_entries = [
            NodeEntry(
                collection=collection,
                content=node.content,
                embedding=embeddings[i].tolist(),
                source_file_uuid=node.metadata.source_file_uuid,
                position=node.metadata.position,
                custom_metadata=node.metadata.custom or {},
            )
            for i, node in enumerate(nodes)
        ]

        created_entries = NodeEntry.objects.bulk_create(node_entries)
        for i, node in enumerate(nodes):
            node.metadata.node_id = created_entries[i].node_id

        logger.info(
            f"Created collection '{collection_name}' with {len(created_entries)} nodes"
        )

    @transaction.atomic
    def add_nodes(self, collection_name: str, nodes: List[Node]) -> None:
        """
        Adds nodes to an existing collection.

        :param collection_name: Name of the collection to update.
        :param nodes: List of Node objects to be added.
        """
        if not nodes:
            logger.warning("No nodes to add")
            return

        logger.info(f"Adding {len(nodes)} nodes to collection: {collection_name}")
        try:
            collection = Collection.objects.get(name=collection_name)
        except Collection.DoesNotExist:
            raise ValueError(f"No collection found with name: {collection_name}")

        text_chunks = [node.content for node in nodes]
        embeddings = self.get_embeddings(text_chunks, parallel=False)

        node_entries = [
            NodeEntry(
                collection=collection,
                content=node.content,
                embedding=embeddings[i].tolist(),
                source_file_uuid=node.metadata.source_file_uuid,
                position=node.metadata.position,
                custom_metadata=node.metadata.custom or {},
            )
            for i, node in enumerate(nodes)
        ]

        created_entries = NodeEntry.objects.bulk_create(node_entries)
        for i, node in enumerate(nodes):
            node.metadata.node_id = created_entries[i].node_id

        logger.info(
            f"Added {len(created_entries)} nodes to collection '{collection_name}'"
        )

    @transaction.atomic
    def delete_nodes(self, collection_name: str, node_ids: List[int]) -> None:
        """
        Deletes nodes from an existing collection.

        :param collection_name: Name of the collection to update.
        :param node_ids: List of node IDs to be deleted.
        """
        if not node_ids:
            logger.warning("No node IDs to delete")
            return

        logger.info(
            f"Deleting {len(node_ids)} nodes from collection: {collection_name}"
        )
        try:
            collection = Collection.objects.get(name=collection_name)
        except Collection.DoesNotExist:
            raise ValueError(f"No collection found with name: {collection_name}")

        existing_ids = set(
            NodeEntry.objects.filter(
                collection=collection, node_id__in=node_ids
            ).values_list("node_id", flat=True)
        )
        missing_ids = set(node_ids) - existing_ids
        if missing_ids:
            logger.warning(
                f"Node ID(s) {missing_ids} not found in collection {collection_name}"
            )

        deleted_count, _ = NodeEntry.objects.filter(
            collection=collection, node_id__in=existing_ids
        ).delete()
        logger.info(
            f"Deleted {deleted_count} nodes from collection '{collection_name}'"
        )

    @transaction.atomic
    def add_files(self, collection_name: str, files: List[VSFile]) -> None:
        """
        Adds file nodes to the specified collection.

        :param collection_name: Name of the collection to update.
        :param files: List of VSFile objects whose nodes are to be added.
        """
        logger.info(f"Adding files to collection: {collection_name}")
        all_nodes = [node for file in files for node in file.nodes]
        self.add_nodes(collection_name, all_nodes)

    @transaction.atomic
    def delete_files(self, collection_name: str, files: List[VSFile]) -> None:
        """
        Deletes file nodes from the specified collection.

        :param collection_name: Name of the collection to update.
        :param files: List of VSFile objects whose nodes are to be deleted.
        """
        logger.info(f"Deleting files from collection: {collection_name}")
        node_ids_to_delete = [
            node.metadata.node_id
            for file in files
            for node in file.nodes
            if node.metadata.node_id
        ]
        if node_ids_to_delete:
            self.delete_nodes(collection_name, node_ids_to_delete)
        else:
            logger.warning("No node IDs found in provided files")

    def list_collections(self) -> List[str]:
        """
        Lists all available collections.

        :return: List of collection names.
        """
        return list(Collection.objects.values_list("name", flat=True))

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Gets information about a collection.

        :param collection_name: Name of the collection.
        :return: Dictionary containing collection information.
        """
        try:
            collection = Collection.objects.get(name=collection_name)
        except Collection.DoesNotExist:
            raise ValueError(f"No collection found with name: {collection_name}")

        node_count = NodeEntry.objects.filter(collection=collection).count()
        return {
            "name": collection.name,
            "embedding_dim": collection.embedding_dim,
            "node_count": node_count,
            "created_at": collection.created_at,
            "updated_at": collection.updated_at,
        }

    @transaction.atomic
    def delete_collection(self, collection_name: str) -> None:
        """
        Deletes a collection and all its nodes.

        :param collection_name: Name of the collection to delete.
        """
        try:
            collection = Collection.objects.get(name=collection_name)
        except Collection.DoesNotExist:
            raise ValueError(f"No collection found with name: {collection_name}")

        node_count = NodeEntry.objects.filter(hourly=collection).count()
        collection.delete()
        logger.info(f"Deleted collection '{collection_name}' with {node_count} nodes")

    # VectorStore interface methods
    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]) -> Any:
        """
        Adds vectors with metadata to the default collection.
        This method implements the VectorStore interface.

        :param vectors: List of embedding vectors to add.
        :param metadatas: List of metadata dictionaries for each vector.
        :return: List of node IDs that were created.
        """
        if not vectors or not metadatas:
            logger.warning("Empty vectors or metadatas provided to add()")
            return []

        if len(vectors) != len(metadatas):
            raise ValueError("Number of vectors must match number of metadatas")

        # Get or create default collection
        collection_name = metadatas[0].get("collection_name", "default_collection")
        collection = self.get_or_create_collection(collection_name)

        # Create nodes from vectors and metadatas
        node_entries = []
        for i, (vector, metadata) in enumerate(zip(vectors, metadatas)):
            content = metadata.get("content", "")
            source_file_uuid = metadata.get("source_file_uuid", "")
            position = metadata.get("position", i)
            custom_metadata = {
                k: v
                for k, v in metadata.items()
                if k not in ["content", "source_file_uuid", "position", "collection_name"]
            }

            node_entries.append(
                NodeEntry(
                    collection=collection,
                    content=content,
                    embedding=vector,
                    source_file_uuid=source_file_uuid,
                    position=position,
                    custom_metadata=custom_metadata,
                )
            )

        created_entries = NodeEntry.objects.bulk_create(node_entries)
        node_ids = [entry.node_id for entry in created_entries]
        logger.info(f"Added {len(node_ids)} vectors to collection '{collection_name}'")
        return node_ids

    def query(
        self, vector: List[float], top_k: int = 5, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Queries the vector store for similar vectors.
        This method implements the VectorStore interface.

        :param vector: Query vector.
        :param top_k: Number of results to return.
        :param kwargs: Additional parameters (collection_name, distance_type, meta_data_filters).
        :return: List of dictionaries containing search results.
        """
        collection_name = kwargs.get("collection_name", "default_collection")
        distance_type = kwargs.get("distance_type", "cosine")
        meta_data_filters = kwargs.get("meta_data_filters")

        try:
            collection = Collection.objects.get(name=collection_name)
        except Collection.DoesNotExist:
            logger.warning(f"Collection '{collection_name}' not found")
            return []

        # Convert vector to numpy array
        query_embedding = np.array(vector, dtype="float32")

        # Normalize if using cosine distance
        if distance_type == "cosine":
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

        # Build queryset
        queryset = NodeEntry.objects.filter(collection=collection)

        # Apply metadata filters
        if meta_data_filters:
            for key, value in meta_data_filters.items():
                queryset = queryset.filter(**{f"custom_metadata__{key}": value})

        # Determine distance operator
        if distance_type == "cosine":
            distance_operator = "<=>"
        elif distance_type == "l2":
            distance_operator = "<->"
        elif distance_type == "dot":
            distance_operator = "<#>"
        else:
            raise ValueError(f"Unsupported distance type: {distance_type}")

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
            cursor.execute(sql_query, [embedding_str, collection.id, top_k])
            results = cursor.fetchall()
            columns = [col[0] for col in cursor.description]

        # Format results
        formatted_results = []
        for row in results:
            result_dict = dict(zip(columns, row))
            formatted_results.append({
                "node_id": result_dict["node_id"],
                "content": result_dict["content"],
                "source_file_uuid": result_dict["source_file_uuid"],
                "position": result_dict["position"],
                "metadata": result_dict["custom_metadata"] or {},
                "distance": float(result_dict["distance"]),
            })

        logger.info(f"Query returned {len(formatted_results)} results")
        return formatted_results