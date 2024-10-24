import logging
import os
import pickle
import time
from typing import Any
from typing import Dict
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rakam_systems.core import VSFile, NodeMetadata, Node

from rakam_systems.core import VSFile



# Configure logging
logging.basicConfig(level=logging.INFO)


class VectorStore:
    """
    A class for managing collection-based vector stores using FAISS and SentenceTransformers.
    """

    def __init__(self, base_index_path: str, embedding_model: str, initialising: bool = False) -> None:
        """
        Initializes the VectorStore with the specified base index path and embedding model.

        :param base_index_path: Base path to store the FAISS indexes.
        :param embedding_model: Pre-trained SentenceTransformer model name.
        """
        self.base_index_path = base_index_path
        if not os.path.exists(self.base_index_path):
            os.makedirs(self.base_index_path)

        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.collections = {}

        if not initialising : self.load_vector_store()

    def load_vector_store(self) -> None:
        """
        Loads all collections from the base directory.
        """
        for collection_name in os.listdir(self.base_index_path):
            store_path = os.path.join(self.base_index_path, collection_name)
            if os.path.isdir(store_path):
                self.collections[collection_name] = self.load_collection(store_path)

    def load_collection(self, store_path: str) -> Dict[str, Any]:
        """
        Loads a single vector store from the specified directory.

        :param store_path: Path to the store directory.
        :return: Dictionary containing the store's index, nodes, and metadata.
        """
        store = {}
        store["index"] = faiss.read_index(os.path.join(store_path, "index"))
        with open(
            os.path.join(store_path, "category_index_mapping.pkl"), "rb"
        ) as f:
            store["category_index_mapping"] = pickle.load(f)
        with open(
            os.path.join(store_path, "metadata_index_mapping.pkl"), "rb"
        ) as f:
            store["metadata_index_mapping"] = pickle.load(f)
        with open(os.path.join(store_path, "nodes.pkl"), "rb") as f:
            store["nodes"] = pickle.load(f)
        logging.info(f"Store loaded successfully from {store_path}.")
        return store

    def predict_embeddings(self, query: str) -> np.ndarray:
        """
        Predicts embeddings for a given query using the embedding model.

        :param query: Query string to encode.
        :return: Embedding vector for the query.
        """
        logging.info(f"Predicting embeddings for query: {query}")
        query_embedding = self.embedding_model.encode(query)
        query_embedding = np.asarray([query_embedding], dtype="float32")
        return query_embedding

    def search(
        self, collection_name: str, query: str, distance_type="cosine", number=5
    ) -> dict:
        """
        Searches the specified collection for the closest embeddings to the query.

        :param collection_name: Name of the collection to search in.
        :param query: Query string to search for.
        :param distance_type: Distance metric to use (default is "cosine").
        :param number: Number of closest embeddings to return.
        :return: Dictionary of search results with suggestion texts and distances.
        """
        logging.info(
            f"Searching in collection: {collection_name} for query: {query} with distance type: {distance_type} and number of results: {number}"
        )

        store = self.collections.get(collection_name)
        if not store:
            raise ValueError(f"No store found with name: {collection_name} in path : {self.base_index_path}")

        query_embedding = self.predict_embeddings(query)
        suggested_nodes = []

        if distance_type == "cosine":
            faiss.normalize_L2(query_embedding)
        D, I = store["index"].search(query_embedding, number)

        seen_texts = set()
        valid_suggestions = {}
        count = 0
        for i, id_ in enumerate(I[0]):
            if count >= number:
                break
            if id_ != -1 and id_ in store["category_index_mapping"]:
                suggestion_text = store["category_index_mapping"][id_]
                node_metadata = store["metadata_index_mapping"][id_]
                suggested_nodes.append(store["nodes"][id_])
                if suggestion_text not in seen_texts:
                    seen_texts.add(suggestion_text)
                    valid_suggestions[str(id_)] = (
                        node_metadata,
                        suggestion_text,
                        float(D[0][i]),
                    )
                    count += 1

        while count < number:
            valid_suggestions[f"placeholder_{count}"] = (
                "No suggestion available",
                float("inf"),
            )
            count += 1

        logging.info(f"Search results: {valid_suggestions}")
        return valid_suggestions, suggested_nodes

    def get_embeddings(self, sentences: List[str], parallel: bool = True, batch_size: int = 8) -> np.ndarray:
        """
        Generates embeddings for a list of sentences.

        :param sentences: List of sentences to encode.
        :param parallel: Whether to use parallel processing (default is True).
        :return: Embedding vectors for the sentences.
        """
        logging.info(f"Generating embeddings for {len(sentences)} sentences.")
        print("GEnerating embeddings...")
        start = time.time()
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
        logging.info(
            f"Time taken to encode {len(sentences)} items: {round(time.time() - start, 2)} seconds"
        )
        return embeddings.cpu().detach().numpy()

    def create_collection_from_files(self, collection_name: str, files: List[VSFile]) -> None:
        """
        Creates FAISS indexes from dictionaries of store names and VSFile objects.

        :param collection_files: Dictionary where keys are store names and values are lists of VSFile objects.
        """
        logging.info(f"Creating FAISS index for store: {collection_name}")
        text_chunks = []
        metadata = []
        nodes = []

        for file in files:
            for node in file.nodes:
                nodes.append(node)
                text_chunks.append(node.content)
                formatted_metadata = {
                    "node_id": node.metadata.node_id,
                    "source_file_uuid": node.metadata.source_file_uuid,
                    "position": node.metadata.position,
                    "custom": node.metadata.custom,
                }
                metadata.append(formatted_metadata)

        self._create_and_save_index(collection_name, nodes, text_chunks, metadata)

    def create_collections_from_files(self, collection_files: Dict[str, List[VSFile]]) -> None:
        """
        Creates FAISS indexes from dictionaries of store names and VSFile objects.

        :param collection_files: Dictionary where keys are store names and values are lists of VSFile objects.
        """
        for collection_name, files in collection_files.items():
            self.create_collection_from_files(collection_name, files)
    
    def create_collection_from_nodes(self, collection_name: str, nodes: List[Any]) -> None:
        """
        Creates a FAISS index from a list of nodes and collection it under the given collection name.

        :param collection_name: Name of the store to create.
        :param nodes: List of nodes containing the content and metadata.
        """
        logging.info(f"Creating FAISS index for store: {collection_name}")
        text_chunks = []
        metadata = []

        for node in nodes:
            text_chunks.append(node.content)
            formatted_metadata = {
                "node_id": node.metadata.node_id,
                "source_file_uuid": node.metadata.source_file_uuid,
                "position": node.metadata.position,
                "custom": node.metadata.custom,
            }
            metadata.append(formatted_metadata)

        self._create_and_save_index(collection_name, nodes, text_chunks, metadata)

    def create_from_nodes(self, nodes: List[Any]) -> None:
        """
        Creates a FAISS index from a list of nodes and collection it under the default name "base".

        :param nodes: List of nodes containing the content and metadata. 
        """
        logging.info("Creating FAISS index for store: base")
        text_chunks = []
        metadata = []

        for node in nodes:
            text_chunks.append(node.content)
            formatted_metadata = {
                "node_id": node.metadata.node_id,
                "source_file_uuid": node.metadata.source_file_uuid,
                "position": node.metadata.position,
                "custom": node.metadata.custom,
            }
            metadata.append(formatted_metadata)

        self._create_and_save_index("base", nodes, text_chunks, metadata)

    def _create_and_save_index(
        self,
        collection_name: str,
        nodes: List[Any],
        text_chunks: List[str],
        metadata: List[Dict[str, Any]],
        ) -> None:
        """
        Helper function to create and save a FAISS index.

        :param collection_name: Name of the store to create.
        :param nodes: List of nodes.
        :param text_chunks: List of text chunks to encode and index.
        :param metadata: List of metadata associated with the text chunks.
        """
        # Check if the list of nodes or text_chunks is empty
        if not nodes or not text_chunks:
            logging.warning(f"Cannot create FAISS index for store '{collection_name}' because nodes or text_chunks are empty.")
            self.collections[collection_name] = {
            "index": None,
            "nodes": [],
            "category_index_mapping": None,
            "metadata_index_mapping": None,
        }
            return

        store_path = os.path.join(self.base_index_path, collection_name)
        if not os.path.exists(store_path):
            os.makedirs(store_path)

        # Get embeddings for the text chunks
        data_embeddings = self.get_embeddings(sentences=text_chunks, parallel=False)
        category_index_mapping = dict(zip(range(len(text_chunks)), text_chunks))

        # Save category index mapping to file
        with open(os.path.join(store_path, "category_index_mapping.pkl"), "wb") as f:
            pickle.dump(category_index_mapping, f)

        # Save nodes to file
        with open(os.path.join(store_path, "nodes.pkl"), "wb") as f:
            pickle.dump(nodes, f)

        # Create FAISS index and add embeddings
        index = faiss.IndexIDMap(faiss.IndexFlatIP(data_embeddings.shape[1]))
        faiss.normalize_L2(data_embeddings)
        index.add_with_ids(
            data_embeddings, np.array(list(category_index_mapping.keys()))
        )
        faiss.write_index(index, os.path.join(store_path, "index"))

        # Save metadata index mapping to file
        metadata_index_mapping = dict(zip(range(len(text_chunks)), metadata))
        with open(os.path.join(store_path, "metadata_index_mapping.pkl"), "wb") as f:
            pickle.dump(metadata_index_mapping, f)

        # Update the collections dictionary
        self.collections[collection_name] = {
            "index": index,
            "nodes": nodes,
            "category_index_mapping": category_index_mapping,
            "metadata_index_mapping": metadata_index_mapping,
        }

        logging.info(
            f"FAISS index for store {collection_name} created and saved successfully."
        )

    def add_nodes(self, collection_name: str, nodes: List[Node]) -> None:
        """
        Adds nodes to an existing store and updates the index.

        :param collection_name: Name of the store to update.
        :param nodes: List of nodes to be added.
        """
        logging.info(f"Adding nodes to store: {collection_name}")

        store = self.collections.get(collection_name)
        if not store:
            raise ValueError(f"No store found with name: {collection_name}")

        new_text_chunks = [node.content for node in nodes]
        new_metadata = [
            {
                "node_id": node.metadata.node_id,
                "source_file_uuid": node.metadata.source_file_uuid,
                "position": node.metadata.position,
                "custom": node.metadata.custom,
            }
            for node in nodes
        ]

        # Add new embeddings to the index
        new_embeddings = self.get_embeddings(sentences=new_text_chunks, parallel=False)

        # Add new entries to the index
        new_ids = range(
            len(store["category_index_mapping"]),
            len(store["category_index_mapping"]) + len(new_text_chunks),
        )
        store["index"].add_with_ids(new_embeddings, np.array(list(new_ids)))

        # Update the mappings
        store["category_index_mapping"].update(dict(zip(new_ids, new_text_chunks)))
        store["metadata_index_mapping"].update(dict(zip(new_ids, new_metadata)))
        store["nodes"].extend(nodes)

        # Save updated store
        self._save_collection(collection_name)

    def delete_nodes(self, collection_name: str, node_ids: List[int]) -> None:
        """
        Deletes nodes from an existing store and updates the index without re-creating it from scratch.

        :param collection_name: Name of the store to update.
        :param node_ids: List of node IDs to be deleted.
        """
        logging.info(f"Deleting nodes {node_ids} from store: {collection_name}")

        store = self.collections.get(collection_name)
        if not store:
            raise ValueError(f"No store found with name: {collection_name}")

        existed_ids = store["category_index_mapping"].keys()
        logging.info(f"Existed IDs before deletion: {existed_ids}")

        missing_ids = []
        ids_to_delete = []
        for node_id in node_ids:
            if node_id not in existed_ids:
                missing_ids.append(node_id)
            else:
                ids_to_delete.append(node_id)
        
        # Filter out the nodes to be deleted
        remaining_ids = [
            id_ for id_ in store["category_index_mapping"].keys() if id_ not in ids_to_delete
        ]

        remaining_text_chunks = [
            store["category_index_mapping"][id_] for id_ in remaining_ids
        ]
        remaining_metadata = [
            store["metadata_index_mapping"][id_] for id_ in remaining_ids
        ]
        remaining_nodes = [store["nodes"][id_] for id_ in remaining_ids]

        # Get existing index and extract embeddings
        existed_index = store["index"]
        base_index = existed_index.index
        num_vectors = base_index.ntotal
        data_embeddings = np.empty((num_vectors, base_index.d), dtype=np.float32)

        # Reconstruct embeddings from the current index
        for i in range(num_vectors):
            embedding = np.empty((base_index.d,), dtype=np.float32)  # Create a 1D array for the reconstructed embedding
            base_index.reconstruct(i, embedding)  # Reconstruct the embedding for index i
            data_embeddings[i] = embedding  # Store it in the embeddings array

        # Remove embeddings for the nodes to be deleted
        delete_indices = [i for i, id_ in enumerate(store["category_index_mapping"].keys()) if id_ in ids_to_delete]
        data_embeddings = np.delete(data_embeddings, delete_indices, axis=0)

        # Create a new FAISS index with the remaining embeddings
        index = faiss.IndexIDMap(faiss.IndexFlatIP(data_embeddings.shape[1]))
        faiss.normalize_L2(data_embeddings)
        index.add_with_ids(
            data_embeddings, np.array(list(remaining_ids))
        )

        # Update the store without re-creating the index from scratch
        store["category_index_mapping"] = {
            i: chunk for i, chunk in zip(remaining_ids, remaining_text_chunks)
        }
        store["metadata_index_mapping"] = {
            i: metadata for i, metadata in zip(remaining_ids, remaining_metadata)
        }
        store["nodes"] = remaining_nodes
        store["index"] = index

        logging.info(f"Nodes {ids_to_delete} deleted and index updated for store: {collection_name} successfully.")
        logging.warning(f"Node ID(s) {missing_ids} does not exist in the collection {collection_name}.")
        logging.info(f"Remaining Node ID(s): {store["category_index_mapping"].keys()}")

    def add_files(self, collection_name: str, files: List[VSFile]) -> None:
        """
        Adds file nodes to the specified store by extracting nodes from the files and adding them to the index.

        :param collection_name: Name of the store to update.
        :param files: List of VSFile objects whose nodes are to be added.
        """
        logging.info(f"Adding files to store: {collection_name}")
        all_nodes = []

        for file in files:
            all_nodes.extend(file.nodes)

        self.add_nodes(collection_name, all_nodes)

    def delete_files(self, collection_name: str, files: List[VSFile]) -> None:
        """
        Deletes file nodes from the specified store by removing nodes corresponding to the given files.

        :param collection_name: Name of the store to update.
        :param files: List of VSFile objects whose nodes are to be deleted.
        """
        logging.info(f"Deleting files from store: {collection_name}")
        node_ids_to_delete = []

        for file in files:
            for node in file.nodes:
                node_ids_to_delete.append(node.metadata.node_id)

        self.delete_nodes(collection_name, node_ids_to_delete)

    def _save_collection(self, collection_name: str) -> None:
        """
        Helper function to save the updated store back to the file system.

        :param collection_name: Name of the store to save.
        """
        store_path = os.path.join(self.base_index_path, collection_name)
        store = self.collections[collection_name]

        # Save category index mapping to file
        with open(os.path.join(store_path, "category_index_mapping.pkl"), "wb") as f:
            pickle.dump(store["category_index_mapping"], f)

        # Save metadata nodes to file
        with open(os.path.join(store_path, "metadata_index_mapping.pkl"), "wb") as f:
            pickle.dump(store["metadata_index_mapping"], f)

        # Save nodes to file
        with open(os.path.join(store_path, "nodes.pkl"), "wb") as f:
            pickle.dump(store["nodes"], f)

        faiss.write_index(store["index"], os.path.join(store_path, "index"))
        logging.info(f"Store {collection_name} saved successfully.")
