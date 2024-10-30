import logging
import os
import pickle
import time
from typing import Any, Dict, List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rakam_systems.core import NodeMetadata, Node
from rakam_systems.components.component import Component

logging.basicConfig(level=logging.INFO)

class VectorStore(Component):
    """
    A class for managing collection-based vector stores using FAISS and SentenceTransformers.
    """

    def __init__(self, base_index_path: str, embedding_model: str, load_from_local: bool = True) -> None:
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

        if load_from_local : self._load_vector_store()

    def _load_vector_store(self) -> None:
        """
        Loads all collections from the base directory.
        """
        for collection_name in os.listdir(self.base_index_path):
            store_path = os.path.join(self.base_index_path, collection_name)
            if os.path.isdir(store_path):
                self.collections[collection_name] = self._load_collection(store_path)

    def _load_collection(self, store_path: str) -> Dict[str, Any]:
        """
        Loads a single vector store from the specified directory.

        :param store_path: Path to the store directory.
        :return: Dictionary containing the store's index, nodes, metadata, and embeddings.
        """
        store = {}
        store["index"] = faiss.read_index(os.path.join(store_path, "index"))
        with open(os.path.join(store_path, "category_index_mapping.pkl"), "rb") as f:
            store["category_index_mapping"] = pickle.load(f)
        with open(os.path.join(store_path, "metadata_index_mapping.pkl"), "rb") as f:
            store["metadata_index_mapping"] = pickle.load(f)
        with open(os.path.join(store_path, "nodes.pkl"), "rb") as f:
            store["nodes"] = pickle.load(f)
        with open(os.path.join(store_path, "embeddings_index_mapping.pkl"), "rb") as f:
            store["embeddings"] = pickle.load(f)


        logging.info(f"Store loaded successfully from {store_path}.")
        return store

    def _predict_embeddings(self, query: str) -> np.ndarray:
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

        query_embedding = self._predict_embeddings(query)
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

    def _get_embeddings(self, sentences: List[str], parallel: bool = True, batch_size: int = 8) -> np.ndarray:
        """
        Generates embeddings for a list of sentences.

        :param sentences: List of sentences to encode.
        :param parallel: Whether to use parallel processing (default is True).
        :return: Embedding vectors for the sentences.
        """
        logging.info(f"Generating embeddings for {len(sentences)} sentences.")
        print(f"Generating embeddings for {len(sentences)} sentences.")
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
        Helper function to create and save a FAISS index and embeddings.

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
                "embeddings": None  # No embeddings
            }
            return

        store_path = os.path.join(self.base_index_path, collection_name)
        if not os.path.exists(store_path):
            os.makedirs(store_path)

        # Get embeddings for the text chunks
        data_embeddings = self._get_embeddings(sentences=text_chunks, parallel=False)
        category_index_mapping = dict(zip(range(len(text_chunks)), text_chunks))

        # Save category index mapping to file
        with open(os.path.join(store_path, "category_index_mapping.pkl"), "wb") as f:
            pickle.dump(category_index_mapping, f)

        # Save nodes to file
        with open(os.path.join(store_path, "nodes.pkl"), "wb") as f:
            pickle.dump(nodes, f)

        # Save embeddings to file
        embeddings_index_mapping = dict(zip(range(len(data_embeddings)), data_embeddings))
        with open(os.path.join(store_path, "embeddings_index_mapping.pkl"), "wb") as f:
            pickle.dump(embeddings_index_mapping, f)

        # Create FAISS index and add embeddings
        index = faiss.IndexIDMap(faiss.IndexFlatIP(data_embeddings.shape[1]))
        faiss.normalize_L2(data_embeddings)
        index.add_with_ids(data_embeddings, np.array(list(category_index_mapping.keys())))
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
            "embeddings": embeddings_index_mapping  # Store the embeddings mapping
        }
        print((f"FAISS index and embeddings for store {collection_name} created and saved successfully."))
        logging.info(f"FAISS index and embeddings for store {collection_name} created and saved successfully.")

    def _add_nodes(self, collection_name: str, nodes: List[Node]) -> None:
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
        new_embeddings = self._get_embeddings(sentences=new_text_chunks, parallel=False)

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
        self.collections[collection_name] = store
        self._save_collection(collection_name)

    def _delete_nodes(self, collection_name: str, node_ids: List[int]) -> None:
        """
        Deletes nodes from an existing store and updates the index using remove_ids method.

        :param collection_name: Name of the store to update.
        :param node_ids: List of node IDs to be deleted.
        """
        logging.info(f"Deleting nodes {node_ids} from store: {collection_name}")

        store = self.collections.get(collection_name)
        if not store:
            raise ValueError(f"No store found with name: {collection_name}")

        existed_ids = set(store["category_index_mapping"].keys())
        logging.info(f"Existed IDs before deletion: {existed_ids}")

        missing_ids = []
        ids_to_delete = []
        
        # Find the valid ids to delete
        for node_id in node_ids:
            if node_id not in existed_ids:
                missing_ids.append(node_id)
            else:
                ids_to_delete.append(node_id)
        
        if not ids_to_delete:
            logging.warning(f"No valid IDs to delete for store: {collection_name}")
            return

        # Remove the IDs from the FAISS index using remove_ids method
        faiss_index = store["index"]
        
        logging.info(f"FAISS Index Size Before Deletion: {faiss_index.ntotal}")
        
        faiss_index.remove_ids(np.array(ids_to_delete))
        
        logging.info(f"FAISS Index Size After Deletion: {faiss_index.ntotal}")

        # Remove the nodes and mappings from the store
        store["category_index_mapping"] = {
            i: chunk for i, chunk in store["category_index_mapping"].items() if i not in ids_to_delete
        }
        store["metadata_index_mapping"] = {
            i: metadata for i, metadata in store["metadata_index_mapping"].items() if i not in ids_to_delete
        }
        
        # Filter the nodes based on the ID, not based on list index
        store["nodes"] = [node for node in store["nodes"] if node.metadata.node_id not in ids_to_delete]
        
        # Filter embeddings to remove those corresponding to deleted IDs
        store["embeddings"] = {
            i: emb for i, emb in store["embeddings"].items() if i not in ids_to_delete
        }

        # Save the updated store
        self.collections[collection_name] = store
        self._save_collection(collection_name)

        logging.info(f"Nodes {ids_to_delete} deleted and index updated for store: {collection_name} successfully.")
        logging.warning(f"Node ID(s) {missing_ids} does not exist in the collection {collection_name}.")
        logging.info(f"Remaining Node ID(s): {store['category_index_mapping'].keys()}")

    def _save_collection(self, collection_name: str) -> None:
        """
        Helper function to save the updated store back to the file system.

        :param collection_name: Name of the store to save.
        """
        store_path = os.path.join(self.base_index_path, collection_name)
        store = self.collections[collection_name]
        if len(store["nodes"]) == 0:
            logging.warning(f"Cannot save FAISS index for store {collection_name} because nodes are empty.")
            return
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
    
    def _get_nodes(self, collection_name: str) -> List[Node]:
        """
        Retrieves the nodes from the specified collection.

        :param collection_name: Name of the collection to retrieve nodes from.
        :return: List of nodes in the collection.
        """
        store = self.collections.get(collection_name)
        if not store:
            raise ValueError(f"No store found with name: {collection_name}")

        return store["nodes"]
    
    def call_get_nodes(self, collection_name: str) -> List[Node]:
        """
        Main method to retrieve nodes from a collection.

        :param collection_name: Name of the collection to retrieve nodes from.
        :return: List of nodes in the collection.
        """
        nodes = self._get_nodes(collection_name)
        serialzied_nodes = [node.to_dict() for node in nodes]
        response = {
            "length": len(serialzied_nodes),
            "nodes": serialzied_nodes
        }
        return response
    
    def call_main(self, collection_name: str, query: str, distance_type="cosine", number=5) -> dict:
        # Perform the search
        valid_suggestions, suggested_nodes = self.search(collection_name, query, distance_type, number)
        
        # Format `valid_suggestions` for JSON compatibility
        formatted_suggestions = {
            suggestion_id: {
                "metadata": metadata,
                "text": text,
                "distance": distance
            } 
            for suggestion_id, (metadata, text, distance) in valid_suggestions.items()
        }
        
        # Return the formatted suggestions as a JSON-compatible dictionary
        return {
            "suggestions": formatted_suggestions,
            # "suggested_nodes": suggested_nodes 
        }

    def test(self, query = "This is the first document.") -> bool:
        """
        Method for testing the VectorStore.
        """
        logging.info("Running test for VectorStore.")
        texts = [
            "This is the first document.",
            "Here is another document.",
            "Final document in the list."
        ]

        nodes = [
            Node(content=text, metadata=NodeMetadata(source_file_uuid=f"file_{i}", position=i))
            for i, text in enumerate(texts)
        ]
        
        self.create_from_nodes(nodes)
        results,_ = self.call_main(collection_name = "base", query = query)
        
        return results["1"][1]

if __name__ == "__main__":
    vector_store = VectorStore(base_index_path="data/vector_stores_for_test/example_baseIDXpath", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    print(vector_store.test())
    
