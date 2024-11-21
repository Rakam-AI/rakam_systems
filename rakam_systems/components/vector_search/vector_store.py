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

    # def load_collection(self, store_path: str) -> Dict[str, Any]:
    #     """
    #     Loads a single vector store from the specified directory.

    #     :param store_path: Path to the store directory.
    #     :return: Dictionary containing the store's index, nodes, and metadata.
    #     """
    #     store = {}
    #     store["index"] = faiss.read_index(os.path.join(store_path, "index"))
    #     with open(
    #         os.path.join(store_path, "category_index_mapping.pkl"), "rb"
    #     ) as f:
    #         store["category_index_mapping"] = pickle.load(f)
    #     with open(
    #         os.path.join(store_path, "metadata_index_mapping.pkl"), "rb"
    #     ) as f:
    #         store["metadata_index_mapping"] = pickle.load(f)
    #     with open(os.path.join(store_path, "nodes.pkl"), "rb") as f:
    #         store["nodes"] = pickle.load(f)
    #     logging.info(f"Store loaded successfully from {store_path}.")
    #     return store
    def load_collection(self, store_path: str) -> Dict[str, Any]:
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

    def get_index_copy(self, store: Dict[str, Any]) -> faiss.IndexIDMap:
        """
        Creates a copy of the index from the store and returns it.
        """
        assert len(store["embeddings"]) == len(store["category_index_mapping"]), "Mismatch between embeddings and category index mapping."

        category_index_mapping = store["category_index_mapping"]
        data_embeddings = np.array(list(store["embeddings"].values()))
        index_copy = faiss.IndexIDMap(faiss.IndexFlatIP(data_embeddings.shape[1]))
        faiss.normalize_L2(data_embeddings)
        index_copy.add_with_ids(data_embeddings, np.array(list(category_index_mapping.keys())))

        return index_copy

    def search(
        self, collection_name: str, query: str, distance_type="cosine", number=5, meta_data_filters: List = None
    ) -> dict:
        """
        Searches the specified collection for the closest embeddings to the query.

        :param collection_name: Name of the collection to search.
        :param query: Query string to search for.
        :param distance_type: Type of distance metric to use (default is cosine).
        :param number: Number of results to return (default is 5).
        :param meta_data_filters: List of Node IDs to filter the search results.
        """
        logging.info(f"Searching in collection: {collection_name} for query: '{query}'")

        # Step 1: Retrieve the collection
        store = self.collections.get(collection_name)
        if not store:
            raise ValueError(f"No store found with name: {collection_name}")

        index_copy = self.get_index_copy(store)    

        # Step 2: Apply metadata filters if provided
        if meta_data_filters:
            logging.info(f"Applying metadata filters: {meta_data_filters}")
            
            all_ids = store["category_index_mapping"].keys()
            logging.info(f"Total IDs in the index: {all_ids}")
            
            ids_to_remove = list(all_ids - set(meta_data_filters))
            logging.info(f"IDs to remove: {ids_to_remove}")

            # filtered_index = faiss.clone_index(store["index"])
            filtered_index = index_copy
            logging.info(f"Original index size: {filtered_index.ntotal}")

            filtered_index.remove_ids(np.array(ids_to_remove))
            logging.info(f"Filtered index size: {filtered_index.ntotal}")
        else:
            # No filters provided; use the original index
            logging.info("No metadata filters provided. Using the entire index for search.")
            filtered_index = index_copy

        # Step 3: Generate the query embedding
        query_embedding = self.predict_embeddings(query)
        logging.info(f"Query embedding shape: {query_embedding.shape}")
        if distance_type == "cosine":
            faiss.normalize_L2(query_embedding)

        # Step 4: Perform the search
        logging.info("Performing search on the index...")
        D, I = filtered_index.search(query_embedding, number)
        logging.debug(f"Search distances: {D}")
        logging.debug(f"Search indices: {I}")

        if I.shape[1] == 0 or np.all(I == -1):
            logging.error("Search returned no results.")
            return {}, []

        # Step 5: Prepare search results
        suggested_nodes = []
        seen_texts = set()
        valid_suggestions = {}
        count = 0

        for i, id_ in enumerate(I[0]):
            if count >= number:
                break
            if id_ != -1 and id_ in store["category_index_mapping"]:
                suggestion_text = store["category_index_mapping"][id_]
                node_metadata = store["metadata_index_mapping"][id_]
                for node in store["nodes"]:
                    if node.metadata.node_id == id_:
                        suggested_nodes.append(node)
                if suggestion_text not in seen_texts:
                    seen_texts.add(suggestion_text)
                    valid_suggestions[str(id_)] = (
                        node_metadata,
                        suggestion_text,
                        float(D[0][i]),
                    )
                    count += 1

        logging.info(f"Final search results: {valid_suggestions}")
        
        return valid_suggestions, suggested_nodes

    def get_embeddings(self, sentences: List[str], parallel: bool = True, batch_size: int = 8) -> np.ndarray:
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

    # def _create_and_save_index(
    #     self,
    #     collection_name: str,
    #     nodes: List[Any],
    #     text_chunks: List[str],
    #     metadata: List[Dict[str, Any]],
    #     ) -> None:
    #     """
    #     Helper function to create and save a FAISS index.

    #     :param collection_name: Name of the store to create.
    #     :param nodes: List of nodes.
    #     :param text_chunks: List of text chunks to encode and index.
    #     :param metadata: List of metadata associated with the text chunks.
    #     """
    #     # Check if the list of nodes or text_chunks is empty
    #     if not nodes or not text_chunks:
    #         logging.warning(f"Cannot create FAISS index for store '{collection_name}' because nodes or text_chunks are empty.")
    #         self.collections[collection_name] = {
    #         "index": None,
    #         "nodes": [],
    #         "category_index_mapping": None,
    #         "metadata_index_mapping": None,
    #     }
    #         return

    #     store_path = os.path.join(self.base_index_path, collection_name)
    #     if not os.path.exists(store_path):
    #         os.makedirs(store_path)

    #     # Get embeddings for the text chunks
    #     data_embeddings = self.get_embeddings(sentences=text_chunks, parallel=False)
    #     category_index_mapping = dict(zip(range(len(text_chunks)), text_chunks))

    #     # Save category index mapping to file
    #     with open(os.path.join(store_path, "category_index_mapping.pkl"), "wb") as f:
    #         pickle.dump(category_index_mapping, f)

    #     # Save nodes to file
    #     with open(os.path.join(store_path, "nodes.pkl"), "wb") as f:
    #         pickle.dump(nodes, f)

    #     # Create FAISS index and add embeddings
    #     index = faiss.IndexIDMap(faiss.IndexFlatIP(data_embeddings.shape[1]))
    #     faiss.normalize_L2(data_embeddings)
    #     index.add_with_ids(
    #         data_embeddings, np.array(list(category_index_mapping.keys()))
    #     )
    #     faiss.write_index(index, os.path.join(store_path, "index"))

    #     # Save metadata index mapping to file
    #     metadata_index_mapping = dict(zip(range(len(text_chunks)), metadata))
    #     with open(os.path.join(store_path, "metadata_index_mapping.pkl"), "wb") as f:
    #         pickle.dump(metadata_index_mapping, f)

    #     # Update the collections dictionary
    #     self.collections[collection_name] = {
    #         "index": index,
    #         "nodes": nodes,
    #         "category_index_mapping": category_index_mapping,
    #         "metadata_index_mapping": metadata_index_mapping,
    #     }

    #     logging.info(
    #         f"FAISS index for store {collection_name} created and saved successfully."
    #     )
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
        
        assert len(nodes) == len(text_chunks) == len(metadata), "Length of nodes, text_chunks, and metadata should be equal."

        store_path = os.path.join(self.base_index_path, collection_name)
        if not os.path.exists(store_path):
            os.makedirs(store_path)

        # Get embeddings for the text chunks
        data_embeddings = self.get_embeddings(sentences=text_chunks, parallel=False)
        category_index_mapping = dict(zip(range(len(text_chunks)), text_chunks))

        # Update the node_id in the metadata for metadata_index_mapping
        for i, meta in enumerate(metadata):
            meta['node_id'] = i
        logging.info(f"Assigned node IDs to metadata successfully. For example: {metadata[0]['node_id']}")

        # Update the node_id in the metadata in the nodes
        for i, node in enumerate(nodes):
            node.metadata.node_id = i

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


    def add_nodes(self, collection_name: str, nodes: List[Node]) -> None:
        """
        Adds nodes to an existing store and updates the index.

        :param collection_name: Name of the store to update.
        :param nodes: List of nodes to be added.
        """
        logging.info(f"Adding nodes to store: {collection_name}")

        if not nodes:
            logging.warning("No nodes to add.")
            return

        store = self.collections.get(collection_name)
        if not store:
            raise ValueError(f"No store found with name: {collection_name}")
        
        assert len(store["category_index_mapping"]) == len(store["metadata_index_mapping"]) == len(store["embeddings"]) == len(store["nodes"]) , "Mismatch between mappings and embeddings."
        assert store["category_index_mapping"].keys() == store["metadata_index_mapping"].keys() == store["embeddings"].keys() == {node.metadata.node_id for node in store["nodes"]}, "Mismatch between mappings and embeddings."
        assert all(node.metadata.node_id not in {n.metadata.node_id for n in store["nodes"]} for node in nodes), "Duplicate node IDs detected in the new nodes."

        # Get the existing text chunks from the given nodes
        new_text_chunks = [node.content for node in nodes]
        
        # Get embeddings for the new text chunks
        new_embeddings = self.get_embeddings(sentences=new_text_chunks, parallel=False)

        existing_ids = set(store["category_index_mapping"].keys())
        max_existing_id = max(existing_ids) if existing_ids else -1
        new_ids = []
        next_id = max_existing_id + 1
        for _ in range(len(new_text_chunks)):
            while next_id in existing_ids:
                next_id += 1
            new_ids.append(next_id)
            next_id += 1

        logging.info(f"New IDs: {new_ids}")
        logging.info(f"Existing IDs: {existing_ids}")
        logging.info(f"New text chunks count: {len(new_text_chunks)}")

        # # Get the new Mapping Indices for the new nodes
        # new_ids = list(range(
        #     len(store["category_index_mapping"]),
        #     len(store["category_index_mapping"]) + len(new_text_chunks),
        # ))

        # Check if the length of new embeddings and new Indices are equal
        assert len(new_embeddings) == len(new_ids), "Mismatch between new embeddings and IDs."

        # Add the new embeddings to the existing index
        store["index"].add_with_ids(new_embeddings, np.array(list(new_ids)))

        # Store new embeddings persistently
        for idx, embedding in zip(new_ids, new_embeddings):
            store["embeddings"][idx] = embedding        

        # Update the node_ids in metadata for the new nodes
        for idx, node in enumerate(nodes):
            node.metadata.node_id = new_ids[idx]
        store["nodes"].extend(nodes)

        # Update the node_id in metadata index mapping from the new nodes
        new_metadata = [
            {
                "node_id": node.metadata.node_id,
                "source_file_uuid": node.metadata.source_file_uuid,
                "position": node.metadata.position,
                "custom": node.metadata.custom,
            }
            for node in nodes
        ]

        # Update the mappings
        store["category_index_mapping"].update(dict(zip(new_ids, new_text_chunks)))
        store["metadata_index_mapping"].update(dict(zip(new_ids, new_metadata)))

        assert len(store["category_index_mapping"]) == len(store["metadata_index_mapping"]) == len(store["embeddings"]) == len(store["nodes"]) , "Mismatch between mappings and embeddings."
        assert store["category_index_mapping"].keys() == store["metadata_index_mapping"].keys() == store["embeddings"].keys() == {node.metadata.node_id for node in store["nodes"]}, "Mismatch between mappings and embeddings."

        # Save updated store
        self.collections[collection_name] = store
        self._save_collection(collection_name)

        self.load_collection(os.path.join(self.base_index_path, collection_name))
        saved_store = self.collections[collection_name]
        assert len(saved_store["category_index_mapping"]) == len(saved_store["metadata_index_mapping"]) == len(saved_store["embeddings"]) == len(saved_store["nodes"]), "Mismatch in saved store mappings."
        assert store["category_index_mapping"].keys() == store["metadata_index_mapping"].keys() == store["embeddings"].keys() == {node.metadata.node_id for node in store["nodes"]}, "Mismatch between mappings and embeddings."
        

    def delete_nodes(self, collection_name: str, node_ids: List[int]) -> None:
        """
        Deletes nodes from an existing store and updates the index using remove_ids method.

        :param collection_name: Name of the store to update.
        :param node_ids: List of node IDs to be deleted.
        """
        logging.info(f"Deleting nodes {node_ids} from store: {collection_name}")

        store = self.collections.get(collection_name)
        if not store:
            raise ValueError(f"No store found with name: {collection_name}")

        assert len(store["category_index_mapping"]) == len(store["metadata_index_mapping"]) == len(store["embeddings"]) == len(store["nodes"]) , "Mismatch between mappings and embeddings."
        assert store["category_index_mapping"].keys() == store["metadata_index_mapping"].keys() == store["embeddings"].keys() == {node.metadata.node_id for node in store["nodes"]}, "Mismatch between mappings and embeddings."

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

        assert len(store["category_index_mapping"]) == len(store["metadata_index_mapping"]) == len(store["embeddings"]) == len(store["nodes"]) , "Mismatch between mappings and embeddings."
        assert store["category_index_mapping"].keys() == store["metadata_index_mapping"].keys() == store["embeddings"].keys() == {node.metadata.node_id for node in store["nodes"]}, "Mismatch between mappings and embeddings."

        # Save the updated store
        self.collections[collection_name] = store
        self._save_collection(collection_name)

        self.load_collection(os.path.join(self.base_index_path, collection_name))
        saved_store = self.collections[collection_name]
        assert len(saved_store["category_index_mapping"]) == len(saved_store["metadata_index_mapping"]) == len(saved_store["embeddings"]) == len(saved_store["nodes"]), "Mismatch in saved store mappings."
        assert store["category_index_mapping"].keys() == store["metadata_index_mapping"].keys() == store["embeddings"].keys() == {node.metadata.node_id for node in store["nodes"]}, "Mismatch between mappings and embeddings."

        logging.info(f"Nodes {ids_to_delete} deleted and index updated for store: {collection_name} successfully.")
        if missing_ids:
            logging.warning(f"Node ID(s) {missing_ids} do not exist in the collection {collection_name}.")
        logging.info(f"Remaining Node ID(s): {store['category_index_mapping'].keys()}")




    # def delete_nodes(self, collection_name: str, node_ids: List[int]) -> None:
    #     """
    #     Deletes nodes from an existing store and updates the index.

    #     :param collection_name: Name of the store to update.
    #     :param node_ids: List of node IDs to be deleted.
    #     """
    #     logging.info(f"Deleting nodes from store: {collection_name}")

    #     store = self.collections.get(collection_name)
    #     if not store:
    #         raise ValueError(f"No store found with name: {collection_name}")

    #     # Filter out the nodes to be deleted
    #     remaining_ids = [
    #         id_ for id_ in store["category_index_mapping"].keys() if id_ not in node_ids
    #     ]

    #     remaining_text_chunks = [
    #         store["category_index_mapping"][id_] for id_ in remaining_ids
    #     ]
    #     remaining_metadata = [
    #         store["metadata_index_mapping"][id_] for id_ in remaining_ids
    #     ]
    #     remaining_nodes = [store["nodes"][id_] for id_ in remaining_ids]

    #     # Re-create the index with remaining nodes
    #     self._create_and_save_index(
    #         collection_name, remaining_nodes, remaining_text_chunks, remaining_metadata
    #     )
    # def delete_nodes(self, collection_name: str, node_ids: List[int]) -> None:
    #     """
    #     Deletes nodes from an existing store and updates the index without re-creating it from scratch.

    #     :param collection_name: Name of the store to update.
    #     :param node_ids: List of node IDs to be deleted.
    #     """
    #     logging.info(f"Deleting nodes {node_ids} from store: {collection_name}")

    #     store = self.collections.get(collection_name)
    #     if not store:
    #         raise ValueError(f"No store found with name: {collection_name}")

    #     existed_ids = store["category_index_mapping"].keys()
    #     logging.info(f"Existed IDs before deletion: {existed_ids}")

    #     missing_ids = []
    #     ids_to_delete = []
    #     for node_id in node_ids:
    #         if node_id not in existed_ids:
    #             missing_ids.append(node_id)
    #         else:
    #             ids_to_delete.append(node_id)

    #     if ids_to_delete:
    #         # Get the embeddings from the store (already loaded during collection load)
    #         data_embeddings = store["embeddings"]

    #         # Create a list of remaining IDs after deletion
    #         remaining_ids = [
    #             id_ for id_ in store["category_index_mapping"].keys() if id_ not in ids_to_delete
    #         ]
    #         remaining_text_chunks = [
    #             store["category_index_mapping"][id_] for id_ in remaining_ids
    #         ]
    #         remaining_metadata = [
    #             store["metadata_index_mapping"][id_] for id_ in remaining_ids
    #         ]
    #         remaining_nodes = [store["nodes"][id_] for id_ in remaining_ids]

    #         # Remove embeddings for the nodes to be deleted
    #         delete_indices = [i for i, id_ in enumerate(store["category_index_mapping"].keys()) if id_ in ids_to_delete]
    #         remaining_embeddings = np.delete(data_embeddings, delete_indices, axis=0)

    #         # Create a new FAISS index with the remaining embeddings
    #         index = faiss.IndexIDMap(faiss.IndexFlatIP(remaining_embeddings.shape[1]))
    #         faiss.normalize_L2(remaining_embeddings)
    #         index.add_with_ids(remaining_embeddings, np.array(remaining_ids))

    #         # Update the store mappings, embeddings, and index
    #         store["category_index_mapping"] = dict(zip(remaining_ids, remaining_text_chunks))
    #         store["metadata_index_mapping"] = dict(zip(remaining_ids, remaining_metadata))
    #         store["nodes"] = remaining_nodes
    #         store["index"] = index
    #         store["embeddings"] = remaining_embeddings

    #         # Save the updated collection
    #         self._save_collection(collection_name)

    #         logging.info(f"Nodes {ids_to_delete} deleted and index updated for store: {collection_name} successfully.")
    #         logging.warning(f"Node ID(s) {missing_ids} do not exist in the collection {collection_name}.")
    #         logging.info(f"Remaining Node ID(s): {remaining_ids}")
    #     else:
    #         logging.warning(f"No valid IDs found to delete in collection {collection_name}.")


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
        if len(store["nodes"]) == 0:
            logging.warning(f"Cannot sve FAISS index for store {collection_name} because nodes are empty.")
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

        # Save embeddings to file
        with open(os.path.join(store_path, "embeddings_index_mapping.pkl"), "wb") as f:
            pickle.dump(store["embeddings"], f)

        faiss.write_index(store["index"], os.path.join(store_path, "index"))
        logging.info(f"Store {collection_name} saved successfully.")



 