import logging
from typing import Dict, List, Any
import os
import pickle
import faiss
import numpy as np
import time

from sentence_transformers import SentenceTransformer

from rakam_systems.core import NodeMetadata, Node
from rakam_systems.system_manager import SystemManager
from rakam_systems.components.component import Component
from rakam_systems.core import VSFile

logging.basicConfig(level=logging.INFO)

class VSManager(Component):
    """
    A class that combines document processing and vector store functionality to ingest documents
    into a vector store for later retrieval.
    """
    def __init__(
        self,
        base_index_path: str,
        embedding_model: str,
        system_manager: SystemManager,
    ) -> None:
        """
        Initialize the DocumentInjector with the necessary components.

        Args:
            base_index_path (str): Base path to store the FAISS indexes
            embedding_model (str): Name of the embedding model to use
            initializing (bool): Whether the vector store is being initialized
        """
        if system_manager is None:
            raise ValueError("system_manager is required for NeedsSystemManagerComponent")
        self.system_manager = system_manager
        self.base_index_path = base_index_path
        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.collections = {}
        if not os.path.exists(self.base_index_path):
            os.makedirs(self.base_index_path)

        self._load_vector_store()

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
        required_files = [
            "index", 
            "category_index_mapping.pkl", 
            "metadata_index_mapping.pkl", 
            "nodes.pkl", 
            "embeddings_index_mapping.pkl"
        ]
        
        if not all(os.path.exists(os.path.join(store_path, file)) for file in required_files):
            logging.warning(f"Store at {store_path} is missing required files and will not be loaded.")
            return None
        
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

    def _create_collections_from_files(self, collection_files: Dict[str, List[VSFile]]) -> None:
        """
        Creates FAISS indexes from dictionaries of store names and VSFile objects.

        :param collection_files: Dictionary where keys are store names and values are lists of VSFile objects.
        """
        for collection_name, files in collection_files.items():
            self._create_collection_from_files(collection_name, files)

    def _create_collection_from_files(self, collection_name: str, files: List[VSFile]) -> None:
        """
        Creates FAISS indexes from dictionaries of store names and VSFile objects.

        :param collection_files: Dictionary where keys are store names and values are lists of VSFile objects.
        """
        logging.info(f"Creating FAISS index for store: {collection_name}")
        text_chunks = []
        metadata = []
        nodes = []
        
        node_id = 0
        for file in files:
            for node in file.nodes:
                node.metadata.node_id = node_id
                node_id += 1
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

    def _create_collection_from_directory(
        self,
        directory_path: str,
        collection_name: str = "base"
    ) -> List[VSFile]:
        """
        Process documents from a directory and inject them into the vector store.

        Args:
            directory_path (str): Path to the directory containing documents
            collection_name (str): Name of the collection to store the documents in

        Returns:
            List[VSFile]: List of processed VSFile objects
        """
        logging.info(f"Processing documents from directory: {directory_path}")
        
        input_data = {"directory": directory_path, "collection_name": collection_name}
        serialized_files = self.system_manager.execute_component_function(
            component_name="DataProcessor",
            function_name="process_from_directory",
            input_data=input_data
        )
        vs_files = [VSFile.from_dict(data=file) for file in serialized_files]
        
        if not vs_files:
            logging.warning(f"No files were processed from directory: {directory_path}")
            return []

        # Create collection in vector store
        collection_files = {collection_name: vs_files}
        self._create_collections_from_files(collection_files)
        
        logging.info(f"Successfully injected {len(vs_files)} files into collection: {collection_name}")
        return vs_files

    def _add_files(self, collection_name: str, files: List[VSFile]) -> None:
        """
        Adds file nodes to the specified store by extracting nodes from the files and adding them to the index.

        :param collection_name: Name of the store to update.
        :param files: List of VSFile objects whose nodes are to be added.
        """
        logging.info(f"Adding files to store: {collection_name}")
        all_nodes = []

        for file in files:
            all_nodes.extend(file.nodes)

        self._add_nodes(collection_name, all_nodes)

    def _add_vsfiles_to_collection_from_directory(
        self,
        directory_path: str,
        collection_name: str = "base"
    ) -> List[VSFile]:
        """
        Process additional documents and add them to an existing collection.

        Args:
            directory_path (str): Path to the directory containing new documents
            collection_name (str): Name of the collection to add the documents to

        Returns:
            List[VSFile]: List of processed VSFile objects
        """
        logging.info(f"Processing additional documents from directory: {directory_path}")
        
        input_data = {"directory": directory_path, "collection_name": collection_name}
        serialized_files = self.system_manager.execute_component_function(
            component_name="DataProcessor",
            function_name="process_from_directory",
            input_data=input_data
        )
        vs_files = [VSFile.from_dict(data=file) for file in serialized_files]
        
        if not vs_files:
            logging.warning(f"No files were processed from directory: {directory_path}")
            return []

        # Add to existing collection
        self._add_files(collection_name, vs_files)
        
        logging.info(f"Successfully added {len(vs_files)} files to collection: {collection_name}")

        return vs_files

    def _delete_files(self, collection_name: str, files: List[VSFile]) -> None:
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

        self._delete_nodes(collection_name, node_ids_to_delete)

    def call_create_from_directory(
        self,
        directory_path: str,
        collection_name: str = "base"
    ) -> List[VSFile]:
        """
        Main method to process and inject documents.

        Args:
            directory_path (str): Path to the directory containing documents
            collection_name (str): Name of the collection to store the documents in

        Returns:
            List[VSFile]: List of processed VSFile objects
        """
        vs_files = self._create_collection_from_directory(directory_path, collection_name)
        serialized_files = [vs_file.to_dict() for vs_file in vs_files]
        return serialized_files
    
    def call_create_from_file(
            self,
            file_path: str,
            collection_name: str = "base",
            file_uuid = None
    ) -> List[VSFile]:
        """
        Main method to process and inject a single document.

        Args:
            file_path (str): Path to the file to process
            collection_name (str): Name of the collection to store the document in

        Returns:
            List[VSFile]: List of processed VSFile objects
        """
        input_data = {"file_path": file_path, "file_uuid": file_uuid}
        serialized_files = self.system_manager.execute_component_function(
            component_name="DataProcessor",
            function_name="process_from_file",
            input_data=input_data
        )
        vs_files = [VSFile.from_dict(data=file) for file in serialized_files]
        
        if not vs_files:
            logging.warning(f"No files were processed from file: {file_path}")
            return []

        # Create collection in vector store
        collection_files = {collection_name: vs_files}
        self._create_collections_from_files(collection_files)
        
        logging.info(f"Successfully injected file into collection: {collection_name}")

        return serialized_files
    
    def call_add_from_directory(
        self,
        directory_path: str,
        collection_name: str = "base"
    ) -> List[VSFile]:
        """
        Main method to process and add documents to an existing collection.

        Args:
            directory_path (str): Path to the directory containing new documents
            collection_name (str): Name of the collection to add the documents to

        Returns:
            List[VSFile]: List of processed VSFile objects
        """
        vs_files = self._add_vsfiles_to_collection_from_directory(directory_path, collection_name)
        serialized_files = [vs_file.to_dict() for vs_file in vs_files]
        return serialized_files
    
    def call_add_from_file(
            self,
            file_path: str,
            collection_name: str = "base"
    ) -> List[VSFile]:
        """
        Main method to process and add a single document to an existing collection.

        Args:
            file_path (str): Path to the file to add
            collection_name (str): Name of the collection to add the document to

        Returns:
            List[VSFile]: List of processed VSFile objects
        """
        input_data = {"file_path": file_path}
        serialized_files = self.system_manager.execute_component_function(
            component_name="DataProcessor",
            function_name="process_from_file",
            input_data=input_data
        )
        vs_files = [VSFile.from_dict(data=file) for file in serialized_files]
        
        if not vs_files:
            logging.warning(f"No files were processed from file: {file_path}")
            return []

        # Add to existing collection
        self._add_files(collection_name, vs_files)
        
        logging.info(f"Successfully added file to collection: {collection_name}")

        return serialized_files
    
    def call_delete_vsfiles(
        self,
        collection_name: str,
        files: List[VSFile]
    ) -> None:
        """
        Main method to delete documents from an existing collection.

        Args:
            collection_name (str): Name of the collection to delete the documents from
            files (List[VSFile]): List of VSFile objects to delete
        """
        self._delete_files(collection_name, files)

    def call_main(self, **kwargs):
        return super().call_main(**kwargs)
    
    def test(
        self,
        test_directory: str = "data/files",
        inject_collection_name: str = "base"
    ) -> None:
        """
        Test the DocumentInjector functionality.

        Args:
            test_directory (str): Directory containing test documents
            test_query (str): Query to test search functionality
        """
        logging.info("Running DocumentInjector test")
        
        vs_files = self._create_collection_from_directory(test_directory,collection_name=inject_collection_name)
        serialized_files = [vs_file.to_dict() for vs_file in vs_files]
        logging.info(f"Processed {len(vs_files)} test files")

        return serialized_files
    