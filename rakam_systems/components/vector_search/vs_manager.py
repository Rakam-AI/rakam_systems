import logging
from typing import Dict, List, Optional

from rakam_systems.components.data_processing.data_processor import DataProcessor
from rakam_systems.components.vector_search.vector_store import VectorStore
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
        vector_store: VectorStore,
        data_processor: DataProcessor,
    ) -> None:
        """
        Initialize the DocumentInjector with the necessary components.

        Args:
            base_index_path (str): Base path to store the FAISS indexes
            embedding_model (str): Name of the embedding model to use
            initializing (bool): Whether the vector store is being initialized
        """
        self.doc_processor = data_processor
        self.vector_store = vector_store

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

        self.vector_store._create_and_save_index(collection_name, nodes, text_chunks, metadata)

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
        
        # Process the documents
        vs_files = self.doc_processor.process_files_from_directory(directory_path)
        
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

        self.vector_store._add_nodes(collection_name, all_nodes)

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
        
        # Process the new documents
        vs_files = self.doc_processor.process_files_from_directory(directory_path)
        
        if not vs_files:
            logging.warning(f"No files were processed from directory: {directory_path}")
            return []

        # Add to existing collection
        self._add_files(collection_name, vs_files)
        
        logging.info(f"Successfully added {len(vs_files)} files to collection: {collection_name}")

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

        self.vector_store._delete_nodes(collection_name, node_ids_to_delete)

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
    
    def call_add_vsfiles(
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

if __name__ == "__main__":
    vector_store = VectorStore(base_index_path="data/vector_stores_for_test/attention_is_all_you_need", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    print("Before injection, len of collections is:")  
    print(len(vector_store.collections))

    # vs_manager = VSManager(vector_store=vector_store)

    # vs_manager.create_collection_from_vsfiles("data/files", collection_name="test")

    print("After injection, len of nodes in the  collections is:")
    print((len(vector_store.collections["test"]["nodes"])))
    print(len(vector_store.collections["test"]["category_index_mapping"]))

    # vs_manager.add_vsfiles("data/files", collection_name="test")

    # vector_store.search(collection_name = "test" ,query="What is attention mechanism?", number=2)