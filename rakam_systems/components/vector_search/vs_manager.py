import os
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
        data_processor: DataProcessor = DataProcessor(),
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

    def create_collection_from_vsfiles(
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
        self.vector_store.create_collections_from_files(collection_files)
        
        logging.info(f"Successfully injected {len(vs_files)} files into collection: {collection_name}")
        return vs_files

    def add_vsfiles(
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
        self.vector_store.add_files(collection_name, vs_files)
        
        logging.info(f"Successfully added {len(vs_files)} files to collection: {collection_name}")

    def delete_vsfiles(
        self,
        vs_files: List[VSFile],
        collection_name: str = "base"
    ) -> None:
        """
        Delete specified documents from a collection.

        Args:
            vs_files (List[VSFile]): List of VSFile objects to delete
            collection_name (str): Name of the collection to delete from
        """
        logging.info(f"Deleting {len(vs_files)} files from collection: {collection_name}")
        self.vector_store.delete_files(collection_name, vs_files)

    def call_main(
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
        return self.create_collection_from_vsfiles(directory_path, collection_name)

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
        
        vs_files = self.create_collection_from_vsfiles(test_directory,collection_name=inject_collection_name)
        logging.info(f"Processed {len(vs_files)} test files")

        return (vs_files[0].nodes[0].content)

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