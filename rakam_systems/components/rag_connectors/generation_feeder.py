import logging
from typing import List, Optional

from rakam_systems.components.connector import Connector
from rakam_systems.components.generation.generation import Generator
from rakam_systems.components.vector_search.vs_manager import VSManager
from rakam_systems.components.vector_search.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)

class GenerationFeeder(Connector):
    """
    A class that feeds generated responses based on documents processed by VSManager.
    """

    def __init__(self, generator_model, vs_manager: VSManager):
        """
        Initialize the GenerationFeeder with the necessary components.

        Args:
            generator_model (str): The LLM model to use for text generation.
            vs_manager (VSManager): An instance of VSManager to manage documents.
        """
        self.generator = generator_model
        self.vs_manager = vs_manager
        logging.info(f"GenerationFeeder initialized with model: {generator_model}")

    def feed_generation(self, query: str, directory_path: str, collection_name: str = "base") -> str:
        """
        Process documents and generate a response based on the provided query.

        Args:
            query (str): The query to generate a response for.
            directory_path (str): Path to the directory containing documents.
            collection_name (str): Name of the collection to store the documents in.

        Returns:
            str: The generated response.
        """
        # Inject documents and get processed files
        self.vs_manager._create_collection_from_directory(directory_path, collection_name)
        
        # Retrieve documents from the vector store for generation
        valid_suggestions, suggested_nodes = self.vs_manager.vector_store.search(query=query, collection_name=collection_name)
        
        # Extract relevant documents (assuming the results contain a list of document texts)
        relevant_documents = [doc[1] for doc in valid_suggestions.values()]

        # Setup the generator with the query and relevant documents
        self.generator.setup_generation(query=query, documents=relevant_documents)

        # Generate text using the LLM
        generated_response = self.generator.call_search_from_collection()
        
        logging.info(f"Generated response for query '{query}': {generated_response}")
        return generated_response

    def call_main(self) -> str:
        """
        Calls the main generation method and returns the response.
        
        Returns:
            str: The generated response.
        """
        vector_store = VectorStore(base_index_path="data/vector_stores_for_test/example_baseIDXpath", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        vs_manager = VSManager(vector_store=vector_store)
        self.vs_manager = vs_manager
        self.generator_model = "gpt-4o-mini"
        test_query = "What is attention mechanism?"
        return self.feed_generation(query=test_query, directory_path="data/files")
        
    def test(self, test_query = "What is attention mechanism?") -> str:
        """
        Test the GenerationFeeder functionality.

        Args:
            query (str): Query to test the generation.
            test_directory (str): Directory containing test documents.

        Returns:
            str: The generated response.
        """
        logging.info("Running GenerationFeeder test")
        vector_store = VectorStore(base_index_path="data/vector_stores_for_test/example_baseIDXpath", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        vs_manager = VSManager(vector_store=vector_store)
        self.vs_manager = vs_manager
        self.generator_model = "gpt-4o-mini"
        
        return self.feed_generation(query=test_query, directory_path="data/files")

if __name__ == "__main__":
    # # Example usage
    vector_store = VectorStore(base_index_path="data/vector_stores_for_test/example_baseIDXpath", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    vs_manager = VSManager(vector_store=vector_store)
    feeder = GenerationFeeder(generator_model="gpt-4o-mini", vs_manager=vs_manager)
    
    test_response = feeder.test()
    print("Test Response:", test_response)
