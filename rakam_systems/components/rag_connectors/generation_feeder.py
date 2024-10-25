import logging
from typing import List, Optional

from rakam_systems.components.connector import Connector
from rakam_systems.components.generation.generation import Generator
from rakam_systems.components.vector_search.vs_manager import VSManager

logging.basicConfig(level=logging.INFO)

class GenerationFeeder(Connector):
    """
    A class that feeds generated responses based on documents processed by VSManager.
    """

    def __init__(self, generator_model: str, vs_manager: VSManager):
        """
        Initialize the GenerationFeeder with the necessary components.

        Args:
            generator_model (str): The LLM model to use for text generation.
            vs_manager (VSManager): An instance of VSManager to manage documents.
        """
        self.generator = Generator(model=generator_model)
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
        self.vs_manager.inject_vsfiles(directory_path, collection_name)
        
        # Retrieve documents from the vector store for generation
        valid_suggestions, suggested_nodes = self.vs_manager.search_documents(query=query, collection_name=collection_name)
        
        # Extract relevant documents (assuming the results contain a list of document texts)
        relevant_documents = [doc[1] for doc in valid_suggestions.values()]

        # Setup the generator with the query and relevant documents
        self.generator.setup_generation(query=query, documents=relevant_documents)

        # Generate text using the LLM
        generated_response = self.generator.call_main()
        
        logging.info(f"Generated response for query '{query}': {generated_response}")
        return generated_response

    def call_main(self) -> str:
        """
        Calls the main generation method and returns the response.
        
        Returns:
            str: The generated response.
        """
        # This should call the appropriate method in the Generator to get the generated output.
        return self.generator.call_main()

    def test(self, query: str, test_directory: str) -> str:
        """
        Test the GenerationFeeder functionality.

        Args:
            query (str): Query to test the generation.
            test_directory (str): Directory containing test documents.

        Returns:
            str: The generated response.
        """
        logging.info("Running GenerationFeeder test")
        return self.feed_generation(query=query, directory_path=test_directory)

if __name__ == "__main__":
    # Example usage
    vs_manager = VSManager(base_index_path="vector_stores_for_test/attention_is_all_you_need")
    feeder = GenerationFeeder(generator_model="gpt-4o-mini", vs_manager=vs_manager)
    
    # Test the functionality
    test_query = "What is attention mechanism?"
    test_response = feeder.test(query=test_query, test_directory="data")
    print("Test Response:", test_response)
