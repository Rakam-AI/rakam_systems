import os
from rakam_systems.core import VSFile, NodeMetadata, Node
from rakam_systems.ingestion.content_extractors import PDFContentExtractor
from rakam_systems.ingestion.data_processor import DataProcessor
from rakam_systems.vector_store import VectorStores
import pytest

# Fixture to set up the vector store for testing
@pytest.fixture
def vector_store_fixture():
    # Define the folder path where documents are stored
    folder_path = "/home/ubuntu/gitlab_work/rakam-systems/tmp_document"  # Replace with the actual folder path containing your documents
    store_name = "my_vector_store"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # You can replace with any SentenceTransformer model you prefer

    # Initialize the VectorStore
    vector_store = VectorStores(
        base_index_path="/home/ubuntu/gitlab_work/rakam-systems/tmp_indices",
        embedding_model=embedding_model,
    )

    # Step 1: Extract content from the folder using the data processor and PDF extractor
    processor = DataProcessor()
    vs_files = processor.process_files_from_directory(folder_path)

    # Step 2: Create the vector store from the VSFiles
    store_files = {store_name: vs_files}
    vector_store.create_from_files(store_files)

    return vector_store, store_name

# Test function using the fixture
def test_vector_store_operations(vector_store_fixture):
    vector_store, store_name = vector_store_fixture

    # Step 3: Test Add Files Functionality
    print("\n--- Testing: Add Files ---")
    additional_file_path = (
        "/home/ubuntu/gitlab_work/rakam-systems/new.pdf"  # Change to an actual path
    )
    additional_pdf_extractor = PDFContentExtractor(parser_name="SimplePDFParser")
    additional_vs_files = additional_pdf_extractor.extract_content(additional_file_path)
    vector_store.add_files(store_name, additional_vs_files)

    # Step 4: Test Search Functionality
    print("\n--- Testing: Search ---")
    query = "Your search query here"
    results, suggested_nodes = vector_store.search(
        store_name, query, distance_type="cosine", number=5
    )
    print(f"Search results: {results}")

    # Step 5: Test Delete Files Functionality
    print("\n--- Testing: Delete Files ---")
    vector_store.delete_files(store_name, additional_vs_files)

    # Step 6: Test Delete Nodes Functionality
    print("\n--- Testing: Delete Nodes ---")
    node_ids_to_delete = [0]  # Replace with actual node IDs to delete
    vector_store.delete_nodes(store_name, node_ids_to_delete)

    # Step 7: Test Add Nodes Functionality
    print("\n--- Testing: Add Nodes ---")
    new_node_metadata = NodeMetadata(source_file_uuid="new_file_uuid", position=1)
    new_node = Node(content="This is some new content", metadata=new_node_metadata)
    vector_store.add_nodes(store_name, [new_node])

    # Step 8: Reload the store and check if everything is loaded correctly
    print("\n--- Testing: Load All Stores ---")
    vector_store.load_all_stores()

    print("Store operations completed successfully!")
