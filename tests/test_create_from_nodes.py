from rakam_systems.core import Node, NodeMetadata
from rakam_systems.components.vector_search.vector_store import VectorStore

BASE_INDEX_PATH = "vector_store_index_1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def setup_vector_store():
    vector_store = VectorStore(base_index_path=BASE_INDEX_PATH, embedding_model=EMBEDDING_MODEL)
    return vector_store

def create_nodes(texts):
    return [
        Node(content=text, metadata=NodeMetadata(source_file_uuid=f"file_{i}", position=i))
        for i, text in enumerate(texts)
    ]

def search_vector_store(vector_store, query, number=3):
    results, _ = vector_store.search(collection_name="base", query=query, number=number)
    return results

def print_search_results(query, results):
    print(f"Search Results for query '{query}':")
    for id_, (metadata, suggestion_text, distance) in results.items():
        print(f"ID: {id_}, Suggestion: {suggestion_text}, Metadata: {metadata}, Distance: {distance}")
    print("\n")

def main():
    texts = [
        "This is the first document.",
        "Here is another document.",
        "Final document in the list."
    ]

    vector_store = setup_vector_store()
    nodes = create_nodes(texts)
    vector_store.create_from_nodes(nodes=nodes)

    query = "first document"
    results = search_vector_store(vector_store, query)

    print_search_results(query, results)

if __name__ == "__main__":
    main()
