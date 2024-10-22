from rakam_systems.components.vector_search import VectorStore
from rakam_systems.components.agents.actions import RAGGeneration
from rakam_systems.components.agents.agents import Agent
from rakam_systems.core import Node, NodeMetadata


# Define the AgentRag class that inherits from the Agent class
class AgentRag(Agent):
    def __init__(self, model):
        super().__init__(model=model)  # Call the parent constructor to initialize the LLM

    # Implement the abstract choose_action method
    def choose_action(self, action_name: str):
        """
        Simple implementation of choose_action that returns the action
        based on the action name.
        """
        if action_name in self.actions:
            return self.actions[action_name]
        else:
            raise ValueError(f"Action {action_name} not found.")

def main():
    # Set the API key before starting
    # os.environ['OPENAI_API_KEY'] = 'your-api-key-here'  # Set your API key here
    # export OPENAI_API_KEY='your-api-key-here' # Or you can export API in your environment
    
    # Sample data to add to the vector store
    texts = [
        "The capital of France is Paris.",
        "The Eiffel Tower is in Paris.",
        "France is known for its cuisine and wine."
    ]
    metadata = [
        {"source": "Fact 1"},
        {"source": "Fact 2"},
        {"source": "Fact 3"}
    ]

    # Create nodes with content and metadata
    nodes = [Node(content=text, metadata=NodeMetadata(source_file_uuid=f"file_{i}", position=i, custom=metadata[i])) for i, text in enumerate(texts)]

    # Create a vector store and add the documents (nodes)
    vector_store = VectorStore(base_index_path="rag_tutorial_store", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    vector_store.create_collection_from_nodes(collection_name="facts_collection", nodes=nodes)

    # Initialize the RAGGeneration action
    sys_prompt = "You are a knowledgeable assistant."
    prompt = "User query: {query}\nRelevant information:\n{search_results}"

    # Create an instance of AgentRag with the desired LLM model (e.g., "gpt-4")
    agent_rag = AgentRag("gpt-4o-mini")

    # Register the RAGGeneration action in the agent's actions
    rag_action = RAGGeneration(
        agent=agent_rag,
        sys_prompt=sys_prompt,
        prompt=prompt,
        vector_stores=[vector_store]  # Assuming vector_store is defined earlier
    )

    # Add the RAGGeneration action to the agent
    agent_rag.add_action("rag_generation", rag_action)

    # Execute the RAGGeneration action via the agent
    query = "Where is the Eiffel Tower?"
    results = agent_rag.execute_action("rag_generation", query=query, collection_names=["facts_collection"], stream = False)

    # Print the generated response
    print("Generated Response:\n", results)

if __name__ == "__main__":
    main()
