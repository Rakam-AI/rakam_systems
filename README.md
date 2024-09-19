
### Overview

This Python library provides an integrated solution for creating **Vector Stores** and **Retrieval-Augmented Generation (RAG)** systems. It offers a modular architecture to build and manage vector-based search systems using embeddings from models such as **SentenceTransformers** and **FAISS** for efficient retrieval. Additionally, it provides flexible interfaces to handle content extraction, node processing, and LLM-driven generation tasks like text classification and RAG-enabled prompt generation.

### Features

-   **Vector Store Management**: Create, manage, and search through vector stores using embeddings for fast retrieval.
-   **Retrieval-Augmented Generation (RAG)**: Combines vector store retrieval and large language model (LLM) generation.
-   **Content Extraction**: Extracts content from PDF, URLs, and JSON files, with different parsers for various formats.
-   **Node Processing**: Processes text data by splitting content based on headers or character count to optimize storage and retrieval.
-   **Modular Action-Based Agents**: Supports classification, prompt generation, and custom RAG generation via agents.

----------

### Key Components

1.  **Vector Stores**  
    Vector Stores allow you to create, store, and query vector embeddings of content (e.g., text). This is useful for information retrieval tasks, where content is indexed using vector representations.
    
2.  **Content Extraction**  
    Extracts content from various file formats such as PDFs, URLs, and JSON. This is a preprocessing step to convert unstructured content into nodes that are used by the Vector Store.
    
3.  **RAG Generation**  
    Combines information retrieval from Vector Stores with LLM prompt generation to produce contextually enriched responses using search results from the Vector Store.
    
4.  **Agents**  
    Agents encapsulate different actions, such as query classification, prompt generation, or RAG, to perform specific tasks using an LLM model.
    

----------

### Installation

Before getting started, make sure you have installed all dependencies. You can install them using `pip` or by adding the necessary packages to your environment.

bash

Copy code

`pip install -r requirements.txt` 

**Dependencies**:

-   `faiss`
-   `sentence-transformers`
-   `pandas`
-   `openai`
-   `pymupdf`
-   `playwright`
-   `joblib`
-   `requests`

----------

### Usage Guide

#### 1. Vector Store Creation

The `VectorStores` class manages creating and searching vector embeddings.

python

Copy code

`from rakam_systems.vector_store import VectorStores
from rakam_systems.core import VSFile, Node, NodeMetadata

# Initialize Vector Store
vector_store = VectorStores(base_index_path="path/to/index", embedding_model="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store from nodes
nodes = [
    Node(content="Text data 1", metadata=NodeMetadata(source_file_uuid="file1", position=1)),
    Node(content="Text data 2", metadata=NodeMetadata(source_file_uuid="file2", position=2))
]
vector_store.create_from_nodes("store_name", nodes)` 

#### 2. Retrieval-Augmented Generation (RAG)

RAG enables combining vector store search with prompt generation using an LLM model.

python

Copy code

`from rakam_systems.generation.agents import RAGGeneration, Agent
from rakam_systems.vector_store import VectorStores

# Initialize Vector Store and Agent
vector_store = VectorStores(base_index_path="path/to/index", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
agent = Agent(model="gpt-3.5-turbo", api_key="your_openai_api_key")

# Create RAG Action
rag_action = RAGGeneration(agent, sys_prompt="System Prompt", prompt="User Prompt", vector_stores=vector_store)

# Execute RAG Action
query = "What is the capital of France?"
result = rag_action.execute(query=query)
print(result)` 

#### 3. Content Extraction

The library provides several content extractors for PDFs, URLs, and JSON files.

python

Copy code

`from rakam_systems.ingestion.content_extractors import PDFContentExtractor

# Initialize PDF Content Extractor
pdf_extractor = PDFContentExtractor(parser_name="SimplePDFParser", output_format="markdown")

# Extract content from a PDF file
vs_files = pdf_extractor.extract_content(source="path/to/file.pdf")` 

#### 4. Node Processing

Node processors split content into chunks for better handling in Vector Stores.

python

Copy code

`from rakam_systems.ingestion.node_processors import CharacterSplitter

# Initialize Node Processor
splitter = CharacterSplitter(max_characters=512, overlap=50)

# Process Nodes
splitter.process(vs_file)` 

#### 5. Classification with Vector Stores

Use vector stores to classify queries based on predefined triggers.

python

Copy code

`from rakam_systems.generation.agents import ClassifyQuery
import pandas as pd

# Sample Data for Classification
trigger_queries = pd.Series(["What is the capital of", "Tell me about"])
class_names = pd.Series(["Geography", "General Info"])

# Initialize Classification Action
classifier = ClassifyQuery(agent=None, trigger_queries=trigger_queries, class_names=class_names)

# Classify a new query
result = classifier.execute("What is the capital of France?")
print(result)` 

----------

### How It Works

1.  **Data Ingestion**: Content is extracted from sources like PDFs, JSON, or URLs.
2.  **Node Processing**: The extracted content is processed into smaller chunks called **nodes**.
3.  **Embedding Generation**: Text content is converted into vector embeddings using SentenceTransformers.
4.  **Vector Store Creation**: The vector embeddings are indexed and stored for fast retrieval using FAISS.
5.  **LLM Augmentation**: When a query is received, the Vector Store retrieves relevant content, and an LLM uses that content to generate a response.

----------

### Example Use Cases

1.  **Intelligent Search Systems**: Build search systems that return relevant documents, paragraphs, or even sentences based on user queries.
2.  **Query Classification**: Automatically classify user queries into categories using vector-based classification.
3.  **RAG Systems**: Use vector search to retrieve relevant documents and use that content to generate augmented answers using LLMs.

----------

### Advanced Features

1.  **Streaming Generation**: Generate responses in real-time using the `stream` option for LLM models.
2.  **Multi-Format Content Extraction**: Extract content from various sources like PDFs, JSON, and websites using different parsers.
3.  **Custom Node Processing**: Customize how the content is chunked or split using different node processors.
4.  **Action-Oriented Agents**: Extend the agents with custom actions for classification, retrieval, or generation tasks.

----------

### Contributing

We welcome contributions! To contribute:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature-branch`).
3.  Commit your changes (`git commit -m 'Add new feature'`).
4.  Push to the branch (`git push origin feature-branch`).
5.  Create a pull request.

----------

### License

This project is licensed under the MIT License.

----------

### Support

For any issues, questions, or suggestions, please contact mohammed@rakam.ai .