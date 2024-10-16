# 🏴‍☠️ Overview 🏴‍☠️ 

`rakam_systems` is a Python library that provides a framework to easily build & deploy AI and Generative AI systems. 

You can build any System by combining **Components**, where you can either use some from our library or completely customise them. Both ways, they come with a suite of features built for production such as integrated evaluation and automatic deployment on your preferred cloud solution. We like to think of it as the child between [Haystack](https://github.com/deepset-ai/haystack) & [Terraform-OpenTofu](https://github.com/opentofu/opentofu).

## 🥵 Problem Statement

Building custom AI and Gen AI systems can be challenging due to the need for flexibility, scalability, and production-readiness. Developers often face problems like:

- **Complexity**: Creating AI systems from scratch is complex, especially when combining different technologies for model management, data processing, and integration.
  
- **Scalability**: Ensuring that AI systems can handle large-scale data and provide efficient, real-time responses.
  
- **Integration**: Standardizing and fluid data communication between the different core components of an AI System, especially when deployed on different servers.
  
- **Maintenance & Updates**: The AI landscape evolves rapidly, and maintaining systems with the latest models and technologies is challenging, stressful and costly.

`rakam_systems` addresses these challenges by offering a flexible, lean framework that helps developers build AI systems efficiently, while minimizing code maintenance overhead and focusing on ease of deployment.

## ✨ Key Features

- **Modular Framework**: `rakam_systems` is a framework for building AI and Gen AI systems, with **Components** serving as the core building blocks.
  
- **Customizability**: Designed to provide robust tools for developing custom Gen AI solutions. Many classes are abstract, offering flexibility while keeping the codebase streamlined by limiting predefined functionality to common use cases.
  
- **Production-Ready**: Built for scalability and ease of deployment:
  - Libraries are chosen for their efficiency and scalability.
  - Components exchange data in a structured way, facilitating API integration.
  - Includes Docker/Django API templates for easy deployment: [Service Template](https://github.com/Rakam-AI/rakam-systems-service-template).
    
- **Lean Design**: Focused on minimizing breaking changes and ensuring code fluidity.
  
- **Best-in-Class Supporting Tools & Approaches**: We select the best libraries and technical approaches for each specific task to keep the codebase lean and manageable, addressing the challenge of evolving technologies. We welcome contributions to improve these approaches and are open to new ideas.
  
- **Selected Libraries**:
  - **Best LLM**: OpenAI has the best models in the world and we've chosen it as the main LLM API [OpenAI](https://github.com/openai/openai-python) 
  - **EU LLM**: Mistral AI is the best European model provider and will have lasting conformity to the AI Act. [Mistral AI](https://github.com/mistralai/client-python)
  - **Transformers & Models**: Hugging Face was chosen for its extensive support for a wide range of pre-trained models and its active community. [Hugging Face (HF)](https://github.com/huggingface/transformers)
  - **Vector Stores**: FAISS was selected for its efficiency and scalability in managing large-scale vector similarity searches. [FAISS](https://github.com/facebookresearch/faiss)
  - **File storage**: While you can work with local files, we allow users to work with buckets using the S3 framework. [S3](https://github.com/facebookresearch/faiss)
 
- **Engine Update**: We also deploy regular **Engine Updates** to ensure that the library stays current with the latest advancements in AI, minimizing maintenance challenges.

## Use Cases

With `rakam_systems`, you can build:

- **Retrieval-Augmented Generation (RAG) Systems**: Combine vector retrieval with LLM prompt generation for enriched responses. [Learn more](https://rsdocs.readthedocs.io/en/latest/usage.html#retrieval-augmented-generation-rag)
  
- **Agent Systems**: Create modular agents that perform specific tasks using LLMs. *Link to come*
  
- **Chained Gen AI Systems**: Develop systems that chain multiple AI tasks together for complex workflows. *Link to come*
  
- **Search Engines**: Develop search engines based on fine-tunned embeddings models on any modality ( text, sound or video ). *Link to come*
  
- **Any Custom AI System**: Use components to create any AI solution tailored to your needs.

## Installation

To install `rakam_systems`, clone the repository and install it in editable mode to include all dependencies:

```bash
git clone <repository_url> rakam_systems
cd rakam_systems
pip install -e .
```

### Dependencies

- `faiss`
- `sentence-transformers`
- `pandas`
- `openai`
- `pymupdf`
- `playwright`
- `joblib`
- `requests`

## Examples

Check out the following links for detailed examples of what you can build using `rakam_systems`:

- **RAG Systems**: [RAG Documentation](https://rsdocs.readthedocs.io/en/latest/usage.html#retrieval-augmented-generation-rag)
- **Vector Search**: [Vector Store Documentation](https://rsdocs.readthedocs.io/en/latest/usage.html#creating-vector-stores)
- **Agent Systems**: *Link to come*
- **Chained Gen AI Systems**: *Link to come*

## Core Components

`rakam_systems` provides several core components to facilitate building AI systems:

- **Vector Stores**: Manage and query vector embeddings for fast retrieval.
- **Content Extraction**: Extract data from PDFs, URLs, and JSON files.
- **Node Processing**: Split content into smaller, manageable chunks.
- **Modular Agents**: Implement custom tasks such as classification, prompt generation, and RAG.

For more details on how to use each of these components, please refer to the [documentation here](https://rsdocs.readthedocs.io/en/latest/usage.html).

## Contributing

We welcome contributions! To contribute:

1. **Fork the Repository**: Start by forking the `rakam_systems` repository to your GitHub account.
2. **Clone the Forked Repository**: Clone the forked repository to your local machine:
   ```bash
   git clone <forked_repository_url> rakam_systems
   cd rakam_systems
   ```
3. **Install in Editable Mode**: Install `rakam_systems` in editable mode to make development easier:
   ```bash
   pip install -e .
   ```
4. **Create a Branch**: Create a feature branch (`git checkout -b feature-branch`).
5. **Make Changes**: Implement your changes and commit them with a meaningful message (`git commit -m 'Add new feature'`).
6. **Push the Branch**: Push your changes to your feature branch (`git push origin feature-branch`).
7. **Submit a Pull Request**: Go to the original `rakam_systems` repository on GitHub and submit a pull request for review.

For more details, refer to the [Contribution Guide](https://rsdocs.readthedocs.io/en/latest/usage.html).

## License

This project is licensed under the Apache-2.0 license.

## Support

For any issues, questions, or suggestions, please contact [mohammed@rakam.ai](mailto:mohammed@rakam.ai).
