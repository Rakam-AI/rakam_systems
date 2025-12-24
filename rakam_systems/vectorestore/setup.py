"""
Setup file for ai_vectorstore standalone submodule.

This allows the ai_vectorstore to be installed independently or as part of rakam-systems.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="rakam-systems-ai-vectorstore",
    version="1.1.0",
    author="Mohamed Hilel, Peng Zheng",
    author_email="mohammedjassemhlel@gmail.com, pengzheng990630@outlook.com",
    description="Modular vector store and RAG components for semantic search and retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rakam-AI/rakam_systems",
    project_urls={
        "Documentation": "https://github.com/Rakam-AI/rakam_systems",
        "Source": "https://github.com/Rakam-AI/rakam_systems",
        "Issues": "https://github.com/Rakam-AI/rakam_systems/issues",
    },
    packages=find_packages(
        where="..",
        include=["rakam_system_vectorstore*", "rakam_systems.core.ai_core*"]
    ),
    package_dir={"": ".."},
    python_requires=">=3.10",
    install_requires=[
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "postgres": [
            "psycopg2-binary>=2.9.9",
            "django>=4.0.0",
        ],
        "faiss": [
            "faiss-cpu>=1.12.0",
        ],
        "local-embeddings": [
            "sentence-transformers>=5.1.0",
            "torch>=2.0.0",
        ],
        "openai": [
            "openai>=1.0.0",
        ],
        "cohere": [
            "cohere>=4.0.0",
        ],
        "loaders": [
            "python-magic>=0.4.27",
            "beautifulsoup4>=4.12.0",
            "python-docx>=1.2.0",
            "pymupdf>=1.24.0",
            "pymupdf4llm>=0.0.17",
        ],
        "all": [
            "psycopg2-binary>=2.9.9",
            "django>=4.0.0",
            "faiss-cpu>=1.12.0",
            "sentence-transformers>=5.1.0",
            "torch>=2.0.0",
            "openai>=1.0.0",
            "cohere>=4.0.0",
            "python-magic>=0.4.27",
            "beautifulsoup4>=4.12.0",
            "python-docx>=1.2.0",
            "pymupdf>=1.24.0",
            "pymupdf4llm>=0.0.17",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-django>=4.5.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="vector-store embeddings rag semantic-search pgvector faiss",
    include_package_data=True,
    package_data={
        "rakam_system_vectorstore": ["**/*.yaml", "**/*.yml", "**/*.json", "**/*.md"],
    },
)
