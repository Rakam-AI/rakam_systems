# AI Vectorstore - Quick Install Reference

## 🚀 TL;DR - Get Started Fast

### As Part of Rakam Systems

```bash
cd app/rakam_systems
pip install -e ".[ai-vectorstore]"
```

**Usage:**

```python
from rakam_systems_vectorstore import ConfigurablePgVectorStore, Node, NodeMetadata

store = ConfigurablePgVectorStore(name="my_store")
store.setup()
store.get_or_create_collection("docs")
```

### Standalone Installation

```bash
cd app/rakam_systems/rakam_systems/ai_vectorstore
pip install -e ".[all]"
```

## 📦 What Gets Installed?

Installing `rakam-systems[ai-vectorstore]` includes:

✅ PostgreSQL backend (Django + pgvector)  
✅ FAISS in-memory store  
✅ SentenceTransformers embeddings  
✅ OpenAI & Cohere API support  
✅ Document loaders (PDF, DOCX, HTML)  
✅ Chunkers, retrievers, re-rankers  
✅ Configuration system

## 🎯 Installation Recipes

### Recipe 1: PostgreSQL Only

```bash
pip install -e ".[postgres,local-embeddings]"
```

### Recipe 2: FAISS + OpenAI

```bash
pip install -e ".[faiss,openai]"
```

### Recipe 3: Development Setup

```bash
pip install -e ".[all,dev]"
```

## 🔧 Environment Setup

```bash
# PostgreSQL with pgvector
docker run -d --name postgres-vectorstore \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=vectorstore_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Environment variables
export POSTGRES_HOST=localhost
export POSTGRES_DB=vectorstore_db
export OPENAI_API_KEY="your-key"  # if using OpenAI
```

## 📚 Next Steps

- **Full Guide**: [INSTALLATION.md](INSTALLATION.md)
- **Usage Examples**: [README.md](README.md)
- **Configuration**: [README.md#configuration](README.md#configuration)

## ❓ Quick Troubleshooting

**Import Error?**

```bash
pip install -e ".[ai-vectorstore]"  # From rakam_systems root
```

**Django Not Found?**

```bash
pip install -e ".[postgres]"
```

**PyTorch Too Large?**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

**Need more details?** See [INSTALLATION.md](INSTALLATION.md) for the complete guide.
