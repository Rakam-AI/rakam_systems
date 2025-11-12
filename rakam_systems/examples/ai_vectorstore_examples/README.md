# AI Vector Store Examples

This directory contains examples demonstrating different vector store implementations in `rakam_systems`.

## Examples

### 1. PostgreSQL Vector Store (`postgres_vectorstore_example.py`)

Demonstrates using PostgreSQL with pgvector extension for production-ready, persistent vector storage.

**Features:**
- ✅ Persistent storage (survives restarts)
- ✅ ACID transactions
- ✅ Hybrid search (vector + full-text)
- ✅ Built-in re-ranking
- ✅ Metadata filtering
- ✅ Scalable for production

**Quick Start:**

```bash
# Run everything with one command
./run_postgres_example.sh
```

**Manual Setup:**

```bash
# 1. Start PostgreSQL
docker-compose up -d

# 2. Run the example (from rakam_systems package root)
cd ../../..
export POSTGRES_PASSWORD=postgres
python -m examples.ai_vectorstore_examples.postgres_vectorstore_example
```

**Managing PostgreSQL:**

```bash
# Check status
docker-compose ps

# Stop PostgreSQL (container stopped, data persists)
docker-compose stop

# Start PostgreSQL again
docker-compose start

# Restart PostgreSQL
docker-compose restart

# Stop and remove containers (data still persists in volume)
docker-compose down

# Stop and remove EVERYTHING including all data
docker-compose down -v

# View logs
docker-compose logs postgres
docker-compose logs -f postgres  # Follow logs in real-time
```

### 2. FAISS Vector Store (`basic_faiss_example.py`)

Demonstrates in-memory vector search using FAISS for fast prototyping and development.

**Features:**
- ✅ No database required
- ✅ Extremely fast in-memory search
- ✅ Easy to get started
- ✅ Perfect for prototyping

**Quick Start:**

```bash
# From rakam_systems package root
cd ../../..
python -m examples.ai_vectorstore_examples.basic_faiss_example
```

## Configuration

### Environment Variables (PostgreSQL)

The PostgreSQL example uses these environment variables:

- `POSTGRES_DB`: Database name (default: `vectorstore_db`)
- `POSTGRES_USER`: Database user (default: `postgres`)
- `POSTGRES_PASSWORD`: Database password (default: `postgres`)
- `POSTGRES_HOST`: Database host (default: `localhost`)
- `POSTGRES_PORT`: Database port (default: `5432`)

### Django Settings

The PostgreSQL example uses a minimal Django configuration located in `django_settings.py`. This is required for Django ORM to manage the database models.

## Files in This Directory

- **`postgres_vectorstore_example.py`** - PostgreSQL vector store example
- **`basic_faiss_example.py`** - FAISS vector store example
- **`django_settings.py`** - Minimal Django configuration for PostgreSQL
- **`docker-compose.yml`** - PostgreSQL + pgvector container configuration
- **`run_postgres_example.sh`** - Helper script to run PostgreSQL example
- **`README.md`** - This file

## Choosing Between PostgreSQL and FAISS

**Use PostgreSQL when:**
- You need persistent storage
- You're building a production system
- You need ACID transactions
- You want hybrid search capabilities
- You need to scale horizontally

**Use FAISS when:**
- You're prototyping
- You need maximum speed for in-memory search
- You don't need persistence
- You want minimal setup

## Troubleshooting

### PostgreSQL Connection Issues

If you get "connection refused" errors:

1. Check if PostgreSQL is running:
   ```bash
   docker ps | grep postgres
   ```

2. Check logs:
   ```bash
   docker-compose logs postgres
   ```

3. Verify the port is not already in use:
   ```bash
   lsof -i :5432
   ```

### Migration Issues

If you get migration errors, try:

```bash
cd ../../..
export POSTGRES_PASSWORD=postgres
export DJANGO_SETTINGS_MODULE=examples.ai_vectorstore_examples.django_settings
python -m django migrate
```

## Additional Resources

- [Main rakam_systems README](../../../README.md)
- [Vector Store Documentation](https://rsdocs.readthedocs.io/en/latest/usage.html#creating-vector-stores)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

