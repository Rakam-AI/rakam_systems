#!/bin/bash
# Helper script to run the PostgreSQL vector store example

set -e

echo "============================================================"
echo "PostgreSQL Vector Store Example Setup"
echo "============================================================"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: docker-compose not found"
    echo "Please install Docker and docker-compose first"
    exit 1
fi

# Check if port 5432 is already in use
if lsof -Pi :5432 -sTCP:LISTEN -t >/dev/null 2>&1 || docker ps | grep -q "0.0.0.0:5432"; then
    echo ""
    echo "ℹ️  PostgreSQL is already running on port 5432"
    echo "   This is fine - we'll use the existing instance."
    POSTGRES_EXISTS=true
else
    echo ""
    echo "1. Starting PostgreSQL with pgvector..."
    docker-compose up -d
    POSTGRES_EXISTS=false
    
    # Wait for PostgreSQL to be ready
    echo "2. Waiting for PostgreSQL to be ready..."
    sleep 5
    
    # Check if PostgreSQL is healthy
    if docker-compose ps | grep -q "healthy"; then
        echo "✓ PostgreSQL is ready"
    else
        echo "⚠️  PostgreSQL may still be starting up..."
    fi
fi

# Run migrations if needed
echo ""
echo "3. Ensuring database migrations are applied..."
# Navigate to the rakam_systems package directory (2 levels up from script location)
# The structure is: rakam_systems/rakam_systems/examples/ai_vectorstore_examples/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RAKAM_PKG_ROOT="$SCRIPT_DIR/../.."

# Change to rakam_systems package root to run Python commands
cd "$RAKAM_PKG_ROOT"
export POSTGRES_PASSWORD=postgres
export DJANGO_SETTINGS_MODULE=examples.ai_vectorstore_examples.django_settings
python -m django migrate --no-input

# Run the example
echo ""
echo "4. Running the example..."
echo "============================================================"
python -m examples.ai_vectorstore_examples.postgres_vectorstore_example

echo ""
echo "============================================================"
echo "Example completed!"
echo ""
if [ "$POSTGRES_EXISTS" = false ]; then
    echo "To stop PostgreSQL: docker-compose stop"
    echo "To remove everything: docker-compose down -v"
else
    echo "Note: PostgreSQL was already running before this script."
    echo "To manage it, use docker-compose commands from the appropriate directory."
fi
echo "============================================================"

