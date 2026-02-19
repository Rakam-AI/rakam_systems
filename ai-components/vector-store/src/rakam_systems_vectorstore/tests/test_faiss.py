import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rakam_systems_vectorstore.components.vectorstore.faiss_vector_store import \
    FaissStore


@pytest.fixture
def temp_index_dir(tmp_path):
    """Temporary FAISS index directory."""
    return str(tmp_path / "faiss_indexes")


@pytest.fixture
def store(temp_index_dir):
    """
    Create FaissStore without loading existing indexes
    and without loading real embedding model.
    """
    with patch(
        "rakam_systems_vectorstore.components.vectorstore.faiss_vector_store.SentenceTransformer"
    ) as mock_st:
        mock_st.return_value = MagicMock()
        store = FaissStore(
            base_index_path=temp_index_dir,
            initialising=True,  # skip load_vector_store
        )
        yield store


def test_add_creates_collection_and_returns_ids(store):
    vectors = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    metadatas = [
        {"content": "doc1"},
        {"content": "doc2"},
    ]

    ids = store.add(vectors, metadatas)

    assert len(ids) == 2
    assert store.count() == 2
    assert "default" in store.collections


def test_add_increments_ids(store):
    vectors = [[0.1, 0.2, 0.3]]
    metadatas = [{"content": "doc"}]

    first_ids = store.add(vectors, metadatas)
    second_ids = store.add(vectors, metadatas)

    assert first_ids[0] != second_ids[0]
    assert store.count() == 2


def test_query_empty_collection_returns_empty(store):
    results = store.query([0.1, 0.2, 0.3])
    assert results == []


def test_query_returns_results(store):
    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    metadatas = [
        {"content": "A"},
        {"content": "B"},
    ]

    store.add(vectors, metadatas)

    query_vector = [1.0, 0.0, 0.0]
    results = store.query(query_vector, top_k=1)

    assert isinstance(results, list)
    assert len(results) <= 1


def test_count_multiple_collections(store):
    vectors = [[0.1, 0.2, 0.3]]
    metadatas = [{"content": "doc"}]

    store.add(vectors, metadatas)
    assert store.count() == 1


