import pickle
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


def test_load_collection_reads_files(tmp_path):
    collection_path = tmp_path / "my_collection"
    collection_path.mkdir()

    fake_index = MagicMock()
    with patch(
        "rakam_systems_vectorstore.components.vectorstore.faiss_vector_store.faiss.read_index",
        return_value=fake_index,
    ):

        with open(collection_path / "category_index_mapping.pkl", "wb") as f:
            pickle.dump({1: "doc1"}, f)

        with open(collection_path / "metadata_index_mapping.pkl", "wb") as f:
            pickle.dump({1: {"content": "doc1"}}, f)

        with open(collection_path / "nodes.pkl", "wb") as f:
            pickle.dump(["node1"], f)

        with open(collection_path / "embeddings_index_mapping.pkl", "wb") as f:
            pickle.dump({1: [0.1, 0.2]}, f)

        store = FaissStore(
            base_index_path=str(tmp_path),
            initialising=True,
        )

        result = store.load_collection(str(collection_path))

        assert result["index"] == fake_index
        assert result["category_index_mapping"] == {1: "doc1"}
        assert result["metadata_index_mapping"] == {1: {"content": "doc1"}}
        assert result["nodes"] == ["node1"]
        assert result["embeddings"] == {1: [0.1, 0.2]}


def test_load_vector_store_loads_all_collections(tmp_path):
    (tmp_path / "collection1").mkdir()
    (tmp_path / "collection2").mkdir()

    with patch.object(
        FaissStore,
        "load_collection",
        return_value={"index": "mocked"},
    ) as mock_load_collection:

        store = FaissStore(
            base_index_path=str(tmp_path),
            initialising=True,
        )

        store.load_vector_store()

        assert "collection1" in store.collections
        assert "collection2" in store.collections
        assert mock_load_collection.call_count == 2


def test_predict_embeddings_local_model(store):
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.1, 0.2, 0.3]
    store.embedding_model = mock_model
    store.use_embedding_api = False

    result = store.predict_embeddings("hello")

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3)
    mock_model.encode.assert_called_once_with("hello")


def test_predict_embeddings_api_mode(tmp_path):
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.5, 0.6, 0.7])
    ]

    with patch(
        "rakam_systems_vectorstore.components.vectorstore.faiss_vector_store.OpenAI",
        return_value=mock_client,
    ):
        store = FaissStore(
            base_index_path=str(tmp_path),
            initialising=True,
            use_embedding_api=True,
        )

        result = store.predict_embeddings("query")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)
        mock_client.embeddings.create.assert_called_once()


def test_get_index_copy_creates_valid_index(store):
    store_data = {
        "category_index_mapping": {1: "doc1", 2: "doc2"},
        "embeddings": {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.0, 1.0], dtype=np.float32)
        },
    }

    index_copy = store.get_index_copy(store_data)

    assert index_copy.ntotal == 2


def test_get_index_copy_mismatch_assert(store):
    store_data = {
        "category_index_mapping": {1: "doc1"},
        "embeddings": {
            1: [1.0, 0.0],
            2: [0.0, 1.0],
        }
    }
    with pytest.raises(AssertionError):
        store.get_index_copy(store_data)
