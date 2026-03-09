import json

import pytest

from rakam_systems_vectorstore.config import (
    DatabaseConfig,
    EmbeddingConfig,
    IndexConfig,
    SearchConfig,
    VectorStoreConfig,
    load_config,
)


def test_embedding_config_defaults():
    cfg = EmbeddingConfig()
    assert cfg.model_type == "sentence_transformer"
    assert cfg.batch_size == 128
    assert cfg.normalize is True


def test_embedding_config_custom():
    cfg = EmbeddingConfig(model_type="openai", model_name="text-embedding-3-small", batch_size=32)
    assert cfg.model_type == "openai"
    assert cfg.model_name == "text-embedding-3-small"


def test_embedding_config_openai_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = EmbeddingConfig(model_type="openai")
    assert cfg.api_key == "sk-test"


def test_embedding_config_cohere_reads_env(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "cohere-key")
    cfg = EmbeddingConfig(model_type="cohere")
    assert cfg.api_key == "cohere-key"


def test_database_config_defaults():
    cfg = DatabaseConfig()
    assert cfg.host == "localhost"
    assert cfg.port == 5432
    assert cfg.database == "vectorstore_db"
    assert cfg.pool_size == 10


def test_database_config_env_override(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "myhost")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "mydb")
    monkeypatch.setenv("POSTGRES_USER", "myuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "mypass")
    cfg = DatabaseConfig()
    assert cfg.host == "myhost"
    assert cfg.port == 5433
    assert cfg.database == "mydb"
    assert cfg.user == "myuser"
    assert cfg.password == "mypass"


def test_database_config_connection_string():
    cfg = DatabaseConfig(
        host="host",
        port=5432,
        database="db",
        user="user",
        password="pass",
    )
    conn_str = cfg.to_connection_string()
    assert "postgresql://user:pass@host:5432/db" == conn_str


def test_search_config_defaults():
    cfg = SearchConfig()
    assert cfg.similarity_metric == "cosine"
    assert cfg.default_top_k == 5
    assert cfg.enable_hybrid_search is True


def test_search_config_validate_valid():
    cfg = SearchConfig()
    cfg.validate()


def test_search_config_invalid_metric():
    cfg = SearchConfig(similarity_metric="invalid")
    with pytest.raises(ValueError, match="Invalid similarity metric"):
        cfg.validate()


def test_search_config_invalid_alpha():
    cfg = SearchConfig(hybrid_alpha=1.5)
    with pytest.raises(ValueError, match="hybrid_alpha"):
        cfg.validate()


def test_search_config_invalid_top_k():
    cfg = SearchConfig(default_top_k=0)
    with pytest.raises(ValueError, match="default_top_k"):
        cfg.validate()


def test_search_config_invalid_keyword_algo():
    cfg = SearchConfig(keyword_ranking_algorithm="unknown")
    with pytest.raises(ValueError, match="Invalid keyword ranking algorithm"):
        cfg.validate()


def test_search_config_invalid_bm25_k1():
    cfg = SearchConfig(bm25_k1=-1.0)
    with pytest.raises(ValueError, match="bm25_k1"):
        cfg.validate()


def test_search_config_invalid_bm25_b():
    cfg = SearchConfig(bm25_b=1.5)
    with pytest.raises(ValueError, match="bm25_b"):
        cfg.validate()


def test_index_config_defaults():
    cfg = IndexConfig()
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 50
    assert cfg.enable_parallel_processing is False


def test_vectorstore_config_defaults():
    cfg = VectorStoreConfig()
    assert cfg.name == "pg_vector_store"
    assert cfg.enable_caching is True
    assert isinstance(cfg.embedding, EmbeddingConfig)
    assert isinstance(cfg.database, DatabaseConfig)
    assert isinstance(cfg.search, SearchConfig)
    assert isinstance(cfg.index, IndexConfig)


def test_vectorstore_config_from_dict():
    d = {
        "name": "custom_store",
        "embedding": {"model_type": "openai"},
        "database": {"host": "db_host"},
        "search": {"default_top_k": 10},
        "index": {"chunk_size": 256},
        "enable_caching": False,
    }
    cfg = VectorStoreConfig.from_dict(d)
    assert cfg.name == "custom_store"
    assert cfg.embedding.model_type == "openai"
    assert cfg.database.host == "db_host"
    assert cfg.search.default_top_k == 10
    assert cfg.index.chunk_size == 256
    assert cfg.enable_caching is False


def test_vectorstore_config_to_dict():
    cfg = VectorStoreConfig(name="test")
    d = cfg.to_dict()
    assert d["name"] == "test"
    assert "embedding" in d
    assert "database" in d
    assert "search" in d
    assert "index" in d


def test_vectorstore_config_validate_valid():
    cfg = VectorStoreConfig()
    cfg.validate()


def test_vectorstore_config_validate_invalid_batch_size():
    cfg = VectorStoreConfig()
    cfg.embedding.batch_size = 0
    with pytest.raises(ValueError, match="embedding.batch_size"):
        cfg.validate()


def test_vectorstore_config_validate_invalid_chunk_size():
    cfg = VectorStoreConfig()
    cfg.index.chunk_size = 0
    with pytest.raises(ValueError, match="index.chunk_size"):
        cfg.validate()


def test_vectorstore_config_validate_invalid_overlap():
    cfg = VectorStoreConfig()
    cfg.index.chunk_size = 100
    cfg.index.chunk_overlap = 100
    with pytest.raises(ValueError, match="chunk_overlap must be less than"):
        cfg.validate()


def test_vectorstore_config_save_and_load_yaml(tmp_path):
    cfg = VectorStoreConfig(name="yaml_test", enable_caching=False)
    yaml_path = str(tmp_path / "config.yaml")
    cfg.save_yaml(yaml_path)
    loaded = VectorStoreConfig.from_yaml(yaml_path)
    assert loaded.name == "yaml_test"
    assert loaded.enable_caching is False


def test_vectorstore_config_save_and_load_json(tmp_path):
    cfg = VectorStoreConfig(name="json_test", cache_size=500)
    json_path = str(tmp_path / "config.json")
    cfg.save_json(json_path)
    loaded = VectorStoreConfig.from_json(json_path)
    assert loaded.name == "json_test"
    assert loaded.cache_size == 500


def test_vectorstore_config_from_yaml_not_found():
    with pytest.raises(FileNotFoundError):
        VectorStoreConfig.from_yaml("/nonexistent/path.yaml")


def test_vectorstore_config_from_json_not_found():
    with pytest.raises(FileNotFoundError):
        VectorStoreConfig.from_json("/nonexistent/path.json")


def test_load_config_defaults():
    cfg = load_config()
    assert isinstance(cfg, VectorStoreConfig)
    assert cfg.name == "pg_vector_store"


def test_load_config_from_dict():
    d = {"name": "from_dict", "enable_caching": False}
    cfg = load_config(d)
    assert cfg.name == "from_dict"
    assert cfg.enable_caching is False


def test_load_config_from_yaml(tmp_path):
    import yaml
    data = {"name": "yaml_load", "log_level": "DEBUG"}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(data))
    cfg = load_config(str(p))
    assert cfg.name == "yaml_load"


def test_load_config_from_json(tmp_path):
    data = {"name": "json_load"}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(data))
    cfg = load_config(str(p))
    assert cfg.name == "json_load"


def test_load_config_auto_detect_unknown_ext(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text("[settings]")
    with pytest.raises(ValueError, match="Cannot auto-detect"):
        load_config(str(p))


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/no/such/file.yaml")
