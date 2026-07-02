"""Tests for ingestion<->runtime embedding consistency."""
import pytest

from rakam_systems_core.config_schema import EmbeddingRef
from rakam_systems_core.embedding_consistency import (
    EmbeddingConsistencyError,
    EmbeddingFingerprint,
    validate_embedding_consistency,
)


def _ref(ref: str, dim: int) -> EmbeddingRef:
    return EmbeddingRef(ref=ref, dim=dim)


class TestFingerprint:
    def test_from_ref_keeps_model_and_dim_only(self):
        fp = EmbeddingFingerprint.from_ref(_ref("azure:text-embedding-3-small", 1536))
        assert fp.model_name == "text-embedding-3-small"
        assert fp.dim == 1536
        assert fp.as_dict() == {"model_name": "text-embedding-3-small", "dim": 1536}


class TestValidate:
    def test_same_model_same_dim_passes(self):
        stored = EmbeddingFingerprint(model_name="text-embedding-3-small", dim=1536)
        validate_embedding_consistency(_ref("openai:text-embedding-3-small", 1536), stored)

    def test_provider_switch_alone_is_compatible(self):
        # Index built via OpenAI, queried via Azure -- same model, identical vectors.
        stored = EmbeddingFingerprint(model_name="text-embedding-3-small", dim=1536)
        validate_embedding_consistency(_ref("azure:text-embedding-3-small", 1536), stored)

    def test_model_mismatch_raises(self):
        stored = EmbeddingFingerprint(model_name="text-embedding-3-small", dim=1536)
        with pytest.raises(EmbeddingConsistencyError, match="does not match"):
            validate_embedding_consistency(_ref("openai:text-embedding-3-large", 1536), stored)

    def test_dim_mismatch_raises(self):
        stored = EmbeddingFingerprint(model_name="text-embedding-3-small", dim=1536)
        with pytest.raises(EmbeddingConsistencyError):
            validate_embedding_consistency(_ref("openai:text-embedding-3-small", 384), stored)

    def test_accepts_mapping(self):
        validate_embedding_consistency(
            _ref("openai:text-embedding-3-small", 1536),
            {"model_name": "text-embedding-3-small", "dim": 1536},
        )
        with pytest.raises(EmbeddingConsistencyError):
            validate_embedding_consistency(
                _ref("openai:text-embedding-3-small", 1536),
                {"model_name": "nomic-embed-text", "dim": 1536},
            )

    def test_none_stored_passes_as_unknown(self):
        # Legacy index with no fingerprint: cannot prove a mismatch, so pass.
        validate_embedding_consistency(_ref("openai:text-embedding-3-small", 1536), None)
