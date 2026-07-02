"""Tests for the AI gateway model-reference schemas (ModelRef / EmbeddingRef)."""
import pytest

from rakam_systems_core.config_schema import EmbeddingRef, ModelRef


class TestModelRef:
    def test_parses_provider_and_model(self):
        ref = ModelRef(ref="azure:gpt-4.1-mini")
        assert ref.provider == "azure"
        assert ref.model_name == "gpt-4.1-mini"

    def test_only_first_colon_splits(self):
        # Bedrock inference-profile ids contain colons; the model portion keeps them.
        ref = ModelRef(ref="bedrock:amazon.nova-micro-v1:0")
        assert ref.provider == "bedrock"
        assert ref.model_name == "amazon.nova-micro-v1:0"

    def test_base_url_optional(self):
        assert ModelRef(ref="openai:gpt-4o").base_url is None
        ref = ModelRef(ref="openai:gemma2", base_url="http://localhost:11434/v1")
        assert ref.base_url == "http://localhost:11434/v1"

    def test_bare_model_name_rejected(self):
        with pytest.raises(ValueError):
            ModelRef(ref="gpt-4o")


class TestEmbeddingRef:
    def test_requires_dim(self):
        with pytest.raises(ValueError):
            EmbeddingRef(ref="openai:text-embedding-3-small")

    def test_dim_must_be_positive(self):
        with pytest.raises(ValueError):
            EmbeddingRef(ref="openai:text-embedding-3-small", dim=0)

    def test_valid(self):
        ref = EmbeddingRef(ref="azure:text-embedding-3-small", dim=1536)
        assert ref.dim == 1536
        assert ref.provider == "azure"
        assert ref.model_name == "text-embedding-3-small"

    def test_inherits_provider_prefix_rule(self):
        with pytest.raises(ValueError):
            EmbeddingRef(ref="text-embedding-3-small", dim=1536)
