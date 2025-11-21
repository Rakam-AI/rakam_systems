"""
Unit tests for LLM Gateway system.

These tests verify the gateway factory, provider implementations,
and configuration system without making actual API calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel, Field

from ai_core.interfaces.llm_gateway import LLMGateway, LLMRequest, LLMResponse
from ai_agents.components.llm_gateway import (
    OpenAIGateway,
    MistralGateway,
    LLMGatewayFactory,
    get_llm_gateway,
)
from ai_core.config_schema import LLMGatewayConfigSchema


class TestLLMRequest:
    """Test LLMRequest data model."""
    
    def test_create_basic_request(self):
        """Test creating a basic request."""
        request = LLMRequest(
            user_prompt="Hello",
        )
        assert request.user_prompt == "Hello"
        assert request.system_prompt is None
        assert request.temperature is None
    
    def test_create_full_request(self):
        """Test creating a request with all fields."""
        request = LLMRequest(
            system_prompt="You are helpful",
            user_prompt="Hello",
            temperature=0.7,
            max_tokens=100,
            response_format="json",
            extra_params={"top_p": 0.9}
        )
        assert request.system_prompt == "You are helpful"
        assert request.user_prompt == "Hello"
        assert request.temperature == 0.7
        assert request.max_tokens == 100
        assert request.response_format == "json"
        assert request.extra_params["top_p"] == 0.9


class TestLLMResponse:
    """Test LLMResponse data model."""
    
    def test_create_response(self):
        """Test creating a response."""
        response = LLMResponse(
            content="Hello, world!",
            usage={"total_tokens": 10},
            model="gpt-4o",
            finish_reason="stop",
        )
        assert response.content == "Hello, world!"
        assert response.usage["total_tokens"] == 10
        assert response.model == "gpt-4o"
        assert response.finish_reason == "stop"


class TestLLMGatewayFactory:
    """Test LLM Gateway Factory."""
    
    def test_parse_model_string_with_provider(self):
        """Test parsing model string with provider."""
        provider, model = LLMGatewayFactory.parse_model_string("openai:gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"
    
    def test_parse_model_string_without_provider(self):
        """Test parsing model string without provider (defaults to OpenAI)."""
        provider, model = LLMGatewayFactory.parse_model_string("gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"
    
    def test_parse_model_string_mistral(self):
        """Test parsing Mistral model string."""
        provider, model = LLMGatewayFactory.parse_model_string("mistral:mistral-large-latest")
        assert provider == "mistral"
        assert model == "mistral-large-latest"
    
    def test_parse_invalid_provider(self):
        """Test parsing with invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMGatewayFactory.parse_model_string("invalid:model")
    
    def test_list_providers(self):
        """Test listing available providers."""
        providers = LLMGatewayFactory.list_providers()
        assert "openai" in providers
        assert "mistral" in providers
    
    def test_get_default_model(self):
        """Test getting default model for provider."""
        openai_model = LLMGatewayFactory.get_default_model("openai")
        assert openai_model == "gpt-4o"
        
        mistral_model = LLMGatewayFactory.get_default_model("mistral")
        assert mistral_model == "mistral-large-latest"
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_create_gateway_from_string(self):
        """Test creating gateway from model string."""
        gateway = LLMGatewayFactory.create_gateway("openai:gpt-4o-mini")
        assert isinstance(gateway, OpenAIGateway)
        assert gateway.model == "gpt-4o-mini"
        assert gateway.provider == "openai"
    
    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    def test_create_mistral_gateway(self):
        """Test creating Mistral gateway."""
        gateway = LLMGatewayFactory.create_gateway("mistral:mistral-small-latest")
        assert isinstance(gateway, MistralGateway)
        assert gateway.model == "mistral-small-latest"
        assert gateway.provider == "mistral"
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_create_gateway_from_config(self):
        """Test creating gateway from configuration dict."""
        config = {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.7,
        }
        gateway = LLMGatewayFactory.create_gateway_from_config(config)
        assert isinstance(gateway, OpenAIGateway)
        assert gateway.model == "gpt-4o"
        assert gateway.default_temperature == 0.7
    
    def test_create_gateway_missing_provider(self):
        """Test creating gateway without provider in config."""
        config = {"model": "gpt-4o"}
        with pytest.raises(ValueError, match="must specify 'provider'"):
            LLMGatewayFactory.create_gateway_from_config(config)
    
    @patch.dict("os.environ", {"DEFAULT_LLM_MODEL": "openai:gpt-4o-mini", "OPENAI_API_KEY": "test-key"})
    def test_get_default_gateway(self):
        """Test getting default gateway from environment."""
        gateway = LLMGatewayFactory.get_default_gateway()
        assert isinstance(gateway, OpenAIGateway)
        assert gateway.model == "gpt-4o-mini"
    
    def test_register_custom_provider(self):
        """Test registering a custom provider."""
        
        class CustomGateway(LLMGateway):
            def generate(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse(content="custom")
            
            def generate_structured(self, request, schema):
                pass
            
            def count_tokens(self, text: str, model=None) -> int:
                return len(text.split())
        
        LLMGatewayFactory.register_provider(
            "custom",
            CustomGateway,
            "custom-model-v1"
        )
        
        assert "custom" in LLMGatewayFactory.list_providers()
        assert LLMGatewayFactory.get_default_model("custom") == "custom-model-v1"


class TestOpenAIGateway:
    """Test OpenAI Gateway."""
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_initialize_gateway(self):
        """Test initializing OpenAI gateway."""
        gateway = OpenAIGateway(model="gpt-4o")
        assert gateway.model == "gpt-4o"
        assert gateway.provider == "openai"
        assert gateway.default_temperature == 0.7
    
    def test_initialize_without_api_key(self):
        """Test initializing without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key must be provided"):
                OpenAIGateway(model="gpt-4o")
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_build_messages(self):
        """Test building messages from request."""
        gateway = OpenAIGateway(model="gpt-4o")
        request = LLMRequest(
            system_prompt="You are helpful",
            user_prompt="Hello"
        )
        messages = gateway._build_messages(request)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_build_messages_no_system(self):
        """Test building messages without system prompt."""
        gateway = OpenAIGateway(model="gpt-4o")
        request = LLMRequest(user_prompt="Hello")
        messages = gateway._build_messages(request)
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("tiktoken.encoding_for_model")
    def test_count_tokens(self, mock_encoding):
        """Test token counting."""
        # Mock tiktoken encoding
        mock_enc = Mock()
        mock_enc.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_encoding.return_value = mock_enc
        
        gateway = OpenAIGateway(model="gpt-4o")
        count = gateway.count_tokens("Hello, world!")
        
        assert count == 5
        mock_enc.encode.assert_called_once()
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("tiktoken.encoding_for_model")
    def test_count_tokens_fallback(self, mock_encoding):
        """Test token counting fallback for unknown models."""
        mock_encoding.side_effect = KeyError("unknown model")
        
        gateway = OpenAIGateway(model="unknown-model")
        count = gateway.count_tokens("Hello")
        
        # Should fall back to character approximation
        assert count == len("Hello") // 4


class TestMistralGateway:
    """Test Mistral Gateway."""
    
    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    def test_initialize_gateway(self):
        """Test initializing Mistral gateway."""
        gateway = MistralGateway(model="mistral-large-latest")
        assert gateway.model == "mistral-large-latest"
        assert gateway.provider == "mistral"
        assert gateway.default_temperature == 0.7
    
    def test_initialize_without_api_key(self):
        """Test initializing without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key must be provided"):
                MistralGateway(model="mistral-large-latest")
    
    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    def test_build_messages(self):
        """Test building messages from request."""
        gateway = MistralGateway(model="mistral-large-latest")
        request = LLMRequest(
            system_prompt="You are helpful",
            user_prompt="Hello"
        )
        messages = gateway._build_messages(request)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
    
    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    def test_count_tokens(self):
        """Test approximate token counting."""
        gateway = MistralGateway(model="mistral-large-latest")
        count = gateway.count_tokens("Hello, world!")
        
        # Approximate count: length / 4
        expected = len("Hello, world!") // 4
        assert count == expected


class TestLLMGatewayConfigSchema:
    """Test LLM Gateway configuration schema."""
    
    def test_valid_openai_config(self):
        """Test valid OpenAI configuration."""
        config = LLMGatewayConfigSchema(
            provider="openai",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=2000,
        )
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
    
    def test_valid_mistral_config(self):
        """Test valid Mistral configuration."""
        config = LLMGatewayConfigSchema(
            provider="mistral",
            model="mistral-large-latest",
            temperature=0.5,
        )
        assert config.provider == "mistral"
        assert config.model == "mistral-large-latest"
    
    def test_config_with_extras(self):
        """Test configuration with extra settings."""
        config = LLMGatewayConfigSchema(
            provider="openai",
            model="gpt-4o",
            base_url="https://custom.api.com",
            organization="org-123",
            extra_settings={"top_p": 0.9}
        )
        assert config.base_url == "https://custom.api.com"
        assert config.organization == "org-123"
        assert config.extra_settings["top_p"] == 0.9
    
    def test_invalid_temperature(self):
        """Test configuration with invalid temperature."""
        with pytest.raises(ValueError):
            LLMGatewayConfigSchema(
                provider="openai",
                model="gpt-4o",
                temperature=3.0,  # Invalid: > 2.0
            )


class TestConvenienceFunction:
    """Test convenience functions."""
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_get_llm_gateway_with_model_string(self):
        """Test get_llm_gateway convenience function."""
        gateway = get_llm_gateway(model="openai:gpt-4o")
        assert isinstance(gateway, OpenAIGateway)
        assert gateway.model == "gpt-4o"
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_get_llm_gateway_with_provider(self):
        """Test get_llm_gateway with separate provider."""
        gateway = get_llm_gateway(model="gpt-4o", provider="openai")
        assert isinstance(gateway, OpenAIGateway)
        assert gateway.model == "gpt-4o"
    
    @patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    def test_get_llm_gateway_mistral(self):
        """Test get_llm_gateway with Mistral."""
        gateway = get_llm_gateway(model="mistral:mistral-small-latest")
        assert isinstance(gateway, MistralGateway)


class TestIntegration:
    """Integration tests for the gateway system."""
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_end_to_end_factory_flow(self):
        """Test complete flow from config to gateway."""
        # Create config
        config = {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.7,
        }
        
        # Create gateway
        gateway = LLMGatewayFactory.create_gateway_from_config(config)
        
        # Verify gateway
        assert isinstance(gateway, OpenAIGateway)
        assert gateway.model == "gpt-4o"
        assert gateway.provider == "openai"
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "MISTRAL_API_KEY": "test-key"})
    def test_multiple_gateways(self):
        """Test creating multiple gateways."""
        openai_gateway = get_llm_gateway("openai:gpt-4o")
        mistral_gateway = get_llm_gateway("mistral:mistral-large-latest")
        
        assert isinstance(openai_gateway, OpenAIGateway)
        assert isinstance(mistral_gateway, MistralGateway)
        assert openai_gateway.provider != mistral_gateway.provider


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

