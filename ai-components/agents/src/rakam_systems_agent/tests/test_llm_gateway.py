# tests/test_llm_gateway_factory.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from rakam_systems_agent.components.llm_gateway.gateway_factory import LLMGatewayFactory


@pytest.fixture
def mock_openai_gateway():
    """Mock OpenAIGateway constructor and instance."""
    mock_class = MagicMock(name="OpenAIGatewayMock")
    instance = mock_class.return_value
    # Example async method on instance
    instance.generate = AsyncMock(return_value="mocked response")
    return mock_class


@pytest.fixture
def mock_mistral_gateway():
    """Mock MistralGateway constructor and instance."""
    mock_class = MagicMock(name="MistralGatewayMock")
    instance = mock_class.return_value
    instance.generate = AsyncMock(return_value="mocked mistral")
    return mock_class


def test_create_gateway_with_mocked_openai(mock_openai_gateway):
    with patch.object(
        LLMGatewayFactory,
        "_PROVIDERS",
        {"openai": mock_openai_gateway}
    ):
        # Provide dummy API key to bypass validation
        gateway = LLMGatewayFactory.create_gateway(
            model_string="openai:gpt-4o",
            temperature=0.5,
            api_key="dummy-key"
        )

        # Ensure the mock constructor was called with expected params
        mock_openai_gateway.assert_called_once_with(
            model="gpt-4o",
            default_temperature=0.5,
            api_key="dummy-key"
        )

        # The returned gateway is the mock instance
        assert gateway is mock_openai_gateway.return_value


def test_create_gateway_from_config_with_mock(mock_openai_gateway):
    with patch.object(
        LLMGatewayFactory,
        "_PROVIDERS",
        {"openai": mock_openai_gateway}
    ):
        config = {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.7,
            "api_key": "dummy-key"
        }
        gateway = LLMGatewayFactory.create_gateway_from_config(config)

        # Ensure the mock constructor was called
        mock_openai_gateway.assert_called_once_with(
            model="gpt-4o",
            default_temperature=0.7,
            api_key="dummy-key"
        )

        assert gateway is mock_openai_gateway.return_value


def test_create_gateway_with_mistral(mock_mistral_gateway):
    with patch.object(
        LLMGatewayFactory,
        "_PROVIDERS",
        {"mistral": mock_mistral_gateway}
    ):
        gateway = LLMGatewayFactory.create_gateway(
            model_string="mistral:mistral-large-latest",
            temperature=0.9,
            api_key="api_test"
        )

        mock_mistral_gateway.assert_called_once_with(
            model="mistral-large-latest",
            default_temperature=0.9,
            api_key='api_test'
        )

        assert gateway is mock_mistral_gateway.return_value


def test_get_default_gateway_with_mock(mock_openai_gateway):
    with patch.object(
        LLMGatewayFactory,
        "_PROVIDERS",
        {"openai": mock_openai_gateway}
    ), patch.dict("os.environ", {"DEFAULT_LLM_MODEL": "openai:gpt-4o", "OPENAI_API_KEY": "test_api"}):
        gateway = LLMGatewayFactory.get_default_gateway()

        mock_openai_gateway.assert_called_once_with(
            model="gpt-4o",
            default_temperature=0.7
        )

        assert gateway is mock_openai_gateway.return_value


def test_register_and_list_providers():
    # Use a dummy gateway class
    class DummyGateway:
        pass

    LLMGatewayFactory.register_provider(
        "dummy", DummyGateway, default_model="dummy-model")
    assert "dummy" in LLMGatewayFactory.list_providers()
    assert LLMGatewayFactory.get_default_model("dummy") == "dummy-model"


def test_create_gateway_from_config_forwards_mistral_base_url(mock_mistral_gateway):
    # provider_specific_keys used to map "mistral" to an empty list, so a base_url
    # in a config file was silently discarded -- no error, no warning, requests
    # quietly going to api.mistral.ai instead of the configured proxy.
    with patch.object(
        LLMGatewayFactory,
        "_PROVIDERS",
        {"mistral": mock_mistral_gateway}
    ):
        LLMGatewayFactory.create_gateway_from_config({
            "provider": "mistral",
            "model": "mistral-large-latest",
            "temperature": 0.3,
            "base_url": "https://gw.internal",
        })

    kwargs = mock_mistral_gateway.call_args.kwargs
    assert kwargs["base_url"] == "https://gw.internal"
    assert kwargs["model"] == "mistral-large-latest"


def test_create_gateway_from_config_omits_base_url_when_absent(mock_mistral_gateway):
    # The forwarding loop is guarded by `if key in config`, so a config without a
    # base_url must not start passing base_url=None -- that would break the exact
    # assert_called_once_with in the tests above and change the client's endpoint
    # handling.
    with patch.object(
        LLMGatewayFactory,
        "_PROVIDERS",
        {"mistral": mock_mistral_gateway}
    ):
        LLMGatewayFactory.create_gateway_from_config({
            "provider": "mistral",
            "model": "mistral-large-latest",
        })

    assert "base_url" not in mock_mistral_gateway.call_args.kwargs
