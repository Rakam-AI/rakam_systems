"""LLM Gateway Factory for provider routing and configuration-driven model selection."""
from __future__ import annotations
import os
from typing import Any, Dict, Optional

from rakam_systems.core.ai_utils import logging
from rakam_systems.core.ai_core.interfaces.llm_gateway import LLMGateway
from .openai_gateway import OpenAIGateway
from .mistral_gateway import MistralGateway

logger = logging.getLogger(__name__)


class LLMGatewayFactory:
    """Factory for creating LLM gateways based on provider and configuration.

    This factory enables:
    - Configuration-driven provider selection
    - Automatic routing to the correct gateway
    - Model string parsing (e.g., "openai:gpt-4o")
    - Fallback to environment-based configuration

    Example:
        >>> # Using model string with provider prefix
        >>> gateway = LLMGatewayFactory.create_gateway("openai:gpt-4o")
        >>> 
        >>> # Using explicit provider and model
        >>> gateway = LLMGatewayFactory.create_gateway_from_config({
        ...     "provider": "mistral",
        ...     "model": "mistral-large-latest",
        ...     "temperature": 0.7
        ... })
    """

    # Registry of available providers
    _PROVIDERS = {
        "openai": OpenAIGateway,
        "mistral": MistralGateway,
    }

    # Default models for each provider
    _DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "mistral": "mistral-large-latest",
    }

    @classmethod
    def parse_model_string(cls, model_string: str) -> tuple[str, str]:
        """Parse a model string into provider and model name.

        Supports formats:
        - "openai:gpt-4o" -> ("openai", "gpt-4o")
        - "mistral:mistral-large-latest" -> ("mistral", "mistral-large-latest")
        - "gpt-4o" -> ("openai", "gpt-4o")  # assumes OpenAI if no prefix

        Args:
            model_string: Model string to parse

        Returns:
            Tuple of (provider, model_name)

        Raises:
            ValueError: If provider is unknown
        """
        if ":" in model_string:
            provider, model = model_string.split(":", 1)
        else:
            # Default to OpenAI if no provider specified
            provider = "openai"
            model = model_string

        if provider not in cls._PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. Supported providers: {list(cls._PROVIDERS.keys())}"
            )

        return provider, model

    @classmethod
    def create_gateway(
        cls,
        model_string: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMGateway:
        """Create an LLM gateway from a model string.

        Args:
            model_string: Model string (e.g., "openai:gpt-4o", "mistral:mistral-large-latest")
                         Falls back to DEFAULT_LLM_MODEL env var
            temperature: Temperature for generation
            api_key: API key for the provider (provider-specific env vars used as fallback)
            **kwargs: Additional parameters passed to gateway constructor

        Returns:
            Configured LLM gateway instance

        Raises:
            ValueError: If provider is unknown or configuration is invalid
        """
        # Get model string from parameter or environment
        model_string = model_string or os.getenv(
            "DEFAULT_LLM_MODEL", "openai:gpt-4o")

        # Parse provider and model
        provider, model = cls.parse_model_string(model_string)

        # Get gateway class
        gateway_class = cls._PROVIDERS[provider]

        # Build gateway parameters
        gateway_params = {
            "model": model,
            "default_temperature": temperature or float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.7")),
        }

        # Add API key if provided
        if api_key:
            gateway_params["api_key"] = api_key

        # Add any additional parameters
        gateway_params.update(kwargs)

        logger.info(
            f"Creating {provider} gateway with model={model}, temperature={gateway_params['default_temperature']}"
        )

        return gateway_class(**gateway_params)

    @classmethod
    def create_gateway_from_config(
        cls,
        config: Dict[str, Any],
    ) -> LLMGateway:
        """Create an LLM gateway from a configuration dictionary.

        Expected config format:
        {
            "provider": "openai",  # or "mistral"
            "model": "gpt-4o",
            "temperature": 0.7,
            "api_key": "...",  # optional
            ... # additional provider-specific params
        }

        Args:
            config: Configuration dictionary

        Returns:
            Configured LLM gateway instance

        Raises:
            ValueError: If provider is unknown or configuration is invalid
        """
        provider = config.get("provider")
        model = config.get("model")

        if not provider:
            raise ValueError("Configuration must specify 'provider'")

        if provider not in cls._PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. Supported providers: {list(cls._PROVIDERS.keys())}"
            )

        # Use default model if not specified
        if not model:
            model = cls._DEFAULT_MODELS.get(provider)
            logger.warning(
                f"No model specified for {provider}, using default: {model}"
            )

        # Get gateway class
        gateway_class = cls._PROVIDERS[provider]

        # Extract common parameters
        gateway_params = {
            "model": model,
            "default_temperature": config.get("temperature", 0.7),
        }

        # Add API key if provided
        if "api_key" in config:
            gateway_params["api_key"] = config["api_key"]

        # Add provider-specific parameters
        provider_specific_keys = {
            "openai": ["base_url", "organization"],
            "mistral": [],
        }

        for key in provider_specific_keys.get(provider, []):
            if key in config:
                gateway_params[key] = config[key]

        logger.info(
            f"Creating {provider} gateway from config with model={model}"
        )

        return gateway_class(**gateway_params)

    @classmethod
    def get_default_gateway(cls) -> LLMGateway:
        """Get a default gateway based on environment configuration.

        Checks environment variables:
        - DEFAULT_LLM_PROVIDER: Provider name (default: "openai")
        - DEFAULT_LLM_MODEL: Model name (default: "gpt-4o" for OpenAI)
        - DEFAULT_LLM_TEMPERATURE: Temperature (default: 0.7)

        Returns:
            Configured default LLM gateway
        """
        provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        model = os.getenv("DEFAULT_LLM_MODEL",
                          cls._DEFAULT_MODELS.get(provider, "gpt-4o"))
        temperature = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.7"))

        # If model doesn't have provider prefix, add it
        if ":" not in model:
            model_string = f"{provider}:{model}"
        else:
            model_string = model

        return cls.create_gateway(
            model_string=model_string,
            temperature=temperature,
        )

    @classmethod
    def register_provider(
        cls,
        provider_name: str,
        gateway_class: type[LLMGateway],
        default_model: Optional[str] = None,
    ) -> None:
        """Register a new provider gateway.

        This allows extending the factory with custom providers.

        Args:
            provider_name: Name of the provider (e.g., "custom")
            gateway_class: Gateway class implementing LLMGateway
            default_model: Optional default model for this provider
        """
        cls._PROVIDERS[provider_name] = gateway_class
        if default_model:
            cls._DEFAULT_MODELS[provider_name] = default_model

        logger.info(
            f"Registered provider '{provider_name}' with gateway class {gateway_class.__name__}"
        )

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered providers.

        Returns:
            List of provider names
        """
        return list(cls._PROVIDERS.keys())

    @classmethod
    def get_default_model(cls, provider: str) -> Optional[str]:
        """Get the default model for a provider.

        Args:
            provider: Provider name

        Returns:
            Default model name or None if provider unknown
        """
        return cls._DEFAULT_MODELS.get(provider)


# Convenience function for creating gateways
def get_llm_gateway(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs: Any,
) -> LLMGateway:
    """Convenience function to create an LLM gateway.

    Args:
        model: Model name or full model string (e.g., "gpt-4o" or "openai:gpt-4o")
        provider: Provider name (optional if model string includes provider)
        temperature: Temperature for generation
        **kwargs: Additional parameters

    Returns:
        Configured LLM gateway

    Example:
        >>> gateway = get_llm_gateway(model="gpt-4o", provider="openai")
        >>> gateway = get_llm_gateway(model="openai:gpt-4o")
    """
    if provider and model:
        # Build model string from separate provider and model
        model_string = f"{provider}:{model}"
    elif model:
        # Use model as-is (may already include provider)
        model_string = model
    else:
        # Use default
        model_string = None

    return LLMGatewayFactory.create_gateway(
        model_string=model_string,
        temperature=temperature,
        **kwargs,
    )
