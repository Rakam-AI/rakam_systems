"""Mistral LLM Gateway implementation with structured output support."""
from __future__ import annotations
import os
from typing import Any, Dict, Iterator, Optional, Type, TypeVar

from mistralai import Mistral
from pydantic import BaseModel

from rakam_systems_tools.utils import logging
from rakam_systems_core.interfaces.llm_gateway import LLMGateway, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class MistralGateway(LLMGateway):
    """Mistral LLM Gateway with support for structured outputs.

    Features:
    - Text generation
    - Structured output using JSON mode
    - Streaming support
    - Token counting (approximate)
    - Support for all Mistral models

    Example:
        >>> gateway = MistralGateway(model="mistral-large-latest", api_key="...")
        >>> request = LLMRequest(
        ...     system_prompt="You are a helpful assistant",
        ...     user_prompt="What is AI?",
        ...     temperature=0.7
        ... )
        >>> response = gateway.generate(request)
        >>> print(response.content)
    """

    def __init__(
        self,
        name: str = "mistral_gateway",
        config: Optional[Dict[str, Any]] = None,
        model: str = "mistral-large-latest",
        default_temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        """Initialize Mistral Gateway.

        Args:
            name: Gateway name
            config: Configuration dictionary
            model: Mistral model name (e.g., "mistral-large-latest", "mistral-small-latest")
            default_temperature: Default temperature for generation
            api_key: Mistral API key (falls back to MISTRAL_API_KEY env var)
        """
        super().__init__(
            name=name,
            config=config,
            provider="mistral",
            model=model,
            default_temperature=default_temperature,
            api_key=api_key or os.getenv("MISTRAL_API_KEY"),
        )

        if not self.api_key:
            raise ValueError(
                "Mistral API key must be provided via api_key parameter or MISTRAL_API_KEY environment variable"
            )

        # Initialize Mistral client
        self.client = Mistral(api_key=self.api_key)

        logger.info(
            f"Initialized Mistral Gateway with model={self.model}, temperature={self.default_temperature}"
        )

    def _build_messages(self, request: LLMRequest) -> list[dict]:
        """Build messages array from request."""
        messages = []

        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })

        messages.append({
            "role": "user",
            "content": request.user_prompt
        })

        return messages

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from Mistral.

        Args:
            request: Standardized LLM request

        Returns:
            Standardized LLM response
        """
        messages = self._build_messages(request)

        # Prepare API call parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature if request.temperature is not None else self.default_temperature,
        }

        if request.max_tokens:
            params["max_tokens"] = request.max_tokens

        # Add extra parameters
        params.update(request.extra_params)

        logger.debug(
            f"Calling Mistral API with model={self.model}, temperature={params['temperature']}")

        try:
            response = self.client.chat.complete(**params)

            # Extract response
            content = response.choices[0].message.content

            # Build usage information
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            llm_response = LLMResponse(
                content=content,
                usage=usage,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "id": response.id,
                    "created": response.created,
                }
            )

            logger.info(
                f"Mistral response received: {usage.get('total_tokens', 'unknown') if usage else 'unknown'} tokens, "
                f"finish_reason={llm_response.finish_reason}"
            )

            return llm_response

        except Exception as e:
            logger.error(f"Mistral API error: {str(e)}")
            raise

    def generate_structured(
        self,
        request: LLMRequest,
        schema: Type[T],
    ) -> T:
        """Generate structured output conforming to a Pydantic schema.

        Uses Mistral's JSON mode and parses the response into the schema.

        Args:
            request: Standardized LLM request
            schema: Pydantic model class to parse response into

        Returns:
            Instance of the schema class
        """
        import json

        messages = self._build_messages(request)

        # Add schema information to the system prompt
        schema_json = schema.model_json_schema()

        # Enhance system prompt with schema information
        enhanced_system = request.system_prompt or ""
        enhanced_system += f"\n\nYou must respond with valid JSON that matches this schema:\n{json.dumps(schema_json, indent=2)}"

        # Update messages with enhanced system prompt
        messages = []
        messages.append({
            "role": "system",
            "content": enhanced_system
        })
        messages.append({
            "role": "user",
            "content": request.user_prompt
        })

        # Prepare API call parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature if request.temperature is not None else self.default_temperature,
            "response_format": {"type": "json_object"},
        }

        if request.max_tokens:
            params["max_tokens"] = request.max_tokens

        # Add extra parameters
        params.update(request.extra_params)

        logger.debug(
            f"Calling Mistral API for structured output with model={self.model}, schema={schema.__name__}"
        )

        try:
            response = self.client.chat.complete(**params)

            # Extract and parse JSON response
            content = response.choices[0].message.content
            parsed_result = schema.model_validate_json(content)

            logger.info(
                f"Mistral structured response received: schema={schema.__name__}"
            )

            return parsed_result

        except Exception as e:
            logger.error(f"Mistral structured output error: {str(e)}")
            raise

    def stream(self, request: LLMRequest) -> Iterator[str]:
        """Stream responses from Mistral.

        Args:
            request: Standardized LLM request

        Yields:
            String chunks from the LLM
        """
        messages = self._build_messages(request)

        # Prepare API call parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature if request.temperature is not None else self.default_temperature,
        }

        if request.max_tokens:
            params["max_tokens"] = request.max_tokens

        # Add extra parameters
        params.update(request.extra_params)

        logger.debug(f"Streaming from Mistral with model={self.model}")

        try:
            stream = self.client.chat.stream(**params)

            for chunk in stream:
                if chunk.data.choices[0].delta.content is not None:
                    yield chunk.data.choices[0].delta.content

        except Exception as e:
            logger.error(f"Mistral streaming error: {str(e)}")
            raise

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text.

        Mistral doesn't provide a native tokenization library, so we use approximation.

        Args:
            text: Text to count tokens for
            model: Model name (unused for Mistral)

        Returns:
            Approximate number of tokens in the text
        """
        # Approximation: average of 4 characters per token
        # This is less accurate than tiktoken but reasonable for most use cases
        token_count = len(text) // 4

        logger.debug(
            f"Counted ~{token_count} tokens (approximation) for text of length {len(text)} characters"
        )

        return token_count
