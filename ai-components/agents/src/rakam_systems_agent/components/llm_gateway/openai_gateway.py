"""OpenAI LLM Gateway implementation with structured output support."""
from __future__ import annotations
import os
from typing import Any, Dict, Iterator, Optional, Type, TypeVar

import tiktoken
from openai import OpenAI
from pydantic import BaseModel

from rakam_systems_tools.utils import logging
from rakam_systems_core.interfaces.llm_gateway import LLMGateway, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAIGateway(LLMGateway):
    """OpenAI LLM Gateway with support for structured outputs.

    Features:
    - Text generation
    - Structured output using response_format
    - Streaming support
    - Token counting with tiktoken
    - Support for all OpenAI chat models

    Example:
        >>> gateway = OpenAIGateway(model="gpt-4o", api_key="...")
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
        name: str = "openai_gateway",
        config: Optional[Dict[str, Any]] = None,
        model: str = "gpt-4o",
        default_temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ):
        """Initialize OpenAI Gateway.

        Args:
            name: Gateway name
            config: Configuration dictionary
            model: OpenAI model name (e.g., "gpt-4o", "gpt-4-turbo")
            default_temperature: Default temperature for generation
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            base_url: Optional base URL for API
            organization: Optional organization ID
        """
        super().__init__(
            name=name,
            config=config,
            provider="openai",
            model=model,
            default_temperature=default_temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided via api_key parameter or OPENAI_API_KEY environment variable"
            )

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            organization=organization,
        )

        logger.info(
            f"Initialized OpenAI Gateway with model={self.model}, temperature={self.default_temperature}"
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
        """Generate a response from OpenAI.

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
            f"Calling OpenAI API with model={self.model}, temperature={params['temperature']}")

        try:
            completion = self.client.chat.completions.create(**params)

            # Extract response
            content = completion.choices[0].message.content

            # Build usage information
            usage = None
            if completion.usage:
                usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }

            response = LLMResponse(
                content=content,
                usage=usage,
                model=completion.model,
                finish_reason=completion.choices[0].finish_reason,
                metadata={
                    "id": completion.id,
                    "created": completion.created,
                }
            )

            logger.info(
                f"OpenAI response received: {usage.get('total_tokens', 'unknown')} tokens, "
                f"finish_reason={response.finish_reason}"
            )

            return response

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    def generate_structured(
        self,
        request: LLMRequest,
        schema: Type[T],
    ) -> T:
        """Generate structured output conforming to a Pydantic schema.

        Uses OpenAI's structured output feature to ensure response matches schema.

        Args:
            request: Standardized LLM request
            schema: Pydantic model class to parse response into

        Returns:
            Instance of the schema class
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
            f"Calling OpenAI API for structured output with model={self.model}, schema={schema.__name__}"
        )

        try:
            # Use beta parse feature for structured outputs
            completion = self.client.beta.chat.completions.parse(
                **params,
                response_format=schema,
            )

            parsed_result = completion.choices[0].message.parsed

            logger.info(
                f"OpenAI structured response received: schema={schema.__name__}"
            )

            return parsed_result

        except Exception as e:
            logger.error(f"OpenAI structured output error: {str(e)}")
            raise

    def stream(self, request: LLMRequest) -> Iterator[str]:
        """Stream responses from OpenAI.

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
            "stream": True,
        }

        if request.max_tokens:
            params["max_tokens"] = request.max_tokens

        # Add extra parameters (excluding stream since we set it)
        extra = {k: v for k, v in request.extra_params.items() if k !=
                 "stream"}
        params.update(extra)

        logger.debug(f"Streaming from OpenAI with model={self.model}")

        try:
            stream = self.client.chat.completions.create(**params)

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming error: {str(e)}")
            raise

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for
            model: Model name to determine encoding (uses instance model if None)

        Returns:
            Number of tokens in the text
        """
        try:
            model_name = model or self.model

            # Try to get encoding for the specific model
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fall back to cl100k_base for unknown models
                logger.warning(
                    f"Unknown model {model_name}, using cl100k_base encoding"
                )
                encoding = tiktoken.get_encoding("cl100k_base")

            token_count = len(encoding.encode(text))

            logger.debug(
                f"Counted {token_count} tokens for text of length {len(text)} characters"
            )

            return token_count

        except Exception as e:
            logger.warning(
                f"Error counting tokens: {e}. Using character approximation.")
            # Fallback to character-based approximation (rough estimate: 4 chars = 1 token)
            return len(text) // 4
