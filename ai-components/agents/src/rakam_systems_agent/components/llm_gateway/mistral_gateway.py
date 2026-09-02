"""Mistral LLM Gateway implementation with structured output support."""
from __future__ import annotations
import os
from typing import Any, Callable, Dict, Iterator, Literal, Optional, Type, TypeVar

from mistralai import Mistral
from pydantic import BaseModel

from rakam_systems_tools.utils import logging
from rakam_systems_core.interfaces.llm_gateway import LLMGateway, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _server_origin(base_url: str) -> str:
    """Turn an OpenAI-style base URL into mistralai's origin-only ``server_url``.

    openai-python's ``base_url`` must include ``/v1``; mistralai's ``server_url``
    is the ORIGIN and the SDK appends ``/v1/chat/completions`` itself. So an
    un-normalised ``https://host/v1`` would request
    ``https://host/v1/v1/chat/completions`` -- a 404 at call time rather than a
    configuration error anyone sees at startup. ``LLMGatewayConfigSchema.base_url``
    is one field shared with :class:`OpenAIGateway`, so the difference is absorbed
    here instead of forcing two spellings on config files. Only an exact trailing
    ``/v1`` is stripped, so a path-mounted proxy like ``https://proxy/v1beta``
    survives untouched.
    """
    trimmed = base_url.rstrip("/")
    return trimmed[: -len("/v1")] if trimmed.endswith("/v1") else trimmed


def _strict_json_schema(node: Any) -> Any:
    """Close every object node so Mistral's strict ``json_schema`` mode accepts it.

    WHY THIS IS NOT ``mistralai.extra.response_format_from_pydantic_model`` (nor
    ``client.chat.parse``, which calls it): that helper's recursion treats only
    str/bool/None as terminal, so ANY schema carrying a numeric default or a
    ge/le/max_length bound hits its ``raise ValueError(f"Unexpected type:
    {node}")`` -- reproduced on mistralai 1.9.11 and 1.12.4 with a plain
    ``confidence: float = Field(0.5, ge=0, le=1)``. Upstream fixed it in 2.x by
    adding int and float as terminals, but 2.x also repackaged the SDK (no
    top-level ``mistralai``), so the ``<2.0.0`` cap this package needs rules that
    release out. The fix is one line; we carry it here.

    Two deliberate departures from a naive "make it strict" rewrite:

    * It **only** sets ``additionalProperties``. It never synthesises
      ``required = list(properties)``: promoting every ``Optional`` field to
      mandatory would silently rewrite the caller's own contract, and would make
      the strict path and the ``json_object`` fallback describe different schemas
      for the same model.
    * It leaves a node alone when ``additionalProperties`` is already present, so
      a ``Dict[str, str]`` field -- which pydantic emits as
      ``{"type": "object", "additionalProperties": {"type": "string"}}`` -- keeps
      its value schema instead of being rewritten to ``false`` and made
      uninhabitable.
    """
    if isinstance(node, dict):
        out = {key: _strict_json_schema(value) for key, value in node.items()}
        if (
            out.get("type") == "object"
            and "properties" in out
            and "additionalProperties" not in out
        ):
            out["additionalProperties"] = False
        return out
    if isinstance(node, list):
        return [_strict_json_schema(item) for item in node]
    # Scalars (including the int/float that upstream's version chokes on) are
    # schema values -- bounds, defaults, enum members -- and pass through as-is.
    return node


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
        base_url: Optional[str] = None,
        structured_mode: Literal["auto", "json_schema", "json_object"] = "auto",
        token_counting: Literal["auto", "exact", "approximate"] = "auto",
        token_counter: Optional[Callable[[str], int]] = None,
    ):
        """Initialize Mistral Gateway.

        Args:
            name: Gateway name
            config: Configuration dictionary
            model: Mistral model name (e.g., "mistral-large-latest", "mistral-small-latest")
            default_temperature: Default temperature for generation
            api_key: Mistral API key (falls back to MISTRAL_API_KEY env var)
            base_url: Optional endpoint override for a proxy or self-hosted
                deployment. Give the origin (``https://gw.internal``); a trailing
                ``/v1`` is stripped for you so the same config value works for
                this gateway and :class:`OpenAIGateway`.
            structured_mode: How :meth:`generate_structured` asks for JSON.
                ``"json_schema"`` sends the schema to Mistral's strict mode;
                ``"json_object"`` reproduces the pre-existing behaviour exactly
                (schema pasted into the system prompt, ``{"type": "json_object"}``
                on the wire); ``"auto"`` (default) tries strict mode and falls back
                to ``"json_object"`` for the life of this instance the first time a
                model rejects it.
            token_counting: How :meth:`count_tokens` counts. ``"exact"`` requires
                ``token_counter`` and raises without it; ``"approximate"`` always
                uses the character heuristic; ``"auto"`` (default) uses
                ``token_counter`` when one was given and the heuristic otherwise.
            token_counter: A callable turning text into a token count. Mistral
                publishes no tokenize endpoint and ships no usable offline
                tokenizer for its current models (``mistral-common``'s
                ``from_model`` is deprecated and recognises no ``-latest`` alias;
                its replacement needs the Hugging Face hub), so an exact local
                count has to come from the caller. See :meth:`count_tokens` for
                the exact count that costs nothing.
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

        # `base_url` is stored under the name the config schema and OpenAIGateway
        # both use; the SDK's own kwarg is `server_url` and wants a bare origin.
        # Not forwarded to super(): the LLMGateway ABC takes no such parameter, and
        # widening it would mean publishing a new rakam-systems-core.
        self.base_url = base_url
        self._structured_mode = structured_mode
        # Per-instance, never module-global: one model refusing strict mode says
        # nothing about the next gateway someone builds.
        self._structured_downgraded = False
        self._token_counting = token_counting
        self._token_counter = token_counter
        if token_counting == "exact" and token_counter is None:
            raise ValueError(
                "MistralGateway(token_counting='exact') needs a token_counter: "
                "Mistral has no tokenize endpoint and no offline tokenizer that "
                "covers its current models, so an exact count must be supplied. "
                "Use token_counting='auto' to fall back to the character "
                "heuristic, or read LLMResponse.usage.prompt_tokens after a call."
            )
        # One warning per gateway, not one per call: count_tokens is often called
        # in a loop over a corpus.
        self._token_counter_warned = False
        self.client = Mistral(
            api_key=self.api_key,
            server_url=_server_origin(base_url) if base_url else None,
        )

        logger.info(
            f"Initialized Mistral Gateway with model={self.model}, "
            f"temperature={self.default_temperature}, endpoint={self.base_url or 'default'}"
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

        Uses Mistral's native strict ``json_schema`` response format when the
        model supports it, which constrains decoding server-side, and otherwise
        the older technique of describing the schema in the system prompt and
        asking for ``json_object``. See ``structured_mode`` in :meth:`__init__`
        for how the choice is made; a model that rejects strict mode downgrades
        this instance once and stays downgraded, so a rejection costs one extra
        request per gateway rather than one per call.

        Args:
            request: Standardized LLM request
            schema: Pydantic model class to parse response into

        Returns:
            Instance of the schema class
        """
        if self._structured_mode == "json_object" or self._structured_downgraded:
            return self._generate_structured_json_object(request, schema)

        try:
            return self._generate_structured_json_schema(request, schema)
        except Exception as e:
            if self._structured_mode != "auto":
                raise
            # Strict mode is per-model, and Mistral signals a model that lacks it
            # with an ordinary API error, so the only way to find out is to ask.
            self._structured_downgraded = True
            logger.warning(
                f"Mistral model={self.model} rejected the native json_schema "
                f"response format ({type(e).__name__}: {e}); falling back to "
                f"json_object with a prompt-described schema for the life of this "
                f"gateway. Pass structured_mode='json_object' to skip this probe, "
                f"or structured_mode='json_schema' to make it a hard error."
            )
            return self._generate_structured_json_object(request, schema)

    def _generate_structured_json_schema(
        self,
        request: LLMRequest,
        schema: Type[T],
    ) -> T:
        """Ask for the schema natively, via Mistral's strict ``json_schema`` mode."""
        from mistralai.models import JSONSchema, ResponseFormat

        params = {
            "model": self.model,
            "messages": self._build_messages(request),
            "temperature": request.temperature
            if request.temperature is not None
            else self.default_temperature,
            "response_format": ResponseFormat(
                type="json_schema",
                json_schema=JSONSchema(
                    name=schema.__name__,
                    schema_definition=_strict_json_schema(schema.model_json_schema()),
                    strict=True,
                ),
            ),
        }

        if request.max_tokens:
            params["max_tokens"] = request.max_tokens

        params.update(request.extra_params)

        logger.debug(
            f"Calling Mistral API with native json_schema, model={self.model}, "
            f"schema={schema.__name__}"
        )

        response = self.client.chat.complete(**params)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            # AssistantMessage.content is Optional[Union[str, List[ContentChunk]]];
            # model_validate_json would raise something unreadable on either.
            raise ValueError(
                f"Mistral returned no text content to parse as {schema.__name__} "
                f"(got {type(content).__name__})"
            )
        parsed_result = schema.model_validate_json(content)

        logger.info(
            f"Mistral structured response received (json_schema): schema={schema.__name__}"
        )
        return parsed_result

    def _generate_structured_json_object(
        self,
        request: LLMRequest,
        schema: Type[T],
    ) -> T:
        """Describe the schema in the system prompt and ask for ``json_object``.

        The original implementation, kept intact: it is the fallback for models
        without strict-mode support, and reproduces the pre-change request exactly
        for anyone who pins ``structured_mode="json_object"``.
        """
        import json

        # Add schema information to the system prompt
        schema_json = schema.model_json_schema()

        # Enhance system prompt with schema information
        enhanced_system = request.system_prompt or ""
        enhanced_system += f"\n\nYou must respond with valid JSON that matches this schema:\n{json.dumps(schema_json, indent=2)}"

        messages = [
            {"role": "system", "content": enhanced_system},
            {"role": "user", "content": request.user_prompt},
        ]

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

        Exactness is opt-in, because Mistral gives no way to be exact for free:
        there is no tokenize endpoint, and ``mistral-common`` -- the obvious
        candidate -- cannot resolve any ``-latest`` model name offline
        (``from_model`` is deprecated and knows only 16 dated names; the
        replacement ``from_hf_hub`` needs the Hugging Face hub at runtime). So a
        caller who needs exact counts supplies ``token_counter``, and everyone
        else gets the character heuristic rather than a heavy dependency that
        would quietly reach for the network.

        The exact count that costs nothing is already on every response:
        ``LLMResponse.usage.prompt_tokens``, reported by the API itself. Prefer it
        whenever you are counting a prompt you are also going to send.

        Args:
            text: Text to count tokens for
            model: Accepted for interface compatibility; Mistral's counting does
                not vary by model here.

        Returns:
            Number of tokens -- exact if a ``token_counter`` produced it,
            otherwise approximated at 4 characters per token.

        Raises:
            ValueError: Only in ``token_counting="exact"`` mode, if the supplied
                counter fails. ``"auto"`` degrades to the heuristic instead, so a
                broken tokenizer never takes down an ingestion run.
        """
        if self._token_counting != "approximate" and self._token_counter is not None:
            try:
                return int(self._token_counter(text))
            except Exception as e:
                if self._token_counting == "exact":
                    raise ValueError(
                        f"token_counter failed on {len(text)} characters of text: {e}"
                    ) from e
                if not self._token_counter_warned:
                    self._token_counter_warned = True
                    logger.warning(
                        f"token_counter failed ({type(e).__name__}: {e}); falling "
                        f"back to the 4-characters-per-token approximation for the "
                        f"life of this gateway"
                    )

        token_count = len(text) // 4
        logger.debug(
            f"Counted ~{token_count} tokens (approximation) for text of length "
            f"{len(text)} characters"
        )
        return token_count
