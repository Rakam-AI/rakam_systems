"""Mistral embeddings backend for the AI gateway.

WHICH ``EmbeddingModel``: this subclasses **pydantic-ai's**
``pydantic_ai.embeddings.base.EmbeddingModel``, not the sibling
``rakam_systems_core.interfaces.embedding_model.EmbeddingModel`` that
``openai_embeddings.py`` and ``configurable_embeddings.py`` implement. The two
share a name and live one directory apart, so the distinction is worth stating:
this class plugs *into* pydantic-ai, and ``build_embedder`` then wraps it in an
``Embedder`` and adapts that to the rakam interface via ``GatewayEmbeddings``.

WHY IT EXISTS AT ALL: pydantic-ai ships no ``pydantic_ai/embeddings/mistral.py``
-- verified absent in every release from 1.39 through 2.37 -- and
``infer_embedding_model`` is a hardcoded if/elif over the provider prefix with no
registry, no entry point, and a ``provider_factory`` hook that cannot help
(the model class is chosen from the raw prefix before the factory is consulted).
So ``mistral:mistral-embed`` raises ``UserError: Unknown embeddings model`` and
the only way to reach Mistral embeddings is to bring our own backend. Delete this
module the release after upstream ships one.

WHY IT BUILDS ITS OWN ``Mistral`` CLIENT rather than taking pydantic-ai's
``MistralProvider``: that provider is the one part of pydantic-ai coupled to
mistralai's major version -- it switched from ``from mistralai import Mistral``
to ``from mistralai.client import Mistral`` in 1.76.0, tracking mistralai 2.x's
removal of the top-level package. Depending on it would force a
``pydantic-ai<1.76`` cap on what is a *hard* dependency of this package, freezing
every other embedding backend too. The ABC used here, by contrast, is stable:
``embeddings/{base,result,settings}.py`` are byte-identical from 1.61.0 through
1.107.5. Constructing the SDK client directly keeps the version coupling to
mistralai alone, which the ``mistral`` extra already pins.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import httpx
from mistralai import Mistral
from mistralai.models import MistralError
from pydantic_ai.embeddings.base import EmbeddingModel
from pydantic_ai.embeddings.result import EmbeddingResult, EmbedInputType
from pydantic_ai.embeddings.settings import EmbeddingSettings
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)
from pydantic_ai.usage import RequestUsage

# The provider name reported to pydantic-ai (instrumentation spans, EmbeddingResult
# .provider_name, and genai-prices lookups all key on it).
_SYSTEM = "mistral"


class MistralEmbeddingModel(EmbeddingModel):
    """pydantic-ai embedding model backed by ``mistralai``'s embeddings endpoint.

    Args:
        model_name: A Mistral embedding model, e.g. ``"mistral-embed"``.
        api_key: Falls back to ``MISTRAL_API_KEY`` (the SDK reads it itself when
            this is ``None``).
        base_url: The endpoint **origin**, e.g. ``"https://gw.internal"``. The SDK
            appends ``/v1/embeddings``, so a URL that already ends in ``/v1``
            would produce ``/v1/v1/embeddings`` -- ``build_embedder`` normalises
            it before calling here.
        http_client: An ``httpx.AsyncClient`` for the SDK to use.
        client: An already built ``Mistral``. Mutually exclusive with the three
            arguments above, which only exist to build one.
        settings: Default ``EmbeddingSettings`` for this model.

    Only ``dimensions`` and ``extra_headers`` from ``EmbeddingSettings`` are
    honoured. ``truncate`` and ``extra_body`` are **silently ignored**: the
    Mistral embeddings API has no counterpart for either, though ``openai.py``
    and ``cohere.py`` both forward them, so this is a real divergence rather than
    an oversight.
    """

    def __init__(
        self,
        model_name: str = "mistral-embed",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        client: Mistral | None = None,
        settings: EmbeddingSettings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self._model_name = model_name
        if client is not None:
            if api_key or base_url or http_client is not None:
                raise ValueError(
                    "MistralEmbeddingModel: pass `client`, or the arguments used to "
                    "build one (api_key / base_url / http_client), but not both -- "
                    "a prebuilt client already carries its own key, endpoint and "
                    "transport, so the others would be silently dropped."
                )
            self._client = client
            self._base_url = None
        else:
            if api_key is None and base_url is None and not os.environ.get(
                "MISTRAL_API_KEY"
            ):
                # Fail here rather than on the first request, mirroring the OpenAI
                # route: openai-python raises at construction when it is pointed at
                # the hosted API with no key, and tolerates a missing key only when
                # a base_url says "I am talking to my own endpoint". mistralai does
                # neither -- it would build happily and 401 mid-ingestion -- so the
                # hosted half of that contract is enforced here. A base_url still
                # means keyless is allowed, so self-hosted deployments keep working.
                raise ValueError(
                    "MistralEmbeddingModel needs an API key for api.mistral.ai: set "
                    "MISTRAL_API_KEY or pass api_key=. (A keyless endpoint is fine "
                    "when you also pass base_url=.)"
                )
            # server_url=None lets the SDK use its own default; api_key=None lets it
            # read MISTRAL_API_KEY. Passing either explicitly as None is what the SDK
            # expects, unlike its Unset() sentinel arguments (see `embed` below).
            self._client = Mistral(
                api_key=api_key,
                server_url=base_url,
                async_client=http_client,
            )
            self._base_url = base_url

    @property
    def model_name(self) -> str:
        """The embedding model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The provider identifier, ``"mistral"``."""
        return _SYSTEM

    @property
    def base_url(self) -> str | None:
        """The configured endpoint origin, or the SDK's default."""
        if self._base_url is not None:
            return self._base_url
        try:
            return self._client.sdk_configuration.get_server_details()[0]
        except Exception:  # pragma: no cover - defensive; SDK-internal accessor
            return None

    # `max_input_tokens` and `count_tokens` are deliberately NOT overridden.
    # The SDK encodes no per-model input limits anywhere, and Mistral exposes no
    # tokenize endpoint, so the base class's `None` / NotImplementedError are the
    # honest answers -- inventing a limit would be worse than admitting we have
    # none. Callers therefore get `Embedder.max_input_tokens() -> None` and an
    # `Embedder.count_tokens()` that raises.

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        # prepare_embed is an INSTANCE method on the ABC: it normalises a bare
        # string to a list and merges per-call settings over the model's own.
        texts, merged = self.prepare_embed(inputs, settings)

        if not texts:
            # Short-circuit before the SDK. mistralai happily serialises an empty
            # `inputs` and the server answers 422; `Embedder` does not guard this
            # (GatewayEmbeddings does, but this model is reachable without it).
            return self._result([], texts, input_type, RequestUsage(), None, None)

        kwargs: dict[str, Any] = {}
        # Only set kwargs we actually have. mistralai defaults these to an Unset()
        # sentinel, NOT to None -- an explicit None is serialised as
        # `"output_dimension": null` in the request body rather than omitted.
        dimensions = merged.get("dimensions")
        if dimensions:
            kwargs["output_dimension"] = dimensions
        extra_headers = merged.get("extra_headers")
        if extra_headers:
            kwargs["http_headers"] = extra_headers
        # `output_dtype` / `encoding_format` are never sent: the SDK types
        # EmbeddingResponseData.embedding as Optional[List[float]], so a base64 or
        # int8 response fails SDK-side validation, and a non-float vector would
        # break GatewayEmbeddings' float contract into pgvector.

        try:
            response = await self._client.embeddings.create_async(
                model=self._model_name, inputs=list(texts), **kwargs
            )
        except MistralError as e:
            # MistralError is the common base of SDKError (4xx/5xx, carrying
            # status_code and body) and HTTPValidationError (422), so one clause
            # covers both -- deliberately broader than pydantic-ai's own
            # models/mistral.py, which catches SDKError only and lets a 422 escape.
            status_code = getattr(e, "status_code", None)
            if status_code is not None and status_code >= 400:
                raise ModelHTTPError(
                    status_code=status_code,
                    model_name=self._model_name,
                    body=getattr(e, "body", None),
                ) from e
            raise ModelAPIError(model_name=self._model_name, message=str(e)) from e
        except httpx.RequestError as e:
            # Connection/timeout failures are not wrapped by mistralai; without
            # this they would surface as a raw httpx error from inside a
            # pydantic-ai call stack.
            raise ModelAPIError(model_name=self._model_name, message=str(e)) from e

        vectors = _ordered_embeddings(response, len(texts), self._model_name)
        usage = RequestUsage(
            input_tokens=(response.usage.prompt_tokens or 0) if response.usage else 0
        )
        return self._result(
            vectors,
            texts,
            input_type,
            usage,
            response.model or self._model_name,
            response.id,
        )

    def _result(
        self,
        vectors: list[list[float]],
        texts: list[str],
        input_type: EmbedInputType,
        usage: RequestUsage,
        model_name: str | None,
        response_id: str | None,
    ) -> EmbeddingResult:
        return EmbeddingResult(
            embeddings=vectors,
            inputs=texts,
            input_type=input_type,
            model_name=model_name or self._model_name,
            provider_name=_SYSTEM,
            usage=usage,
            provider_response_id=response_id,
        )


def _ordered_embeddings(
    response: Any, expected: int, model_name: str
) -> list[list[float]]:
    """Return the vectors in the caller's input order, or fail loudly.

    Order is not cosmetic here: ``GatewayEmbeddings`` hands the list straight on,
    and the pgvector loader zips it positionally against the rows being indexed.
    A silently permuted response therefore attaches every embedding to the wrong
    document and corrupts the index in a way no later check would catch, so every
    shape this cannot prove correct raises instead.
    """
    data = list(response.data or [])
    if len(data) != expected:
        raise UnexpectedModelBehavior(
            f"Mistral returned {len(data)} embeddings for {expected} inputs "
            f"(model {model_name!r}); refusing to guess which input each belongs to"
        )

    indexes = [getattr(d, "index", None) for d in data]
    present = [i for i in indexes if i is not None]
    if present and len(present) != len(indexes):
        # Some indexed, some not: unorderable. Guessing would be the corrupting move.
        raise UnexpectedModelBehavior(
            f"Mistral returned a mix of indexed and un-indexed embeddings "
            f"(model {model_name!r}); cannot restore input order"
        )
    if present:
        # sorted() is stable, so duplicates or gaps would silently survive a plain
        # sort and yield a wrong-but-plausible order -- require a true permutation
        # of range(expected) before trusting the indexes at all.
        if sorted(present) != list(range(expected)):
            raise UnexpectedModelBehavior(
                f"Mistral returned embedding indexes {sorted(present)} for "
                f"{expected} inputs (model {model_name!r}); expected each of "
                f"0..{expected - 1} exactly once"
            )
        data = sorted(data, key=lambda d: d.index)

    vectors: list[list[float]] = []
    for position, item in enumerate(data):
        if item.embedding is None:
            raise UnexpectedModelBehavior(
                f"Mistral returned a null embedding at position {position} "
                f"(model {model_name!r})"
            )
        vectors.append(list(item.embedding))
    return vectors
