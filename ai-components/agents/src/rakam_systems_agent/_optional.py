"""Deferred imports for the symbols that need an optional extra.

``psycopg2`` (extra ``postgres``) and ``openai`` / ``mistralai`` / ``tiktoken``
(extra ``llm-providers``) are optional *by declaration* in ``pyproject.toml``,
but every ``__init__`` on the way down used to import them eagerly -- so a
correct minimal install could not even ``import rakam_systems_agent``:

    rakam_systems_agent/__init__.py -> components -> chat_history
      -> postgres_chat_history -> psycopg2.pool   # ModuleNotFoundError

That made the extras mandatory in practice, and locked out consumers that need
none of them (``ModelGateway`` uses pydantic-ai only).

Each re-exporting ``__init__`` now declares the affected names here and gets a
PEP 562 module ``__getattr__``: the third-party import is attempted on first
attribute access instead of at import time. Import paths are unchanged for
anyone who did install the extra, and anyone who did not gets an ``ImportError``
naming the extra to install, at the point of use.
"""
from __future__ import annotations

import sys
from importlib import import_module
from typing import Callable, Dict, Iterable, Tuple

_PACKAGE = "rakam_systems_agent"

# symbol -> (module that defines it, extra whose dependency it needs)
OPTIONAL_SYMBOLS: Dict[str, Tuple[str, str]] = {
    "PostgresChatHistory": (
        f"{_PACKAGE}.components.chat_history.postgres_chat_history",
        "postgres",
    ),
    "OpenAIGateway": (
        f"{_PACKAGE}.components.llm_gateway.openai_gateway",
        "llm-providers",
    ),
    "MistralGateway": (
        f"{_PACKAGE}.components.llm_gateway.mistral_gateway",
        "llm-providers",
    ),
    "LLMGatewayFactory": (
        f"{_PACKAGE}.components.llm_gateway.gateway_factory",
        "llm-providers",
    ),
    "get_llm_gateway": (
        f"{_PACKAGE}.components.llm_gateway.gateway_factory",
        "llm-providers",
    ),
}


def load(name: str):
    """Import and return the optional symbol ``name``.

    Raises:
        ImportError: If the extra that ships the dependency is not installed.
            The message names the extra; the original ``ModuleNotFoundError``
            stays attached as ``__cause__``.
    """
    module_name, extra = OPTIONAL_SYMBOLS[name]
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{name} needs the optional '{extra}' extra: "
            f"install it with `pip install \"rakam-systems-agent[{extra}]\"` "
            f"(missing dependency: {exc.name})"
        ) from exc
    return getattr(module, name)


def make_getattr(module_name: str, names: Iterable[str]) -> Callable[[str], object]:
    """Build the PEP 562 ``__getattr__`` for a package exporting ``names``.

    The resolved symbol is written back onto the module, so the deferred import
    is paid once and ``dir()`` reports the name from then on.
    """
    deferred = frozenset(names)

    def __getattr__(name: str) -> object:
        if name not in deferred:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        value = load(name)
        setattr(sys.modules[module_name], name, value)
        return value

    return __getattr__
