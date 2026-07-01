"""
Stable import surface for the PostgreSQL vector store.

This is the documented entry point for consumers (Django, FastAPI, or plain
Python)::

    from rakam_systems_vectorstore.stores import (
        ConfigurablePgVectorStore,
        VectorStoreConfig,
    )

Importing this module configures Django from the ``POSTGRES_*`` environment
variables when running outside a Django process (see
:mod:`rakam_systems_vectorstore._django_setup`). Inside a Django host it relies
on the host's existing configuration and does nothing.

To point the store at a database other than the ``POSTGRES_*`` defaults, call
:func:`ensure_django_configured` with an explicit ``DatabaseConfig`` *before*
importing from this module.
"""

from __future__ import annotations

from rakam_systems_vectorstore._django_setup import ensure_django_configured

# Bootstrap Django before importing the model-dependent store. Safe no-op when a
# Django host has already configured settings.
ensure_django_configured()

from rakam_systems_vectorstore.components.vectorstore.configurable_pg_vector_store import (  # noqa: E402
    ConfigurablePgVectorStore,
)
from rakam_systems_vectorstore.config import VectorStoreConfig  # noqa: E402

__all__ = [
    "ConfigurablePgVectorStore",
    "VectorStoreConfig",
    "ensure_django_configured",
]
