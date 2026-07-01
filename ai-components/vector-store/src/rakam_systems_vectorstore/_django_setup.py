"""
Lazy Django bootstrap for standalone (non-Django) consumers.

The PostgreSQL vector store talks to Postgres through Django's ORM and
``django.db.connection``. Inside a Django process this is already wired up, but
when the package is used from a plain Python / FastAPI service there is no
``DJANGO_SETTINGS_MODULE`` and importing the store raises
``django.core.exceptions.ImproperlyConfigured``.

``ensure_django_configured`` configures a minimal Django settings object from the
package's :class:`~rakam_systems_vectorstore.config.DatabaseConfig` (i.e. the
``POSTGRES_*`` environment variables) and runs ``django.setup()`` — but **only**
when Django has not already been configured. A real Django host has
``settings.configured`` set, so this is a no-op there and never overrides the
host's configuration.

The store always connects through Django's ``default`` database connection (it
never reads ``vs_config.database`` for connecting), so bootstrapping from the
same ``POSTGRES_*`` env vars is consistent with the in-Django behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rakam_systems_vectorstore.config import DatabaseConfig

# App that owns the Collection / NodeEntry models. Its AppConfig (apps.py)
# registers under label "application", which the models declare explicitly.
_VECTORSTORE_APP = "rakam_systems_vectorstore.components.vectorstore"


def ensure_django_configured(db_config: "Optional[DatabaseConfig]" = None) -> None:
    """Configure and set up Django for standalone use, if not already done.

    Args:
        db_config: Database connection settings. Defaults to ``DatabaseConfig()``,
            which reads the ``POSTGRES_*`` environment variables.

    This is idempotent and safe to call from any number of import sites: once
    Django is configured (either by us or by a host Django project) it returns
    immediately.
    """
    from django.conf import settings

    if settings.configured:
        return

    import django
    from rakam_systems_vectorstore.config import DatabaseConfig

    db = db_config or DatabaseConfig()

    settings.configure(
        INSTALLED_APPS=[_VECTORSTORE_APP],
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": db.database,
                "USER": db.user,
                "PASSWORD": db.password,
                "HOST": db.host,
                "PORT": str(db.port),
            }
        },
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
        USE_TZ=True,
    )
    django.setup()
