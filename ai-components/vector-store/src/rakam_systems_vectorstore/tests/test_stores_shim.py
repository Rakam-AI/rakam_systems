"""Regression tests for the public ``stores`` import surface and standalone
Django bootstrap (issue #122).

Django configuration is process-global, so each scenario runs in its own
subprocess to keep them isolated from one another and from the rest of the
suite. No live database is required: Django connects lazily, so importing the
store and its models only needs settings to be configured.
"""

import os
import subprocess
import sys

import pytest

django = pytest.importorskip("django")  # only meaningful with the [postgres] extra

# Make the parent's import paths visible to the subprocesses.
_ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
_ENV.pop("DJANGO_SETTINGS_MODULE", None)


def _run(snippet: str, extra_env: dict | None = None) -> str:
    env = {**_ENV, **(extra_env or {})}
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_stores_import_resolves_documented_path():
    """The exact import from the issue must resolve outside Django."""
    out = _run(
        "from rakam_systems_vectorstore.stores import "
        "ConfigurablePgVectorStore, VectorStoreConfig;"
        "print(ConfigurablePgVectorStore.__name__, VectorStoreConfig.__name__)"
    )
    assert out == "ConfigurablePgVectorStore VectorStoreConfig"


def test_standalone_bootstrap_uses_postgres_env():
    """Importing the shim configures Django from POSTGRES_* when unconfigured."""
    out = _run(
        "import rakam_systems_vectorstore.stores;"
        "from django.conf import settings;"
        "print(settings.DATABASES['default']['NAME'], settings.DATABASES['default']['HOST'])",
        extra_env={"POSTGRES_DB": "tickets", "POSTGRES_HOST": "db.example.internal"},
    )
    assert out == "tickets db.example.internal"


def test_django_dependent_models_import_standalone():
    """pg_models (the original ImproperlyConfigured trigger) imports cleanly."""
    out = _run(
        "import rakam_systems_vectorstore.stores;"
        "from rakam_systems_vectorstore.components.vectorstore.pg_models "
        "import Collection, NodeEntry;"
        "print(Collection.__name__, NodeEntry.__name__)"
    )
    assert out == "Collection NodeEntry"


def test_existing_django_config_is_not_overridden():
    """A host that already configured Django keeps its own DATABASES."""
    out = _run(
        "import django;"
        "from django.conf import settings;"
        "settings.configure("
        "  INSTALLED_APPS=['rakam_systems_vectorstore.components.vectorstore'],"
        "  DATABASES={'default': {'ENGINE': 'django.db.backends.postgresql',"
        "    'NAME': 'HOST_OWNED_DB', 'USER': 'u', 'PASSWORD': 'p',"
        "    'HOST': 'host.db', 'PORT': '5432'}},"
        "  DEFAULT_AUTO_FIELD='django.db.models.BigAutoField', USE_TZ=True);"
        "django.setup();"
        "import rakam_systems_vectorstore.stores;"
        "print(settings.DATABASES['default']['NAME'])",
        extra_env={"POSTGRES_DB": "SHOULD_NOT_WIN"},
    )
    assert out == "HOST_OWNED_DB"
