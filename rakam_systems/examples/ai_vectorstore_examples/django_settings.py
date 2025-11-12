"""
Minimal Django settings for running PostgreSQL Vector Store examples.

This is a standalone Django configuration that allows the PgVectorStore
to work independently of any Django project.
"""

import os
from pathlib import Path

# Build paths
BASE_DIR = Path(__file__).resolve().parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "django-insecure-example-key-for-vectorstore-only-change-in-production"
)

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ["*"]

# Application definition
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.postgres",
    "ai_vectorstore.components.vectorstore.apps.VectorStoreConfig",  # For PgVectorStore models
]

# Database
# Read from environment variables or use defaults
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("POSTGRES_DB", "vectorstore_db"),
        "USER": os.getenv("POSTGRES_USER", "postgres"),
        "PASSWORD": os.getenv("POSTGRES_PASSWORD", "postgres"),
        "HOST": os.getenv("POSTGRES_HOST", "localhost"),
        "PORT": os.getenv("POSTGRES_PORT", "5432"),
    }
}

# Required for Django to work
USE_TZ = True
TIME_ZONE = "UTC"

# Middleware (minimal)
MIDDLEWARE = []

# Templates (not needed but Django requires it)
TEMPLATES = []

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
