"""
Django App Configuration for Vector Store
"""
from django.apps import AppConfig


class VectorStoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "rakam_systems_vectorstore.components.vectorstore"
    label = "application"  # Match the app_label in models
