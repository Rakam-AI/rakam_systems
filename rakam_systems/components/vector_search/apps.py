from django.apps import AppConfig


class VectorStoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "vector_store"

    def ready(self):
        """
        Ensure pgvector extension is installed when the Django app starts.
        """
        # Import is done here to avoid import loops
        from django.db import connection

        with connection.cursor() as cursor:
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except Exception as e:
                print(f"Error creating pgvector extension: {e}")
                # Don't raise exception here to allow app to start even if extension creation fails
                # This allows for more graceful error handling in the application
