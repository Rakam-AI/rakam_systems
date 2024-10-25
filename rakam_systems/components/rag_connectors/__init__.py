# components/__init__.py

# Importing using absolute imports
from rakam_systems.components.rag_connectors.generation_feeder import GenerationFeeder

# Define what is available when importing * from this package
__all__ = [
    "GenerationFeeder"
    ]