from typing import Any
from typing import Dict
from typing import Optional
from typing import TYPE_CHECKING

from django.db import models
from pgvector.django import VectorField

if TYPE_CHECKING:
    from rakam_systems.ai_vectorstore.core import Node


class Collection(models.Model):
    """
    Represents a collection of vector embeddings in the database.
    Model with explicit app_label to avoid Django app registration issues.
    """

    name = models.CharField(max_length=255, unique=True)
    embedding_dim = models.IntegerField(
        default=384
    )  # Default to a common embedding dimension
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "application"
        db_table = "application_collection"

    def __str__(self):
        return f"Collection: {self.name}"


class NodeEntry(models.Model):
    """
    Represents a stored node entry with content, metadata, and vector embedding.
    """

    collection = models.ForeignKey(
        Collection, on_delete=models.CASCADE, related_name="nodes"
    )
    content = models.TextField()
    # Use a standard embedding dimension (384 for all-MiniLM-L6-v2)
    embedding = VectorField(dimensions=384)

    # Node metadata
    node_id = models.AutoField(primary_key=True)
    source_file_uuid = models.CharField(max_length=255)
    position = models.IntegerField(null=True, blank=True)
    custom_metadata = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "application"
        db_table = "application_nodeentry"
        indexes = [
            models.Index(fields=["source_file_uuid"], name="application_source__idx"),
            models.Index(fields=["collection", "source_file_uuid"], name="application_collect_idx"),
        ]

    def __str__(self):
        return f"Node {self.node_id}: {self.content[:30]}..."

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node entry to a dictionary."""
        return {
            "node_id": self.node_id,
            "content": self.content,
            "source_file_uuid": self.source_file_uuid,
            "position": self.position,
            "custom": self.custom_metadata,
        }

    def to_node(self) -> "Node":
        """Convert the database entry to a Node object."""
        from rakam_systems.ai_vectorstore.core import Node, NodeMetadata

        metadata = NodeMetadata(
            source_file_uuid=self.source_file_uuid,
            position=self.position,
            custom=self.custom_metadata,
        )
        metadata.node_id = self.node_id

        node = Node(content=self.content, metadata=metadata)
        # Convert the database vector field to a numpy array
        node.embedding = self.embedding

        return node