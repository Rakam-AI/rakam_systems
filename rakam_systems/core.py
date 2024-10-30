import mimetypes
import uuid
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


class VSFile:
    """
    A data source to be processed. Its nodes will become entries in the VectorStore.
    """

    def __init__(self, file_path: str) -> None:
        self.uuid: str = uuid.uuid4()
        self.file_path: str = file_path
        self.file_name: str = file_path.split("/")[-1]
        self.mime_type, _ = mimetypes.guess_type(self.file_path)
        self.nodes: Optional[Node] = []
        self.processed: bool = (
            False  # whether the nodes of this file have been processed
        )
    
    def to_dict(self) -> Dict:
        return {
            "uuid": self.uuid,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "mime_type": self.mime_type,
            "nodes": [node.to_dict() for node in self.nodes] if self.nodes else [],
            "processed": self.processed,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "VSFile":
        vs_file = cls(file_path=data["file_path"])
        vs_file.uuid = data["uuid"]
        vs_file.file_name = data["file_name"]
        vs_file.mime_type = data["mime_type"]
        vs_file.nodes = [Node.from_dict(node_data) for node_data in data["nodes"]]
        vs_file.processed = data["processed"]
        return vs_file


class NodeMetadata:
    def __init__(
        self, source_file_uuid: str, position: int, custom: dict = None
    ) -> None:
        self.node_id: Optional[int] = None
        self.source_file_uuid: str = source_file_uuid
        self.position: Optional[int] = position  # page_number
        self.custom: Optional[Dict] = custom

    def __str__(self) -> str:
        custom_str = ", ".join(
            f"{key}: {value}" for key, value in (self.custom or {}).items()
        )
        return (
            f"NodeMetadata(node_id={self.node_id}, source_file_uuid='{self.source_file_uuid}', "
            f"position={self.position}, custom={{ {custom_str} }})"
        )

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "source_file_uuid": self.source_file_uuid,
            "position": self.position,
            "custom": self.custom,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "NodeMetadata":
        return cls(
            source_file_uuid=data.get("source_file_uuid", ""),
            position=data.get("position", None),
            custom=data.get("custom", {})
        )
    


class Node:
    """
    A node with content and associated metadata.
    """

    def __init__(self, content: str, metadata: NodeMetadata) -> None:
        self.content: str = content
        self.metadata: Optional[NodeMetadata] = metadata
        self.embedding: Optional[Any] = None

    def __str__(self) -> str:
        return f"Node(content='{self.content[:30]}...', metadata={self.metadata})"
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "embedding": None,  # Modify if embedding is serializable
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        return cls(
            content=data["content"],
            metadata=NodeMetadata.from_dict(data["metadata"]),
        )