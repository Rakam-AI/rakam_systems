import os
import re
import sys
from abc import ABC
from abc import abstractmethod
from typing import List


RAKAM_SYSTEMS_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))  # ingestion  # this file
)
sys.path.append(RAKAM_SYSTEMS_DIR)

from rakam_systems.core import VSFile, NodeMetadata, Node


# This takes big chunks and split them in more granularity

class NodeProcessor(ABC):
    """
    Abstract base class for processing nodes in a VSFile.
    Subclasses should implement the `process` method.
    """

    @abstractmethod
    def process(self, vs_file: VSFile) -> None:
        """
        Process the nodes of a VSFile and modify the VSFile in-place.
        """
        pass

    def _build_node_from_chunk(self, content: str, metadata: NodeMetadata) -> Node:
        chunk_metadata = NodeMetadata(
            source_file_uuid=metadata.source_file_uuid,
            position=metadata.position,  # eg. page_number
            custom=metadata.custom.copy() if metadata.custom else {},
        )
        return Node(content=content, metadata=chunk_metadata)


class CharacterSplitter(NodeProcessor):
    """
    Splits the content of each Node based on a specified number of characters.

    Attributes:
        max_characters(int): The maximum number of characters in each chunk.
        overlap(int): The number of characters to overlap between chunks.

    Methods:
        process(vs_file: VSFile) -> None:
            Split the content of each Node in the VSFile into chunks.
    """

    def __init__(self, max_characters: int = 1024, overlap: int = 20):
        self.max_characters = max_characters
        self.overlap = overlap

    def process(self, vs_file: VSFile) -> None:
        chunked_nodes = []
        for node in vs_file.nodes:
            content = node.content
            metadata = node.metadata
            start_idx = 0
            while start_idx < len(content):
                # Calculate the start and end idxs for the chunk
                end_idx = min(start_idx + self.max_characters, len(content))
                chunk = content[start_idx:end_idx]

                chunk_node = self._build_node_from_chunk(
                    content=chunk, metadata=metadata
                )
                chunked_nodes.append(chunk_node)

                # Update the position for the next chunk, considering the overlap
                start_idx = end_idx - self.overlap

        vs_file.nodes = (
            chunked_nodes  # replace the original nodes with the chunked nodes
        )
        vs_file.processed = True  # mark file as processed


class MarkdownSplitter(NodeProcessor):
    """
    Splits the content of each Node in a VSFile based on Markdown headers.

    Methods:
        process(vs_file: VSFile) -> None:
            Split the content of each Node in the VSFile into chunks based on Markdown headers.
    """

    def process(self, vs_file: VSFile) -> None:
        chunked_nodes = []
        for node in vs_file.nodes:
            content = node.content
            metadata = node.metadata
            markdown_nodes = self._split_content(content, metadata, node)
            chunked_nodes.extend(markdown_nodes)

        vs_file.nodes = (
            chunked_nodes  # replace the original nodes with the chunked nodes
        )
        vs_file.processed = True  # mark file as processed

        return markdown_nodes

    def _split_content(
        self, content: str, metadata: NodeMetadata, node: Node
    ) -> List[Node]:
        markdown_nodes = []
        lines = content.split("\n")
        current_chunk_lines = []
        current_metadata = metadata
        code_block = False

        for line in lines:
            if line.lstrip().startswith("```"):
                code_block = (
                    not code_block
                )  # switch from True to False or vice versa (beginning or end of code block)

            header_match = re.match(r"^(#+)\s(.*)", line)
            if header_match and not code_block:
                if current_chunk_lines:
                    chunk_content = "\n".join(current_chunk_lines).strip()
                    chunk_node = self._build_node_from_chunk(
                        content=chunk_content, metadata=current_metadata
                    )
                    markdown_nodes.append(chunk_node)
                    current_chunk_lines = []

                level = len(header_match.group(1))
                header = header_match.group(2)
                current_metadata = self._update_metadata(metadata, header, level)
            current_chunk_lines.append(line)

        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines).strip()
            chunk_node = self._build_node_from_chunk(
                content=chunk_content, metadata=current_metadata
            )
            markdown_nodes.append(chunk_node)

        return markdown_nodes

    def _update_metadata(
        self, metadata: NodeMetadata, header: str, level: int
    ) -> NodeMetadata:
        new_metadata = metadata.custom.copy() if metadata.custom else {}
        new_metadata.update({"header": header, "level": level})
        return NodeMetadata(
            source_file_uuid=metadata.source_file_uuid,
            position=metadata.position,  # eg. page_number
            custom=new_metadata,
        )


if __name__ == "__main__":  # example usage
    vs_file = VSFile("example.md")
    node = Node(
        content="# Header 1\n\nSome content\n\n## Header 2\n\nMore content",
        metadata=NodeMetadata(source_file_uuid=vs_file.uuid, position=0),
    )
    vs_file.nodes = [node]

    splitter = MarkdownSplitter()
    splitter.process(vs_file)
