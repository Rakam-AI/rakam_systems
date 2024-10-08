import os
import sys
from typing import List


RAKAM_SYSTEMS_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))  # ingestion  # this file
)
sys.path.append(RAKAM_SYSTEMS_DIR)

from rakam_systems.core import VSFile, NodeMetadata, Node

def parsed_url_to_VSFile(
    url: str, extracted_content: str, other_info: dict = None
) -> VSFile:
    """
    Convert a parsed URL to a VSFile.
    """
    vs_file = VSFile(file_path=url)
    node_metadata = NodeMetadata(
        source_file_uuid=vs_file.uuid,
        position=0,  # what is position for urls?
        custom=other_info,
    )
    node = Node(content=extracted_content, metadata=node_metadata)
    vs_file.nodes = [node]  # All content is in one node before processing
    return vs_file
