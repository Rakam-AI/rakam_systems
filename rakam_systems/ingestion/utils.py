import os
import sys
from typing import List


RAKAM_SYSTEMS_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))  # ingestion  # this file
)
sys.path.append(RAKAM_SYSTEMS_DIR)

from rakam_systems.core import VSFile, NodeMetadata, Node


def llama_documents_to_VSFiles(llama_documents) -> List[VSFile]:
    """
    Convert a list of LlamaIndex documents (from different sources) to a list of VSFiles.
    """
    vs_files = []
    page_number_tracker = 1
    current_file_name = None

    for doc in llama_documents:
        file_name = doc.metadata["file_name"]
        if file_name != current_file_name:  # new file
            current_file_name = file_name
            page_number_tracker = 1
            vs_file = VSFile(file_name)  # create a new VSFile
            vs_files.append(vs_file)

        # create a new node
        node_metadata = NodeMetadata(
            source_file_uuid=vs_file.uuid, position=page_number_tracker
        )
        node = Node(doc.text, node_metadata)
        vs_file.nodes.append(node)
        page_number_tracker += 1

    return vs_files


def llama_documents_to_VSFile(llama_documents) -> VSFile:
    """
    Convert a list of LlamaIndex documents (from the same source) to a VSFile.
    """
    file_name = llama_documents[0].metadata["file_name"]
    vs_file = VSFile(file_name)
    page_number_tracker = 1

    for doc in llama_documents:
        node_metadata = NodeMetadata(
            source_file_uuid=vs_file.uuid, position=page_number_tracker
        )
        node = Node(doc.text, node_metadata)
        vs_file.nodes.append(node)
        page_number_tracker += 1

    return vs_file


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
