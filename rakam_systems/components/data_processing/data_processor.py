import mimetypes
import os
import sys
from typing import Dict
from typing import List
import logging
logging.basicConfig(level=logging.INFO)

RAKAM_SYSTEMS_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))  # ingestion  # this file
)
sys.path.append(RAKAM_SYSTEMS_DIR)

import re

from docling.document_converter import DocumentConverter

from rakam_systems.core import VSFile, NodeMetadata, Node
from rakam_systems.components.data_processing import (
    PDFContentExtractor,
    JSONContentExtractor,
)
from rakam_systems.components.data_processing.node_processors import MarkdownSplitter

class DataProcessor:
    def __init__(self) -> None:
        self.default_content_extractors: Dict[str, callable] = {
            "application/pdf": PDFContentExtractor(
                parser_name="AdvancedPDFParser", output_format="markdown", persist=True
            ),
            "application/json": JSONContentExtractor(),
        }
        self.default_node_processors = MarkdownSplitter()

    def process_files_from_directory(self, directory_path: str) -> List[VSFile]:
        # Extract content from files
        print(f"Extracting content from files in {directory_path}")
        vs_files = []
        for root, _, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                mime_type, _ = mimetypes.guess_type(file_path)

                if mime_type in self.default_content_extractors:
                    content_extractor = self.default_content_extractors[mime_type]
                    vs_files.extend(content_extractor.extract_content(file_path))
                else:
                    print(
                        f"Skipping file {filename} with unsupported MIME type {mime_type}"
                    )

        # Process nodes
        print(f"Processing nodes in {len(vs_files)} files")
        for vs_file in vs_files:
            self.default_node_processors.process(vs_file)

        return vs_files

class Advanced_Data_Processor:
    def __init__(self) -> None:
        self.converter = DocumentConverter()
        self.doc_types = {
            "application/pdf", 
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # docx
            "application/vnd.openxmlformats-officedocument.presentationml.presentation", # pptx
            }

    def chunk_markdown_by_headers(self, md_text):
        """
        Chunk a Markdown text based on H1 and H2 headers.
        
        Parameters:
            md_text (str): The Markdown text string.
        
        Returns:
            list of dict: A list of chunks where each chunk is a dictionary with 'header' and 'content'.
        """
        # Regular expression to match H1 or H2 headers
        pattern = r"(^(#|##) .*)"
        
        # Find all headers and their positions in the text
        headers = [(match.group(0), match.start()) for match in re.finditer(pattern, md_text, re.MULTILINE)]
        
        chunks = []
        for i, (header, start_pos) in enumerate(headers):
            # Get the end position of the current section
            end_pos = headers[i + 1][1] if i + 1 < len(headers) else len(md_text)
            
            # Extract the content between the current header and the next one
            content = md_text[start_pos:end_pos].strip()
            
            # Add the chunk to the list
            chunks.append({'header': header, 'content': content})
        
        return chunks
    
    def process_files_from_directory(self, directory_path: str) -> List[VSFile]:
        vs_files = []
        for root, _, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                mime_type, _ = mimetypes.guess_type(file_path)

                if mime_type in self.doc_types:
                    vs_file = self.process_file(file_path)
                    vs_files.extend(vs_file)
                    logging.info(f"Processed file {filename}")
                else:
                    print(
                        f"Skipping file {filename} with unsupported MIME type {mime_type}"
                    )

        return vs_files
    
    def process_file(self, file_path: str) -> VSFile:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type in self.doc_types:
            result = self.converter.convert(file_path)
            md_text = result.document.export_to_markdown()
            chunks = self.chunk_markdown_by_headers(md_text)
            vs_file = VSFile(file_path)
            vs_file.nodes = [Node(metadata=NodeMetadata(file_path, i, {"header": chunk["header"]}), content=chunk['content']) for i, chunk in enumerate(chunks)]
            return [vs_file]
        else:
            raise ValueError(f"Unsupported MIME type {mime_type}")


if __name__ == "__main__":  # example usage
    processor = DataProcessor()
    vs_files = processor.process_files_from_directory("path/to/directory")

    advanced_processor = Advanced_Data_Processor()
    vs_files_advanced = advanced_processor.process_files_from_directory("path/to/directory")
