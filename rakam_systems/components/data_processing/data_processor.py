import logging
import mimetypes
import os
import sys
from typing import Dict
from typing import List

RAKAM_SYSTEMS_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))  # ingestion  # this file
)
sys.path.append(RAKAM_SYSTEMS_DIR)

from rakam_systems.system_manager import SystemManager
from rakam_systems.core import VSFile
from rakam_systems.components.data_processing import (
    PDFContentExtractor,
    JSONContentExtractor,
)
from rakam_systems.components.data_processing.node_processors import MarkdownSplitter
from rakam_systems.components.component import Component


class DataProcessor(Component):
    def __init__(self, system_manager: SystemManager) -> None:
        self.default_content_extractors: Dict[str, callable] = {
            "application/pdf": PDFContentExtractor(
                parser_name="AdvancedPDFParser", output_format="markdown", persist=True
            ),
            "application/json": JSONContentExtractor(),
        }
        self.default_node_processors = MarkdownSplitter()
        self.system_manager = system_manager
        logging.info("Data Processor initialized")

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
    
    def call_process_file(self, file_path: str) -> List[VSFile]:
        """
        Processes a single file, extracting content and processing nodes if the MIME type is supported.

        Parameters:
        file_path (str): Path to the file to be processed.

        Returns:
        List[VSFile]: A list of processed VSFile objects.
        """
        mime_type, _ = mimetypes.guess_type(file_path)

        # Check if the file's MIME type is supported
        if mime_type in self.default_content_extractors:
            content_extractor = self.default_content_extractors[mime_type]
            vs_files = content_extractor.extract_content(file_path)  # This should be a list

            # Process nodes in each extracted VSFile object
            for vs_file in vs_files:
                self.default_node_processors.process(vs_file)
            
            serialized_files = [vs_file.to_dict() for vs_file in vs_files]

            return serialized_files
        else:
            print(f"Skipping file {os.path.basename(file_path)} with unsupported MIME type {mime_type}")
            return []
    
    def call_main(self,directory_path):
        vs_files = self.process_files_from_directory(directory_path)
        serialized_files = [vs_file.to_dict() for vs_file in vs_files]
        return serialized_files
    
    def test(self, directory_path = "data"):
        vs_files = self.call_search_from_collection(directory_path)
        
        return vs_files[0].nodes[0].content
    
if __name__ == "__main__":  # example usage
    system_manager = SystemManager(system_config_path="system_config.yaml")
    processor = DataProcessor(system_manager=system_manager)
    
    vs_files = processor.call_process_file("data/files/1706.03762v7.pdf")
    print(vs_files)
