import logging
import mimetypes
import requests

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
    
    def call_process_file(self, file_path: str, file_uuid = None, download_dir: str = "downloads") -> List[dict]:
        """
        Processes a single file or URL, extracting content and processing nodes if the MIME type is supported.
        Downloads the file to a specified directory if it's a URL.

        Parameters:
        file_path (str): Path to the file or URL to be processed.
        download_dir (str): Directory where files are downloaded (if file_path is a URL).

        Returns:
        List[dict]: A list of processed VSFile objects in dictionary format.
        """
        def is_url(path: str) -> bool:
            """Check if the given path is a URL."""
            return path.startswith("http://") or path.startswith("https://")

        if is_url(file_path):
            # Handle URL case
            try:
                # Ensure the download directory exists
                os.makedirs(download_dir, exist_ok=True)
                
                # Get the file name from Content-Disposition or fallback to the URL's file name
                response = requests.get(file_path, stream=True)
                response.raise_for_status()
                
                content_disposition = response.headers.get("Content-Disposition", "")
                if "filename=" in content_disposition:
                    file_name = content_disposition.split("filename=")[-1].strip('"')
                else:
                    file_name = os.path.basename(file_path)
                
                # Save the file with the determined file name
                local_path = os.path.join(download_dir, file_name)
                with open(local_path, 'wb') as file:
                    file.write(response.content)
                
                temp_file_path = local_path
                mime_type, _ = mimetypes.guess_type(temp_file_path)
            except Exception as e:
                print(f"Failed to download or process URL {file_path}: {e}")
                return []
        else:
            # Handle local file case
            temp_file_path = file_path
            mime_type, _ = mimetypes.guess_type(file_path)

        # Check if the file's MIME type is supported
        if mime_type in self.default_content_extractors:
            content_extractor = self.default_content_extractors[mime_type]
            vs_files = content_extractor.extract_content(temp_file_path, file_uuid)  # This should be a list

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
    

