import mimetypes
import os
import sys
from typing import Dict
from typing import List

RAKAM_SYSTEMS_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))  # ingestion  # this file
)
sys.path.append(RAKAM_SYSTEMS_DIR)

from rakam_systems.core import VSFile
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


if __name__ == "__main__":  # example usage
    processor = DataProcessor()
    vs_files = processor.process_files_from_directory("path/to/directory")
