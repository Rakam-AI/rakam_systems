import logging
import os
import re
from typing import List

from docling.document_converter import DocumentConverter

from rakam_systems.system_manager import SystemManager
from rakam_systems.components.component import Component
from rakam_systems.core import VSFile, NodeMetadata, Node

class Advanced_Data_Processor(Component):
    def __init__(self, system_manager: SystemManager = None) -> None:
        self.converter = DocumentConverter()
        self.system_manager = system_manager
        logging.info("Advanced Data Processor initialized")

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
                result = self.converter.convert(file_path)
                md_text = result.document.export_to_markdown()
                chunks = self.chunk_markdown_by_headers(md_text)
                vs_file = VSFile(file_path)
                vs_file.nodes = [Node(metadata=NodeMetadata(vs_file.uuid, i), content=chunk['content']) for i, chunk in enumerate(chunks)]
                vs_files.append(vs_file)

        return vs_files

    def call_main(self, **kwargs) -> dict:
        return super().call_main(**kwargs)
    
    def test(self, **kwargs) -> bool:
        return super().test(**kwargs)

# if __name__ == "__main__":
#     processor = Advanced_Data_Processor()
#     vs_files = processor.process_files_from_directory("data/pdfs")
#     for vs_file in vs_files:
#         print(vs_file)
#         for node in vs_file.nodes:
#             print(node)
#             print("-" * 50)
#             print(node.content)
#             print("#" * 50)
#         print("=" * 50)
#         print()


