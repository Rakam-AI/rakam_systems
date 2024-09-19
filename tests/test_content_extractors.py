# import pytest
# import os
# import fitz  # PyMuPDF for PDF generation
# import json
# from rakam_systems.ingestion.data_processor import DataProcessor
# from rakam_systems.ingestion.node_processors import CharacterSplitter, MarkdownSplitter
# from rakam_systems.core import VSFile, Node, NodeMetadata


# @pytest.fixture
# def sample_pdf(tmp_path):
#     pdf_file = tmp_path / "sample.pdf"

#     # Create a simple PDF file
#     doc = fitz.open()
#     page = doc.new_page()
#     page.insert_text((72, 72), "Dummy PDF content for testing")
#     doc.save(pdf_file)
#     doc.close()

#     return str(pdf_file)


# @pytest.fixture
# def sample_json(tmp_path):
#     json_file = tmp_path / "sample.json"

#     # Create a simple JSON file
#     content = {
#         "content": "Sample content for testing.",
#         "metadata": {"author": "Test Author"},
#     }
#     with open(json_file, "w") as f:
#         json.dump(content, f)

#     return str(json_file)


# @pytest.fixture
# def sample_directory(tmp_path, sample_pdf, sample_json):
#     directory = tmp_path / "test_directory"
#     os.makedirs(directory)

#     # Move files to directory
#     os.rename(sample_pdf, directory / "sample.pdf")
#     os.rename(sample_json, directory / "sample.json")

#     return str(directory)


# def test_data_processor_with_pdf(sample_directory):
#     processor = DataProcessor()
#     vs_files = processor.process_files_from_directory(sample_directory)

#     assert len(vs_files) == 2  # Expecting 1 VSFile from PDF and 1 from JSON
#     assert isinstance(vs_files[0], VSFile)
#     assert len(vs_files[0].nodes) > 0


# def test_data_processor_with_unsupported_file_type(tmp_path):
#     processor = DataProcessor()

#     # Create a text file (unsupported file type)
#     text_file = tmp_path / "sample.txt"
#     text_file.write_text("This is a plain text file.")

#     vs_files = processor.process_files_from_directory(tmp_path)

#     # Since the text file is unsupported, no VSFile should be created
#     assert len(vs_files) == 0


# def test_character_splitter():
#     vs_file = VSFile("example.md")
#     content = "a" * 1050  # content longer than the max_characters
#     node = Node(
#         content=content,
#         metadata=NodeMetadata(source_file_uuid=vs_file.uuid, position=0),
#     )
#     vs_file.nodes = [node]

#     splitter = CharacterSplitter(max_characters=1024, overlap=20)
#     splitter.process(vs_file)

#     assert len(vs_file.nodes) == 2  # Should split into 2 nodes
#     assert vs_file.nodes[0].content == "a" * 1024
#     assert vs_file.nodes[1].content == "a" * 46  # Remaining characters + overlap


# def test_markdown_splitter():
#     vs_file = VSFile("example.md")
#     content = "# Header 1\n\nSome content\n\n## Header 2\n\nMore content"
#     node = Node(
#         content=content,
#         metadata=NodeMetadata(source_file_uuid=vs_file.uuid, position=0),
#     )
#     vs_file.nodes = [node]

#     splitter = MarkdownSplitter()
#     splitter.process(vs_file)

#     assert len(vs_file.nodes) == 2  # Two headers, so two nodes
#     assert vs_file.nodes[0].content == "# Header 1\n\nSome content"
#     assert vs_file.nodes[1].content == "## Header 2\n\nMore content"


# def test_process_files_from_directory_with_json(sample_directory):
#     processor = DataProcessor()
#     vs_files = processor.process_files_from_directory(sample_directory)

#     assert len(vs_files) == 2  # One PDF and one JSON file
#     assert any(
#         "Sample content for testing." in node.content
#         for vs_file in vs_files
#         for node in vs_file.nodes
#     )
# import pytest
# import os
# import fitz  # PyMuPDF for PDF generation
# import json
# from rakam_systems.ingestion.data_processor import DataProcessor
# from rakam_systems.ingestion.node_processors import CharacterSplitter, MarkdownSplitter
# from rakam_systems.core import VSFile, Node, NodeMetadata


# @pytest.fixture
# def sample_pdf(tmp_path):
#     pdf_file = tmp_path / "sample.pdf"

#     # Create a simple PDF file
#     doc = fitz.open()
#     page = doc.new_page()
#     page.insert_text((72, 72), "Dummy PDF content for testing")
#     doc.save(pdf_file)
#     doc.close()

#     return str(pdf_file)


# @pytest.fixture
# def sample_json(tmp_path):
#     json_file = tmp_path / "sample.json"

#     # Create a simple JSON file
#     content = {
#         "content": "Sample content for testing.",
#         "metadata": {"author": "Test Author"},
#     }
#     with open(json_file, "w") as f:
#         json.dump(content, f)

#     return str(json_file)


# @pytest.fixture
# def sample_directory(tmp_path, sample_pdf, sample_json):
#     directory = tmp_path / "test_directory"
#     os.makedirs(directory)

#     # Move files to directory
#     os.rename(sample_pdf, directory / "sample.pdf")
#     os.rename(sample_json, directory / "sample.json")

#     return str(directory)


# def test_data_processor_with_pdf(sample_directory):
#     processor = DataProcessor()
#     vs_files = processor.process_files_from_directory(sample_directory)

#     assert len(vs_files) == 2  # Expecting 1 VSFile from PDF and 1 from JSON
#     assert isinstance(vs_files[0], VSFile)
#     assert len(vs_files[0].nodes) > 0


# def test_data_processor_with_unsupported_file_type(tmp_path):
#     processor = DataProcessor()

#     # Create a text file (unsupported file type)
#     text_file = tmp_path / "sample.txt"
#     text_file.write_text("This is a plain text file.")

#     vs_files = processor.process_files_from_directory(tmp_path)

#     # Since the text file is unsupported, no VSFile should be created
#     assert len(vs_files) == 0


# def test_character_splitter():
#     vs_file = VSFile("example.md")
#     content = "a" * 1050  # content longer than the max_characters
#     node = Node(
#         content=content,
#         metadata=NodeMetadata(source_file_uuid=vs_file.uuid, position=0),
#     )
#     vs_file.nodes = [node]

#     splitter = CharacterSplitter(max_characters=1024, overlap=20)
#     splitter.process(vs_file)

#     assert len(vs_file.nodes) == 2  # Should split into 2 nodes
#     assert vs_file.nodes[0].content == "a" * 1024
#     assert vs_file.nodes[1].content == "a" * 46  # Remaining characters + overlap


# def test_markdown_splitter():
#     vs_file = VSFile("example.md")
#     content = "# Header 1\n\nSome content\n\n## Header 2\n\nMore content"
#     node = Node(
#         content=content,
#         metadata=NodeMetadata(source_file_uuid=vs_file.uuid, position=0),
#     )
#     vs_file.nodes = [node]

#     splitter = MarkdownSplitter()
#     splitter.process(vs_file)

#     assert len(vs_file.nodes) == 2  # Two headers, so two nodes
#     assert vs_file.nodes[0].content == "# Header 1\n\nSome content"
#     assert vs_file.nodes[1].content == "## Header 2\n\nMore content"


# def test_process_files_from_directory_with_json(sample_directory):
#     processor = DataProcessor()
#     vs_files = processor.process_files_from_directory(sample_directory)

#     assert len(vs_files) == 2  # One PDF and one JSON file
#     assert any(
#         "Sample content for testing." in node.content
#         for vs_file in vs_files
#         for node in vs_file.nodes
#     )
