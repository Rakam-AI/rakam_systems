"""
Unit tests for Adaptive Loader.
"""

import os
import tempfile
import unittest
from pathlib import Path

from rakam_systems.ai_vectorstore.components.loader.adaptive_loader import AdaptiveLoader, create_adaptive_loader
from rakam_systems.ai_vectorstore.core import Node


class TestAdaptiveLoader(unittest.TestCase):
    """Test AdaptiveLoader class."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.loader = AdaptiveLoader()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test loader initialization."""
        self.assertEqual(self.loader.name, "adaptive_loader")
        self.assertEqual(self.loader._encoding, "utf-8")
        self.assertEqual(self.loader._chunk_size, 512)
        self.assertEqual(self.loader._chunk_overlap, 50)
    
    def test_custom_config(self):
        """Test loader with custom configuration."""
        config = {
            "chunk_size": 256,
            "chunk_overlap": 25,
            "encoding": "latin-1"
        }
        loader = AdaptiveLoader(config=config)
        
        self.assertEqual(loader._chunk_size, 256)
        self.assertEqual(loader._chunk_overlap, 25)
        self.assertEqual(loader._encoding, "latin-1")
    
    def test_file_type_detection(self):
        """Test file type detection."""
        test_cases = [
            ("file.txt", "text"),
            ("file.md", "markdown"),
            ("file.pdf", "pdf"),
            ("file.docx", "docx"),
            ("file.json", "json"),
            ("file.csv", "csv"),
            ("file.html", "html"),
            ("file.py", "code"),
            ("file.js", "code"),
            ("file.unknown", "unknown"),
        ]
        
        for filename, expected_type in test_cases:
            path = Path(filename)
            detected_type = self.loader._detect_file_type(path)
            self.assertEqual(detected_type, expected_type, f"Failed for {filename}")
    
    def test_load_text_string(self):
        """Test loading raw text string."""
        text = "This is a test string for the adaptive loader."
        chunks = self.loader.run(text)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIn("test string", chunks[0])
    
    def test_load_text_file(self):
        """Test loading text file."""
        # Create test file
        test_content = "This is line 1.\nThis is line 2.\nThis is line 3."
        test_file = os.path.join(self.temp_dir, "test.txt")
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        chunks = self.loader.run(test_file)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    def test_load_json_file(self):
        """Test loading JSON file."""
        import json
        
        test_data = {"key1": "value1", "key2": "value2"}
        test_file = os.path.join(self.temp_dir, "test.json")
        
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        chunks = self.loader.run(test_file)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    def test_load_csv_file(self):
        """Test loading CSV file."""
        import csv
        
        test_file = os.path.join(self.temp_dir, "test.csv")
        
        with open(test_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'age', 'city'])
            writer.writeheader()
            writer.writerow({'name': 'Alice', 'age': '30', 'city': 'NYC'})
            writer.writerow({'name': 'Bob', 'age': '25', 'city': 'LA'})
        
        chunks = self.loader.run(test_file)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIn("Alice", chunks[0])
    
    def test_load_markdown_file(self):
        """Test loading markdown file."""
        markdown_content = """# Title

This is a paragraph.

## Section 1

Content for section 1.

## Section 2

Content for section 2.
"""
        test_file = os.path.join(self.temp_dir, "test.md")
        
        with open(test_file, 'w') as f:
            f.write(markdown_content)
        
        chunks = self.loader.run(test_file)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    def test_load_as_nodes(self):
        """Test loading data as Node objects."""
        text = "Test content for nodes."
        custom_metadata = {"category": "test", "priority": "high"}
        
        nodes = self.loader.load_as_nodes(text, source_id="test_source", custom_metadata=custom_metadata)
        
        self.assertIsInstance(nodes, list)
        self.assertGreater(len(nodes), 0)
        
        for node in nodes:
            self.assertIsInstance(node, Node)
            self.assertEqual(node.metadata.source_file_uuid, "test_source")
            self.assertEqual(node.metadata.custom.get("category"), "test")
            self.assertEqual(node.metadata.custom.get("priority"), "high")
    
    def test_load_as_vsfile(self):
        """Test loading file as VSFile object."""
        test_content = "This is test content for VSFile."
        test_file = os.path.join(self.temp_dir, "test.txt")
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        custom_metadata = {"type": "document"}
        vsfile = self.loader.load_as_vsfile(test_file, custom_metadata=custom_metadata)
        
        self.assertEqual(vsfile.file_path, test_file)
        self.assertIsInstance(vsfile.nodes, list)
        self.assertGreater(len(vsfile.nodes), 0)
        self.assertTrue(vsfile.processed)
        
        # Check metadata
        for node in vsfile.nodes:
            self.assertEqual(node.metadata.custom.get("type"), "document")
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        # Should treat as text
        result = self.loader.run("/nonexistent/file.txt")
        self.assertIsInstance(result, list)
    
    def test_chunking(self):
        """Test text chunking."""
        # Create long text
        long_text = "word " * 1000  # Creates text > chunk_size
        
        chunks = self.loader._chunk_text(long_text, "test")
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)  # Should be split into multiple chunks
        
        # Verify chunk sizes
        for chunk in chunks:
            self.assertLessEqual(len(chunk), self.loader._chunk_size * 1.2)  # Allow some variance
    
    def test_empty_input(self):
        """Test with empty input."""
        chunks = self.loader._process_text("", "empty")
        self.assertEqual(chunks, [])
        
        chunks = self.loader._chunk_text("   ", "whitespace")
        self.assertEqual(chunks, [])
    
    def test_process_markdown_structure(self):
        """Test markdown structure preservation."""
        markdown = """# Header 1
Content 1

# Header 2
Content 2

# Header 3
Content 3
"""
        chunks = self.loader._process_markdown(markdown)
        
        # Should have multiple chunks (one per section)
        self.assertGreater(len(chunks), 1)
        self.assertIn("# Header 1", chunks[0])
    
    def test_process_json_dict(self):
        """Test JSON dict processing."""
        data = {"key1": "value1", "key2": {"nested": "value"}}
        chunks = self.loader._process_json(data)
        
        self.assertEqual(len(chunks), 1)
        self.assertIn("key1", chunks[0])
    
    def test_process_json_list(self):
        """Test JSON list processing."""
        data = [{"item": 1}, {"item": 2}, {"item": 3}]
        chunks = self.loader._process_json(data)
        
        self.assertEqual(len(chunks), 3)
    
    def test_process_csv_rows(self):
        """Test CSV rows processing."""
        rows = [
            {"name": "Alice", "age": "30"},
            {"name": "Bob", "age": "25"}
        ]
        chunks = self.loader._process_csv(rows)
        
        self.assertEqual(len(chunks), 2)
        self.assertIn("Alice", chunks[0])
        self.assertIn("Bob", chunks[1])


class TestCreateAdaptiveLoader(unittest.TestCase):
    """Test factory function."""
    
    def test_default_creation(self):
        """Test creating loader with defaults."""
        loader = create_adaptive_loader()
        
        self.assertIsInstance(loader, AdaptiveLoader)
        self.assertEqual(loader._chunk_size, 512)
        self.assertEqual(loader._chunk_overlap, 50)
    
    def test_custom_creation(self):
        """Test creating loader with custom parameters."""
        loader = create_adaptive_loader(
            chunk_size=1024,
            chunk_overlap=100,
            encoding='utf-16'
        )
        
        self.assertEqual(loader._chunk_size, 1024)
        self.assertEqual(loader._chunk_overlap, 100)
        self.assertEqual(loader._encoding, 'utf-16')


if __name__ == '__main__':
    unittest.main()

