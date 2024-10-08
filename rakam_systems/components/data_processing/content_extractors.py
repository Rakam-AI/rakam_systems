import glob
import json
import os
import re
import sys
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Type

import dotenv
import joblib
import pymupdf
import requests
import pymupdf4llm
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


# Content extractors look at a raw file ( PDF, URL,... ), transforms to markdown, splits into big chungs (eg. page)

dotenv.load_dotenv()

import logging

from rakam_systems.components.data_processing import utils


RAKAM_SYSTEMS_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))  # ingestion  # this file
)
sys.path.append(RAKAM_SYSTEMS_DIR)

from rakam_systems.core import VSFile, NodeMetadata, Node


class ContentExtractor(ABC):
    """
    Interface for content extractors.

    Attributes:
        output_format (str): The desired output format of the extracted content

    Methods:
        extract_content: Fetch and parse content from the source
        format_content: Format the parsed content before returning it
    """

    def __init__(self, output_format: str = "markdown"):
        self.output_format = output_format

    @abstractmethod
    def extract_content(self, source):
        pass

    def format_content(self, content):
        # Optional
        # To format the content before returning it, override this method.
        return content


### --- PDFs --- ###
class SimplePDFParser:
    """
    A simple PDF parser that extracts text from PDF files using PyMuPDF.
    """

    def __init__(self) -> None:
        pass

    def parse_one_pdf(self, file_path: str):
        try:
            doc = pymupdf.open(file_path)
        except Exception as e:
            logging.error(f"Error reading PDF file: {file_path}")
            return None

        vs_file = VSFile(file_path)  # create a new VSFile
        nodes = []
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            text = self._clean_text(text)
            if text:
                metadata = NodeMetadata(
                    source_file_uuid=file_path, position=(page_num + 1)
                )
                node = Node(text, metadata)
                nodes.append(node)
        vs_file.nodes = nodes
        return vs_file

    def _clean_text(self,text):
        # Remove non-printable and control characters
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


class AdvancedPDFParser:
    """
    A PDF parser that extracts text from PDF files using pymupdf4llm.
    """

    def __init__(self, output_format: str = "markdown") -> None:
        self.output_format = output_format

    def parse_one_pdf(self, file_path: str):
        # try:
            # Returns a list of dicts, each representing a page
        doc = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        # except Exception as e:
        #     logging.error(f"Error reading PDF file: {file_path}")
        #     return None

        vs_file = VSFile(file_path)  # create a new VSFile
        nodes = []
        for page_num, page_content in enumerate(doc):
            text = page_content["text"].strip()
            if text:
                metadata = NodeMetadata(
                    source_file_uuid=file_path, position=(page_num + 1)
                )
                node = Node(text, metadata)
                nodes.append(node)
        vs_file.nodes = nodes
        return vs_file


class PDFContentExtractor(ContentExtractor):
    """
    Extracts content from PDF files.

    Attributes:
        output_format (str): The desired output format of the extracted content
        parser_name (str): The name of the parser to use (default: "AdvancedPDFParser")
            - "SimplePDFParser"
            - "AdvancedPDFParser"

        persist (bool): Persist the parsed documents (default: True) (LlamaParse only)

    Methods:
        get_parser: Initialize the parser
        extract_content: Parse content from the PDFs (pass path to file or directory)
    """

    def __init__(self, parser_name: str = "AdvancedPDFParser", persist: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.parser_name = parser_name
        self.persist = persist
        self.parser = self._get_parser(parser_name, persist=persist)

    def _get_parser(self, parser_name: str, persist: bool):
        if parser_name == "SimplePDFParser":
            return SimplePDFParser()
        elif parser_name == "AdvancedPDFParser":
            return AdvancedPDFParser(output_format=self.output_format)
        else:
            raise NotImplementedError(
                f"Unsupported parser: {parser_name}. Use 'SimplePDFParser', or 'AdvancedPDFParser'."
            )

    def _load_parsed_files(self, parsed_files) -> VSFile:
        """
        Load previously parsed files (all pkls for a PDF).
        """
        documents = []
        for file_path in parsed_files:
            doc = joblib.load(file_path)
            documents.append(doc)
        VSFile = utils.llama_documents_to_VSFile(documents)
        return VSFile

    def _load_or_parse_pdf(self, file_path) -> VSFile:
        """
        Determine whether to load or parse a PDF file.
        """
        filename = os.path.basename(file_path)
        parsed_dir = os.path.dirname(file_path) + "_parsed"
        if os.path.exists(parsed_dir):
            parsed_files = glob.glob(
                os.path.join(parsed_dir, filename.replace(".pdf", "_p*.pkl"))
            )
            if parsed_files:
                return self._load_parsed_files(parsed_files)
        return self.parser.parse_one_pdf(file_path)

    def _parse_directory(self, directory) -> List[VSFile]:
        """
        Parse all PDF files in a directory.
        """
        VSFiles = []
        for filename in os.listdir(directory):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(directory, filename)
                VSFile = self._load_or_parse_pdf(file_path)
                VSFiles.append(VSFile)
        return VSFiles

    def extract_content(self, source) -> List[VSFile]:
        """
        Wrapper method to extract content from PDF files.
        """
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source not found: {source}")

        if os.path.isdir(source):
            VSFiles = self._parse_directory(source)
        elif os.path.isfile(source) and source.lower().endswith(".pdf"):
            VSFile = self._load_or_parse_pdf(source)
            VSFiles = [VSFile]
        else:
            raise ValueError(
                "Source must be a PDF file or a directory containing PDF files."
            )
        return VSFiles

### --- URLs --- ###
class URLContentExtractor(ContentExtractor):
    """
    Extracts content from URLs. Returns a VSFile with the extracted content.

    Attributes:
        output_format (str): The desired output format of the extracted content
        scraper (str): The name of the scraper to use (default: "BeautifulSoup")

    Methods:
        extract_content: Fetch and parse content from the URL
        format_content: Format the parsed content before returning it as a VSFile
    """

    def __init__(self, scraper: str = "BeautifulSoup", **kwargs):
        super().__init__(**kwargs)
        self.scraper = scraper
        self.soup = None
        self.unique_lines = set()

    def extract_content(self, source):
        if self.scraper == "BeautifulSoup":
            response = requests.get(source)
            response.raise_for_status()
            self.soup = BeautifulSoup(response.content, "html.parser")
        elif self.scraper == "Playwright":
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(source)
                html = page.content()
                browser.close()
            self.soup = BeautifulSoup(html, "html.parser")
        else:
            raise NotImplementedError(
                f"Unsupported scraper: {self.scraper}. Use 'BeautifulSoup' or 'Playwright'."
            )

        return self.format_content(self.soup, source)

    def format_content(self, content, source):
        if self.output_format == "markdown":
            md_content = self.convert_to_markdown(content)
            # Create VSFile
            return utils.parsed_url_to_VSFile(url=source, extracted_content=md_content)
        elif self.output_format == "text":
            # Create VSFile
            return utils.parsed_url_to_VSFile(url=source, extracted_content=content)

    def convert_to_markdown(self):
        markdown_content = ""
        for element in self.soup.find_all(True):  # All tags
            markdown_content += self._extract_text(element)
        return markdown_content

    def _extract_text(self, element):
        # Markdown conversion logic
        # Headings
        if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            prefix = "#" * int(element.name[1])
            line = f"\n{prefix} {element.get_text().strip()}\n\n"
            if line not in self.unique_lines:
                self.unique_lines.add(line)
                return line
            return ""

        # Paragraphs
        elif element.name == "p":
            line = f"{element.get_text().strip()}\n\n"
            if line not in self.unique_lines:
                self.unique_lines.add(line)
                return line
            return ""

        # Lists
        elif element.name == "li":
            line = f"- {element.get_text().strip()}\n"
            if line not in self.unique_lines:
                self.unique_lines.add(line)
                return line
            return ""
        elif element.name == "ul":
            return "".join(
                [self.extract_text(child) for child in element.find_all("li")]
            )
        elif element.name == "ol":
            return "".join(
                [
                    f"{idx+1}. {child.get_text().strip()}\n"
                    for idx, child in enumerate(element.find_all("li"))
                ]
            )

        # Images
        elif element.name == "img":
            line = f"![{element.get('alt', '')}]({element.get('src', '')})\n\n"
            if line not in self.unique_lines:
                self.unique_lines.add(line)
                return line
            return ""

        # Links
        elif element.name == "a":
            line = f"[{element.get_text().strip()}]({element.get('href', '')})"
            if line not in self.unique_lines:
                self.unique_lines.add(line)
                return line
            return ""

        # Tables
        elif element.name == "table":
            rows = element.find_all("tr")
            headers = rows[0].find_all("th")
            header_row = (
                "| "
                + " | ".join([header.get_text().strip() for header in headers])
                + " |\n"
            )
            separator_row = "| " + " | ".join(["---" for _ in headers]) + " |\n"
            body_rows = [
                "| "
                + " | ".join([td.get_text().strip() for td in row.find_all("td")])
                + " |\n"
                for row in rows[1:]
            ]
            table_text = header_row + separator_row + "".join(body_rows) + "\n\n"
            if table_text not in self.unique_lines:
                self.unique_lines.add(table_text)
                return table_text
            return ""

        # Text formatting
        elif element.name in ["strong", "b"]:
            line = f"**{element.get_text().strip()}**"
            if line not in self.unique_lines:
                self.unique_lines.add(line)
                return line
            return ""
        elif element.name in ["em", "i"]:
            line = f"*{element.get_text().strip()}*"
            if line not in self.unique_lines:
                self.unique_lines.add(line)
                return line
            return ""

        else:
            # If the element has child elements, recursively extract their text
            if element.find_all(recursive=False):
                child_texts = "".join(
                    [
                        self.extract_text(child)
                        for child in element.find_all(recursive=False)
                    ]
                )
                return child_texts
            else:
                # If the element has no child elements, return its stripped text
                line = element.get_text().strip()
                if line and line not in self.unique_lines:
                    self.unique_lines.add(line)
                    return line
                return ""

### --- JSONs --- ###
class JSONContentExtractor(ContentExtractor):
    """
    Extracts content from JSON files.

    Methods:
        extract_content: Load and parse content from the JSON file(s)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_one_json(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            content = data.get("content", "")
            if not content:
                raise ValueError("No content found in the JSON file.")

            # all other keys become custom metadata
            other_info = {key: value for key, value in data.items() if key != "content"}

            vs_file = VSFile(file_path=file_path)
            node_metadata = NodeMetadata(
                source_file_uuid=vs_file.uuid, position=0, custom=other_info
            )
            node = Node(content, node_metadata)
            vs_file.nodes = [node]
            return vs_file

    def extract_content(self, source):
        if os.path.isfile(source) and source.lower().endswith(".json"):
            vs_file = self._load_one_json(source)
            return [vs_file]
        elif os.path.isdir(source):
            vs_files = []
            for filename in os.listdir(source):
                if filename.lower().endswith(".json"):
                    file_path = os.path.join(source, filename)
                    vs_file = self._load_one_json(file_path)
                    vs_files.append(vs_file)
            return vs_files
        else:
            raise ValueError(
                "Source must be a JSON file or a directory containing JSON files."
            )


if __name__ == "__main__":  # example usage
    # one file, vs_files will be a list with one element
    pdf_extractor = PDFContentExtractor(
        parser_name="SimplePDFParser", output_format="markdown"
    )
    vs_files = pdf_extractor.extract_content(source="example.pdf")

    # directory
    pdf_extractor = PDFContentExtractor(parser_name="SimplePDFParser")
    vs_files = pdf_extractor.extract_content(source="path/to/pdf_directory")

    url_extractor = URLContentExtractor(
        scraper="BeautifulSoup", output_format="markdown"
    )
    vs_file = url_extractor.extract_content(source="https://www.example.com")

    json_extractor = JSONContentExtractor()
    vs_files = json_extractor.extract_content(source="example.json")
