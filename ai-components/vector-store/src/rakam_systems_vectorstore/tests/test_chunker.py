import pytest
from unittest.mock import MagicMock, patch

from rakam_systems_vectorstore.components.chunker.text_chunker import TextChunker


class FakeChunk:
    def __init__(self, text, token_count, start, end):
        self.text = text
        self.token_count = token_count
        self.start_index = start
        self.end_index = end


def test_chunk_text_returns_empty_for_blank_input():
    chunker = TextChunker()

    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   ") == []


def test_chunk_text_raises_if_chonkie_not_available():
    chunker = TextChunker()

    with patch(
        "rakam_systems_vectorstore.components.chunker.text_chunker.CHONKIE_AVAILABLE",
        False
    ):
        with pytest.raises(ImportError):
            chunker.chunk_text("Some text")


def test_chunk_text_returns_formatted_chunks():
    fake_chunks = [
        FakeChunk("chunk1", 10, 0, 20),
        FakeChunk("chunk2", 8, 21, 40),
    ]

    mock_sentence_chunker = MagicMock()
    mock_instance = mock_sentence_chunker.return_value
    mock_instance.return_value = fake_chunks

    with patch(
        "rakam_systems_vectorstore.components.chunker.text_chunker.CHONKIE_AVAILABLE",
        True
    ), patch(
        "rakam_systems_vectorstore.components.chunker.text_chunker.SentenceChunker",
        mock_sentence_chunker
    ):
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)

        result = chunker.chunk_text("Some text", context="doc_0")

        # Verify SentenceChunker constructed correctly
        mock_sentence_chunker.assert_called_once_with(
            tokenizer="character",
            chunk_size=100,
            chunk_overlap=10,
            min_sentences_per_chunk=1,
        )

        assert result == [
            {
                "text": "chunk1",
                "token_count": 10,
                "start_index": 0,
                "end_index": 20,
            },
            {
                "text": "chunk2",
                "token_count": 8,
                "start_index": 21,
                "end_index": 40,
            },
        ]


def test_run_aggregates_multiple_documents():
    fake_chunks = [
        FakeChunk("chunkA", 5, 0, 10),
        FakeChunk("chunkB", 6, 11, 20),
    ]

    mock_sentence_chunker = MagicMock()
    mock_instance = mock_sentence_chunker.return_value
    mock_instance.return_value = fake_chunks

    with patch(
        "rakam_systems_vectorstore.components.chunker.text_chunker.CHONKIE_AVAILABLE",
        True
    ), patch(
        "rakam_systems_vectorstore.components.chunker.text_chunker.SentenceChunker",
        mock_sentence_chunker
    ):
        chunker = TextChunker()

        result = chunker.run(["Doc 1 text", "Doc 2 text"])

        # Should flatten chunks from both documents
        assert result == [
            "chunkA",
            "chunkB",
            "chunkA",
            "chunkB",
        ]
