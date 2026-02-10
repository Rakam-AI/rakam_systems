from __future__ import annotations
from typing import List
from rakam_systems_core.ai_core.interfaces.embedding_model import EmbeddingModel
from openai import OpenAI
from rakam_systems_core.ai_utils import logging

logger = logging.getLogger(__name__)


class OpenAIEmbeddings(EmbeddingModel):
    """OpenAI embeddings implementation using the OpenAI API.

    This module is integrated into ConfigurableEmbeddings and can be used via
    the "openai" provider configuration.

    Args:
        model: The OpenAI embedding model to use (default: "text-embedding-3-small")
        api_key: Optional API key. If not provided, will use OPENAI_API_KEY environment variable
        max_tokens: Maximum tokens allowed per text (default: 8191 for text-embedding-3-small)
        batch_size: Batch size for API calls (default: 100, recommended by OpenAI)
    """

    # Model-specific token limits (leaving some margin for safety)
    MODEL_TOKEN_LIMITS = {
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
        "text-embedding-ada-002": 8191,
    }

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None, max_tokens: int = None, batch_size: int = 100):
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.max_tokens = max_tokens or self.MODEL_TOKEN_LIMITS.get(
            model, 8191)
        self.batch_size = batch_size

        # Initialize tiktoken for token counting
        try:
            import tiktoken
            self.encoding = tiktoken.encoding_for_model(model)
        except Exception as e:
            logger.warning(
                f"Failed to initialize tiktoken for model {model}: {e}. Using character-based estimation.")
            self.encoding = None

    def _truncate_batch_with_encode_batch(self, texts: List[str]) -> List[str]:
        """Truncate texts using encode_batch to find maximum embeddable length.

        Uses encode_batch to determine the actual token count for each text,
        then truncates only those that exceed the limit.
        """
        if not self.encoding:
            # Fallback: character-based truncation
            max_chars = self.max_tokens * 4
            return [text[:max_chars] if len(text) > max_chars else text for text in texts]

        # Clean texts
        cleaned_texts = [text.replace("\n", " ") for text in texts]

        # Use encode_batch to get token counts for all texts at once
        try:
            encoded_batch = self.encoding.encode_batch(cleaned_texts)
        except Exception as e:
            logger.warning(
                f"encode_batch failed: {e}, falling back to individual encoding")
            encoded_batch = [self.encoding.encode(
                text) for text in cleaned_texts]

        # Process each text based on its actual token count
        processed_texts = []
        for i, (text, tokens) in enumerate(zip(cleaned_texts, encoded_batch)):
            if len(tokens) <= self.max_tokens:
                # Text is within limit, use as-is
                processed_texts.append(text)
            else:
                # Text exceeds limit, truncate to max_tokens
                truncated_tokens = tokens[:self.max_tokens]
                truncated_text = self.encoding.decode(truncated_tokens)
                logger.warning(
                    f"Text {i} truncated from {len(tokens)} to {self.max_tokens} tokens")
                processed_texts.append(truncated_text)

        return processed_texts

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        # Use batch truncation for consistency
        processed_texts = self._truncate_batch_with_encode_batch([text])
        return self.client.embeddings.create(input=processed_texts, model=self.model).data[0].embedding

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Get embeddings for multiple texts using batch processing.

        Uses encode_batch to determine the maximum embeddable length for each text,
        then truncates if necessary before sending to the API.
        Also respects OpenAI's 300K tokens per request limit.

        Args:
            texts: List of texts to embed
            batch_size: Maximum number of texts to send in a single API call (default: 100)
                       OpenAI recommends batches of 100 or fewer for optimal performance

        Returns:
            List of embedding vectors, one for each input text
        """
        if not texts:
            return []

        import time

        # OpenAI API limits
        MAX_TOKENS_PER_REQUEST = 300000  # Total tokens per request limit
        MAX_TOKENS_PER_TEXT = self.max_tokens  # Individual text limit (8191)

        all_embeddings = []
        total_texts = len(texts)

        # Log initial info
        logger.info(
            f"Starting OpenAI embedding generation for {total_texts} texts")
        logger.info(f"Initial batch size: {batch_size}")
        logger.info(f"Model: {self.model}")

        start_time = time.time()

        # Process texts with dynamic batching based on token count
        i = 0
        batch_num = 0
        while i < total_texts:
            batch_num += 1
            batch_start_time = time.time()

            # Collect texts for this batch, respecting token limits
            batch = []
            batch_indices = []
            current_batch_tokens = 0

            # Try to fill batch up to batch_size or token limit
            while i < total_texts and len(batch) < batch_size:
                # Peek at next text and process it
                next_batch = [texts[i]]
                processed_next = self._truncate_batch_with_encode_batch(
                    next_batch)

                # Count tokens in processed text
                if self.encoding:
                    try:
                        text_tokens = len(
                            self.encoding.encode(processed_next[0]))
                    except:
                        # Fallback estimation
                        text_tokens = len(processed_next[0]) // 4
                else:
                    text_tokens = len(processed_next[0]) // 4

                # Check if adding this text would exceed the request limit
                if batch and (current_batch_tokens + text_tokens > MAX_TOKENS_PER_REQUEST):
                    # Batch is full, stop here
                    logger.info(
                        f"[OpenAI Batch {batch_num}] Batch token limit reached: {current_batch_tokens} tokens, stopping before adding text with {text_tokens} tokens")
                    break

                # Add text to batch
                batch.append(texts[i])
                batch_indices.append(i)
                current_batch_tokens += text_tokens
                i += 1

            if not batch:
                # Edge case: single text exceeds request limit (shouldn't happen with 8191 limit)
                logger.error(
                    f"Single text at index {i} exceeds request token limit, skipping")
                all_embeddings.append([0.0] * 1536)
                i += 1
                continue

            # Log batch start
            progress_pct = i / total_texts * 100
            logger.info(f"[OpenAI Batch {batch_num}] Processing texts {batch_indices[0]+1}-{batch_indices[-1]+1} "
                        f"({len(batch)} texts, ~{current_batch_tokens} tokens, {progress_pct:.1f}% complete)")

            # Process the batch
            processed_batch = self._truncate_batch_with_encode_batch(batch)

            # Send batch to OpenAI API
            try:
                response = self.client.embeddings.create(
                    input=processed_batch,
                    model=self.model
                )

                # Extract embeddings in order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                batch_elapsed = time.time() - batch_start_time

                # Calculate statistics
                texts_processed = len(all_embeddings)
                overall_elapsed = time.time() - start_time
                overall_rate = texts_processed / overall_elapsed if overall_elapsed > 0 else 0
                eta_seconds = (total_texts - texts_processed) / \
                    overall_rate if overall_rate > 0 else 0

                # Log batch completion with detailed stats
                logger.info(f"[OpenAI Batch {batch_num}] ✓ Completed in {batch_elapsed:.2f}s | "
                            f"Progress: {texts_processed}/{total_texts} ({texts_processed/total_texts*100:.1f}%) | "
                            f"Rate: {overall_rate:.1f} texts/s | ETA: {eta_seconds:.0f}s")

            except Exception as e:
                logger.error(
                    f"[OpenAI Batch {batch_num}] Batch processing failed: {e}")
                logger.info(
                    f"[OpenAI Batch {batch_num}] Falling back to individual processing...")

                # Fallback: process texts individually
                for idx, text in enumerate(batch):
                    try:
                        processed_text = self._truncate_batch_with_encode_batch([text])[
                            0]
                        embedding = self.client.embeddings.create(
                            input=[processed_text],
                            model=self.model
                        ).data[0].embedding
                        all_embeddings.append(embedding)
                    except Exception as inner_e:
                        logger.error(
                            f"[OpenAI Batch {batch_num}] Error processing individual text {batch_indices[idx] + 1}: {inner_e}")
                        # Return zero vector as fallback
                        # Default dimension for text-embedding-3-small
                        all_embeddings.append([0.0] * 1536)

        # Log final summary
        total_elapsed = time.time() - start_time
        overall_rate = total_texts / total_elapsed if total_elapsed > 0 else 0
        logger.info(f"✓ OpenAI embedding generation completed!")
        logger.info(f"  Total texts: {total_texts}")
        logger.info(f"  Total time: {total_elapsed:.2f}s")
        logger.info(f"  Average rate: {overall_rate:.1f} texts/s")
        logger.info(f"  Batches processed: {batch_num}")

        return all_embeddings

    def run(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.

        Uses batch processing for efficiency when multiple texts are provided.
        """
        if not texts:
            return []

        # Use batch processing for multiple texts
        if len(texts) > 1:
            return self.get_embeddings_batch(texts, batch_size=self.batch_size)
        else:
            # Single text - use direct method
            return [self.get_embedding(texts[0])]
