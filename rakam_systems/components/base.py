import os
import logging
from typing import List

import dotenv
import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.DEBUG)
dotenv.load_dotenv()

class LLM:
    def __init__(self, model: str = "gpt-4o", api_key=None) -> None:
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") if not api_key else api_key)

    def call_llm(self, sys_prompt: str, prompt: str, temperature: float = 0 ) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return completion.choices[0].message.content

    def call_llm_stream(self, sys_prompt: str, prompt: str, temperature: float = 0, seed: int = 0):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            temperature=temperature,
            seed=seed
        )

        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:  # Only yield if content is not None
                yield content


    def call_llm_output_json(self, sys_prompt: str, prompt: str, temperature: float = 0, seed: int = 0):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            seed=seed
        )
        return completion.choices[0].message.content


class EmbeddingModel:
    _instance = None
    _model = None
    _device = None

    @classmethod
    def get_instance(cls, model_name: str = "all-MiniLM-L6-v2") -> 'EmbeddingModel':
        """Get or create the singleton instance of the embedding model."""
        if cls._instance is None:
            logging.info("Creating new EmbeddingModel instance")
            cls._instance = cls()
            
            # Set device to GPU if available (cuda or mps), otherwise use CPU
            # Note: mps not available when inside Docker container
            if torch.cuda.is_available():
                cls._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                cls._device = "mps"
            else:
                cls._device = "cpu"
            logging.info(f"DEVICE: {cls._device}")

            cls._model = SentenceTransformer(model_name, device=cls._device, trust_remote_code=True)
        return cls._instance

    def get_embeddings(self, sentences: List[str], parallel: bool = True, batch_size: int = 8) -> np.ndarray:
        """
        Generates embeddings for a list of sentences.

        :param sentences: List of sentences to encode.
        :param parallel: Whether to use parallel processing (default is True).
        :param batch_size: Batch size for encoding (default is 8).
        :return: Embedding vectors for the sentences.
        """
        logging.info(f"Generating embeddings for {len(sentences)} sentences.")
        logging.info(f"DEVICE: {self._device}")
        
        if parallel:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            pool = self._model.start_multi_process_pool(target_devices=[self._device] * 5)
            embeddings = self._model.encode_multi_process(
                sentences, 
                pool, 
                batch_size=batch_size
            )
            self._model.stop_multi_process_pool(pool)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            embeddings = self._model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
            )
        
        return embeddings.cpu().detach().numpy() if torch.is_tensor(embeddings) else embeddings

    def encode(self, sentences: List[str]) -> np.ndarray:
        """
        Simple wrapper for direct encoding without parallel processing options.
        
        :param sentences: List of sentences to encode.
        :return: Embedding vectors for the sentences.
        """
        return self._model.encode(sentences)

    def __call__(self, sentences: List[str]) -> np.ndarray:
        """Allow the class instance to be called like a function."""
        return self.encode(sentences)