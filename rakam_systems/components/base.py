import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import dotenv
from openai import OpenAI
from mistralai import Mistral

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()

class LLM:
    def __init__(self, model: str = "gpt-4o", api_key=None, api_source: str = "OpenAI") -> None:
        self.model = model
        self.api_source = api_source
        if self.api_source == "OpenAI":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") if not api_key else api_key)
        elif self.api_source == "Mistral":
            self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY") if not api_key else api_key)
        else:
            raise ValueError("Invalid API source, must be either OpenAI or Mistral")

    def call_llm(self, sys_prompt: str, prompt: str, temperature: float = 0 ) -> str:
        if self.api_source == "OpenAI":
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return completion.choices[0].message.content
        elif self.api_source == "Mistral":
            completion = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return completion.choices[0].message.content
        else:
            raise ValueError("Invalid API source, must be either OpenAI or Mistral")

    def call_llm_stream(self, sys_prompt: str, prompt: str, temperature: float = 0, seed: int = 0):
        if self.api_source == "OpenAI":
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
        elif self.api_source == "Mistral":
            stream_response = self.client.chat.stream(
                model = self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            for chunk in stream_response:
                yield chunk.data.choices[0].delta.content

    def call_llm_output_json(self, sys_prompt: str, prompt: str, temperature: float = 0, seed: int = 0):
        if self.api_source == "OpenAI":
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
        elif self.api_source == "Mistral":
            raise ValueError("Mistral does not support JSON output format")