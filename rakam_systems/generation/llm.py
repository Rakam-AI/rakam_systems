import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO)


class LLM:
    def __init__(self, model: str, api_key: str = None) -> None:
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY") if not api_key else api_key
        )

    def call_llm(self, sys_prompt: str, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content

    def call_llm_stream(self, sys_prompt: str, prompt: str):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        for chunk in completion:
            yield chunk.choices[0].delta.content
