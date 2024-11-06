import os

from openai import OpenAI
from mistralai import Mistral

from rakam_systems.system_manager import SystemManager
from rakam_systems.components.component import Component

class LLMManager(Component):
    def __init__(self, system_manager: SystemManager, model: str = "gpt-4o", api_key=None) -> None:
        self.model = model
        if model in ["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"]:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") if not api_key else api_key)
        else:
            self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY") if not api_key else api_key)
        self.system_manager = system_manager

    def call_llm(self, sys_prompt: str, prompt: str, temperature: float = 0 ) -> str:
        if self.model in ["gpt-4o", "gpt-4o-mini"]:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return completion.choices[0].message.content
        else:
            completion = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return completion.choices[0].message.content

    def call_llm_stream(self, sys_prompt: str, prompt: str, temperature: float = 0, seed: int = 0):
        if self.model in ["gpt-4o", "gpt-4o-mini"]:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                temperature=temperature
            )
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:  # Only yield if content is not None
                    yield content
        else:
            stream_response = self.client.chat.stream(
                model = self.model,
                messages = [
                        {
                            "role": "user",
                            "content": "What is the best French cheese?",
                        },
                    ]
                )

            for chunk in stream_response:
                yield chunk
    
    def call_llm_output_json(self, sys_prompt: str, prompt: str, temperature: float = 0, seed: int = 0):
        if self.model in ["gpt-4o", "gpt-4o-mini"]:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temperature
            )
            return completion.choices[0].message.content
        else:
            raise NotImplementedError("Output JSON is not supported for Mistral models.")
    
    def call_main(self, **kwargs) -> dict:
        return super().call_main(**kwargs)
    
    def test(self, **kwargs) -> bool:
        return super().test(**kwargs)
