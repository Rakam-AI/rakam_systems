import logging
from typing import List
import dotenv

from rakam_systems.system_manager import SystemManager
from rakam_systems.components.base import LLM
from rakam_systems.components.component import Component

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)

class RAGGenerator(Component):
    def __init__(self, model: str, system_manager: SystemManager, temperature: float = 0.7):
        """Initialize the Generator with the specified LLM model and temperature.

        Args:
            model (str): The LLM model to use.
            temperature (float): A parameter to control randomness in generation (default is 0.7).
        """

        self.llm = LLM(model=model)
        self.temperature = temperature
        self.system_manager = system_manager
        self.sys_prompt = "You are a helpful assistant."

    def _retrive(self, query, collection_name = "base"):
        """Retrieve the documents relevant to the query.

        Args:
            query (str): The query to retrieve documents for.

        Returns:
            List[str]: The retrieved documents.
        """
        input_data = {"query": query, "collection_name": collection_name}
        response = self.system_manager.execute_component_function(
            component_name="VectorStore",
            function_name="search",
            input_data=input_data
        )

        # Extract the texts from the response
        documents = [item["text"] for item in response.values() if "text" in item]
        return documents

    def _get_user_prompt(self, documents: List[str], query: str):
        """Generate a prompt based on the loaded documents and the set query.

        Args:
            mode (str): The mode of prompt generation (e.g., "RAG").
        
        Raises:
            ValueError: If no documents are loaded.
        """
        combined_documents = "---\n".join(documents)
        
        return (
            f"Based on the following documents:\n{combined_documents}\n"
            f"Please generate a response for this query: {query}"
        )

    def _generate_text(self, sys_prompt: str, user_prompt:str,  stream=False) -> str:
        """Generate text using the LLM.

        Returns:
            str: The generated text.

        Raises:
            ValueError: If the prompt is not set.
        """
        if stream:
            self.llm.call_llm_stream(
                sys_prompt=sys_prompt,
                prompt=user_prompt,
                temperature=self.temperature
            )
        else:
            logging.info(f"******System Prompt Sent: {sys_prompt}\n User Prompt Sent: {user_prompt} \n ******")
            return self.llm.call_llm(
                sys_prompt=sys_prompt,
                prompt=user_prompt,
                temperature=self.temperature
            )

    def set_sys_prompt(self, sys_prompt: str):
        """Set the system prompt for text generation.

        Args:
            sys_prompt (str): The system prompt.
        """
        self.sys_prompt = sys_prompt

    def set_temperature(self, temperature: float):
        """Set the temperature for text generation.

        Args:
            temperature (float): A value between 0 and 1 to control randomness.

        Raises:
            ValueError: If the temperature is not between 0 and 1.
        """
        if not (0 <= temperature <= 1):
            raise ValueError("Temperature must be between 0 and 1.")
        self.temperature = temperature

    def call_main(self, query):
        documents = self._retrive(query)
        user_prompt = self._get_user_prompt(documents, query)
        generated_text = self._generate_text(self.sys_prompt, user_prompt, stream=False)
        response = {
            "generated_text": generated_text,
            "query": query,
            "retreived_documents": documents,
            "status": "success",
        }
        return response

    def test(self) -> bool:
        
        pass