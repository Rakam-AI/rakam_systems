import logging
from typing import List, Optional
import dotenv

from rakam_systems.components.base import LLM
from rakam_systems.components.component import Component

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)

class Generator(Component):
    def __init__(self, model: str, temperature: float = 0.7):
        """Initialize the Generator with the specified LLM model and temperature.

        Args:
            model (str): The LLM model to use.
            temperature (float): A parameter to control randomness in generation (default is 0.7).
        """
        self.llm = LLM(model=model)  # Initialize the LLM
        self.temperature: float = temperature
        self.query: Optional[str] = None
        self.documents: Optional[List[str]] = None
        self.sys_prompt: str = "You are a helpful assistant."  # Default system prompt
        self.user_prompt: Optional[str] = None
        logging.info(f"Generator initialized with model: {model}")

    def _setup_generation(self, query: str, documents: List[str], sys_prompt: Optional[str] = None, temperature: Optional[float] = None):
        """Set the query and documents for generation.

        Args:
            query (str): The query to respond to.
            documents (List[str]): The documents to consider for generation.
            sys_prompt (Optional[str]): The system prompt for the LLM.
        """
        self.documents = documents
        self.query = query
        if sys_prompt is not None:
            self.sys_prompt = sys_prompt
        if temperature is not None:
            self.set_temperature(temperature)
        
        self._setup_user_prompt(mode="RAG")

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

    def _setup_user_prompt(self, mode: str):
        """Generate a prompt based on the loaded documents and the set query.

        Args:
            mode (str): The mode of prompt generation (e.g., "RAG").
        
        Raises:
            ValueError: If no documents are loaded.
        """
        if self.documents is None:
            raise ValueError("No documents loaded. Please load documents before setting up a prompt.")
        
        if mode == "RAG":
            combined_documents = "\n".join(self.documents)
            self.user_prompt = (
                f"Based on the following documents:\n{combined_documents}\n"
                f"Please generate a response for this query: {self.query}"
            )

    def _generate_text(self, stream=False) -> str:
        """Generate text using the LLM.

        Returns:
            str: The generated text.

        Raises:
            ValueError: If the prompt is not set.
        """
        if self.user_prompt is None:
            raise ValueError("Prompt not set. Please set a prompt before generating text.")

        if stream:
            self.llm.call_llm_stream(
                sys_prompt=self.sys_prompt,
                prompt=self.user_prompt,
                temperature=self.temperature
            )
        else:
            logging.info(f"******System Prompt Sent: {self.sys_prompt}\n User Prompt Sent: {self.user_prompt} \n ******")
            return self.llm.call_llm(
                sys_prompt=self.sys_prompt,
                prompt=self.user_prompt,
                temperature=self.temperature
            )

    def call_direct_generation(self, query:str) -> str:
        """Generate text directly using the LLM.

        Args:
            query (str): The query to respond to.

        Returns:
            str: The generated text.
        """
        self.query = query
        return self.llm.call_llm(
            sys_prompt=self.sys_prompt,
            prompt=query,
            temperature=self.temperature
        )

    def call_main(self, query, documents) -> str:
        """Main method to generate text.

        Returns:
            str: The generated text.
        """
        self._setup_generation(query=query, documents=documents)
        return self._generate_text()

    def test(self) -> bool:
        """Method for testing the VectorStore.

        Returns:
            bool: The result of the test (currently a placeholder).
        """
        logging.info("Running test for VectorStore.")
        return self.call_main()

