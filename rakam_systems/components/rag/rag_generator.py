import logging
from typing import List
from textwrap import dedent

import dotenv

from rakam_systems.system_manager import SystemManager
from rakam_systems.components.component import Component

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)

class RAGGenerator(Component):
    def __init__(self, system_manager: SystemManager, temperature: float = 0.7):
        """Initialize the Generator with the specified LLM model and temperature.

        Args:
            temperature (float): A parameter to control randomness in generation (default is 0.7).
        """
        self.temperature = temperature
        self.system_manager = system_manager
        self.sys_prompt = "You are a helpful assistant."

    def _retrive(self, query, collection_name = "base", source_file_uuids = []) -> List[str]:
        """Retrieve the documents relevant to the query.

        Args:
            query (str): The query to retrieve documents for.

        Returns:
            List[str]: The retrieved documents.
        """
        nodeID_response = self.system_manager.execute_component_function(
            component_name="VectorStore",
            function_name="get_nodes",
            input_data={"collection_name": collection_name}
        )

        nodeID_filters = []
        for node in nodeID_response["nodes"]:
            if node["metadata"]["source_file_uuid"] in source_file_uuids:
                nodeID_filters.append(node["metadata"]["node_id"])
        logging.info(f"NodeID filters: {nodeID_filters}")

        input_data = {"query": query, "collection_name": collection_name, "nodeID_filters": nodeID_filters}
        response = self.system_manager.execute_component_function(
            component_name="VectorStore",
            function_name="search",
            input_data=input_data
        )
        logging.info(f"Retrieved: {response}")
        response = response["response"]
        documents = [item["text"] for item in response.values() if "text" in item]
        logging.info(f"Retrieved {len(documents)} documents.")
        return documents, nodeID_filters

    def _get_user_prompt(self, documents: List[str], query: str) -> str:
        """Generate a prompt based on the loaded documents and the set query.

        Args:
            documents (List[str]): List of document strings.
            query (str): The query for which the response is generated.
        
        Raises:
            ValueError: If no documents are loaded.
        """
        if not documents:
            raise ValueError("No documents loaded.")

        try:
            # Clean each document to ensure no special characters interfere with the formatting.
            cleaned_documents = [doc.replace("\n", " ").replace("\t", " ") for doc in documents]
            combined_documents = "\n".join(cleaned_documents)
        except Exception as e:
            # Handle any unexpected errors in document processing.
            raise ValueError(f"Error processing documents: {e}")

        return dedent(f"""\
            Given the following documents, generate a response to the query: {query}

            Documents:
            {combined_documents}
        """)

    def _generate_text(self, sys_prompt: str, user_prompt:str,  stream=False) -> str:
        """Generate text using the LLM.

        Returns:
            str: The generated text.

        Raises:
            ValueError: If the prompt is not set.
        """
        if stream:
            self.system_manager.execute_component_function(
                component_name="LLMConnector",
                function_name="call_llm_stream",
                input_data={
                    "sys_prompt": sys_prompt,
                    "prompt": user_prompt,
                    "temperature": self.temperature
                }
            )
        else:
            logging.info(f"******System Prompt Sent: {sys_prompt}\n User Prompt Sent: {user_prompt} \n ******")
            response = self.system_manager.execute_component_function(
                component_name="LLMConnector",
                function_name="call_llm",
                input_data={
                    "sys_prompt": sys_prompt,
                    "prompt": user_prompt,
                    "temperature": self.temperature
                }
            )
            return response['response']
    
    def _split_query(self, query: str) -> List[str]:
        """Split the query into multiple queries.

        Args:
            query (str): The query to split.

        Returns:
            List[str]: The split queries.
        """
        sys_prompt = """
        Given a query, split it into multiple(at most 5) simple sub-queries.
        
        Respond in this format:
        ["Sub-query 1", "Sub-query 2", ...]

        Only respond the list of sub-queries and nothing else.
        """
        user_prompt = f"Split the following query into multiple simple sub-queries: {query}"
        generated_text = self._generate_text(sys_prompt, user_prompt)
        try:
            sub_queries = eval(generated_text)
            # Ensure that the evaluated response is a list
            if isinstance(sub_queries, list):
                return sub_queries
            else:
                # If not a list, log an error and return as a single-item list
                logging.error("Generated text is not in the expected list format.")
                return [generated_text]
        except Exception as e:
            logging.error(f"Error splitting query: {e}")
            return [generated_text]

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

    def call_split_query(self, query: str):
        """
        External function to split a query into multiple sub-queries using the LLM.
        """
        sub_queries = self._split_query(query)
        response = {
            "sub_queries": sub_queries,
            "query": query,
            "status": "success",
        }
        return response

    def call_rag_with_split_query(self, query: str):
        sub_queries = self._split_query(query)
        documents = []
        for sub_query in sub_queries:
            documents.extend(self._retrive(sub_query))
        user_prompt = self._get_user_prompt(documents, query)
        generated_text = self._generate_text(self.sys_prompt, user_prompt, stream=False)
        response = {
            "generated_text": generated_text,
            "query": query,
            "sub_queries": sub_queries,
            "retreived_documents": documents,
            "status": "success",
        }
        return response

    def call_main(self, query, collection_name="base", source_file_uuids=[]):
        """
        Simple RAG generation with a single query.
        Vector Search in default 'base' collection.
        """
        documents,nodeID_filters = self._retrive(query, collection_name=collection_name, source_file_uuids=source_file_uuids)
        user_prompt = self._get_user_prompt(documents, query)
        # user_prompt = user_prompt[:200]
        generated_text = self._generate_text(self.sys_prompt, user_prompt, stream=False)
        response = {
            "generated_text": generated_text,
            "retrieval nodeID filters": nodeID_filters,
            "query": query,
            "sys_prompt": self.sys_prompt,
            "user_prompt": user_prompt,
            "retreived_documents": documents,
            "status": "success",
        }
        return response

    def test(self) -> bool:
        
        pass