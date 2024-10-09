import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

import dotenv
import pandas as pd

from rakam_systems.core import Node
from rakam_systems.core import NodeMetadata
from rakam_systems.custom_loggers import prompt_logger
from rakam_systems.components.base import LLM
from rakam_systems.components.vector_search import VectorStores

dotenv.load_dotenv()

# Abstract Action class
class Action(ABC):
    @abstractmethod
    def __init__(self, agent, **kwargs):
        pass

    @abstractmethod
    def execute(self, **kwargs):
        pass

class TextSearchMetadata(Action):
    def __init__(
        self,
        agent,
        text_items: pd.Series,
        metadatas: pd.Series,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.agent = agent
        self.text_items = text_items
        self.metadatas = metadatas

        self.embedding_model = embedding_model
        self.vector_store = self.build_vector_store(text_items, metadatas)

    def build_vector_store(
        self, text_items: pd.Series, metadatas: pd.Series
    ) -> VectorStores:
        """
        Builds a VectorStores object from the trigger queries and class names.
        """
        # Create nodes from trigger queries and class names
        nodes = []
        for query, metadata in zip(text_items, metadatas):
            metadata = NodeMetadata(
                source_file_uuid=query, position=None, custom=metadata
            )
            node = Node(content=query, metadata=metadata)
            nodes.append(node)

        # Initialize VectorStores
        vector_store = VectorStores(
            base_index_path="temp_path", embedding_model=self.embedding_model
        )

        # Use the create_from_nodes method to build the index
        vector_store.create_from_nodes(store_name="query_classification", nodes=nodes)

        return vector_store

    def execute(self, query: str) -> list:
        """
        Classifies the query by finding the closest match in the FAISS index.
        """
        # Perform the search using the vector store
        valid_suggestions, _ = self.vector_store.search(
            store_name="query_classification", query=query, number=2
        )

        # print("Successfully classified query:", query, "\nFound : ", valid_suggestions)
        return valid_suggestions

class ClassifyQuery(Action):
    def __init__(
        self,
        agent,
        trigger_queries: pd.Series,
        class_names: pd.Series,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.agent = agent
        self.trigger_queries = trigger_queries
        self.class_names = class_names
        self.embedding_model = embedding_model
        self.vector_store = self.build_vector_store(trigger_queries, class_names)

    def build_vector_store(
        self, trigger_queries: pd.Series, class_names: pd.Series
    ) -> VectorStores:
        """
        Builds a VectorStores object from the trigger queries and class names.
        """
        # Create nodes from trigger queries and class names
        nodes = []
        for query, class_name in zip(trigger_queries, class_names):
            metadata = NodeMetadata(
                source_file_uuid=query, position=None, custom={"class_name": class_name}
            )
            node = Node(content=query, metadata=metadata)
            nodes.append(node)

        # Initialize VectorStores
        vector_store = VectorStores(
            base_index_path="temp_path", embedding_model=self.embedding_model
        )

        # Use the create_from_nodes method to build the index
        vector_store.create_from_nodes(store_name="query_classification", nodes=nodes)

        return vector_store

    def execute(self, query: str):
        """
        Classifies the query by finding the closest match in the FAISS index.
        """
        # Perform the search using the vector store
        _, node_search_results = self.vector_store.search(
            store_name="query_classification", query=query, number=2
        )

        # Extract the matched trigger query and class name
        matched_node = node_search_results[0]
        matched_trigger_query = matched_node.content
        matched_class_name = matched_node.metadata.custom["class_name"]

        print("Successfully classified query:", query)
        return matched_class_name, matched_trigger_query

class RAGGeneration(Action):
    def __init__(
        self,
        agent,
        sys_prompt: str,
        prompt: str,
        vector_stores: VectorStores,
        vs_descriptions: dict = None,
    ):
        self.agent = agent
        self.sys_prompt = sys_prompt
        self.prompt = prompt
        self.vector_stores = vector_stores
        self.vs_descriptions = vs_descriptions

        self.store_separator = "\n====\n"
        self.result_separator = "\n----\n"

    def execute(self, query, prompt_kwargs: dict = {}, stream: bool = False):
        ### --- Vector Store Search --- ###
        formatted_search_results = self._perform_vector_store_search(query)

        ### --- Format Prompt --- ###
        formatted_prompt = self.prompt.format(
            query=query,
            search_results=formatted_search_results,
            **prompt_kwargs,
        )
        prompt_logger.info(f"\nSYSPROMPT:\n---\n{self.sys_prompt}\n---\n")
        prompt_logger.info(f"\nPROMPT:\n---\n{self.prompt}\n---\n")
        prompt_logger.info(
            f"\nFORMATTED PROMPT (RAGGeneration):\n---\n{formatted_prompt}\n---\n"
        )

        ### --- LLM Generation --- ###
        if stream:
            return self._generate_stream(formatted_prompt)
        else:
            return self._generate_non_stream(formatted_prompt)

    # STREAMING (returns a generator)
    def _generate_stream(self, formatted_prompt):
        # Call the LLM to generate the final answer in streaming mode
        response_generator = self.agent.llm.call_llm_stream(
            self.sys_prompt, formatted_prompt
        )
        for chunk in response_generator:
            yield chunk

    # NON-STREAMING (returns a string)
    def _generate_non_stream(self, formatted_prompt) -> str:
        # Call the LLM to generate the final answer
        answer = self.agent.llm.call_llm(self.sys_prompt, formatted_prompt)
        return answer

    def _perform_vector_store_search(self, query: str) -> str:
        """
        Perform a search across all vector stores and format the results.
        """
        search_results = {}
        for store_name, _ in self.vector_stores.stores.items():
            _, node_search_results = self.vector_stores.search(
                store_name=store_name, query=query
            )
            if node_search_results:
                # Format the search results for one vector store
                search_results[store_name] = self._format_search_results(
                    node_search_results
                )

        # Format the search results for all vector stores
        formatted_search_results = []
        for store_name, results in search_results.items():
            if store_name in self.vs_descriptions:
                description = self.vs_descriptions[store_name]
            else:
                description = store_name
            formatted_search_results.append(f"\n**Source:** {description}\n\n")
            formatted_search_results.append(results)
            formatted_search_results.append(self.store_separator)

        return "".join(formatted_search_results).rstrip(
            self.store_separator + self.result_separator
        )

    def _format_search_results(self, search_results: list) -> str:
        """
        Formats the search results from one vector store into a string.
        """
        if not search_results:
            return f"No results found. {self.result_separator}"

        formatted_results = [
            f"{node.content}{self.result_separator}" for node in search_results
        ]

        return "".join(formatted_results)

class GenericLLMResponse(Action):
    def __init__(self, agent, sys_prompt: str, prompt: str):
        self.agent = agent
        self.sys_prompt = sys_prompt
        self.prompt = prompt

    def execute(self, query, prompt_kwargs: dict = {}, stream: bool = False):
        ### --- Format Prompt --- ###
        formatted_prompt = self.prompt.format(query=query, **prompt_kwargs)
        prompt_logger.info(
            f"\nFORMATTED PROMPT (GenericLLMResponse):\n---\n{formatted_prompt}\n---\n"
        )

        ### --- LLM Generation --- ###
        if stream:
            return self._generate_stream(formatted_prompt)
        else:
            return self._generate_non_stream(formatted_prompt)

    # STREAMING (returns a generator)
    def _generate_stream(self, formatted_prompt):
        # Call the LLM to generate the final answer in streaming mode
        response_generator = self.agent.llm.call_llm_stream(
            self.sys_prompt, formatted_prompt
        )
        for chunk in response_generator:
            yield chunk

    # NON-STREAMING (returns a string)
    def _generate_non_stream(self, formatted_prompt) -> str:
        # Call the LLM to generate the final answer
        answer = self.agent.llm.call_llm(self.sys_prompt, formatted_prompt)
        return answer
