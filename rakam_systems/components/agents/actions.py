import logging
from abc import ABC
from abc import abstractmethod
from typing import Any, Optional
from textwrap import dedent

import dotenv
import pandas as pd
from typing import List
import os
from rakam_systems.core import Node
from rakam_systems.core import NodeMetadata
from rakam_systems.custom_loggers import prompt_logger
from rakam_systems.components.base import LLM
from rakam_systems.components.base import EmbeddingModel
from rakam_systems.components.vector_search import VectorStore

logger = logging.getLogger(__name__)

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
        collections: dict = {},
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_store_path: str = "temp_path"
    ):
        self.agent = agent
        self.collections = collections
        self.vector_store_path = vector_store_path

        # Initialize the embedding model
        self.embedding_model = EmbeddingModel.get_instance(embedding_model)

        # Initialize the VectorStore
        self.vector_store = self.initialize_vector_store()

    def initialize_vector_store(self) -> VectorStore:
        # VectorStore will contain multiple collections
        
        # If vector_store_path doesn't exist or the dir is empty, create new VectorStore
        if not os.path.exists(self.vector_store_path) or not os.listdir(self.vector_store_path):
            # Check the format of collections
            if self._validate_collections_format(self.collections):
                # Create directory if it doesn't exist
                os.makedirs(self.vector_store_path, exist_ok=True)
                # Build the vector store
                return self._build_vector_store()
            else:
                raise ValueError("Invalid collections format.")
        else:
            # Load the existing vector store
            vector_store = VectorStore(
                base_index_path=self.vector_store_path,
                embedding_model=self.embedding_model,
                initialising=False
            )
            vector_store.load_vector_store()

            # Verify collections if provided
            if self.collections:
                self._verify_and_build_collections(vector_store)

            return vector_store

    def _validate_collections_format(self, collections: dict) -> bool:
        # Validate that collections is a dictionary with the correct structure
        for _, data in collections.items():
            if not isinstance(data, dict) or "triggers" not in data or "metadata" not in data:
                return False
        return True

    def _verify_and_build_collections(self, vector_store: VectorStore):
        # Verify that all collections are present in the vector store
        for collection_name, data in self.collections.items():
            if collection_name not in vector_store.collections:
                # Build missing collections
                self._build_collection(vector_store, collection_name, data)

    def _build_collection(self, vector_store: VectorStore, collection_name: str, data: dict):
        # Create nodes from trigger queries and metadata
        nodes = []
        for query, metadata in zip(data["triggers"], data["metadata"]):
            node_metadata = NodeMetadata(
                source_file_uuid=query, position=None, custom=metadata
            )
            node = Node(content=query, metadata=node_metadata)
            nodes.append(node)

        # Use the create_collection_from_nodes method to build the index
        vector_store.create_collection_from_nodes(collection_name=collection_name, nodes=nodes)

    def _build_vector_store(self) -> VectorStore:
        print("Building vector store:", self.vector_store_path)
        # Initialize VectorStore
        vector_store = VectorStore(
            base_index_path=self.vector_store_path,
            embedding_model=self.embedding_model,
            initialising=True
        )

        # Build all collections
        for collection_name, data in self.collections.items():
            self._build_collection(vector_store, collection_name, data)

        return vector_store

    def execute(self, query: str, collection: str) -> list:
        """
        Classifies the query by finding the closest match in the FAISS index.
        """
        # Perform the search using the vector store
        valid_suggestions, _ = self.vector_store.search(
            collection_name=collection, query=query, number=2
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
    ) -> VectorStore:
        """
        Builds a VectorStore object from the trigger queries and class names.
        """
        # Create nodes from trigger queries and class names
        nodes = []
        for query, class_name in zip(trigger_queries, class_names):
            metadata = NodeMetadata(
                source_file_uuid=query, position=None, custom={"class_name": class_name}
            )
            node = Node(content=query, metadata=metadata)
            nodes.append(node)

        # Initialize VectorStore
        vector_store = VectorStore(
            base_index_path="temp_path", embedding_model=self.embedding_model
        )

        # Use the create_collection_from_nodes method to build the index
        vector_store.create_collection_from_nodes(collection_name="query_classification", nodes=nodes)

        return vector_store

    def execute(self, query: str):
        """
        Classifies the query by finding the closest match in the FAISS index.
        """
        # Perform the search using the vector store
        node_search_results, _ = self.vector_store.search(
            collection_name="query_classification", query=query, number=2
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
        vector_stores: List[VectorStore],
        vs_descriptions: List[str] = None,
    ):
        self.agent = agent
        self.default_sys_prompt = sys_prompt
        self.prompt = prompt
        self.vector_stores = vector_stores
        self.vs_descriptions = vs_descriptions

        self.store_separator = "\n====\n"
        self.result_separator = "\n----\n"

    def execute(self, query, collection_names: List [str], prompt_kwargs: dict = {}, stream: bool = False, sys_prompt: str = None, **kwargs):
        ### --- Vector Store Search --- ###
        formatted_search_results = self._perform_vector_store_search(query, collection_names)

        ### --- Format Prompt --- ###
        formatted_prompt = self.prompt.format(
            query=query,
            search_results=formatted_search_results,
            **prompt_kwargs,
        )

        # Use the provided sys_prompt or fall back to the default
        sys_prompt = sys_prompt or self.default_sys_prompt

        #(f"\nSYSPROMPT:\n---\n{self.default_sys_prompt}\n---\n")
        #prompt_logger.info(f"\nPROMPT:\n---\n{self.prompt}\n---\n")
        #prompt_logger.info(
        #    f"\nFORMATTED PROMPT (RAGGeneration):\n---\n{formatted_prompt}\n---\n"
        #)

        ### --- LLM Generation --- ###
        if stream: return self._generate_stream(
            sys_prompt=sys_prompt,
            formatted_prompt=formatted_prompt
        )
        else: return self._generate_non_stream(
            sys_prompt=sys_prompt,
            formatted_prompt=formatted_prompt
        )

    # STREAMING (returns a generator)
    def _generate_stream(self, sys_prompt, formatted_prompt):
        # Call the LLM to generate the final answer in streaming mode
        response_generator = self.agent.llm.call_llm_stream(
            sys_prompt, formatted_prompt
        )
        for chunk in response_generator:
            yield chunk

    # NON-STREAMING (returns a string)
    def _generate_non_stream(self, sys_prompt, formatted_prompt) -> str:
        # Call the LLM to generate the final answer
        answer = self.agent.llm.call_llm(sys_prompt, formatted_prompt)
        return answer

    def _perform_vector_store_search(self, query: str, collection_names: List[str]) -> str:
        """
        Perform a search across all vector stores and format the results.
        """

        formatted_search_results = []

        # If vector_store_descriptions is None, use collection_names as descriptions
        if self.vs_descriptions is None:
            self.vs_descriptions = collection_names

        print("self.vector_stores, collection_names, self.vs_descriptions : ", self.vector_stores, collection_names, self.vs_descriptions)
        for store, collection_name, collection_description in zip(self.vector_stores, collection_names, self.vs_descriptions) :
            node_search_results, _ = store.search(
                collection_name=collection_name, query=query
            )
            if node_search_results:
                # Format the search results for one vector store
                formatted_search_results.append(f"\n**Source:** {collection_description}\n\n")
                formatted_search_results.append(self._format_search_results(node_search_results))
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

        # TMP:
        search_results = list(search_results.values())
        formatted_results = [
            f"{result[1]}{self.result_separator}" for result in search_results
        ]

        #formatted_results = [
        #    f"{node.content}{self.result_separator}" for node in search_results
        #]

        return "".join(formatted_results)

class GenericLLMResponse(Action):
    def __init__(self, agent, sys_prompt: str, prompt: str):
        self.agent = agent
        self.default_sys_prompt = sys_prompt
        self.prompt = prompt

    def execute(self, query, prompt_kwargs: dict = {}, stream: bool = False, sys_prompt: str = None, **kwargs):
        ### --- Format Prompt --- ###
        formatted_prompt = self.prompt.format(query=query, **prompt_kwargs)

        # Use the provided sys_prompt or fall back to the default
        sys_prompt = sys_prompt or self.default_sys_prompt

        #prompt_logger.info(
        #    f"\nFORMATTED PROMPT (GenericLLMResponse):\n---\n{formatted_prompt}\n---\n"
        #)

        ### --- LLM Generation --- ###
        if stream:
            return self._generate_stream(
                sys_prompt=sys_prompt,
                formatted_prompt=formatted_prompt
            )
        else:
            return self._generate_non_stream(
                sys_prompt=sys_prompt,
                formatted_prompt=formatted_prompt
            )

    # STREAMING (returns a generator)
    def _generate_stream(self, sys_prompt, formatted_prompt):
        # Call the LLM to generate the final answer in streaming mode
        response_generator = self.agent.llm.call_llm_stream(
            sys_prompt, formatted_prompt
        )
        for chunk in response_generator:
            yield chunk

    # NON-STREAMING (returns a string)
    def _generate_non_stream(self, sys_prompt, formatted_prompt) -> str:
        # Call the LLM to generate the final answer
        answer = self.agent.llm.call_llm(sys_prompt, formatted_prompt)
        return answer

class RAGComparison(Action):
    def __init__(
        self,
        agent,
        sys_prompt: str,
        prompt: str,
        vector_stores_map: dict,
        vector_stores_info: dict,
        external_vs: Optional[VectorStore] = None,
        include_external: bool = True,
    ):
        self.agent = agent
        self.default_sys_prompt = sys_prompt
        self.prompt = prompt
        self.vector_stores_map = vector_stores_map
        self.vector_stores_info = vector_stores_info
        self.comparison_use_cases = ["lesfurets-test"]
        self.include_external = include_external
        self.external_vs = external_vs

        self.offering_separator = "\n=== {} ===\n"
        self.result_separator = "\n----\n"

    def execute(
        self, 
        query: str, 
        collection_names: List[str], 
        prompt_kwargs: dict = {}, 
        stream: bool = False, 
        sys_prompt: str = None, 
        **kwargs
    ):
        ### --- Vector Store Search --- ###
        formatted_search_results = self._perform_vector_store_search(
            query=query, 
            sources=collection_names
        )

        ### --- Format Prompt --- ###
        formatted_prompt = self.prompt.format(
            query=query,
            search_results=formatted_search_results,
            **prompt_kwargs,
        )

        # Use the provided sys_prompt or fall back to the default
        sys_prompt = sys_prompt or self.default_sys_prompt

        #prompt_logger.info(f"\nSYSPROMPT:\n---\n{sys_prompt}\n---\n")
        #prompt_logger.info(f"\nPROMPT:\n---\n{self.prompt}\n---\n")
        #prompt_logger.info(
        #    f"\nFORMATTED PROMPT (RAGComparison):\n---\n{formatted_prompt}\n---\n"
        #)

        ### --- LLM Generation --- ###
        if stream:
            return self._generate_stream(
                sys_prompt=sys_prompt,
                formatted_prompt=formatted_prompt
            )
        else:
            return self._generate_non_stream(
                sys_prompt=sys_prompt,
                formatted_prompt=formatted_prompt
            )

    def _generate_stream(self, sys_prompt, formatted_prompt):
        response_generator = self.agent.llm.call_llm_stream(
            sys_prompt, formatted_prompt
        )
        for chunk in response_generator:
            yield chunk

    def _generate_non_stream(self, sys_prompt, formatted_prompt) -> str:
        answer = self.agent.llm.call_llm(sys_prompt, formatted_prompt)
        return answer

    def _perform_vector_store_search(self, query: str, sources: List[str]) -> str:
        """
        Perform a search across all vector stores and format the results by offering.
        'sources' may contain directly a collection name (e.g. "External_vs"), 
        or a vector store name (which itself contains various collections).
        """
        formatted_search_results = []

        logger.debug(f"sources: {sources}")
        for source in sources:
            logger.debug(f"source: {source}")
            if source in self.comparison_use_cases:
                logger.debug(f"source in comparison_use_cases: {source}")
                for offering_collection, vs in self.vector_stores_map.items():
                    logger.debug(f"offering_collection: {offering_collection}")
                    
                    offering_dict = self.vector_stores_info[offering_collection]
                    collection_name = offering_dict['collection_name']
                    
                    logger.debug(f"offering_dict: {offering_dict}")
                    logger.debug(f"collection_name: {collection_name}")
                    
                    node_search_results, _ = vs.search(
                        collection_name=collection_name,
                        query=query, 
                        number=3
                    )

                    offering_full_name = f"{offering_dict['name']} ({offering_dict['provider']})"
                    formatted_search_results.append(
                        self.offering_separator.format(offering_full_name)
                    ) # e.g.: ===== LMF Santé (Mutuelle Familiale) =====
                    if node_search_results:
                        formatted_search_results.append(
                            self._format_search_results(node_search_results)
                        )
                    else:
                        formatted_search_results.append(
                            "No relevant information found for this offering."
                        )
            else:
                logger.info(f"source not in comparison_use_cases : {source}")
                if self.include_external:
                    logger.info(f"searching in external vs - include_external is True")
                    collection_name = source # 'External_vs'
                    vs = self.external_vs
                    logger.debug(f"collection_name : {collection_name}")
                    
                    node_search_results, _ = vs.search(
                        collection_name=collection_name,
                        query=query, 
                        number=3  # Get top 3 documents from the external collection (ameli)
                    )

                    if node_search_results:
                        formatted_search_results.append(
                            dedent(f"""
                            {self.offering_separator.format('Non-offering Specific Information')}\n
                            **Source:** Site officiel de l'Assurance Maladie (Ameli.fr)
                            """
                            )
                        )
                        formatted_search_results.append(
                            self._format_search_results(node_search_results)
                        )

        return "".join(formatted_search_results)

    def _format_search_results(self, search_results: list) -> str:
        """
        Formats the search results from one vector store into a string.
        """
        # TMP:
        search_results = list(search_results.values())
        formatted_results = [
            f"{result[1]}{self.result_separator}" for result in search_results
        ]
        
        #formatted_results = [
        #    f"{node.content}{self.result_separator}" for node in search_results
        #]
        return "".join(formatted_results)