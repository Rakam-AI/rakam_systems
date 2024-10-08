import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

import dotenv

from rakam_systems.components.agents.actions import Action
from rakam_systems.custom_loggers import prompt_logger
from rakam_systems.components.base import LLM

dotenv.load_dotenv()

class Agent(ABC):
    def __init__(self, model: str):
        self.llm = LLM(model=model)  # Initialize the LLM with the specified model
        self.state = {}  # Initialize an empty state dictionary
        self.actions = {}  # Dictionary to store actions by name

    def process_state(self, **kwargs) -> dict:
        """
        Build a temporary state based on the input query.

        :param input: The input query string.
        :return: A dictionary representing the state.
        """
        # Example: Building a state based on keyword analysis
        state = {}
        return state

    def add_action(self, action_name: str, action: Action):
        """
        Adds an action to the agent's set of available actions.

        :param action_name: A string name for the action (e.g., 'rag_generation').
        :param action: An instance of an Action subclass.
        """
        self.actions[action_name] = action

    @abstractmethod
    def choose_action(self, input: str, state: Any) -> Action:
        """
        Abstract method to select an action based on input.

        :param input: A string input based on which the action is selected.
        :return: The selected Action instance.
        """
        pass

    def execute_action(self, action_name: str, **kwargs):
        """
        Executes the selected action with the provided arguments.

        :param action_name: The name of the action to execute.
        :param kwargs: Arguments to pass to the action's execute method.
        :return: The result of the action's execute method.
        """
        action = self.choose_action(action_name)
        return action.execute(**kwargs)

