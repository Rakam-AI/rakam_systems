import logging
from abc import ABC, abstractmethod
from typing import Any

import dotenv
from rakam_systems.components.agents.actions import Action
from rakam_systems.custom_loggers import prompt_logger
from rakam_systems.components.base import LLM
from rakam_systems.components.component import Component

dotenv.load_dotenv()

class Agent(Component):
    def __init__(self, model: str):
        super().__init__()  # Ensure proper initialization of the parent Component
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

    def call_main(self, **kwargs) -> dict:
        """
        Main method for executing the component's functionality.
        Processes the state, selects, and executes an action.

        :param kwargs: Additional keyword arguments required for processing.
        :return: Result of executing the selected action.
        """
        input_data = kwargs.get('input', '')
        state = self.process_state(input=input_data)
        chosen_action = self.choose_action(input=input_data, state=state)
        result = chosen_action.execute(**kwargs)
        return result

    def test(self, **kwargs) -> bool:
        """
        Method for testing the component's functionality.
        
        :param kwargs: Optional arguments to simulate input or state.
        :return: Boolean indicating if the test passed.
        """
        try:
            # Perform a mock process to verify everything works as expected
            mock_state = self.process_state(input="test input")
            mock_action = self.choose_action(input="test input", state=mock_state)
            mock_result = mock_action.execute(**kwargs)
            return True  # Test passes if no exceptions are raised
        except Exception as e:
            logging.error(f"Test failed: {e}")
            return False
