from rakam_systems.components.agents.actions import Action


class MockAgent:
    """
    A mock version of the Agent class to simulate the behavior of the actual Agent.
    """
    def __init__(self):
        self.actions = {}
        self.state = {}

    def add_action(self, action_name: str, action: Action):
        self.actions[action_name] = action

    def choose_action(self, input: str):
        """
        Select the appropriate action based on the input.
        For simplicity, we just return the action associated with the input.
        """
        return self.actions.get(input, None)

    def execute_action(self, action_name: str, **kwargs):
        action = self.choose_action(action_name)
        if action:
            return action.execute(**kwargs)
        return None


class MockAction(Action):
    """
    Mock version of Action class to simulate action execution.
    """
    def __init__(self, agent, **kwargs):
        self.agent = agent

    def execute(self, **kwargs):
        """
        Return a mock response simulating action execution.
        """
        return f"Mock action executed with args: {kwargs}"
