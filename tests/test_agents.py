import pytest
from rakam_systems.components.agents.mock_agent import MockAgent, MockAction

@pytest.fixture
def mock_agent_fixture():
    """
    Fixture for creating a mock agent instance.
    """
    return MockAgent()

def test_add_action(mock_agent_fixture):
    """
    Test adding actions to the mock agent.
    """
    action = MockAction(mock_agent_fixture)
    mock_agent_fixture.add_action("mock_action", action)
    
    assert "mock_action" in mock_agent_fixture.actions
    assert isinstance(mock_agent_fixture.actions["mock_action"], MockAction)

def test_choose_action(mock_agent_fixture):
    """
    Test choosing an action from the mock agent.
    """
    action = MockAction(mock_agent_fixture)
    mock_agent_fixture.add_action("mock_action", action)
    
    chosen_action = mock_agent_fixture.choose_action("mock_action")
    
    assert chosen_action is not None
    assert isinstance(chosen_action, MockAction)

def test_execute_action(mock_agent_fixture):
    """
    Test executing an action using the mock agent.
    """
    action = MockAction(mock_agent_fixture)
    mock_agent_fixture.add_action("mock_action", action)
    
    result = mock_agent_fixture.execute_action("mock_action", param="test_param")
    
    assert result == "Mock action executed with args: {'param': 'test_param'}"

def test_execute_action_with_no_action(mock_agent_fixture):
    """
    Test executing an action that doesn't exist in the agent.
    """
    result = mock_agent_fixture.execute_action("non_existent_action", param="test_param")
    
    assert result is None
