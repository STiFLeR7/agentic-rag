
import pytest
from unittest.mock import MagicMock
from agentic_rag.agent import Agent, CompletionRequest
from agentic_rag.tools import ToolRegistry, Tool
from agentic_rag.llm import InferenceEngine

class MockTool:
    name = "mock_tool"
    description = "A mock tool."
    parameters = {"type": "object", "properties": {"arg": {"type": "string"}}}
    
    def execute(self, arg: str) -> str:
        return f"Executed with {arg}"

@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=InferenceEngine)
    return llm

@pytest.fixture
def agent(mock_llm):
    registry = ToolRegistry()
    registry.register(MockTool())
    return Agent(llm=mock_llm, tools=registry)

def test_agent_single_step(agent, mock_llm):
    # Setup LLM response to immediately answer
    mock_llm.generate.return_value = {
        "choices": [{"text": "Final Answer: The answer is 42."}]
    }
    
    response = agent.run("What is the answer?")
    assert response == "The answer is 42."
    assert mock_llm.generate.call_count == 1

def test_agent_tool_use(agent, mock_llm):
    # Setup LLM response to use tool then answer
    # Step 1: Think -> Action
    # Step 2: Answer based on observation
    
    step1_response = {
        "choices": [{"text": 'Think: I need to use the mock tool.\nAction: mock_tool\nAction Input: {"arg": "test"}'}]
    }
    step2_response = {
        "choices": [{"text": "Final Answer: Tool said Executed with test."}]
    }
    
    mock_llm.generate.side_effect = [step1_response, step2_response]
    
    response = agent.run("Run the tool.")
    
    assert "Tool said Executed with test" in response
    assert mock_llm.generate.call_count == 2
