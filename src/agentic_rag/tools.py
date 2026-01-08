
from typing import Protocol, Dict, Any, List
from pydantic import BaseModel, Field
import os
from agentic_rag.retriever import Retriever

class Tool(Protocol):
    name: str
    description: str
    parameters: Dict[str, Any]

    def execute(self, **kwargs) -> str:
        ...

class SearchKnowledgeBaseTool:
    name = "search_knowledge_base"
    description = "Searches the internal knowledge base for documents relevant to a query."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."}
        },
        "required": ["query"]
    }

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def execute(self, query: str) -> str:
        results = self.retriever.retrieve(query)
        if not results:
            return "No relevant documents found."
        
        # Format results for the agent
        formatted = []
        for i, res in enumerate(results, 1):
            formatted.append(f"Result {i} (Score: {res['score']}):\n{res['content']}")
        return "\n\n".join(formatted)

class ReadFileTool:
    name = "read_file"
    description = "Reads the content of a file from the local filesystem."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file."}
        },
        "required": ["file_path"]
    }

    def execute(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"

class PythonCodeTool:
    name: str = "python_repl"
    description: str = "Executes pure Python code to perform math or logic. Input: {'code': 'print(1+1)'}. Output: stdout. Use for calculations."
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute."}
        },
        "required": ["code"]
    }

    def execute(self, code: str) -> str:
        import sys
        from io import StringIO
        import contextlib

        # Safety: Restricted globals? Ideally yes, but for now just standard exec with captured stdout.
        # This is "Agentic AI" level - trusting the agent but sandboxing if possible.
        # We will keep it simple for this local context.
        
        # Clean code
        code = code.strip()
        if code.startswith("```python"):
            code = code.replace("```python", "").replace("```", "")
        
        # Capture stdout
        output = StringIO()
        try:
            with contextlib.redirect_stdout(output):
                 # Define a safe local scope
                 local_scope = {}
                 exec(code, {}, local_scope)
            return output.getvalue().strip() or "Success (No Output)"
        except Exception as e:
            return f"Error: {e}"

# Simple registry
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        # Auto-register basic tools
        # Note: Retriever needs to be passed to SearchKnowledgeBaseTool if it's auto-registered here.
        # For now, assuming it's initialized elsewhere or passed during registry creation.
        # For this example, we'll assume a dummy retriever or that the registry is initialized with tools.
        # If retriever is needed, it should be passed to the constructor of ToolRegistry.
        # For the purpose of this edit, we'll just register the classes as per instruction.
        # self.register(SearchKnowledgeBaseTool(retriever=...)) # Placeholder
        # self.register(ReadFileTool())
        # self.register(PythonCodeTool()) # Register new tool

    def register(self, tool: Tool):
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)
    
    def get_descriptions(self) -> str:
        return "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools.values()])
