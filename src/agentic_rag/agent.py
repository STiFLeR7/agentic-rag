

from typing import TypedDict, Annotated, List, Union, Dict, Any
import operator
import json
import logging

# Setup Logger
logging.basicConfig(filename='d:/agentic-rag/debug_agent.log', level=logging.INFO, format='%(asctime)s %(message)s')

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from agentic_rag.llm import InferenceEngine
from agentic_rag.tools import ToolRegistry

# --- State Application Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    # We can add more state here like 'scratchpad' or 'iterations'

class Agent:

    def __init__(self, llm: InferenceEngine, tools: ToolRegistry, system_prompt: str = ""):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt or self._default_prompt()
        
        # Build Graph
        builder = StateGraph(AgentState)
        builder.add_node("rewrite", self.rewrite_query) # New Node
        builder.add_node("agent", self.call_model)
        builder.add_node("action", self.call_tools)
        
        builder.set_entry_point("rewrite") # Start at Rewrite
        
        builder.add_edge("rewrite", "agent") # Rewrite -> Agent
        
        # Conditional Edge
        builder.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "action",
                "end": END
            }
        )
        
        # Normal Edge
        builder.add_edge("action", "agent")
        
        self.graph = builder.compile()

    def _default_prompt(self) -> str:
        return f"""You are a Function Calling Engine.
You have access to a local knowledge base.
You are NOT a chat assistant. You are a process that retrieves data.

TOOLS:
{self.tools.get_descriptions()}

INSTRUCTIONS:
1. You must output an 'Action' to find the answer.
2. You must NOT apologize or refuse.
3. You must NOT say "I cannot".
4. Format:
Action: <tool_name>
Action Input: <json_params>

Example:
User: Status?
Action: search_knowledge_base
Action Input: "{{"query": "status"}}"
Observation: OK
Final Answer: OK

Begin.
"""

    def rewrite_query(self, state: AgentState) -> Dict:
        """
        Node: Rewrites the initial user query to be more specific.
        """
        messages = state['messages']
        # Assume the last message is the User's query (or the first human message)
        # In this flow, we usually just entered, so messages[-1] is User.
        original_query = messages[-1].content
        
        print(f"--- Rewriting Query: '{original_query}' ---")
        logging.info(f"Rewriting Query: '{original_query}'")
        
        # Prompt for rewriting
        rewrite_prompt = [
            {"role": "system", "content": """You are a Query Expert. 
Your job is to rewrite the user's query to be specific to 'Project Orion' (a confidential technical project).
Do NOT answer the question.
Do NOT add usage instructions.

Examples:
Query: "What mitigates Cascade Resonance?"
Rewritten: "In the context of Project Orion, what mitigates 'Cascade Resonance'?"

Query: "Who is the Chief Architect?"
Rewritten: "Who is the Chief Architect of Project Orion?"

Query: "Status?"
Rewritten: "What is the current status of Project Orion?"
"""},
            {"role": "user", "content": f"Rewrite this query: {original_query}"}
        ]
        
        response = self.llm.chat(rewrite_prompt, temperature=0.0)
        rewritten_query = response['choices'][0]['message']['content'].strip()
        
        # Clean up quotes if present
        rewritten_query = rewritten_query.strip('"').strip("'")
        
        print(f"Rewritten Query: '{rewritten_query}'")
        logging.info(f"Rewritten Query: '{rewritten_query}'")
        
        # Replace the user's message in the state with the rewritten one?
        # Or just append? 
        # Better to REPLACE the last HumanMessage so the Agent sees the Good Query.
        # But StateGraph usually appends.
        # We can return a modification to the last message if we want, or just append a new HumanMessage?
        # If we append, the agent sees: User: <bad>, User: <good>.
        # Let's try replacing if LangGraph allows, or just act as if the User said the new thing.
        # Since we are using annotated add, we return a new message.
        # Let's return a System note + the new Query as the "Effective" user query?
        # Actually, simpler: The Agent just sees the message list.
        # Let's add a System message saying "Rewritten Query: ..." then the Agent can use it?
        # Or better: We construct a new list for the agent node?
        
        # Strategy: Return a NEW HumanMessage that overrides/supplements.
        return {"messages": [HumanMessage(content=rewritten_query)]}

    def call_model(self, state: AgentState) -> Dict:
        messages = state['messages']
        
        # Convert LangChain messages to ChatML format for local LLM
        # This is the "Adapter" layer
        formatted_messages = []
        # specific handling: if first message is not system, add system
        if not isinstance(messages[0], SystemMessage):
             formatted_messages.append({"role": "system", "content": self.system_prompt})
        
        for m in messages:
            if isinstance(m, SystemMessage):
                formatted_messages.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                formatted_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                formatted_messages.append({"role": "assistant", "content": m.content})
            # We treat Tool output as User message "Observation: ..." in this specific local LLM prompting strategy
            # If we had a "ToolMessage", we would convert it too.
            
        print(f"--- Calling LLM with {len(formatted_messages)} messages ---")
            
        response = self.llm.chat(
            formatted_messages,
            stop=["Observation:"],
            temperature=0.0
        )
        content = response['choices'][0]['message']['content'].strip()
        print(f"LLM Output:\n{content}")
        
        return {"messages": [AIMessage(content=content)]}

    def should_continue(self, state: AgentState) -> str:
        last_message = state['messages'][-1]
        content = last_message.content
        
        # Check for Final Answer
        if "Final Answer:" in content:
            return "end"
            
        # Check for Action
        if "Action:" in content:
            return "continue"
            
        # Heuristic: Loop detection (Repeated content)
        # Note: LangGraph manages state append, so we check history
        messages = state['messages']
        if len(messages) >= 3:
             # prev assistant message
             # messages are [System, Human, AI, Human(Obs), AI]
             # simple check: if last AI message == previous AI message
             ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
             if len(ai_msgs) >= 2:
                 if ai_msgs[-1].content.strip() == ai_msgs[-2].content.strip():
                     print("DEBUG: Loop detected in Edge. Forcing END.")
                     return "end"
        
        return "end" # Default if no action found

    def call_tools(self, state: AgentState) -> Dict:
        last_message = state['messages'][-1]
        content = last_message.content
        
        # Parse Action
        lines = content.split('\n')
        action_line = next((l for l in lines if l.startswith("Action:")), None)
        input_line = next((l for l in lines if l.startswith("Action Input:")), None)
        
        observation = "Error: Could not parse action."
        
        if action_line and input_line:
            tool_name = action_line.replace("Action:", "").strip()
            try:
                raw_input = input_line.replace("Action Input:", "").strip()
                # Heuristic fix for quotes
                if not raw_input.startswith("{") and not raw_input.startswith("\""):
                        raw_input = f'"{raw_input}"'
                
                try:
                    tool_input = json.loads(raw_input)
                except:
                    tool_input = raw_input.strip('"')

                # Robustness mapping
                if not isinstance(tool_input, dict):
                        if tool_name == "search_knowledge_base":
                            tool_input = {"query": str(tool_input)}
                        elif tool_name == "read_file":
                            tool_input = {"file_path": str(tool_input)}

                tool = self.tools.get(tool_name)
                if tool:
                    print(f"Executing {tool_name} with {tool_input}...")
                    observation = tool.execute(**tool_input)
                else:
                    observation = f"Error: Tool '{tool_name}' not found."
            except Exception as e:
                observation = f"Error: {e}"
        
        print(f"Observation: {observation}")
        # Return as HumanMessage to prompt next step
        return {"messages": [HumanMessage(content=f"Observation: {observation}")]}

    def run(self, query: str, max_steps: int = 5) -> str: # max_steps is handled by recursion_limit in LangGraph usually
        # Initial State
        # Rewriter will handle enhancement
        
        inputs = {"messages": [SystemMessage(content=self.system_prompt), HumanMessage(content=query)]}
        
        # Execute Graph
        config = {"recursion_limit": max_steps * 2} # *2 because distinct User/Asst steps
        try:
            final_state = self.graph.invoke(inputs, config=config)
            return final_state['messages'][-1].content
        except Exception as e:
            return f"Error: {e}"
