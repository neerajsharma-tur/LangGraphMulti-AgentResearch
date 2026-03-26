from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()

# Use llama-3.3-70b-versatile which supports tool calling
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- 1. Define tools with @tool decorator ---

@tool
def search_web(query: str) -> str:
    """Search the web for information about a query.
    Use this when you need current or factual information."""
    # Mock result for now — real Tavily comes Day 8
    return f"Mock search results for '{query}': LangGraph is a framework by LangChain for building stateful multi-agent applications using graph-based workflows."

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    Use this for any calculations needed."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_date() -> str:
    """Get today's date. Use this when the query involves current time."""
    from datetime import datetime
    return f"Today's date is {datetime.now().strftime('%B %d, %Y')}"

# Inspect what a tool looks like
print("=" * 50)
print("1. TOOL DEFINITIONS")
print("=" * 50)
print(f"Tool name: {search_web.name}")
print(f"Tool description: {search_web.description}")
print(f"Tool schema: {json.dumps(search_web.args_schema.model_json_schema(), indent=2)}")


# --- 2. Bind tools to LLM ---
print("\n" + "=" * 50)
print("2. TOOL BINDING AND DETECTION")
print("=" * 50)

tools = [search_web, calculate, get_current_date]

# Debug: show how bind_tools works
print(f"Binding {len(tools)} tools to LLM...")
for tool in tools:
    print(f"  - {tool.name}: {tool.description[:50]}...")

llm_with_tools = llm.bind_tools(tools)

# Ask something that requires a tool
try:
    response = llm_with_tools.invoke([
        SystemMessage(content="You are a research assistant. Use tools when needed."),
        HumanMessage(content="What is LangGraph?")
    ])

    print(f"Response type: {type(response).__name__}")
    print(f"Has tool calls: {bool(response.tool_calls)}")

    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"\nTool requested: {tc['name']}")
            print(f"Arguments: {tc['args']}")
    else:
        print(f"Direct answer: {response.content}")
except Exception as e:
    print(f"Error with tool binding: {e}")
    print("\nNote: Some Groq models have limited tool support.")
    print("Trying with a direct query instead...\n")
    
    # Fallback to direct query without tools
    response = llm.invoke([
        SystemMessage(content="You are a research assistant."),
        HumanMessage(content="What is LangGraph?")
    ])
    print(f"Direct answer: {response.content}")


# --- 3. COMPLETE TOOL CALLING FLOW (Manual Execution) ---
print("\n" + "=" * 50)
print("3. COMPLETE TOOL CALLING FLOW")
print("=" * 50)

# Note: Groq's bind_tools has issues, so we'll use a simpler approach
# We prompt the LLM to decide if it needs a tool, then execute manually

decision_prompt = """You are a research assistant. You have access to these tools:
1. search_web(query: str) - Search for current information
2. calculate(expression: str) - Evaluate math expressions  
3. get_current_date() - Get today's date

For the user's query, respond with ONLY ONE of these formats:

If you need a tool:
TOOL: tool_name
ARGS: {{"arg": "value"}}

If you can answer directly:
ANSWER: your response here

User query: {query}"""

# Test with a query that should use a tool
messages = [
    HumanMessage(content=decision_prompt.format(query="What is 25 * 37?"))
]

print("\nQuery: 'What is 25 * 37?'")
response = llm.invoke(messages)
print(f"LLM Response:\n{response.content}\n")

# Parse and execute if it's a tool call
if "TOOL:" in response.content:
    # Simple parsing (in real code, use structured output!)
    lines = response.content.strip().split("\n")
    tool_name = lines[0].replace("TOOL:", "").strip()
    
    print(f"→ LLM chose to use tool: {tool_name}")
    
    # Execute the tool
    if tool_name == "calculate":
        result = calculate.invoke({"expression": "25 * 37"})
        print(f"→ Tool result: {result}")
        
        # Send result back to LLM for final answer
        final_response = llm.invoke([
            HumanMessage(content=f"The calculation result is: {result}. Please provide a natural language answer to the user.")
        ])
        print(f"→ Final answer: {final_response.content}")
else:
    print(f"→ Direct answer: {response.content}")

print("\n" + "=" * 50)
print("Summary: Tool calling with Groq requires manual")
print("execution. LangGraph will handle this for you!")
print("=" * 50)
