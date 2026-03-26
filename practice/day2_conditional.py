from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]
    clarity_status: str  # "clear" or "needs_clarification"

# --- Nodes ---

def clarity_node(state: State) -> State:
    last_msg = state["messages"][-1].content.lower()
    # Mock logic — no LLM yet, we add that on Day 7
    if len(last_msg.split()) >= 5:  # long enough = clear
        status = "clear"
    else:
        status = "needs_clarification"
    print(f"--- clarity_node: status = '{status}' ---")
    return {"clarity_status": status}

def ask_user_node(state: State) -> State:
    print("--- ask_user_node: query needs clarification ---")
    reply = AIMessage(content="Could you clarify? Please add more detail to your question.")
    return {"messages": [reply]}

def research_node(state: State) -> State:
    print("--- research_node: query is clear, proceeding ---")
    reply = AIMessage(content="Researching your query now...")
    return {"messages": [reply]}

# --- Router function ---

def route_by_clarity(state: State) -> Literal["ask_user", "research"]:
    if state["clarity_status"] == "needs_clarification":
        return "ask_user"
    return "research"

# --- Build Graph ---

builder = StateGraph(State)

builder.add_node("clarity", clarity_node)
builder.add_node("ask_user", ask_user_node)
builder.add_node("research", research_node)

builder.add_edge(START, "clarity")
builder.add_conditional_edges("clarity", route_by_clarity)
builder.add_edge("ask_user", END)
builder.add_edge("research", END)

graph = builder.compile()

# --- Test both paths ---

print("=== Test 1: short query (needs clarification) ===")
result = graph.invoke({
    "messages": [HumanMessage(content="Tell me")]
})
print(f"Last reply: {result['messages'][-1].content}")
print(f"Clarity status: {result['clarity_status']}")

print("\n=== Test 2: detailed query (clear) ===")
result = graph.invoke({
    "messages": [HumanMessage(content="What are the benefits of using LangGraph for agent workflows?")]
})
print(f"Last reply: {result['messages'][-1].content}")
print(f"Clarity status: {result['clarity_status']}")