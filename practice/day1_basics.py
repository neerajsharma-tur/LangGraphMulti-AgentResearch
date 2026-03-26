from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

# 1. Define State — this is the "memory" that flows through your graph
class State(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages = append, not overwrite
    user_name: str

# 2. Define Nodes — pure functions: State in, State out
def greet_node(state: State) -> State:
    print(f"--- greet_node running ---")
    name = state.get("user_name", "stranger")
    reply = AIMessage(content=f"Hello {name}! I got your message.")
    return {"messages": [reply]}

def farewell_node(state: State) -> State:
    print(f"--- farewell_node running ---")
    reply = AIMessage(content="Goodbye! Come back anytime.")
    return {"messages": [reply]}

# 3. Build the Graph
builder = StateGraph(State)

builder.add_node("greet", greet_node)
builder.add_node("farewell", farewell_node)

# 4. Add Edges — define the flow
builder.add_edge(START, "greet")
builder.add_edge("greet", "farewell")
builder.add_edge("farewell", END)

# 5. Compile — this validates and locks the graph
graph = builder.compile()

# 6. Invoke — run it!
initial_state = {
    "messages": [HumanMessage(content="Hey there!")],
    "user_name": "Arjun"
}

result = graph.invoke(initial_state)

print(result["messages"])
for msg in result["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content}")