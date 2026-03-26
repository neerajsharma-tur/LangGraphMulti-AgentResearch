from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

# 1. Define State — this is the "memory" that flows through your graph
class State(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages = append, not overwrite
    sentiment: str  # "happy" or "sad" or "help"

def analyze_node(state: State) -> str:
    last_msg = state["messages"][-1].content.lower()
    sentiment = "happy" if any(w in last_msg for w in ["good","great","love","happy"]) else "sad" if any(w in last_msg for w in ["bad","terrible","hate","sad"]) else "help"
    print(f"--- analyze_node: detected '{sentiment}' ---")
    return {"sentiment": sentiment}


# 2. Define Nodes — pure functions: State in, State out
def happy_node(state: State) -> State:
    reply = AIMessage(content="Glad you're feeling great!")
    return {"messages": [reply]}


def sad_node(state: State) -> State:
    reply = AIMessage(content="I'm sorry to hear that. How can I help?")
    return {"messages": [reply]}

def help_node(state: State) -> State:
    reply = AIMessage(content="I can help you with that. What do you need?")
    return {"messages": [reply]}

# The router function — returns the NAME of the next node
def route_by_sentiment(state: State) -> Literal["happy_response", "sad_response", "help_response"]:
    return "happy_response" if state["sentiment"] == "happy" else "sad_response" if state["sentiment"] == "sad" else "help_response"


# 3. Build the Graph
builder = StateGraph(State)

builder.add_node("analyze", analyze_node)
builder.add_node("happy_response", happy_node)
builder.add_node("sad_response", sad_node)
builder.add_node("help_response", help_node)


    
# 4. Add Edges — define the flow
builder.add_edge(START, "analyze")
builder.add_conditional_edges("analyze",route_by_sentiment)
builder.add_edge("happy_response", END)
builder.add_edge("sad_response", END)
builder.add_edge("help_response", END)

# 5. Compile — this validates and locks the graph
graph = builder.compile()


    # Test both paths
for msg in ["I love this!", "I feel terrible today","I need help from you"]:
    print(f"\nInput: {msg}")
    result = graph.invoke({"messages": [HumanMessage(content=msg)]})
    print(f"Reply: {result['messages'][-1].content}")

