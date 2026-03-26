from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Replace ChatOpenAI with ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


class State(TypedDict):
    messages: Annotated[list, add_messages]

# --- Node ---

def chat_node(state: State) -> State:
    system = SystemMessage(content="You are a helpful research assistant. Keep answers concise.")
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

# --- Graph ---

builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# --- Stream node events ---

print("=" * 50)
print("STREAMING — node by node updates")
print("=" * 50)

config = {"configurable": {"thread_id": "stream-001"}}

inputs = {"messages": [HumanMessage(content="What is LangGraph in one sentence?")]}

# stream_mode="updates" gives you each node's output as it completes
for chunk in graph.stream(inputs, config=config, stream_mode="updates"):
    for node_name, node_output in chunk.items():
        print(f"\n[{node_name}] output:")
        for msg in node_output["messages"]:
            print(f"  {msg.content}")

print("\n" + "=" * 50)
print("TOKEN STREAMING — word by word")
print("=" * 50)

config2 = {"configurable": {"thread_id": "stream-002"}}

inputs2 = {"messages": [HumanMessage(content="Explain LangGraph state in 2 sentences.")]}

print("\nResponse: ", end="", flush=True)

# stream_mode="messages" streams individual tokens
for token, metadata in graph.stream(inputs2, config=config2, stream_mode="messages"):
    if hasattr(token, "content") and token.content:
        print(token.content, end="", flush=True)

print("\n")


print("=" * 50)
print("MULTI-TURN CONVERSATION")
print("=" * 50)

config3 = {"configurable": {"thread_id": "conversation-001"}}

# Turn 1
print("\nYou: What is LangGraph?")
result = graph.invoke(
    {"messages": [HumanMessage(content="What is LangGraph?")]},
    config=config3
)
print(f"AI: {result['messages'][-1].content}")

# Turn 2 — LLM remembers Turn 1 because same thread_id
print("\nYou: What was my previous question?")
result = graph.invoke(
    {"messages": [HumanMessage(content="What was my previous question?")]},
    config=config3
)
print(f"AI: {result['messages'][-1].content}")

# Turn 3
print("\nYou: Give me one use case for it.")
result = graph.invoke(
    {"messages": [HumanMessage(content="Give me one use case for it.")]},
    config=config3
)
print(f"AI: {result['messages'][-1].content}")

# Inspect full conversation history
print("\n--- Full conversation history in state ---")
snapshot = graph.get_state(config3)
for msg in snapshot.values["messages"]:
    role = "You" if isinstance(msg, HumanMessage) else "AI"
    print(f"  {role}: {msg.content}")