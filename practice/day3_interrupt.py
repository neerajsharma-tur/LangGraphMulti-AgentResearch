from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]
    clarity_status: str

# --- Nodes ---

def clarity_node(state: State) -> State:
    last_msg = state["messages"][-1].content.lower()
    status = "clear" if len(last_msg.split()) >= 5 else "needs_clarification"
    print(f"--- clarity_node: status = '{status}' ---")
    return {"clarity_status": status}

def ask_user_node(state: State) -> State:
    # This fires AFTER the interrupt resumes
    # By this point the user has already added their clarification
    print("--- ask_user_node: processing clarification ---")
    reply = AIMessage(content="Thanks for clarifying! Let me research that now.")
    return {"messages": [reply]}

def research_node(state: State) -> State:
    print("--- research_node: researching... ---")
    reply = AIMessage(content="Here are the research results for your query.")
    return {"messages": [reply]}

# --- Router ---

def route_by_clarity(state: State) -> Literal["ask_user", "research"]:
    return "ask_user" if state["clarity_status"] == "needs_clarification" else "research"


# Add a second interrupt — also pause interrupt_before=["research"] so you can inspect what's 
# in state right before research fires. Check the snapshot's next value at that point.
#  This will teach you how to debug your pipeline later when agents misbehave.
# --- Build Graph ---

builder = StateGraph(State)

builder.add_node("clarity", clarity_node)
builder.add_node("ask_user", ask_user_node)
builder.add_node("research", research_node)

builder.add_edge(START, "clarity")
builder.add_conditional_edges("clarity", route_by_clarity)
builder.add_edge("ask_user", "research")  # after clarification, still research
builder.add_edge("research", END)

# MemorySaver — stores state between invocations
memory = MemorySaver()

# interrupt_before=["ask_user"] — pause BEFORE ask_user runs
graph = builder.compile(checkpointer=memory, interrupt_before=["ask_user","research"])

# --- thread config — every conversation needs a unique thread_id ---
config = {"configurable": {"thread_id": "thread-001"}}

print("=" * 50)
print("TURN 1: Send a short/unclear query")
print("=" * 50)

result = graph.invoke(
    {"messages": [HumanMessage(content="Tell me stuff")]},
    config=config
)

# Graph paused — let's inspect where it stopped
snapshot = graph.get_state(config)
print(f"\nGraph paused at: {snapshot.next}")
print(f"Clarity status: {snapshot.values['clarity_status']}")
print(f"Messages so far: {[m.content for m in snapshot.values['messages']]}")

print("\n" + "=" * 50)
print("TURN 2: User provides clarification")
print("=" * 50)

# Inject the user's clarification into state and resume
graph.update_state(
    config,
    {"messages": [HumanMessage(content="I want to know about LangGraph interrupt patterns in detail")]}
)

# Resume by passing None as input — it picks up from the checkpoint
result = graph.invoke(None, config=config)


# Graph paused — let's inspect where it stopped
snapshot = graph.get_state(config)
print(f"\nGraph paused at: {snapshot.next}")
print(f"Clarity status: {snapshot.values['clarity_status']}")
print(f"Messages so far: {[m.content for m in snapshot.values['messages']]}")


print("\n" + "=" * 50)
print("TURN 3: Resume research after second interrupt")
print("=" * 50)

# No new input needed — just resume
result = graph.invoke(None, config=config)

print(f"\nFinal messages:")
for msg in result["messages"]:
    print(f"  {msg.__class__.__name__}: {msg.content}")