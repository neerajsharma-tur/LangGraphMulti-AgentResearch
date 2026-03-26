from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    clarity_status: str
    clarifying_question: str
    original_query: str
    clarification_count: int    # NEW — tracks how many times we asked

# --- Structured output schema ---
class ClarityOutput(BaseModel):
    status: Literal["clear", "needs_clarification"] = Field(
        description="Whether the query is clear enough to research"
    )
    clarifying_question: str = Field(
        description="Question to ask if needs_clarification, empty string if clear"
    )
    refined_query: str = Field(
        description="Refined query for research if clear, empty string if needs_clarification"
    )
    reason: str = Field(
        description="Brief reason for your decision"
    )

# --- Prompt ---
clarity_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query clarity analyzer for a research assistant.

Your job is to determine if a user query is clear enough to research effectively.

A query is CLEAR if it:
- Has a specific topic or subject
- Has a clear intent (what they want to know)
- Is specific enough to search for

A query NEEDS CLARIFICATION if it:
- Is too vague or ambiguous (e.g. "tell me stuff", "explain things")
- Has multiple possible interpretations
- Is missing key context (e.g. "how does it work?" — what is "it"?)
- Is too broad to research effectively

Be practical — most reasonable questions are clear enough."""),
    ("human", "Analyze this query: {query}")
])

structured_llm = llm.with_structured_output(ClarityOutput)
clarity_chain = clarity_prompt | structured_llm

# --- Clarity Node ---
def clarity_node(state: AgentState) -> AgentState:
    print("\n--- clarity_node running ---")

    last_msg = state["messages"][-1].content
    print(f"  Analyzing: '{last_msg}'")

    result = clarity_chain.invoke({"query": last_msg})
    print(f"  Status: {result.status}")
    print(f"  Reason: {result.reason}")

    # Read current clarification count — default 0 if first run
    current_count = state.get("clarification_count", 0)
    print(f"  Clarification count: {current_count}")

    updates = {
        "original_query": last_msg,
        "clarifying_question": result.clarifying_question,
    }

    # Force to research if query is clear OR hit the limit of 2
    if result.status == "clear" or current_count >= 2:
        if current_count >= 2 and result.status == "needs_clarification":
            print("  Max clarifications reached — routing to research anyway")
        updates["clarity_status"] = "clear"
        updates["clarification_count"] = current_count
        updates["messages"] = [
            AIMessage(content=f"Query understood. Researching: '{last_msg}'")
        ]

    else:
        # Still under limit — ask for clarification
        updates["clarity_status"] = "needs_clarification"
        updates["clarification_count"] = current_count + 1  # increment
        updates["messages"] = [AIMessage(content=result.clarifying_question)]
        print(f"  Asking clarification {current_count + 1} of 2")

    return updates

# --- Router ---
def route_clarity(state: AgentState) -> Literal["ask_user", "research"]:
    return "ask_user" if state["clarity_status"] == "needs_clarification" else "research"

# --- Ask User Node ---
def ask_user_node(state: AgentState) -> AgentState:
    print("\n--- ask_user_node: waiting for user clarification ---")
    return {}

# --- Mock Research Node ---
def research_node(state: AgentState) -> AgentState:
    print("\n--- research_node: researching... ---")
    query = state.get("original_query", state["messages"][-1].content)
    reply = AIMessage(content=f"[Mock Research] Found results for: '{query}'")
    return {"messages": [reply]}

# --- Build Graph ---
memory = MemorySaver()
builder = StateGraph(AgentState)

builder.add_node("clarity", clarity_node)
builder.add_node("ask_user", ask_user_node)
builder.add_node("research", research_node)

builder.add_edge(START, "clarity")
builder.add_conditional_edges("clarity", route_clarity)
builder.add_edge("ask_user", "clarity")  # loop back after clarification
builder.add_edge("research", END)

graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["ask_user"]
)

# --- Test helper ---
def run_conversation(thread_id: str, query: str, clarifications: list[str] = []):
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 50)
    print(f"Thread: {thread_id}")
    print(f"Query: '{query}'")
    print("=" * 50)

    # Turn 1 — send initial query
    graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )

    # Loop through clarifications
    for i, clarification in enumerate(clarifications):
        snapshot = graph.get_state(config)

        if snapshot.next and "ask_user" in snapshot.next:
            print(f"\nAgent asks: {snapshot.values['messages'][-1].content}")
            print(f"User replies: '{clarification}'")

            graph.update_state(
                config,
                {"messages": [HumanMessage(content=clarification)]}
            )
            graph.invoke(None, config=config)
        else:
            print("\nGraph already finished — no more clarifications needed")
            break

    # Final state
    final = graph.get_state(config)
    print(f"\nFinal answer: {final.values['messages'][-1].content}")
    print(f"Clarity status: {final.values['clarity_status']}")
    print(f"Clarification count: {final.values.get('clarification_count', 0)}")


# --- Test 1: Clear query — goes straight to research ---
run_conversation(
    thread_id="test-001",
    query="What are the main advantages of LangGraph over vanilla LangChain?"
)

# --- Test 2: One clarification needed then clear ---
run_conversation(
    thread_id="test-002",
    query="Tell me about agents",
    clarifications=[
        "I want to know about LangGraph agent patterns for production"
    ]
)

# --- Test 3: Hits max clarifications limit ---
run_conversation(
    thread_id="test-003",
    query="how does it work?",
    clarifications=[
        "I don't know",       # still vague — count becomes 1
        "still not sure",     # still vague — count becomes 2, limit hit
    ]
)