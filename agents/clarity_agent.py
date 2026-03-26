from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from state.agent_state import AgentState
from config.settings import llm

# --- Output schema ---
class ClarityOutput(BaseModel):
    status: Literal["clear", "needs_clarification"] = Field(
        description="Whether the query is clear enough to research"
    )
    clarifying_question: str = Field(
        description="Question to ask if needs_clarification, empty if clear"
    )
    refined_query: str = Field(
        description="Refined query if clear, empty if needs_clarification"
    )
    reason: str = Field(
        description="Brief reason for your decision"
    )

# --- Prompt ---
_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query clarity analyzer for a research assistant.

A query is CLEAR if it:
- Has a specific topic or subject
- Has a clear intent
- Is specific enough to search for

A query NEEDS CLARIFICATION if it:
- Is too vague or ambiguous
- Has multiple possible interpretations
- Is missing key context

Be practical — most reasonable questions are clear enough."""),
    ("human", "Analyze this query: {query}")
])

_chain = _prompt | llm.with_structured_output(ClarityOutput)

# --- Node ---
def clarity_node(state: AgentState) -> AgentState:
    print("\n[ClarityAgent] Running...")
    last_msg = state["messages"][-1].content
    print(f"  Query: '{last_msg}'")

    result = _chain.invoke({"query": last_msg})
    current_count = state.get("clarification_count", 0)

    print(f"  Status: {result.status}")
    print(f"  Reason: {result.reason}")
    print(f"  Clarification count: {current_count}")

    updates = {
        "original_query": last_msg,
        "clarifying_question": result.clarifying_question,
    }

    if result.status == "clear" or current_count >= 2:
        if current_count >= 2 and result.status == "needs_clarification":
            print("  Max clarifications hit — forcing to research")
        updates["clarity_status"] = "clear"
        updates["clarification_count"] = current_count
        updates["messages"] = [AIMessage(
            content=f"Got it. Researching: '{last_msg}'"
        )]
    else:
        updates["clarity_status"] = "needs_clarification"
        updates["clarification_count"] = current_count + 1
        updates["messages"] = [AIMessage(content=result.clarifying_question)]
        print(f"  Asking clarification {current_count + 1} of 2")

    return updates

# --- Ask user node ---
def ask_user_node(state: AgentState) -> AgentState:
    print("\n[ClarityAgent] Paused — waiting for user clarification")
    return {}

# --- Router ---
def route_clarity(state: AgentState) -> Literal["ask_user", "research"]:
    return "ask_user" if state["clarity_status"] == "needs_clarification" else "research"