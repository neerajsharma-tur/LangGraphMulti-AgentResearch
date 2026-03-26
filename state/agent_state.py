from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Conversation
    messages:            Annotated[list, add_messages]

    # Clarity agent fields
    clarity_status:      str   # "clear" | "needs_clarification"
    clarifying_question: str
    original_query:      str
    clarification_count: int

    # Research agent fields
    research_findings:   str
    confidence_score:    int   # 1-10
    research_attempts:   int

    # Validator agent fields
    validation_result:   str   # "sufficient" | "insufficient"
    validation_reason:   str

    # Synthesis agent fields
    final_answer:        str