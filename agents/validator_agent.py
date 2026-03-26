from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from state.agent_state import AgentState
from config.settings import llm

# --- Output schema ---
class ValidationOutput(BaseModel):
    result: Literal["sufficient", "insufficient"] = Field(
        description="Whether the findings sufficiently answer the query"
    )
    reason: str = Field(
        description="Why the findings are sufficient or insufficient"
    )
    missing: str = Field(
        description="What key information is missing if insufficient, empty if sufficient"
    )

# --- Prompt ---
_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research quality validator.

Given a query and research findings, decide if the findings are sufficient.

Findings are SUFFICIENT if they:
- Directly answer the main question
- Provide enough detail to be useful
- Cover the key aspects of the query

Findings are INSUFFICIENT if they:
- Miss the main point of the query
- Are too vague or generic
- Lack important details"""),
    ("human", """Query: {query}

Research findings:
{findings}

Are these findings sufficient to answer the query?""")
])

_chain = _prompt | llm.with_structured_output(ValidationOutput)

# --- Node ---
def validator_node(state: AgentState) -> AgentState:
    attempts = state.get("research_attempts", 0)
    print(f"\n[ValidatorAgent] Running — attempt {attempts} of 3")

    query = state.get("original_query", "")
    findings = state.get("research_findings", "")

    # Rule check first — if max attempts hit, force sufficient
    if attempts >= 3:
        print("  Max attempts reached — forcing sufficient")
        return {
            "validation_result": "sufficient",
            "validation_reason": "Max research attempts reached"
        }

    # LLM judge
    try:
        result = _chain.invoke({
            "query": query,
            "findings": findings
        })
        print(f"  Validation: {result.result}")
        print(f"  Reason: {result.reason}")
        if result.missing:
            print(f"  Missing: {result.missing}")

        return {
            "validation_result": result.result,
            "validation_reason": result.reason
        }

    except Exception as e:
        print(f"  Validation failed: {e} — defaulting to sufficient")
        return {
            "validation_result": "sufficient",
            "validation_reason": "Validation error — proceeding"
        }

# --- Router ---
def route_validator(state: AgentState) -> Literal["research", "synthesis"]:
    result = state.get("validation_result", "sufficient")
    attempts = state.get("research_attempts", 0)

    if result == "insufficient" and attempts < 3:
        print(f"  Insufficient — retrying research (attempt {attempts})")
        return "research"

    print(f"  Sufficient (or max attempts) — routing to synthesis")
    return "synthesis"