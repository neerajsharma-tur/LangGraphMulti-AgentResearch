from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from state.agent_state import AgentState
from config.settings import llm
from tools.search import search_tool

# --- Output schema ---
class ResearchOutput(BaseModel):
    findings: str = Field(
        description="Clear structured summary of what was found"
    )
    confidence_score: int = Field(
        description="Confidence 1-10 on how well results answer the query",
        ge=1, le=10
    )
    is_sufficient: bool = Field(
        description="True if findings sufficiently answer the query"
    )
    reasoning: str = Field(
        description="Why you assigned this confidence score"
    )

# --- Prompt ---
_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research analyst. Summarize web search results clearly.

Confidence scoring guide:
- 8-10: Comprehensive direct answer found with good sources
- 6-7:  Good information but some gaps
- 4-5:  Partial information, missing key details
- 1-3:  Very little useful information found

Be honest — do not inflate scores."""),
    ("human", "Query: {query}\n\nSearch results:\n{search_results}\n\nAnalyze these results.")
])

_chain = _prompt | llm.with_structured_output(ResearchOutput)

def _run_search(query: str) -> str:
    """Run Tavily search and format results for LLM."""
    try:
        raw = search_tool.invoke({"query": query})
        results = raw.get("results", [])
        print(f"  Got {len(results)} results from Tavily")

        formatted = ""
        for i, r in enumerate(results, 1):
            formatted += f"\nResult {i}:\n"
            formatted += f"  Title: {r.get('title', 'N/A')}\n"
            formatted += f"  URL: {r.get('url', 'N/A')}\n"
            formatted += f"  Content: {r.get('content', 'N/A')[:300]}\n"

        if raw.get("answer"):
            formatted = f"Direct answer: {raw['answer']}\n\n" + formatted

        return formatted

    except Exception as e:
        print(f"  Search failed: {e}")
        return "Search failed — no results available"

def _analyze_results(query: str, search_results: str) -> ResearchOutput:
    """Run LLM analysis on search results."""
    try:
        return _chain.invoke({
            "query": query,
            "search_results": search_results
        })
    except Exception as e:
        print(f"  Analysis failed: {e}")
        return ResearchOutput(
            findings="Analysis failed",
            confidence_score=1,
            is_sufficient=False,
            reasoning=str(e)
        )

# --- Node ---
def research_node(state: AgentState) -> AgentState:
    print("\n[ResearchAgent] Running...")
    query = state.get("original_query") or state["messages"][-1].content
    print(f"  Searching: '{query}'")

    search_results = _run_search(query)
    analysis = _analyze_results(query, search_results)

    print(f"  Confidence: {analysis.confidence_score}/10")
    print(f"  Sufficient: {analysis.is_sufficient}")
    print(f"  Reasoning: {analysis.reasoning}")

    current_attempts = state.get("research_attempts", 0) + 1

    return {
        "research_findings": analysis.findings,
        "confidence_score": analysis.confidence_score,
        "research_attempts": current_attempts,
        "messages": [AIMessage(
            content=f"Research complete. Confidence: {analysis.confidence_score}/10\n\n{analysis.findings}"
        )]
    }

# --- Router ---
def route_confidence(state: AgentState) -> Literal["synthesis", "validator"]:
    score = state.get("confidence_score", 0)
    route = "synthesis" if score >= 6 else "validator"
    print(f"\n[ResearchAgent] Confidence {score}/10 — routing to {route}")
    return route