from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- Tavily tool — real web search ---
tavily_tool = TavilySearch(
    max_results=3,          # top 3 results per search
    search_depth="basic",   # "basic" or "advanced"
    include_answer=True,    # get a direct answer if available
)

tools = [tavily_tool]
llm_with_tools = llm.bind_tools(tools)

print("Tavily tool schema:")
print(f"  Name: {tavily_tool.name}")
print(f"  Description: {tavily_tool.description[:80]}...")

# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    clarity_status: str
    original_query: str
    clarification_count: int
    research_findings: str      # NEW — stores research summary
    confidence_score: int       # NEW — 1 to 10
    research_attempts: int      # NEW — retry counter for validator

# --- Research output schema ---
class ResearchOutput(BaseModel):
    findings: str = Field(
        description="A clear structured summary of what was found"
    )
    confidence_score: int = Field(
        description="Confidence score 1-10. 10 = very comprehensive answer found, 1 = nothing useful found",
        ge=1, le=10
    )
    is_sufficient: bool = Field(
        description="True if findings are sufficient to answer the query, False if more research needed"
    )
    reasoning: str = Field(
        description="Why you assigned this confidence score"
    )



    # --- Research prompt ---
research_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research analyst. You have just searched the web for information.

Given the search results, your job is to:
1. Summarize the key findings clearly and concisely
2. Assign a confidence score (1-10) based on how well the results answer the query
3. Decide if the findings are sufficient

Confidence scoring guide:
- 8-10: Comprehensive, direct answer found with good sources
- 6-7:  Good information found but some gaps
- 4-5:  Partial information, missing key details
- 1-3:  Very little useful information found

Be honest about confidence — do not inflate scores."""),
    ("human", """Original query: {query}

Search results:
{search_results}

Analyze these results and provide your assessment.""")
])

research_structured_llm = llm.with_structured_output(ResearchOutput)
research_chain = research_prompt | research_structured_llm

# --- Research Node ---
def research_node(state: AgentState) -> AgentState:
    print("\n--- research_node running ---")

    query = state.get("original_query") or state["messages"][-1].content
    print(f"  Searching for: '{query}'")

    # Step 1 — run Tavily search
    try:
        raw = tavily_tool.invoke({"query": query})

        # New langchain-tavily returns a dict with 'results' key
        search_results = raw.get("results", [])
        print(f"  Got {len(search_results)} results from Tavily")

        # Format results for LLM
        formatted_results = ""
        for i, result in enumerate(search_results, 1):
            formatted_results += f"\nResult {i}:\n"
            formatted_results += f"  Title: {result.get('title', 'N/A')}\n"
            formatted_results += f"  URL: {result.get('url', 'N/A')}\n"
            formatted_results += f"  Content: {result.get('content', 'N/A')[:300]}\n"
            formatted_results += f"  Score: {result.get('score', 'N/A')}\n"

        # Use direct answer if Tavily provides one
        if raw.get("answer"):
            formatted_results = f"Direct answer: {raw['answer']}\n\n" + formatted_results

    except Exception as e:
        print(f"  Tavily search failed: {e}")
        formatted_results = "Search failed — no results available"

    # Step 2 — LLM analyzes and scores the results
    try:
        analysis = research_chain.invoke({
            "query": query,
            "search_results": formatted_results
        })

        print(f"  Confidence score: {analysis.confidence_score}/10")
        print(f"  Is sufficient: {analysis.is_sufficient}")
        print(f"  Reasoning: {analysis.reasoning}")

    except Exception as e:
        print(f"  Analysis failed: {e}")
        analysis = ResearchOutput(
            findings="Analysis failed",
            confidence_score=1,
            is_sufficient=False,
            reasoning=str(e)
        )

    # Step 3 — update state
    current_attempts = state.get("research_attempts", 0) + 1

    return {
        "research_findings": analysis.findings,
        "confidence_score": analysis.confidence_score,
        "research_attempts": current_attempts,
        "messages": [AIMessage(
            content=f"Research complete. Confidence: {analysis.confidence_score}/10\n\n{analysis.findings}"
        )]
    }
# --- Confidence router ---
def route_confidence(state: AgentState) -> Literal["synthesis", "validator"]:
    confidence = state.get("confidence_score", 0)
    if confidence >= 6:
        print(f"\n  Confidence {confidence} >= 6 — routing to synthesis")
        return "synthesis"
    else:
        print(f"\n  Confidence {confidence} < 6 — routing to validator")
        return "validator"

# --- Mock Validator and Synthesis (real ones come Day 10-12) ---
def validator_node(state: AgentState) -> AgentState:
    attempts = state.get("research_attempts", 0)
    print(f"\n--- validator_node: attempt {attempts} of 3 ---")
    return {}

def route_validator(state: AgentState) -> Literal["research", "synthesis"]:
    attempts = state.get("research_attempts", 0)
    if attempts < 3:
        print(f"  Retrying research (attempt {attempts})")
        return "research"
    else:
        print(f"  Max attempts reached — forcing synthesis")
        return "synthesis"

def synthesis_node(state: AgentState) -> AgentState:
    print("\n--- synthesis_node: generating final answer ---")
    findings = state.get("research_findings", "No findings available")
    query = state.get("original_query", "Unknown query")
    reply = AIMessage(content=f"Based on research about '{query}':\n\n{findings}")
    return {"messages": [reply]}


# --- Build Graph ---
memory = MemorySaver()
builder = StateGraph(AgentState)

builder.add_node("research", research_node)
builder.add_node("validator", validator_node)
builder.add_node("synthesis", synthesis_node)

builder.add_edge(START, "research")
builder.add_conditional_edges("research", route_confidence)
builder.add_conditional_edges("validator", route_validator)
builder.add_edge("synthesis", END)

graph = builder.compile(checkpointer=memory)

# --- Test helper ---
def run_research(thread_id: str, query: str):
    config = {"configurable": {"thread_id": thread_id}}
    print("\n" + "=" * 50)
    print(f"Query: '{query}'")
    print("=" * 50)

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "clarification_count": 0,
            "research_attempts": 0
        },
        config=config
    )

    print(f"\nFinal answer:\n{result['messages'][-1].content}")
    print(f"\nConfidence score: {result['confidence_score']}/10")
    print(f"Research attempts: {result['research_attempts']}")

# Test 1 — well known topic, should get high confidence
run_research(
    thread_id="research-001",
    query="What is LangGraph and how is it used for building AI agents?"
)

# Test 2 — specific technical question
run_research(
    thread_id="research-002",
    query="What are the main differences between LangChain and LangGraph?"
)