from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from state.agent_state import AgentState
from config.settings import llm

# --- Prompt ---
_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful research assistant writing a final answer.

Given the user's original query and research findings, write a clear,
well-structured, user-friendly response.

Guidelines:
- Answer the question directly and completely
- Use clear language — avoid jargon unless necessary
- Structure with paragraphs or bullet points where helpful
- Cite key facts from the research
- Be concise but comprehensive"""),
    ("human", """Original query: {query}

Research findings:
{findings}

Write a comprehensive final answer.""")
])

_chain = _prompt | llm

# --- Node ---
def synthesis_node(state: AgentState) -> AgentState:
    print("\n[SynthesisAgent] Writing final answer...")

    query = state.get("original_query", "")
    findings = state.get("research_findings", "No research findings available")

    try:
        response = _chain.invoke({
            "query": query,
            "findings": findings
        })
        final_answer = response.content
        print(f"  Answer length: {len(final_answer)} chars")

    except Exception as e:
        print(f"  Synthesis failed: {e}")
        final_answer = f"I found some information about '{query}' but encountered an error generating the final response."

    return {
        "final_answer": final_answer,
        "messages": [AIMessage(content=final_answer)]
    }