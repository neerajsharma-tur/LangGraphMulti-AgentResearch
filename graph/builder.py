from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state.agent_state import AgentState
from agents.clarity_agent import clarity_node, ask_user_node, route_clarity
from agents.research_agent import research_node, route_confidence
from agents.validator_agent import validator_node, route_validator
from agents.synthesis_agent import synthesis_node

def build_graph():
    builder = StateGraph(AgentState)

    # Register all nodes
    builder.add_node("clarity",   clarity_node)
    builder.add_node("ask_user",  ask_user_node)
    builder.add_node("research",  research_node)
    builder.add_node("validator", validator_node)
    builder.add_node("synthesis", synthesis_node)

    # Wire all edges
    builder.add_edge(START, "clarity")
    builder.add_conditional_edges("clarity",   route_clarity)
    builder.add_edge("ask_user",  "clarity")
    builder.add_conditional_edges("research",  route_confidence)
    builder.add_conditional_edges("validator", route_validator)
    builder.add_edge("synthesis", END)

    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["ask_user"]
    )

    print("[Graph] Built successfully")
    return graph