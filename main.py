from graph.builder import build_graph
from langchain_core.messages import HumanMessage

graph = build_graph()

def run(thread_id: str, query: str, clarification: str = None):
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 60)
    print(f"Query: '{query}'")
    print("=" * 60)

    # Initial invoke
    graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "clarification_count": 0,
            "research_attempts": 0,
        },
        config=config
    )

    # Handle clarification interrupt
    snapshot = graph.get_state(config)
    if snapshot.next and "ask_user" in snapshot.next:
        agent_question = snapshot.values["messages"][-1].content
        print(f"\nAgent asks: {agent_question}")

        if clarification:
            print(f"User replies: '{clarification}'")
            graph.update_state(
                config,
                {"messages": [HumanMessage(content=clarification)]}
            )
            graph.invoke(None, config=config)
        else:
            print("Waiting for clarification...")
            return

    # Final result
    final = graph.get_state(config).values
    print(f"\n{'=' * 60}")
    print("FINAL ANSWER:")
    print('=' * 60)
    print(final["messages"][-1].content)
    print(f"\nConfidence: {final.get('confidence_score', 'N/A')}/10")
    print(f"Attempts:   {final.get('research_attempts', 'N/A')}")
    print(f"Validation: {final.get('validation_result', 'N/A')}")

if __name__ == "__main__":
    # Test 1 — clear query
    run(
        thread_id="prod-001",
        query="Give me information about Google Alphabet Company?"
    )

    # Test 2 — needs clarification
    run(
        thread_id="prod-002",
        query="Tell me about agents",
        clarification="I want to know about multi-agent orchestration with LangGraph"
    )