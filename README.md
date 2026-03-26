# LangGraph Multi-Agent Research Pipeline

A production-grade multi-agent research pipeline built with LangGraph, LangChain and Groq. The pipeline takes a user query, clarifies it if needed, searches the web, validates the results, and synthesizes a final answer — all driven by a stateful graph of specialized AI agents.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Clarity Agent  │ ◄─── loops back if clarification needed
└────────┬────────┘
         │ clear
         ▼
┌─────────────────┐
│ Research Agent  │ ◄─── retries if confidence < 6
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
confidence   confidence
  >= 6        < 6
    │         │
    │    ┌────▼────────────┐
    │    │ Validator Agent  │
    │    └────────┬─────────┘
    │         sufficient / max attempts
    │             │
    └─────────────┘
         │
         ▼
┌──────────────────┐
│ Synthesis Agent  │
└────────┬─────────┘
         │
         ▼
    Final Answer
```

### Agents

**Clarity Agent** — Analyzes the user query using an LLM with structured output. Determines if the query is specific enough to research. If vague, asks a clarifying question and waits for user input via LangGraph interrupt. Has a retry guard — after 2 clarifications, forces the query through regardless.

**Research Agent** — Runs a real web search via Tavily, formats the results, and passes them to an LLM that scores confidence (1–10) and summarizes findings. If confidence is 6 or above, routes directly to synthesis. Below 6, routes to the validator.

**Validator Agent** — Applies rule checks (attempt count, completeness) and runs an LLM judge that decides whether the findings are sufficient to answer the original query. Routes back to research for a retry or forward to synthesis if max attempts are reached.

**Synthesis Agent** — Takes the original query, full conversation history and research findings, and writes a clean, comprehensive final answer using an LLM.

---

## Project Structure

```
langgraph-research-agent/
├── .env                        # API keys
├── requirements.txt            # Python dependencies
├── main.py                     # CLI entry point
├── api.py                      # FastAPI REST API server
├── graph/
│   ├── __init__.py
│   └── builder.py              # Graph wiring — nodes and edges
├── agents/
│   ├── __init__.py
│   ├── clarity_agent.py        # Clarity node, ask_user node, router
│   ├── research_agent.py       # Research node, confidence router
│   ├── validator_agent.py      # Validator node, validation router
│   └── synthesis_agent.py      # Synthesis node
├── state/
│   ├── __init__.py
│   └── agent_state.py          # Shared AgentState TypedDict
├── tools/
│   ├── __init__.py
│   └── search.py               # Tavily search tool
└── config/
    ├── __init__.py
    └── settings.py             # LLM config, env loading
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/yourname/langgraph-research-agent
cd langgraph-research-agent

conda create -n langgraph-agent python=3.11 -y
conda activate langgraph-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
python -m pip install langgraph langchain langchain-groq tavily-python
python -m pip install python-dotenv pydantic fastapi uvicorn
```

### 3. Set up API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

Get your keys from:
- Groq: https://console.groq.com (free)
- Tavily: https://tavily.com (free — 1000 searches/month)

### 4. Run the pipeline

```bash
python main.py
```

---

## Usage

### Basic query

```python
from graph.builder import build_graph
from langchain_core.messages import HumanMessage

graph = build_graph()
config = {"configurable": {"thread_id": "session-001"}}

result = graph.invoke(
    {
        "messages": [HumanMessage(content="What are the key features of LangGraph?")],
        "clarification_count": 0,
        "research_attempts": 0,
    },
    config=config
)

print(result["messages"][-1].content)
```

### Query with clarification handling

```python
graph.invoke(
    {"messages": [HumanMessage(content="Tell me about agents")], ...},
    config=config
)

snapshot = graph.get_state(config)

if snapshot.next and "ask_user" in snapshot.next:
    # Agent needs clarification — inject user reply
    graph.update_state(
        config,
        {"messages": [HumanMessage(content="I want to know about LangGraph multi-agent patterns")]}
    )
    result = graph.invoke(None, config=config)
```

### Multi-turn conversation

Each `thread_id` maintains its own isolated conversation history via `MemorySaver`. Use different thread IDs for different users or sessions:

```python
config_user_1 = {"configurable": {"thread_id": "user-001"}}
config_user_2 = {"configurable": {"thread_id": "user-002"}}
```

---

## REST API

### Starting the API server

```bash
python api.py
```

Or with uvicorn directly:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

Interactive documentation (Swagger UI): `http://localhost:8000/docs`

### API Endpoints

#### `POST /query`
Process a research query through the pipeline.

**Request body:**
```json
{
  "query": "What are the key features of LangGraph?",
  "thread_id": "optional-session-id",
  "clarification": "optional clarification text"
}
```

**Response (completed):**
```json
{
  "thread_id": "thread-123",
  "status": "completed",
  "message": "Query processed successfully",
  "final_answer": "LangGraph is...",
  "confidence_score": 8,
  "research_attempts": 1,
  "validation_result": "sufficient",
  "needs_clarification": false
}
```

**Response (needs clarification):**
```json
{
  "thread_id": "thread-123",
  "status": "pending_clarification",
  "message": "Agent needs clarification",
  "needs_clarification": true,
  "clarifying_question": "What aspect of agents would you like to know about?"
}
```

#### `POST /clarify`
Provide clarification for a pending query.

**Request body:**
```json
{
  "thread_id": "thread-123",
  "clarification": "I want to know about multi-agent orchestration"
}
```

#### `GET /thread/{thread_id}`
Get the full conversation history and state.

**Response:**
```json
{
  "thread_id": "thread-123",
  "messages": [...],
  "state": {
    "clarity_status": "clear",
    "confidence_score": 8,
    "research_attempts": 1
  },
  "next_steps": []
}
```

### Example API usage with curl

**Simple query:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the benefits of using LangGraph?"}'
```

**Query with clarification:**
```bash
# Initial query
RESPONSE=$(curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about agents"}')

THREAD_ID=$(echo $RESPONSE | jq -r '.thread_id')

# Provide clarification
curl -X POST http://localhost:8000/clarify \
  -H "Content-Type: application/json" \
  -d "{\"thread_id\": \"$THREAD_ID\", \"clarification\": \"I want to know about LangGraph agents\"}"
```

### Example with Python requests

```python
import requests

# Simple query
response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What is LangGraph?"}
)
result = response.json()
print(result["final_answer"])

# Query that needs clarification
response = requests.post(
    "http://localhost:8000/query",
    json={"query": "Tell me about agents"}
)
result = response.json()

if result["needs_clarification"]:
    print(f"Agent asks: {result['clarifying_question']}")
    
    # Provide clarification
    response = requests.post(
        "http://localhost:8000/clarify",
        json={
            "thread_id": result["thread_id"],
            "clarification": "I want to know about LangGraph multi-agent patterns"
        }
    )
    final_result = response.json()
    print(final_result["final_answer"])
```

---

## State schema

All agents share a single `AgentState` TypedDict:

| Field | Type | Owner | Description |
|---|---|---|---|
| `messages` | `list` | All | Full conversation history with `add_messages` reducer |
| `clarity_status` | `str` | Clarity | `"clear"` or `"needs_clarification"` |
| `clarifying_question` | `str` | Clarity | Question asked to user if vague |
| `original_query` | `str` | Clarity | Preserved original query for all downstream agents |
| `clarification_count` | `int` | Clarity | Retry guard — max 2 clarifications |
| `research_findings` | `str` | Research | Summarized web search results |
| `confidence_score` | `int` | Research | 1–10 score driving routing logic |
| `research_attempts` | `int` | Research | Retry guard — max 3 attempts |
| `validation_result` | `str` | Validator | `"sufficient"` or `"insufficient"` |
| `validation_reason` | `str` | Validator | LLM judge reasoning |
| `final_answer` | `str` | Synthesis | The final user-facing response |

---

## Configuration

### Changing the LLM model

Edit `config/settings.py`:

```python
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
fast_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
```

### Switching to OpenAI

```python
# config/settings.py
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

### Changing confidence threshold

Edit `agents/research_agent.py`:

```python
def route_confidence(state: AgentState) -> Literal["synthesis", "validator"]:
    score = state.get("confidence_score", 0)
    return "synthesis" if score >= 7 else "validator"  # raise to 7 for stricter pipeline
```

### Switching to persistent storage

Edit `graph/builder.py`:

```python
# Development — in memory
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

# Production — SQLite (survives restarts)
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string("checkpoints.db")

# Production — PostgreSQL
from langgraph.checkpoint.postgres import PostgresSaver
memory = PostgresSaver.from_conn_string("postgresql://user:pass@host/db")
```

---

## How routing works

```
clarity_node runs
    clarity_status == "needs_clarification" and count < 2
        → INTERRUPT → ask_user → user replies → clarity_node re-runs
    clarity_status == "clear" or count >= 2
        → research_node

research_node runs
    confidence_score >= 6
        → synthesis_node
    confidence_score < 6
        → validator_node

validator_node runs
    validation_result == "insufficient" and attempts < 3
        → research_node (retry)
    validation_result == "sufficient" or attempts >= 3
        → synthesis_node
```

---

## Key concepts used

| Concept | Where used |
|---|---|
| `TypedDict` state | `state/agent_state.py` — shared memory across agents |
| `add_messages` reducer | Appends messages instead of overwriting |
| `with_structured_output` | All agents — reliable typed responses from LLM |
| `MemorySaver` checkpointer | Persists state between invocations |
| `interrupt_before` | Pauses graph at `ask_user` for human input |
| `graph.update_state` | Injects user clarification into frozen state |
| `graph.invoke(None, config)` | Resumes graph from checkpoint |
| Conditional edges | Route between agents based on state values |

---

## Learning path

This project was built incrementally across 9 days:

| Day | Topic | File |
|---|---|---|
| 1 | State, nodes, edges | `day1_basics.py` |
| 2 | Conditional edges, routing | `day2_conditional.py` |
| 3 | MemorySaver, interrupts | `day3_interrupt.py` |
| 4 | Streaming, multi-turn | `day4_streaming.py` |
| 5 | Messages, prompts, structured output | `day5_langchain.py` |
| 6 | Tool calling, ToolNode | `day6_tools.py` |
| 7 | Real Clarity Agent | `day7_clarity_agent.py` |
| 8 | Real Tavily search | `day8_research_agent.py` |
| 9 | Combined pipeline | `day9_combined.py` |

---

## Tech stack

| Tool | Purpose |
|---|---|
| LangGraph 1.1+ | Graph orchestration, state management, interrupts |
| LangChain | LLM abstractions, prompt templates, tool calling |
| Groq + Llama 3.3 70B | Fast, free LLM inference |
| Tavily | Real-time web search API |
| FastAPI | REST API endpoints with automatic OpenAPI documentation |
| Pydantic | Structured output validation |
| Python 3.11 | Runtime |

---

## Roadmap

- [x] FastAPI endpoint — expose pipeline as REST API
- [ ] Streamlit UI — chat interface for the pipeline
- [ ] LangSmith tracing — observe every agent run
- [ ] SqliteSaver — persistent conversation history
- [ ] Query refinement on retry — improve search query between attempts
- [ ] Multiple search tools — fallback from Tavily to DuckDuckGo
- [ ] Async execution — parallel tool calls

---

## License

MIT
