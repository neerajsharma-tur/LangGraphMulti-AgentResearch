from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# --- 1. Basic invoke ---
print("=" * 50)
print("1. BASIC INVOKE")
print("=" * 50)

response = llm.invoke([
    SystemMessage(content="You are a helpful assistant. Be concise."),
    HumanMessage(content="What is a LangGraph node in one sentence?")
])

print(f"Content: {response.content}")
print(f"Model: {response.response_metadata['model_name']}")
print(f"Input tokens: {response.usage_metadata['input_tokens']}")
print(f"Output tokens: {response.usage_metadata['output_tokens']}")
print(f"Total tokens: {response.usage_metadata['total_tokens']}")


# --- 2. Prompt Templates ---
print("\n" + "=" * 50)
print("2. PROMPT TEMPLATES")
print("=" * 50)

# ChatPromptTemplate lets you reuse prompts with variables
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Your job is to analyze queries."),
    ("human", "Analyze this query and tell me if it is clear or vague: {query}")
])

# Format the prompt with actual values
formatted = prompt.invoke({"query": "Tell me stuff"})
print(f"Formatted messages: {formatted.messages}")

# Chain prompt -> llm using pipe operator
chain = prompt | llm
response = chain.invoke({"query": "What are the benefits of LangGraph for building agents?"})
print(f"\nResponse: {response.content}")



# --- 3. Structured Output ---
print("\n" + "=" * 50)
print("3. STRUCTURED OUTPUT")
print("=" * 50)

# Define the exact shape you want back
class ClarityResult(BaseModel):
    status: str = Field(description="Either 'clear' or 'needs_clarification'")
    reason: str = Field(description="Why you made this decision")
    clarifying_question: str = Field(description="Question to ask user if needs_clarification, else empty string")

# Bind the structure to the LLM
structured_llm = llm.with_structured_output(ClarityResult)

clarity_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query clarity analyzer.
     Determine if a user query is clear enough to research.
     A query is clear if it has a specific topic and intent.
     A query needs clarification if it is vague, too short, or ambiguous."""),
    ("human", "Analyze this query: {query}")
])

clarity_chain = clarity_prompt | structured_llm

# Test with vague query
print("\nTest 1 — vague query:")
result = clarity_chain.invoke({"query": "Tell me stuff"})
print(f"  Status: {result.status}")
print(f"  Reason: {result.reason}")
print(f"  Question: {result.clarifying_question}")

# Test with clear query
print("\nTest 2 — clear query:")
result = clarity_chain.invoke({"query": "What are the main use cases of LangGraph in production?"})
print(f"  Status: {result.status}")
print(f"  Reason: {result.reason}")
print(f"  Question: {result.clarifying_question}")



# --- 4. Conversation history ---
print("\n" + "=" * 50)
print("4. CONVERSATION HISTORY PATTERN")
print("=" * 50)

# This is exactly how your Synthesis Agent will work
history = [
    SystemMessage(content="You are a helpful research assistant."),
    HumanMessage(content="What is LangGraph?"),
    AIMessage(content="LangGraph is a framework for building stateful multi-agent applications."),
    HumanMessage(content="Give me one real world use case.")
]

response = llm.invoke(history)
print(f"Response with history: {response.content}")
