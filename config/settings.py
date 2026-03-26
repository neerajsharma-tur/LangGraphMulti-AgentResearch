from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Central LLM config — change model here once for all agents
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
fast_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)  # for simple tasks    