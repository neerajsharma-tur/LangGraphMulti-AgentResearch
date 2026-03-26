# test_tavily.py
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

tavily_tool = TavilySearch(max_results=2)

raw = tavily_tool.invoke({"query": "What is LangGraph?"})

print(f"Type: {type(raw)}")
print(f"Value: {raw}")