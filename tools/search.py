from langchain_tavily import TavilySearch

def get_search_tool(max_results: int = 3, depth: str = "basic") -> TavilySearch:
    return TavilySearch(
        max_results=max_results,
        search_depth=depth,
        include_answer=True,
    )

# Default instance used by research agent
search_tool = get_search_tool()