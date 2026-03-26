"""
FastAPI REST API for LangGraph Research Agent
Exposes endpoints to query the multi-agent research pipeline
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uuid
from graph.builder import build_graph
from langchain_core.messages import HumanMessage

app = FastAPI(
    title="LangGraph Research Agent API",
    description="Multi-agent research pipeline with web search, validation and synthesis",
    version="1.0.0"
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build graph once at startup
graph = build_graph()


class QueryRequest(BaseModel):
    query: str = Field(..., description="Research query to process")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for conversation continuity")
    clarification: Optional[str] = Field(None, description="Optional clarification if needed")


class QueryResponse(BaseModel):
    thread_id: str
    status: str
    message: str
    final_answer: Optional[str] = None
    confidence_score: Optional[int] = None
    research_attempts: Optional[int] = None
    validation_result: Optional[str] = None
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None


class ClarificationRequest(BaseModel):
    thread_id: str = Field(..., description="Thread ID to continue conversation")
    clarification: str = Field(..., description="User's clarification response")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "LangGraph Research Agent API",
        "version": "1.0.0"
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a research query through the multi-agent pipeline
    
    - **query**: The research question to answer
    - **thread_id**: Optional thread ID for conversation continuity (generated if not provided)
    - **clarification**: Optional clarification if continuing a previous query
    
    Returns the research results or a clarifying question if needed
    """
    try:
        # Generate thread_id if not provided
        thread_id = request.thread_id or f"thread-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Initial invoke
        graph.invoke(
            {
                "messages": [HumanMessage(content=request.query)],
                "clarification_count": 0,
                "research_attempts": 0,
            },
            config=config
        )
        
        # Check for clarification interrupt
        snapshot = graph.get_state(config)
        
        if snapshot.next and "ask_user" in snapshot.next:
            # Agent needs clarification
            agent_question = snapshot.values["messages"][-1].content
            
            if request.clarification:
                # User provided clarification — continue processing
                graph.update_state(
                    config,
                    {"messages": [HumanMessage(content=request.clarification)]}
                )
                graph.invoke(None, config=config)
                
                # Get final result
                final_state = graph.get_state(config).values
                return QueryResponse(
                    thread_id=thread_id,
                    status="completed",
                    message="Query processed successfully with clarification",
                    final_answer=final_state["messages"][-1].content,
                    confidence_score=final_state.get("confidence_score"),
                    research_attempts=final_state.get("research_attempts"),
                    validation_result=final_state.get("validation_result"),
                    needs_clarification=False
                )
            else:
                # Return clarifying question to user
                return QueryResponse(
                    thread_id=thread_id,
                    status="pending_clarification",
                    message="Agent needs clarification",
                    needs_clarification=True,
                    clarifying_question=agent_question
                )
        
        # No clarification needed — return final result
        final_state = graph.get_state(config).values
        return QueryResponse(
            thread_id=thread_id,
            status="completed",
            message="Query processed successfully",
            final_answer=final_state["messages"][-1].content,
            confidence_score=final_state.get("confidence_score"),
            research_attempts=final_state.get("research_attempts"),
            validation_result=final_state.get("validation_result"),
            needs_clarification=False
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/clarify", response_model=QueryResponse)
async def provide_clarification(request: ClarificationRequest):
    """
    Provide clarification for a pending query
    
    - **thread_id**: Thread ID from the previous query response
    - **clarification**: User's clarification response
    
    Returns the final research results
    """
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Check if thread exists and is waiting for clarification
        snapshot = graph.get_state(config)
        
        if not snapshot.next or "ask_user" not in snapshot.next:
            raise HTTPException(
                status_code=400, 
                detail="Thread is not waiting for clarification or doesn't exist"
            )
        
        # Update with clarification and continue
        graph.update_state(
            config,
            {"messages": [HumanMessage(content=request.clarification)]}
        )
        graph.invoke(None, config=config)
        
        # Get final result
        final_state = graph.get_state(config).values
        return QueryResponse(
            thread_id=request.thread_id,
            status="completed",
            message="Query completed with clarification",
            final_answer=final_state["messages"][-1].content,
            confidence_score=final_state.get("confidence_score"),
            research_attempts=final_state.get("research_attempts"),
            validation_result=final_state.get("validation_result"),
            needs_clarification=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing clarification: {str(e)}")


@app.get("/thread/{thread_id}")
async def get_thread_state(thread_id: str):
    """
    Get the current state of a conversation thread
    
    - **thread_id**: Thread ID to query
    
    Returns the full conversation history and state
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(config)
        
        if not snapshot.values:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        return {
            "thread_id": thread_id,
            "messages": [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content
                }
                for msg in snapshot.values.get("messages", [])
            ],
            "state": {
                "clarity_status": snapshot.values.get("clarity_status"),
                "confidence_score": snapshot.values.get("confidence_score"),
                "research_attempts": snapshot.values.get("research_attempts"),
                "validation_result": snapshot.values.get("validation_result"),
            },
            "next_steps": list(snapshot.next) if snapshot.next else []
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving thread: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
