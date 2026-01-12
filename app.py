"""
FastAPI Application for Trip Planner Chat API

Endpoints:
- POST /chat: Main chat endpoint with conversation_id and user_message
- GET /health: Health check

Session Management:
- In-memory store for TripPlannerState by conversation_id
- State persists across requests for same conversation_id
"""

import os
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph import get_graph
from stategraph import TripPlannerState
from logger_config import setup_logger

# Setup logger
logger = setup_logger(__name__)


# =============================================================================
# FastAPI App Setup
# =============================================================================

app = FastAPI(
    title="Trip Planner Chat API",
    description="LangGraph-powered chat API with intent routing for trip planning",
    version="1.0.0",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Session Store (In-Memory)
# =============================================================================

# In-memory store: conversation_id -> TripPlannerState
_session_store: Dict[str, TripPlannerState] = {}


def get_or_create_state(conversation_id: str) -> TripPlannerState:
    """Get existing state or create new one for conversation_id."""
    if conversation_id not in _session_store:
        _session_store[conversation_id] = {
            "conversation_id": conversation_id,
            "core": [{}],
            "errors": [],
        }
    return _session_store[conversation_id]


def update_state(conversation_id: str, new_state: TripPlannerState) -> None:
    """Update stored state for conversation_id."""
    _session_store[conversation_id] = new_state


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    conversation_id: str = Field(
        ...,
        description="Unique identifier for the conversation",
        example="conv-12345"
    )
    user_message: str = Field(
        ...,
        description="The user's message",
        example="I want to plan a trip to Paris"
    )


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    conversation_id: str
    assistant_response: str
    intent: str
    errors: List[str] = []


class HealthResponse(BaseModel):
    """Response body for health check."""
    status: str
    service: str
    timestamp: str


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint.
    
    Flow:
    1. Retrieve or create state for conversation_id
    2. Set user_message in state
    3. Invoke LangGraph
    4. Store updated state
    5. Return assistant_response
    """
    try:
        logger.info(f"Received chat request for conversation_id: {request.conversation_id}")
        logger.debug(f"User message: {request.user_message}")

        # Get or create state
        state = get_or_create_state(request.conversation_id)
        
        # Update state with user message
        state["user_message"] = request.user_message
        
        # Get graph and config
        graph = get_graph()
        config = {"configurable": {"thread_id": request.conversation_id}}
        
        # Invoke graph
        result = graph.invoke(state, config)
        
        # Update stored state
        update_state(request.conversation_id, result)
        
        # Return response
        response = ChatResponse(
            conversation_id=request.conversation_id,
            assistant_response=result.get("assistant_response", ""),
            intent=result.get("node_to_run", ""),
            errors=result.get("errors", []),
        )
        
        logger.info(f"Response sent for conversation_id: {request.conversation_id}. Intent: {result.get('node_to_run', '')}")
        return response
    
    except Exception as e:
        logger.error(f"Error in chat_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="trip-planner-chat-api",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/session/{conversation_id}")
async def get_session(conversation_id: str):
    """
    Debug endpoint: Get current state for a conversation.
    
    Useful for debugging and understanding state progression.
    """
    if conversation_id not in _session_store:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    state = _session_store[conversation_id]
    
    # Return sanitized state (exclude internal fields)
    return {
        "conversation_id": state.get("conversation_id"),
        "current_phase": str(state.get("current_phase", "")),
        "core": state.get("core", []),
        "draft": state.get("draft", {}),
        "last_intent": state.get("node_to_run", ""),
        "errors": state.get("errors", []),
    }


@app.delete("/session/{conversation_id}")
async def clear_session(conversation_id: str):
    """Clear/reset a conversation session."""
    if conversation_id in _session_store:
        del _session_store[conversation_id]
        return {"message": f"Session {conversation_id} cleared"}
    return {"message": f"Session {conversation_id} not found (already cleared)"}


@app.get("/itinerary/{conversation_id}")
async def get_itinerary(conversation_id: str):
    """
    Get the final itinerary for a conversation.
    """
    if conversation_id not in _session_store:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    state = _session_store[conversation_id]
    final_itinerary = state.get("final_itinerary")
    
    if not final_itinerary:
       raise HTTPException(status_code=404, detail="Final itinerary not found for this conversation")
       
    return final_itinerary


# =============================================================================
# Run with: uvicorn app:app --reload --port 8000
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8006, reload=True)
