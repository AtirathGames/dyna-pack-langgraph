"""
Chat Conversation API with Intent Extractor/Router using LangGraph

Flow:
1. Request lands at intent_extractor_node (entry point)
2. Conditional edge routes to ONE of three handler nodes based on intent
3. Handler node generates response using its agent + system prompt
4. Graph ends, response returned via API

Key state fields:
- user_message: Current user input
- conversation_history: List of {role, content} messages
- assistant_response: Final response to return
- errors: List[str] - any errors passed to handlers
- current_context_from_intent_node: Context data for handler agents
"""

import os
import json
from typing import Dict, List, Any, TypedDict, Literal, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import state types from your existing stategraph
from stategraph import TripPlannerState


# =============================================================================
# Configuration
# =============================================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
)


# =============================================================================
# Chat State Definition
# =============================================================================

class ConversationMessage(TypedDict):
    """A single message in the conversation."""
    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str]


class IntentContext(TypedDict, total=False):
    """Context extracted by intent node for handler agents."""
    detected_intent: str
    confidence: float
    summary: str

class ChatState(TypedDict, total=False):
    """State for the chat conversation graph."""
    # Core conversation
    user_message: str
    conversation_history: List[ConversationMessage]
    assistant_response: str

    # Routing
    intent: str  

    # Context from intent node for handler agents
    current_context_from_intent_node: IntentContext

    # Error handling
    errors: List[str]

    # Session tracking
    session_id: str

    # Optional: Reference to full trip planner state if needed
    trip_state: Optional[TripPlannerState]


# =============================================================================
# Intent Types
# =============================================================================

INTENT_EXPLORATION = "exploration"
INTENT_TRIP_PLANNER = "trip_planner"
INTENT_PRE_CURATED_PACKAGE = "pre_curated_package"

VALID_INTENTS = [INTENT_EXPLORATION, INTENT_TRIP_PLANNER, INTENT_PRE_CURATED_PACKAGE]


# =============================================================================
# System Prompts for Each Handler
# =============================================================================

EXPLORATION_AGENT_SYSTEM_PROMPT = """You are a specialized Exploration Assistant for a travel planning chatbot.

Your role is to help users with:
- Searching for flights
- Comparing flight options
- Understanding flight details (layovers, duration, prices)
- Making flight selections
- Answering flight-related questions

Guidelines:
1. Be helpful and conversational
2. Ask clarifying questions if needed (origin, destination, dates, passengers, cabin class)
3. Provide clear, organized flight information
4. Highlight important details like layovers, total travel time, and price
5. If you don't have flight data, acknowledge it and ask what the user needs

Context from the conversation will be provided to help you understand the user's needs.
Always respond in a helpful, professional manner focused on flights."""


TRIP_PLANNER_AGENT_SYSTEM_PROMPT = """You are a specialized Trip Planner Assistant for a travel planning chatbot.

Your role is to help users with:
- Searching for hotels and accommodations
- Comparing hotel options (price, rating, amenities)
- Understanding hotel details (location, facilities, room types)
- Making hotel selections
- Answering hotel-related questions

Guidelines:
1. Be helpful and conversational
2. Ask clarifying questions if needed (destination, dates, budget, preferences)
3. Provide clear, organized hotel information
4. Highlight important details like star rating, price per night, and key amenities
5. If you don't have hotel data, acknowledge it and ask what the user needs

Context from the conversation will be provided to help you understand the user's needs.
Always respond in a helpful, professional manner focused on accommodations."""


PRE_CURATED_PACKAGE_AGENT_SYSTEM_PROMPT = """You are a Pre-Curated Package Travel Assistant for a travel planning chatbot.

Your role is to help users with:
- General travel questions and advice
- Trip planning discussions
- Destination recommendations
- Travel tips and best practices
- Answering FAQs about travel
- Handling greetings and casual conversation
- Routing users to specific assistants when they have flight or hotel needs

Guidelines:
1. Be warm, friendly, and conversational
2. Provide helpful travel advice and recommendations
3. If the user clearly needs flight or hotel help, acknowledge it and provide general guidance
4. Answer travel-related questions comprehensively
5. For off-topic queries, gently redirect to travel planning

Context from the conversation will be provided to help you understand the user's needs.
Always respond in a helpful, professional manner."""


# =============================================================================
# Intent Extractor/Router Node
# =============================================================================

def intent_extractor_node(state: ChatState) -> Dict[str, Any]:
    """
    Intent Extractor/Router Node (Entry Point)

    Analyzes user message and conversation history to:
    1. Detect intent (flights, hotels, general)
    2. Extract relevant context for the handler agent
    3. Populate current_context_from_intent_node

    Returns updated state with intent and context.
    """
    user_message = state.get("user_message", "")
    conversation_history = state.get("conversation_history", [])
    existing_errors = state.get("errors", [])

    # Build conversation context for the LLM
    history_text = ""
    if conversation_history:
        recent_history = conversation_history[-6:]  # Last 6 messages
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_text += f"{role.upper()}: {content}\n"

    # Intent classification prompt
    classification_prompt = f"""You are an intent classifier for a travel planning chatbot.

CONVERSATION HISTORY:
{history_text if history_text else "No prior conversation."}

CURRENT USER MESSAGE:
{user_message}

TASK:
Classify the user's intent into ONE of these categories:

1. "exploration" - User wants to:
   

2. "trip_planner" - User wants to:
   - Search for hotels/accommodations
   - Ask about hotel options, prices, amenities
   - Compare hotels
   - Select or book a hotel
   - Ask hotel-related questions

3. "pre_curated_package" - User wants to:
   - General travel advice or tips
   - Destination recommendations
   - Greeting or casual conversation
   - FAQ questions about travel
   - Anything that doesn't clearly fit flights or hotels

RESPOND WITH ONLY A JSON OBJECT (no markdown, no explanation):
{{
    "intent": "exploration" | "trip_planner" | "pre_curated_package",
    "confidence": 0.0-1.0,
    "summary": "brief summary of what the user wants (1-2 sentences)",
    "reasoning": "brief explanation of why this intent was chosen"
}}"""

    messages = [
        SystemMessage(content=classification_prompt),
        HumanMessage(content=user_message),
    ]

    errors = list(existing_errors)  # Copy existing errors

    try:
        response = llm.invoke(messages)
        response_text = response.content.strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)

        intent = result.get("intent", INTENT_GENERAL)
        if intent not in VALID_INTENTS:
            intent = INTENT_GENERAL

        confidence = float(result.get("confidence", 0.7))
        summary = result.get("summary", "")

    except json.JSONDecodeError as e:
        errors.append(f"Intent parsing error: {str(e)}")
        intent = INTENT_GENERAL
        confidence = 0.5
        entities = {}
        summary = "Could not parse user intent, defaulting to general assistance."

    except Exception as e:
        errors.append(f"Intent extraction error: {str(e)}")
        intent = INTENT_GENERAL
        confidence = 0.5
        entities = {}
        summary = "Error during intent extraction, defaulting to general assistance."

    # Build context for handler agent
    relevant_history = []
    if conversation_history:
        for msg in conversation_history[-4:]:  # Last 4 messages
            relevant_history.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")

    context_from_intent: IntentContext = {
        "detected_intent": intent,
        "confidence": confidence,
        "summary": summary,
        "relevant_history": relevant_history,
        "additional_context": {
            "has_prior_conversation": len(conversation_history) > 0,
            "message_count": len(conversation_history),
            "timestamp": datetime.now().isoformat(),
        }
    }

    return {
        "intent": intent,
        "current_context_from_intent_node": context_from_intent,
        "errors": errors,
    }


# =============================================================================
# Handler Nodes (Each with Agent + System Prompt)
# =============================================================================

def _build_handler_messages(
    state: ChatState,
    system_prompt: str
) -> List:
    """Build messages for a handler agent using context from intent node."""
    user_message = state.get("user_message", "")
    context = state.get("current_context_from_intent_node", {})
    errors = state.get("errors", [])

    # Build context section for the agent
    context_section = f"""
--- CONTEXT FROM INTENT ANALYSIS ---
Intent: {context.get('detected_intent', 'unknown')}
Confidence: {context.get('confidence', 0)}
Summary: {context.get('summary', 'No summary available')}

Extracted Entities:
{json.dumps(context.get('entities', {}), indent=2)}

Recent Conversation:
{chr(10).join(context.get('relevant_history', ['No prior conversation']))}

Additional Context:
{json.dumps(context.get('additional_context', {}), indent=2)}
--- END CONTEXT ---
"""

    # Add errors if any
    if errors:
        context_section += f"\n--- ERRORS (handle gracefully) ---\n"
        for err in errors:
            context_section += f"- {err}\n"
        context_section += "--- END ERRORS ---\n"

    # Combine system prompt with context
    full_system_prompt = f"{system_prompt}\n\n{context_section}"

    return [
        SystemMessage(content=full_system_prompt),
        HumanMessage(content=user_message),
    ]


def _invoke_handler(state: ChatState, system_prompt: str) -> Dict[str, Any]:
    """Generic handler invocation logic."""
    user_message = state.get("user_message", "")
    conversation_history = state.get("conversation_history", [])
    errors = list(state.get("errors", []))

    messages = _build_handler_messages(state, system_prompt)

    try:
        response = llm.invoke(messages)
        assistant_response = response.content.strip()

    except Exception as e:
        errors.append(f"Handler error: {str(e)}")
        assistant_response = "I apologize, but I encountered an error processing your request. Could you please try again?"

    # Update conversation history
    updated_history = list(conversation_history)
    updated_history.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now().isoformat(),
    })
    updated_history.append({
        "role": "assistant",
        "content": assistant_response,
        "timestamp": datetime.now().isoformat(),
    })

    return {
        "assistant_response": assistant_response,
        "conversation_history": updated_history,
        "errors": errors,
    }


def flights_handler_node(state: ChatState) -> Dict[str, Any]:
    """
    Flights Handler Node

    Specialized agent for flight-related queries.
    Uses FLIGHTS_AGENT_SYSTEM_PROMPT and context from intent node.
    """
    return _invoke_handler(state, FLIGHTS_AGENT_SYSTEM_PROMPT)


def hotels_handler_node(state: ChatState) -> Dict[str, Any]:
    """
    Hotels Handler Node

    Specialized agent for hotel-related queries.
    Uses HOTELS_AGENT_SYSTEM_PROMPT and context from intent node.
    """
    return _invoke_handler(state, HOTELS_AGENT_SYSTEM_PROMPT)


def general_handler_node(state: ChatState) -> Dict[str, Any]:
    """
    General Handler Node

    Handles general travel queries, greetings, and fallback.
    Uses GENERAL_AGENT_SYSTEM_PROMPT and context from intent node.
    """
    return _invoke_handler(state, GENERAL_AGENT_SYSTEM_PROMPT)


# =============================================================================
# Conditional Routing Function
# =============================================================================

def route_by_intent(state: ChatState) -> Literal["flights_handler", "hotels_handler", "general_handler"]:
    """
    Conditional routing function for the intent extractor.

    Routes to exactly ONE handler based on detected intent.
    """
    intent = state.get("intent", INTENT_GENERAL)

    if intent == INTENT_FLIGHTS:
        return "flights_handler"
    elif intent == INTENT_HOTELS:
        return "hotels_handler"
    else:
        return "general_handler"


# =============================================================================
# Graph Construction
# =============================================================================

def create_chat_graph() -> StateGraph:
    """
    Create and compile the chat conversation graph.

    Structure:
    - Entry: intent_extractor_node
    - Conditional Edge: routes to ONE of three handlers
    - Each handler: generates response and ends
    """
    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node("intent_extractor", intent_extractor_node)
    graph.add_node("flights_handler", flights_handler_node)
    graph.add_node("hotels_handler", hotels_handler_node)
    graph.add_node("general_handler", general_handler_node)

    # Set entry point
    graph.set_entry_point("intent_extractor")

    # Add conditional edges from intent_extractor to handlers
    graph.add_conditional_edges(
        "intent_extractor",
        route_by_intent,
        {
            "flights_handler": "flights_handler",
            "hotels_handler": "hotels_handler",
            "general_handler": "general_handler",
        }
    )

    # Each handler goes to END
    graph.add_edge("flights_handler", END)
    graph.add_edge("hotels_handler", END)
    graph.add_edge("general_handler", END)

    # Compile the graph
    compiled_graph = graph.compile()

    return compiled_graph


# =============================================================================
# API Endpoint Function
# =============================================================================

# Global graph instance (create once, reuse)
_chat_graph = None


def get_chat_graph():
    """Get or create the chat graph singleton."""
    global _chat_graph
    if _chat_graph is None:
        _chat_graph = create_chat_graph()
    return _chat_graph


def chat_api_handler(
    user_message: str,
    conversation_history: List[ConversationMessage] = None,
    session_id: str = None,
    trip_state: TripPlannerState = None,
) -> Dict[str, Any]:
    """
    API endpoint handler for chat conversations.

    Args:
        user_message: The user's input message
        conversation_history: List of prior conversation messages
        session_id: Optional session identifier
        trip_state: Optional reference to full trip planner state

    Returns:
        Dict containing:
        - assistant_response: The generated response
        - conversation_history: Updated conversation history
        - intent: Detected intent
        - context: Context extracted by intent node
        - errors: Any errors that occurred
    """
    graph = get_chat_graph()

    # Prepare input state
    input_state: ChatState = {
        "user_message": user_message,
        "conversation_history": conversation_history or [],
        "assistant_response": "",
        "intent": "",
        "current_context_from_intent_node": {},
        "errors": [],
        "session_id": session_id or "",
        "trip_state": trip_state,
    }

    # Invoke the graph
    result = graph.invoke(input_state)

    # Return API response
    return {
        "assistant_response": result.get("assistant_response", ""),
        "conversation_history": result.get("conversation_history", []),
        "intent": result.get("intent", ""),
        "context": result.get("current_context_from_intent_node", {}),
        "errors": result.get("errors", []),
    }


# =============================================================================
# FastAPI Integration Example
# =============================================================================

def create_fastapi_app():
    """
    Create a FastAPI app with the chat endpoint.

    Usage:
        from chat_graph import create_fastapi_app
        app = create_fastapi_app()
        # Run with: uvicorn chat_graph:app --reload
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="Chat Conversation API",
        description="Intent-based routing chat API with specialized agents",
        version="1.0.0",
    )

    class ChatRequest(BaseModel):
        user_message: str = Field(..., description="The user's input message")
        conversation_history: List[Dict] = Field(
            default=[],
            description="Prior conversation messages"
        )
        session_id: str = Field(default="", description="Optional session ID")

    class ChatResponse(BaseModel):
        assistant_response: str
        conversation_history: List[Dict]
        intent: str
        context: Dict
        errors: List[str]

    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest):
        """
        Chat conversation endpoint.

        Receives user message, routes through intent extractor,
        processes with appropriate handler, returns response.
        """
        try:
            result = chat_api_handler(
                user_message=request.user_message,
                conversation_history=request.conversation_history,
                session_id=request.session_id,
            )
            return ChatResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "chat-graph"}

    return app


# Create app instance for uvicorn
app = create_fastapi_app()


# =============================================================================
# CLI Test Mode
# =============================================================================

def main():
    """CLI test mode for the chat graph."""
    print("=" * 60)
    print("Chat Graph - Intent Router Test Mode")
    print("=" * 60)
    print("Type 'quit' to exit, 'clear' to reset history")
    print("-" * 60)

    conversation_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("\nGoodbye!")
                break
            if user_input.lower() == "clear":
                conversation_history = []
                print("\n[History cleared]")
                continue

            result = chat_api_handler(
                user_message=user_input,
                conversation_history=conversation_history,
            )

            conversation_history = result["conversation_history"]

            print(f"\n[Intent: {result['intent']}]")
            print(f"\nAssistant: {result['assistant_response']}")

            if result["errors"]:
                print(f"\n[Errors: {result['errors']}]")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
