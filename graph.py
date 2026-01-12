"""
LangGraph Definition for Trip Planner Chat API

Graph Structure:
- Entry: intent_router (classifies intent from user_message)
- Conditional Edge: routes to ONE of three handler nodes
- Handlers: exploration, trip_planner, pre_curated_package
- Each handler updates state and ends

Uses TripPlannerState from stategraph.py for full conversation state.
"""

import os
import json
from typing import Dict, Any, Literal, List, Optional
from datetime import datetime
import requests

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from pydantic import BaseModel, Field
from stategraph import TripPlannerState, Phase, DraftItinerary
from query_qdrant import qdrant_pdf_search
from logger_config import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Lazy LLM initialization - only create when needed
_llm = None


def get_llm():
    """Get or create LLM instance. Raises error if API key not set."""
    global _llm
    if _llm is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.2,
        )
    return _llm


# =============================================================================
# Structured Output Models
# =============================================================================

class IntentOutput(BaseModel):
    """Pydantic model for intent classification output."""
    intent: Literal["exploration", "trip_planner", "pre_curated_package"] = Field(..., description="The detected user intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the intent")
    summary: str = Field(..., description="A brief summary of what the user wants")
    exploration_complete: bool = Field(..., description="Whether the exploration phase is complete")
    resume_intent: Literal["exploration", "trip_planner", "pre_curated_package"] = Field(..., description="The intent to resume from")
    reasoning_tags: List[Literal["itinerary_request", "package_request", "faq", "parameter_edit", "recommendation", "hook_choice_dynamic", "hook_choice_precurated", "unclear"]] = Field(..., description="Tags explaining the reasoning")


class ExtractedParams(BaseModel):
    """Extracted parameters from user message."""
    destination: Optional[List[str]] = Field(..., description="Trip destinations")
    origin: Optional[str] = Field(..., description="Trip origin city")
    start_date: Optional[str] = Field(..., description="Trip start date")
    end_date: Optional[str] = Field(..., description="Trip end date")
    budget_total: Optional[int] = Field(..., description="Total budget in INR")
    travelers: Optional[int] = Field(..., description="Number of travelers")
    type_of_trip: Optional[Literal["dynamic", "curated"]] = Field(..., description="Chosen trip type")


class ExplorationOutput(BaseModel):
    """Pydantic model for exploration node output."""
    response: str = Field(..., description="The helpful message for the user")
    extracted_params: ExtractedParams = Field(..., description="Parameters extracted from the conversation")
    draft: Optional[Dict[str, Any]] = Field(None, description="The updated draft itinerary")


# =============================================================================
# Intent Types
# =============================================================================

INTENT_EXPLORATION = "exploration"
INTENT_TRIP_PLANNER = "trip_planner"
INTENT_PRE_CURATED = "pre_curated_package"

VALID_INTENTS = [INTENT_EXPLORATION, INTENT_TRIP_PLANNER, INTENT_PRE_CURATED]


# =============================================================================
# Helper Functions
# =============================================================================

def _build_conversation_context(state: TripPlannerState) -> str:
    """Build context string from state for LLM prompts."""
    context_parts = []
    
    # Core requirements
    core = state.get("core", [])
    if core:
        context_parts.append("Trip Requirements:")
        for req in core:
            if req.get("destination"):
                context_parts.append(f"  - Destination: {req.get('destination')}")
            if req.get("origin"):
                context_parts.append(f"  - Origin: {req.get('origin')}")
            if req.get("start_date"):
                context_parts.append(f"  - Start Date: {req.get('start_date')}")
            if req.get("end_date"):
                context_parts.append(f"  - End Date: {req.get('end_date')}")
            if req.get("budget_total"):
                context_parts.append(f"  - Budget: {req.get('budget_total')}")
            if req.get("travelers"):
                context_parts.append(f"  - Travelers: {req.get('travelers')}")
    
    # Current phase
    phase = state.get("current_phase")
    if phase:
        context_parts.append(f"\nCurrent Phase: {phase}")
    
    # Draft itinerary
    draft = state.get("draft")
    if draft and draft.get("summary"):
        context_parts.append(f"\nDraft Summary: {draft.get('summary')}")
    
    # Conversation History (Last 5 turns for context)
    history = state.get("conversation_history", [])
    if history:
        context_parts.append("\nReference Conversation History:")
        # Get last 10 messages to keep context window manageable
        recent_history = history[-10:]
        for msg in recent_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            context_parts.append(f"{role.upper()}: {content}")
            
    return "\n".join(context_parts) if context_parts else "No prior context."


# =============================================================================
# Intent Router Node (Entry Point)
# =============================================================================

def intent_router_node(state: TripPlannerState) -> Dict[str, Any]:
    """
    Intent Router Node (Entry Point)
    
    Analyzes user_message + conversation context to:
    1. Detect intent (exploration, trip_planner, pre_curated_package)
    2. Set node_to_run for conditional routing
    
    Returns updated state with node_to_run set.
    """
    user_message = state.get("user_message", "")
    context = _build_conversation_context(state)
    errors = list(state.get("errors", []))
    core = state.get("core", [{}])
    
    classification_prompt = f"""You are an intent classifier + intelligent router for a travel planning chatbot.

CONVERSATION CONTEXT (includes dialogue + any stored state/notes if present):
{context}

CURRENT USER MESSAGE:
{user_message}
core: {core}

YOUR JOB:
Route the user to the correct node based on:
- What the user is asking NOW or answer for the agent question
- What has ALREADY happened in the conversation (history)
- Whether the **exploration agenda is complete** (core parameters captured) — this is NOT captured at intent-level, but you MUST infer it from context/history.

IMPORTANT ARCHITECTURE NOTE:
- Core parameter collection happens ONLY inside the **exploration node**, not in this classifier output.
- However, the classifier MUST infer from the history whether the exploration node likely finished collecting core parameters, so it can route intelligently to:
  - exploration (keep collecting / refining / recommending/answer FAQ)
  - trip_planner (dynamic itinerary creation / modification)
  - pre_curated_package (show curated packages)

CORE FLOW:
1) Default start → exploration
2) exploration continues until core agenda is complete (inferred from history)
3) Once exploration agenda is complete, the system asks a hook question:
   "Do you want a dynamic itinerary or a pre-curated package?"
   - If user chooses dynamic itinerary → trip_planner
   - If user chooses pre-curated → pre_curated_package
4) If user asks an FAQ or side question anytime → exploration node.

NODES (INTENTS):
A) "exploration"
Route here when:
- Conversation is starting / early stage OR
- Core agenda likely NOT complete OR
- User is still refining/confirming/editing trip inputs (even if some parameters exist) OR
- User asks for recommendations to decide destinations/dates/budget/style, etc.
Exploration node responsibility: capture missing core parameters + run recommendation tool + confirm/modify parameters.
- User want to change any core parameters 

B) "trip_planner"
Route here when:
- User explicitly wants a dynamic itinerary / day-wise plan / schedule OR
- User explicitly chooses “dynamic itinerary” in the hook question OR
- User wants to modify an existing itinerary (add/remove days, reorder places, add activities, change pace)
- User is clearly past exploration and wants the full plan now.
- for trip_planner,core should be filled 

C) "pre_curated_package"
Route here when:
- User explicitly asks for packages, curated trips, ready-made itineraries, “best packages”, “show options” OR
- User explicitly chooses “pre-curated package” in the hook question.
-for pre_curated_package,core should be filled 


SUB-INTENTS (must be handled without breaking the core flow):
You MUST classify these correctly:
- Parameter edits: “Change dates”, “Increase budget”, “We are 6 people now” → usually exploration (unless already planning itinerary and edits are itinerary-specific → trip_planner)
- Recommendation-only: “Suggest places in Thailand for 5 days” → exploration
- Itinerary modification: “Add 1 day in Kyoto”, “Swap day 2 and 3” → trip_planner
- Package browsing: “Give me 3 Goa packages under 30k” → pre_curated_package
- Confirming hook choice: “Dynamic”, “Make it a package”, “I want curated” → trip_planner or pre_curated_package accordingly

HOW TO INFER WHETHER EXPLORATION AGENDA IS COMPLETE:
-core is filled or changes made reached the confirmmed parameters of core


OUTPUT REQUIREMENTS:
Respond with ONLY a JSON object (no markdown). Use EXACTLY this schema:

{{
  "intent": "exploration" | "trip_planner" | "pre_curated_package",
  "confidence": 0.0-1.0,
  "summary": "what the user wants right now",
  "exploration_complete": true | false,
  "resume_intent": "exploration" | "trip_planner" | "pre_curated_package",
  "reasoning_tags": ["itinerary_request" | "package_request" | "faq" | "parameter_edit" | "recommendation" | "hook_choice_dynamic" | "hook_choice_precurated" | "unclear"]
}}

"""


    messages = [
        SystemMessage(content=classification_prompt),
        HumanMessage(content=user_message),
    ]
    
    try:
        # Use structured output for robust parsing
        structured_llm = get_llm().with_structured_output(IntentOutput)
        result = structured_llm.invoke(messages)
        
        intent = result.intent
        if intent not in VALID_INTENTS:
            intent = INTENT_EXPLORATION
        
        context_summary = result.summary
        
        logger.info(f"Intent classified: {intent} (confidence: {result.confidence})")
        logger.debug(f"Reasoning: {result.reasoning_tags}")
        
    except Exception as e:
        logger.error(f"Intent classification error: {str(e)}", exc_info=True)
        errors.append(f"Intent classification error: {str(e)}")
        intent = INTENT_EXPLORATION
        context_summary = "Defaulting to exploration due to error."
    
    return {
        "node_to_run": intent,
        "current_context_from_intent_node_": context_summary,
        "errors": errors,
    }


# =============================================================================
# Handler Nodes
# =============================================================================

def exploration_node(state: TripPlannerState) -> Dict[str, Any]:
    """
    Exploration Node
    
    Handles parameter collection, FAQs, destination exploration.
    Uses Qdrant PDF search for knowledge retrieval.
    """
    user_message = state.get("user_message", "")
    context = _build_conversation_context(state)
    errors = list(state.get("errors", []))
    core = state.get("core", [{}])
    draft = state.get("draft", {})
    
    # Search Qdrant for relevant info
    try:
        search_results = qdrant_pdf_search.invoke({"query": user_message, "k": 3})
    except Exception as e:
        search_results = f"Search unavailable: {str(e)}"
    
    # Serialize draft to JSON string for prompt, ensuring valid JSON format (double quotes)
    draft_json = json.dumps(draft)
    
    system_prompt = f"""You are a friendly Travel Exploration Assistant. Your job is to help the user explore trip ideas *and* quickly converge on a confirmed plan with a solid draft itinerary.

CONTEXT:
{context}

CURRENT CORE:
{core}

KNOWLEDGE BASE RESULTS:
{search_results}

GOALS
1) Collect and confirm core trip parameters (as efficiently as possible):
   - origin
   - destination(s)
   - date window (start_date, end_date or duration)
   - total budget
   - number of travelers
   - preferences (pace, interests, must-dos, food, accessibility, etc.) when relevant
2) Answer travel FAQs using the knowledge base (and then return to parameter collection/confirmation).
3) After *every* meaningful new parameter (especially destination/date/budget/travelers), immediately:
   - Generate/refresh a **draft itinerary** that best fits the CURRENT CORE using the knowledge base.
   - Explain the itinerary in the response so the user can confirm or tweak quickly.
   - Use the itinerary to “bundle” questions (avoid one-question-per-turn).
4) type_of_trip must be decided LAST, only after the user confirms the key core parameters and is happy with the draft:
   - Ask the final “hook” question: whether they want a **dynamic** trip (flexible, exploring options) or a **curated** trip (locked-in plan with reservations/step-by-step schedule).

DRAFT ITINERARY SCHEMA (use this structure for the "draft" field):
{{
  "summary": "Brief overview of the trip concept",
  "days": [
    {{
      "day_number": 1,
      "morning": "Activity idea",
      "afternoon": "Activity idea",
      "evening": "Activity idea",
      "hotel_idea": "Type of stay (e.g. City Center Boutique)",
      "notes": "Any logic notes"
    }}
  ],
  "estimated_budget": 0,
  "hotel_type": "Brief preference summary",
  "flight_type": "Brief preference summary"
}}

CONVERSATION STYLE RULES
- Be proactive and reduce back-and-forth: ask at most 2–4 targeted questions per turn, grouped.
- Use the draft itinerary to confirm details (“If this looks right, I’ll lock X and Y into the core.”).
- If the user dislikes the draft, offer 2–3 alternative directions (e.g., budget vs comfort, relaxed vs packed, city-first vs nature-first) and update the itinerary accordingly.
- Always use KNOWLEDGE BASE RESULTS for facts/recommendations. If KB is thin, make reasonable assumptions but mark them as assumptions and ask the user to confirm.

WHEN THE TASK IS “DONE”
- The task is completed only when:
  (a) core parameters are collected and confirmed by the user, AND
  (b) a draft itinerary exists that the user approves, AND
  (c) you ask the final hook question about type_of_trip (dynamic vs curated).

Respond with JSON:
{{"response": "your helpful message", "extracted_params": {{"destination": [...], "origin": null, "start_date": null, "end_date": null, "budget_total": null, "travelers": null,"type_of_trip": null}}, "draft": {draft_json}}}

Always include all extracted_params fields. Use null for unmentioned parameters.
If updating the draft, provide the FULL updated draft object, not just diffs."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User: {user_message}"),
    ]
    
    try:
        # Use structured output for robust parsing
        structured_llm = get_llm().with_structured_output(ExplorationOutput)
        result = structured_llm.invoke(messages)
        
        assistant_response = result.response
        extracted = result.extracted_params.model_dump(exclude_none=True)
        draft = result.draft if result.draft else {}
        
        # Update core requirements with extracted params
        if core:
            updated_core = core[0].copy()
        else:
            updated_core = {}
        
        for key in ["destination", "origin", "start_date", "end_date", "budget_total", "travelers", "type_of_trip"]:
            if key in extracted:
                updated_core[key] = extracted[key]
        
        core = [updated_core] if updated_core else core

        logger.info("Exploration node finished.")
        logger.debug(f"Extracted params: {extracted}")
        
    except Exception as e:
        logger.error(f"Exploration error: {str(e)}", exc_info=True)
        errors.append(f"Exploration error: {str(e)}")
        assistant_response = "I'd be happy to help you plan your trip! Could you tell me where you'd like to go?"
    
    # Update conversation history
    new_history = state.get("conversation_history", []) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response}
    ]
    
    return {
        "assistant_response": assistant_response,
        "core": core,
        "current_phase": Phase.GATHERING,
        "errors": errors,
        "conversation_history": new_history,
        "draft": draft
    }


def trip_planner_node(state: TripPlannerState) -> Dict[str, Any]:
    """
    Trip Planner Node
    
    Generates itineraries, searches for flights/hotels/activities.
    Uses collected parameters to create comprehensive plans.
    """
    user_message = state.get("user_message", "")
    context = _build_conversation_context(state)
    intent_context = state.get("current_context_from_intent_node_", "")
    errors = list(state.get("errors", []))
    core = state.get("core", [{}])
    draft = state.get("draft", {})
    
    
    # Ensure all core parameters are present, defaulting to None
    core_data = core[0] if core and isinstance(core, list) else {}
    hydrated_core = {
        "destination": core_data.get("destination"),
        "origin": core_data.get("origin"),
        "start_date": core_data.get("start_date"),
        "end_date": core_data.get("end_date"),
        "budget_total": core_data.get("budget_total"),
        "travelers": core_data.get("travelers"),
        "type_of_trip": core_data.get("type_of_trip")
    }
    
    # Prepare payload for API
    payload = {
        "core": hydrated_core,
        "draft": draft,
        "user_message": user_message
    }
    
    try:
        # Call external API
        logger.info(f"Calling Trip Planner API at http://35.200.247.133:8002/chat for conversation {state.get('conversation_id', 'unknown')} payload :{payload}")
        api_response = requests.post("http://35.200.247.133:8002/chat", json=payload)
        api_response.raise_for_status()
        result = api_response.json()
        
        logger.info(f"Trip Planner API successful. the response is : {result}")
        
        # Extract fields as requested
        # Note: API returns 'final_itenary' (sic)
        final_itinerary = result.get("final_itenary", {})
        assistant_response = result.get("assistant_response", "Here is your trip plan!")
        
    except Exception as e:
        logger.error(f"Trip planner API error: {str(e)}", exc_info=True)
        errors.append(f"Trip planner API error: {str(e)}")
        assistant_response = "I'm having trouble connecting to the planner service. Please try again."
        final_itinerary = {}
    
    # Update conversation history
    new_history = state.get("conversation_history", []) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response}
    ]

    return {
        "assistant_response": assistant_response,
        "final_itinerary": final_itinerary,
        "current_phase": Phase.DRAFTING,
        "errors": errors,
        "conversation_history": new_history
    }


def pre_curated_package_node(state: TripPlannerState) -> Dict[str, Any]:
    """
    Pre-Curated Package Node
    
    Provides curated travel packages and quick recommendations.
    """
    user_message = state.get("user_message", "")
    context = _build_conversation_context(state)
    intent_context = state.get("current_context_from_intent_node_", "")
    errors = list(state.get("errors", []))
    
    # Prepare payload
    core_list = state.get("core", [])
    core_data = core_list[0] if core_list and isinstance(core_list, list) else {}
    
    hydrated_core = {
        "destination": core_data.get("destination"),
        "origin": core_data.get("origin"),
        "start_date": core_data.get("start_date"),
        "end_date": core_data.get("end_date"),
        "budget_total": core_data.get("budget_total"),
        "travelers": core_data.get("travelers"),
        "type_of_trip": core_data.get("type_of_trip")
    }
    
    payload = {
        "core": hydrated_core,
        "user_message": user_message
    }
    
    try:
        # Call external API
        logger.info(f"Calling Package API at http://localhost:8001/v1/package-response payload :{payload}")
        api_response = requests.post("http://localhost:8001/v1/package-response", json=payload)
        api_response.raise_for_status()
        result = api_response.json()
        
        logger.info(f"Package API successful. the response is : {result}")

        assistant_response = result.get("assistant_response", "Here are some packages for you.")
        packages = result.get("packages")
        
    except Exception as e:
        logger.error(f"Package API error: {str(e)}", exc_info=True)
        errors.append(f"Package API error: {str(e)}")
        assistant_response = "I'm having trouble fetching packages right now. Please try again later."
        packages = None
    
    # Update conversation history
    new_history = state.get("conversation_history", []) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response}
    ]
    
    # Build state update
    update_dict = {
        "assistant_response": assistant_response,
        "errors": errors,
        "conversation_history": new_history
    }
    
    # Only update packages if present in response (or if we want to clear it on error, but usually we keep old if simplified. 
    # User said "if in response add it to state", implying if not present, don't touch or don't add. 
    # I'll include it if it's not None).
    if packages is not None:
        update_dict["packages"] = packages
        
    return update_dict


# =============================================================================
# Routing Function
# =============================================================================

def route_by_intent(state: TripPlannerState) -> Literal["exploration", "trip_planner", "pre_curated_package"]:
    """Route to appropriate handler based on node_to_run."""
    node = state.get("node_to_run", INTENT_EXPLORATION)
    
    if node == INTENT_TRIP_PLANNER:
        return "trip_planner"
    elif node == INTENT_PRE_CURATED:
        return "pre_curated_package"
    else:
        return "exploration"


# =============================================================================
# Graph Construction
# =============================================================================

def create_trip_chat_graph() -> StateGraph:
    """
    Create and compile the trip planner chat graph.
    
    Structure:
    - Entry: intent_router
    - Conditional Edge: routes to ONE of three handlers
    - Each handler: updates state and ends
    """
    graph = StateGraph(TripPlannerState)
    
    # Add nodes
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("exploration", exploration_node)
    graph.add_node("trip_planner", trip_planner_node)
    graph.add_node("pre_curated_package", pre_curated_package_node)
    
    # Set entry point
    graph.set_entry_point("intent_router")
    
    # Conditional edges from router
    graph.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "exploration": "exploration",
            "trip_planner": "trip_planner",
            "pre_curated_package": "pre_curated_package",
        }
    )
    
    # Each handler ends
    graph.add_edge("exploration", END)
    graph.add_edge("trip_planner", END)
    graph.add_edge("pre_curated_package", END)
    
    # Compile with memory checkpointer for persistence
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)
    
    return compiled_graph


# Global graph instance
_graph = None


def get_graph():
    """Get or create the graph singleton."""
    global _graph
    if _graph is None:
        _graph = create_trip_chat_graph()
    return _graph


# =============================================================================
# CLI Test Mode
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Trip Chat Graph - Test Mode")
    print("=" * 60)
    print("Type 'quit' to exit, 'clear' to reset")
    print("-" * 60)
    
    graph = get_graph()
    session_id = "test-session"
    config = {"configurable": {"thread_id": session_id}}
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("\nGoodbye!")
                break
            if user_input.lower() == "clear":
                session_id = f"test-{datetime.now().isoformat()}"
                config = {"configurable": {"thread_id": session_id}}
                print("\n[Session cleared]")
                continue
            
            result = graph.invoke(
                {"user_message": user_input},
                config
            )
            
            print(f"\n[Intent: {result.get('node_to_run', 'unknown')}]")
            print(f"\nAssistant: {result.get('assistant_response', 'No response')}")
            
            if result.get("errors"):
                print(f"\n[Errors: {result['errors']}]")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
