"""
Dynamic Trip Planner Chatbot using LangGraph

A conversational, agentic trip planner with ReAct-style agents and HITL support.
Uses Gemini LLM via Google Generative AI.

Author: AI Assistant
Date: 2026-01-07
"""

import os
import uuid
import json
from typing import Dict, List, Any, Annotated, TypedDict, Literal
from operator import add

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from query_qdrant import qdrant_pdf_search
# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import Qdrant search tool
from query_qdrant import qdrant_pdf_search

# =============================================================================
# Configuration
# =============================================================================

# Get API key from environment
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
)

# =============================================================================
# State Definition
# =============================================================================

class TripPlannerState(TypedDict):
    """State for the trip planner graph."""
    user_input: str
    history: Annotated[List[str], add]  # Accumulating conversation history
    parameters: Dict[str, str]  # destination, dates, budget, activities
    recommendations: List[Dict]
    current_itinerary: Dict
    intent: str  # collect_params, faq, generate_itinerary, edit_itinerary, unintended
    waiting_for_hitl: bool
    session_id: str


# =============================================================================
# Tool Definitions (Placeholder)
# =============================================================================

class RecommendationInput(BaseModel):
    """Input for recommendation tool."""
    category: str = Field(description="Category for recommendations: destination, activity, hotel, restaurant")
    context: str = Field(description="Context for personalized recommendations", default="")


def get_recommendations(category: str, context: str = "") -> List[Dict]:
    """Get recommendations for destinations, activities, etc. (Placeholder)"""
    recommendations = {
        "destination": [
            {"name": "Paris, France", "highlight": "City of Lights, Eiffel Tower, Louvre"},
            {"name": "Tokyo, Japan", "highlight": "Culture, Technology, Amazing Food"},
            {"name": "Bali, Indonesia", "highlight": "Beaches, Temples, Tropical Paradise"},
        ],
        "activity": [
            {"name": "City Walking Tour", "duration": "3 hours", "price": "$30"},
            {"name": "Museum Visit", "duration": "4 hours", "price": "$25"},
            {"name": "Local Food Tour", "duration": "3 hours", "price": "$50"},
        ],
        "hotel": [
            {"name": "Grand Hotel", "rating": "5-star", "price": "$300/night"},
            {"name": "Comfort Inn", "rating": "3-star", "price": "$100/night"},
            {"name": "Boutique Stay", "rating": "4-star", "price": "$180/night"},
        ],
        "restaurant": [
            {"name": "Le Gourmet", "cuisine": "French", "price": "$$$"},
            {"name": "Street Eats", "cuisine": "Local", "price": "$"},
            {"name": "Fusion Kitchen", "cuisine": "International", "price": "$$"},
        ],
    }
    return recommendations.get(category, [{"suggestion": "No recommendations available"}])


class FlightSearchInput(BaseModel):
    """Input for flight search."""
    origin: str = Field(description="Origin city/airport")
    destination: str = Field(description="Destination city/airport")
    date: str = Field(description="Travel date")


def search_flights(origin: str, destination: str, date: str) -> Dict:
    """Search for flights (Placeholder)"""
    return {
        "flights": [
            {
                "airline": "Air Global",
                "flight_no": "AG123",
                "departure": f"{date} 08:00",
                "arrival": f"{date} 14:00",
                "price": "$450",
                "class": "Economy",
            },
            {
                "airline": "Sky Wings",
                "flight_no": "SW456",
                "departure": f"{date} 10:30",
                "arrival": f"{date} 16:30",
                "price": "$520",
                "class": "Economy",
            },
        ],
        "origin": origin,
        "destination": destination,
    }


class HotelSearchInput(BaseModel):
    """Input for hotel search."""
    destination: str = Field(description="Destination city")
    check_in: str = Field(description="Check-in date")
    check_out: str = Field(description="Check-out date")
    budget: str = Field(description="Budget level: budget, moderate, luxury")


def search_hotels(destination: str, check_in: str, check_out: str, budget: str = "moderate") -> Dict:
    """Search for hotels (Placeholder)"""
    hotels_by_budget = {
        "budget": [
            {"name": "Hostel Central", "rating": "3-star", "price": "$50/night", "amenities": ["WiFi", "Breakfast"]},
        ],
        "moderate": [
            {"name": "City Comfort Hotel", "rating": "4-star", "price": "$150/night", "amenities": ["WiFi", "Pool", "Gym"]},
        ],
        "luxury": [
            {"name": "Grand Palace Resort", "rating": "5-star", "price": "$400/night", "amenities": ["WiFi", "Pool", "Spa", "Concierge"]},
        ],
    }
    return {
        "hotels": hotels_by_budget.get(budget, hotels_by_budget["moderate"]),
        "destination": destination,
        "dates": f"{check_in} to {check_out}",
    }


class ActivitySearchInput(BaseModel):
    """Input for activity search."""
    destination: str = Field(description="Destination city")
    interests: str = Field(description="Interests/preferences for activities")


def get_activities(destination: str, interests: str = "") -> Dict:
    """Get activity suggestions (Placeholder)"""
    return {
        "activities": [
            {"name": f"Walking Tour of {destination}", "duration": "3 hours", "price": "$35", "category": "Sightseeing"},
            {"name": "Local Cuisine Experience", "duration": "4 hours", "price": "$60", "category": "Food"},
            {"name": "Historical Museum Visit", "duration": "2 hours", "price": "$20", "category": "Culture"},
            {"name": "Sunset River Cruise", "duration": "2 hours", "price": "$45", "category": "Relaxation"},
        ],
        "destination": destination,
    }


# Create structured tools
recommendation_tool = StructuredTool.from_function(
    func=get_recommendations,
    name="get_recommendations",
    description="Get travel recommendations for destinations, activities, hotels, or restaurants",
    args_schema=RecommendationInput,
)

flight_tool = StructuredTool.from_function(
    func=search_flights,
    name="search_flights",
    description="Search for available flights between two cities",
    args_schema=FlightSearchInput,
)

hotel_tool = StructuredTool.from_function(
    func=search_hotels,
    name="search_hotels",
    description="Search for hotels at a destination",
    args_schema=HotelSearchInput,
)

activity_tool = StructuredTool.from_function(
    func=get_activities,
    name="get_activities",
    description="Get activity suggestions for a destination",
    args_schema=ActivitySearchInput,
)


# =============================================================================
# Agent/Node Functions
# =============================================================================

def get_context_prompt(state: TripPlannerState) -> str:
    """Build context from state for agent prompts."""
    context_parts = []
    
    # Add conversation history
    if state.get("history"):
        recent_history = state["history"][-10:]  # Last 10 exchanges
        context_parts.append("Conversation History:")
        for entry in recent_history:
            context_parts.append(f"  {entry}")
    
    # Add collected parameters
    if state.get("parameters"):
        params = state["parameters"]
        context_parts.append("\nCollected Trip Parameters:")
        for key, value in params.items():
            if value:
                context_parts.append(f"  - {key}: {value}")
    
    # Add current itinerary if exists
    if state.get("current_itinerary"):
        context_parts.append("\nCurrent Itinerary (summary available)")
    
    return "\n".join(context_parts)


def intent_router_node(state: TripPlannerState) -> Dict:
    """
    Intent Extractor/Router Node
    
    Classifies user intent and routes accordingly.
    Intents: collect_params, faq, generate_itinerary, edit_itinerary, unintended
    """
    user_input = state.get("user_input", "")
    context = get_context_prompt(state)
    
    # Check if we have all required parameters
    params = state.get("parameters", {})
    has_all_params = all([
        params.get("destination"),
        params.get("dates"),
        params.get("budget"),
    ])
    
    system_prompt = f"""You are an intent classifier for a trip planning chatbot.

Current Context:
{context}

All required parameters collected: {has_all_params}

Classify the user's intent into ONE of these categories:
1. collect_params - User wants to plan a trip, provide trip details, or answer questions about their preferences
2. faq - User is asking a general travel question (best time to visit, visa requirements, etc.)
3. generate_itinerary - User wants to see/create their trip itinerary (only valid if all params collected)
4. edit_itinerary - User wants to modify an existing itinerary
5. unintended - Completely unrelated to travel planning

Respond with ONLY a JSON object in this format:
{{"intent": "category_name", "reasoning": "brief explanation"}}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User message: {user_input}"),
    ]
    
    response = llm.invoke(messages)
    response_text = response.content.strip()
    
    # Parse intent from response
    try:
        # Handle markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        intent = result.get("intent", "collect_params")
    except (json.JSONDecodeError, IndexError):
        # Default to collect_params if parsing fails
        intent = "collect_params"
    
    # Handle unintended intent directly
    if intent == "unintended":
        return {
            "intent": "unintended",
            "history": [f"User: {user_input}", "Bot: I'm a trip planning assistant! I can help you plan your perfect trip. Would you like to start planning a vacation?"],
            "waiting_for_hitl": True,
        }
    
    # If trying to generate itinerary without all params, redirect to param collection
    if intent == "generate_itinerary" and not has_all_params:
        intent = "collect_params"
    
    return {
        "intent": intent,
        "history": [f"User: {user_input}"],
        "waiting_for_hitl": False,
    }


def exploration_node(state: TripPlannerState) -> Dict:
    """
    Exploration Node - ReAct Agent for Parameter Collection
    
    Collects trip parameters slot-by-slot:
    - Required: destination, dates, budget
    - Optional: activities, preferences
    
    Uses recommendation tool to provide suggestions.
    """
    user_input = state.get("user_input", "")
    context = get_context_prompt(state)
    params = state.get("parameters", {})
    
    # Determine what's missing
    missing_params = []
    if not params.get("destination"):
        missing_params.append("destination")
    if not params.get("dates"):
        missing_params.append("dates (when do you want to travel?)")
    if not params.get("budget"):
        missing_params.append("budget (budget, moderate, or luxury)")
    
    # Check if this is a FAQ
    intent = state.get("intent", "")
    
    system_prompt = f"""You are a friendly travel planning assistant helping collect trip details.

Current Context:
{context}

Missing Information: {', '.join(missing_params) if missing_params else 'All required info collected!'}

Your tasks:
1. If the user mentions a destination, dates, budget, or activities, extract and acknowledge them
2. If there are missing required parameters, ask for the NEXT missing one conversationally
3. If the intent is 'faq', answer their travel question helpfully, then gently redirect to trip planning
4. If all params are collected, ask: "Great! I have all the details. Would you like me to generate your trip itinerary?"
5. Use recommendations when suggesting destinations or activities

Always be warm, helpful, and conversational. Don't be robotic.

Based on the user's message, provide:
1. Any extracted parameters (as JSON: {{"extracted": {{"destination": "...", "dates": "...", etc}}}})
2. Your conversational response

Respond with a JSON object:
{{"extracted": {{}}, "response": "your message to user", "all_params_collected": false}}"""

    tools_info = f"""
Available tools:
- get_recommendations(category, context): Get suggestions for destinations, activities, hotels, restaurants
- qdrant_pdf_search(query, k): Search travel knowledge base, FAQs, and past successful itineraries. Use this for specific travel questions and u can use this as recomendation.
"""
    
    messages = [
        SystemMessage(content=system_prompt + "\n" + tools_info),
        HumanMessage(content=f"User says: {user_input}"),
    ]

    # Check if the user is asking an FAQ that might benefit from PDF search
    if intent == "faq":
        search_results = qdrant_pdf_search.invoke({"query": user_input})
        messages.append(SystemMessage(content=f"Qdrant Search Results for '{user_input}':\n{search_results}"))
    
    response = llm.invoke(messages)
    response_text = response.content.strip()
    
    # Parse response
    try:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        extracted = result.get("extracted", {})
        bot_response = result.get("response", "I'd love to help plan your trip! Where would you like to go?")
        all_collected = result.get("all_params_collected", False)
    except (json.JSONDecodeError, IndexError):
        extracted = {}
        bot_response = "I'd love to help plan your trip! Could you tell me where you'd like to go?"
        all_collected = False
    
    # Update parameters with extracted values
    updated_params = params.copy()
    for key in ["destination", "dates", "budget", "activities"]:
        if extracted.get(key):
            updated_params[key] = extracted[key]
    
    # Check if all required params are now collected
    has_all_params = all([
        updated_params.get("destination"),
        updated_params.get("dates"),
        updated_params.get("budget"),
    ])
    
    return {
        "parameters": updated_params,
        "history": [f"Bot: {bot_response}"],
        "waiting_for_hitl": True,  # Always wait for user response
        "intent": "generate_itinerary" if has_all_params and all_collected else "collect_params",
    }


def trip_planner_node(state: TripPlannerState) -> Dict:
    """
    Trip Planner Node - ReAct Agent for Itinerary Generation
    
    Generates detailed trip itinerary using:
    - Flight search tool
    - Hotel search tool  
    - Activity suggestions tool
    
    Also handles edit requests for existing itineraries.
    """
    user_input = state.get("user_input", "")
    context = get_context_prompt(state)
    params = state.get("parameters", {})
    current_itinerary = state.get("current_itinerary", {})
    intent = state.get("intent", "")
    
    destination = params.get("destination", "Unknown")
    dates = params.get("dates", "TBD")
    budget = params.get("budget", "moderate")
    activities = params.get("activities", "sightseeing")
    
    # Get data from placeholder tools
    flights = search_flights("Your City", destination, dates.split(" to ")[0] if " to " in dates else dates)
    hotels = search_hotels(destination, dates.split(" to ")[0] if " to " in dates else dates, 
                          dates.split(" to ")[1] if " to " in dates else dates, budget)
    activities_data = get_activities(destination, activities)
    
    # Check if this is an edit request
    is_edit = intent == "edit_itinerary" and current_itinerary
    
    if is_edit:
        system_prompt = f"""You are a trip planning assistant helping modify an existing itinerary.

Current Context:
{context}

Current Itinerary:
{json.dumps(current_itinerary, indent=2)}

User's Edit Request: {user_input}

Available updated data:
- Flights: {json.dumps(flights, indent=2)}
- Hotels: {json.dumps(hotels, indent=2)}
- Activities: {json.dumps(activities_data, indent=2)}

Modify the itinerary based on the user's request and provide the updated version.
Be specific about what changed.

Respond with JSON:
{{"response": "your explanation of changes", "updated_itinerary": {{...}}, "needs_confirmation": true}}"""
    else:
        system_prompt = f"""You are a trip planning assistant creating a detailed itinerary.

Trip Parameters:
- Destination: {destination}
- Dates: {dates}
- Budget: {budget}
- Interests: {activities}

Available Data:
- Flights: {json.dumps(flights, indent=2)}
- Hotels: {json.dumps(hotels, indent=2)}  
- Activities: {json.dumps(activities_data, indent=2)}

Create a comprehensive day-by-day itinerary. Be specific and helpful.

IMPORTANT: Your "response" field MUST contain the FULL FORMATTED ITINERARY with all details:
- Flight information (airline, times, price)
- Hotel details (name, rating, price, amenities)
- Day-by-day schedule with activities, times, and prices
- Total estimated cost

Format it nicely with sections and bullet points for readability.

Respond with JSON:
{{"response": "YOUR COMPLETE FORMATTED ITINERARY HERE - include ALL flight, hotel, and daily schedule details", "itinerary": {{"destination": "...", "dates": "...", "flights": [...], "hotel": {{...}}, "daily_schedule": [...]}}, "needs_confirmation": true}}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User: {user_input}"),
    ]
    
    response = llm.invoke(messages)
    response_text = response.content.strip()
    
    # Parse response
    try:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        bot_response = result.get("response", "Here's your trip itinerary!")
        itinerary = result.get("itinerary", result.get("updated_itinerary", {}))
        needs_confirmation = result.get("needs_confirmation", True)
    except (json.JSONDecodeError, IndexError):
        # Fallback: format itinerary manually
        itinerary = {
            "destination": destination,
            "dates": dates,
            "flights": flights.get("flights", [])[:1],
            "hotel": hotels.get("hotels", [{}])[0],
            "activities": activities_data.get("activities", []),
        }
        
        # Format a nice response with all details
        flight_info = flights.get("flights", [{}])[0]
        hotel_info = hotels.get("hotels", [{}])[0]
        activities_list = activities_data.get("activities", [])
        
        bot_response = f"""Here's your trip itinerary for {destination}!

📅 TRAVEL DATES: {dates}
💰 BUDGET: {budget}

✈️ FLIGHT:
  • {flight_info.get('airline', 'Air Global')} - Flight {flight_info.get('flight_no', 'TBD')}
  • Departure: {flight_info.get('departure', 'TBD')}
  • Arrival: {flight_info.get('arrival', 'TBD')}
  • Price: {flight_info.get('price', 'TBD')}

🏨 HOTEL:
  • {hotel_info.get('name', 'TBD')}
  • Rating: {hotel_info.get('rating', 'TBD')}
  • Price: {hotel_info.get('price', 'TBD')}
  • Amenities: {', '.join(hotel_info.get('amenities', ['WiFi']))}

🎯 RECOMMENDED ACTIVITIES:
"""
        for i, act in enumerate(activities_list, 1):
            bot_response += f"  {i}. {act.get('name', 'Activity')} - {act.get('duration', 'TBD')} - {act.get('price', 'TBD')}\n"
        
        bot_response += "\nWould you like to make any changes to this itinerary?"
        needs_confirmation = True
    
    return {
        "current_itinerary": itinerary,
        "history": [f"Bot: {bot_response}"],
        "waiting_for_hitl": needs_confirmation,
        "intent": "edit_itinerary",  # Ready for potential edits
    }


# =============================================================================
# Graph Setup
# =============================================================================

def route_by_intent(state: TripPlannerState) -> Literal["exploration", "trip_planner", END]:
    """Route based on detected intent."""
    intent = state.get("intent", "collect_params")
    waiting = state.get("waiting_for_hitl", False)
    
    # If waiting for user input, go to END
    if waiting:
        return END
    
    # Route based on intent
    if intent in ["collect_params", "faq"]:
        return "exploration"
    elif intent in ["generate_itinerary", "edit_itinerary"]:
        return "trip_planner"
    else:
        return END  # unintended or unknown


def should_continue_exploration(state: TripPlannerState) -> Literal["trip_planner", END]:
    """Decide whether to continue to trip planner or wait for user."""
    intent = state.get("intent", "")
    waiting = state.get("waiting_for_hitl", True)
    
    if waiting:
        return END
    
    if intent == "generate_itinerary":
        return "trip_planner"
    
    return END


def should_continue_planning(state: TripPlannerState) -> Literal[END]:
    """Trip planner always ends (waits for user feedback)."""
    return END


def create_graph() -> StateGraph:
    """Create and compile the trip planner graph."""
    # Initialize graph with state
    graph = StateGraph(TripPlannerState)
    
    # Add nodes
    graph.add_node("router", intent_router_node)
    graph.add_node("exploration", exploration_node)
    graph.add_node("trip_planner", trip_planner_node)
    
    # Set entry point
    graph.set_entry_point("router")
    
    # Add conditional edges from router
    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "exploration": "exploration",
            "trip_planner": "trip_planner",
            END: END,
        }
    )
    
    # Add conditional edges from exploration
    graph.add_conditional_edges(
        "exploration",
        should_continue_exploration,
        {
            "trip_planner": "trip_planner",
            END: END,
        }
    )
    
    # Add edge from trip_planner
    graph.add_conditional_edges(
        "trip_planner",
        should_continue_planning,
        {
            END: END,
        }
    )
    
    # Compile with memory checkpointer
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)
    
    return compiled_graph


# =============================================================================
# CLI Main Loop
# =============================================================================

def get_initial_state() -> Dict:
    """Create initial state for a new session."""
    return {
        "user_input": "",
        "history": [],
        "parameters": {},
        "recommendations": [],
        "current_itinerary": {},
        "intent": "",
        "waiting_for_hitl": False,
        "session_id": "",
    }


def main():
    """Main CLI loop for the trip planner chatbot."""
    print("=" * 60)
    print("🌍 Welcome to the Dynamic Trip Planner! 🌍")
    print("=" * 60)
    print("I can help you plan your perfect trip.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("Type 'reset' to start a new planning session.")
    print("-" * 60)
    
    # Create graph
    graph = create_graph()
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    # Track state between invocations
    current_state = get_initial_state()
    current_state["session_id"] = session_id
    
    print("\nHow can I help you plan your trip today?\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Handle special commands
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("\n✈️ Safe travels! Goodbye! ✈️")
                break
            if user_input.lower() == "reset":
                session_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": session_id}}
                current_state = get_initial_state()
                current_state["session_id"] = session_id
                print("\n🔄 Session reset. Let's start fresh!\n")
                print("How can I help you plan your trip today?\n")
                continue
            
            # Prepare input state
            input_state = {
                "user_input": user_input,
                "session_id": session_id,
                "parameters": current_state.get("parameters", {}),
                "current_itinerary": current_state.get("current_itinerary", {}),
                "waiting_for_hitl": False,  # Reset HITL flag for new input
            }
            
            # Invoke graph
            result = graph.invoke(input_state, config)
            
            # Update tracked state
            if result:
                current_state.update({
                    "parameters": result.get("parameters", current_state.get("parameters", {})),
                    "current_itinerary": result.get("current_itinerary", current_state.get("current_itinerary", {})),
                    "intent": result.get("intent", ""),
                })
            
            # Print bot response (last item in history from this turn)
            if result and result.get("history"):
                # Get the most recent bot response
                history = result["history"]
                for entry in reversed(history):
                    if entry.startswith("Bot:"):
                        print(f"\n{entry}\n")
                        break
        
        except KeyboardInterrupt:
            print("\n\n✈️ Safe travels! Goodbye! ✈️")
            break
        except Exception as e:
            print(f"\n⚠️ An error occurred: {e}")
            print("Let's continue. What would you like to do?\n")


if __name__ == "__main__":
    main()
