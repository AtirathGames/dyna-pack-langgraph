
import os
import requests
from graph import trip_planner_node
from stategraph import TripPlannerState, Phase

def test_tripplanner_api():
    # Construct a sample state
    state = {
        "conversation_id": "test-123",
        "user_message": "Plan a trip to Paris for 5 days in March",
        "core": {
            "destination": ["Paris"],
            "origin": "London",
            "start_date": "2026-03-10",
            "end_date": "2026-03-15",
            "budget_total": 50000,
            "travelers": 2,
            "type_of_trip": "dynamic"
        },
        "flights_state": {"conversation": [], "searches": {}},
        "hotels_state": {"conversation": [], "searches": {}},
        "attractions_state": {"conversation": [], "searches": {}},
        "transfers_state": {"conversation": [], "searches": {}},
        "draft": {},
        "final_itinerary": {},
        "assistant_response": "",
        "assistant_message": "",
        "should_end_conversation": False,
        "ended_at": None,
        "trip_planner_history": [],
        "trip_planner_queue": [],
        "trip_planner_waiting_for_order": False,
        "trip_planner_waiting_for_continue": False,
        "trip_planner_active_agent": None,
        "trip_planner_last_active_agent": None,
        "trip_planner_collected_responses": {},
        "errors": []
    }

    print("Testing trip_planner_node calling external API...")
    try:
        updated_state = trip_planner_node(state)
        
        print("\nAPI Call Successful!")
        print("-" * 30)
        print(f"Assistant Response: {updated_state.get('assistant_response')}")
        
        if updated_state.get('errors'):
            print(f"Errors: {updated_state['errors']}")
        
        # Check if some expected fields are returned
        if "final_itinerary" in updated_state:
            print("Received final_itinerary from API!")
        
        print("-" * 30)
        # print("Full Updated State Keys:", updated_state.keys())
        
    except Exception as e:
        print(f"\nTest Failed with exception: {e}")

if __name__ == "__main__":
    test_tripplanner_api()
