import os
import sys

# Add the current directory to sys.path so we can import the tool
sys.path.append(os.getcwd())

from live_packages_tool import fetch_live_packages

def test_fetch_live_packages():
    print("Testing fetch_live_packages tool...")
    
    # Test case 1: Destination only
    print("\nTest Case 1: Destination='Paris'")
    result = fetch_live_packages.invoke({"search_term": "Paris"})
    print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")
    
    # Test case 2: Destination and theme
    print("\nTest Case 2: Destination='London', theme='adventure'")
    result = fetch_live_packages.invoke({"search_term": "London", "theme": "adventure"})
    print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")

    # Test case 3: Destination, travelers, and budget
    print("\nTest Case 3: Destination='Dubai', number_of_people=2, budget=50000")
    result = fetch_live_packages.invoke({"search_term": "Dubai", "number_of_people": 2, "budget": 50000.0})
    print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")

if __name__ == "__main__":
    test_fetch_live_packages()
