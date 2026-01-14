from live_packages_tool import fetch_live_packages
import json

def test_tool():
    print("Testing fetch_live_packages tool...")
    
    # Test case 1: Basic search
    for term in ["Rome", "Thailand", "Paris"]:
        print(f"\n--- Test Case: {term} ---")
        result = fetch_live_packages.invoke({"search_term": term})
        print(f"Result (first 500 chars): {result[:500]}...")
        
        try:
            if result.startswith("["):
                import ast
                parsed_result = ast.literal_eval(result)
                if isinstance(parsed_result, list):
                    print(f"Number of packages returned: {len(parsed_result)}")
                    if len(parsed_result) > 0:
                        print(f"First package fields: {list(parsed_result[0].keys())}")
                        print(f"Sample: {json.dumps(parsed_result[0], indent=2)}")
                else:
                    print("Result is not a list.")
            else:
                print(f"Tool returned message: {result}")
        except Exception as e:
            print(f"Error parsing result: {e}")


if __name__ == "__main__":
    test_tool()
