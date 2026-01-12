
import requests
import json

url = "http://localhost:8001/v1/package-response"
payload = {
    "core": {"destination": ["Rome"], "duration": 5},
    "user_message": "Show me packages for Rome"
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload, timeout=10)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\nResponse Keys:", list(data.keys()))
        
        if "packages" in data:
            packages = data["packages"]
            print(f"\nPackages found: {len(packages) if packages else 0}")
            if packages:
                print(f"Sample Package: {json.dumps(packages[0], indent=2)}")
        else:
            print("\n'packages' key NOT found (Optional field).")
            
        if "assistant_response" in data:
            print(f"\nAssistant Response: {data['assistant_response']}")
        else:
            print("\nWARNING: 'assistant_response' key NOT found.")
            
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Exception: {e}")
