import json
import requests
from typing import Optional
from langchain.tools import tool
from pydantic import BaseModel, Field

class LivePackagesRequest(BaseModel):
    search_term: str = Field(description="Search term or destination (e.g. 'Paris')")
    theme: Optional[str] = Field(default=None, description="Trip theme (e.g. 'romantic')")
    number_of_people: Optional[int] = Field(
        default=None, description="Number of travelers (e.g. 4)"
    )
    days: Optional[int] = Field(default=None, description="Trip length in days (e.g. 5)")
    budget: Optional[float] = Field(default=None, description="Maximum budget (e.g. 1500.0)")
    departureCity: Optional[str] = Field(default=None, description="Departure city (e.g. 'Mumbai')")
    monthOfTravel: Optional[str] = Field(default=None, description="Month of travel (e.g. '2024-12')")

class TravelPackage(BaseModel):
    title: Optional[str] = None
    days: Optional[str] = None
    price: Optional[str] = None
    tumbnail: Optional[str] = None
    prompt: Optional[str] = None
    packageId: Optional[str] = None

    def to_dict(self) -> dict[str, Optional[str]]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()

@tool("fetch_live_packages")
def fetch_live_packages(
    search_term: str,
    theme: Optional[str] = None,
    number_of_people: Optional[int] = None,
    days: Optional[int] = None,
    budget: Optional[float] = None,
    departureCity: Optional[str] = None,
    monthOfTravel: Optional[str] = None
) -> str:
    """
    Fetch live travel packages based on destination, theme, budget, and other parameters.
    Use this tool when the user asks for available packages, curated trips, or specific travel deals.
    """
    url = "http://35.244.47.248:8000/v1/livepackages"
    
    payload = {
        "search_term": search_term,
        "theme": theme,
        "number_of_people": number_of_people,
        "days": days,
        "budget": budget,
        "departureCity": departureCity,
        "monthOfTravel": monthOfTravel
    }
    
    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        body = data.get("body", [])
        if not body:
            return "No packages found for the given criteria."
            
        packages = []
        for item in body:
            if isinstance(item, dict):
                itinerary = item.get("itinerary_data", {})
                # Extract fields using the TravelPackage model with correct mappings
                pkg = TravelPackage(
                    title=itinerary.get("packageName"),
                    days=str(itinerary.get("days")) if itinerary.get("days") else None,
                    price=str(itinerary.get("price")) if itinerary.get("price") else None,
                    tumbnail=itinerary.get("thumbnailImage"),
                    prompt=itinerary.get("packageSummary"),
                    packageId=itinerary.get("packageId")
                )
                packages.append(pkg.to_dict())
        
        # Limit to first 4 results
        if len(packages) > 4:
            packages = packages[:4]
            
        return json.dumps(packages) if packages else "No packages found for the given criteria."
    except Exception as e:
        return f"Error fetching live packages: {str(e)}"
