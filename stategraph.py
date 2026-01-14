# product/state_demo.py

# NOTE:
try: 
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
except Exception:  # pragma: no cover
    StateGraph = None  # type: ignore[assignment]
    END = None  # type: ignore[assignment]
    MemorySaver = None  # type: ignore[assignment]
from typing import Any, TypedDict, List, Dict, Optional, Literal, Required
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
import json


# ============================================================
# UTILITY: FlexibleDateTime parser
# ============================================================
class FlexibleDateTime:
    """
    Utility to parse datetime from various formats (ISO, timestamps, etc.)
    Used by ItineraryPlan validators.
    """

    @staticmethod
    def parse(value: Any) -> Optional[datetime]:
        """
        Parse a value into a datetime object.
        Supports:
        - datetime objects (pass through)
        - ISO 8601 strings (e.g., "2026-03-10T09:15:00")
        - date objects (convert to datetime at midnight)
        - timestamps (int/float)
        """
        if isinstance(value, datetime):
            return value

        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())

        if isinstance(value, str):
            # Try ISO 8601 format
            try:
                # Handle with/without timezone
                if "T" in value:
                    # Try with timezone first
                    for fmt in [
                        "%Y-%m-%dT%H:%M:%S%z",
                        "%Y-%m-%dT%H:%M:%S.%f%z",
                        "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%dT%H:%M:%S.%f",
                        "%d-%m-%YT%H:%M:%S",
                        "%d-%m-%YT%H:%M:%S.%f",
                    ]:
                        try:
                            return datetime.strptime(value, fmt)
                        except ValueError:
                            continue
                # Try date-only format
                for fmt in ["%Y-%m-%d", "%d-%m-%Y"]:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue
                return datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                pass

        if isinstance(value, (int, float)):
            # Assume Unix timestamp
            try:
                return datetime.fromtimestamp(value)
            except (ValueError, OSError):
                pass

        return None


# ============================================================
# PART 1: STATE STRUCTURE
# Clear separation of concerns
# ============================================================

class Phase(str, Enum):
    """Which phase of planning are we in?"""
    GATHERING = "gathering"      # Collecting user requirements
    DRAFTING = "drafting"        # Creating initial plan from RAG
    REFINING = "refining"        # User is editing the plan
    VERIFYING = "verifying"      # Checking real availability
    FINALIZING = "finalizing"    # Creating bookable itinerary


class ChangeType(str, Enum):
    """What kind of change did user make?"""
    NONE = "none"
    DESTINATION = "destination"     # Major: reset everything
    DATES = "dates"                 # Major: reset flights, hotels, cabs
    BUDGET = "budget"               # Medium: re-filter options
    DURATION = "duration"           # Medium: adjust itinerary length
    ADD_ACTIVITY = "add_activity"   # Minor: just add one thing
    REMOVE_ACTIVITY = "remove_activity"
    CHANGE_HOTEL_PREF = "hotel_preference"
    CHANGE_FLIGHT_PREF = "flight_preference"
    MINOR_EDIT = "minor_edit"       # Small tweaks


# Core requirements - what user wants (small, always in context)
class CoreRequirements(TypedDict, total=False):
    destination: List[str]
    origin: str
    start_date: str
    end_date: str
    budget_total: int
    travelers: int  # Total count (deprecated - use pax breakdown instead)
    type_of_trip: str


# Preferences for each category (loaded only when needed)
class HotelPreferences(TypedDict):
    location: Required[str]
    hotel_name:Optional[str]
    country: Optional[str]
    hotel_budget:Optional[int]
    star_rating: Optional[int]
    amenities: Optional[List[str]]  # ["pool", "beach-access", "wifi"]
    location_preference: Optional[str]  # "city-center", "beachside", "quiet"


class FlightPreferences(TypedDict, total=False):
    budget: Optional[int]
    preferred_time: Optional[str]  # "morning", "evening", "any"
    departure_date: Required[str]
    departure_city: Required[str]
    arrival_city: Required[str]
    flight_preference: Optional[str]
    class_preference: Optional[str]  # "economy", "business"
    pax_adults: Optional[int]
    pax_children: Optional[int]
    pax_infants: Optional[int]


class ActivityPreferences(TypedDict, total=False):
    types: List[str]  # ["water-sports", "sightseeing", "relaxation"]
    intensity: str  # "relaxed", "moderate", "packed"
    must_do: List[str]  # Activities user specifically wants


# Draft itinerary - ideas, not real bookings
class DraftDay(TypedDict, total=False):
    day_number: int
    date: str
    morning: str
    afternoon: str
    evening: str
    hotel_idea: str  # "beachside resort" - not a real hotel yet
    notes: str


class DraftItinerary(TypedDict, total=False):
    summary: str
    days: List[DraftDay]
    estimated_budget: int
    hotel_type: str
    flight_type: str


# Verified options - real data from APIs
class VerifiedHotel(TypedDict):
    id: str
    name: str
    total_price: int
    rating: float
    amenities: List[str]
    location: str
    available: bool


class VerifiedFlight(TypedDict):
    id: str
    airline: str
    flight_number: str
    departure_time: str
    arrival_time: str
    price: int
    stops: int
    available: bool


class VerifiedActivity(TypedDict):
    id: str
    name: str
    operator: str
    price: int
    duration: str
    day_suggested: int
    available: bool


class VerifiedCab(TypedDict):
    id: str
    type: str  # "airport-pickup", "day-rental", "point-to-point"
    provider: str
    price: int
    available: bool


class VerifiedOptions(TypedDict, total=False):
    hotels: List[VerifiedHotel]
    flights_outbound: List[VerifiedFlight]
    flights_return: List[VerifiedFlight]
    activities: List[VerifiedActivity]
    cabs: List[VerifiedCab]


FlightsRole = Literal["user", "assistant", "tool"]


class FlightsConversationMessage(TypedDict, total=False):
    role: FlightsRole
    content: str
    timestamp: Optional[str]  # ISO 8601


# ============================================================
# Per-leg search schema (NO nested "legs")
# Each item inside searches == one leg search record
# Multi-leg trips are represented by multiple search records
# linked via trip_id (or journey_id)
# ============================================================

LegKey = Literal["leg_1", "leg_2", "leg_3"]


class PassengerCounts(TypedDict, total=False):
    adults: int
    children: int
    infants: int


class FlightSearchCriteria(TypedDict, total=False):
    """
    Criteria captured for ONE LEG search.
    """
    origin: str
    destination: str

    departure_date: str         # YYYY-MM-DD (user-level)
    return_date: Optional[str]  # optional (generally unused in per-leg, but kept compatible)
    date_of_travel: Optional[str]  # DD-MM-YYYY (tool-level)

    pax: PassengerCounts
    cabin: str                  # e.g., "E" / "B"
    trip_type: str              # e.g., "intl" / "dom"


class FlightPreferences(TypedDict, total=False):
    """
    User preferences for THIS LEG only (separate from criteria and selection).
    Keep it flexible to support future filters.
    """
    filters: Dict[str, Any]     # layover, airline, dep time window, budget, baggage etc.
    notes: Optional[str]


class FlightSelection(TypedDict, total=False):
    """
    A single user selection for a given leg search.

    IMPORTANT:
    - No metadata here.
    - Store the full selected flight object (copied from raw_response).
    """
    selected_option_id: Optional[str]   # if tool provides an id
    selected_index: Optional[int]       # 1-based index shown to user
    selected_key: Optional[str]         # e.g., "Flight 7" (if raw_response uses keys)
    selected_display_key: Optional[str] # e.g., "Flight 2" in filtered/renumbered view
    flight: Any                         # FULL flight dict/object from raw_response
    selected_at: Optional[str]          # ISO 8601


class AirportDisambiguation(TypedDict, total=False):
    """
    Airport disambiguation workflow FOR THIS LEG.
    """
    from_city_name: Optional[str]
    to_city_name: Optional[str]
    from_airports: Optional[List[Dict[str, Any]]]
    to_airports: Optional[List[Dict[str, Any]]]
    waiting_for_from_airport_selection: Optional[bool]
    waiting_for_to_airport_selection: Optional[bool]


class FlightSearchRecord(TypedDict, total=False):
    """
    ONE LEG search record.
    Stored as searches["Search 1"], searches["Search 2"], ... (append-only).
    """
    # identity
    search_id: str
    created_at: str                # ISO 8601

    # trip grouping (links legs together)
    trip_id: Optional[str]         # same for outbound+return (or multicity)
    leg_key: LegKey                # "outbound", "return", or "leg_N"
    leg_order: Optional[int]       # 1,2,3... useful for multicity

    # core data
    search_criteria: FlightSearchCriteria
    preferences: FlightPreferences

    # tool output for THIS LEG
    raw_response: Any
    result_count: int

    # UI / presentation state for THIS LEG (helps map "Option N" back to raw_response)
    presented_source_keys: Optional[List[str]]  # e.g., ["Flight 12", "Flight 3", ...] in the order shown

    # selections for THIS LEG (append-only)
    selections: List[FlightSelection]

    # disambiguation state for THIS LEG (optional)
    airport_disambiguation: Optional[AirportDisambiguation]

    # errors
    error: Optional[str]


class FlightsState(TypedDict, total=False):
    """
    Master flights state:
    - conversation history
    - append-only searches (each search is ONE LEG)
    - latest pointers/errors
    """
    conversation: List[FlightsConversationMessage]
    searches: Dict[str, FlightSearchRecord]
    latest_search_key: Optional[str]
    latest_error: Optional[str]
    should_end_conversation: Optional[bool]
    ended_at: Optional[str]


HotelsRole = Literal["user", "assistant", "tool"]


class HotelsConversationMessage(TypedDict, total=False):
    role: HotelsRole
    content: str
    timestamp: Optional[str]  # ISO 8601 string


class HotelPassengerCounts(TypedDict, total=False):
    """Passenger breakdown for hotel search."""
    adults: int
    children: int
    infants: int


class HotelSearchCriteria(TypedDict, total=False):
    """
    Core search criteria - only changes to destination trigger new search.
    Other changes (dates, pax, star rating) are applied as filters.
    """
    country: str
    city: str
    city_code: str

    check_in_date: str   # YYYY-MM-DD
    check_out_date: str  # YYYY-MM-DD

    rooms: int
    pax: HotelPassengerCounts

    star_rating: Optional[int]
    budget_per_night: Optional[int]


class HotelSearchPreferences(TypedDict, total=False):
    """
    User preferences for filtering and sorting results.
    These don't trigger new searches - just filter existing results.
    """
    filters: Dict[str, Any]  # e.g., {"refundable": true, "near": "Termini"}
    amenities: Optional[List[str]]  # ["breakfast", "wifi", "gym"]
    location_preference: Optional[str]  # "Near Termini station", "City center"
    notes: Optional[str]  # Any additional user notes


class HotelRoomOption(TypedDict, total=False):
    """Individual room option within a hotel."""
    room_name: str
    meal_plan: str
    refundable: bool
    price: Dict[str, Any]  # {"currency": "EUR", "total": 420}


class HotelLocation(TypedDict, total=False):
    """Hotel location coordinates."""
    lat: float
    lng: float


class HotelInfo(TypedDict, total=False):
    """Complete hotel information from raw response."""
    hotel_name: str
    hotel_code: str
    star_rating: int
    address: Optional[str]
    location: Optional[HotelLocation]
    amenities: Optional[List[str]]
    room_options: Optional[List[HotelRoomOption]]
    rating: Optional[float]  # User rating (e.g., 9.2)


class HotelSelection(TypedDict, total=False):
    """
    A single user selection for a hotel.
    Stores the FULL hotel object from raw_response.
    """
    selected_index: Optional[int]       # 1-based index shown to user
    selected_key: Optional[str]         # e.g., "Hotel 1" (key from raw_response)
    selected_display_key: Optional[str] # e.g., "Hotel 2" in filtered/renumbered view
    hotel: HotelInfo                    # FULL hotel dict from raw_response
    selected_at: Optional[str]          # ISO 8601


class HotelSearchRecord(TypedDict, total=False):
    """
    ONE hotel search record for ONE city/location.
    Stored as searches["Search 1"], searches["Search 2"], ... (append-only).

    Design principles:
    1. Only destination changes trigger new searches
    2. Date/pax/rating changes are applied as filters to existing results
    3. Supports multi-city trips via stay_key/stay_order
    4. Selections are append-only within each search
    """
    # Identity
    search_id: str                      # e.g., "hs_b7c4d1a9"
    created_at: str                     # ISO 8601
    signature: str                      # Deduplication key: "Italy|Rome|134622|2026-03-10|2026-03-12|rooms=1|adults=2|children=0|infants=0|stars=4"

    # Multi-city trip grouping
    stay_key: Optional[str]             # e.g., "stay_1", "stay_2" (links searches to stays)
    stay_order: Optional[int]           # 1, 2, 3... (sequence of stays in trip)

    # Core search data
    search_criteria: HotelSearchCriteria
    preferences: HotelSearchPreferences

    # Tool output for THIS search
    raw_response: Dict[str, Any]        # Full provider response (stores hotels as dict)
    result_count: int

    # Selections for THIS search (append-only)
    selections: List[HotelSelection]

    # Errors
    error: Optional[str]


class HotelsState(TypedDict, total=False):
    """
    Master hotels state:
    - conversation history
    - append-only searches (each search is ONE city/location)
    - latest pointers/errors

    Key behaviors:
    1. Only destination changes trigger new tool searches
    2. Other changes (dates, pax, rating) apply filters to existing results
    3. Supports multiple city searches for multi-city trips
    """
    conversation: List[HotelsConversationMessage]
    searches: Dict[str, HotelSearchRecord]  # keys: "Search 1", "Search 2", ...
    latest_search_key: Optional[str]
    latest_error: Optional[str]
    should_end_conversation: Optional[bool]
    ended_at: Optional[str]  # ISO 8601


# ============================================================
# ATTRACTIONS STATE (Unified schema for state_demo.py)
# - Mirrors Flights/Hotels style: searches + latest_search_key
# - Supports MULTI-SELECTION: selected is a list (only user-picked items)
# ============================================================

AttractionsRole = Literal["user", "assistant", "tool"]


class AttractionsConversationMessage(TypedDict, total=False):
    """A single message in the Attractions Agent conversation history."""
    role: AttractionsRole
    content: str
    timestamp: Optional[str]  # ISO 8601 string


class AttractionInfo(TypedDict, total=False):
    """
    Minimal normalized attraction shape.

    Your tool can return: [{"name": "..."}]
    Keep permissive so you can extend later (address, tags, hours, etc).
    """
    name: str
    address: Optional[str]
    category: Optional[str]
    city: Optional[str]
    country: Optional[str]
    metadata: Dict[str, Any]


class AttractionSearchCriteria(TypedDict, total=False):
    """
    Search criteria captured for an attractions search.
    Keep flexible (like flights/hotels) because tools/providers vary.
    """
    country: Optional[str]
    city: Optional[str]
    limit: Optional[int]
    categories: Optional[List[str]]
    filters: Dict[str, Any]  # provider-specific / future


class AttractionSearchRecord(TypedDict, total=False):
    """
    One attractions search run + its results.
    Stored in searches dict keyed by "Search 1", "Search 2", ...
    """
    search_id: str
    signature: str
    criteria: AttractionSearchCriteria

    raw_response: Any  # store tool payload as-is (list/dict/etc)

    result_count: int
    created_at: str  # ISO 8601 string
    selections: List["AttractionSelection"]  # append-only selections for THIS search
    error: Optional[str]


class AttractionSelection(TypedDict, total=False):
    """
    A user selection from a prior search.
    IMPORTANT: Only selected attractions go here (not all results).
    """
    selected_key: Optional[str]      # e.g., "Attraction 3"
    selected_display_key: Optional[str]  # e.g., "Attraction 1" in renumbered view
    selected_source_index: Optional[int] # original 1-based index if tool returned a list
    selected_index: Optional[int]    # 1-based index shown to user
    attraction: Any                  # FULL attraction dict/object from raw_response
    selected_at: Optional[str]       # ISO 8601 string


class AttractionsState(TypedDict, total=False):
    """
    Dedicated state bucket for Attractions.

    Key points:
    - searches keeps all results (append-only)
    - selections are stored per-search under searches["Search N"]["selections"]
    """
    conversation: List[AttractionsConversationMessage]

    searches: Dict[str, AttractionSearchRecord]  # keys: "Search 1", "Search 2", ...
    latest_search_key: Optional[str]

    latest_error: Optional[str]
    should_end_conversation: Optional[bool]
    ended_at: Optional[str]  # ISO 8601 string


#============================Itineary JSON======================================

class InputDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination: List[str]
    duration: int
    hub: str
    travel_start: date
    travel_end: date


class Passengers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adult: int
    children: int
    infants: int
    total: int


class TripSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str


# ----------------------------
# Hotels
# ----------------------------
class Hotel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hotel_code: str
    hotel_name: str
    city_name: str
    address: Optional[str] = None
    rating: Optional[float] = None
    check_in_time: Optional[str] = None
    check_out_time: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    facilities: Optional[List[str]] = None


# ----------------------------
# Flights
# ----------------------------
class Layover(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str
    airport_code: str
    duration: str  # keep as string like "0:02:40"


class RouteSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    departure_city: str
    departure_airport_code: str
    arrival_city: str
    arrival_airport_code: str

    departure_time: datetime
    arrival_time: datetime

    flight_number: str
    operating_carrier: Optional[str] = None
    aircraft_model: Optional[str] = None
    segment_duration: Optional[str] = None

    @field_validator("departure_time", "arrival_time", mode="before")
    @classmethod
    def _parse_dt(cls, v: Any) -> datetime:
        dt = FlexibleDateTime.parse(v)
        if dt is None:
            raise ValueError("datetime is required")
        return dt


class Fare(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_price: float
    base_fare: float
    taxes: float
    fare_class_type: Optional[str] = None
    fare_type: Optional[str] = None


class Flight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    airline_name: str
    departure_city: str
    arrival_city: str
    total_travel_duration: str  # e.g. "0:04:05"
    total_layover_duration: str
    cabin_class_type: str

    aircraft_models: List[str] = Field(default_factory=list)
    layovers: List[Layover] = Field(default_factory=list)
    route: List[RouteSegment]

    fare: Optional[Fare] = None


# ----------------------------
# Transfers
# ----------------------------

class Transfer(BaseModel):
    """
    Normalized itinerary transfer entry.

    Note: This is used by `DayPlan.transfers` in the itinerary JSON section and is
    intentionally permissive (`extra="ignore"`) because transfer providers/tools
    may include additional fields.
    """

    model_config = ConfigDict(extra="ignore")

    transfer_id: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    pickup_datetime: Optional[str] = None  # ISO 8601

    vehicle_models: Optional[str] = None
    vehicle_category: Optional[str] = None  # e.g., "BUS", "EXCLUSIVE_MINIVAN"
    vehicle_type: Optional[str] = None
    price: Optional[float] = None
    currency_code: Optional[str] = None

    addons: List[Dict[str, Any]] = Field(default_factory=list)
    notes: Optional[str] = None


TransfersRole = Literal["user", "assistant", "tool"]


class TransfersConversationMessage(TypedDict, total=False):
    role: TransfersRole
    content: str
    timestamp: Optional[str]  # ISO 8601


# ----------------------------
# Trip Planner (orchestrator) conversation
# ----------------------------
TripPlannerRole = Literal["user", "assistant"]


class TripPlannerConversationMessage(TypedDict, total=False):
    role: TripPlannerRole
    content: str
    timestamp: Optional[str]  # ISO 8601


class TransferAddonInfo(TypedDict, total=False):
    type: Optional[str]
    price: Optional[float]
    max_allowed: Optional[int]


class TransferVehicleInfo(TypedDict, total=False):
    """
    One vehicle option from the tool output, stored as-is (normalized shape).
    """
    vehicle_models: Optional[str]          # tool gives comma-separated models string
    vehicle_category: Optional[str]        # vehicle category (e.g., "BUS", "EXCLUSIVE_MINIVAN")
    price: Optional[float]
    currency_code: Optional[str]
    available_addons: List[TransferAddonInfo]


class TransferSearchCriteria(TypedDict, total=False):
    """
    Criteria captured for ONE transfer search.
    Keep flexible because providers differ.
    """
    origin: str
    destination: str

    # Optional fields (use if your tool/provider supports them later)
    transfer_type: Optional[str]           # "airport_to_hotel" / "hotel_to_airport"
    pickup_datetime: Optional[str]         # ISO 8601
    pax: Optional[PassengerCounts]         # reuse your PassengerCounts TypedDict
    notes: Optional[str]


class TransferPreferences(TypedDict, total=False):
    """
    Filters/sort that should NOT force a new tool call (like hotels).
    """
    filters: Dict[str, Any]                # e.g., {"max_price": 70, "addon": "BABY_SEAT"}
    sort_by: Optional[str]                 # "price_asc", "price_desc"
    notes: Optional[str]


class TransferSelection(TypedDict, total=False):
    """
    A single user selection.
    IMPORTANT:
    - Store the FULL selected vehicle object (copied from raw_response["vehicles"][...]).
    - No derived metadata required to reconstruct the selection.
    """
    selected_key: Optional[str]            # e.g., "Vehicle 2"
    selected_display_key: Optional[str]    # e.g., "Vehicle 1" in filtered/renumbered view
    selected_index: Optional[int]          # 1-based index shown to user
    vehicle: Any                           # FULL vehicle dict/object from raw_response
    selected_at: Optional[str]             # ISO 8601


class TransferSearchRecord(TypedDict, total=False):
    """
    ONE transfer search record.
    Stored as searches["Search 1"], searches["Search 2"], ... (append-only).
    """
    search_id: str
    created_at: str                        # ISO 8601
    signature: str                         # e.g., "BCN|Comte Borrell 73, ..."

    search_criteria: TransferSearchCriteria
    preferences: TransferPreferences

    # tool output for THIS search (store payload exactly)
    raw_response: Any                      # your tool result dict

    # Optional normalized view for UI/LLM convenience ("Vehicle 1"...)
    vehicles: Dict[str, TransferVehicleInfo]
    result_count: int

    # user selections (append-only)
    selections: List[TransferSelection]

    error: Optional[str]


class TransfersState(TypedDict, total=False):
    """
    Master transfers state (mirrors Flights/Hotels patterns):
    - conversation history
    - append-only searches
    - latest pointers/errors
    """
    conversation: List[TransfersConversationMessage]
    searches: Dict[str, TransferSearchRecord]
    latest_search_key: Optional[str]
    latest_error: Optional[str]
    should_end_conversation: Optional[bool]
    ended_at: Optional[str]                # ISO 8601


# ----------------------------
# Activities
# ----------------------------
ActivityType = Literal["sightseeing"]


class Activity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ActivityType
    time: str
    location: str
    description: str

    attraction: str


TimelineItemType = Literal[
    "arrival",
    "transfer",
    "check_in",
    "check_out",
    "activity",
]

TransportMode = Literal[
    "airport_transfer",
]

class ActivityTimelineItem(BaseModel):
    """Timeline item for activities/attractions.

    REQUIRED fields:
    - type: must be "activity"
    - title: activity/attraction name
    - description: brief description
    - start_time: start time in HH:MM format (e.g., "09:00")
    - end_time: end time in HH:MM format (e.g., "12:00")

    DO NOT include: duration_minutes, transport_mode, time, or any other fields
    """
    model_config = ConfigDict(extra="forbid")

    type: Literal["activity"]
    title: str
    description: str = ""
    start_time: str  # Required: HH:MM format
    end_time: str    # Required: HH:MM format

class ArrivalTimelineItem(BaseModel):
    """Timeline item for flight arrivals."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["arrival"]
    title: str
    description: str = ""
    time: str  # Single time field for arrivals

class CheckInTimelineItem(BaseModel):
    """Timeline item for hotel check-in."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["check_in"]
    title: str
    description: str = ""

class CheckOutTimelineItem(BaseModel):
    """Timeline item for hotel check-out."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["check_out"]
    title: str
    description: str = ""

class TransferTimelineItem(BaseModel):
    """Timeline item for transfers."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["transfer"]
    title: str
    description: str = ""
    start_time: str | None = None
    end_time: str | None = None
    duration_minutes: int | None = None
    transport_mode: TransportMode | None = None

# Union type for all timeline items
DayTimelineItem = (
    ActivityTimelineItem
    | ArrivalTimelineItem
    | CheckInTimelineItem
    | CheckOutTimelineItem
    | TransferTimelineItem
)

def _is_hhmm(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip()
    if len(v) != 5 or v[2] != ":":
        return False
    hh, mm = v.split(":", 1)
    return hh.isdigit() and mm.isdigit() and 0 <= int(hh) <= 23 and 0 <= int(mm) <= 59


def _infer_end_time(start_time: str, *, minutes: int = 120) -> str:
    """Infer an end time HH:MM by adding a default duration to start_time."""
    try:
        dt = datetime.strptime(start_time.strip(), "%H:%M")
        dt2 = dt + timedelta(minutes=minutes)
        return dt2.strftime("%H:%M")
    except Exception:
        return "12:00"


def _to_hhmm(value: Any, *, default: str = "09:00") -> str:
    """Best-effort convert a value (HH:MM, ISO datetime, etc.) to HH:MM."""
    if isinstance(value, str):
        v = value.strip()
        if _is_hhmm(v):
            return v
        # Try ISO / other datetime strings using FlexibleDateTime
        dt = FlexibleDateTime.parse(v)
        if dt is not None:
            return dt.strftime("%H:%M")
    dt = FlexibleDateTime.parse(value)
    if dt is not None:
        return dt.strftime("%H:%M")
    return default


def _normalize_day_timeline_item(item: Any) -> DayTimelineItem | None:
    """Normalize a raw timeline item dict into the strict Pydantic models.

    This is intentionally defensive because LLM output often includes:
    - extra keys (e.g., transport_mode=None on activity items)
    - missing required keys (e.g., end_time on activity items)

    We sanitize the dict first to satisfy `extra="forbid"` schemas.
    """
    # If it's already a validated model, keep it
    if isinstance(
        item,
        (
            ActivityTimelineItem,
            ArrivalTimelineItem,
            CheckInTimelineItem,
            CheckOutTimelineItem,
            TransferTimelineItem,
        ),
    ):
        return item

    if not isinstance(item, dict):
        return None

    item_type = item.get("type")

    if item_type == "activity":
        start = _to_hhmm(item.get("start_time") or item.get("time"), default="10:00")
        end = _to_hhmm(item.get("end_time"), default=_infer_end_time(start))
        # EXACTLY 5 keys for ActivityTimelineItem (no extras, no null extras)
        clean = {
            "type": "activity",
            "title": (item.get("title") or item.get("attraction") or "").strip(),
            "description": (item.get("description") or "").strip(),
            "start_time": start,
            "end_time": end,
        }
        return ActivityTimelineItem.model_validate(clean)

    if item_type == "arrival":
        clean = {
            "type": "arrival",
            "title": (item.get("title") or "Flight Arrival").strip(),
            "description": (item.get("description") or "").strip(),
            "time": _to_hhmm(item.get("time") or item.get("arrival_time"), default="09:00"),
        }
        return ArrivalTimelineItem.model_validate(clean)

    if item_type == "check_in":
        clean = {
            "type": "check_in",
            "title": (item.get("title") or "Hotel Check-in").strip(),
            "description": (item.get("description") or "").strip(),
        }
        return CheckInTimelineItem.model_validate(clean)

    if item_type == "check_out":
        clean = {
            "type": "check_out",
            "title": (item.get("title") or "Hotel Check-out").strip(),
            "description": (item.get("description") or "").strip(),
        }
        return CheckOutTimelineItem.model_validate(clean)

    if item_type == "transfer":
        # Keep only allowed keys; drop everything else to satisfy extra="forbid"
        clean = {
            "type": "transfer",
            "title": (item.get("title") or "Transfer").strip(),
            "description": (item.get("description") or "").strip(),
            "start_time": item.get("start_time"),
            "end_time": item.get("end_time"),
            "duration_minutes": item.get("duration_minutes"),
            "transport_mode": item.get("transport_mode"),
        }
        return TransferTimelineItem.model_validate(clean)

    # Unknown timeline items are dropped (safer with strict unions + extra=forbid)
    return None

# ----------------------------
# Day + City itinerary
# ----------------------------
class DayPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    day: int
    label: str
    lead: str
    full_day_plan: List[DayTimelineItem] = Field(default_factory=list)
    date: str 
    hotels: Dict[str, Any] = Field(default_factory=dict)
    flights: Dict[str, Any] = Field(default_factory=dict)
    transfers: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("full_day_plan", mode="before")
    @classmethod
    def _normalize_full_day_plan(cls, v: Any) -> Any:
        if v is None:
            return []
        if not isinstance(v, list):
            return v
        return [x for x in (_normalize_day_timeline_item(item) for item in v) if x is not None]

class CityItinerary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city_name: str
    start_date: date
    end_date: date
    duration: str 
    days: List[DayPlan]


# ----------------------------
# Costing
# ----------------------------
class Costing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flights: float
    hotels: float
    attractions: float
    total: float


# ----------------------------
# Root model
# ----------------------------
class ItineraryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_details: InputDetails
    passengers: Passengers
    trip_summary: TripSummary
    itinerary: List[CityItinerary]
    costing: Costing


# Main state - everything the system knows
class TripPlannerState(TypedDict, total=False):
    # Conversation tracking
    conversation_id: str
    current_phase: Phase
    version: int

    # What changed in last message
    last_change_type: ChangeType
    changed_fields: List[str]

    # User's core requirements (always small)
    core: CoreRequirements

    # Track which services user actually requested (for partial trip support)
    requested_services: Optional[List[Literal["flights", "hotels", "activities", "transfers"]]]

    flights_state: FlightsState
    hotels_state: HotelsState
    attractions_state: AttractionsState
    transfers_state: TransfersState

    # The plans
    draft: DraftItinerary
    final_itinerary: ItineraryPlan

    # For RAG
    similar_trips_ids: List[str]  # Just IDs, not full data

    # Current context for LLM (minimal)
    current_context: str

    # What nodes need to run
    nodes_to_run: List[str]

    # Conversation
    user_message: str
    assistant_response: str  # Final synthesized response from Trip Planner
    # Back-compat / UI field name used by some modules
    assistant_message: str
    conversation_history: List[Dict[str, str]]  # List of {role, content} messages
    should_end_conversation: Optional[bool]
    ended_at: Optional[str]  # ISO 8601

    # Trip Planner memory / orchestration (owned by TripPlannerAgent)
    trip_planner_history: List[TripPlannerConversationMessage]
    trip_planner_queue: List[Literal["flights", "hotels", "activities", "transfers"]]
    trip_planner_waiting_for_order: Optional[bool]
    trip_planner_waiting_for_continue: Optional[bool]
    trip_planner_active_agent: Optional[Literal["flights", "hotels", "activities", "transfers"]]
    trip_planner_last_active_agent: Optional[Literal["flights", "hotels", "activities", "transfers"]]
    trip_planner_collected_responses: Dict[str, str]

    # Errors
    errors: List[str]

    # Internal routing
    _router_index: int
