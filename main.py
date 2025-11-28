import os
import json
import time
from dotenv import load_dotenv
import openrouteservice
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from typing import TypedDict, List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from fastapi import FastAPI

# LOAD ENV & INITIALIZE MODELS 
load_dotenv()

# Model 1: Fast (Llama 3.1 8B)
llm_fast = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Model 2: Tool User (Gemini 2.5 Flash)
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Model 3: Hero (Llama 3.3 70B)
llm_hero = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Define Structure for Vibe Agent output
class KeywordList(BaseModel):
    keywords: List[str] = Field(description="A list of specific, real place names.")

# DEFINE CLIENTS & TOOLS

# Initialize OpenRouteService (Only for Directions now)
try:
    ors_key = os.environ.get("ORS_API_KEY")
    if ors_key:
        ors_client = openrouteservice.Client(key=ors_key)
    else:
        ors_client = None
except Exception:
    ors_client = None

# Initialize Nominatim (Free Geocoder for Search)
geolocator = Nominatim(user_agent="hybrid_travel_agent_project_final")

@tool
def query_places_nominatim(query: str, location_name: str):
    """
    Smart search that adapts its radius based on the city's 'Importance'.
    """
    print(f"--- TOOL: Searching for '{query}' around '{location_name}' ---")
    
    try:
        # Get City Details & "Importance"
        # The 'importance' field (0.0 to 1.0) tells us if it's a Megacity or Town
        city_loc = geolocator.geocode(location_name, timeout=10, addressdetails=True)
        if not city_loc:
            print(f"   > Error: Target city '{location_name}' not found.")
            return []
        
        city_coords = (city_loc.latitude, city_loc.longitude)
        importance = city_loc.raw.get("importance", 0.5)
        
        # DYNAMIC RADIUS LOGIC
        # If importance > 0.75 (London, Tokyo, NYC), use small radius (City limits)
        # If importance <= 0.75 (Visakhapatnam, Bath), use large radius (Day trips allowed)
        if importance > 0.75:
            radius_limit = 30  # Strict city limit
            print(f"   > Detected MEGACITY (Score: {importance}). Radius set to {radius_limit}km.")
        else:
            radius_limit = 200 # Allow day trips
            print(f"   > Detected REGIONAL HUB (Score: {importance}). Radius set to {radius_limit}km.")

        # Strategy A: Strict Search ("Place, City")
        full_query = f"{query}, {location_name}"
        time.sleep(1.1) 
        place_loc = geolocator.geocode(full_query, timeout=10)
        
        # Strategy B: Global Search + Smart Distance Filter
        if not place_loc:
            print(f"   > Strict search failed. Trying global search for '{query}'...")
            time.sleep(1.1)
            # Fetch top 5 global matches
            candidates = geolocator.geocode(query, exactly_one=False, limit=5, timeout=10)
            
            if candidates:
                for cand in candidates:
                    cand_coords = (cand.latitude, cand.longitude)
                    dist = geodesic(city_coords, cand_coords).km
                    
                    # Check against our DYNAMIC radius
                    if dist <= radius_limit: 
                        place_loc = cand
                        print(f"   > Found match via global search: {cand.address} ({int(dist)}km away)")
                        break
                    else:
                        print(f"   > Skipping candidate: {int(dist)}km away (Limit: {radius_limit}km)")
        
        if not place_loc:
            print(f"   > No results found for '{query}'")
            return []

        return [{
            "name": query,
            "address": place_loc.address,
            "coordinates": [place_loc.longitude, place_loc.latitude]
        }]

    except Exception as e:
        print(f"   > Tool Error: {e}")
        return []

@tool
def get_ors_directions(start_coords: List[float], end_coords: List[float], profile: str = "foot-walking"):
    """Gets directions between two locations [lon, lat] using ORS."""
    if not ors_client: 
        return {"duration_minutes": 0, "distance_km": 0, "note": "Directions unavailable (No API Key)"}
    
    try:
        directions_result = ors_client.directions(
            coordinates=[start_coords, end_coords],
            profile=profile
        )
        if directions_result['routes']:
            summary = directions_result['routes'][0]['summary']
            return {
                "duration_minutes": round(summary.get('duration', 0) / 60, 1),
                "distance_km": round(summary.get('distance', 0) / 1000, 1)
            }
        return "No directions found."
    except Exception as e:
        return f"Error using OpenRouteService API: {e}"

# BIND TOOLS
llm_gemini_tools = llm_gemini.bind_tools([query_places_nominatim])
llm_hero_tools = llm_hero.bind_tools([get_ors_directions])

# AGENT STATE
class TravelGraphState(TypedDict):
    destination: str
    duration_days: int
    vibe: str
    user_feedback: Optional[str]
    places_to_avoid: List[str]
    keywords: List[str]
    search_results: List[dict]
    itinerary_draft: str

# AGENT NODES

def vibe_interpreter_agent(state: TravelGraphState):
    print(f"--- 1. VIBE AGENT (Analysing {state['destination']}) ---")
    
    llm_hero_structured = llm_hero.with_structured_output(KeywordList)
    
    # We aim for ~4 places per day to fill Morning/Afternoon/Evening
    target_count = max(10, state['duration_days'] * 4)
    
    prompt = f"""
    You are an expert travel consultant.
    Destination: {state['destination']}
    Trip Duration: {state['duration_days']} days.
    Vibe: {state['vibe']}
    Avoid: {state['places_to_avoid']}
    
    Task: Generate exactly {target_count} search terms.
    (We need enough places to fill a {state['duration_days']}-day itinerary).
    
    STRATEGY:
    1. **Megacities:** Return SPECIFIC NAMES (e.g., "The Louvre").
    2. **Smaller Regions/Cities:** Return GENERIC CATEGORIES (e.g., "Beach", "Seafood Restaurant").
       - Vary the categories! Don't just say "Beach" 5 times. 
       - Use: "Quiet Beach", "Busy Beach", "Sunset Viewpoint", "Local Market", "History Museum", "Portuguese Church", "Spicy Restaurant".
    """
    
    response = llm_hero_structured.invoke(prompt)
    return {"keywords": response.keywords}

def search_agent(state: TravelGraphState):
    print("--- 2. SEARCH AGENT ---")
    
    location_fixed = state['destination']
    # Minimal normalization if needed
    if "London" in location_fixed and "UK" not in location_fixed:
        location_fixed = "London, UK"
    
    search_prompt = f"""
    Find coordinates for these: {state['keywords']}. 
    Use the tool `Maps_nominatim`.
    CRITICAL: You MUST pass '{location_fixed}' as the 'location_name' argument for every call.
    """
    
    response = llm_gemini_tools.invoke(search_prompt)
    
    all_places = []
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'query_places_nominatim':
                tool_output = query_places_nominatim.invoke(tool_call['args'])
                if isinstance(tool_output, list):
                    all_places.extend(tool_output)

    # Deduplicate results
    seen = set()
    final_places = []
    for p in all_places:
        if p['name'] not in seen and p['name'] not in state['places_to_avoid']:
            final_places.append(p)
            seen.add(p['name'])
            
    return {"search_results": final_places}

def itinerary_agent(state: TravelGraphState):
    print("--- 3. EXECUTING: Itinerary Agent (Llama 3 70B) ---")
    
    prompt = f"""
    You are a professional travel itinerary curator for {state['destination']}.
    
    Context:
    - Trip Duration: {state['duration_days']} days.
    - Desired Vibe: {state['vibe']}.
    - Verified Locations Found: {json.dumps(state['search_results'], indent=2)}
    
    Your Task:
    Construct a logical, narrative-driven itinerary using ONLY the verified locations provided above.
    
    Guidelines:
    1. **Clean Output:** Do NOT display coordinates (lat/long) or full street addresses in the final text. Just use the venue name.
    2. **Logical Flow:** Group activities by neighborhood to minimize travel time.
    3. **Narrative:** Explain *why* each spot fits the '{state['vibe']}' vibe. Write in an engaging, travel-blog style.
    4. **Gaps:** If you have few verified locations, suggest general activities (e.g. "Walk along the beach") to fill the day, but prioritize the verified spots.
    5. **Format:** Use clean Markdown with headers for each Day.
    """
    
    response = llm_hero.invoke(prompt)
    return {"itinerary_draft": response.content}

# GRAPH DEFINITION

def await_feedback(state: TravelGraphState):
    return {"user_feedback": None}

def check_feedback(state: TravelGraphState):
    if state.get("user_feedback"):
        return "vibe_interpreter"
    else:
        return END

workflow = StateGraph(TravelGraphState)
workflow.add_node("vibe_interpreter", vibe_interpreter_agent)
workflow.add_node("search_agent", search_agent)
workflow.add_node("itinerary_agent", itinerary_agent)
workflow.add_node("await_feedback", await_feedback)

workflow.set_entry_point("vibe_interpreter")
workflow.add_edge("vibe_interpreter", "search_agent")
workflow.add_edge("search_agent", "itinerary_agent")
workflow.add_edge("itinerary_agent", "await_feedback")
workflow.add_conditional_edges(
    "await_feedback", check_feedback, {"vibe_interpreter": "vibe_interpreter", END: END}
)

app = workflow.compile()

# FASTAPI SERVER
api = FastAPI(title="Hybrid-Model Travel Planner")

class PlanRequest(BaseModel):
    destination: str
    duration_days: int
    vibe: str
    places_to_avoid: List[str] = []

class ResumeRequest(BaseModel):
    current_state: Dict[str, Any]
    user_feedback: str
    place_to_avoid: Optional[str] = None

@api.post("/plan/start")
def start_plan(request: PlanRequest):
    initial_input = request.model_dump()
    initial_input["user_feedback"] = "Start"
    
    final_state = initial_input.copy()
    for event in app.stream(initial_input):
        for node_name, node_output in event.items():
            if isinstance(node_output, dict):
                final_state.update(node_output)
    
    return {"itinerary_draft": final_state.get('itinerary_draft'), "current_state": final_state}

@api.post("/plan/resume")
def resume_plan(request: ResumeRequest):
    resume_state = request.current_state
    resume_state["user_feedback"] = request.user_feedback
    if request.place_to_avoid:
        resume_state["places_to_avoid"].append(request.place_to_avoid)

    final_state = resume_state.copy()
    for event in app.stream(resume_state):
        for node_name, node_output in event.items():
            if isinstance(node_output, dict):
                final_state.update(node_output)
            
    return {"itinerary_draft": final_state.get('itinerary_draft'), "current_state": final_state}