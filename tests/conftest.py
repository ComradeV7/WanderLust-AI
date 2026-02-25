"""
conftest.py — Shared pytest fixtures for WanderLust-AI tests.

All external services (Nominatim, ORS, Groq, Gemini) are mocked here
so tests run fast, offline, and without spending API credits.

IMPORTANT — Why env vars are set here at module level:
  main.py initialises ChatGroq / ChatGoogleGenerativeAI at *module import
  time* (lines 20-26). Those SDK clients validate the API key during
  __init__, raising an error before any patch() can intercept the call.
  By setting dummy env vars here — which runs when pytest loads conftest,
  before any test file imports main — we satisfy the SDK's key check
  without making real API calls.  All actual network calls are mocked
  individually inside each test.
"""
import os

# Must be set BEFORE any test file imports main.py
os.environ.setdefault("GROQ_API_KEY",   "dummy-groq-key-for-tests")
os.environ.setdefault("GOOGLE_API_KEY", "dummy-google-key-for-tests")
os.environ.setdefault("ORS_API_KEY",    "dummy-ors-key-for-tests")
import pytest
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# Helpers: Fake location objects that mimic geopy's Location namedtuple
# ---------------------------------------------------------------------------

def make_fake_location(lat, lon, address, importance=0.5):
    """Create a fake geopy Location object."""
    loc = MagicMock()
    loc.latitude = lat
    loc.longitude = lon
    loc.address = address
    loc.raw = {"importance": importance}
    return loc


# ---------------------------------------------------------------------------
# Fixtures: Geocoder (Nominatim)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_megacity_location():
    """A location with high importance (megacity like London)."""
    return make_fake_location(51.5074, -0.1278, "London, UK", importance=0.9)


@pytest.fixture
def fake_regional_location():
    """A location with low importance (regional hub like Visakhapatnam)."""
    return make_fake_location(17.6868, 83.2185, "Visakhapatnam, India", importance=0.5)


@pytest.fixture
def fake_place_location():
    """A fake place found by geocoder (e.g. a museum)."""
    return make_fake_location(51.5081, -0.1281, "The British Museum, London, UK")


# ---------------------------------------------------------------------------
# Fixtures: Full TravelGraphState for agent node tests
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_state():
    """A fully populated TravelGraphState dict for testing agent nodes."""
    return {
        "destination": "London, UK",
        "duration_days": 3,
        "vibe": "historic, literary, cozy",
        "user_feedback": None,
        "places_to_avoid": ["Buckingham Palace"],
        "keywords": ["The British Museum", "Borough Market", "Tate Modern"],
        "search_results": [
            {"name": "The British Museum", "address": "Great Russell St, London", "coordinates": [-0.1270, 51.5194]},
            {"name": "Borough Market", "address": "8 Southwark St, London", "coordinates": [-0.0901, 51.5055]},
        ],
        "itinerary_draft": "",
    }


# ---------------------------------------------------------------------------
# Fixtures: FastAPI async test client
# ---------------------------------------------------------------------------

@pytest.fixture
async def api_client():
    """Async HTTP client for testing FastAPI endpoints via ASGI."""
    from main import api
    async with AsyncClient(transport=ASGITransport(app=api), base_url="http://test") as client:
        yield client


# ---------------------------------------------------------------------------
# Fixtures: Mock LLM responses
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_keyword_response():
    """Fake structured output from the vibe interpreter LLM."""
    from pydantic import BaseModel
    from typing import List

    class FakeKeywordList(BaseModel):
        keywords: List[str]

    return FakeKeywordList(keywords=["The British Museum", "Borough Market", "Tate Modern", "Hyde Park"])


@pytest.fixture
def mock_llm_text_response():
    """Fake plain-text LLM response for itinerary drafts."""
    response = MagicMock()
    response.content = (
        "**Day 1:**\n\n"
        "**Morning:** Visit The British Museum\n"
        "**Afternoon:** Explore Borough Market\n"
        "**Evening:** Walk along the Thames (General Suggestion)\n"
    )
    return response


# ---------------------------------------------------------------------------
# Fixtures: Mock ORS client
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ors_directions_response():
    """Fake ORS directions API response."""
    return {
        "routes": [
            {
                "summary": {
                    "duration": 1500,   # seconds → 25 minutes
                    "distance": 2300,   # metres → 2.3 km
                }
            }
        ]
    }
