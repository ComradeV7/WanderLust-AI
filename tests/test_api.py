"""Integration tests for the FastAPI endpoints in main.py.
LangGraph app.stream is mocked so tests run without real agents or API calls.
"""
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport


def make_stream_events(itinerary_text: str, extra_state: dict = None):
    """Simulate the event iterator yielded by LangGraph's app.stream()."""
    state_updates = {"itinerary_draft": itinerary_text, "keywords": ["Museum"], "search_results": []}
    if extra_state:
        state_updates.update(extra_state)
    return iter([{"itinerary_agent": state_updates}])


VALID_START_PAYLOAD = {
    "destination": "London, UK",
    "duration_days": 3,
    "vibe": "historic, literary",
    "places_to_avoid": [],
}

SAMPLE_ITINERARY = (
    "**Day 1:**\n**Morning:** The British Museum\n**Afternoon:** Borough Market\n"
)


@pytest.mark.asyncio
class TestStartPlanEndpoint:
    """Integration tests for the /plan/start endpoint."""

    async def test_returns_200_with_itinerary(self):
        """Valid request returns HTTP 200 with an itinerary_draft."""
        from main import api

        with patch("main.app") as mock_graph_app:
            mock_graph_app.stream.return_value = make_stream_events(SAMPLE_ITINERARY)

            async with AsyncClient(transport=ASGITransport(app=api), base_url="http://test") as client:
                response = await client.post("/plan/start", json=VALID_START_PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        assert "itinerary_draft" in data
        assert data["itinerary_draft"] == SAMPLE_ITINERARY

    async def test_returns_current_state(self):
        """Response includes current_state so the frontend can resume the plan."""
        from main import api

        with patch("main.app") as mock_graph_app:
            mock_graph_app.stream.return_value = make_stream_events(SAMPLE_ITINERARY)

            async with AsyncClient(transport=ASGITransport(app=api), base_url="http://test") as client:
                response = await client.post("/plan/start", json=VALID_START_PAYLOAD)

        data = response.json()
        assert "current_state" in data
        assert isinstance(data["current_state"], dict)
        assert data["current_state"]["destination"] == "London, UK"

    async def test_missing_required_field_returns_422(self):
        """Omitting a required field (e.g. 'vibe') returns HTTP 422."""
        from main import api

        async with AsyncClient(transport=ASGITransport(app=api), base_url="http://test") as client:
            response = await client.post("/plan/start", json={
                "destination": "London, UK",
                "duration_days": 3,
            })

        assert response.status_code == 422

    async def test_invalid_duration_type_returns_422(self):
        """Passing a string for duration_days fails validation."""
        from main import api

        async with AsyncClient(transport=ASGITransport(app=api), base_url="http://test") as client:
            response = await client.post("/plan/start", json={
                "destination": "London, UK",
                "duration_days": "three",
                "vibe": "historic",
            })

        assert response.status_code == 422

    async def test_places_to_avoid_defaults_to_empty_list(self):
        """places_to_avoid is optional and defaults to an empty list."""
        from main import api

        payload_without_avoid = {
            "destination": "London, UK",
            "duration_days": 2,
            "vibe": "modern",
        }

        with patch("main.app") as mock_graph_app:
            mock_graph_app.stream.return_value = make_stream_events(SAMPLE_ITINERARY)

            async with AsyncClient(transport=ASGITransport(app=api), base_url="http://test") as client:
                response = await client.post("/plan/start", json=payload_without_avoid)

        assert response.status_code == 200


@pytest.mark.asyncio
class TestResumePlanEndpoint:
    """Integration tests for the /plan/resume endpoint."""

    def _base_current_state(self):
        return {
            "destination": "London, UK",
            "duration_days": 3,
            "vibe": "historic",
            "user_feedback": None,
            "places_to_avoid": [],
            "keywords": ["Museum"],
            "search_results": [],
            "itinerary_draft": "Old itinerary...",
        }

    async def test_returns_updated_itinerary(self):
        """Resume with feedback returns an updated itinerary."""
        from main import api

        new_itinerary = "**Updated Day 1:** More food spots..."
        resume_payload = {
            "current_state": self._base_current_state(),
            "user_feedback": "More food, fewer museums!",
            "place_to_avoid": None,
        }

        with patch("main.app") as mock_graph_app:
            mock_graph_app.stream.return_value = make_stream_events(new_itinerary)

            async with AsyncClient(transport=ASGITransport(app=api), base_url="http://test") as client:
                response = await client.post("/plan/resume", json=resume_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["itinerary_draft"] == new_itinerary

    async def test_place_to_avoid_is_appended_to_state(self):
        """place_to_avoid should appear in the returned current_state."""
        from main import api

        current_state = self._base_current_state()
        resume_payload = {
            "current_state": current_state,
            "user_feedback": "Remove the Tower of London",
            "place_to_avoid": "Tower of London",
        }

        with patch("main.app") as mock_graph_app:
            mock_graph_app.stream.return_value = make_stream_events(
                SAMPLE_ITINERARY,
                extra_state={"places_to_avoid": ["Tower of London"]}
            )

            async with AsyncClient(transport=ASGITransport(app=api), base_url="http://test") as client:
                response = await client.post("/plan/resume", json=resume_payload)

        assert response.status_code == 200
        data = response.json()
        assert "Tower of London" in data["current_state"].get("places_to_avoid", [])

    async def test_missing_current_state_returns_422(self):
        """Omitting current_state fails validation."""
        from main import api

        async with AsyncClient(transport=ASGITransport(app=api), base_url="http://test") as client:
            response = await client.post("/plan/resume", json={
                "user_feedback": "More food please"
            })

        assert response.status_code == 422

    async def test_missing_user_feedback_returns_422(self):
        """Omitting user_feedback fails validation."""
        from main import api

        async with AsyncClient(transport=ASGITransport(app=api), base_url="http://test") as client:
            response = await client.post("/plan/resume", json={
                "current_state": self._base_current_state(),
            })

        assert response.status_code == 422
