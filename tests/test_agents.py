"""Unit tests for each agent node function in main.py.
All LLM calls (Groq/Gemini) are mocked so tests run offline without API keys.
"""
import pytest
from unittest.mock import MagicMock, patch
from langgraph.graph import END


def build_state(**overrides):
    """Return a valid TravelGraphState dict, optionally overriding fields."""
    base = {
        "destination": "London, UK",
        "duration_days": 3,
        "vibe": "historic, literary",
        "user_feedback": None,
        "places_to_avoid": [],
        "keywords": [],
        "search_results": [],
        "itinerary_draft": "",
    }
    base.update(overrides)
    return base


class TestVibeInterpreterAgent:
    """Unit tests for the first agent node (keyword generation)."""

    def test_returns_keywords_in_state(self, mock_keyword_response):
        """Agent should update the 'keywords' key in the returned state dict."""
        with patch("main.llm_hero") as mock_llm:
            mock_llm.with_structured_output.return_value.invoke.return_value = mock_keyword_response

            from main import vibe_interpreter_agent
            state = build_state(destination="London, UK", duration_days=3, vibe="historic")
            result = vibe_interpreter_agent(state)

        assert "keywords" in result
        assert isinstance(result["keywords"], list)
        assert len(result["keywords"]) > 0

    def test_target_count_scales_with_duration(self, mock_keyword_response):
        """Prompt requests duration_days * 4 keywords (min 10); both calls should succeed."""
        with patch("main.llm_hero") as mock_llm:
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = mock_keyword_response
            mock_llm.with_structured_output.return_value = mock_structured

            from main import vibe_interpreter_agent
            result_1day = vibe_interpreter_agent(build_state(duration_days=1))
            result_5day = vibe_interpreter_agent(build_state(duration_days=5))

        assert "keywords" in result_1day
        assert "keywords" in result_5day

    def test_places_to_avoid_passed_to_prompt(self):
        """Agent should include places_to_avoid in the LLM prompt."""
        from pydantic import BaseModel
        from typing import List

        class FakeKW(BaseModel):
            keywords: List[str]

        fake_response = FakeKW(keywords=["Hyde Park", "Tate Modern"])

        with patch("main.llm_hero") as mock_llm:
            mock_llm.with_structured_output.return_value.invoke.return_value = fake_response

            from main import vibe_interpreter_agent
            state = build_state(places_to_avoid=["Buckingham Palace", "Tower of London"])
            result = vibe_interpreter_agent(state)

        mock_llm.with_structured_output.return_value.invoke.assert_called_once()
        call_args = mock_llm.with_structured_output.return_value.invoke.call_args[0][0]
        assert "Buckingham Palace" in call_args or "Tower of London" in call_args


class TestSearchAgent:
    """Unit tests for the second agent node (geospatial search)."""

    def _make_tool_call_response(self, tool_calls):
        """Build a mock LLM response with tool calls."""
        response = MagicMock()
        response.tool_calls = tool_calls
        return response

    def test_valid_tool_calls_populate_search_results(self):
        """Agent calls the geocoding tool and returns results in state."""
        fake_tool_call = {
            "name": "query_places_nominatim",
            "args": {"query": "The British Museum", "location_name": "London, UK"},
        }
        fake_result = [{"name": "The British Museum", "address": "London", "coordinates": [-0.127, 51.519]}]

        with patch("main.llm_gemini_tools") as mock_llm, \
             patch("main.query_places_nominatim") as mock_tool:
            mock_llm.invoke.return_value = self._make_tool_call_response([fake_tool_call])
            mock_tool.invoke.return_value = fake_result

            from main import search_agent
            state = build_state(keywords=["The British Museum"])
            result = search_agent(state)

        assert "search_results" in result
        assert len(result["search_results"]) == 1
        assert result["search_results"][0]["name"] == "The British Museum"

    def test_deduplicates_results(self):
        """Duplicate place names should appear only once in search_results."""
        fake_tool_call_1 = {"name": "query_places_nominatim", "args": {"query": "Museum", "location_name": "London"}}
        fake_tool_call_2 = {"name": "query_places_nominatim", "args": {"query": "Museum", "location_name": "London"}}
        dupe_result = [{"name": "Museum", "address": "London", "coordinates": [0.0, 51.0]}]

        with patch("main.llm_gemini_tools") as mock_llm, \
             patch("main.query_places_nominatim") as mock_tool:
            mock_llm.invoke.return_value = self._make_tool_call_response([fake_tool_call_1, fake_tool_call_2])
            mock_tool.invoke.return_value = dupe_result

            from main import search_agent
            result = search_agent(build_state(keywords=["Museum", "Museum"]))

        names = [p["name"] for p in result["search_results"]]
        assert names.count("Museum") == 1

    def test_filters_places_to_avoid(self):
        """Places in the avoid list should not appear in search_results."""
        fake_tool_call = {"name": "query_places_nominatim", "args": {"query": "Buckingham Palace", "location_name": "London"}}
        avoid_result = [{"name": "Buckingham Palace", "address": "London", "coordinates": [-0.14, 51.50]}]

        with patch("main.llm_gemini_tools") as mock_llm, \
             patch("main.query_places_nominatim") as mock_tool:
            mock_llm.invoke.return_value = self._make_tool_call_response([fake_tool_call])
            mock_tool.invoke.return_value = avoid_result

            from main import search_agent
            state = build_state(places_to_avoid=["Buckingham Palace"], keywords=["Buckingham Palace"])
            result = search_agent(state)

        assert all(p["name"] != "Buckingham Palace" for p in result["search_results"])

    def test_no_tool_calls_returns_empty_results(self):
        """If the LLM makes no tool calls, search_results is empty."""
        with patch("main.llm_gemini_tools") as mock_llm:
            mock_llm.invoke.return_value = self._make_tool_call_response([])

            from main import search_agent
            result = search_agent(build_state(keywords=["Anything"]))

        assert result["search_results"] == []


class TestItineraryAgent:
    """Unit tests for the third agent node (itinerary generation)."""

    def test_returns_itinerary_draft(self, mock_llm_text_response):
        """Agent should update 'itinerary_draft' in the returned state dict."""
        with patch("main.llm_hero") as mock_llm:
            mock_llm.invoke.return_value = mock_llm_text_response

            from main import itinerary_agent
            state = build_state(
                search_results=[{"name": "The British Museum", "address": "London", "coordinates": [-0.127, 51.519]}]
            )
            result = itinerary_agent(state)

        assert "itinerary_draft" in result
        assert len(result["itinerary_draft"]) > 0
        assert "Morning" in result["itinerary_draft"]

    def test_empty_search_results_still_generates_draft(self, mock_llm_text_response):
        """Even with no verified locations, the LLM should still produce a draft."""
        with patch("main.llm_hero") as mock_llm:
            mock_llm.invoke.return_value = mock_llm_text_response

            from main import itinerary_agent
            result = itinerary_agent(build_state(search_results=[]))

        assert "itinerary_draft" in result
        assert isinstance(result["itinerary_draft"], str)


class TestCheckFeedback:
    """Tests for the conditional edge function that routes the graph."""

    def test_with_feedback_routes_to_vibe_interpreter(self):
        """Non-empty feedback loops back to 'vibe_interpreter'."""
        from main import check_feedback
        state = build_state(user_feedback="More food spots please!")
        assert check_feedback(state) == "vibe_interpreter"

    def test_empty_string_feedback_routes_to_end(self):
        """Empty string feedback ends the graph (falsy check)."""
        from main import check_feedback
        state = build_state(user_feedback="")
        assert check_feedback(state) == END

    def test_none_feedback_routes_to_end(self):
        """None feedback ends the graph."""
        from main import check_feedback
        state = build_state(user_feedback=None)
        assert check_feedback(state) == END


class TestAwaitFeedback:
    """Tests for the passthrough await_feedback node."""

    def test_resets_user_feedback_to_none(self):
        """await_feedback resets user_feedback to None to pause the loop."""
        from main import await_feedback
        state = build_state(user_feedback="Some old feedback")
        result = await_feedback(state)
        assert result == {"user_feedback": None}
