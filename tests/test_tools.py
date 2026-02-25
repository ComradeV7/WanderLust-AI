"""Unit tests for query_places_nominatim and get_ors_directions in main.py.
All external calls (Nominatim, ORS) are mocked so tests run offline.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestQueryPlacesNominatim:
    """Tests for the dynamic geocoding tool."""

    def test_megacity_strict_search_success(self, fake_megacity_location, fake_place_location):
        """Megacity (importance > 0.75) uses 30km radius; strict search returns result immediately."""
        with patch("main.geolocator") as mock_geo:
            mock_geo.geocode.side_effect = [fake_megacity_location, fake_place_location]

            from main import query_places_nominatim
            result = query_places_nominatim.invoke({"query": "The British Museum", "location_name": "London, UK"})

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "The British Museum"
        assert "coordinates" in result[0]
        assert result[0]["coordinates"] == [fake_place_location.longitude, fake_place_location.latitude]

    def test_regional_hub_uses_larger_radius(self, fake_regional_location, fake_place_location):
        """Regional hub (importance <= 0.75) uses 200km radius; strict search returns result."""
        with patch("main.geolocator") as mock_geo:
            mock_geo.geocode.side_effect = [fake_regional_location, fake_place_location]

            from main import query_places_nominatim
            result = query_places_nominatim.invoke({"query": "Beach", "location_name": "Visakhapatnam"})

        assert isinstance(result, list)
        assert len(result) == 1

    def test_city_not_found_returns_empty_list(self):
        """If the city can't be geocoded, return an empty list gracefully."""
        with patch("main.geolocator") as mock_geo:
            mock_geo.geocode.return_value = None

            from main import query_places_nominatim
            result = query_places_nominatim.invoke({"query": "Beach", "location_name": "NonExistentCity"})

        assert result == []

    def test_fallback_to_global_search_when_strict_fails(self, fake_megacity_location, fake_place_location):
        """If strict search returns None, fall back to global search with distance filtering."""
        nearby_candidate = MagicMock()
        nearby_candidate.latitude = 51.5194   # ~1.5km from London center
        nearby_candidate.longitude = -0.1270
        nearby_candidate.address = "The British Museum, London, UK"

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return fake_megacity_location
            elif call_count[0] == 2:
                return None  # strict search fails
            else:
                return [nearby_candidate]

        with patch("main.geolocator") as mock_geo:
            mock_geo.geocode.side_effect = side_effect

            from main import query_places_nominatim
            result = query_places_nominatim.invoke({"query": "The British Museum", "location_name": "London, UK"})

        assert isinstance(result, list)
        assert len(result) == 1

    def test_global_search_rejects_distant_candidates(self, fake_megacity_location):
        """Candidates beyond the city radius are rejected; returns empty list."""
        distant_candidate = MagicMock()
        distant_candidate.latitude = 48.8606   # Paris, ~340km from London
        distant_candidate.longitude = 2.3376
        distant_candidate.address = "Louvre Museum, Paris, France"

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return fake_megacity_location
            elif call_count[0] == 2:
                return None
            else:
                return [distant_candidate]

        with patch("main.geolocator") as mock_geo:
            mock_geo.geocode.side_effect = side_effect

            from main import query_places_nominatim
            result = query_places_nominatim.invoke({"query": "Louvre", "location_name": "London, UK"})

        assert result == []

    def test_tool_error_returns_empty_list(self):
        """Unexpected exceptions return [] without crashing."""
        with patch("main.geolocator") as mock_geo:
            mock_geo.geocode.side_effect = Exception("Network timeout")

            from main import query_places_nominatim
            result = query_places_nominatim.invoke({"query": "anything", "location_name": "London"})

        assert result == []


class TestGetOrsDirections:
    """Tests for the ORS routing tool."""

    def test_success_returns_duration_and_distance(self, mock_ors_directions_response):
        """Valid ORS response returns duration_minutes and distance_km."""
        with patch("main.ors_client") as mock_client:
            mock_client.directions.return_value = mock_ors_directions_response

            from main import get_ors_directions
            result = get_ors_directions.invoke({
                "start_coords": [-0.1270, 51.5194],
                "end_coords": [-0.0901, 51.5055],
                "profile": "foot-walking"
            })

        assert isinstance(result, dict)
        assert result["duration_minutes"] == 25.0   # 1500s / 60
        assert result["distance_km"] == 2.3         # 2300m / 1000

    def test_no_api_key_returns_graceful_note(self):
        """When ors_client is None (missing key), return a note instead of crashing."""
        with patch("main.ors_client", None):
            from main import get_ors_directions
            result = get_ors_directions.invoke({
                "start_coords": [-0.1270, 51.5194],
                "end_coords": [-0.0901, 51.5055],
            })

        assert "note" in result
        assert result["duration_minutes"] == 0
        assert result["distance_km"] == 0

    def test_ors_api_error_returns_error_string(self):
        """ORS exceptions (e.g. rate limit) return an error string."""
        with patch("main.ors_client") as mock_client:
            mock_client.directions.side_effect = Exception("403 Forbidden")

            from main import get_ors_directions
            result = get_ors_directions.invoke({
                "start_coords": [-0.1270, 51.5194],
                "end_coords": [-0.0901, 51.5055],
            })

        assert "Error" in str(result)

    def test_no_routes_in_response(self):
        """Empty routes list returns 'No directions found.'"""
        with patch("main.ors_client") as mock_client:
            mock_client.directions.return_value = {"routes": []}

            from main import get_ors_directions
            result = get_ors_directions.invoke({
                "start_coords": [-0.1270, 51.5194],
                "end_coords": [-0.0901, 51.5055],
            })

        assert result == "No directions found."
