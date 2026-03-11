from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app import app
from src.models.response_models import ReturnResponse

client = TestClient(app)


def _valid_payload() -> dict:
    return {
        "provider": "Bank of Baroda",
        "service_name": "BOB iBanking",
        "login_type": "Direct",
        "url": "https://example.com/login",
        "country": "India",
    }


def test_check_endpoint_requires_request_fields():
    response = client.post("/check", json={})

    assert response.status_code == 422


def test_check_endpoint_returns_service_response():
    mocked_response = ReturnResponse(
        url="https://example.com/login",
        inactive_flagged=False,
        reason="Page appears active.",
        health_check=True,
        page_match_score=82,
        direct_match_score=91,
        notes="langfuse_session_id=test-session",
        updated_name="Example Bank Login",
        marked_for_human_review=False,
        marked_for_deletion=False,
        errors="",
        time="2026-03-10T12:00:00",
    )

    with patch(
        "src.controllers.verification_controller.verify_url",
        new=AsyncMock(return_value=mocked_response),
    ):
        response = client.post("/check", json=_valid_payload())

    assert response.status_code == 200
    assert response.json() == mocked_response.model_dump()


def test_check_endpoint_maps_service_value_error_to_http_400():
    with patch(
        "src.controllers.verification_controller.verify_url",
        new=AsyncMock(side_effect=ValueError("URL must include a valid scheme and host.")),
    ):
        response = client.post("/check", json=_valid_payload())

    assert response.status_code == 400
    assert response.json() == {
        "detail": "URL must include a valid scheme and host."
    }
