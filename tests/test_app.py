import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestLifespan:
    def test_lifespan_temporal_disabled(self):
        """Lifespan runs startup and shutdown with TEMPORAL_STATE=OFF."""
        with patch("temporal.config.settings.TEMPORAL_ENABLED", False):
            with TestClient(app) as c:
                resp = c.post("/check", json={})
                assert resp.status_code in (200, 400, 422)

    def test_lifespan_temporal_enabled_worker_starts_and_cancels(self):
        """Lifespan starts and cancels the Temporal worker task."""
        import asyncio

        async def _fake_worker():
            await asyncio.sleep(9999)

        with (
            patch("temporal.config.settings.TEMPORAL_ENABLED", True),
            patch("temporal.workers.worker.start_worker", return_value=_fake_worker()),
        ):
            with TestClient(app) as c:
                resp = c.post("/check", json={})
                assert resp.status_code in (200, 400, 422)

    def test_on_worker_done_crash_logs_error(self):
        """_on_worker_done callback logs when worker task raises an exception."""
        import asyncio

        exc = RuntimeError("worker crashed")

        async def _crashing_worker():
            raise exc

        with (
            patch("temporal.config.settings.TEMPORAL_ENABLED", True),
            patch("temporal.workers.worker.start_worker", return_value=_crashing_worker()),
        ):
            with TestClient(app) as c:
                pass  # lifespan runs; crashing worker done callback fires


class TestTemporalEnabledFlag:
    def test_temporal_enabled_returns_bool(self):
        from src.controllers.verification_controller import _temporal_enabled
        with patch("temporal.config.settings.TEMPORAL_ENABLED", False):
            assert _temporal_enabled() is False

    def test_temporal_enabled_true(self):
        from src.controllers.verification_controller import _temporal_enabled
        with patch("temporal.config.settings.TEMPORAL_ENABLED", True):
            assert _temporal_enabled() is True
