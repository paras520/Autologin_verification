"""Tests for src/services/m103_ingest.py — all external HTTP mocked."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.models.response_models import BatchCheckResponse, CbLinkResult, RowResult


def _make_batch_response() -> BatchCheckResponse:
    row = RowResult(
        cb_link_id="B-IN-test1",
        service_id="svc-uuid-1",
        login_service="NetBanking",
        url="https://netbanking.example.com/login",
        health_check=True,
        page_match_score=85,
        direct_match_score=90,
        display_name_score=None,
        notes="ok",
        inactive_flagged=False,
        marked_for_deletion=False,
        marked_for_human_review=False,
        is_duplicate=False,
        duplicate_of_url=None,
        reason="Match confirmed",
        status="active",
    )
    cb_result = CbLinkResult(cb_link_id="B-IN-test1", total_rows=1, rows=[row])
    return BatchCheckResponse(results=[cb_result])


class TestIngestToM103:
    @pytest.mark.asyncio
    async def test_success_path(self):
        """POST succeeds, raise_for_status does not raise, logs run_id/inserted/flagged."""
        batch = _make_batch_response()

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()  # does not raise
        mock_resp.json = MagicMock(return_value={"data": {"run_id": "r1", "inserted": 5, "flagged": 1}})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            from src.services.m103_ingest import ingest_to_m103
            # Should complete without raising
            await ingest_to_m103("run-uuid-1", batch, triggered_by="test_suite")

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert "json" in call_kwargs.kwargs
        payload = call_kwargs.kwargs["json"]
        assert payload["run_mode"] == "batch"
        assert payload["triggered_by"] == "test_suite"
        assert len(payload["results"]) == 1

    @pytest.mark.asyncio
    async def test_success_path_default_triggered_by(self):
        """When triggered_by is None, falls back to 'autologin_verification'."""
        batch = _make_batch_response()

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={"data": {}})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            from src.services.m103_ingest import ingest_to_m103
            await ingest_to_m103("run-uuid-2", batch, triggered_by=None)

        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["json"]["triggered_by"] == "autologin_verification"

    @pytest.mark.asyncio
    async def test_http_status_error_is_non_fatal(self):
        """HTTPStatusError is caught and logged; function does not re-raise."""
        batch = _make_batch_response()

        mock_response_obj = MagicMock()
        mock_response_obj.status_code = 502
        mock_response_obj.text = "Bad Gateway"
        error = httpx.HTTPStatusError(
            "502 Bad Gateway",
            request=MagicMock(),
            response=mock_response_obj,
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock(side_effect=error)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            from src.services.m103_ingest import ingest_to_m103
            # Must not raise
            await ingest_to_m103("run-uuid-3", batch, triggered_by="test")

    @pytest.mark.asyncio
    async def test_generic_exception_is_non_fatal(self):
        """Any unexpected exception is caught and logged; function does not re-raise."""
        batch = _make_batch_response()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            from src.services.m103_ingest import ingest_to_m103
            # Must not raise
            await ingest_to_m103("run-uuid-4", batch, triggered_by="test")

    @pytest.mark.asyncio
    async def test_payload_contains_meta_activity_run_id(self):
        """Payload must embed activity_run_id in meta."""
        batch = _make_batch_response()

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={"data": {}})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            from src.services.m103_ingest import ingest_to_m103
            await ingest_to_m103("specific-run-id", batch, triggered_by="test")

        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["meta"]["activity_run_id"] == "specific-run-id"
