"""Unit tests for VerificationController — all external services mocked."""
import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, patch, MagicMock

from src.controllers.verification_controller import VerificationController
from src.models.request_models import BatchCheckRequest, CheckRequest
from src.models.response_models import (
    AsyncBatchResponse,
    BatchCheckResponse,
    CbLinkResult,
    ReturnResponse,
    RowResult,
)


def _check_payload() -> CheckRequest:
    return CheckRequest(
        provider="HDFC Bank",
        service_name="NetBanking",
        login_type="Direct",
        url="https://netbanking.hdfcbank.com/login",
        country="India",
        cb_link_id="test-001",
    )


def _mock_return_response() -> ReturnResponse:
    return ReturnResponse(
        url="https://netbanking.hdfcbank.com/login",
        inactive_flagged=False,
        reason="Matches",
        health_check=True,
        page_match_score=88,
        direct_match_score=90,
        notes=None,
        updated_name=None,
        marked_for_human_review=False,
        marked_for_deletion=False,
        errors="",
        time="2026-01-01T00:00:00",
    )


class TestHandleRequest:
    @pytest.mark.asyncio
    async def test_returns_service_response(self):
        controller = VerificationController()
        mock_resp = _mock_return_response()
        with patch(
            "src.controllers.verification_controller.verify_url",
            new=AsyncMock(return_value=mock_resp),
        ):
            result = await controller.handle_request(_check_payload())
        assert result == mock_resp

    @pytest.mark.asyncio
    async def test_value_error_raises_400(self):
        controller = VerificationController()
        with patch(
            "src.controllers.verification_controller.verify_url",
            new=AsyncMock(side_effect=ValueError("bad url")),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await controller.handle_request(_check_payload())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_unexpected_error_raises_500(self):
        controller = VerificationController()
        with patch(
            "src.controllers.verification_controller.verify_url",
            new=AsyncMock(side_effect=RuntimeError("crash")),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await controller.handle_request(_check_payload())
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_http_exception_propagated(self):
        controller = VerificationController()
        with patch(
            "src.controllers.verification_controller.verify_url",
            new=AsyncMock(side_effect=HTTPException(status_code=503, detail="upstream")),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await controller.handle_request(_check_payload())
        assert exc_info.value.status_code == 503


class TestHandleBatchInline:
    def _batch_payload(self) -> BatchCheckRequest:
        return BatchCheckRequest(cb_link_ids=["B-IN-test01"], include_inactive=True)

    def _mock_rows(self):
        return [
            {
                "id": "row-1",
                "cb_link_id": "B-IN-test01",
                "login_service": "NetBanking",
                "login_url": "https://netbanking.hdfcbank.com/login",
                "display_name": "HDFC Bank",
                "sorting_order": 1,
                "status": "active",
                "is_duplicate": False,
            }
        ]

    @pytest.mark.asyncio
    async def test_batch_inline_returns_response(self):
        controller = VerificationController()
        rows = self._mock_rows()
        mock_verify = AsyncMock(return_value=_mock_return_response())

        with (
            patch("src.controllers.verification_controller._temporal_enabled", return_value=False),
            patch("src.controllers.verification_controller.fetch_rows", new=AsyncMock(return_value=rows)),
            patch("src.controllers.verification_controller.detect_duplicates", return_value=rows),
            patch("src.controllers.verification_controller.verify_row", new=mock_verify),
            patch("src.controllers.verification_controller.resolve_same_name_duplicates", side_effect=lambda x: x),
            patch("src.controllers.verification_controller.resolve_parent_child", side_effect=lambda x: x),
            patch("src.controllers.verification_controller.ingest_to_m103", new=AsyncMock()),
            patch("src.controllers.verification_controller.create_activity_run", new=AsyncMock(return_value="run-1")),
            patch("src.controllers.verification_controller.start_activity_run", new=AsyncMock()),
            patch("src.controllers.verification_controller.upsert_run_item_result", new=AsyncMock()),
            patch("src.controllers.verification_controller.finalize_activity_run", new=AsyncMock()),
        ):
            result = await controller.handle_batch(self._batch_payload())

        assert isinstance(result, BatchCheckResponse)
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_batch_error_raises_500(self):
        controller = VerificationController()
        with (
            patch("src.controllers.verification_controller._temporal_enabled", return_value=False),
            patch("src.controllers.verification_controller.fetch_rows", new=AsyncMock(side_effect=RuntimeError("db down"))),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await controller.handle_batch(self._batch_payload())
        assert exc_info.value.status_code == 500


class TestHandleBatchAsync:
    @pytest.mark.asyncio
    async def test_async_batch_returns_response(self):
        controller = VerificationController()
        payload = BatchCheckRequest(cb_link_ids=["B-IN-test01"])

        with (
            patch("src.controllers.verification_controller._temporal_enabled", return_value=False),
            patch("src.controllers.verification_controller.create_activity_run", new=AsyncMock(return_value="run-123")),
        ):
            result = await controller.handle_batch_async(payload, MagicMock())

        assert isinstance(result, AsyncBatchResponse)
        assert result.run_id == "run-123"
        assert result.status == "queued"

    @pytest.mark.asyncio
    async def test_async_batch_db_failure_raises_500(self):
        controller = VerificationController()
        payload = BatchCheckRequest(cb_link_ids=["B-IN-test01"])

        with (
            patch("src.controllers.verification_controller._temporal_enabled", return_value=False),
            patch("src.controllers.verification_controller.create_activity_run", new=AsyncMock(side_effect=RuntimeError("db down"))),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await controller.handle_batch_async(payload, MagicMock())
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_async_batch_total_links_count(self):
        controller = VerificationController()
        payload = BatchCheckRequest(cb_link_ids=["B-IN-001", "B-IN-002", "B-IN-003"])

        with (
            patch("src.controllers.verification_controller._temporal_enabled", return_value=False),
            patch("src.controllers.verification_controller.create_activity_run", new=AsyncMock(return_value="run-456")),
        ):
            result = await controller.handle_batch_async(payload, MagicMock())

        assert result.total_links == 3


class TestRunVerificationBackground:
    def _mock_rows(self):
        return [
            {
                "id": "row-1",
                "cb_link_id": "B-IN-test01",
                "login_service": "NetBanking",
                "login_url": "https://netbanking.hdfcbank.com/login",
                "display_name": "HDFC Bank",
                "sorting_order": 1,
                "status": "active",
                "is_duplicate": False,
            }
        ]

    @pytest.mark.asyncio
    async def test_background_run_completes(self):
        controller = VerificationController()
        payload = BatchCheckRequest(cb_link_ids=["B-IN-test01"])
        rows = self._mock_rows()

        with (
            patch("src.controllers.verification_controller._temporal_enabled", return_value=False),
            patch("src.controllers.verification_controller.fetch_rows", new=AsyncMock(return_value=rows)),
            patch("src.controllers.verification_controller.detect_duplicates", return_value=rows),
            patch("src.controllers.verification_controller.verify_row", new=AsyncMock(return_value=_mock_return_response())),
            patch("src.controllers.verification_controller.resolve_same_name_duplicates", side_effect=lambda x: x),
            patch("src.controllers.verification_controller.resolve_parent_child", side_effect=lambda x: x),
            patch("src.controllers.verification_controller.start_activity_run", new=AsyncMock()),
            patch("src.controllers.verification_controller.upsert_run_item_result", new=AsyncMock()),
            patch("src.controllers.verification_controller.finalize_activity_run", new=AsyncMock()),
            patch("src.controllers.verification_controller.ingest_to_m103", new=AsyncMock()),
        ):
            await controller._run_verification_background("run-001", payload)

    @pytest.mark.asyncio
    async def test_background_run_handles_inline_failure(self):
        controller = VerificationController()
        payload = BatchCheckRequest(cb_link_ids=["B-IN-test01"])

        with (
            patch("src.controllers.verification_controller._temporal_enabled", return_value=False),
            patch("src.controllers.verification_controller.fetch_rows", new=AsyncMock(side_effect=RuntimeError("db crash"))),
            patch("src.controllers.verification_controller.start_activity_run", new=AsyncMock()),
            patch("src.controllers.verification_controller.finalize_activity_run", new=AsyncMock()),
        ):
            # Should not raise — errors are swallowed in background tasks
            await controller._run_verification_background("run-001", payload)
