"""Tests for src/services/analysis_service.py — all LLM calls mocked."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.models.request_models import CheckRequest


def _make_request(provider="HDFC Bank", service_name="NetBanking") -> CheckRequest:
    return CheckRequest(
        provider=provider,
        service_name=service_name,
        login_type="Direct",
        url="https://netbanking.hdfcbank.com/login",
        country="India",
        cb_link_id="B-IN-test1",
    )


def _page_result() -> dict:
    return {
        "final_url": "https://netbanking.hdfcbank.com/login",
        "title": "HDFC NetBanking Login",
        "headings": ["Login", "Sign In"],
        "visible_text": "Welcome to HDFC Bank NetBanking",
        "buttons": ["Login"],
        "login_form_present": True,
    }


class TestRunChecks:
    @pytest.mark.asyncio
    async def test_returns_none_when_provider_and_service_both_empty(self):
        """run_checks short-circuits and returns None if both fields are blank."""
        from src.services.analysis_service import run_checks
        payload = _make_request(provider="", service_name="")
        result = await run_checks(payload, "https://x.com", _page_result(), session_id="s1")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_only_provider_empty_and_service_empty(self):
        """Edge: provider non-empty but service_name empty — should still run."""
        mock_match = AsyncMock(return_value={
            "bank_matched": True,
            "service_matched": True,
            "confidence_score": 80,
            "url_confidence_score": 75,
            "login_type": "direct",
            "reason": "Matched",
            "notes": [],
        })
        with patch("src.services.analysis_service.assess_full_match", mock_match):
            from src.services.analysis_service import run_checks
            payload = _make_request(provider="HDFC Bank", service_name="")
            result = await run_checks(payload, "https://x.com", _page_result(), session_id="s1")
        assert result is not None
        assert result["bank_matched"] is True

    @pytest.mark.asyncio
    async def test_delegates_to_assess_full_match(self):
        """run_checks passes correct args to assess_full_match."""
        expected = {
            "bank_matched": True,
            "service_matched": True,
            "confidence_score": 90,
            "url_confidence_score": 88,
            "login_type": "direct",
            "reason": "Perfect match",
            "notes": ["langfuse_session_id=s1"],
        }
        mock_match = AsyncMock(return_value=expected)
        with patch("src.services.analysis_service.assess_full_match", mock_match):
            from src.services.analysis_service import run_checks
            payload = _make_request()
            result = await run_checks(payload, "https://netbanking.hdfcbank.com/login", _page_result(), session_id="s1")
        assert result == expected
        mock_match.assert_called_once_with(
            provider="HDFC Bank",
            service_name="NetBanking",
            url="https://netbanking.hdfcbank.com/login",
            page_result=_page_result(),
            session_id="s1",
        )

    @pytest.mark.asyncio
    async def test_returns_none_when_service_empty_and_provider_empty(self):
        from src.services.analysis_service import run_checks
        payload = _make_request(provider="", service_name="")
        result = await run_checks(payload, "https://x.com", {})
        assert result is None


class TestClassifyPageAudience:
    @pytest.mark.asyncio
    async def test_delegates_to_classify_customer_facing(self):
        """classify_page_audience must call classify_customer_facing with correct args."""
        expected = {
            "is_customer_facing": True,
            "confidence": 95,
            "category": "customer_login",
            "reason": "Login page detected",
            "notes": [],
        }
        mock_classify = AsyncMock(return_value=expected)
        with patch("src.services.analysis_service.classify_customer_facing", mock_classify):
            from src.services.analysis_service import classify_page_audience
            payload = _make_request()
            result = await classify_page_audience(
                payload, "https://netbanking.hdfcbank.com/login", _page_result(), session_id="s2"
            )
        assert result == expected
        mock_classify.assert_called_once_with(
            provider="HDFC Bank",
            service_name="NetBanking",
            url="https://netbanking.hdfcbank.com/login",
            page_result=_page_result(),
            session_id="s2",
        )

    @pytest.mark.asyncio
    async def test_empty_provider_passes_empty_string(self):
        """When payload.provider is None/empty, classify_customer_facing receives ''."""
        mock_classify = AsyncMock(return_value={
            "is_customer_facing": True, "confidence": 0, "category": "unknown",
            "reason": "skipped", "notes": [],
        })
        with patch("src.services.analysis_service.classify_customer_facing", mock_classify):
            from src.services.analysis_service import classify_page_audience
            payload = _make_request(provider="", service_name="")
            result = await classify_page_audience(payload, "https://x.com", {})
        assert result["is_customer_facing"] is True
        call_kwargs = mock_classify.call_args.kwargs
        assert call_kwargs["provider"] == ""
        assert call_kwargs["service_name"] == ""

    @pytest.mark.asyncio
    async def test_always_returns_dict(self):
        """classify_page_audience never returns None."""
        mock_classify = AsyncMock(return_value={
            "is_customer_facing": True, "confidence": 0,
            "category": "unknown", "reason": "x", "notes": [],
        })
        with patch("src.services.analysis_service.classify_customer_facing", mock_classify):
            from src.services.analysis_service import classify_page_audience
            result = await classify_page_audience(_make_request(), "https://x.com", {})
        assert isinstance(result, dict)
