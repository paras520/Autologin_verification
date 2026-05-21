"""Unit tests for verification_service — all external calls mocked."""
import pytest
from unittest.mock import AsyncMock, patch
from src.models.request_models import CheckRequest
from src.services.verification_service import _is_valid_url, verify_url


def _payload(**overrides) -> CheckRequest:
    defaults = dict(
        provider="HDFC Bank",
        service_name="NetBanking",
        login_type="Direct",
        url="https://netbanking.hdfcbank.com/login",
        country="India",
        cb_link_id="test-cb-001",
    )
    defaults.update(overrides)
    return CheckRequest(**defaults)


def _health_ok(url="https://netbanking.hdfcbank.com/login"):
    return {
        "health": "OK",
        "status": 200,
        "reason": None,
        "load_time_ms": 350,
        "soft_errors": [],
        "page_result": {
            "final_url": url,
            "title": "HDFC NetBanking Login",
            "visible_text": "Login to HDFC NetBanking",
            "headings": ["Secure Login"],
            "buttons": ["Login"],
            "login_form_present": True,
        },
    }


def _health_fail(reason="NOT_FOUND"):
    return {
        "health": "INACTIVE",
        "status": 404,
        "reason": reason,
        "load_time_ms": 100,
        "soft_errors": [],
        "page_result": {},
    }


# ---------------------------------------------------------------------------
# _is_valid_url
# ---------------------------------------------------------------------------

class TestIsValidUrl:
    def test_valid_https(self):
        assert _is_valid_url("https://example.com/login") is True

    def test_valid_http(self):
        assert _is_valid_url("http://example.com") is True

    def test_no_scheme(self):
        assert _is_valid_url("example.com/login") is False

    def test_no_netloc(self):
        assert _is_valid_url("https://") is False

    def test_empty(self):
        assert _is_valid_url("") is False


# ---------------------------------------------------------------------------
# verify_url — early exit paths
# ---------------------------------------------------------------------------

class TestVerifyUrlEmptyUrl:
    @pytest.mark.asyncio
    async def test_empty_url_raises_value_error(self):
        payload = _payload(url="   ")
        with pytest.raises(ValueError, match="non-empty"):
            await verify_url(payload)

    @pytest.mark.asyncio
    async def test_invalid_url_raises_value_error(self):
        payload = _payload(url="not-a-url")
        with pytest.raises(ValueError, match="scheme"):
            await verify_url(payload)


class TestVerifyUrlHealthFail:
    @pytest.mark.asyncio
    async def test_health_fail_returns_inactive(self):
        payload = _payload()
        with patch("src.services.verification_service.check_url_health", new=AsyncMock(return_value=_health_fail())):
            result = await verify_url(payload)
        assert result.inactive_flagged is True
        assert result.health_check is False
        assert result.marked_for_deletion is True

    @pytest.mark.asyncio
    async def test_empty_server_response_not_deleted(self):
        payload = _payload()
        with patch(
            "src.services.verification_service.check_url_health",
            new=AsyncMock(return_value=_health_fail("EMPTY_SERVER_RESPONSE")),
        ):
            result = await verify_url(payload)
        assert result.inactive_flagged is True
        assert result.marked_for_deletion is False

    @pytest.mark.asyncio
    async def test_bot_blocked_not_deleted(self):
        payload = _payload()
        with patch(
            "src.services.verification_service.check_url_health",
            new=AsyncMock(return_value=_health_fail("BOT_BLOCKED")),
        ):
            result = await verify_url(payload)
        assert result.marked_for_deletion is False

    @pytest.mark.asyncio
    async def test_timeout_not_deleted(self):
        payload = _payload()
        with patch(
            "src.services.verification_service.check_url_health",
            new=AsyncMock(return_value=_health_fail("TIMEOUT")),
        ):
            result = await verify_url(payload)
        assert result.marked_for_deletion is False


class TestVerifyUrlAudienceShortCircuit:
    @pytest.mark.asyncio
    async def test_non_customer_facing_high_confidence_deleted(self):
        payload = _payload(country="")
        audience = {
            "is_customer_facing": False,
            "confidence": 90,
            "category": "HRMS",
            "reason": "HR portal",
            "notes": [],
        }
        with (
            patch("src.services.verification_service.check_url_health", new=AsyncMock(return_value=_health_ok())),
            patch("src.services.verification_service.classify_page_audience", new=AsyncMock(return_value=audience)),
        ):
            result = await verify_url(payload)
        assert result.inactive_flagged is True
        assert result.marked_for_deletion is True

    @pytest.mark.asyncio
    async def test_non_customer_facing_low_confidence_routed_to_review(self):
        payload = _payload(country="")
        audience = {
            "is_customer_facing": False,
            "confidence": 50,
            "category": "HRMS",
            "reason": "Uncertain",
            "notes": [],
        }
        with (
            patch("src.services.verification_service.check_url_health", new=AsyncMock(return_value=_health_ok())),
            patch(
                "src.services.verification_service.classify_page_audience",
                new=AsyncMock(return_value=audience),
            ),
            patch(
                "src.services.verification_service.run_checks",
                new=AsyncMock(return_value={"bank_matched": True, "service_matched": True, "reason": "ok", "notes": [], "confidence_score": 80, "url_confidence_score": 85, "login_type": "Direct"}),
            ),
        ):
            result = await verify_url(payload)
        assert result.marked_for_human_review is True


class TestVerifyUrlFullPipeline:
    def _full_mocks(self, match_result=None, audience=None, country="India"):
        audience = audience or {
            "is_customer_facing": True, "confidence": 95,
            "category": "banking", "reason": "login page", "notes": [],
        }
        match_result = match_result or {
            "bank_matched": True, "service_matched": True,
            "reason": "Matches", "notes": [], "confidence_score": 88,
            "url_confidence_score": 90, "login_type": "Direct",
        }
        return audience, match_result

    @pytest.mark.asyncio
    async def test_full_pass_not_flagged(self):
        payload = _payload(country="")
        audience, match = self._full_mocks()
        with (
            patch("src.services.verification_service.check_url_health", new=AsyncMock(return_value=_health_ok())),
            patch("src.services.verification_service.classify_page_audience", new=AsyncMock(return_value=audience)),
            patch("src.services.verification_service.run_checks", new=AsyncMock(return_value=match)),
        ):
            result = await verify_url(payload)
        assert result.inactive_flagged is False
        assert result.marked_for_deletion is False

    @pytest.mark.asyncio
    async def test_bank_mismatch_flagged(self):
        payload = _payload(country="")
        audience, _ = self._full_mocks()
        bad_match = {
            "bank_matched": False, "service_matched": True,
            "reason": "Wrong bank", "notes": [], "confidence_score": 10,
            "url_confidence_score": 5, "login_type": "Direct",
        }
        with (
            patch("src.services.verification_service.check_url_health", new=AsyncMock(return_value=_health_ok())),
            patch("src.services.verification_service.classify_page_audience", new=AsyncMock(return_value=audience)),
            patch("src.services.verification_service.run_checks", new=AsyncMock(return_value=bad_match)),
        ):
            result = await verify_url(payload)
        assert result.inactive_flagged is True
        assert result.marked_for_deletion is True

    @pytest.mark.asyncio
    async def test_response_has_all_fields(self):
        payload = _payload(country="")
        audience, match = self._full_mocks()
        with (
            patch("src.services.verification_service.check_url_health", new=AsyncMock(return_value=_health_ok())),
            patch("src.services.verification_service.classify_page_audience", new=AsyncMock(return_value=audience)),
            patch("src.services.verification_service.run_checks", new=AsyncMock(return_value=match)),
        ):
            result = await verify_url(payload)
        assert result.url is not None
        assert result.time is not None
        assert isinstance(result.errors, str)
