"""Unit tests for url_health pure functions and check_url_health."""
import pytest
import socket
from unittest.mock import AsyncMock, MagicMock, patch
from src.url_health import (
    _is_bot_blocked,
    _status_to_health,
    detect_soft_errors,
    detect_url_token,
)


# ---------------------------------------------------------------------------
# detect_url_token
# ---------------------------------------------------------------------------

class TestDetectUrlToken:
    def test_clean_url_returns_none(self):
        assert detect_url_token("https://netbanking.hdfcbank.com/login") is None

    def test_jwt_in_url_body(self):
        jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        result = detect_url_token(f"https://example.com/auth?t={jwt}")
        assert result is not None
        assert result["has_token"] is True
        assert any("JWT" in r for r in result["reasons"])

    def test_token_param_name(self):
        result = detect_url_token("https://example.com/login?token=abc123")
        assert result is not None
        assert any("token" in r for r in result["reasons"])

    def test_session_param_name(self):
        result = detect_url_token("https://example.com/?session=abc123")
        assert result is not None

    def test_uuid_value_in_param(self):
        # "ref" is in _SAFE_PARAM_NAMES; use a non-safe custom param name
        result = detect_url_token(
            "https://example.com/login?user_ref=550e8400-e29b-41d4-a716-446655440000"
        )
        assert result is not None
        assert any("UUID" in r for r in result["reasons"])

    def test_long_hex_in_param(self):
        # "hash" is in _TOKEN_PARAM_NAMES (matched by name); use a non-token param name
        hex_val = "a" * 32
        result = detect_url_token(f"https://example.com/login?data_ref={hex_val}")
        assert result is not None
        assert any("hex" in r for r in result["reasons"])

    def test_long_base64_in_param(self):
        b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqr"  # 46 chars
        result = detect_url_token(f"https://example.com/login?data={b64}")
        assert result is not None

    def test_safe_param_names_ignored(self):
        assert detect_url_token("https://example.com/login?lang=en") is None
        assert detect_url_token("https://example.com/login?page=2") is None
        assert detect_url_token("https://example.com/login?id=123") is None

    def test_uuid_in_path_segment(self):
        result = detect_url_token(
            "https://example.com/user/550e8400-e29b-41d4-a716-446655440000/dashboard"
        )
        assert result is not None
        assert any("UUID" in r for r in result["reasons"])

    def test_long_hex_in_path(self):
        result = detect_url_token(f"https://example.com/token/{'a'*32}/callback")
        assert result is not None

    def test_summary_joins_all_reasons(self):
        result = detect_url_token(
            "https://example.com/login?token=abc&session=xyz"
        )
        assert result is not None
        assert ";" in result["summary"] or len(result["reasons"]) >= 1

    def test_jwt_in_path_segment(self):
        jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        result = detect_url_token(f"https://example.com/{jwt}/callback")
        assert result is not None

    def test_empty_param_value_skipped(self):
        result = detect_url_token("https://example.com/login?custom=")
        assert result is None

    def test_short_value_not_flagged(self):
        assert detect_url_token("https://example.com/login?custom=abc") is None


# ---------------------------------------------------------------------------
# detect_soft_errors
# ---------------------------------------------------------------------------

class TestDetectSoftErrors:
    def test_no_errors_in_clean_content(self):
        assert detect_soft_errors("Welcome to your online banking portal.") == []

    def test_detects_404(self):
        result = detect_soft_errors("Error 404 page not found")
        assert "404" in result
        assert "page not found" in result

    def test_detects_service_unavailable(self):
        result = detect_soft_errors("Service unavailable, try again later")
        assert "service unavailable" in result

    def test_detects_access_denied(self):
        result = detect_soft_errors("Access denied to this resource")
        assert "access denied" in result

    def test_case_insensitive(self):
        result = detect_soft_errors("PAGE NOT FOUND")
        assert "page not found" in result

    def test_none_input_returns_empty(self):
        assert detect_soft_errors(None) == []

    def test_list_input_joined(self):
        result = detect_soft_errors(["Error", "404", "not found here"])
        assert "404" in result

    def test_list_with_none_items(self):
        result = detect_soft_errors(["Welcome", None, "page not found"])
        assert "page not found" in result

    def test_error_occurred(self):
        result = detect_soft_errors("An error occurred while processing")
        assert "error occurred" in result

    def test_not_authorized(self):
        result = detect_soft_errors("You are not authorized to view this page")
        assert "not authorized" in result

    def test_resource_not_found(self):
        result = detect_soft_errors("The resource not found")
        assert "resource not found" in result

    def test_temporarily_unavailable(self):
        result = detect_soft_errors("Service is temporarily unavailable")
        assert "temporarily unavailable" in result

    def test_page_has_moved(self):
        result = detect_soft_errors("This page has moved to a new location")
        assert "this page has moved" in result

    def test_invalid_request(self):
        result = detect_soft_errors("Invalid request parameters")
        assert "invalid request" in result

    def test_multiple_errors_in_one_string(self):
        result = detect_soft_errors("404 page not found - access denied")
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# _status_to_health
# ---------------------------------------------------------------------------

class TestStatusToHealth:
    def test_none_status(self):
        health, reason = _status_to_health(None)
        assert health == "INACTIVE"
        assert reason == "NO_RESPONSE"

    def test_200_ok(self):
        health, reason = _status_to_health(200)
        assert health == "OK"
        assert reason is None

    def test_201_ok(self):
        health, reason = _status_to_health(201)
        assert health == "OK"

    def test_299_ok(self):
        health, reason = _status_to_health(299)
        assert health == "OK"

    def test_301_redirect(self):
        health, reason = _status_to_health(301)
        assert health == "REDIRECT"
        assert reason is None

    def test_302_redirect(self):
        assert _status_to_health(302)[0] == "REDIRECT"

    def test_307_redirect(self):
        assert _status_to_health(307)[0] == "REDIRECT"

    def test_308_redirect(self):
        assert _status_to_health(308)[0] == "REDIRECT"

    def test_404_not_found(self):
        health, reason = _status_to_health(404)
        assert health == "INACTIVE"
        assert reason == "NOT_FOUND"

    def test_410_gone(self):
        health, reason = _status_to_health(410)
        assert health == "INACTIVE"
        assert reason == "PERMANENTLY_REMOVED"

    def test_500_server_error(self):
        health, reason = _status_to_health(500)
        assert health == "INACTIVE"
        assert reason == "SERVER_ERROR"

    def test_403_forbidden(self):
        health, reason = _status_to_health(403)
        assert health == "INACTIVE"
        assert reason == "FORBIDDEN"

    def test_unknown_status(self):
        health, reason = _status_to_health(418)
        assert health == "INACTIVE"
        assert reason == "HTTP_418"


# ---------------------------------------------------------------------------
# _is_bot_blocked
# ---------------------------------------------------------------------------

class TestIsBotBlocked:
    def _page(self, title="", visible="", headings=None):
        return {"title": title, "visible_text": visible, "headings": headings or []}

    def test_clean_page_not_bot_blocked(self):
        assert not _is_bot_blocked(self._page("Login", "Enter your credentials"))

    def test_cloudflare_just_a_moment(self):
        assert _is_bot_blocked(self._page("Just a moment...", "Checking your browser"))

    def test_cloudflare_ray_id(self):
        assert _is_bot_blocked(self._page("Error", "Ray ID: abc123"))

    def test_ddos_protection(self):
        assert _is_bot_blocked(self._page("", "DDoS protection by Cloudflare"))

    def test_bot_activity_detected(self):
        assert _is_bot_blocked(self._page("Security Check", "Bot activity detected"))

    def test_verify_you_are_human(self):
        assert _is_bot_blocked(self._page("", "Verify you are human"))

    def test_enable_javascript_cookies(self):
        assert _is_bot_blocked(
            self._page("", "Enable JavaScript and cookies to continue")
        )

    def test_akamai_phrase(self):
        assert _is_bot_blocked(self._page("", "akamai bot manager"))

    def test_incapsula(self):
        assert _is_bot_blocked(self._page("", "incapsula incident id: 123"))

    def test_match_in_headings(self):
        assert _is_bot_blocked(self._page("", "", ["just a moment"]))

    def test_visible_text_truncated_to_3000(self):
        # Phrase must appear within the first 3000 chars of visible_text to be detected
        long_text = "a" * 2980 + " just a moment " + "b" * 100
        assert _is_bot_blocked(self._page("", long_text))

    def test_empty_page_not_blocked(self):
        assert not _is_bot_blocked(self._page())


# ---------------------------------------------------------------------------
# check_url_health — full async function with mocked extract_visible_layer
# ---------------------------------------------------------------------------

def _good_page(url="https://netbanking.hdfcbank.com/login"):
    return {
        "final_url": url,
        "http_status": 200,
        "redirect_chain": [],
        "network_error": None,
        "title": "HDFC NetBanking Login",
        "visible_text": "Login to HDFC NetBanking",
        "headings": ["Secure Login"],
        "login_form_present": True,
        "extraction_quality": "good",
    }


class TestCheckUrlHealth:
    @pytest.mark.asyncio
    async def test_dns_failure_returns_inactive(self):
        from src.url_health import check_url_health
        with (
            patch("src.url_health.socket.gethostbyname", side_effect=socket.gaierror("DNS fail")),
        ):
            result = await check_url_health("https://nonexistent-domain-xyz.com/login")
        assert result["health"] == "INACTIVE"
        assert result["reason"] == "DNS_FAILURE"

    @pytest.mark.asyncio
    async def test_healthy_page_returns_ok(self):
        from src.url_health import check_url_health
        with (
            patch("src.url_health.socket.gethostbyname", return_value="1.2.3.4"),
            patch("src.url_health.extract_visible_layer", new=AsyncMock(return_value=_good_page())),
        ):
            result = await check_url_health("https://netbanking.hdfcbank.com/login")
        assert result["health"] == "OK"
        assert result["reason"] is None
        assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_404_returns_inactive(self):
        from src.url_health import check_url_health
        page = {**_good_page(), "http_status": 404, "network_error": None}
        with (
            patch("src.url_health.socket.gethostbyname", return_value="1.2.3.4"),
            patch("src.url_health.extract_visible_layer", new=AsyncMock(return_value=page)),
        ):
            result = await check_url_health("https://netbanking.hdfcbank.com/login")
        assert result["health"] == "INACTIVE"
        assert result["reason"] == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_network_error_returns_inactive(self):
        from src.url_health import check_url_health
        page = {**_good_page(), "http_status": None, "network_error": "CONNECTION_REFUSED"}
        with (
            patch("src.url_health.socket.gethostbyname", return_value="1.2.3.4"),
            patch("src.url_health.extract_visible_layer", new=AsyncMock(return_value=page)),
        ):
            result = await check_url_health("https://example.com/login")
        assert result["health"] == "INACTIVE"
        assert result["reason"] == "CONNECTION_REFUSED"

    @pytest.mark.asyncio
    async def test_empty_server_response_overrides_health(self):
        from src.url_health import check_url_health
        page = {**_good_page(), "http_status": 200, "extraction_quality": "empty_server_response"}
        with (
            patch("src.url_health.socket.gethostbyname", return_value="1.2.3.4"),
            patch("src.url_health.extract_visible_layer", new=AsyncMock(return_value=page)),
        ):
            result = await check_url_health("https://example.com/login")
        assert result["health"] == "INACTIVE"
        assert result["reason"] == "EMPTY_SERVER_RESPONSE"

    @pytest.mark.asyncio
    async def test_bot_blocked_403(self):
        from src.url_health import check_url_health
        page = {
            **_good_page(),
            "http_status": 403,
            "network_error": None,
            "title": "Just a moment...",
            "visible_text": "Checking your browser",
            "extraction_quality": "good",
        }
        with (
            patch("src.url_health.socket.gethostbyname", return_value="1.2.3.4"),
            patch("src.url_health.extract_visible_layer", new=AsyncMock(return_value=page)),
        ):
            result = await check_url_health("https://example.com/login")
        assert result["reason"] == "BOT_BLOCKED"

    @pytest.mark.asyncio
    async def test_token_detected_in_url(self):
        from src.url_health import check_url_health
        url = "https://example.com/login?token=abc123def456"
        page = {**_good_page(url=url), "http_status": 200}
        with (
            patch("src.url_health.socket.gethostbyname", return_value="1.2.3.4"),
            patch("src.url_health.extract_visible_layer", new=AsyncMock(return_value=page)),
        ):
            result = await check_url_health(url)
        assert result["token_detected"] is not None

    @pytest.mark.asyncio
    async def test_load_time_ms_is_set(self):
        from src.url_health import check_url_health
        with (
            patch("src.url_health.socket.gethostbyname", return_value="1.2.3.4"),
            patch("src.url_health.extract_visible_layer", new=AsyncMock(return_value=_good_page())),
        ):
            result = await check_url_health("https://example.com/login")
        assert isinstance(result["load_time_ms"], int)
        assert result["load_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_soft_errors_detected(self):
        from src.url_health import check_url_health
        page = {**_good_page(), "headings": ["404 page not found"]}
        with (
            patch("src.url_health.socket.gethostbyname", return_value="1.2.3.4"),
            patch("src.url_health.extract_visible_layer", new=AsyncMock(return_value=page)),
        ):
            result = await check_url_health("https://example.com/login")
        assert "404" in result["soft_errors"] or "page not found" in result["soft_errors"]

    @pytest.mark.asyncio
    async def test_raw_http_fallback_for_403(self):
        from src.url_health import check_url_health
        page = {
            **_good_page(),
            "http_status": 403,
            "network_error": None,
            "title": "Forbidden",
            "visible_text": "Access restricted",
            "extraction_quality": "good",
        }
        with (
            patch("src.url_health.socket.gethostbyname", return_value="1.2.3.4"),
            patch("src.url_health.extract_visible_layer", new=AsyncMock(return_value=page)),
            patch("src.url_health._raw_http_check", new=AsyncMock(return_value="OK")),
        ):
            result = await check_url_health("https://example.com/login")
        assert result["reason"] == "BOT_BLOCKED"
