import logging
import re
import socket
import time
from urllib.parse import parse_qs, urlparse

import httpx

from src.page_extraction import extract_visible_layer, MAX_LOAD_TIME, _STEALTH_UA

logger = logging.getLogger("autologin.url_health")

_TOKEN_PARAM_NAMES = re.compile(
    r"^(token|auth|session|sess|sid|key|apikey|api_key|access_token|"
    r"refresh_token|id_token|csrf|xsrf|nonce|otp|ticket|code|sso|"
    r"jsessionid|phpsessid|asp\.net_sessionid|__start_tran_flag__|"
    r"saml|assertion|bearer|signature|sig|hmac|hash|digest)$",
    re.IGNORECASE,
)

_JWT_PATTERN = re.compile(
    r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
)

_LONG_HEX = re.compile(r"[0-9a-fA-F]{32,}")

_LONG_BASE64 = re.compile(r"[A-Za-z0-9+/=_-]{40,}")

_UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

_SAFE_PARAM_NAMES = {
    "url", "redirect", "redirect_uri", "return", "returnurl",
    "next", "callback", "continue", "goto", "destination", "ref",
    "lang", "language", "locale", "hl", "page", "id", "type",
    "action", "event", "mode", "view", "tab", "format",
    "bank_id", "language_id",
}


def detect_url_token(url: str) -> dict | None:
    """Check if the URL contains an embedded token that may expire.

    Returns a dict with token details if found, or None if clean.
    """
    parsed = urlparse(url)
    reasons: list[str] = []

    full_url = url

    if _JWT_PATTERN.search(full_url):
        reasons.append("URL contains a JWT token")

    query_params = parse_qs(parsed.query, keep_blank_values=True)
    for param_name, values in query_params.items():
        if param_name.lower() in _SAFE_PARAM_NAMES:
            continue

        if _TOKEN_PARAM_NAMES.match(param_name):
            reasons.append(f"query parameter '{param_name}' is a known token/session key")
            continue

        for val in values:
            if not val:
                continue
            if _UUID.fullmatch(val):
                reasons.append(
                    f"query parameter '{param_name}' contains a UUID ({val[:36]})"
                )
            elif _JWT_PATTERN.search(val):
                reasons.append(
                    f"query parameter '{param_name}' contains a JWT"
                )
            elif _LONG_HEX.fullmatch(val):
                reasons.append(
                    f"query parameter '{param_name}' contains a long hex string (len={len(val)})"
                )
            elif len(val) >= 40 and _LONG_BASE64.fullmatch(val):
                reasons.append(
                    f"query parameter '{param_name}' contains a long opaque token (len={len(val)})"
                )

    path_segments = [s for s in parsed.path.split("/") if s]
    for segment in path_segments:
        if _JWT_PATTERN.search(segment):
            reasons.append(f"path segment contains a JWT")
        elif _UUID.fullmatch(segment):
            reasons.append(f"path segment contains a UUID ({segment[:36]})")
        elif _LONG_HEX.fullmatch(segment):
            reasons.append(f"path segment contains a long hex token (len={len(segment)})")

    if not reasons:
        return None

    return {
        "has_token": True,
        "reasons": reasons,
        "summary": "; ".join(reasons),
    }


def detect_soft_errors(content):
    error_indicators = [
        "404",
        "page not found",
        "service unavailable",
        "temporarily unavailable",
        "invalid request",
        "resource not found",
        "this page has moved",
        "error occurred",
        "not authorized",
        "access denied",
    ]

    if isinstance(content, list):
        normalized_content = " ".join(
            str(item) for item in content if item is not None
        )
    elif content is None:
        normalized_content = ""
    else:
        normalized_content = str(content)

    visible_text_lower = normalized_content.lower()

    found_errors = [
        phrase for phrase in error_indicators if phrase in visible_text_lower
    ]

    return found_errors


def _status_to_health(status: int | None) -> tuple[str, str | None]:
    """Map an HTTP status code to (health, reason)."""
    if status is None:
        return "INACTIVE", "NO_RESPONSE"
    if 200 <= status < 300:
        return "OK", None
    if status in (301, 302, 303, 307, 308):
        return "REDIRECT", None
    if status == 404:
        return "INACTIVE", "NOT_FOUND"
    if status == 410:
        return "INACTIVE", "PERMANENTLY_REMOVED"
    if status in (500, 502, 503):
        return "INACTIVE", "SERVER_ERROR"
    if status == 403:
        return "INACTIVE", "FORBIDDEN"
    return "INACTIVE", f"HTTP_{status}"


# ---------------------------------------------------------------------------
# Bot-block detection helpers
# ---------------------------------------------------------------------------

# Phrases that appear in WAF challenge / bot-detection pages but never in a
# real login portal.  Checked against title + headings + first 3 000 chars of
# visible text so we avoid false-positives from marketing copy deeper in the page.
_BOT_BLOCK_PHRASES: frozenset[str] = frozenset({
    # Cloudflare
    "just a moment", "checking your browser", "ray id",
    "ddos protection by cloudflare", "performance & security by cloudflare",
    "enable javascript and cookies to continue",
    # Akamai / Citrix Bot Manager
    "your access to this service has been limited",
    "reference #", "akamai",
    # DataDome
    "datadome",
    # PerimeterX / HUMAN
    "px-captcha", "perimeterx", "bot activity detected",
    # Imperva / Incapsula
    "incapsula", "incident id",
    # Generic WAF / challenge pages
    "automated traffic", "unusual activity",
    "verify you are human", "complete the security check",
    "one more step", "please enable javascript",
    "javascript is required to view this page",
})


def _is_bot_blocked(page_result: dict) -> bool:
    """Return True when the 403 page content matches known WAF/bot-detection patterns."""
    title   = (page_result.get("title")        or "").lower()
    visible = (page_result.get("visible_text") or "").lower()[:3000]
    headings = " ".join(page_result.get("headings") or []).lower()
    combined = f"{title} {headings} {visible}"
    return any(phrase in combined for phrase in _BOT_BLOCK_PHRASES)


_HTTPX_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": _STEALTH_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


async def _raw_http_check(url: str) -> str | None:
    """Fire a plain HTTP GET with browser-like headers — no automation fingerprint.

    Returns 'OK', 'FORBIDDEN', or None (network/timeout error).
    Used as a second opinion when Playwright returned 403 but page content gave
    no clear WAF signal: if raw HTTP succeeds, the 403 was automation-specific.
    """
    try:
        async with httpx.AsyncClient(
            timeout=10.0, follow_redirects=True
        ) as client:
            resp = await client.get(url, headers=_HTTPX_BROWSER_HEADERS)
            if 200 <= resp.status_code < 400:
                return "OK"
            if resp.status_code == 403:
                return "FORBIDDEN"
            return f"HTTP_{resp.status_code}"
    except Exception:
        return None


async def check_url_health(url):
    result = {
        "original_url": url,
        "final_url": None,
        "status": None,
        "health": "UNKNOWN",
        "reason": None,
        "redirect_chain": [],
        "load_time_ms": None,
        "token_detected": None,
        "soft_errors": [],
        "page_result": {},
    }

    token_info = detect_url_token(url)
    if token_info:
        result["token_detected"] = token_info
        logger.warning(
            "Embedded token detected in URL: %s", token_info["summary"],
        )

    # Fast DNS pre-flight — skip launching a browser for dead domains
    parsed = urlparse(url)
    domain = parsed.netloc
    try:
        socket.gethostbyname(domain)
    except socket.gaierror:
        result["health"] = "INACTIVE"
        result["reason"] = "DNS_FAILURE"
        # Stub a page_result so downstream consumers (incl. scrape-quality
        # reports) see a consistent shape for dead-domain entries.
        result["page_result"] = {
            "original_url": url,
            "network_error": "DNS_FAILURE",
            "extraction_quality": "unreachable",
            "visible_text_length": 0,
        }
        logger.warning("DNS resolution failed for %s", domain)
        return result

    # Single Playwright pass: health check + page extraction combined
    start_time = time.time()
    page_result = await extract_visible_layer(url, MAX_LOAD_TIME)
    result["load_time_ms"] = int((time.time() - start_time) * 1000)

    result["page_result"] = page_result
    result["final_url"] = page_result.get("final_url")
    result["redirect_chain"] = page_result.get("redirect_chain") or []
    result["status"] = page_result.get("http_status")

    # If Playwright hit a network-level error before getting an HTTP status
    net_error = page_result.get("network_error")
    if net_error:
        result["health"] = "INACTIVE"
        result["reason"] = net_error
        logger.warning("Health determined INACTIVE: reason=%s", net_error)
    else:
        health, reason = _status_to_health(page_result.get("http_status"))

        # 403 disambiguation: was the server blocking our automation, or is the
        # URL genuinely inaccessible?  Two-stage check:
        #   1. Content fingerprint — WAF challenge pages have recognisable text.
        #   2. Raw HTTP fallback   — if curl-like GET succeeds where Playwright
        #      failed, the block is automation-specific (not a dead URL).
        if reason == "FORBIDDEN":
            if _is_bot_blocked(page_result):
                reason = "BOT_BLOCKED"
                logger.warning(
                    "Health determined BOT_BLOCKED: WAF/anti-bot pattern detected for %s", url
                )
            else:
                raw = await _raw_http_check(url)
                if raw == "OK":
                    reason = "BOT_BLOCKED"
                    logger.warning(
                        "Health confirmed BOT_BLOCKED via raw HTTP cross-check for %s", url
                    )
                # raw == "FORBIDDEN" or None → genuine block, keep FORBIDDEN

        result["health"] = health
        result["reason"] = reason
        if health == "INACTIVE":
            logger.warning("Health determined INACTIVE: reason=%s", reason)

    # Server returned a 2xx status but a 0-byte body — the URL is technically
    # "alive" but useless for an end user. Override health so downstream
    # consumers (verifier, golden dataset) flag it for human review.
    if page_result.get("extraction_quality") == "empty_server_response":
        result["health"] = "INACTIVE"
        result["reason"] = "EMPTY_SERVER_RESPONSE"
        logger.warning(
            "Health overridden INACTIVE: server returned empty body for %s", url,
        )

    soft_errors = detect_soft_errors(page_result.get("headings") or [])
    if soft_errors:
        logger.warning("Soft errors detected in page headings: %s", soft_errors)
    result["soft_errors"] = soft_errors

    return result
