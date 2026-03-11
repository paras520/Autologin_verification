import logging
import re
import socket
import time
from urllib.parse import parse_qs, urlparse

import requests

from src.page_extraction import extract_visible_layer, MAX_LOAD_TIME

logger = logging.getLogger("autologin.url_health")

TIMEOUT = 20  # seconds
MAX_REDIRECTS = 5

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


async def check_url_health(url):
    logger.info("=== Health check started for %s ===", url)

    result = {
        "original_url": url,
        "final_url": None,
        "status": None,
        "health": "UNKNOWN",
        "reason": None,
        "redirect_chain": [],
        "load_time_ms": None,
        "token_detected": None,
    }

    token_info = detect_url_token(url)
    if token_info:
        result["token_detected"] = token_info
        logger.warning(
            "Embedded token detected in URL: %s", token_info["summary"],
        )

    start_time = time.time()

    try:
        parsed = urlparse(url)
        domain = parsed.netloc

        try:
            socket.gethostbyname(domain)
            logger.info("DNS resolved for %s", domain)
        except socket.gaierror:
            result["health"] = "INACTIVE"
            result["reason"] = "DNS_FAILURE"
            logger.warning("DNS resolution failed for %s", domain)
            return result

        session = requests.Session()
        session.max_redirects = MAX_REDIRECTS

        response = session.get(url, timeout=TIMEOUT, allow_redirects=True)

        for resp in response.history:
            result["redirect_chain"].append(resp.url)

        result["final_url"] = response.url
        result["status"] = response.status_code

        logger.info(
            "HTTP %d from %s → %s  content_type=%s  redirects=%d",
            response.status_code,
            url,
            response.url,
            response.headers.get("Content-Type", "unknown"),
            len(response.history),
        )

        status = response.status_code

        if status == 200:
            result["health"] = "OK"
        elif status in [301, 302]:
            result["health"] = "REDIRECT"
        elif status == 404:
            result["health"] = "INACTIVE"
            result["reason"] = "NOT_FOUND"
        elif status == 410:
            result["health"] = "INACTIVE"
            result["reason"] = "PERMANENTLY_REMOVED"
        elif status in [500, 503]:
            result["health"] = "INACTIVE"
            result["reason"] = "SERVER_ERROR"
        else:
            result["health"] = "INACTIVE"
            result["reason"] = f"HTTP_{status}"

        if result["health"] == "INACTIVE":
            logger.warning("Health determined INACTIVE: reason=%s", result["reason"])

    except requests.exceptions.Timeout:
        result["health"] = "INACTIVE"
        result["reason"] = "TIMEOUT"
        logger.warning("Request timed out for %s", url)

    except requests.exceptions.ConnectionError:
        result["health"] = "INACTIVE"
        result["reason"] = "CONNECTION_REFUSED"
        logger.warning("Connection refused for %s", url)

    except requests.exceptions.TooManyRedirects:
        result["health"] = "INACTIVE"
        result["reason"] = "INFINITE_REDIRECT"
        logger.warning("Too many redirects for %s", url)

    except Exception as e:
        result["health"] = "INACTIVE"
        result["reason"] = str(e)
        logger.error("Unexpected error during health check for %s: %s", url, e)

    finally:
        result["load_time_ms"] = int((time.time() - start_time) * 1000)
        logger.info("HTTP check completed in %dms  health=%s", result["load_time_ms"], result["health"])

    page_result = await extract_visible_layer(url, MAX_LOAD_TIME)

    logger.info(
        "Page extraction done: title=%s  visible_text_length=%s  load_error=%s",
        page_result.get("title"),
        page_result.get("visible_text_length"),
        page_result.get("load_error"),
    )

    soft_errors = detect_soft_errors(page_result["headings"])
    if soft_errors:
        logger.info("Soft errors detected: %s", soft_errors)

    result["soft_errors"] = soft_errors
    result["page_result"] = page_result

    return result
