import logging
import socket
import time
from urllib.parse import urlparse

import requests

from src.page_extraction import extract_visible_layer, MAX_LOAD_TIME

logger = logging.getLogger("autologin.url_health")

TIMEOUT = 20  # seconds
MAX_REDIRECTS = 5


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
    }

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
