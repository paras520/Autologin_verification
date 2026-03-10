import requests
import socket
import json
import time
from urllib.parse import urlparse

from src.page_extraction import extract_visible_layer, MAX_LOAD_TIME


TIMEOUT = 20  # seconds
MAX_REDIRECTS = 5
DEBUG_LOG_PATH = "debug-a55a2e.log"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as debug_file:
        debug_file.write(json.dumps({
            "sessionId": "a55a2e",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }, ensure_ascii=False) + "\n")


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
        "access denied"
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
        phrase for phrase in error_indicators
        if phrase in visible_text_lower
    ]

    return found_errors


async def check_url_health(url):
    run_id = f"url-health-{int(time.time() * 1000)}"

    result = {
        "original_url": url,
        "final_url": None,
        "status": None,
        "health": "UNKNOWN",
        "reason": None,
        "redirect_chain": [],
        "load_time_ms": None
    }

    start_time = time.time()

    try:
        # Step 1 — DNS Resolution Check
        parsed = urlparse(url)
        domain = parsed.netloc

        try:
            socket.gethostbyname(domain)
        except socket.gaierror:
            result["health"] = "INACTIVE"
            result["reason"] = "DNS_FAILURE"
            return result

        # Step 2 — HTTP Request with Redirect Handling
        session = requests.Session()
        session.max_redirects = MAX_REDIRECTS

        response = session.get(
            url,
            timeout=TIMEOUT,
            allow_redirects=True
        )

        # Capture redirect chain
        for resp in response.history:
            result["redirect_chain"].append(resp.url)

        result["final_url"] = response.url
        result["status"] = response.status_code
        # region agent log
        _debug_log(run_id, "H5", "src/url_health.py:97", "http_response_received", {
            "url": url,
            "status_code": response.status_code,
            "final_url": response.url,
            "content_type": response.headers.get("Content-Type"),
            "response_preview": response.text[:300],
        })
        # endregion

        # Step 3 — Status Code Validation
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

    except requests.exceptions.Timeout:
        result["health"] = "INACTIVE"
        result["reason"] = "TIMEOUT"

    except requests.exceptions.ConnectionError:
        result["health"] = "INACTIVE"
        result["reason"] = "CONNECTION_REFUSED"

    except requests.exceptions.TooManyRedirects:
        result["health"] = "INACTIVE"
        result["reason"] = "INFINITE_REDIRECT"

    except Exception as e:
        result["health"] = "INACTIVE"
        result["reason"] = str(e)

    finally:
        result["load_time_ms"] = int((time.time() - start_time) * 1000)

    # page matching logic
    page_result = await extract_visible_layer(url, MAX_LOAD_TIME)
    # region agent log
    _debug_log(run_id, "H6", "src/url_health.py:152", "page_result_summary", {
        "load_error": page_result.get("load_error"),
        "title": page_result.get("title"),
        "final_url": page_result.get("final_url"),
        "visible_text_length": page_result.get("visible_text_length"),
        "visible_text_preview": (page_result.get("visible_text") or "")[:300],
    })
    # endregion

    soft_errors = detect_soft_errors(page_result["headings"])

    result["soft_errors"] = soft_errors
    result["page_result"] = page_result


    return result