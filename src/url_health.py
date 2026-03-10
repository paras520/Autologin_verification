import requests
import socket
from urllib.parse import urlparse
import time

from page_extraction import extract_visible_layer, MAX_LOAD_TIME


TIMEOUT = 20  # seconds
MAX_REDIRECTS = 5

def detect_soft_errors(visible_text):
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

    visible_text_lower = visible_text.lower()

    found_errors = [
        phrase for phrase in error_indicators
        if phrase in visible_text_lower
    ]

    return found_errors


def check_url_health(url):

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
    page_result = extract_visible_layer(url, MAX_LOAD_TIME)

    soft_errors = detect_soft_errors(page_result"headings")
    


    return result