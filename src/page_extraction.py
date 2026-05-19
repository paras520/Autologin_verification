import asyncio
import logging
import re
import sys

from playwright.async_api import (
    Error as PlaywrightError,
    Frame,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

logger = logging.getLogger("autologin.page_extraction")

MAX_LOAD_TIME = 30000  # 30 seconds

_STEALTH_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
window.chrome = { runtime: {} };

Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});
Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});
Object.defineProperty(navigator, 'platform', {get: () => 'Win32'});
Object.defineProperty(navigator, 'maxTouchPoints', {get: () => 0});

const _origPermQuery = window.navigator.permissions.query.bind(navigator.permissions);
window.navigator.permissions.query = (params) =>
    params.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : _origPermQuery(params);
"""

_CHROMIUM_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
    # Required in Linux/Docker containers (no user namespace for Chrome sandbox)
    *( ["--no-sandbox"] if sys.platform != "win32" else [] ),
]

_LOGIN_INPUT_KEYWORDS = re.compile(
    r"user|login|email|mobile|phone|account|customer|id|acct|usr|uname|corp",
    re.IGNORECASE,
)

_BUTTON_SELECTOR = (
    "button, "
    "input[type='submit'], "
    "input[type='button'], "
    "a[role='button'], "
    "[role='button'], "
    "a.btn, a.button, "
    "input[type='image']"
)

_MIN_GOOD_TEXT = 200   # characters — threshold for "high quality" extraction
_MIN_OK_TEXT = 100     # characters — below this we retry
_CONTENT_WAIT_MS = 12000  # how long to poll for body.innerText to hydrate
_RETRY_SETTLE_MS = 3000   # extra wait before re-extracting on retry
_NETWORKIDLE_FALLBACK_MS = 8000  # networkidle wait for the 3rd-attempt fallback
_TITLE_TIMEOUT_MS = 4000
_HEADINGS_TIMEOUT_MS = 5000
_BUTTONS_TIMEOUT_MS = 5000
_LOGIN_FORM_TIMEOUT_MS = 5000

_AUTO_SCROLL_JS = """
async () => {
    await new Promise(resolve => {
        let total = 0;
        const step = 400;
        const maxScroll = Math.max(document.body.scrollHeight, 5000);
        const timer = setInterval(() => {
            window.scrollBy(0, step);
            total += step;
            if (total >= maxScroll) {
                window.scrollTo(0, 0);
                clearInterval(timer);
                resolve();
            }
        }, 80);
    });
}
"""

_HTML_TAG_STRIP = re.compile(r"<[^>]+>")
_HTML_SCRIPT_STYLE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE,
)


async def _collect_frame_text(frame: Frame) -> str:
    """Extract innerText from a single frame, silently returning '' on error.

    Tries `innerText` first (visible text only), then falls back to
    `textContent` for SPAs that hide the real text behind CSS.
    """
    try:
        text = await frame.evaluate(
            "() => document.body ? document.body.innerText : ''"
        )
        if text and text.strip():
            return text
    except Exception:
        pass
    try:
        return await frame.evaluate(
            "() => document.body ? document.body.textContent || '' : ''"
        )
    except Exception:
        return ""


async def _collect_all_text(page) -> str:
    """Collect visible text from the main page and every nested iframe."""
    parts: list[str] = []
    parts.append(await _collect_frame_text(page.main_frame))

    for frame in page.frames:
        if frame == page.main_frame:
            continue
        text = await _collect_frame_text(frame)
        if text and text.strip():
            parts.append(text)
            logger.debug("Collected %d chars from child frame %s", len(text), frame.url)

    return "\n".join(parts)


async def _collect_elements_across_frames(page, selector: str) -> list[str]:
    """Gather inner-text of elements matching *selector* across all frames."""
    texts: list[str] = []
    for frame in page.frames:
        try:
            entries = await frame.locator(selector).all_inner_texts()
            texts.extend(t for t in entries if t and t.strip())
        except Exception:
            pass
    return texts


async def _wait_for_content(page, timeout_ms: int = _CONTENT_WAIT_MS) -> bool:
    """Wait for document.body to have meaningful text content (SPA hydration).

    Returns True if content appeared within the timeout, False otherwise.
    """
    try:
        await page.wait_for_function(
            f"() => document.body && document.body.innerText.trim().length > {_MIN_GOOD_TEXT}",
            timeout=timeout_ms,
        )
        return True
    except PlaywrightTimeoutError:
        return False
    except PlaywrightError:
        return False


async def _auto_scroll(page) -> None:
    """Scroll the page to bottom and back to trigger lazy-loaded content."""
    try:
        await page.evaluate(_AUTO_SCROLL_JS)
    except Exception:
        pass


async def _extract_html_fallback_text(page) -> str:
    """Fallback: strip tags from outerHTML when innerText returns empty.

    Handles cases where content is in shadow DOM or unusual structures.
    """
    try:
        html = await page.content()
    except Exception:
        return ""
    cleaned = _HTML_SCRIPT_STYLE.sub(" ", html)
    cleaned = _HTML_TAG_STRIP.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _score_extraction_quality(
    text: str,
    title: str | None = None,
    headings: list[str] | None = None,
    buttons: list[str] | None = None,
    login_form_present: bool = False,
    http_status: int | None = None,
) -> str:
    """Classify extraction quality using all available signals.

    A login page can be a perfectly successful scrape with only 30 chars
    of visible text if we captured its title, buttons, and login form.
    Quality is scored on "effective content" = text + title + headings +
    buttons, boosted further when a login form is detected.

    When the server returned a 2xx HTTP status but the page yields zero
    effective content, we emit "empty_server_response" instead of "empty"
    to distinguish a server-side empty body (URL is broken / deprecated)
    from a true scraping failure.
    """
    text = text or ""
    title = title or ""
    headings = headings or []
    buttons = buttons or []

    # Title weight scales inversely with body text: when body text is sparse,
    # a title is often the only reliable signal we have.
    title_weight = 3 if len(text) < 30 else 2

    # Effective content length combines every useful signal.
    effective = (
        len(text)
        + (len(title) * title_weight if title else 0)
        + sum(len(h) for h in headings if h)
        + sum(len(b) for b in buttons if b)
    )
    # A detected login form is strong evidence of a successful scrape.
    if login_form_present:
        effective += 100

    if effective == 0:
        if http_status is not None and 200 <= http_status < 300:
            return "empty_server_response"
        return "empty"
    if effective < 18:   # very sparse — truly no usable content
        return "low"
    if effective < 120:
        return "medium"
    return "high"


async def _detect_login_form(page) -> bool:
    """Detect whether the page (including iframes) contains a login form.

    Signals checked:
      1. Password input field  (strongest signal)
      2. Text / email / tel inputs whose name, id, placeholder, or aria-label
         contain login-related keywords (covers multi-step logins)
    """
    for frame in page.frames:
        try:
            if await frame.locator("input[type='password']").count() > 0:
                logger.debug("Password field found in frame %s", frame.url)
                return True

            text_inputs = frame.locator(
                "input[type='text'], input[type='email'], input[type='tel'], "
                "input:not([type])"
            )
            count = await text_inputs.count()
            for i in range(count):
                el = text_inputs.nth(i)
                attrs = " ".join(
                    str(await el.get_attribute(a) or "")
                    for a in ("name", "id", "placeholder", "aria-label")
                )
                if _LOGIN_INPUT_KEYWORDS.search(attrs):
                    logger.debug(
                        "Login-related input found in frame %s (attrs: %s)",
                        frame.url, attrs.strip(),
                    )
                    return True
        except Exception:
            pass
    return False


async def extract_visible_layer(
    url,
    load_timeout: int = MAX_LOAD_TIME,
    max_visible_text_chars: int | None = None,
):
    """Load a URL in a stealth Playwright browser and extract page data.

    Uses a multi-stage resilient pipeline:
      1. goto(domcontentloaded) — don't wait for networkidle, it lies for SPAs
      2. wait_for_function until body.innerText has >200 chars (SPA hydration)
      3. auto-scroll to trigger lazy-loaded content
      4. extract visible text + structured elements
      5. if text is still too short, retry once with longer wait
      6. if innerText is empty, fall back to stripping outerHTML

    Returns a dict that also includes HTTP health fields:
      http_status, redirect_chain, network_error, extraction_quality
    """
    result = {
        "original_url": url,
        "final_url": None,
        "http_status": None,
        "redirect_chain": [],
        "network_error": None,
        "title": None,
        "visible_text": None,
        "visible_text_length": 0,
        "visible_text_truncated": False,
        "headings": [],
        "buttons": [],
        "login_form_present": False,
        "error_text_detected": [],
        "load_error": None,
        "extraction_quality": "empty",
        "extraction_attempts": 0,
        "used_html_fallback": False,
    }

    redirect_chain: list[str] = []

    async def _do_extraction(page) -> str:
        """Collect text from all frames and return cleaned visible_text."""
        raw_text = await _collect_all_text(page)
        return re.sub(r"\s+", " ", raw_text or "").strip()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=_CHROMIUM_ARGS)
        context = await browser.new_context(
            user_agent=_STEALTH_UA,
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            java_script_enabled=True,
            ignore_https_errors=True,
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Sec-Ch-Ua": '"Chromium";v="131", "Not_A Brand";v="24"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
            },
        )
        await context.add_init_script(_STEALTH_INIT_SCRIPT)
        page = await context.new_page()

        def _on_response(resp):
            if resp.request.is_navigation_request() and resp.request.frame == page.main_frame:
                if 300 <= resp.status < 400:
                    redirect_chain.append(resp.url)

        page.on("response", _on_response)

        try:
            # ----- Attempt 1: goto(commit) + wait for content + scroll -----
            # "commit" fires as soon as first bytes are received — much faster
            # than "domcontentloaded" for slow or large SPAs. We then use
            # wait_for_function to wait for actual content hydration.
            nav_response = None
            try:
                nav_response = await page.goto(
                    url, timeout=load_timeout, wait_until="commit",
                )
            except PlaywrightTimeoutError:
                result["network_error"] = "TIMEOUT"
                logger.warning("Page load timed out for %s", url)
            except PlaywrightError as nav_err:
                err_msg = str(nav_err)
                if "net::ERR_NAME_NOT_RESOLVED" in err_msg:
                    result["network_error"] = "DNS_FAILURE"
                elif "net::ERR_CONNECTION_REFUSED" in err_msg:
                    result["network_error"] = "CONNECTION_REFUSED"
                elif "net::ERR_CONNECTION_RESET" in err_msg:
                    result["network_error"] = "CONNECTION_RESET"
                elif "net::ERR_HTTP2_PROTOCOL_ERROR" in err_msg:
                    result["network_error"] = "HTTP2_ERROR"
                elif "net::ERR_CERT" in err_msg:
                    result["network_error"] = "SSL_ERROR"
                else:
                    result["network_error"] = "CONNECTION_ERROR"
                result["load_error"] = err_msg
                result["extraction_quality"] = "unreachable"
                logger.warning("Navigation failed for %s: %s", url, nav_err)
                return result

            if nav_response is not None:
                result["http_status"] = nav_response.status
            result["redirect_chain"] = redirect_chain
            result["final_url"] = page.url

            # Wait for SPA hydration (body.innerText > MIN_GOOD_TEXT chars)
            await _wait_for_content(page, timeout_ms=_CONTENT_WAIT_MS)

            # Trigger lazy loads + intersection observers
            await _auto_scroll(page)

            result["extraction_attempts"] = 1
            cleaned_text = await _do_extraction(page)

            # ----- Attempt 2: retry with longer wait if text is too short -----
            http_ok = result["http_status"] is None or 200 <= (result["http_status"] or 0) < 400
            if len(cleaned_text) < _MIN_OK_TEXT and http_ok:
                logger.info(
                    "[extract] retry for %s (first attempt %d chars)", url, len(cleaned_text),
                )
                try:
                    await page.wait_for_timeout(_RETRY_SETTLE_MS)
                    await _auto_scroll(page)
                    await _wait_for_content(page, timeout_ms=8000)
                    retry_text = await _do_extraction(page)
                    result["extraction_attempts"] = 2
                    if len(retry_text) > len(cleaned_text):
                        cleaned_text = retry_text
                except Exception as exc:
                    logger.debug("[extract] retry failed for %s: %s", url, exc)

            # ----- Fallback: strip HTML if innerText is still empty -----
            if not cleaned_text and http_ok:
                logger.info("[extract] HTML fallback for %s", url)
                fallback_text = await _extract_html_fallback_text(page)
                if fallback_text:
                    cleaned_text = fallback_text
                    result["used_html_fallback"] = True

            # ----- 3rd attempt: wait for networkidle then re-extract -----
            # Some SPAs (shadow DOM, complex loaders) still yield empty content
            # after both the polling wait and HTML fallback. Give them a final
            # networkidle window before giving up.
            if not cleaned_text and http_ok:
                logger.info("[extract] networkidle fallback for %s", url)
                try:
                    await page.wait_for_load_state(
                        "networkidle", timeout=_NETWORKIDLE_FALLBACK_MS,
                    )
                except Exception:
                    pass
                await _auto_scroll(page)
                third_text = await _do_extraction(page)
                if third_text:
                    cleaned_text = third_text
                    result["extraction_attempts"] = (result["extraction_attempts"] or 0) + 1
                else:
                    # Last resort: HTML fallback after networkidle
                    third_fallback = await _extract_html_fallback_text(page)
                    if third_fallback:
                        cleaned_text = third_fallback
                        result["used_html_fallback"] = True
                        result["extraction_attempts"] = (result["extraction_attempts"] or 0) + 1

            result["visible_text_length"] = len(cleaned_text)

            if max_visible_text_chars is not None:
                result["visible_text"] = cleaned_text[:max_visible_text_chars]
                result["visible_text_truncated"] = len(cleaned_text) > max_visible_text_chars
            else:
                result["visible_text"] = cleaned_text

            # Title + structured elements (each with short timeout so no single
            # call can hang the pipeline on slow/half-loaded pages).
            try:
                page.set_default_timeout(_HEADINGS_TIMEOUT_MS)
            except Exception:
                pass

            try:
                result["title"] = await asyncio.wait_for(
                    page.title(), timeout=_TITLE_TIMEOUT_MS / 1000,
                )
            except Exception:
                result["title"] = None

            try:
                result["headings"] = await asyncio.wait_for(
                    _collect_elements_across_frames(page, "h1, h2, h3"),
                    timeout=_HEADINGS_TIMEOUT_MS / 1000,
                )
            except Exception:
                result["headings"] = []

            try:
                result["buttons"] = await asyncio.wait_for(
                    _collect_elements_across_frames(page, _BUTTON_SELECTOR),
                    timeout=_BUTTONS_TIMEOUT_MS / 1000,
                )
            except Exception:
                result["buttons"] = []

            try:
                result["login_form_present"] = await asyncio.wait_for(
                    _detect_login_form(page),
                    timeout=_LOGIN_FORM_TIMEOUT_MS / 1000,
                )
            except Exception:
                result["login_form_present"] = False

            # Quality score now uses ALL signals, not just text length.
            result["extraction_quality"] = _score_extraction_quality(
                cleaned_text,
                title=result["title"],
                headings=result["headings"],
                buttons=result["buttons"],
                login_form_present=result["login_form_present"],
                http_status=result["http_status"],
            )

        except PlaywrightError as e:
            err_msg = str(e)
            if "net::ERR_NAME_NOT_RESOLVED" in err_msg:
                result["network_error"] = "DNS_FAILURE"
            elif "net::ERR_CONNECTION_REFUSED" in err_msg:
                result["network_error"] = "CONNECTION_REFUSED"
            elif "net::ERR_CONNECTION_RESET" in err_msg:
                result["network_error"] = "CONNECTION_RESET"
            elif "net::ERR_HTTP2_PROTOCOL_ERROR" in err_msg:
                result["network_error"] = "HTTP2_ERROR"
            elif "net::ERR_CERT" in err_msg:
                result["network_error"] = "SSL_ERROR"
            else:
                result["network_error"] = "CONNECTION_ERROR"
            result["load_error"] = err_msg
            result["extraction_quality"] = "unreachable"
            logger.error("Page extraction error for %s: %s", url, e)

        except Exception as e:
            result["load_error"] = str(e)
            result["extraction_quality"] = "unreachable"
            logger.error("Page extraction error for %s: %s", url, e)

        finally:
            await context.close()
            await browser.close()

    return result
