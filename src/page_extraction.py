import logging
import re

from playwright.async_api import (
    Frame,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

logger = logging.getLogger("autologin.page_extraction")

MAX_LOAD_TIME = 30000  # 30 seconds

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


async def _collect_frame_text(frame: Frame) -> str:
    """Extract innerText from a single frame, silently returning '' on error."""
    try:
        return await frame.evaluate("() => document.body ? document.body.innerText : ''")
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
    MAX_LOAD_TIME: int = MAX_LOAD_TIME,
    max_visible_text_chars: int | None = None,
):
    result = {
        "original_url": url,
        "final_url": None,
        "title": None,
        "visible_text": None,
        "visible_text_length": 0,
        "visible_text_truncated": False,
        "headings": [],
        "buttons": [],
        "login_form_present": False,
        "error_text_detected": [],
        "load_error": None,
    }

    logger.info("Starting page extraction for %s (timeout=%dms)", url, MAX_LOAD_TIME)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            try:
                await page.goto(url, timeout=MAX_LOAD_TIME, wait_until="networkidle")
                logger.info("Page loaded with networkidle: %s", url)
            except PlaywrightTimeoutError as timeout_error:
                logger.warning(
                    "networkidle timed out for %s, falling back to domcontentloaded: %s",
                    url, timeout_error,
                )
                await page.goto(url, timeout=MAX_LOAD_TIME, wait_until="domcontentloaded")

            result["final_url"] = page.url
            result["title"] = await page.title()
            logger.info(
                "Page metadata: final_url=%s  title=%s",
                result["final_url"], result["title"],
            )

            raw_text = await _collect_all_text(page)
            cleaned_text = re.sub(r"\s+", " ", raw_text or "").strip()
            result["visible_text_length"] = len(cleaned_text)

            if max_visible_text_chars is not None:
                result["visible_text"] = cleaned_text[:max_visible_text_chars]
                result["visible_text_truncated"] = len(cleaned_text) > max_visible_text_chars
            else:
                result["visible_text"] = cleaned_text

            logger.info(
                "Visible text extracted: length=%d  truncated=%s  preview=%.200s",
                result["visible_text_length"],
                result["visible_text_truncated"],
                result["visible_text"][:200] if result["visible_text"] else "",
            )

            result["headings"] = await _collect_elements_across_frames(
                page, "h1, h2, h3",
            )
            result["buttons"] = await _collect_elements_across_frames(
                page, _BUTTON_SELECTOR,
            )

            result["login_form_present"] = await _detect_login_form(page)

            logger.info(
                "Page elements: headings=%d  buttons=%d  login_form=%s",
                len(result["headings"]),
                len(result["buttons"]),
                result["login_form_present"],
            )

        except Exception as e:
            result["load_error"] = str(e)
            logger.error("Page extraction error for %s: %s", url, e)

        finally:
            await browser.close()
            logger.debug("Browser closed for %s", url)

    return result
