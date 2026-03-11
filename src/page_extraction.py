import logging
import re

from playwright.async_api import TimeoutError as PlaywrightTimeoutError, async_playwright

logger = logging.getLogger("autologin.page_extraction")

MAX_LOAD_TIME = 30000  # 30 seconds


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

            visible_text = await page.evaluate("() => document.body.innerText")
            cleaned_text = re.sub(r"\s+", " ", visible_text or "").strip()
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

            result["headings"] = await page.locator("h1, h2, h3").all_inner_texts()
            result["buttons"] = await page.locator("button").all_inner_texts()

            password_fields = await page.locator("input[type='password']").count()
            result["login_form_present"] = password_fields > 0

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
