from playwright.async_api import TimeoutError as PlaywrightTimeoutError, async_playwright
import json
import re
import time





# Has some url health checks below, will check that later 
MAX_LOAD_TIME = 30000  # 30 seconds
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


async def extract_visible_layer(
    url,
    MAX_LOAD_TIME: int = MAX_LOAD_TIME,
    max_visible_text_chars: int | None = None,
):
    run_id = f"page-extract-{int(time.time() * 1000)}"
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
        "load_error": None
    }

    # region agent log
    _debug_log(run_id, "H1", "src/page_extraction.py:33", "extract_visible_layer_start", {
        "url": url,
        "max_load_time": MAX_LOAD_TIME,
        "max_visible_text_chars": max_visible_text_chars,
    })
    # endregion

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            try:
                await page.goto(url, timeout=MAX_LOAD_TIME, wait_until="networkidle")
                # region agent log
                _debug_log(run_id, "H2", "src/page_extraction.py:42", "page_loaded_with_networkidle", {
                    "url": url,
                })
                # endregion
            except PlaywrightTimeoutError as timeout_error:
                # region agent log
                _debug_log(run_id, "H2", "src/page_extraction.py:48", "networkidle_timeout_fallback", {
                    "error": str(timeout_error),
                    "url": url,
                })
                # endregion
                await page.goto(url, timeout=MAX_LOAD_TIME, wait_until="domcontentloaded")

            result["final_url"] = page.url
            result["title"] = await page.title()
            # region agent log
            _debug_log(run_id, "H2", "src/page_extraction.py:47", "page_loaded", {
                "final_url": result["final_url"],
                "title": result["title"],
            })
            # endregion

            # Visible body text
            visible_text = await page.evaluate("""
                () => {
                    return document.body.innerText;
                }
            """)

            # Clean whitespace
            cleaned_text = re.sub(r'\s+', ' ', visible_text or "").strip()
            result["visible_text_length"] = len(cleaned_text)

            if max_visible_text_chars is not None:
                result["visible_text"] = cleaned_text[:max_visible_text_chars]
                result["visible_text_truncated"] = len(cleaned_text) > max_visible_text_chars
            else:
                result["visible_text"] = cleaned_text

            # region agent log
            _debug_log(run_id, "H3", "src/page_extraction.py:67", "visible_text_extracted", {
                "visible_text_length": result["visible_text_length"],
                "visible_text_truncated": result["visible_text_truncated"],
                "visible_text_preview": result["visible_text"][:300],
            })
            # endregion

            # Headings (H1-H3)
            result["headings"] = await page.locator("h1, h2, h3").all_inner_texts()

            # Visible buttons
            result["buttons"] = await page.locator("button").all_inner_texts()

            # Detect login form
            password_fields = await page.locator("input[type='password']").count()
            result["login_form_present"] = password_fields > 0
            # region agent log
            _debug_log(run_id, "H4", "src/page_extraction.py:79", "page_metadata_extracted", {
                "headings_count": len(result["headings"]),
                "buttons_count": len(result["buttons"]),
                "login_form_present": result["login_form_present"],
            })
            # endregion


        except Exception as e:
            result["load_error"] = str(e)
            # region agent log
            _debug_log(run_id, "H2", "src/page_extraction.py:88", "page_extraction_error", {
                "error": str(e),
                "url": url,
            })
            # endregion

        finally:
            await browser.close()

    return result