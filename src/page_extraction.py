from playwright.sync_api import sync_playwright
import re

# Has some url health checks below, will check that later 
MAX_LOAD_TIME = 30000  # 30 seconds


def extract_visible_layer(url, MAX_LOAD_TIME: int = MAX_LOAD_TIME):
    result = {
        "original_url": url,
        "final_url": None,
        "title": None,
        "visible_text": None,
        "headings": [],
        "buttons": [],
        "login_form_present": False,
        "error_text_detected": [],
        "load_error": None
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(url, timeout=MAX_LOAD_TIME, wait_until="networkidle")

            result["final_url"] = page.url
            result["title"] = page.title()

            # Visible body text
            visible_text = page.evaluate("""
                () => {
                    return document.body.innerText;
                }
            """)

            # Clean whitespace
            cleaned_text = re.sub(r'\s+', ' ', visible_text).strip()
            result["visible_text"] = cleaned_text[:15000]

            # Headings (H1-H3)
            result["headings"] = page.locator("h1, h2, h3").all_inner_texts()

            # Visible buttons
            result["buttons"] = page.locator("button").all_inner_texts()

            # Detect login form
            password_fields = page.locator("input[type='password']").count()
            result["login_form_present"] = password_fields > 0


        except Exception as e:
            result["load_error"] = str(e)

        finally:
            browser.close()

    return result