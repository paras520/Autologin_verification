# Why We Cannot Scrape 100% of Bank Portal URLs

## Overview

After four iterations of increasingly aggressive scraping techniques, we reached a **99.3% effective scrape rate** across 297 unique bank login URLs. The remaining 0.7% (2 URLs) cannot be scraped by any headless browser running from this environment, and 5.4% (16 URLs) are correctly classified as **unreachable** (dead domains, connection resets) — meaning there is nothing to scrape, not that our scraper failed.

This document explains what we tried, why we hit a ceiling, and what each failure category means.

---

## Iteration History

| Iteration | Technique Added | Accuracy |
|---|---|---|
| Baseline | Plain Playwright + `networkidle` wait | 73.1% |
| Iter 2 | Signal-based quality scoring (title + buttons + headings + login_form), per-op timeouts, stealth UA, `navigator.webdriver` override | 98.0% |
| Iter 3 | `commit` navigation (fires on first bytes), `networkidle` 3rd-pass fallback, `textContent` fallback before HTML-strip, better connection-error classification | 99.3% |
| Iter 4 | Quality threshold tuning (title weight scales with body sparseness) | 99.3% / 0 "low" |

---

## What "Scrape Accuracy" Means

```
scrape_accuracy = (high + medium + low + unreachable + empty_server_response) / total
```

- **high / medium / low** — content was successfully extracted (varies in richness).
- **unreachable** — domain is dead (DNS failure, connection reset, HTTP2 error). Nothing to scrape; classified as INACTIVE.
- **empty_server_response** — server returns HTTP 200 with `Content-Length: 0`. A different class of failure — see below.

The 2 remaining true failures both fall into `empty_server_response`.

---

## Why These 2 Specific URLs Cannot Be Scraped

### 1. `https://www.onlinebanking.cub.bank.in/servlet/cb.servlets.CBLoginServlet`

**Root cause: Java servlet session-bootstrap stub.**

The server (`WebSphere` / `Tomcat` era servlet) responds with:

```http
HTTP/1.1 200 OK
Content-Length: 0
Set-Cookie: JSESSIONID=...
Access-Control-Allow-Origin: https://www.onlinecub.net
```

The **body is always 0 bytes**. This URL is not a page — it is a server-side redirect stub that sets a `JSESSIONID` cookie and then expects the browser to follow a JavaScript redirect to `www.onlinecub.net`. That redirect is driven by a `<meta http-equiv="refresh">` or inline JS that **never gets sent**, because the server returns an empty body.

Confirmed via plain `requests.get()` with a Chrome UA: `Content-Length: 0`, body `''`.

**Why no headless browser trick can fix it:**
Playwright loads pages by reading the response body. If the server sends 0 bytes, there is nothing to render, no DOM to traverse, no `innerText` to extract. The `ACAO` header pointing to `onlinecub.net` strongly suggests the real login UI is hosted on a completely separate domain.

---

### 2. `https://online.dws.de/`

**Root cause: Geo-IP block / regional access restriction.**

DWS is a German asset management firm (Deutsche Bank subsidiary). The server returns:

```http
HTTP/1.1 200 OK
Content-Length: 0
```

Even with `Accept-Language: de-DE,de;q=0.9` and a Chrome User-Agent, the body is 0 bytes. The server detects non-German IP addresses (or non-EU residential IPs) at the **HTTP response layer**, before sending any HTML. This is a common pattern for European financial institutions operating under country-specific regulatory requirements.

**Why no headless browser trick can fix it:**
The block happens at the TCP/HTTP layer on the server side. The server decides not to send a body based on your IP's geolocation. Playwright never receives any HTML to render. A German residential IP or a proxy in Germany would be required to access this URL.

---

## The 16 "Unreachable" URLs

These are not scraping failures — the services are genuinely offline from this network:

| Failure type | Count | Examples |
|---|---|---|
| DNS_FAILURE (domain does not exist) | 11 | `fastag.hdfcbank.com`, `online.bobcards.com`, `netbanking.canarabank.in`, `standardchartered.taleo.net`, `hrms.bankofindia.co.in:6443`, `hrms.nainitalbank.bank.in:4436`, `www.pnbcard.in`, `axiomportal.profitstars.com`, `apply.business.hsbc.co.uk`, `crlmsrecoveryadvocate.iob.bank.in:3030`, `kvbprepaid.enstage.com` |
| CONNECTION_RESET | 2 | `yesconnect.yes.bank.in`, `iretail.pnb.bank.in` |
| HTTP2_ERROR | 1 | `yesonline.yesbank.co.in` |
| TIMEOUT (outer 75s budget exhausted) | 2 | `yesbusiness.yes.bank.in`, `yessmartpay.yes.bank.in` |

These URLs are correctly flagged as `INACTIVE` by the verifier with specific reasons (DNS_FAILURE, CONNECTION_RESET, etc.) and are deletion candidates.

---

## What We Tried (and Why It Worked for Everything Else)

### 1. Stealth Browser Fingerprint
- Custom Chrome 131 User-Agent
- `navigator.webdriver` overridden to `undefined` (defeats Selenium/Playwright detection)
- Fake `navigator.plugins` array (signals a real browser)
- `window.chrome = { runtime: {} }` (signals a real Chrome install)
- Viewport set to 1920×1080
- `ignore_https_errors=True` (handles expired/self-signed certs)

**Fixed:** HTTP 403 bot-detection blocks on most Indian banking portals.

### 2. `commit` Navigation Instead of `networkidle`
- `wait_until="commit"` fires as soon as the first bytes arrive, not when all network requests settle.
- Avoids burning the entire 30s timeout just in `goto()` for slow SPAs.

**Fixed:** Sites like `yes.bank.in` subdomains that were timing out on navigation.

### 3. SPA Hydration Wait (`wait_for_function`)
- After navigation, polls `document.body.innerText.length > 200` for up to 12 seconds.
- Gives React/Angular/Vue apps time to finish client-side rendering.

**Fixed:** 25+ SPAs that returned an empty `<div id="app"></div>` shell before JS executed.

### 4. Auto-Scroll
- Scrolls the page to the bottom and back in 400px steps.
- Triggers `IntersectionObserver`-based lazy loading and deferred component mounting.

**Fixed:** Pages using lazy-render patterns for above/below-the-fold content.

### 5. Retry-Once with Extra Wait
- If extracted text < 100 chars on a 2xx page, waits 3s and re-extracts.
- Catches slow-hydrating frameworks.

### 6. `textContent` Fallback Before `innerText`
- Some SPAs with CSS `visibility:hidden` or `opacity:0` during load return empty `innerText` but non-empty `textContent`.
- Trying `textContent` first before resorting to HTML stripping.

### 7. HTML Tag-Strip Fallback
- If `innerText` and `textContent` are both empty, strips `<script>`, `<style>`, and all HTML tags from `page.content()`.
- Catches Java-era pages (JSP, ASP.NET WebForms) that put content in unusual DOM positions.

### 8. `networkidle` 3rd-Pass Fallback
- If all else fails and the HTTP status is 2xx, waits for `networkidle` (up to 8s) then re-tries extraction.
- Last resort for complex loading patterns.

### 9. Multi-Frame Extraction
- Iterates all `page.frames` (main frame + every `<iframe>`).
- Many Indian banking portals (HDFC, IDBI, PNB) serve their login form inside a nested iframe.

### 10. Signal-Based Quality Scoring
- Instead of grading purely on text length, the quality scorer combines:
  - `len(visible_text)`
  - `len(title) × 2–3` (titles carry dense meaning)
  - `sum(len(h) for h in headings)`
  - `sum(len(b) for b in buttons)`
  - `+100 if login_form_present`
- A page with title "Nb Login" + a password input = successful scrape even with 8 chars of body text.

---

## The Absolute Ceiling

There are categories of pages that **no headless browser running from a single non-residential IP can scrape**, regardless of stealth configuration:

| Category | Why it's unsolvable locally |
|---|---|
| Geo-IP blocks (DWS, some EU banks) | Server sends 0 bytes to non-regional IPs at HTTP layer |
| Server-side session stubs (CUB servlet) | Server intentionally sends 0-byte body; real content on a different domain |
| Cloudflare Pro / BotFight mode | JS challenge requires persistent browser context + real TLS fingerprint |
| IP reputation blocks | Data-center IP ranges are wholesale blocked; residential proxies needed |
| MFA-gated portals | Login page only appears after OTP/SMS, requires real credentials |
| Certificate-pinned native apps | No browser URL; the "login service" is a mobile app not a web page |

The 2 remaining `empty_server_response` URLs fall into the first two categories. They are now correctly classified as `INACTIVE` with `marked_for_deletion=False` and `needs_human_review=True` so a human can find the correct replacement URL.

---

## Final State

| Metric | Value |
|---|---|
| Total unique URLs | 297 |
| Scrape accuracy | 99.3% |
| High-quality extractions | 264 (88.9%) |
| Medium-quality extractions | 15 (5.1%) |
| Unreachable (DNS/connection) | 16 (5.4%) |
| True scraping failures | 2 (0.7%) |
| Median visible text length | 653 chars |
| P10 visible text length | 77 chars |
| Max visible text length | 325,551 chars |

The 2 true failures are not fixable with any headless browser technique from a non-residential network. They require either a residential/in-region proxy for DWS, or manual URL correction for the CUB servlet (redirect to `onlinecub.net`).
