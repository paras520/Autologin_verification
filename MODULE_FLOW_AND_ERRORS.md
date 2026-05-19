# Module Flow & Error Analysis — Autologin Verification Pipeline

> Deep technical breakdown of every check, the logic it uses, and every error / edge-case the system can produce.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [End-to-End Data Flow](#end-to-end-data-flow)
3. [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
   - Phase 0: Deduplication (batch only)
   - Phase 1: URL Health Check & Page Extraction
   - Phase 1.5: Audience Classifier (customer-facing check)
   - Phase 2: Two-Stage LLM Match Pipeline
   - Phase 3: Country Match (deterministic)
   - Phase 4: Final Decision Assembly
4. [Complete Error Catalog](#complete-error-catalog)
5. [Key File Map](#key-file-map)

---

## Architecture Overview

```
HTTP Request (FastAPI)
        |
        v
+-----------------------------------+
|  Router  (verification_router.py) |
|    POST /check        (single)    |
|    POST /check/batch  (batch)     |
+-----------------------------------+
        |
        v
+-----------------------------------+
|  Controller (verification_controller.py) |
|  - _process_cb_link() fetches rows     |
|  - detect_duplicates()   Phase 0       |
|  - _process_row() per non-duplicate    |
+-----------------------------------+
        |
        v
+-----------------------------------+
|  Service (verification_service.py) |
|  verify_url() / verify_row()       |
|    Phase 1  -> check_url_health()  |
|    Phase 1.5 -> classify_page_audience() |
|    Phase 2  -> run_checks()        |
|    Phase 3  -> assess_country_match() |
|    Phase 4  -> assemble response   |
+-----------------------------------+
```

---

## End-to-End Data Flow

### Single URL (`POST /check`)

```
CheckRequest arrives
    |
    v
[Input validation]  url must be non-empty + have scheme+host
    |
    v
Phase 1:  check_url_health(url)
    |-- DNS pre-flight (socket.gethostbyname)
    |-- Playwright launch (headless Chromium)
    |-- Page load + extraction
    |-- Token detection in URL
    |-- Soft-error scan
    |
    v
Phase 1.5:  classify_page_audience()
    |-- LLM call: is this page customer-facing?
    |-- Returns: is_customer_facing, confidence, category, reason
    |-- High-confidence non-customer -> early exit (marked_for_deletion)
    |-- Low-confidence non-customer -> continue, flag human_review
    |
    v
Phase 2:  run_checks()  (two-stage LLM pipeline)
    |-- Stage A (cheap): extract_and_score()
    |      -> extracts bank_identifiers, relevant_page_sections,
    |         login_signals, is_login_page, login_type_suggestion
    |-- Stage B (smart): assess_match_with_identifiers()
    |      -> bank_matched, service_matched, confidence_score,
    |         url_confidence_score, login_type, reason
    |
    v
Phase 3:  assess_country_match()  (deterministic, no LLM)
    |-- Scans page text for country signals (ccTLD, city names, currency, phone)
    |-- Returns: matched=True|False|None, scores, reason
    |
    v
Phase 4:  Assemble ReturnResponse
    |-- Priority stack builds final_reason
    |-- Flags: inactive_flagged, marked_for_deletion, marked_for_human_review
    +-- Response returned
```

### Batch URL (`POST /check/batch`)

```
BatchCheckRequest arrives (list of cb_link_ids)
    |
    v
For each cb_link_id:
    |-- fetch_rows() from DB (or local dump)
    |-- detect_duplicates()   Phase 0 (same URL -> Case 1/3)
    |-- For each non-duplicate row:
    |       run verify_row() -> full pipeline above
    |-- resolve_same_name_duplicates()  Phase 2 (same name, diff URL -> Case 2)
    |-- resolve_parent_child()          Phase 3 stub (no-op today)
    +-- Return CbLinkResult
```

---

## Phase-by-Phase Breakdown

### Phase 0: Deduplication (`src/services/duplicate_service.py`)

**When:** Batch only, before any verification.

**Logic:**
1. **Normalize URLs** — lowercase domain, strip ephemeral query params (`token`, `session`, `utm_*`, etc.), drop trailing slash.
2. **Group by normalized URL** — rows with the same normalized URL enter the same group.
3. **Pick canonical survivor** per group using priority:
   - `status == 'active'` beats inactive
   - Lower `sorting_order` wins
   - More descriptive display name (more non-noise words, then longer)
   - Stable tiebreak by row `id` string
4. **Case 1** (same name + same URL): losers get `dedupe_action='delete_same_url'`
5. **Case 3** (different names + same URL): losers get `delete_same_url`; survivor gets `canonical_display_name` = best name in group

**Post-verification Phase 2:**
- Groups surviving rows by normalized `login_service` name.
- If 2+ rows share the name but have **different** normalized URLs:
  - Winner picked by: not marked_for_deletion > health_check=True > higher page_match_score > lower sorting_order
  - Losers get `dedupe_action='delete_same_name_worse'`

**Phase 3:** `resolve_parent_child()` is a stub — returns rows unchanged.

---

### Phase 1: URL Health Check & Page Extraction (`src/utils/url_health.py` + `src/page_extraction.py`)

**Entry:** `check_url_health(url)`

#### Step 1A — Token Detection (deterministic, no browser)

`detect_url_token(url)` scans for:
- JWT tokens in full URL (`eyJ...` pattern)
- Known session/auth query param names (`token`, `session`, `csrf`, `__START_TRAN_FLAG__`, etc.)
- UUIDs in query values
- Long hex strings (32+ chars)
- Long base64 opaque tokens (40+ chars)
- UUIDs / JWTs / hex in URL path segments

Safe params that are **ignored**: `url`, `redirect`, `return`, `lang`, `page`, `id`, `action`, `bank_id`, etc.

**If token found:** `result["token_detected"]` is populated; downstream pipeline treats this as deletion + human review.

#### Step 1B — DNS Pre-flight

```python
socket.gethostbyname(domain)
```
- If DNS fails (`socket.gaierror`): returns immediately with `health=INACTIVE`, `reason=DNS_FAILURE`
- No browser is launched for dead domains.

#### Step 1C — Playwright Page Extraction (`extract_visible_layer()`)

**Browser setup:**
- Headless Chromium
- Stealth user-agent: `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36`
- Stealth init script: hides `navigator.webdriver`, fakes plugins/languages, sets `window.chrome`
- Viewport: 1920x1080
- `ignore_https_errors=True`

**Extraction pipeline (multi-stage):**

| Attempt | Trigger | Action |
|---------|---------|--------|
| **1** | Always | `goto(url, wait_until="commit")` — fires as soon as first bytes received |
| | | `wait_for_function()` until `body.innerText.length > 200` (SPA hydration, up to 12s) |
| | | Auto-scroll to bottom and back (triggers lazy-loaded content) |
| | | Extract visible text from **main frame + every iframe** |
| | | Extract title, h1/h2/h3 headings, buttons |
| | | Detect login form (password input OR login-keyword text inputs across all frames) |
| **2** | If text < 100 chars and HTTP OK | Wait 3s extra, scroll again, wait up to 8s for content |
| **3 (HTML fallback)** | If innerText still empty and HTTP OK | Strip HTML tags from `outerHTML` to catch shadow-DOM content |
| **4 (networkidle)** | If still empty and HTTP OK | Wait for `networkidle` (up to 8s), scroll, re-extract |
| | | Final HTML fallback again |

**Text collection across frames:**
- `_collect_frame_text()` tries `document.body.innerText` first, falls back to `textContent`
- `_collect_all_text()` concatenates main frame + all child frames

**Login form detection (`_detect_login_form()`):**
- Scans every frame for:
  1. `input[type='password']` (strong signal)
  2. `input[type='text'|'email'|'tel']` whose `name/id/placeholder/aria-label` matches login keywords (`user`, `login`, `email`, `mobile`, `account`, `customer`, etc.)

**Extraction quality scoring (`_score_extraction_quality()`):**
- Computes "effective content" = visible_text + (title * weight) + headings + buttons + 100 bonus if login_form_present
- `empty_server_response`: HTTP 2xx but zero effective content (server returned empty body)
- `empty`: no content and no HTTP status
- `low`: < 18 effective chars
- `medium`: < 120 effective chars
- `high`: >= 120 effective chars

**Health mapping (`_status_to_health()`):**

| HTTP Status | Health | Reason |
|-------------|--------|--------|
| 200-299 | OK | — |
| 301-308 | REDIRECT | — |
| 404 | INACTIVE | NOT_FOUND |
| 410 | INACTIVE | PERMANENTLY_REMOVED |
| 500/502/503 | INACTIVE | SERVER_ERROR |
| 403 | INACTIVE | FORBIDDEN |
| Anything else | INACTIVE | HTTP_{status} |
| No response | INACTIVE | NO_RESPONSE |

**Empty server response override:**
- If `extraction_quality == "empty_server_response"` and HTTP was 2xx, health is forced to `INACTIVE` with reason `EMPTY_SERVER_RESPONSE`.
- This prevents auto-deletion; instead it flags for human review because the URL may need replacement (e.g., different host).

**Soft error detection (`detect_soft_errors()`):**
- Scans headings for error phrases: "404", "page not found", "service unavailable", "temporarily unavailable", "invalid request", "resource not found", "this page has moved", "error occurred", "not authorized", "access denied"
- Returns list of matched phrases; appended to result notes.

---

### Phase 1.5: Audience Classifier (`src/heuristics.py` -> `classify_customer_facing()`)

**When:** After health check passes (or even if health failed? No — actually called in `verify_url()` after health check, regardless of health result? Let me check...)

Actually looking at `verification_service.py`, the classifier is called unconditionally after health check but before the early-exit return. Wait, no — the health early exit happens at line 96. The classifier is called at line 117. So the classifier only runs if health_check passed.

**Logic:**
1. Builds prompt variables from page data: provider, service_name, url, title, headings, buttons, login_form_present, visible_text
2. Calls LLM via Langfuse + LiteLLM (model: gemini-2.5-flash via proxy)
3. Parses JSON response expecting:
   - `is_customer_facing`: bool
   - `confidence`: int (0-100)
   - `category`: one of `customer_login`, `hrms`, `careers`, `internal_admin`, `vendor_portal`, `marketing_only`, `placeholder`, `unknown`
   - `reason`: str

**Thresholds:**
- `AUDIENCE_DELETE_THRESHOLD = 70`
- `is_customer_facing == False` AND `confidence >= 70` -> **early exit**: `inactive_flagged=True`, `marked_for_deletion=True`, `marked_for_human_review=False`
- `is_customer_facing == False` AND `confidence < 70` -> **continue pipeline**, but set `marked_for_human_review=True`, append audience notes
- `is_customer_facing == True` -> append audience info to notes, continue normally

**Fail-open behavior:** If Langfuse not configured or LLM fails, returns `is_customer_facing=True, confidence=0` so pipeline never deletes a real customer page due to classifier failure.

---

### Phase 2: Two-Stage LLM Match Pipeline (`src/heuristics.py`)

**Entry:** `assess_full_match()` -> orchestrates `extract_and_score()` then `assess_match_with_identifiers()`

#### Stage A — Cheap Extractor LLM (`extract_and_score()`)

**Inputs:** Full visible text (no cap), title, headings, buttons, login_form_present, url, provider, service_name

**What it returns:**
```json
{
  "bank_identifiers": ["Canara Bank", "logo alt text", ...],
  "relevant_page_sections": ["headings about the service"],
  "login_signals": ["password field", "login button"],
  "is_login_page": true,
  "login_type_suggestion": "direct" | "navigation",
  "notes": []
}
```

**Fallback if LLM fails:** Uses `login_form_present` from Playwright — if form exists, `is_login_page=True` and `login_type_suggestion=direct`, else `navigation`.

**Shared hosting awareness:** `_is_shared_host(url)` checks if domain is in a known multi-bank host list (HDFC, SBI, Axis, etc.). This flag is passed to the LLM so it knows domain matching may be unreliable.

#### Stage B — Smart Matcher LLM (`assess_match_with_identifiers()`)

**Inputs:** URL + compact signals from Stage A (NOT raw page text). This saves tokens and gives the smart LLM focused data.

**What it returns:**
```json
{
  "bank_matched": true,
  "service_matched": true,
  "confidence_score": 95,
  "url_confidence_score": 98,
  "login_type": "direct" | "navigation",
  "reason": "Page matches the claimed provider and service.",
  "notes": []
}
```

**Skip conditions:**
- If both `provider` and `service_name` are empty -> skip match, return `bank_matched=True, service_matched=True` with note
- If Langfuse not configured -> skip with warning

---

### Phase 3: Country Match (`src/heuristics.py` -> `assess_country_match()`)

**When:** Only if health_check=True AND provider/service match passed.

**Logic:** Fully deterministic — no LLM.

1. **Resolve expected country** using aliases map ("US" -> "united states", "UK" -> "united kingdom", etc.)
2. **Extract ccTLD** from URL via `tldextract` (e.g., `.in`, `.de`)
3. **Build page zones:** url text (lowercase), title, headings, visible_text (first 6000 chars)
4. **Score expected country:**
   - Strong markers: ccTLD match (+45), country name in URL (+25), in title (+20), in headings (+10)
   - Weak markers: city names, currencies, phone prefixes in any zone (+5 each, max +20)
5. **Score all foreign countries** — find the highest-scoring non-expected country
6. **Decision thresholds:**
   - `foreign_score >= 70` AND `expected_score < 30` -> `matched=False` (definite mismatch, delete)
   - `foreign_score >= 40` AND `expected_score <= foreign_score` -> `matched=None` (uncertain, human review)
   - Otherwise -> `matched=True`

**Supported countries:** 40+ including India, Germany, Spain, US, UK, France, Australia, Canada, Japan, China, Brazil, UAE, Singapore, South Africa, Kenya, Nigeria, Italy, Netherlands, Switzerland, Russia, South Korea, Mexico, Indonesia, Malaysia, Thailand, Turkey, Saudi Arabia, Pakistan, Bangladesh, Sri Lanka, Hong Kong, New Zealand, Poland, Sweden, Norway, Denmark, Finland, Ireland, Portugal, Austria, Belgium, Greece, Egypt, Vietnam, Philippines, Qatar, Bahrain, Kuwait, Oman.

---

### Phase 4: Final Decision Assembly (`src/services/verification_service.py`)

**Priority stack for `final_reason` (highest to lowest):**
1. Token detected in URL
2. Bank mismatch (`bank_match_failed`)
3. Service mismatch (`service_match_failed`)
4. Country mismatch (`country_mismatch`)
5. Uncertain audience (low-confidence non-customer)
6. Match reason / health reason

**Flags:**
| Flag | Condition |
|------|-----------|
| `inactive_flagged` | provider_match_failed OR country_mismatch OR token_detected OR audience_uncertain_review |
| `marked_for_deletion` | token_detected OR bank_match_failed OR country_mismatch |
| `marked_for_human_review` | provider_match_failed OR country_mismatch OR country_uncertain OR token_detected OR audience_uncertain_review |

**Note:** Service mismatch alone does NOT trigger `marked_for_deletion` — only `inactive_flagged + marked_for_human_review`.

---

## QA Report Findings — What the System Reports About URLs

This catalog lists every **verdict, flag, and reason** the QA system produces about the URLs/pages it checks — organized by the type of problem detected on the target website.

---

### A. URL Unreachable / Network Failures

These are reported when the target URL cannot be reached at all.

| QA Finding | Reported Reason | Downstream Flags |
|------------|-----------------|------------------|
| **DNS failure** | Domain does not resolve | `health_check=False`, `inactive_flagged=True`, `marked_for_deletion=True` |
| **Connection refused** | Server actively refuses connection | `health_check=False`, `inactive_flagged=True`, `marked_for_deletion=True` |
| **Connection reset** | TCP reset during handshake | `health_check=False`, `inactive_flagged=True`, `marked_for_deletion=True` |
| **SSL/TLS error** | Certificate invalid or expired | `health_check=False`, `inactive_flagged=True`, `marked_for_deletion=True` |
| **HTTP/2 protocol error** | Low-level HTTP/2 failure | `health_check=False`, `inactive_flagged=True`, `marked_for_deletion=True` |
| **Timeout** | Page load exceeds 30 seconds | `health_check=False`, `inactive_flagged=True`, `marked_for_deletion=True` |
| **Generic connection error** | Any other network failure | `health_check=False`, `inactive_flagged=True`, `marked_for_deletion=True` |

---

### B. HTTP Error Responses

The target server responds but returns an error status code.

| QA Finding | HTTP Status | Reported Reason | Downstream Flags |
|------------|-------------|-----------------|------------------|
| **Not Found** | 404 | `URL returned 404 Not Found` | `health_check=False`, `inactive_flagged=True`, `marked_for_deletion=True` |
| **Permanently Removed** | 410 | `URL returned 410 Gone` | `health_check=False`, `inactive_flagged=True`, `marked_for_deletion=True` |
| **Forbidden** | 403 | `URL returned 403 Forbidden` | `health_check=False`, `inactive_flagged=True`, `marked_for_deletion=True` |
| **Server Error** | 500 / 502 / 503 | `URL returned server error` | `health_check=False`, `inactive_flagged=True`, `marked_for_deletion=True` |
| **Other HTTP error** | Any non-2xx/3xx | `URL returned HTTP {status}` | `health_check=False`, `inactive_flagged=True`, `marked_for_deletion=True` |

---

### C. Content / Page Quality Issues

The URL loads with a healthy HTTP status but the page content is unusable.

| QA Finding | Reported Reason | Downstream Flags |
|------------|-----------------|------------------|
| **Empty server response** | `Server returned an empty (0-byte) response — URL likely deprecated or requires a different host. Needs human review/replacement.` | `health_check=False`, `inactive_flagged=True`, `marked_for_deletion=False`, `marked_for_human_review=True` |
| **Empty extraction** | No visible text, title, headings, or buttons captured | `health_check=True` (HTTP OK), but match scores may be low |
| **Low extraction** | Very sparse content (< 18 effective chars) | `health_check=True`, match confidence likely low |

---

### D. Token / Session Key in URL

The URL itself contains an embedded token that may expire, making it unsuitable for a permanent login link.

| QA Finding | Reported Reason | Downstream Flags |
|------------|-----------------|------------------|
| **JWT in URL** | `URL contains a JWT token` | `marked_for_deletion=True`, `marked_for_human_review=True` |
| **Session parameter** | `query parameter '{name}' is a known token/session key` | `marked_for_deletion=True`, `marked_for_human_review=True` |
| **UUID in query** | `query parameter '{name}' contains a UUID` | `marked_for_deletion=True`, `marked_for_human_review=True` |
| **Long hex token** | `query parameter '{name}' contains a long hex string` | `marked_for_deletion=True`, `marked_for_human_review=True` |
| **Long opaque token** | `query parameter '{name}' contains a long opaque token` | `marked_for_deletion=True`, `marked_for_human_review=True` |
| **SBI transaction flag** | `query parameter '__START_TRAN_FLAG__' is a known token/session key` | `marked_for_deletion=True`, `marked_for_human_review=True` |

---

### E. Page Does Not Match Expected Bank (Phase 2)

The smart matcher LLM determines the page content does not correspond to the claimed bank/provider.

| QA Finding | Reported Reason | Downstream Flags |
|------------|-----------------|------------------|
| **Bank mismatch** | LLM-generated reason (e.g., `Page does not match the expected bank.`) | `inactive_flagged=True`, `marked_for_deletion=True`, `marked_for_human_review=True` |

---

### F. Page Does Not Match Expected Service (Phase 2)

The smart matcher LLM determines the page content does not correspond to the claimed service/product name.

| QA Finding | Reported Reason | Downstream Flags |
|------------|-----------------|------------------|
| **Service mismatch** | LLM-generated reason (e.g., `Page does not match the expected service.`) | `inactive_flagged=True`, `marked_for_human_review=True`, **NOT** `marked_for_deletion` |

---

### G. Page Is Not Customer-Facing (Phase 1.5)

The audience classifier identifies the page as intended for non-customer audiences.

| QA Finding | Reported Reason | Downstream Flags |
|------------|-----------------|------------------|
| **High-confidence non-customer** | `Page is not a customer-facing service ({category}): {reason}` | `inactive_flagged=True`, `marked_for_deletion=True`, `marked_for_human_review=False` |
| **Low-confidence non-customer** | `Page may not be customer-facing ({category}, conf={confidence}) — needs human review. {reason}` | `inactive_flagged=True`, `marked_for_deletion=False`, `marked_for_human_review=True` |

**Categories the classifier can assign:** `customer_login`, `hrms`, `careers`, `internal_admin`, `vendor_portal`, `marketing_only`, `placeholder`, `unknown`

---

### H. Country Mismatch (Phase 3)

The page signals indicate it belongs to a different country than expected.

| QA Finding | Reported Reason | Downstream Flags |
|------------|-----------------|------------------|
| **Definite foreign country** | `Page appears to be for {country} (score {foreign}), not {expected} (score {expected_score}). Likely a foreign-country login.` | `inactive_flagged=True`, `marked_for_deletion=True`, `marked_for_human_review=True` |
| **Uncertain country signals** | `Mixed country signals: expected {expected} ({expected_score}) vs {foreign} ({foreign_score}). Needs human review.` | `inactive_flagged=True`, `marked_for_deletion=False`, `marked_for_human_review=True` |

---

### I. Soft Errors Detected on Page

The page loads but contains error-indicating text in its headings.

| QA Finding | Trigger Text | Downstream Flags |
|------------|--------------|------------------|
| **Page not found text** | "404", "page not found", "resource not found" | Appended to `notes`; does NOT auto-flag unless combined with other issues |
| **Service unavailable** | "service unavailable", "temporarily unavailable" | Appended to `notes` |
| **Access denied** | "not authorized", "access denied" | Appended to `notes` |
| **Invalid request** | "invalid request", "error occurred" | Appended to `notes` |
| **Page moved** | "this page has moved" | Appended to `notes` |

---

### J. Deduplication Findings (Batch Only)

Multiple rows under the same `cb_link_id` are found to represent the same logical login service.

| QA Finding | Reported Reason | Downstream Flags |
|------------|-----------------|------------------|
| **Duplicate — same URL, same name** (Case 1) | `case_1: same URL as canonical row {id}` | `is_duplicate=True`, `dedupe_action='delete_same_url'` |
| **Duplicate — same URL, different names** (Case 3) | `case_3: same URL as canonical row {id}` | `is_duplicate=True`, `dedupe_action='delete_same_url'`, survivor gets `canonical_display_name` |
| **Duplicate — same name, worse URL** (Case 2) | `case_2: same login_service '{name}' as row {id} but lower verification score` | `is_duplicate=True`, `dedupe_action='delete_same_name_worse'` |
| **Missing URL** | `no URL supplied, skipping dedupe grouping` | `dedupe_action='keep'` |

---

### K. Login Page Type Detection

The extractor classifies what kind of login page this is. This is **informational only** — it does not currently trigger any flag.

| QA Finding | Meaning |
|------------|---------|
| **Direct login** | Page contains a login form (password field or login-keyword inputs) |
| **Navigation page** | Page is a landing/menu page that links to the actual login |

---

## Decision Tree — Flag Logic

Below is the exact decision tree the pipeline follows for a single URL. Read it top-to-bottom; the first branch that matches determines the flags.

```
START: verify_url(payload)
│
├─ [Input validation]
│  ├─ url empty → 400 Bad Request (no response object)
│  └─ url invalid (no scheme/host) → 400 Bad Request (no response object)
│
├─ Phase 1: check_url_health(url)
│  ├─ DNS fails → health=INACTIVE, reason=DNS_FAILURE
│  ├─ Playwright network error → health=INACTIVE, reason=TIMEOUT|CONNECTION_REFUSED|...
│  ├─ HTTP error (404|410|403|5xx|...) → health=INACTIVE, reason=NOT_FOUND|FORBIDDEN|...
│  ├─ HTTP OK but EMPTY_SERVER_RESPONSE → health=INACTIVE, reason=EMPTY_SERVER_RESPONSE
│  └─ HTTP OK (2xx/3xx) with content → health=OK, page_result populated
│
├─ Health check FAILED?
│  ├─ YES → inactive_flagged=True, marked_for_human_review=True
│  │      ├─ reason == EMPTY_SERVER_RESPONSE?
│  │      │   ├─ YES → marked_for_deletion=False (needs replacement URL)
│  │      │   └─ NO  → marked_for_deletion=True
│  │      └─ token_detected present? → reason overwritten with token message
│  │
│  └─ NO (health OK) → continue to Phase 1.5
│
├─ Phase 1.5: classify_page_audience()
│  ├─ High-confidence non-customer (confidence >= 70)
│  │   → EARLY EXIT:
│  │      inactive_flagged=True
│  │      marked_for_deletion=True
│  │      marked_for_human_review=False
│  │      health_check=True
│  │      reason = "Page is not a customer-facing service ({category})"
│  │
│  └─ Low-confidence non-customer (confidence 0-69) OR customer-facing
│      → Continue to Phase 2 (audience info stored in notes for final assembly)
│
├─ Phase 2: run_checks() → match_result
│  ├─ match_result is None (provider+service both empty) → bank_matched=True, service_matched=True (skip)
│  ├─ bank_matched=False  → bank_match_failed=True
│  ├─ service_matched=False → service_match_failed=True
│  └─ Both match=True → provider_match_failed=False
│
├─ Phase 3: assess_country_match()
│  ├─ Skipped if: health_check=False OR provider_match_failed=True OR no country provided
│  ├─ matched=False (definite foreign) → country_mismatch=True
│  └─ matched=None (uncertain) → country_uncertain= True
│
├─ Phase 4: Final Decision Assembly
│  │
│  ├─ REASON priority (highest wins):
│  │   1. token_detected → "URL contains an embedded token..."
│  │   2. bank_match_failed → match_result["reason"] or "Page does not match the expected bank."
│  │   3. service_match_failed → match_result["reason"] or "Page does not match the expected service."
│  │   4. country_mismatch → country_check["reason"]
│  │   5. audience_uncertain_review → "Page may not be customer-facing..."
│  │   6. else → match["reason"] or health["reason"]
│  │
│  └─ FLAGS:
│      ├─ inactive_flagged = provider_match_failed
│      │                   OR country_mismatch
│      │                   OR token_detected
│      │                   OR audience_uncertain_review
│      │
│      ├─ marked_for_deletion = token_detected
│      │                      OR bank_match_failed
│      │                      OR country_mismatch
│      │   (NOTE: service_match_failed alone does NOT trigger deletion)
│      │
│      └─ marked_for_human_review = provider_match_failed
│                                   OR country_mismatch
│                                   OR country_uncertain
│                                   OR token_detected
│                                   OR audience_uncertain_review
│
├─ Batch-only: Deduplication (Phase 0)
│  ├─ Phase 0a (pre-verify): detect_duplicates()
│  │   ├─ Same normalized URL → is_duplicate=True, dedupe_action='delete_same_url'
│  │   └─ Different names, same URL → is_duplicate=True, canonical_display_name set on survivor
│  │
│  └─ Phase 0b (post-verify): resolve_same_name_duplicates()
│      └─ Same normalized name, different URLs → is_duplicate=True, dedupe_action='delete_same_name_worse'
│
└─ RESPONSE: ReturnResponse(url, inactive_flagged, reason, health_check,
                            page_match_score, direct_match_score, notes,
                            marked_for_human_review, marked_for_deletion, ...)
```

---

## Flag Cross-Reference Table

| Scenario | `inactive_flagged` | `marked_for_deletion` | `marked_for_human_review` | `health_check` | Typical `reason` |
|----------|-------------------|----------------------|--------------------------|---------------|----------------|
| DNS failure | True | True | True | False | DNS_FAILURE |
| HTTP 404 | True | True | True | False | NOT_FOUND |
| HTTP 403 | True | True | True | False | FORBIDDEN |
| Empty server response (HTTP OK, 0 bytes) | True | **False** | True | False | EMPTY_SERVER_RESPONSE |
| URL has token | True | True | True | True/False | Token in URL |
| High-confidence non-customer | True | True | **False** | True | Not customer-facing |
| Low-confidence non-customer | True | False | True | True | May not be customer-facing |
| Bank mismatch | True | True | True | True | Bank mismatch |
| Service mismatch | True | **False** | True | True | Service mismatch |
| Country mismatch (definite) | True | True | True | True | Foreign country detected |
| Country uncertain | True | False | True | True | Mixed country signals |
| Everything passes | **False** | **False** | **False** | True | Page appears active |

---

## Summary: What Each Flag Means

| Flag | When It Appears | What It Tells the Reviewer |
|------|-----------------|---------------------------|
| `inactive_flagged=True` | Any health, match, country, token, or audience issue | This URL is problematic and should not be treated as active |
| `marked_for_deletion=True` | High-confidence failure (token, bank mismatch, country mismatch, high-conf non-customer) | Safe to auto-remove from the database |
| `marked_for_human_review=True` | Ambiguous or risky result | A human must review before taking action |
| `is_duplicate=True` | Row lost a deduplication tie | This row is redundant; keep the canonical survivor instead |
| `health_check=False` | URL unreachable or returned HTTP error | The page cannot be loaded successfully |
| `page_match_score` (low) | LLM confidence in bank+service match is low | The page may not belong to the claimed provider/service |

---

## Key File Map

| File | What it does |
|------|--------------|
| `app.py` | FastAPI bootstrap, event-loop policy fix, logging setup |
| `src/routers/verification_router.py` | Route definitions (`/check`, `/check/batch`) |
| `src/controllers/verification_controller.py` | Orchestrates batch fetch -> dedupe -> verify per row |
| `src/services/verification_service.py` | Core pipeline: health -> audience -> match -> country -> decision |
| `src/services/analysis_service.py` | Thin wrappers around LLM-based match + audience classification |
| `src/services/duplicate_service.py` | Dedupe logic (Phase 0/1/2/3) |
| `src/utils/url_health.py` | Token detection, DNS check, health mapping, soft-error scan |
| `src/page_extraction.py` | Playwright browser control, multi-stage extraction, iframe handling, login form detection, quality scoring |
| `src/heuristics.py` | Two-stage LLM match (extractor + matcher), audience classifier, country match (deterministic), service name QA |
| `src/db.py` | Async Postgres fetch for cb_link_id rows |
| `src/models/request_models.py` | Pydantic request schemas |
| `src/models/response_models.py` | Pydantic response schemas |
