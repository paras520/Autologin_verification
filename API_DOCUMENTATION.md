# API Documentation — Autologin Verification Service

> This doc describes the **public API behavior**, request/response contracts, and the internal verification pipeline so any developer can integrate with or maintain the service without needing to read the DeepEval / prompt-optimization code.

---

## 1. Overview

The Autologin Verification API checks whether a bank login URL is still valid, matches the expected bank/service, and is customer-facing. It is built with **FastAPI** and exposes two HTTP endpoints.

**Entry point:** `app.py`  
**Router:** `src/routers/verification_router.py`  
**Controller:** `src/controllers/verification_controller.py`  
**Core service:** `src/services/verification_service.py`

---

## 2. Endpoints

### `POST /check` — Single URL check (legacy)

Runs the full verification pipeline on one URL.

#### Request body (`CheckRequest`)

```json
{
  "provider": "Bank of Baroda",
  "service_name": "BOB iBanking",
  "login_type": "Direct",
  "url": "https://example.com/login",
  "country": "India",
  "cb_link_id": ""
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | string | yes | Bank / institution name (used for matching) |
| `service_name` | string | yes | Product/portal name (used for matching) |
| `login_type` | string | yes | e.g. `Direct`, `OAuth` |
| `url` | string | yes | URL to verify |
| `country` | string | no | Expected country (used in Phase 3 country match) |
| `cb_link_id` | string | no | Internal grouping id |

#### Response (`ReturnResponse`)

```json
{
  "url": "https://example.com/login",
  "inactive_flagged": false,
  "reason": "Page appears active.",
  "health_check": true,
  "page_match_score": 82,
  "direct_match_score": 82,
  "notes": "langfuse_session_id=test-session",
  "updated_name": null,
  "marked_for_human_review": false,
  "marked_for_deletion": false,
  "errors": "",
  "time": "2026-03-10T12:00:00"
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `url` | string | The URL that was checked |
| `inactive_flagged` | bool | **True** when any check failed (health, match, country, token, or uncertain audience) |
| `reason` | string \| null | Human-readable explanation of the final verdict |
| `health_check` | bool | Whether the URL returned a healthy HTTP status (2xx / 3xx) |
| `page_match_score` | int \| null | LLM confidence (0-100) that the page matches expected bank **and** service |
| `direct_match_score` | int \| null | Currently mirrors `page_match_score` |
| `notes` | string \| null | Pipe-delimited diagnostic strings (soft errors, token info, match notes, country notes, audience info) |
| `updated_name` | string \| null | Reserved for future name-correction feature |
| `marked_for_human_review` | bool | **True** when the result is ambiguous and needs a human to decide |
| `marked_for_deletion` | bool | **True** when the row should be removed from the database |
| `errors` | string | Raw error text (usually empty on success) |
| `time` | string | ISO-8601 timestamp of response generation |

#### Error responses

| Scenario | Status | Detail |
|----------|--------|--------|
| Missing / empty `url` | `400 Bad Request` | `Request body must include a non-empty 'url' field.` |
| Invalid URL (no scheme or host) | `400 Bad Request` | `URL must include a valid scheme and host.` |
| Unexpected server error | `500 Internal Server Error` | `Internal server error while processing /check request.` |

---

### `POST /check/batch` — Batch check by `cb_link_id`

Fetches all rows belonging to one or more `cb_link_id`s from the database, runs deduplication, then verifies each non-duplicate row.

#### Request body (`BatchCheckRequest`)

```json
{
  "cb_link_ids": ["B-IN-i9i6wf", "B-IN-SORltG"],
  "include_inactive": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `cb_link_ids` | string[] | yes | One or more ids to process |
| `include_inactive` | bool | no | Whether to also fetch rows with `status != 'active'` (default `true`) |

#### Response (`BatchCheckResponse`)

```json
{
  "results": [
    {
      "cb_link_id": "B-IN-i9i6wf",
      "total_rows": 5,
      "rows": [
        {
          "cb_link_id": "B-IN-i9i6wf",
          "login_service": "BOB iBanking",
          "url": "https://example.com/login",
          "health_check": true,
          "page_match_score": 82,
          "direct_match_score": 82,
          "display_name_score": null,
          "notes": "...",
          "inactive_flagged": false,
          "marked_for_deletion": false,
          "marked_for_human_review": false,
          "is_duplicate": false,
          "duplicate_of_url": null,
          "duplicate_of_id": null,
          "dedupe_action": "keep",
          "dedupe_reason": null,
          "canonical_display_name": null,
          "reason": "Page appears active.",
          "status": "active"
        }
      ]
    }
  ]
}
```

**Key differences from `/check` response:**
- Results are grouped per `cb_link_id`.
- Each row includes **deduplication fields** (`is_duplicate`, `duplicate_of_id`, `duplicate_of_url`, `dedupe_action`, `dedupe_reason`, `canonical_display_name`).
- Duplicate rows skip the expensive verification pipeline; their check fields (`health_check`, `page_match_score`, etc.) are `null`.

#### Deduplication actions

| `dedupe_action` | Meaning |
|-----------------|---------|
| `keep` | Survivor — this row should be retained |
| `delete_same_url` | Another row shares the same normalized URL; this one is the loser |
| `delete_same_name_worse` | Same service name but a different URL with a worse verification score |

#### Error responses

| Scenario | Status | Detail |
|----------|--------|--------|
| `cb_link_ids` empty or missing | `422 Unprocessable Entity` | FastAPI validation error |
| Unexpected server error | `500 Internal Server Error` | `Internal server error while processing /check/batch request.` |

---

## 3. Verification Pipeline (single URL)

When you call `/check` (or a non-duplicate row inside `/check/batch`), the following phases run in order:

```
Phase 1   URL Health Check  →  scrape page, check HTTP status, detect tokens
Phase 1.5 Audience Classifier →  LLM decides if page is customer-facing
Phase 2   Two-Stage Match   →  cheap extractor + smart LLM (bank + service match)
Phase 3   Country Match     →  deterministic heuristics (if prior phases passed)
Phase 4   Final Decision    →  assemble flags, reason, notes
```

### Phase 1 — URL Health Check (`src/utils/url_health.py`)

- Sends a headless-browser request (Playwright) with stealth headers.
- Returns HTTP status, load time, and a `page_result` dict containing visible text, title, links, and meta tags.
- Detects **tokens in the URL** (session ids, auth codes, etc.).
- **Early exit:** If the URL is unreachable, returns `health_check=false`, `inactive_flagged=true`, `marked_for_deletion=true` (except for empty-server responses, which go to human review).

### Phase 1.5 — Audience Classifier (`src/services/analysis_service.py`)

- Sends scraped page data to an LLM prompt (`customer_facing_classifier`).
- Returns:
  - `is_customer_facing` (`true` / `false`)
  - `confidence` (0-100)
  - `category` — e.g. `customer_login`, `hrms`, `careers`, `internal_admin`, `vendor_portal`, `marketing_only`, `placeholder`, `unknown`
- **Early exit (high-confidence non-customer):** If `is_customer_facing == false` **and** `confidence >= 70`, the pipeline short-circuits with `inactive_flagged=true`, `marked_for_deletion=true`, `marked_for_human_review=false`.
- **Human review (low-confidence non-customer):** If `is_customer_facing == false` **and** `confidence < 70`, the pipeline continues but sets `marked_for_human_review=true` and appends audience notes.

### Phase 2 — Two-Stage Match (`src/utils/heuristics.py`)

1. **Cheap extractor LLM** — extracts signals from the page (bank mentions, service mentions, login type indicators).
2. **Smart matcher LLM** — uses the extracted signals plus the expected `provider` and `service_name` to return:
   - `bank_matched` (bool)
   - `service_matched` (bool)
   - `confidence_score` (0-100)
   - `reason` and `notes`

If either match fails, `inactive_flagged` becomes `true` and `marked_for_deletion` is set for bank mismatches (service mismatches only flag inactive + human review).

### Phase 3 — Country Match (`src/utils/heuristics.py`)

Only runs when `health_check=true` and the provider/service match passed.

- Heuristic scan of the page text for country indicators (domain TLDs, phone prefixes, address snippets, currency symbols).
- Returns `matched` (`true` / `false` / `null` for uncertain).
- A definite mismatch sets `inactive_flagged=true`, `marked_for_deletion=true`, and `marked_for_human_review=true`.

### Phase 4 — Final Decision Assembly

The controller builds the final `ReturnResponse` by combining all phase outputs. Priority of the `reason` field (highest to lowest):

1. Token detected in URL
2. Bank mismatch
3. Service mismatch
4. Country mismatch
5. Uncertain audience (low-confidence non-customer)
6. Match reason / health reason

`marked_for_deletion` is `true` for: token, bank mismatch, country mismatch.  
`marked_for_human_review` is `true` for: any match failure, country mismatch/uncertainty, token, or uncertain audience.

---

## 4. Deduplication Pipeline (batch only)

Before any verification runs, rows sharing a `cb_link_id` go through deduplication in **two phases**:

### Phase 1 — Same normalized URL (`detect_duplicates`)

- Groups rows by normalized URL (strips ephemeral query params, lowercases domain, drops trailing slash).
- Within each group, picks a **canonical survivor** using priority:
  1. `status == 'active'`
  2. Lower `sorting_order`
  3. More descriptive display name
  4. Stable tiebreak by row `id`
- **Case 1** (same name + same URL): losers marked `delete_same_url`.
- **Case 3** (different names + same URL): losers marked `delete_same_url`; survivor gets `canonical_display_name` set to the best name in the group.

### Phase 2 — Same name, different URL (`resolve_same_name_duplicates`)

- Runs **after** verification so real scores are available.
- Groups surviving rows by normalized `login_service` name.
- If the group has 2+ rows with **different** normalized URLs, picks a winner using:
  1. Not `marked_for_deletion`
  2. `health_check == true`
  3. Higher `page_match_score`
  4. Lower `sorting_order`
- Losers are marked `delete_same_name_worse`.

### Phase 3 — Parent / Child URLs (`resolve_parent_child`)

Currently a **stub**. The DB schema does not expose a `parent_id` column, so no parent/child resolution happens yet. The integration point exists for future implementation.

---

## 5. Flag Behaviour Summary

| Flag | Set when | Downstream action |
|------|----------|-------------------|
| `inactive_flagged` | Any health, match, country, token, or audience issue | UI should grey-out or warn |
| `marked_for_deletion` | High-confidence failure (token, bank mismatch, country mismatch, high-conf non-customer) | Safe to auto-delete |
| `marked_for_human_review` | Ambiguous or risky result (service mismatch, country uncertainty, low-conf non-customer, token) | Queue for manual review |
| `is_duplicate` | Row lost a dedupe tie | Skip verification; do not display separately |

---

## 6. Running the API locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure `.env` is present (DATABASE_URL, LiteLLM / Langfuse keys, etc.)

# 3. Start server
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 5000 --loop asyncio
```

On Windows, `app.py` automatically switches the asyncio event-loop policy to `WindowsSelectorEventLoopPolicy` so Playwright can spawn Chromium subprocesses.

---

## 7. Existing API-level Tests

File: `tests/test_app.py`

| Test | What it verifies |
|------|------------------|
| `test_check_endpoint_requires_request_fields` | `POST /check` with empty body returns `422` |
| `test_check_endpoint_returns_service_response` | Mocked successful verification returns `200` and correct JSON shape |
| `test_check_endpoint_maps_service_value_error_to_http_400` | Invalid URL input returns `400` with correct detail message |

These are lightweight FastAPI `TestClient` tests that mock `verify_url` so they run without a database or browser.

---

## 8. Key Files Map

| File | Role |
|------|------|
| `app.py` | FastAPI bootstrap, event-loop policy fix, logging setup |
| `src/routers/verification_router.py` | Route definitions (`/check`, `/check/batch`) |
| `src/controllers/verification_controller.py` | Orchestrates batch fetch → dedupe → verify per row |
| `src/services/verification_service.py` | Core pipeline: health → audience → match → country → decision |
| `src/services/analysis_service.py` | Thin wrappers around LLM-based match + audience classification |
| `src/services/duplicate_service.py` | Dedupe logic (Phase 1 / 2 / 3) |
| `src/models/request_models.py` | Pydantic request schemas |
| `src/models/response_models.py` | Pydantic response schemas (`ReturnResponse`, `RowResult`, `CbLinkResult`, `BatchCheckResponse`) |
| `src/db.py` | Async DB fetch for `cb_link_id` rows |
| `tests/test_app.py` | FastAPI endpoint smoke tests |
