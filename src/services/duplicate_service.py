"""Duplicate detection and resolution for login_services rows sharing a cb_link_id.

Implements the four Jira dedupe cases in two phases so the pipeline can skip
verifying known duplicates and still use real verification scores to break
same-name ties:

  Phase 1 (pre-verification, detect_duplicates):
    Case 1  same name + same URL                    -> keep one by URL
    Case 3  different names + same URL              -> keep one, standardize name

  Phase 2 (post-verification, resolve_same_name_duplicates):
    Case 2  same name + different URLs              -> keep valid/preferred URL
             (decided by health_check, page_match_score, marked_for_deletion)

  Phase 3 (Case 4, resolve_parent_child):
    Parent + Child URLs                             -> STUB (no parent_id column
             in current schema; helper is pass-through so integration point
             exists when schema is confirmed).

Every row gets `dedupe_action` set to exactly one of:
    keep, delete_same_url, delete_same_name_worse
`is_duplicate=True` iff dedupe_action != 'keep'. No DB mutation happens here --
downstream consumers decide what "cleaning" means based on these flags.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

logger = logging.getLogger("autologin.duplicate_service")

# Query-param names that are ephemeral and should be stripped before comparing URLs
_EPHEMERAL_PARAMS = re.compile(
    r"^(token|auth|session|sess|sid|key|apikey|api_key|access_token|"
    r"refresh_token|id_token|csrf|xsrf|nonce|otp|ticket|code|sso|"
    r"jsessionid|phpsessid|asp\.net_sessionid|__start_tran_flag__|"
    r"saml|assertion|bearer|signature|sig|hmac|hash|digest|"
    r"utm_source|utm_medium|utm_campaign|utm_term|utm_content)$",
    re.IGNORECASE,
)

# Generic words that shouldn't bias canonical-name selection (Case 3)
_NAME_NOISE_WORDS: frozenset[str] = frozenset({
    "login", "online", "portal", "banking", "bank", "internet", "secure",
    "authenticated", "authentication", "access", "signin", "sign-in",
    "loginpage", "loginportal", "website", "site", "home", "homepage",
})

# Actions on a row after dedupe runs. Exactly one per row.
ACTION_KEEP = "keep"
ACTION_DELETE_SAME_URL = "delete_same_url"
ACTION_DELETE_SAME_NAME_WORSE = "delete_same_name_worse"


# ---------------------------------------------------------------------------
# URL + name normalization
# ---------------------------------------------------------------------------

def _normalize_url(url: str) -> str:
    """Lowercase the domain, strip ephemeral query params, drop trailing slash."""
    if not url:
        return ""
    parsed = urlparse(url.strip())
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"

    remaining_params = {
        k: v
        for k, v in parse_qs(parsed.query, keep_blank_values=True).items()
        if not _EPHEMERAL_PARAMS.match(k)
    }
    stable_query = urlencode(
        {k: remaining_params[k][0] for k in sorted(remaining_params)},
        safe="",
    )

    normalized = urlunparse((parsed.scheme.lower(), netloc, path, "", stable_query, ""))
    return normalized


def _normalize_name(name: str) -> str:
    """Lowercase + collapse whitespace so 'Bob World' == 'bob  world'."""
    if not name:
        return ""
    return re.sub(r"\s+", " ", name.strip().lower())


def _name_signal_score(name: str) -> tuple[int, int]:
    """Scoring tuple for canonical display-name selection.

    Returned as (non_noise_word_count, total_length) so it can be compared
    directly with max(). A name with more non-noise words beats a short one;
    within the same non-noise count, the longer name wins (more descriptive).
    """
    if not name:
        return (0, 0)
    tokens = re.findall(r"[a-zA-Z0-9]+", name)
    non_noise = sum(1 for t in tokens if t.lower() not in _NAME_NOISE_WORDS)
    return (non_noise, len(name))


# ---------------------------------------------------------------------------
# Phase 1 — same normalized URL (Cases 1 + 3)
# ---------------------------------------------------------------------------

def _canonical_row_for_url_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the row to keep when multiple rows share a normalized URL.

    Priority (higher wins):
      1. status == 'active' beats anything else
      2. lower sorting_order wins (None treated as +inf)
      3. longer non-noise display_name wins (then longer overall)
      4. stable tiebreak by string id
    """
    def _key(row: dict[str, Any]) -> tuple:
        status_active = 1 if (row.get("status") or "").lower() == "active" else 0
        sorting_order = row.get("sorting_order")
        # invert sorting_order so lower wins under max()
        sorting_priority = -sorting_order if isinstance(sorting_order, int) else -10**9
        name_score = _name_signal_score(row.get("display_name") or row.get("login_service") or "")
        id_tiebreak = str(row.get("id") or "")
        return (status_active, sorting_priority, name_score, id_tiebreak)

    return max(group, key=_key)


def _canonical_display_name(group: list[dict[str, Any]]) -> str | None:
    """Pick the best-looking display name from a group of rows (Case 3)."""
    best_name: str | None = None
    best_score: tuple[int, int] = (-1, -1)
    for row in group:
        candidate = row.get("display_name") or row.get("login_service") or ""
        if not candidate:
            continue
        score = _name_signal_score(candidate)
        if score > best_score:
            best_score = score
            best_name = candidate
    return best_name


def detect_duplicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Phase 1 (pre-verification). Groups rows by normalized URL; resolves
    Case 1 (same name, same URL) and Case 3 (different names, same URL).

    Injects these keys on every row (backwards compatible with the previous
    flag-only version):
        is_duplicate: bool
        duplicate_of_id: str | None
        duplicate_of_url: str | None
        dedupe_action: 'keep' | 'delete_same_url'
        dedupe_reason: str
        canonical_display_name: str | None   (only populated on the survivor
                                              of a Case 3 group)

    Rows with no URL at all are kept (dedupe_action='keep') and skip grouping.
    """
    # First pass: group rows by normalized URL
    groups: dict[str, list[dict[str, Any]]] = {}
    missing_url_rows: list[dict[str, Any]] = []

    for row in rows:
        norm = _normalize_url(row.get("login_url") or "")
        if not norm:
            missing_url_rows.append(row)
            continue
        groups.setdefault(norm, []).append(row)

    # Initialize all rows with the default 'keep' action (Phase 1 defaults)
    for row in rows:
        row["is_duplicate"] = False
        row["duplicate_of_id"] = None
        row["duplicate_of_url"] = None
        row["dedupe_action"] = ACTION_KEEP
        row["dedupe_reason"] = None
        row["canonical_display_name"] = None

    for row in missing_url_rows:
        row["dedupe_reason"] = "no URL supplied, skipping dedupe grouping"

    # Second pass: within each URL group, pick a canonical row
    for norm_url, group in groups.items():
        if len(group) == 1:
            continue

        canonical = _canonical_row_for_url_group(group)
        canonical_id = str(canonical.get("id") or "")
        canonical_url = canonical.get("login_url") or norm_url

        # Detect whether this is Case 1 (all same name) or Case 3 (different names)
        normalized_names = {_normalize_name(r.get("login_service") or "") for r in group}
        is_case_3 = len(normalized_names) > 1
        case_tag = "case_3" if is_case_3 else "case_1"

        # Case 3: also pick the canonical display name across the group
        if is_case_3:
            canonical.setdefault("canonical_display_name", None)
            canonical["canonical_display_name"] = _canonical_display_name(group)

        for row in group:
            if row is canonical:
                # Survivor keeps action=keep; annotate why
                existing = row.get("dedupe_reason")
                row["dedupe_reason"] = (
                    f"{case_tag}: canonical survivor of {len(group)}-row "
                    f"group sharing URL"
                    + (f" ({existing})" if existing else "")
                )
                continue

            row["is_duplicate"] = True
            row["duplicate_of_id"] = canonical_id
            row["duplicate_of_url"] = canonical_url
            row["dedupe_action"] = ACTION_DELETE_SAME_URL
            row["dedupe_reason"] = (
                f"{case_tag}: same URL as canonical row {canonical_id}"
            )
            logger.info(
                "[dedupe] %s: row %s duplicate of %s (url=%s)",
                case_tag, row.get("id"), canonical_id, row.get("login_url"),
            )

    return rows


# ---------------------------------------------------------------------------
# Phase 2 — same normalized name with different URLs (Case 2)
# ---------------------------------------------------------------------------

def _case2_primary_score(row: dict[str, Any]) -> tuple:
    """Primary scoring tuple for Case 2. Higher beats lower under max().

    Tuple order:
        1. not marked_for_deletion   (True=1 beats False=0)
        2. health_check True         (True=1 beats False/None=0)
        3. page_match_score          (None treated as -1)
        4. lower sorting_order       (inverted so higher key = lower order)

    Ties are broken separately using the lexicographically lowest id.
    """
    marked_for_del = bool(row.get("marked_for_deletion"))
    health_ok = row.get("health_check") is True
    score = row.get("page_match_score")
    score_val = score if isinstance(score, (int, float)) else -1
    sorting_order = row.get("sorting_order")
    sorting_priority = -sorting_order if isinstance(sorting_order, int) else -10**9
    return (
        1 if not marked_for_del else 0,
        1 if health_ok else 0,
        score_val,
        sorting_priority,
    )


def _pick_case2_winner(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the Case 2 winner from a group of rows sharing a login_service.

    Two-stage: first find the max primary score, then tiebreak by the
    lexicographically lowest id string so results are deterministic.
    """
    best_primary = max(_case2_primary_score(r) for r in group)
    top = [r for r in group if _case2_primary_score(r) == best_primary]
    return min(top, key=lambda r: str(r.get("id") or ""))


def resolve_same_name_duplicates(row_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Phase 2 (post-verification). Among rows that survived Phase 1
    (dedupe_action == 'keep'), groups by normalized `login_service` name.
    If two or more share the name but have *different* normalized URLs,
    picks the winner using verification scores and marks the losers
    `dedupe_action='delete_same_name_worse'`.

    Inputs are dicts (already-dumped RowResult or raw dicts). The same list
    is returned with action/reason fields updated in place.

    Inputs read per row:
        login_service, url, id, health_check, page_match_score,
        marked_for_deletion, sorting_order, dedupe_action

    Mutates per loser:
        is_duplicate=True, dedupe_action='delete_same_name_worse',
        duplicate_of_id=<winner.id>, duplicate_of_url=<winner.url>,
        dedupe_reason=<why winner won>
    """
    # Only consider rows that survived Phase 1; Phase 1 losers already have
    # their duplicate_of set and shouldn't be reconsidered.
    eligible = [r for r in row_results if r.get("dedupe_action") == ACTION_KEEP]

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        name = _normalize_name(row.get("login_service") or "")
        if not name:
            continue
        groups.setdefault(name, []).append(row)

    for name, group in groups.items():
        if len(group) < 2:
            continue

        # Only act if URLs actually differ after normalization — otherwise
        # Phase 1 would already have handled it.
        distinct_urls = {_normalize_url(r.get("url") or r.get("login_url") or "") for r in group}
        if len(distinct_urls) < 2:
            continue

        winner = _pick_case2_winner(group)
        winner_id = str(winner.get("id") or "")
        winner_url = winner.get("url") or winner.get("login_url") or ""
        winner_score = winner.get("page_match_score")
        winner_health = winner.get("health_check")
        winner_marked = winner.get("marked_for_deletion")

        for row in group:
            if row is winner:
                continue
            row["is_duplicate"] = True
            row["duplicate_of_id"] = winner_id
            row["duplicate_of_url"] = winner_url
            row["dedupe_action"] = ACTION_DELETE_SAME_NAME_WORSE
            row["dedupe_reason"] = (
                f"case_2: same login_service '{name}' as row {winner_id} "
                f"but lower verification score "
                f"(winner health={winner_health} score={winner_score} "
                f"deleted={winner_marked}; "
                f"this health={row.get('health_check')} "
                f"score={row.get('page_match_score')} "
                f"deleted={row.get('marked_for_deletion')})"
            )
            logger.info(
                "[dedupe] case_2: row %s duplicate of %s (name=%r)",
                row.get("id"), winner_id, name,
            )

    return row_results


# ---------------------------------------------------------------------------
# Phase 3 — Parent vs Child URLs (Case 4)  — STUB
# ---------------------------------------------------------------------------

def resolve_parent_child(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Phase 3 (Case 4) — STUB.

    The current DB schema (see `src/db.py`) does not SELECT a `parent_id`
    column and the offline dump has zero parent/child instances, so
    implementing parent/child resolution today would be guesswork.

    Returns rows unchanged. The integration point exists so once the DB
    schema is confirmed (or URL-prefix inference is approved) the logic can
    be added here without touching callers.
    """
    # TODO: implement once schema confirmed. Candidate policy:
    #   - if a row's normalized URL is a strict path prefix of another row's,
    #     treat the shorter URL as the parent and keep either parent-only
    #     or child-only per business rule.
    return rows
