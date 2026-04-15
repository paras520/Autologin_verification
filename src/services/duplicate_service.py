"""Duplicate detection across login_services rows for a single cb_link_id.

Two rows are considered duplicates when they share the same resolved
final URL after stripping ephemeral query parameters (tokens, sessions, etc.).

The first-seen row (lowest index, i.e. most recently created per DB ordering)
is kept as the canonical entry; every subsequent row pointing to the same
destination is marked as a duplicate of it.
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


def detect_duplicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark duplicate rows within a single cb_link_id batch.

    Rows arrive in DB order (most recently created first, per ORDER BY created_at DESC).
    The first row with a given normalized URL is canonical; duplicates after it
    get is_duplicate=True and duplicate_of_id pointing to the canonical row's id.

    Returns the same list with two extra keys injected per row:
        is_duplicate     bool
        duplicate_of_id  str | None
    """
    seen: dict[str, tuple[str, str]] = {}  # normalized_url -> (canonical row id, original url)

    for row in rows:
        url = row.get("login_url") or ""
        norm = _normalize_url(url)

        if not norm:
            row["is_duplicate"] = False
            row["duplicate_of_id"] = None
            row["duplicate_of_url"] = None
            logger.debug("[dupe] row %s has no URL — skipping dupe check", row.get("id"))
            continue

        if norm in seen:
            canonical_id, canonical_url = seen[norm]
            row["is_duplicate"] = True
            row["duplicate_of_id"] = canonical_id
            row["duplicate_of_url"] = canonical_url
            logger.info(
                "[dupe] row %s is a duplicate of %s  (url=%s)",
                row.get("id"), canonical_id, url,
            )
        else:
            seen[norm] = (str(row["id"]), url)
            row["is_duplicate"] = False
            row["duplicate_of_id"] = None
            row["duplicate_of_url"] = None

    return rows
