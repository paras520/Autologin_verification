"""Heuristics for URL verification checks.

LLM-based checks use Langfuse + LiteLLM for intelligent analysis.
The country-match check is fully deterministic (no LLM call).

Prompt paths are configurable through environment variables so you can manage
and version them in Langfuse independently.

Expected env vars (on top of the shared Langfuse credentials):
  LANGFUSE_DIRECT_LOGIN_PROMPT   — default: test/autologin_Direct_Login_Check
  LANGFUSE_PROVIDER_MATCH_PROMPT — default: test/autologin_service_matcher
"""

from __future__ import annotations

import json
import logging
import os
import re
from urllib.parse import urlparse
from uuid import uuid4

try:
    import tldextract
except ImportError:
    tldextract = None

logger = logging.getLogger("autologin.heuristics")

DEFAULT_DIRECT_LOGIN_PROMPT = "test/autologin_Direct_Login_Check"
DEFAULT_PROVIDER_MATCH_PROMPT = "test/autologin_service_matcher"

# Cap visible text sent to these LLM checks (separate from the main LLM limit)
_HEURISTIC_VISIBLE_TEXT_LIMIT = 10_000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_text(value) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if item is not None)
    elif value is None:
        value = ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _langfuse_is_configured() -> bool:
    return all(
        os.getenv(k)
        for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")
    )


def _load_langfuse_helpers():
    from langfuse_helper import (
        build_messages,
        call_litellm,
        get_prompts_from_langfuse,
        parse_response,
    )
    return build_messages, call_litellm, get_prompts_from_langfuse, parse_response


def _build_page_variables(provider: str, service_name: str, page_result: dict) -> dict:
    """Build the template variables dict that will be injected into Langfuse prompts."""
    visible_text = _normalize_text(page_result.get("visible_text"))
    return {
        "provider": provider or "",
        "service_name": service_name or "",
        "url": page_result.get("final_url") or "",
        "page_title": _normalize_text(page_result.get("title")),
        "headings": json.dumps(page_result.get("headings") or [], ensure_ascii=False),
        "buttons": json.dumps(page_result.get("buttons") or [], ensure_ascii=False),
        "login_form_present": str(page_result.get("login_form_present", False)).lower(),
        "visible_text": visible_text[:_HEURISTIC_VISIBLE_TEXT_LIMIT],
    }


def _parse_notes(raw) -> list[str]:
    """Ensure notes is always a flat list of strings."""
    if isinstance(raw, list):
        return [str(n) for n in raw if n]
    if raw:
        return [str(raw)]
    return []


# ---------------------------------------------------------------------------
# Direct login-page check
# ---------------------------------------------------------------------------

async def assess_direct_login_page(
    provider: str,
    service_name: str,
    page_result: dict,
) -> dict:
    """Ask the LLM whether *page_result* represents a login portal.

    Returns:
        {
          "is_login_page": bool,
          "score": int  (0-100),
          "reason": str,
          "notes": list[str],
        }
    """
    prompt_path = os.getenv("LANGFUSE_DIRECT_LOGIN_PROMPT", DEFAULT_DIRECT_LOGIN_PROMPT)

    if not _langfuse_is_configured():
        logger.warning("[direct-login] Langfuse not configured — skipping")
        return {
            "is_login_page": False,
            "score": 0,
            "reason": "Langfuse not configured; direct login check skipped.",
            "notes": ["direct-login LLM check skipped — Langfuse credentials missing"],
        }

    session_id = f"direct-login-{uuid4()}"
    variables = _build_page_variables(provider, service_name, page_result)

    logger.info(
        "[direct-login] Calling LLM  prompt=%s  session=%s  provider=%s  service=%s",
        prompt_path, session_id, provider, service_name,
    )

    try:
        build_messages, call_litellm, get_prompts_from_langfuse, parse_response = (
            _load_langfuse_helpers()
        )

        system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
            prompt_path=prompt_path,
            session_id=session_id,
            variables=variables,
        )

        messages = build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        logger.debug("[direct-login] Sending %d messages to model=%s", len(messages), config.get("model"))

        response = await call_litellm(
            config=config,
            messages=messages,
            session_id=session_id,
            api_endpoint="/check/direct-login",
            tag_suffix="2",
            prompt=prompt_obj,
        )

        parsed = parse_response(response, has_functions=False, has_tools=False)
        logger.info("[direct-login] Raw LLM response: %s", str(parsed)[:500])

        if isinstance(parsed, dict):
            is_login = bool(parsed.get("is_login_page", False))
            score = max(0, min(int(parsed.get("score", 0)), 100))
            reason = parsed.get("reason") or (
                "Page identified as a login portal."
                if is_login
                else "Page does not appear to be a login portal."
            )
            notes = _parse_notes(parsed.get("notes"))
            notes.append(f"langfuse_session_id={session_id}")

            logger.info(
                "[direct-login] Result: is_login_page=%s  score=%d  reason=%s",
                is_login, score, reason,
            )
            return {
                "is_login_page": is_login,
                "score": score,
                "reason": reason,
                "notes": notes,
            }

        logger.warning("[direct-login] LLM returned unstructured response: %s", str(parsed)[:300])
        return {
            "is_login_page": False,
            "score": 0,
            "reason": f"Direct login LLM returned unstructured response: {str(parsed)[:300]}",
            "notes": [f"langfuse_session_id={session_id}"],
        }

    except Exception as exc:
        logger.error("[direct-login] LLM call failed: %s", exc, exc_info=True)
        return {
            "is_login_page": False,
            "score": 0,
            "reason": f"Direct login LLM check failed: {exc}",
            "notes": [f"llm_error={exc}", f"langfuse_session_id={session_id}"],
        }


# ---------------------------------------------------------------------------
# Provider / service-name page-match check
# ---------------------------------------------------------------------------

async def assess_provider_service_match(
    provider: str,
    service_name: str,
    page_result: dict,
) -> dict:
    """Ask the LLM whether *page_result* belongs to the claimed provider/service.

    Returns:
        {
          "matched": bool,
          "score": int  (0-100),
          "reason": str,
          "notes": list[str],
        }
    """
    if not provider and not service_name:
        logger.info("[provider-match] Skipped — both provider and service_name are empty")
        return {
            "matched": True,
            "score": 0,
            "reason": "No provider or service name provided — match check skipped.",
            "notes": ["provider/service match skipped — both fields empty"],
        }

    prompt_path = os.getenv(
        "LANGFUSE_PROVIDER_MATCH_PROMPT", DEFAULT_PROVIDER_MATCH_PROMPT
    )

    if not _langfuse_is_configured():
        logger.warning("[provider-match] Langfuse not configured — skipping")
        return {
            "matched": True,
            "score": 0,
            "reason": "Langfuse not configured; provider/service match check skipped.",
            "notes": ["provider/service LLM check skipped — Langfuse credentials missing"],
        }

    session_id = f"provider-match-{uuid4()}"
    variables = _build_page_variables(provider, service_name, page_result)

    logger.info(
        "[provider-match] Calling LLM  prompt=%s  session=%s  provider=%s  service=%s",
        prompt_path, session_id, provider, service_name,
    )

    try:
        build_messages, call_litellm, get_prompts_from_langfuse, parse_response = (
            _load_langfuse_helpers()
        )

        system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
            prompt_path=prompt_path,
            session_id=session_id,
            variables=variables,
        )

        messages = build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        logger.debug("[provider-match] Sending %d messages to model=%s", len(messages), config.get("model"))

        response = await call_litellm(
            config=config,
            messages=messages,
            session_id=session_id,
            api_endpoint="/check/provider-match",
            tag_suffix="3",
            prompt=prompt_obj,
        )

        parsed = parse_response(response, has_functions=False, has_tools=False)
        logger.info("[provider-match] Raw LLM response: %s", str(parsed)[:500])

        if isinstance(parsed, dict):
            matched = bool(parsed.get("matched", False))
            score = max(0, min(int(parsed.get("score", 0)), 100))
            reason = parsed.get("reason") or (
                "Page content matches the claimed provider/service."
                if matched
                else "Page content does NOT match the claimed provider/service."
            )
            notes = _parse_notes(parsed.get("notes"))
            notes.append(f"langfuse_session_id={session_id}")

            logger.info(
                "[provider-match] Result: matched=%s  score=%d  reason=%s",
                matched, score, reason,
            )
            return {
                "matched": matched,
                "score": score,
                "reason": reason,
                "notes": notes,
            }

        logger.warning("[provider-match] LLM returned unstructured response: %s", str(parsed)[:300])
        return {
            "matched": True,
            "score": 0,
            "reason": f"Provider match LLM returned unstructured response: {str(parsed)[:300]}",
            "notes": [f"langfuse_session_id={session_id}"],
        }

    except Exception as exc:
        logger.error("[provider-match] LLM call failed: %s", exc, exc_info=True)
        return {
            "matched": True,
            "score": 0,
            "reason": f"Provider/service match LLM check failed: {exc}",
            "notes": [f"llm_error={exc}", f"langfuse_session_id={session_id}"],
        }


# ---------------------------------------------------------------------------
# Service name QA check (deterministic — no LLM, read-only / pass-fail only)
# ---------------------------------------------------------------------------

# Tokens that are always noise in a display name regardless of context
_NAME_NOISE_WORDS: set[str] = {
    "login", "online", "portal", "banking", "bank", "internet", "secure",
    "authenticated", "authentication", "access",
}

# Parenthetical suffixes that just duplicate audience info already conveyed
# by the kind/audience fields — strip them before comparing
_AUDIENCE_PARENS: set[str] = {
    "personal", "business", "corporate", "retail", "nri", "staff",
    "individual", "sme", "msme",
}

# Words that indicate a banking-type service (used to detect generics)
_BANKING_TYPE_WORDS: set[str] = {
    "net banking", "internet banking", "online banking",
    "netbanking", "ibanking", "e-banking",
}


def _tokenize(text: str) -> list[str]:
    """Split on whitespace and punctuation, lower-case."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _bank_abbreviations(provider: str) -> set[str]:
    """
    Derive the common abbreviations for a bank name so we can strip them
    from the service name.

    Rules (mirrors the prompt logic):
      1. All-caps acronym of initial letters of each word ≥ 2 chars
         (e.g. "Bank of India" → "BOI", "HDFC Bank" → "HB" … not useful;
          only emit if ≥ 2 words give initials, else skip)
      2. First word of the bank name (e.g. "Kotak" from "Kotak Mahindra Bank")
      3. The raw words themselves (e.g. "HDFC" already appears as a token)
    """
    if not provider:
        return set()

    words = re.findall(r"[a-zA-Z0-9]+", provider)
    abbrevs: set[str] = set()

    for w in words:
        abbrevs.add(w.lower())

    # Acronym from first letters of words >= 2 chars
    significant = [w for w in words if len(w) >= 2]
    if len(significant) >= 2:
        acronym = "".join(w[0] for w in significant).lower()
        abbrevs.add(acronym)

    # First word alone (handles "Kotak", "ICICI", "HDFC" etc.)
    if words:
        abbrevs.add(words[0].lower())

    # Remove generic banking words from the abbrev set so we don't
    # accidentally strip meaningful tokens like "national" from a name
    abbrevs -= {"bank", "ltd", "limited", "inc", "co"}

    return abbrevs


def _strip_parentheticals(text: str) -> str:
    """Remove (...) and [...] groups whose inner text is an audience word or country."""
    def _should_strip(match: re.Match) -> str:
        inner = match.group(1).strip().lower()
        inner_tokens = re.findall(r"[a-z]+", inner)
        if all(t in _AUDIENCE_PARENS for t in inner_tokens if t):
            return ""
        return match.group(0)

    return re.sub(r"\(([^)]*)\)", _should_strip, text)


def _normalize_service_name(service_name: str, provider: str) -> str:
    """
    Apply the canonical naming rules from the classification prompt and return
    a normalised display name.  This is intentionally lightweight — it covers
    the deterministic rules only (no LLM inference).

    Steps applied (mirrors the prompt chain-of-thought):
      1. Strip bank name tokens / abbreviations from the service name.
      2. Strip parentheticals that only carry audience info.
      3. Strip trailing / leading noise words.
      4. Title-case and collapse whitespace.
    """
    if not service_name:
        return ""

    name = service_name.strip()

    # Step 2 — remove audience-only parentheticals
    name = _strip_parentheticals(name)

    # Remove bracket variants too
    name = re.sub(r"\[([^\]]*)\]", lambda m: (
        "" if all(t in _AUDIENCE_PARENS for t in re.findall(r"[a-z]+", m.group(1).lower()) if t)
        else m.group(0)
    ), name)

    # Step 1 — strip bank-name tokens
    abbrevs = _bank_abbreviations(provider)
    tokens = _tokenize(name)
    cleaned_tokens = [t for t in tokens if t not in abbrevs]

    # Reconstruct from cleaned tokens preserving any non-bank words
    # (we work token-level to avoid accidentally removing substrings)
    cleaned = " ".join(cleaned_tokens)

    # Step 3 — strip pure noise words from both ends
    words = cleaned.split()
    while words and words[0].lower() in _NAME_NOISE_WORDS:
        words.pop(0)
    while words and words[-1].lower() in _NAME_NOISE_WORDS:
        words.pop()

    cleaned = " ".join(words)

    # Step 4 — title case, collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip().title()

    return cleaned


def assess_service_name(
    service_name: str,
    provider: str,
    page_result: dict,
) -> str | None:
    """
    QA-only, read-only check: does *service_name* already look canonical?

    Returns:
        None            — name passes (looks correct / already canonical)
        "NAME_MISMATCH" — name fails (stored name deviates from canonical form)

    The check is purely deterministic — no LLM, no external calls.
    It never modifies any data; callers must treat the return value as a
    diagnostic flag only.

    Mismatch is flagged when ANY of the following are true:
      a) Bank name tokens appear verbatim in the stored service name
         (e.g. "HDFC Net Banking" → "HDFC" is redundant)
      b) Audience-only parentheticals appear
         (e.g. "Net Banking (Personal)" → "(Personal)" is noise)
      c) The normalised name differs from the stored name by more than
         case/whitespace (i.e. meaningful tokens were stripped or reordered)
    """
    if not service_name:
        return None

    normalised = _normalize_service_name(service_name, provider)
    stored_normalised = re.sub(r"\s+", " ", service_name.strip()).title()

    if not normalised:
        return None

    # Check (a): bank abbreviation tokens present in stored name
    abbrevs = _bank_abbreviations(provider)
    stored_tokens = set(_tokenize(service_name))
    if abbrevs and stored_tokens & abbrevs:
        logger.debug(
            "[name-qa] MISMATCH — bank tokens %s found in service_name=%r",
            stored_tokens & abbrevs,
            service_name,
        )
        return "NAME_MISMATCH"

    # Check (b): audience-only parentheticals present
    if re.search(r"\(([^)]*)\)", service_name):
        inner_groups = re.findall(r"\(([^)]*)\)", service_name)
        for group in inner_groups:
            inner_tokens = re.findall(r"[a-z]+", group.lower())
            if inner_tokens and all(t in _AUDIENCE_PARENS for t in inner_tokens):
                logger.debug(
                    "[name-qa] MISMATCH — audience parenthetical %r in service_name=%r",
                    group,
                    service_name,
                )
                return "NAME_MISMATCH"

    # Check (c): meaningful token difference after normalisation
    if normalised != stored_normalised:
        logger.debug(
            "[name-qa] MISMATCH — normalised=%r  stored=%r",
            normalised,
            stored_normalised,
        )
        return "NAME_MISMATCH"

    logger.debug("[name-qa] PASS — service_name=%r", service_name)
    return None


# ---------------------------------------------------------------------------
# Country match check (deterministic — no LLM)
# ---------------------------------------------------------------------------

# Each entry maps a canonical lowercase country name to:
#   "strong"  — high-confidence markers (ccTLDs, official names)
#   "weak"    — supporting markers (cities, demonyms, language tags)
COUNTRY_SIGNALS: dict[str, dict[str, list[str]]] = {
    "india": {
        "strong": [".in", "india", "bharat"],
        "weak":   ["mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad",
                   "indian", "rupee", "inr", "+91", "hindi"],
    },
    "germany": {
        "strong": [".de", "germany", "deutschland"],
        "weak":   ["berlin", "frankfurt", "munich", "german", "deutsch", "eur", "+49"],
    },
    "spain": {
        "strong": [".es", "spain", "espana", "españa"],
        "weak":   ["madrid", "barcelona", "spanish", "español", "eur", "+34"],
    },
    "united states": {
        "strong": [".us", "united states", "usa"],
        "weak":   ["new york", "california", "american", "usd", "+1"],
    },
    "united kingdom": {
        "strong": [".uk", ".co.uk", "united kingdom"],
        "weak":   ["london", "british", "gbp", "+44"],
    },
    "france": {
        "strong": [".fr", "france"],
        "weak":   ["paris", "french", "français", "francais", "eur", "+33"],
    },
    "australia": {
        "strong": [".au", "australia"],
        "weak":   ["sydney", "melbourne", "australian", "aud", "+61"],
    },
    "canada": {
        "strong": [".ca", "canada"],
        "weak":   ["toronto", "ontario", "canadian", "cad", "+1"],
    },
    "japan": {
        "strong": [".jp", "japan", "日本"],
        "weak":   ["tokyo", "japanese", "jpy", "yen", "+81"],
    },
    "china": {
        "strong": [".cn", "china", "中国"],
        "weak":   ["beijing", "shanghai", "chinese", "cny", "yuan", "+86"],
    },
    "brazil": {
        "strong": [".br", "brazil", "brasil"],
        "weak":   ["sao paulo", "rio", "brazilian", "brl", "real", "+55"],
    },
    "uae": {
        "strong": [".ae", "united arab emirates", "uae"],
        "weak":   ["dubai", "abu dhabi", "emirati", "aed", "dirham", "+971"],
    },
    "singapore": {
        "strong": [".sg", "singapore"],
        "weak":   ["singaporean", "sgd", "+65"],
    },
    "south africa": {
        "strong": [".za", "south africa"],
        "weak":   ["johannesburg", "cape town", "zar", "rand", "+27"],
    },
    "kenya": {
        "strong": [".ke", "kenya"],
        "weak":   ["nairobi", "kenyan", "kes", "+254"],
    },
    "nigeria": {
        "strong": [".ng", "nigeria"],
        "weak":   ["lagos", "nigerian", "ngn", "naira", "+234"],
    },
    "italy": {
        "strong": [".it", "italy", "italia"],
        "weak":   ["rome", "milan", "italian", "italiano", "eur", "+39"],
    },
    "netherlands": {
        "strong": [".nl", "netherlands", "nederland"],
        "weak":   ["amsterdam", "dutch", "eur", "+31"],
    },
    "switzerland": {
        "strong": [".ch", "switzerland", "schweiz", "suisse", "svizzera"],
        "weak":   ["zurich", "geneva", "swiss", "chf", "+41"],
    },
    "russia": {
        "strong": [".ru", "russia", "россия"],
        "weak":   ["moscow", "russian", "rub", "ruble", "+7"],
    },
    "south korea": {
        "strong": [".kr", "south korea", "korea", "한국"],
        "weak":   ["seoul", "korean", "krw", "won", "+82"],
    },
    "mexico": {
        "strong": [".mx", "mexico", "méxico"],
        "weak":   ["mexico city", "mexican", "mxn", "peso", "+52"],
    },
    "indonesia": {
        "strong": [".id", "indonesia"],
        "weak":   ["jakarta", "indonesian", "idr", "rupiah", "+62"],
    },
    "malaysia": {
        "strong": [".my", "malaysia"],
        "weak":   ["kuala lumpur", "malaysian", "myr", "ringgit", "+60"],
    },
    "thailand": {
        "strong": [".th", "thailand"],
        "weak":   ["bangkok", "thai", "thb", "baht", "+66"],
    },
    "turkey": {
        "strong": [".tr", "turkey", "türkiye", "turkiye"],
        "weak":   ["istanbul", "ankara", "turkish", "try", "lira", "+90"],
    },
    "saudi arabia": {
        "strong": [".sa", "saudi arabia", "saudi"],
        "weak":   ["riyadh", "jeddah", "sar", "riyal", "+966"],
    },
    "pakistan": {
        "strong": [".pk", "pakistan"],
        "weak":   ["karachi", "lahore", "islamabad", "pakistani", "pkr", "+92"],
    },
    "bangladesh": {
        "strong": [".bd", "bangladesh"],
        "weak":   ["dhaka", "bangladeshi", "bdt", "taka", "+880"],
    },
    "sri lanka": {
        "strong": [".lk", "sri lanka"],
        "weak":   ["colombo", "sri lankan", "lkr", "+94"],
    },
    "hong kong": {
        "strong": [".hk", "hong kong"],
        "weak":   ["hkd", "+852"],
    },
    "new zealand": {
        "strong": [".nz", "new zealand"],
        "weak":   ["auckland", "wellington", "nzd", "+64"],
    },
    "poland": {
        "strong": [".pl", "poland", "polska"],
        "weak":   ["warsaw", "polish", "pln", "zloty", "+48"],
    },
    "sweden": {
        "strong": [".se", "sweden", "sverige"],
        "weak":   ["stockholm", "swedish", "sek", "krona", "+46"],
    },
    "norway": {
        "strong": [".no", "norway", "norge"],
        "weak":   ["oslo", "norwegian", "nok", "krone", "+47"],
    },
    "denmark": {
        "strong": [".dk", "denmark", "danmark"],
        "weak":   ["copenhagen", "danish", "dkk", "krone", "+45"],
    },
    "finland": {
        "strong": [".fi", "finland", "suomi"],
        "weak":   ["helsinki", "finnish", "eur", "+358"],
    },
    "ireland": {
        "strong": [".ie", "ireland"],
        "weak":   ["dublin", "irish", "eur", "+353"],
    },
    "portugal": {
        "strong": [".pt", "portugal"],
        "weak":   ["lisbon", "portuguese", "eur", "+351"],
    },
    "austria": {
        "strong": [".at", "austria", "österreich"],
        "weak":   ["vienna", "austrian", "eur", "+43"],
    },
    "belgium": {
        "strong": [".be", "belgium", "belgique", "belgië"],
        "weak":   ["brussels", "belgian", "eur", "+32"],
    },
    "greece": {
        "strong": [".gr", "greece", "ελλάδα"],
        "weak":   ["athens", "greek", "eur", "+30"],
    },
    "egypt": {
        "strong": [".eg", "egypt"],
        "weak":   ["cairo", "egyptian", "egp", "+20"],
    },
    "vietnam": {
        "strong": [".vn", "vietnam", "việt nam"],
        "weak":   ["hanoi", "ho chi minh", "vietnamese", "vnd", "dong", "+84"],
    },
    "philippines": {
        "strong": [".ph", "philippines"],
        "weak":   ["manila", "filipino", "php", "peso", "+63"],
    },
    "qatar": {
        "strong": [".qa", "qatar"],
        "weak":   ["doha", "qatari", "qar", "riyal", "+974"],
    },
    "bahrain": {
        "strong": [".bh", "bahrain"],
        "weak":   ["manama", "bahraini", "bhd", "dinar", "+973"],
    },
    "kuwait": {
        "strong": [".kw", "kuwait"],
        "weak":   ["kuwaiti", "kwd", "dinar", "+965"],
    },
    "oman": {
        "strong": [".om", "oman"],
        "weak":   ["muscat", "omani", "omr", "rial", "+968"],
    },
}

# Aliases so users can pass "US", "UK", "UAE" etc. and still match
_COUNTRY_ALIASES: dict[str, str] = {
    "us": "united states",
    "usa": "united states",
    "uk": "united kingdom",
    "gb": "united kingdom",
    "uae": "uae",
    "south korea": "south korea",
    "korea": "south korea",
    "hong kong": "hong kong",
    "hk": "hong kong",
    "nz": "new zealand",
    "sa": "saudi arabia",
}


def _resolve_country(raw: str) -> str:
    """Normalize user-supplied country to a canonical key in COUNTRY_SIGNALS."""
    key = raw.strip().lower()
    return _COUNTRY_ALIASES.get(key, key)


def _extract_cctld(url: str) -> str | None:
    """Return the ccTLD suffix (e.g. '.in', '.de') from the URL."""
    if tldextract is not None:
        ext = tldextract.extract(url)
        if ext.suffix:
            parts = ext.suffix.split(".")
            return f".{parts[-1]}"
        return None
    parsed = urlparse(url)
    domain = parsed.netloc.lower().split(":")[0]
    parts = domain.rsplit(".", 1)
    if len(parts) == 2:
        return f".{parts[-1]}"
    return None


def _score_country(signals: dict[str, list[str]], zones: dict[str, str]) -> tuple[int, list[str]]:
    """Score how strongly a single country's signals appear across page zones.

    Returns (score, notes).
    """
    score = 0
    notes: list[str] = []

    cctld = zones.get("cctld", "")
    url_text = zones["url"]
    title = zones["title"]
    headings = zones["headings"]
    visible = zones["visible"]

    for marker in signals["strong"]:
        if marker.startswith(".") and cctld == marker:
            score += 45
            notes.append(f"ccTLD={marker}")
        elif not marker.startswith("."):
            if marker in url_text:
                score += 25
                notes.append(f"'{marker}' in URL")
            if marker in title:
                score += 20
                notes.append(f"'{marker}' in title")
            if marker in headings:
                score += 10
                notes.append(f"'{marker}' in headings")

    weak_hits = 0
    for marker in signals["weak"]:
        if marker in title or marker in headings or marker in visible:
            weak_hits += 1
            if weak_hits <= 3:
                zone = "title" if marker in title else ("headings" if marker in headings else "text")
                notes.append(f"'{marker}' in {zone}")

    score += min(weak_hits * 5, 20)

    return score, notes


def assess_country_match(
    expected_country: str,
    page_result: dict,
) -> dict:
    """Deterministic check: does the page belong to *expected_country*?

    Returns:
        {
          "matched": True | False | None,  (None = unknown)
          "expected_country": str,
          "detected_country": str | None,
          "expected_score": int,
          "best_foreign_score": int,
          "reason": str,
          "notes": list[str],
        }

    Decision thresholds:
        - foreign score >= 70 AND expected score < 30  → matched=False  (delete)
        - foreign score >= 40 AND expected > foreign    → matched=None   (uncertain)
        - otherwise                                     → matched=True
    """
    country_key = _resolve_country(expected_country or "")

    if not country_key or country_key not in COUNTRY_SIGNALS:
        logger.info(
            "[country-match] Skipped — country '%s' not in signal map",
            expected_country,
        )
        return {
            "matched": None,
            "expected_country": expected_country,
            "detected_country": None,
            "expected_score": 0,
            "best_foreign_score": 0,
            "reason": f"Country '{expected_country}' not recognized — check skipped.",
            "notes": [f"country '{expected_country}' not in signal map"],
        }

    url_text = _normalize_text(
        page_result.get("final_url") or page_result.get("url", "")
    ).lower()
    title = _normalize_text(page_result.get("title")).lower()
    headings = _normalize_text(page_result.get("headings")).lower()
    visible = _normalize_text(page_result.get("visible_text")).lower()[:6000]
    cctld = _extract_cctld(url_text) or ""

    zones = {
        "cctld": cctld,
        "url": url_text,
        "title": title,
        "headings": headings,
        "visible": visible,
    }

    expected_score, expected_notes = _score_country(COUNTRY_SIGNALS[country_key], zones)

    best_foreign_name: str | None = None
    best_foreign_score = 0
    best_foreign_notes: list[str] = []

    for name, signals in COUNTRY_SIGNALS.items():
        if name == country_key:
            continue
        f_score, f_notes = _score_country(signals, zones)
        if f_score > best_foreign_score:
            best_foreign_score = f_score
            best_foreign_name = name
            best_foreign_notes = f_notes

    notes = [f"expected({country_key})={expected_score}"]
    notes.extend(expected_notes)
    if best_foreign_name:
        notes.append(f"strongest_foreign({best_foreign_name})={best_foreign_score}")
        notes.extend(best_foreign_notes)

    if best_foreign_score >= 70 and expected_score < 30:
        matched = False
        reason = (
            f"Page appears to be for {best_foreign_name.title()} "
            f"(score {best_foreign_score}), not {country_key.title()} "
            f"(score {expected_score}). Likely a foreign-country login."
        )
    elif best_foreign_score >= 40 and expected_score <= best_foreign_score:
        matched = None
        reason = (
            f"Mixed country signals: expected {country_key.title()} ({expected_score}) "
            f"vs {best_foreign_name.title()} ({best_foreign_score}). Needs human review."
        )
    else:
        matched = True
        reason = (
            f"Country signals consistent with {country_key.title()} "
            f"(score {expected_score})."
        )

    logger.info(
        "[country-match] expected=%s  expected_score=%d  "
        "best_foreign=%s  foreign_score=%d  matched=%s",
        country_key, expected_score,
        best_foreign_name, best_foreign_score, matched,
    )

    return {
        "matched": matched,
        "expected_country": country_key,
        "detected_country": best_foreign_name if matched is False else country_key,
        "expected_score": expected_score,
        "best_foreign_score": best_foreign_score,
        "reason": reason,
        "notes": notes,
    }
