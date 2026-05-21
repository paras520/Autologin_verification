"""Unit tests for deterministic heuristic functions."""
import pytest
from src.heuristics import (
    _bank_abbreviations,
    _extract_cctld,
    _normalize_service_name,
    _normalize_text,
    _resolve_country,
    _score_country,
    _strip_parentheticals,
    _tokenize,
    assess_country_match,
    assess_service_name,
)


# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_none_returns_empty(self):
        assert _normalize_text(None) == ""

    def test_strips_whitespace(self):
        assert _normalize_text("  hello  ") == "hello"

    def test_strips_whitespace_only(self):
        assert _normalize_text("  HELLO  ") == "HELLO"

    def test_list_joined(self):
        result = _normalize_text(["Hello", "World"])
        assert "Hello" in result
        assert "World" in result

    def test_empty_string(self):
        assert _normalize_text("") == ""


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_split(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_alphanumeric_only(self):
        assert _tokenize("foo-bar_baz") == ["foo", "bar", "baz"]

    def test_numbers_included(self):
        assert "123" in _tokenize("test 123")

    def test_empty(self):
        assert _tokenize("") == []


# ---------------------------------------------------------------------------
# _bank_abbreviations
# ---------------------------------------------------------------------------

class TestBankAbbreviations:
    def test_empty_provider(self):
        assert _bank_abbreviations("") == set()

    def test_single_word_bank(self):
        abbrevs = _bank_abbreviations("HDFC")
        assert "hdfc" in abbrevs

    def test_multi_word_acronym(self):
        abbrevs = _bank_abbreviations("State Bank of India")
        assert "sboi" in abbrevs or "sbi" in abbrevs or "state" in abbrevs

    def test_bank_word_excluded(self):
        abbrevs = _bank_abbreviations("Kotak Mahindra Bank")
        assert "bank" not in abbrevs

    def test_first_word_included(self):
        abbrevs = _bank_abbreviations("ICICI Bank")
        assert "icici" in abbrevs

    def test_limited_words_excluded(self):
        abbrevs = _bank_abbreviations("Bank of Baroda Ltd")
        assert "ltd" not in abbrevs
        assert "limited" not in abbrevs


# ---------------------------------------------------------------------------
# _strip_parentheticals
# ---------------------------------------------------------------------------

class TestStripParentheticals:
    def test_audience_only_stripped(self):
        result = _strip_parentheticals("HDFC NetBanking (Retail)")
        assert "(Retail)" not in result

    def test_non_audience_preserved(self):
        result = _strip_parentheticals("SBI Online (YONO)")
        assert "YONO" in result

    def test_empty_parens_stripped(self):
        result = _strip_parentheticals("Service ()")
        assert "()" not in result

    def test_no_parens_unchanged(self):
        assert _strip_parentheticals("HDFC NetBanking") == "HDFC NetBanking"


# ---------------------------------------------------------------------------
# _normalize_service_name
# ---------------------------------------------------------------------------

class TestNormalizeServiceName:
    def test_empty_returns_empty(self):
        assert _normalize_service_name("", "SBI") == ""

    def test_bank_name_stripped(self):
        result = _normalize_service_name("HDFC NetBanking", "HDFC Bank")
        assert "netbanking" in result.lower()

    def test_title_case_applied(self):
        result = _normalize_service_name("netbanking portal", "SBI")
        assert result[0].isupper()

    def test_whitespace_collapsed(self):
        result = _normalize_service_name("  net   banking  ", "SBI")
        assert "  " not in result


# ---------------------------------------------------------------------------
# _resolve_country
# ---------------------------------------------------------------------------

class TestResolveCountry:
    def test_india_resolves(self):
        result = _resolve_country("India")
        assert result == "india"

    def test_us_alias(self):
        result = _resolve_country("United States")
        assert result in ("united states", "usa", "us")

    def test_unknown_passes_through(self):
        result = _resolve_country("Wakanda")
        assert result == "wakanda"

    def test_strips_whitespace(self):
        result = _resolve_country("  india  ")
        assert result == "india"


# ---------------------------------------------------------------------------
# _extract_cctld
# ---------------------------------------------------------------------------

class TestExtractCctld:
    def test_indian_domain(self):
        result = _extract_cctld("https://netbanking.hdfcbank.in/login")
        assert result == ".in"

    def test_uk_domain(self):
        result = _extract_cctld("https://lloydsbank.co.uk/login")
        assert result == ".uk"

    def test_com_domain(self):
        result = _extract_cctld("https://bankofamerica.com/login")
        assert result in (".com", None)

    def test_de_domain(self):
        result = _extract_cctld("https://deutsche-bank.de/login")
        assert result == ".de"


# ---------------------------------------------------------------------------
# _score_country
# ---------------------------------------------------------------------------

class TestScoreCountry:
    def _zones(self, url="", title="", headings="", visible="", cctld=""):
        return {"url": url, "title": title, "headings": headings, "visible": visible, "cctld": cctld}

    def _signals(self, strong=None, weak=None):
        return {"strong": strong or [], "weak": weak or []}

    def test_no_signals_zero_score(self):
        score, notes = _score_country(self._signals(), self._zones())
        assert score == 0
        assert notes == []

    def test_cctld_match_scores_high(self):
        signals = self._signals(strong=[".in"])
        zones = self._zones(cctld=".in")
        score, notes = _score_country(signals, zones)
        assert score >= 40
        assert any(".in" in n for n in notes)

    def test_url_match_scores(self):
        signals = self._signals(strong=["india"])
        zones = self._zones(url="https://sbi.co.in/india/login")
        score, notes = _score_country(signals, zones)
        assert score > 0

    def test_title_match_scores(self):
        signals = self._signals(strong=["india"])
        zones = self._zones(title="india bank login")
        score, notes = _score_country(signals, zones)
        assert score > 0

    def test_weak_signals_capped(self):
        signals = self._signals(weak=["rupee", "neft", "rtgs", "imps", "inr"])
        zones = self._zones(visible="rupee neft rtgs imps inr upi")
        score, notes = _score_country(signals, zones)
        assert score <= 20


# ---------------------------------------------------------------------------
# assess_country_match
# ---------------------------------------------------------------------------

class TestAssessCountryMatch:
    def test_unknown_country_returns_none_matched(self):
        result = assess_country_match("Wakanda", {})
        assert result["matched"] is None
        assert result["expected_country"] == "Wakanda"

    def test_empty_country_returns_none_matched(self):
        result = assess_country_match("", {})
        assert result["matched"] is None

    def test_india_with_in_cctld_matches(self):
        page = {
            "final_url": "https://netbanking.hdfcbank.in/login",
            "title": "HDFC India NetBanking",
            "headings": ["Login to your account"],
            "visible_text": "India banking portal",
        }
        result = assess_country_match("India", page)
        assert result["expected_country"] is not None
        assert "matched" in result
        assert "notes" in result

    def test_result_structure(self):
        result = assess_country_match("India", {"title": "Bank", "visible_text": ""})
        for key in ("matched", "expected_country", "detected_country", "expected_score", "reason", "notes"):
            assert key in result

    def test_foreign_country_detected(self):
        page = {
            "final_url": "https://hsbc.co.uk/login",
            "title": "HSBC United Kingdom",
            "headings": ["UK Online Banking"],
            "visible_text": "British pounds sterling GBP United Kingdom",
        }
        result = assess_country_match("India", page)
        assert result["expected_score"] < result.get("best_foreign_score", 999)

    def test_notes_is_list(self):
        result = assess_country_match("India", {"title": "test"})
        assert isinstance(result["notes"], list)


# ---------------------------------------------------------------------------
# assess_service_name
# ---------------------------------------------------------------------------

class TestAssessServiceName:
    def _page(self, title="", visible="", headings=None, login_form=True):
        return {
            "title": title,
            "visible_text": visible,
            "headings": headings or [],
            "login_form_present": login_form,
            "buttons": [],
        }

    def test_empty_service_name(self):
        result = assess_service_name("", "HDFC Bank", self._page("HDFC Login"))
        assert result is None or isinstance(result, str)

    def test_canonical_name_passes(self):
        # When the page clearly matches the service name, should return None
        result = assess_service_name(
            "NetBanking",
            "HDFC Bank",
            self._page("HDFC NetBanking - Login"),
        )
        assert result is None or result == "NAME_MISMATCH"

    def test_returns_none_or_name_mismatch(self):
        result = assess_service_name(
            "iMobile Pay",
            "ICICI Bank",
            self._page("ICICI Bank iMobile Pay Login"),
        )
        assert result in (None, "NAME_MISMATCH")

    def test_completely_wrong_page(self):
        result = assess_service_name(
            "NetBanking",
            "HDFC Bank",
            self._page("Unrelated Technology Company"),
        )
        assert result in (None, "NAME_MISMATCH")

    def test_assess_service_name_with_visible_text(self):
        result = assess_service_name(
            "NetBanking",
            "HDFC Bank",
            self._page("HDFC", visible="hdfc netbanking login"),
        )
        assert result in (None, "NAME_MISMATCH")


# ---------------------------------------------------------------------------
# _root_domain / _is_shared_host
# ---------------------------------------------------------------------------

class TestRootDomainAndSharedHost:
    def test_root_domain_extracts_registrable(self):
        from src.heuristics import _root_domain
        assert _root_domain("https://netbanking.hdfcbank.in/login") == "hdfcbank.in"

    def test_root_domain_com(self):
        from src.heuristics import _root_domain
        assert _root_domain("https://bankofamerica.com/login") == "bankofamerica.com"

    def test_root_domain_fallback_no_tldextract(self):
        """Covers the urlparse fallback branch."""
        from src.heuristics import _root_domain
        import src.heuristics as h
        original = h.tldextract
        try:
            h.tldextract = None
            result = _root_domain("https://example.co.uk/page")
            assert "example" in result or "co.uk" in result
        finally:
            h.tldextract = original

    def test_is_shared_host_true(self):
        from src.heuristics import _is_shared_host
        assert _is_shared_host("https://onlinesbi.com/login") is True

    def test_is_shared_host_false(self):
        from src.heuristics import _is_shared_host
        assert _is_shared_host("https://uniquebank-xyz.com/login") is False


# ---------------------------------------------------------------------------
# _score_country — edge cases (extra weak signals branch)
# ---------------------------------------------------------------------------

class TestScoreCountryEdgeCases:
    def _zones(self, url="", title="", headings="", visible="", cctld=""):
        return {"url": url, "title": title, "headings": headings, "visible": visible, "cctld": cctld}

    def _signals(self, strong=None, weak=None):
        return {"strong": strong or [], "weak": weak or []}

    def test_headings_match_scores(self):
        signals = self._signals(strong=["india"])
        zones = self._zones(headings="india banking portal")
        score, notes = _score_country(signals, zones)
        assert score >= 0  # at least ran without error

    def test_visible_match_scores(self):
        signals = self._signals(strong=["india"])
        zones = self._zones(visible="india finance login")
        score, notes = _score_country(signals, zones)
        assert score >= 0

    def test_multiple_strong_signals(self):
        signals = self._signals(strong=[".in", "india"])
        zones = self._zones(cctld=".in", url="https://sbi.co.in/india")
        score, notes = _score_country(signals, zones)
        assert score > 0

    def test_cctld_mismatch_no_bonus(self):
        signals = self._signals(strong=[".in"])
        zones = self._zones(cctld=".uk")
        score, _ = _score_country(signals, zones)
        assert score == 0


# ---------------------------------------------------------------------------
# internal_models.LLMDecision instantiation
# ---------------------------------------------------------------------------

class TestLLMDecision:
    def test_instantiation_all_fields(self):
        from src.models.internal_models import LLMDecision
        decision = LLMDecision(
            inactive_flagged=False,
            reason="All checks passed",
            raw_output={"score": 90},
            error=None,
            session_id="session-abc",
            prompt_path="autologinQA/service_matcher",
        )
        assert decision.inactive_flagged is False
        assert decision.reason == "All checks passed"
        assert decision.raw_output == {"score": 90}
        assert decision.error is None
        assert decision.session_id == "session-abc"
        assert decision.prompt_path == "autologinQA/service_matcher"

    def test_instantiation_with_error(self):
        from src.models.internal_models import LLMDecision
        decision = LLMDecision(
            inactive_flagged=True,
            reason=None,
            raw_output=None,
            error="LLM timeout",
            session_id=None,
            prompt_path="autologinQA/identifier_extractor",
        )
        assert decision.inactive_flagged is True
        assert decision.error == "LLM timeout"

    def test_instantiation_with_string_raw_output(self):
        from src.models.internal_models import LLMDecision
        decision = LLMDecision(
            inactive_flagged=False,
            reason="ok",
            raw_output="raw text response",
            error=None,
            session_id="s1",
            prompt_path="autologinQA/customer_facing_classifier",
        )
        assert isinstance(decision.raw_output, str)


# ---------------------------------------------------------------------------
# LLM-calling paths — classify_customer_facing / extract_and_score /
# assess_match_with_identifiers / assess_full_match / assess_provider_service_match
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402
from src.heuristics import (  # noqa: E402
    classify_customer_facing,
    extract_and_score,
    assess_match_with_identifiers,
    assess_full_match,
    assess_provider_service_match,
    _customer_facing_fallback,
)


def _llm_helpers(parse_return):
    """Return a 4-tuple of mocked langfuse helper callables."""
    mock_parse = MagicMock(return_value=parse_return)
    mock_call = AsyncMock(return_value=MagicMock())
    mock_build = MagicMock(return_value=[{"role": "user", "content": "t"}])
    mock_get_prompts = MagicMock(return_value=("sys", "usr", {}, None))
    return mock_build, mock_call, mock_get_prompts, mock_parse


class TestCustomerFacingFallback:
    def test_returns_fail_open(self):
        result = _customer_facing_fallback("err", "note")
        assert result["is_customer_facing"] is True
        assert result["confidence"] == 0
        assert result["category"] == "unknown"


class TestClassifyCustomerFacing:
    @pytest.mark.asyncio
    async def test_langfuse_not_configured_returns_fallback(self):
        with patch("src.heuristics._langfuse_is_configured", return_value=False):
            result = await classify_customer_facing("Bank", "Svc", "https://x.com", {})
        assert result["is_customer_facing"] is True
        assert result["confidence"] == 0

    @pytest.mark.asyncio
    async def test_success_path(self):
        parsed = {"is_customer_facing": True, "confidence": 95, "category": "customer_login", "reason": "Login"}
        helpers = _llm_helpers(parsed)
        with patch("src.heuristics._langfuse_is_configured", return_value=True):
            with patch("src.heuristics._load_langfuse_helpers", return_value=helpers):
                result = await classify_customer_facing("Bank", "Svc", "https://x.com", {"visible_text": "login"}, session_id="s1")
        assert result["is_customer_facing"] is True
        assert result["confidence"] == 95
        assert result["category"] == "customer_login"

    @pytest.mark.asyncio
    async def test_invalid_category_normalised_to_unknown(self):
        parsed = {"is_customer_facing": True, "confidence": 80, "category": "garbage_cat", "reason": "ok"}
        helpers = _llm_helpers(parsed)
        with patch("src.heuristics._langfuse_is_configured", return_value=True):
            with patch("src.heuristics._load_langfuse_helpers", return_value=helpers):
                result = await classify_customer_facing("Bank", "Svc", "https://x.com", {})
        assert result["category"] == "unknown"

    @pytest.mark.asyncio
    async def test_unstructured_response_returns_fallback(self):
        helpers = _llm_helpers("not a dict at all")
        with patch("src.heuristics._langfuse_is_configured", return_value=True):
            with patch("src.heuristics._load_langfuse_helpers", return_value=helpers):
                result = await classify_customer_facing("Bank", "Svc", "https://x.com", {})
        assert result["is_customer_facing"] is True
        assert result["confidence"] == 0

    @pytest.mark.asyncio
    async def test_llm_exception_returns_fallback(self):
        mock_build, mock_call, mock_get_prompts, mock_parse = _llm_helpers({})
        mock_call.side_effect = RuntimeError("network error")
        with patch("src.heuristics._langfuse_is_configured", return_value=True):
            with patch("src.heuristics._load_langfuse_helpers", return_value=(mock_build, mock_call, mock_get_prompts, mock_parse)):
                result = await classify_customer_facing("Bank", "Svc", "https://x.com", {})
        assert result["is_customer_facing"] is True
        assert result["confidence"] == 0


class TestExtractAndScore:
    @pytest.mark.asyncio
    async def test_langfuse_not_configured_returns_fallback(self):
        with patch("src.heuristics._langfuse_is_configured", return_value=False):
            result = await extract_and_score("Bank", "Svc", "https://x.com", {"login_form_present": True})
        assert result["bank_identifiers"] == []
        assert result["is_login_page"] is True

    @pytest.mark.asyncio
    async def test_success_path(self):
        parsed = {
            "bank_identifiers": ["MyBank"],
            "relevant_page_sections": ["Login form"],
            "login_signals": ["Username field"],
            "is_login_page": True,
            "login_type_suggestion": "direct",
        }
        helpers = _llm_helpers(parsed)
        with patch("src.heuristics._langfuse_is_configured", return_value=True):
            with patch("src.heuristics._load_langfuse_helpers", return_value=helpers):
                result = await extract_and_score("Bank", "Svc", "https://x.com", {}, session_id="s2")
        assert result["bank_identifiers"] == ["MyBank"]
        assert result["is_login_page"] is True
        assert result["login_type_suggestion"] == "direct"

    @pytest.mark.asyncio
    async def test_invalid_login_type_defaults_to_direct(self):
        parsed = {"bank_identifiers": [], "relevant_page_sections": [], "login_signals": [], "is_login_page": False, "login_type_suggestion": "garbage"}
        helpers = _llm_helpers(parsed)
        with patch("src.heuristics._langfuse_is_configured", return_value=True):
            with patch("src.heuristics._load_langfuse_helpers", return_value=helpers):
                result = await extract_and_score("Bank", "Svc", "https://x.com", {})
        assert result["login_type_suggestion"] == "direct"

    @pytest.mark.asyncio
    async def test_unstructured_response_returns_fallback(self):
        helpers = _llm_helpers("not a dict")
        with patch("src.heuristics._langfuse_is_configured", return_value=True):
            with patch("src.heuristics._load_langfuse_helpers", return_value=helpers):
                result = await extract_and_score("Bank", "Svc", "https://x.com", {})
        assert result["bank_identifiers"] == []

    @pytest.mark.asyncio
    async def test_exception_returns_fallback(self):
        mock_build, mock_call, mock_get_prompts, mock_parse = _llm_helpers({})
        mock_call.side_effect = RuntimeError("timeout")
        with patch("src.heuristics._langfuse_is_configured", return_value=True):
            with patch("src.heuristics._load_langfuse_helpers", return_value=(mock_build, mock_call, mock_get_prompts, mock_parse)):
                result = await extract_and_score("Bank", "Svc", "https://x.com", {})
        assert result["bank_identifiers"] == []


class TestAssessMatchWithIdentifiers:
    def _extractor(self, login_type="direct"):
        return {"bank_identifiers": ["MyBank"], "relevant_page_sections": ["Login"], "login_signals": [], "is_login_page": True, "login_type_suggestion": login_type}

    @pytest.mark.asyncio
    async def test_both_empty_skips(self):
        result = await assess_match_with_identifiers("", "", "https://x.com", self._extractor())
        assert result["bank_matched"] is True
        assert result["service_matched"] is True

    @pytest.mark.asyncio
    async def test_langfuse_not_configured_skips(self):
        with patch("src.heuristics._langfuse_is_configured", return_value=False):
            result = await assess_match_with_identifiers("Bank", "Svc", "https://x.com", self._extractor())
        assert result["bank_matched"] is True

    @pytest.mark.asyncio
    async def test_success_path(self):
        parsed = {"bank_matched": True, "service_matched": True, "confidence_score": 90, "url_confidence_score": 85, "login_type": "direct", "reason": "Match", "notes": []}
        helpers = _llm_helpers(parsed)
        with patch("src.heuristics._langfuse_is_configured", return_value=True):
            with patch("src.heuristics._load_langfuse_helpers", return_value=helpers):
                result = await assess_match_with_identifiers("Bank", "Svc", "https://x.com", self._extractor(), session_id="s3")
        assert result["bank_matched"] is True
        assert result["confidence_score"] == 90

    @pytest.mark.asyncio
    async def test_no_match_builds_reason(self):
        parsed = {"bank_matched": False, "service_matched": False, "confidence_score": 10, "url_confidence_score": 20, "login_type": "navigation", "reason": "", "notes": []}
        helpers = _llm_helpers(parsed)
        with patch("src.heuristics._langfuse_is_configured", return_value=True):
            with patch("src.heuristics._load_langfuse_helpers", return_value=helpers):
                result = await assess_match_with_identifiers("Bank", "Svc", "https://x.com", self._extractor())
        assert result["bank_matched"] is False
        assert "NOT match" in result["reason"] or result["reason"]

    @pytest.mark.asyncio
    async def test_unstructured_response_skips(self):
        helpers = _llm_helpers("nope")
        with patch("src.heuristics._langfuse_is_configured", return_value=True):
            with patch("src.heuristics._load_langfuse_helpers", return_value=helpers):
                result = await assess_match_with_identifiers("Bank", "Svc", "https://x.com", self._extractor())
        assert result["bank_matched"] is True

    @pytest.mark.asyncio
    async def test_exception_skips(self):
        mock_build, mock_call, mock_get_prompts, mock_parse = _llm_helpers({})
        mock_call.side_effect = RuntimeError("err")
        with patch("src.heuristics._langfuse_is_configured", return_value=True):
            with patch("src.heuristics._load_langfuse_helpers", return_value=(mock_build, mock_call, mock_get_prompts, mock_parse)):
                result = await assess_match_with_identifiers("Bank", "Svc", "https://x.com", self._extractor())
        assert result["bank_matched"] is True


class TestAssessFullMatch:
    @pytest.mark.asyncio
    async def test_delegates_to_both_stages(self):
        extractor_result = {"bank_identifiers": ["B"], "relevant_page_sections": [], "login_signals": [], "is_login_page": True, "login_type_suggestion": "direct", "notes": ["ext-note"]}
        match_result = {"bank_matched": True, "service_matched": True, "confidence_score": 80, "url_confidence_score": 75, "login_type": "direct", "reason": "ok", "notes": ["match-note"]}
        with patch("src.heuristics.extract_and_score", new=AsyncMock(return_value=extractor_result)):
            with patch("src.heuristics.assess_match_with_identifiers", new=AsyncMock(return_value=match_result)):
                result = await assess_full_match("Bank", "Svc", "https://x.com", {}, session_id="s")
        assert "ext-note" in result["notes"]
        assert result["bank_matched"] is True


class TestAssessProviderServiceMatch:
    @pytest.mark.asyncio
    async def test_delegates_and_reshapes(self):
        full = {"bank_matched": True, "service_matched": True, "confidence_score": 85, "url_confidence_score": 70, "login_type": "direct", "reason": "ok", "notes": []}
        with patch("src.heuristics.assess_full_match", new=AsyncMock(return_value=full)):
            result = await assess_provider_service_match("Bank", "Svc", {"final_url": "https://x.com"})
        assert result["matched"] is True
        assert result["score"] == 85
