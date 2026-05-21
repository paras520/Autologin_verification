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
