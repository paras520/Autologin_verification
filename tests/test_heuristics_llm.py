"""Tests for heuristics.py LLM-gated functions — Langfuse fully mocked."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _page(title="Login", visible="Enter credentials", headings=None, login_form=True):
    return {
        "title": title,
        "visible_text": visible,
        "headings": headings or ["Sign In"],
        "buttons": ["Login"],
        "login_form_present": login_form,
        "final_url": "https://example.com/login",
    }


# ---------------------------------------------------------------------------
# _langfuse_is_configured / _load_langfuse_helpers / _build_page_variables / _parse_notes
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_langfuse_not_configured_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        from src.heuristics import _langfuse_is_configured
        assert _langfuse_is_configured() is False

    def test_langfuse_configured_when_all_env_set(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.test")
        from src.heuristics import _langfuse_is_configured
        assert _langfuse_is_configured() is True

    def test_build_page_variables_returns_dict(self):
        from src.heuristics import _build_page_variables
        result = _build_page_variables("HDFC", "NetBanking", _page())
        assert "provider" in result
        assert "visible_text" in result
        assert "page_title" in result

    def test_parse_notes_list(self):
        from src.heuristics import _parse_notes
        assert _parse_notes(["a", "b"]) == ["a", "b"]

    def test_parse_notes_string(self):
        from src.heuristics import _parse_notes
        assert _parse_notes("single note") == ["single note"]

    def test_parse_notes_empty(self):
        from src.heuristics import _parse_notes
        assert _parse_notes(None) == []

    def test_parse_notes_filters_empty_strings(self):
        from src.heuristics import _parse_notes
        assert _parse_notes(["ok", "", None]) == ["ok"]


# ---------------------------------------------------------------------------
# classify_customer_facing — Langfuse not configured (fail-open)
# ---------------------------------------------------------------------------

class TestClassifyCustomerFacingNotConfigured:
    @pytest.mark.asyncio
    async def test_returns_fail_open_when_langfuse_missing(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        from src.heuristics import classify_customer_facing
        result = await classify_customer_facing("HDFC", "NetBanking", "https://x.com", _page())
        assert result["is_customer_facing"] is True
        assert result["confidence"] == 0

    @pytest.mark.asyncio
    async def test_fail_open_has_required_keys(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        from src.heuristics import classify_customer_facing
        result = await classify_customer_facing("SBI", "YONO", "https://x.com", _page())
        for key in ("is_customer_facing", "confidence", "category", "reason", "notes"):
            assert key in result


class TestClassifyCustomerFacingWithLLM:
    def _mock_helpers(self, parsed_response):
        build_messages = MagicMock(return_value=[])
        call_litellm = AsyncMock(return_value="raw_response")
        get_prompts = MagicMock(return_value=("sys", "usr", {}, None))
        parse_response = MagicMock(return_value=parsed_response)
        return build_messages, call_litellm, get_prompts, parse_response

    @pytest.mark.asyncio
    async def test_customer_facing_true(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
        bm, cl, gp, pr = self._mock_helpers({
            "is_customer_facing": True, "confidence": 90,
            "category": "customer_login", "reason": "login page",
        })
        with patch("src.heuristics._load_langfuse_helpers", return_value=(bm, cl, gp, pr)):
            from src.heuristics import classify_customer_facing
            result = await classify_customer_facing("HDFC", "Net", "https://x.com", _page())
        assert result["is_customer_facing"] is True
        assert result["confidence"] == 90

    @pytest.mark.asyncio
    async def test_customer_facing_false(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
        bm, cl, gp, pr = self._mock_helpers({
            "is_customer_facing": False, "confidence": 85,
            "category": "hrms", "reason": "HR system",
        })
        with patch("src.heuristics._load_langfuse_helpers", return_value=(bm, cl, gp, pr)):
            from src.heuristics import classify_customer_facing
            result = await classify_customer_facing("HDFC", "Net", "https://x.com", _page())
        assert result["is_customer_facing"] is False

    @pytest.mark.asyncio
    async def test_unstructured_response_returns_fail_open(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
        bm, cl, gp, pr = self._mock_helpers("not a dict")
        with patch("src.heuristics._load_langfuse_helpers", return_value=(bm, cl, gp, pr)):
            from src.heuristics import classify_customer_facing
            result = await classify_customer_facing("HDFC", "Net", "https://x.com", _page())
        assert result["is_customer_facing"] is True

    @pytest.mark.asyncio
    async def test_llm_exception_returns_fail_open(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
        bm = MagicMock()
        cl = AsyncMock(side_effect=RuntimeError("LLM down"))
        gp = MagicMock(return_value=("sys", "usr", {}, None))
        pr = MagicMock()
        with patch("src.heuristics._load_langfuse_helpers", return_value=(bm, cl, gp, pr)):
            from src.heuristics import classify_customer_facing
            result = await classify_customer_facing("HDFC", "Net", "https://x.com", _page())
        assert result["is_customer_facing"] is True

    @pytest.mark.asyncio
    async def test_unknown_category_mapped_to_unknown(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
        bm, cl, gp, pr = self._mock_helpers({
            "is_customer_facing": True, "confidence": 70,
            "category": "totally_invalid_category", "reason": "ok",
        })
        with patch("src.heuristics._load_langfuse_helpers", return_value=(bm, cl, gp, pr)):
            from src.heuristics import classify_customer_facing
            result = await classify_customer_facing("HDFC", "Net", "https://x.com", _page())
        assert result["category"] == "unknown"

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_100(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
        bm, cl, gp, pr = self._mock_helpers({
            "is_customer_facing": True, "confidence": 150,
            "category": "customer_login", "reason": "ok",
        })
        with patch("src.heuristics._load_langfuse_helpers", return_value=(bm, cl, gp, pr)):
            from src.heuristics import classify_customer_facing
            result = await classify_customer_facing("HDFC", "Net", "https://x.com", _page())
        assert result["confidence"] <= 100


# ---------------------------------------------------------------------------
# extract_and_score — Langfuse not configured fallback
# ---------------------------------------------------------------------------

class TestExtractAndScore:
    @pytest.mark.asyncio
    async def test_not_configured_returns_fallback(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        from src.heuristics import extract_and_score
        result = await extract_and_score("HDFC", "NetBanking", "https://x.com", _page())
        assert "bank_identifiers" in result
        assert "is_login_page" in result

    @pytest.mark.asyncio
    async def test_configured_returns_llm_result(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
        llm_response = {
            "bank_identifiers": ["HDFC Bank"],
            "relevant_page_sections": ["HDFC NetBanking Login"],
            "login_signals": ["form present"],
            "is_login_page": True,
            "login_type_suggestion": "direct",
            "notes": [],
        }
        bm = MagicMock(return_value=[])
        cl = AsyncMock(return_value="raw")
        gp = MagicMock(return_value=("sys", "usr", {}, None))
        pr = MagicMock(return_value=llm_response)
        with patch("src.heuristics._load_langfuse_helpers", return_value=(bm, cl, gp, pr)):
            from src.heuristics import extract_and_score
            result = await extract_and_score("HDFC", "NetBanking", "https://x.com", _page())
        assert result["bank_identifiers"] == ["HDFC Bank"]
        assert result["is_login_page"] is True

    @pytest.mark.asyncio
    async def test_llm_exception_returns_fallback(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
        bm = MagicMock(return_value=[])
        cl = AsyncMock(side_effect=RuntimeError("LLM error"))
        gp = MagicMock(return_value=("sys", "usr", {}, None))
        pr = MagicMock()
        with patch("src.heuristics._load_langfuse_helpers", return_value=(bm, cl, gp, pr)):
            from src.heuristics import extract_and_score
            result = await extract_and_score("HDFC", "NetBanking", "https://x.com", _page())
        assert "bank_identifiers" in result

    @pytest.mark.asyncio
    async def test_non_dict_response_returns_fallback(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
        bm = MagicMock(return_value=[])
        cl = AsyncMock(return_value="raw")
        gp = MagicMock(return_value=("sys", "usr", {}, None))
        pr = MagicMock(return_value="not a dict")
        with patch("src.heuristics._load_langfuse_helpers", return_value=(bm, cl, gp, pr)):
            from src.heuristics import extract_and_score
            result = await extract_and_score("HDFC", "NetBanking", "https://x.com", _page())
        assert "bank_identifiers" in result


# ---------------------------------------------------------------------------
# assess_match_with_identifiers
# ---------------------------------------------------------------------------

class TestAssessMatchWithIdentifiers:
    def _extractor(self):
        return {
            "bank_identifiers": ["HDFC"],
            "relevant_page_sections": ["NetBanking"],
            "login_signals": ["form"],
            "is_login_page": True,
            "login_type_suggestion": "direct",
            "notes": [],
        }

    @pytest.mark.asyncio
    async def test_empty_provider_returns_skip(self):
        from src.heuristics import assess_match_with_identifiers
        result = await assess_match_with_identifiers("", "", "https://x.com", self._extractor())
        assert result["bank_matched"] is True  # skip = fail-open

    @pytest.mark.asyncio
    async def test_not_configured_returns_skip(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        from src.heuristics import assess_match_with_identifiers
        result = await assess_match_with_identifiers("HDFC", "Net", "https://x.com", self._extractor())
        assert result["bank_matched"] is True

    @pytest.mark.asyncio
    async def test_llm_bank_matched(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
        llm_result = {
            "bank_matched": True, "service_matched": True,
            "confidence_score": 90, "url_confidence_score": 85,
            "login_type": "direct", "reason": "Match",
        }
        bm = MagicMock(return_value=[])
        cl = AsyncMock(return_value="raw")
        gp = MagicMock(return_value=("sys", "usr", {}, None))
        pr = MagicMock(return_value=llm_result)
        with patch("src.heuristics._load_langfuse_helpers", return_value=(bm, cl, gp, pr)):
            from src.heuristics import assess_match_with_identifiers
            result = await assess_match_with_identifiers("HDFC", "Net", "https://x.com", self._extractor())
        assert result["bank_matched"] is True
        assert result["confidence_score"] == 90

    @pytest.mark.asyncio
    async def test_llm_returns_invalid_login_type_defaults_to_direct(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
        llm_result = {
            "bank_matched": True, "service_matched": True,
            "confidence_score": 80, "url_confidence_score": 75,
            "login_type": "INVALID_TYPE", "reason": "Match",
        }
        bm = MagicMock(return_value=[])
        cl = AsyncMock(return_value="raw")
        gp = MagicMock(return_value=("sys", "usr", {}, None))
        pr = MagicMock(return_value=llm_result)
        with patch("src.heuristics._load_langfuse_helpers", return_value=(bm, cl, gp, pr)):
            from src.heuristics import assess_match_with_identifiers
            result = await assess_match_with_identifiers("HDFC", "Net", "https://x.com", self._extractor())
        assert result["login_type"] == "direct"

    @pytest.mark.asyncio
    async def test_llm_unstructured_returns_skip(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
        bm = MagicMock(return_value=[])
        cl = AsyncMock(return_value="raw")
        gp = MagicMock(return_value=("sys", "usr", {}, None))
        pr = MagicMock(return_value="unstructured")
        with patch("src.heuristics._load_langfuse_helpers", return_value=(bm, cl, gp, pr)):
            from src.heuristics import assess_match_with_identifiers
            result = await assess_match_with_identifiers("HDFC", "Net", "https://x.com", self._extractor())
        assert result["bank_matched"] is True  # skip = fail-open

    @pytest.mark.asyncio
    async def test_llm_exception_returns_skip(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_HOST", "https://lf.test")
        bm = MagicMock(return_value=[])
        cl = AsyncMock(side_effect=RuntimeError("down"))
        gp = MagicMock(return_value=("sys", "usr", {}, None))
        pr = MagicMock()
        with patch("src.heuristics._load_langfuse_helpers", return_value=(bm, cl, gp, pr)):
            from src.heuristics import assess_match_with_identifiers
            result = await assess_match_with_identifiers("HDFC", "Net", "https://x.com", self._extractor())
        assert result["bank_matched"] is True  # fail-open


# ---------------------------------------------------------------------------
# assess_full_match — orchestrator
# ---------------------------------------------------------------------------

class TestAssessFullMatch:
    @pytest.mark.asyncio
    async def test_full_match_orchestrates_both_steps(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        from src.heuristics import assess_full_match
        result = await assess_full_match("HDFC", "Net", "https://x.com", _page())
        assert "bank_matched" in result
        assert "notes" in result

    @pytest.mark.asyncio
    async def test_full_match_merges_notes(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        from src.heuristics import assess_full_match
        result = await assess_full_match("HDFC", "Net", "https://x.com", _page())
        assert isinstance(result["notes"], list)


# ---------------------------------------------------------------------------
# Legacy stubs
# ---------------------------------------------------------------------------

class TestLegacyStubs:
    @pytest.mark.asyncio
    async def test_assess_direct_login_page_with_form(self):
        from src.heuristics import assess_direct_login_page
        result = await assess_direct_login_page("HDFC", "Net", _page(login_form=True))
        assert result["is_login_page"] is True
        assert result["score"] > 50

    @pytest.mark.asyncio
    async def test_assess_direct_login_page_no_form(self):
        from src.heuristics import assess_direct_login_page
        result = await assess_direct_login_page("HDFC", "Net", _page(login_form=False))
        assert result["is_login_page"] is False
        assert result["score"] < 50

    @pytest.mark.asyncio
    async def test_assess_provider_service_match_delegates(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        from src.heuristics import assess_provider_service_match
        result = await assess_provider_service_match("HDFC", "Net", _page())
        assert "matched" in result
        assert "score" in result
        assert "reason" in result
