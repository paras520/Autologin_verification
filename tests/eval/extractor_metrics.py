"""G-Eval metrics for Stage 1 — Identifier Extractor.

Reference-free LLM-as-a-judge metrics. No expected_output needed.
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from tests.eval.qwen_judge import get_qwen_judge

JUDGE_MODEL = get_qwen_judge()

extraction_completeness = GEval(
    name="Extraction Completeness",
    criteria=(
        "The `input` contains the full visible text, title, headings, and "
        "buttons of a bank/financial service web page. The `actual_output` is "
        "a JSON object with `bank_identifiers` and `relevant_page_sections`. "
        "Evaluate whether the extractor captured ALL meaningful bank/service "
        "identification signals present in the input page text. "
        "Missing important identifiers (bank brand name, legal entity name, "
        "product names, logo text) should heavily penalize the score. "
        "Minor omissions of low-value text (cookie banners, generic footer "
        "links) should NOT penalize. "
        "If the page content is genuinely empty or clearly blocked (no "
        "extractable signals), a low-confidence or empty extraction output "
        "is NOT a penalty — it correctly recognizes the limitation. "
        "Penalize only when meaningful signals exist in the actual_output "
        "that were omitted due to poor extraction, not due to page inaccessibility."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.6,
    model=JUDGE_MODEL,
    async_mode=True,
)

signal_relevance = GEval(
    name="Signal Relevance",
    criteria=(
        "The `actual_output` is a JSON object with `relevant_page_sections` "
        "containing text snippets extracted from a bank/financial service page. "
        "Evaluate whether these sections are genuinely useful for identifying "
        "which bank and which service the page belongs to. "
        "High-value sections include: bank name mentions, service/product "
        "names, legal entity text, branding, login portal headings. "
        "Low-value noise includes: generic cookie notices, unrelated footer "
        "links, marketing fluff, privacy policy boilerplate, social media "
        "links. A high noise-to-signal ratio should penalize the score."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.6,
    model=JUDGE_MODEL,
    async_mode=True,
)

login_detection = GEval(
    name="Login Detection",
    criteria=(
        "The `input` contains data from a bank/financial service web page "
        "including whether a login form is present, button labels, and page "
        "headings. The `actual_output` is a JSON object with `is_login_page` "
        "(boolean) and `login_type_suggestion` ('direct' or 'navigation'). "
        "Evaluate: "
        "1) Is `is_login_page` correct given the page evidence? A page with "
        "   username/password fields, login buttons, or OTP inputs IS a "
        "   login page. A homepage with only a 'Login' nav link is NOT. "
        "2) Is `login_type_suggestion` appropriate? 'direct' means the form "
        "   is immediately usable; 'navigation' means the user must click "
        "   through to reach login. "
        "Incorrect boolean or clearly wrong login_type should heavily "
        "penalize the score."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model=JUDGE_MODEL,
    async_mode=True,
)

EXTRACTOR_METRICS = [extraction_completeness, signal_relevance, login_detection]
