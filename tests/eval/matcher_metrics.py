"""Deterministic correctness metric for Stage 2 — Service Matcher.

Why this isn't a DAGMetric anymore:
  DeepEval's DAGMetric has two issues that make it unreliable for our case:
    1. metric.score is OVERWRITTEN by every VerdictNode that fires (line 151
       in deepeval/metrics/dag/nodes.py: `metric.score = self.score / 10`),
       not aggregated across leaves. So when bank=False fires a 0 and
       url_calibration fires a 10 in parallel, the final score depends on
       execution order — not on intent.
    2. The TaskNode parses `actual_output` via an LLM judge, which can
       hallucinate boolean values when comparing structured fields. We saw
       a known-mismatch (bank_matched: false vs expected true) get scored
       1.0 because the judge hallucinated agreement.

  Matcher correctness is purely deterministic: parse JSON, compare booleans,
  bucket integer deltas. No LLM is needed (or appropriate). This metric does
  exactly that, weighted as:
      bank_matched      40%   (gates everything; mismatch -> 0 contribution)
      service_matched   30%   (only counted if bank_matched)
      confidence_score  15%   (within 5pts: 1.0, within 15pts: 0.7, else 0.4)
      url_score         15%   (independent, same buckets)

  Score in [0.0, 1.0]. Threshold 0.5 means "bank+service correct" passes.
"""
from __future__ import annotations

import json
from typing import Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def _safe_load(s: Optional[str]) -> dict:
    if not s:
        return {}
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}


def _bucket_score(actual: Optional[int], expected: Optional[int]) -> float:
    if actual is None or expected is None:
        return 0.0
    try:
        delta = abs(int(actual) - int(expected))
    except (TypeError, ValueError):
        return 0.0
    if delta <= 5:
        return 1.0
    if delta <= 15:
        return 0.7
    return 0.4


class MatcherCorrectness(BaseMetric):
    """Deterministic matcher correctness. No LLM calls."""

    _required_params = [
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        async_mode: bool = True,
        verbose_mode: bool = False,
        include_reason: bool = True,
        weights: Optional[dict] = None,
    ):
        self.threshold = threshold
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.include_reason = include_reason
        self.weights = weights or {
            "bank": 0.40,
            "service": 0.30,
            "confidence": 0.15,
            "url": 0.15,
        }
        self.score: Optional[float] = None
        self.reason: Optional[str] = None
        self.success: Optional[bool] = None
        self.error: Optional[str] = None
        self.evaluation_cost = 0.0
        self.evaluation_model = "deterministic"
        self.verbose_logs: Optional[str] = None

    @property
    def __name__(self) -> str:
        return "Matcher Correctness"

    def _compute(self, test_case: LLMTestCase) -> tuple[float, str]:
        actual = _safe_load(test_case.actual_output)
        expected = _safe_load(test_case.expected_output)

        bank_a = bool(actual.get("bank_matched", False))
        bank_e = bool(expected.get("bank_matched", False))
        srv_a = bool(actual.get("service_matched", False))
        srv_e = bool(expected.get("service_matched", False))

        bank_ok = bank_a == bank_e
        srv_ok = srv_a == srv_e

        # Bank gate: if bank wrong, only the bank component contributes.
        bank_pts = 1.0 if bank_ok else 0.0
        srv_pts = 1.0 if (bank_ok and srv_ok) else 0.0

        conf_pts = (
            _bucket_score(actual.get("confidence_score"), expected.get("confidence_score"))
            if (bank_ok and srv_ok)
            else 0.0
        )
        url_pts = _bucket_score(
            actual.get("url_confidence_score"),
            expected.get("url_confidence_score"),
        )  # url is always evaluated, independent of bank gate

        w = self.weights
        score = (
            w["bank"] * bank_pts
            + w["service"] * srv_pts
            + w["confidence"] * conf_pts
            + w["url"] * url_pts
        )

        reason = (
            f"bank_match={bank_a}/exp={bank_e} ({'OK' if bank_ok else 'MISMATCH'})  "
            f"service_match={srv_a}/exp={srv_e} ({'OK' if srv_ok else 'MISMATCH' if bank_ok else 'GATED'})  "
            f"conf={actual.get('confidence_score')}/exp={expected.get('confidence_score')} -> {conf_pts:.2f}  "
            f"url={actual.get('url_confidence_score')}/exp={expected.get('url_confidence_score')} -> {url_pts:.2f}  "
            f"=>  total={score:.3f}"
        )
        return score, reason

    # Sync
    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        try:
            score, reason = self._compute(test_case)
            self.score = score
            self.reason = reason if self.include_reason else None
            self.success = score >= self.threshold
            self.verbose_logs = reason if self.verbose_mode else None
            return score
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.score = 0.0
            self.success = False
            self.reason = self.error
            return 0.0

    # Async (just delegates; metric is pure CPU)
    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)


matcher_correctness_metric = MatcherCorrectness(
    threshold=0.5,
    async_mode=True,
    include_reason=True,
)

MATCHER_METRICS = [matcher_correctness_metric]
