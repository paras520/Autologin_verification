"""G-Eval metrics for Stage 1.5 — Customer-Facing Classifier.

Reference-based + reference-free LLM-as-a-judge metrics for audience
classification quality.
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from tests.eval.qwen_judge import get_qwen_judge

JUDGE_MODEL = get_qwen_judge()

audience_verdict_correctness = GEval(
    name="Audience Verdict Correctness",
    criteria=(
        "The `actual_output` is a JSON object with `is_customer_facing` (boolean). "
        "The `expected_output` also contains `is_customer_facing` (boolean). "
        "Compare the two booleans. "
        "If they match, the score should be high (>= 0.85). "
        "If they do NOT match (a flip), the score should be very low (<= 0.2). "
        "Partial credit only if the actual_output shows clear uncertainty."
    ),
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model=JUDGE_MODEL,
    async_mode=True,
)

category_consistency = GEval(
    name="Category Consistency",
    criteria=(
        "The `actual_output` is a JSON object with `category` and `is_customer_facing`. "
        "Evaluate whether the category is logically consistent with the verdict: "
        "- 'customer_login' MUST pair with is_customer_facing=true. "
        "- 'hrms', 'careers', 'internal_admin', 'vendor_portal', 'marketing_only' "
        "  MUST pair with is_customer_facing=false. "
        "- 'placeholder' or 'unknown' may pair with either, but the confidence "
        "  should be low when paired with true. "
        "Any logical inconsistency should heavily penalize the score."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model=JUDGE_MODEL,
    async_mode=True,
)

reason_grounding = GEval(
    name="Reason Grounding",
    criteria=(
        "The `actual_output` contains a `reason` string (<= 200 chars). "
        "The `input` contains the page title, headings, buttons, visible text, and URL. "
        "Evaluate whether the `reason` cites concrete page evidence "
        "(specific words from title, headings, visible text, or URL path) "
        "rather than guessing from the bank name alone. "
        "A reason like 'The title says Employee Portal' is good. "
        "A reason like 'This does not look like a bank login' is bad."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.6,
    model=JUDGE_MODEL,
    async_mode=True,
)

CLASSIFIER_METRICS = [audience_verdict_correctness, category_consistency, reason_grounding]
