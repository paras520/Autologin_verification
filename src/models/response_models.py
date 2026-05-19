from pydantic import BaseModel


# Internal model used by verification_service — not exposed directly in batch response
class ReturnResponse(BaseModel):
    url: str
    inactive_flagged: bool
    reason: str | None
    health_check: bool
    page_match_score: int | None
    direct_match_score: int | None
    notes: str | None
    updated_name: str | None
    marked_for_human_review: bool = False
    marked_for_deletion: bool = False
    errors: str
    time: str


class RowResult(BaseModel):
    """Flat verification result for a single login_services row."""
    cb_link_id: str
    service_id: str | None = None  # login_services.id UUID — required by m103 ingest
    login_service: str
    url: str
    # checks
    health_check: bool | None
    page_match_score: int | None
    direct_match_score: int | None
    display_name_score: int | None  # not yet implemented
    notes: str | None
    # flags
    inactive_flagged: bool | None
    marked_for_deletion: bool | None
    marked_for_human_review: bool | None
    # meta
    is_duplicate: bool
    duplicate_of_url: str | None
    duplicate_of_id: str | None = None
    dedupe_action: str | None = None
    dedupe_reason: str | None = None
    canonical_display_name: str | None = None
    reason: str | None
    status: str


class CbLinkResult(BaseModel):
    """All row results for a single cb_link_id."""
    cb_link_id: str
    total_rows: int
    rows: list[RowResult]


class BatchCheckResponse(BaseModel):
    """Top-level response for POST /check/batch."""
    results: list[CbLinkResult]


class AsyncBatchResponse(BaseModel):
    """Immediate response for POST /check/batch/async — verification runs in background."""
    run_id: str
    status: str = "queued"
    total_links: int
