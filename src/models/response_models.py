from pydantic import BaseModel


# DO NOT TOUCH THIS STRUCTURE, IT IS USED FOR THE RETURN RESPONSE
class ReturnResponse(BaseModel):
    url: str
    inactive_flagged: bool
    reason: str | None
    health_check: bool
    page_match_score: int | None
    direct_match_score: int | None
    notes: str | None
    updated_name: str | None
    marked_for_human_review: bool
    marked_for_deletion: bool
    errors: str
    time: str
