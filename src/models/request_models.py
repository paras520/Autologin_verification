from pydantic import BaseModel, Field


class CheckRequest(BaseModel):
    """Internal per-row request built from a fetched DB row."""
    provider: str
    service_name: str
    login_type: str
    url: str
    country: str
    cb_link_id: str = ""


class BatchCheckRequest(BaseModel):
    """Public API request body.

    Example:
        {
            "cb_link_ids": ["B-IN-i9i6wf"],
            "include_inactive": true,
            "triggered_by": "m114"
        }
    """
    cb_link_ids: list[str] = Field(..., min_length=1)
    include_inactive: bool = True
    triggered_by: str | None = None
