from pydantic import BaseModel


class CheckRequest(BaseModel):
    provider: str
    service_name: str
    login_type: str
    url: str
    country: str
