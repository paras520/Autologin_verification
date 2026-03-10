from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.page_extraction import extract_visible_layer
from src.url_health import check_url_health

app = FastAPI(title="URL Verification API")


class CheckRequest(BaseModel):
    url: str


def _is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)


@app.post("/check")
def check_url(payload: CheckRequest):
    url = payload.url

    if not isinstance(url, str) or not url.strip():
        raise HTTPException(
            status_code=400,
            detail="Request body must include a non-empty 'url' field.",
        )

    url = url.strip()
    if not _is_valid_url(url):
        raise HTTPException(
            status_code=400,
            detail="URL must include a valid scheme and host.",
        )

    health_result = check_url_health(url)
    #url health file calls the visual extraction module from page extraction file so rest everything happens there


    return {
        "url": url,
        "health_check": health_result,
        "page_check": page_result,
        "confidence_score": 0,
        "marked_for_human_review": False,
        "marked_for_deletion": False,
        "errors": "",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
