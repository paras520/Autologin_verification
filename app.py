from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI  # noqa: E402

from src.routers.verification_router import router as verification_router  # noqa: E402
from src.utils.logging_utils import configure_logging  # noqa: E402

configure_logging()

app = FastAPI(title="URL Verification API")
app.include_router(verification_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
