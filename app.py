from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from src.routers.verification_router import router as verification_router
from src.utils.logging_utils import configure_logging

load_dotenv(Path(__file__).resolve().parent / ".env")
configure_logging()

app = FastAPI(title="URL Verification API")
app.include_router(verification_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
