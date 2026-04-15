import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Playwright requires SelectorEventLoop on Windows — ProactorEventLoop
# (the default on Windows) cannot spawn subprocesses needed by Chromium.
# WindowsSelectorEventLoopPolicy is deprecated in 3.14 but still works.
# Suppress the DeprecationWarning since we have no alternative until
# Playwright natively supports ProactorEventLoop.
if sys.platform == "win32":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI  # noqa: E402

from src.routers.verification_router import router as verification_router  # noqa: E402
from src.utils.logging_utils import configure_logging  # noqa: E402

configure_logging()


app = FastAPI(title="URL Verification API")
app.include_router(verification_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        loop="asyncio",
    )
