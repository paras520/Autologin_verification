import asyncio
import contextlib
import logging
import sys
from contextlib import asynccontextmanager
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
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from src.config import config  # noqa: E402
from src.routers.verification_router import router as verification_router  # noqa: E402
from src.utils.logging_utils import configure_logging  # noqa: E402

configure_logging()
logger = logging.getLogger("autologin.app")
logger.info("Starting autologin_verification [env=%s]", config.app_env)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from temporal.config.settings import TEMPORAL_ENABLED  # noqa: PLC0415
    from temporal.workers.worker import start_worker  # noqa: PLC0415

    _worker_task: asyncio.Task | None = None

    if TEMPORAL_ENABLED:
        def _on_worker_done(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logger.error("Temporal worker crashed: %s", task.exception(), exc_info=task.exception())

        _worker_task = asyncio.create_task(start_worker())
        _worker_task.add_done_callback(_on_worker_done)
        logger.info("Temporal worker started as background task")
    else:
        logger.info("Temporal disabled (TEMPORAL_STATE=OFF) — skipping worker startup")

    yield

    if _worker_task and not _worker_task.done():
        _worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _worker_task


app = FastAPI(title="URL Verification API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(verification_router, prefix="/autourl-qa")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=config.uvicorn_host,
        port=config.port,
        loop="asyncio",
    )
