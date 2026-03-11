from __future__ import annotations

import logging

from fastapi import HTTPException

from src.models.request_models import CheckRequest
from src.models.response_models import ReturnResponse
from src.services.verification_service import verify_url

logger = logging.getLogger("autologin.verification_controller")


class VerificationController:
    async def handle_request(self, payload: CheckRequest) -> ReturnResponse:
        try:
            return await verify_url(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Unexpected controller error: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error while processing /check request.",
            ) from exc
