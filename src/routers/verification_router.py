from fastapi import APIRouter

from src.controllers.verification_controller import VerificationController
from src.models.request_models import BatchCheckRequest, CheckRequest
from src.models.response_models import BatchCheckResponse, ReturnResponse

router = APIRouter(tags=["URL Verification"])


@router.post("/check", response_model=ReturnResponse)
async def check_url(payload: CheckRequest) -> ReturnResponse:
    controller = VerificationController()
    return await controller.handle_request(payload)


@router.post("/check/batch", response_model=BatchCheckResponse)
async def check_batch(payload: BatchCheckRequest) -> BatchCheckResponse:
    controller = VerificationController()
    return await controller.handle_batch(payload)
