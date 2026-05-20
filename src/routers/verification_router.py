from fastapi import APIRouter, BackgroundTasks

from src.controllers.verification_controller import VerificationController
from src.models.request_models import BatchCheckRequest, CheckRequest
from src.models.response_models import AsyncBatchResponse, BatchCheckResponse, ReturnResponse

router = APIRouter(tags=["URL Verification"])


@router.post("/check")
async def check_url(payload: CheckRequest) -> ReturnResponse:
    controller = VerificationController()
    return await controller.handle_request(payload)


@router.post("/check/batch")
async def check_batch(payload: BatchCheckRequest) -> BatchCheckResponse:
    controller = VerificationController()
    return await controller.handle_batch(payload)


@router.post("/check/batch/async")
async def check_batch_async(
    payload: BatchCheckRequest,
    background_tasks: BackgroundTasks,
) -> AsyncBatchResponse:
    controller = VerificationController()
    return await controller.handle_batch_async(payload, background_tasks)
