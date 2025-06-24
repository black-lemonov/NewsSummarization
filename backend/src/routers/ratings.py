from fastapi import APIRouter

from src.dependencies import SessionDep
from src.dto.rating import RateType, RateAction
from src.dto.responses import BaseResponse
from src.services.rating_service import update_summary_rate


ratings_router = APIRouter(prefix="/summaries", tags=["Оценки"])


@ratings_router.patch(
    "/{cluster_n}/{rate_type}/{action}",
    summary="Обновить рейтинг реферата",
    response_model=BaseResponse
)
async def update_summary_rate_endpoint(
        cluster_n: int,
        rate_type: RateType,
        action: RateAction,
        session: SessionDep,
):
    await update_summary_rate(
        session,
        cluster_n,
        rate_type,
        action
    )

    return {"status": "OK", "message": "Оценка добавлена"}