from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.dto.rating import RateAction, RateType
from src.services.summaries_service import get_cluster_summary


async def update_summary_rate(
        session: AsyncSession,
        cluster_n: int,
        rate_field: str,
        action: RateAction
) -> None:
    summary = await get_cluster_summary(session, cluster_n)
    if not summary:
        raise HTTPException(status_code=404, detail="Реферат не найден")

    rate_field = "positive_rates" if rate_field == RateType.LIKE else "negative_rates"

    current_value = getattr(summary, rate_field)
    if action == RateAction.ADD:
        setattr(summary, rate_field, current_value + 1)
    elif action == RateAction.REMOVE and current_value > 0:
        setattr(summary, rate_field, current_value - 1)

    await session.commit()