from datetime import datetime

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.news import News


async def get_all_news_urls(session: AsyncSession) -> list[str]:
    result = await session.execute(
        select(News.url)
    )
    return list(result.scalars().all())


async def get_news_content_by_urls(session: AsyncSession, urls: list[str]) -> list[str]:
    result = await session.execute(
        select(News.content)
        .where(News.url.in_(urls))
    )
    return list(result.scalars().all())


async def set_cluster_n(session: AsyncSession, news_url: str, cluster_n: int) -> None:
    await session.execute(
        update(News)
        .where(News.url == news_url)
        .values(cluster_n=cluster_n)
    )


def add_news(
        session: AsyncSession,
        url: str,
        title: str,
        published_at: datetime,
        content: str
):
    session.add(
        News(
            url=url,
            title=title,
            published_at=published_at,
            content=content
        )
    )


async def del_cluster_in_news(session: AsyncSession, cluster_n: int) -> None:
    await session.execute(
        update(News)
        .where(News.cluster_n == cluster_n)
        .values(cluster_n=None)
    )

async def del_news_by_cluster(session: AsyncSession, cluster_n: int) -> None:
    await session.execute(
        delete(News).where(News.cluster_n == cluster_n)
    )
