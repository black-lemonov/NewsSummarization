from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from src.dependencies import get_summarizer
from src.models.news import News
from src.models.summary import Summary
from src.dto.summaries import SourceScheme
from src.services.news_service import get_news_content_by_urls


async def get_cluster_news_urls(session: AsyncSession, cluster_n: int) -> list[str]:
    result = await session.execute(
        select(News.url)
        .where(News.cluster_n == cluster_n)
    )
    return list(result.scalars().all())


async def get_cluster_summary(session: AsyncSession, cluster_n: int) -> Summary | None:
    news_urls = await get_cluster_news_urls(session, cluster_n)
    if not news_urls:
        return None
    result = await session.execute(
        select(Summary)
        .where(Summary.news_url.in_(news_urls))
    )
    return result.scalars().first()


async def get_paginated_summaries(
        session: AsyncSession,
        page: int,
        size: int
) -> tuple[list, int]:
    offset = page * size
    query = (
        select(
            News.title,
            Summary.content,
            News.published_at,
            News.cluster_n
        )
        .join(Summary.news)
        .where(News.cluster_n.is_not(None))
        .order_by(desc(News.published_at))
        .offset(offset)
        .limit(size)
    )
    result = await session.execute(query)
    summaries = list(result.all())

    count_query = (
        select(func.count())
        .select_from(News)
        .join(Summary.news)
    )
    total_count = (await session.execute(count_query)).scalar() or 0

    return summaries, total_count


async def get_summary_with_sources(session: AsyncSession, cluster_n: int) -> tuple | None:
    sources_query = select(News.url, News.title).where(News.cluster_n == cluster_n)
    sources_result = await session.execute(sources_query)

    summary_query = (
        select(News.title, Summary.content, News.published_at)
        .join(Summary.news)
        .where(News.cluster_n == cluster_n)
        .limit(1)
    )
    summary_result = await session.execute(summary_query)
    summary = summary_result.first()

    if not summary:
        return None

    return summary, [
        SourceScheme(url=row.url, title=row.title)
        for row in sources_result.all()
    ]


async def check_if_summary_exist(session: AsyncSession, news_url: str) -> bool:
    query = (
        select(Summary)
        .where(Summary.news_url == news_url)
    )
    summary = (await session.execute(query)).first()

    return summary is not None


def add_summary(session: AsyncSession, news_url: str, summary: str) -> None:
    summary = Summary(
        news_url=news_url,
        content=summary
    )
    session.add(summary)


async def create_summary_for_news(session: AsyncSession, news_url: str):
    content = (await get_news_content_by_urls(session, [news_url]))[0]

    summarizer = get_summarizer()
    summary = summarizer.summarize(content)

    summary = Summary(
        news_url=news_url,
        content=summary
    )
    session.add(summary)
