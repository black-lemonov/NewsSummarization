from datetime import datetime

from pydantic import BaseModel

from src.dto.base import Pagination


class SummaryScheme(BaseModel):
    title: str
    summary: str
    created_at: datetime
    cluster_n: int


class SummarySchemeWithPagination(BaseModel):
    data: list[SummaryScheme]
    pagination: Pagination


class SourceScheme(BaseModel):
    url: str
    title: str


class SummaryWithSourcesScheme(SummaryScheme):
    news: list[SourceScheme]