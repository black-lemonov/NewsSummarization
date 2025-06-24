from datetime import datetime

from sqlalchemy import DateTime, ForeignKey
from sqlalchemy.orm import mapped_column, Mapped, relationship

from src.models.base import Base


class News(Base):
    __tablename__ = "news"

    url: Mapped[str] = mapped_column(primary_key=True, index=True)
    title: Mapped[str]
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    content: Mapped[str]
    cluster_n: Mapped[int | None]

    summary = relationship(
        "Summary",
        back_populates="news",
        cascade="all, delete-orphan"
    )