from sqlalchemy import ForeignKey
from sqlalchemy.orm import mapped_column, Mapped, relationship

from src.models.base import Base


class Summary(Base):
    __tablename__ = "summary"

    news_url: Mapped[str] = mapped_column(
        ForeignKey("news.url", ondelete="CASCADE", onupdate='CASCADE'),
        primary_key=True
    )
    content: Mapped[str]
    positive_rates: Mapped[int] = mapped_column(default=0)
    negative_rates: Mapped[int] = mapped_column(default=0)

    news = relationship(
        "News",
        back_populates="summary",
        single_parent=True
    )