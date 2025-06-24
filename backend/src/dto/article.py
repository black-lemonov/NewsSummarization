from dataclasses import dataclass
from datetime import datetime


@dataclass
class Article:
    url: str
    title: str
    content: str
    date: datetime