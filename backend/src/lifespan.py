import asyncio
from contextlib import asynccontextmanager

import nltk
from fastapi import FastAPI

from src.database import init_db
from src.logger import init_logger
from src.services.bg_service import start_bg_task


@asynccontextmanager
async def lifespan(app: FastAPI):
    nltk.download("stopwords")
    init_logger()
    await init_db()
    asyncio.create_task(start_bg_task())
    yield