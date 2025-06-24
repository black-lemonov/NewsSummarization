import uvicorn
from fastapi import FastAPI

from src.lifespan import lifespan
from src.routers.ratings import ratings_router
from src.routers.summaries import summaries_router
from src.routers.admin import admin_router


app = FastAPI(lifespan=lifespan)

app.include_router(summaries_router)
app.include_router(ratings_router)
app.include_router(admin_router)


if __name__ == "__main__":
    uvicorn.run("src.main:app", reload=True)