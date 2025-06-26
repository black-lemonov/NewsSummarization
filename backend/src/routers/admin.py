import csv
import json
import secrets
from io import StringIO

from fastapi import APIRouter, UploadFile, Depends, HTTPException, status
from fastapi.params import Security
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload
from starlette.responses import JSONResponse, Response

from src.config import SUMM_MODELS_FILEPATHS, ADMIN_USERNAME, ADMIN_PASSWORD
from src.dependencies import SessionDep
from src.models import News
from src.parsers.parsers_selection import get_parsers_sites_urls, remove_parser, add_new_parser
from src.services.bg_service import get_last_parsing_time_from_config, start_bg_task
from src.services.summaries_service import create_summary_for_news
from src.summarizers.utils.model_selection import set_model_by_name, get_selected_model_name


def verify_admin(credentials: HTTPBasicCredentials = Security(HTTPBasic())):
    username_correct = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    password_correct = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)

    if not (username_correct and password_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверные учетные данные",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


admin_router = APIRouter(
    prefix="/admin",
    tags=["Управление 🤖"],
    dependencies=[Depends(verify_admin)]
)


@admin_router.post("/summaries/", summary="Сгенерировать реферат для новости")
async def generate_summary(news_url: str, session: SessionDep):
    await create_summary_for_news(session, news_url)
    return {"status": "OK", "message": "Реферат создан"}


@admin_router.get("/models", summary="Получить список доступных моделей")
def get_available_models():
    return {"available_models": list(SUMM_MODELS_FILEPATHS.keys())}


@admin_router.get("/models/selected", summary="Получить название выбранной модели")
def get_selected_model():
    name = get_selected_model_name()
    return {"selected_model": name}


@admin_router.post("/models/selected", summary="Выбрать модель")
def set_model(model_name: str):
    set_model_by_name(model_name)
    return {"status": "OK", "message": "Модель успешно установлена"}


@admin_router.get("/parsers/sites", summary="Получить список новостных сайтов для парсинга")
def get_all_parsers():
    return {"sites_urls": get_parsers_sites_urls()}


@admin_router.delete("/parsers/sites", summary="Удалить новостной сайт из парсинга")
def delete_parser(site_url: str):
    remove_parser(site_url)
    return {"status": "OK", "message": "Парсер удален"}


@admin_router.post("/parsers", summary="Загрузить JSON-файл с конфигурацией парсера")
def load_parser(file: UploadFile):
    parser_config = json.load(file.file)
    add_new_parser(parser_config)
    return {"status": "OK", "message": "Парсер добавлен"}


@admin_router.get("/parsers/template", summary="Получить шаблон файла конфигурации")
def get_parser_template():
    return JSONResponse(
        {
            "site_url": "",
            "article_selector": "",
            "title_selector": "",
            "url_selector": "",
            "date_selector": "",
            "content_selector": "",
            "stop_words": []
        }
    )

@admin_router.get("/system", summary="Запустить парсинг и алгоритм кластеризации")
async def start_bg_task_():
    await start_bg_task()
    return {"status": "OK", "message": "Алгоритм парсинга запущен"}


@admin_router.get("/system/timestamp", summary="Получить дату последнего парсинга")
async def get_last_parsing_time():
    last_time = get_last_parsing_time_from_config()
    return {"last_parsing_time": last_time}


@admin_router.delete("/cluster/{cluster_n}", summary="Удалить кластер")
async def delete_cluster(cluster_n: int, session: SessionDep):
    await session.execute(
        delete(News)
        .where(News.cluster_n == cluster_n)
    )
    await session.commit()
    return {"status": "OK", "message": "Кластер был удален"}


@admin_router.get("/news/export", summary="Скачать таблицу .csv", response_class=Response)
async def export_news_with_summaries(session: SessionDep):
        query = (
            select(News)
            .options(selectinload(News.summary))
            .where(News.summary.any())

        )
        result = await session.execute(query)
        news_items = result.scalars().all()

        if not news_items:
            raise HTTPException(status_code=404, detail="Нет новостей с рефератами")

        output = StringIO()
        writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        headers = [
            "url", "title", "date", "content",
            "summary_content", "positive_rates", "negative_rates"
        ]
        writer.writerow(headers)

        for news in news_items:
            if not news.summary:
                continue

            summary = news.summary[0]
            writer.writerow([
                news.url,
                news.title,
                news.published_at.isoformat(),
                news.content,
                summary.content,
                summary.positive_rates,
                summary.negative_rates
            ])

        output.seek(0)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=news_with_summaries.csv",
                "Content-Type": "text/csv; charset=utf-8"
            }
        )