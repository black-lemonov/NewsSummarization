from fastapi import APIRouter, BackgroundTasks, File, Body

from src.config import SUMM_MODELS_FILEPATHS
from src.dependencies import SessionDep
from src.parsers.parsers_selection import get_parsers_sites_urls, remove_parser
from src.summarizers.utils.model_selection import set_model_by_name, get_selected_model_name

admin_router = APIRouter(prefix="/admin", tags=["Управление 🤖"])


@admin_router.post("/summaries/{cluster_n}", summary="Создать новый реферат для кластера")
async def generate_new_summary(
        cluster_n: int,
        session: SessionDep,
        bg_tasks: BackgroundTasks
):
    return {"status": "OK", "message": "Кластер удален"}


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


@admin_router.get("/parsers", summary="Получить список новостных сайтов для парсинга")
def get_all_parsers():
    return {"sites_urls": get_parsers_sites_urls()}


@admin_router.delete("/parsers/{site_url}", summary="Удалить новостной сайт из парсинга")
def delete_parser(site_url: str):
    remove_parser(site_url)
    return {"status": "OK", "message": "Парсер удален"}


@admin_router.post("/parsers", summary="Загрузить JSON-файл с конфигурацией парсера")
def load_parser():
    pass


@admin_router.get("/data", summary="Получить .csv статей и рефератов")
def load_csv_data():
    pass