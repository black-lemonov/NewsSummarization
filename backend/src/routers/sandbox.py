from fastapi import APIRouter

from src.config import SUMM_MODELS_FILEPATHS
from src.dependencies import get_summarizer
from src.summarizers.utils.model_selection import set_model_by_name, get_selected_model_name

sandbox_router = APIRouter(
    prefix="/sandbox",
    tags=["Песочница"],
)

@sandbox_router.get("/summary", summary="Получить реферат")
def get_summary(text: str):
    summarizer = get_summarizer()
    summary = summarizer.summarize(text)
    return {"summary": summary}

@sandbox_router.put("/summarizers", summary="Выбрать модель для реферирования")
def set_summarizer(model_name: str):
    set_model_by_name(model_name)
    return {"status": "OK", "message": "Модель успешно установлена"}

@sandbox_router.get("/summarizers", summary="Получить список доступных моделей")
def get_available_summarizers():
    return {"available_models": list(SUMM_MODELS_FILEPATHS.keys())}
