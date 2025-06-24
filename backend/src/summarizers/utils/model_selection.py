import json

from src.config import SUMM_MODELS_FILEPATHS


def get_selected_model_name() -> str:
    with open("config.json") as f:
        summ_config = json.load(f)
        return summ_config["selected_model"]


def set_model_by_name(model_name: str) -> None:
    with open("config.json") as f:
        summ_config = json.load(f)
        if model_name in SUMM_MODELS_FILEPATHS.keys():
            summ_config["selected_model"] = model_name

    with open("config.json", 'w') as f:
        json.dump(summ_config, f)
