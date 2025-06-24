from xgboost import XGBClassifier

from src.summarizers.utils.model_loader import ModelLoader


class XGBLoader(ModelLoader):
    def load_model(self, filepath: str):
        return XGBClassifier().load_model(filepath)
