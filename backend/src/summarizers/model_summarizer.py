from src.preprocessing.features_extractor import FeaturesExtractor
from src.summarizers.base_summarizer import BaseSummarizer
from src.summarizers.utils.model_loader import ModelLoader


class ModelSummarizer[T](BaseSummarizer):
    def __init__(self, model_filepath: str, model_loader: ModelLoader) -> None:
        self._fe = FeaturesExtractor()
        self.__model = model_loader.load_model(model_filepath)
    
    def summarize(self, text: str) -> str:
        features_df = FeaturesExtractor().convert_text_to_df(text)
        labels = self.__model.predict(features_df.drop("text", axis=1))

        return ' '.join(
            [
                sent
                for sent, label in zip(features_df["text"], labels)
                if label == 1
            ]
        )