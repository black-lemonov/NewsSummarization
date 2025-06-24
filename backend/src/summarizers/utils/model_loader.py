from abc import ABC, abstractmethod


class ModelLoader(ABC):
    @abstractmethod
    def load_model(self, filepath: str):
        pass