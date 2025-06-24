from abc import ABC, abstractmethod


class BaseSummarizer(ABC):
    @abstractmethod
    def summarize(self, source_text: str) -> str:
        pass
    