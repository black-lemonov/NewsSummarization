from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

from src.config import SUMMARY_SIZE
from src.summarizers.base_summarizer import BaseSummarizer


class TRSummarizer(BaseSummarizer):
    def summarize(self, text: str) -> str:
        if text is None or len(text) == 0:
            return ""
        parser = PlaintextParser.from_string(text, Tokenizer("russian"))
        summarizer = TextRankSummarizer()
        sentences_in_summary = min(SUMMARY_SIZE, len(parser.document.sentences))
        summary = summarizer(parser.document, sentences_in_summary)

        return ' '.join([str(sent) for sent in summary])