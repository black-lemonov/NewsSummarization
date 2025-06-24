from string import punctuation

from nltk import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


class StemmedTfidfVectorizer(TfidfVectorizer):
    """
    TF-IDF векторизатор со стеммингом
    """
    def build_analyzer(self):
        russian_stemmer = SnowballStemmer("russian")
        stop_words = stopwords.words("russian") + list(punctuation)
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (
            russian_stemmer.stem(w) for w in analyzer(doc)
            if w not in stop_words
        )
