import re
import pandas as pd
from razdel import sentenize


class FeaturesExtractor():
    """Преобразователь текста в набор признаков"""
    
    __ABBR_PAT = r"[А-Я][А-Я]+, \d{1,2}.{5,10}[A-Я][А-Я]+. " 
    __ABBR_PAT2 = r'[А-Я][а-я]+, , \d{1,2}:\d{1,2}.{3,5}[A-Z][A-Z]{1,10}.'
    __TASS_ABBR_PAT = r"[А-Я][А-Я]+, \d{1,2} [а-я]{3,8}. /[А-Я][А-Я]+/. "
    __URL_PAT = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)'
        r'(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+'
        r'(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
    )
    __QUOTE_PAT = "«|\"|\'"    

    def __get_sentences(self, raw_text: str) -> list[str]:
        """Разбивает текст на предложения"""
        raw_text = re.sub(self.__ABBR_PAT, "", raw_text)
        raw_text = re.sub(self.__ABBR_PAT2, "", raw_text)
        raw_text = re.sub(self.__TASS_ABBR_PAT, "", raw_text)
        
        return [s.text for s in sentenize(raw_text) if not self.__URL_PAT.search(s.text)]  


    def __sents_to_features_vec(self, text_sents: list[str]) -> list[list[str]]:
        """
        Вычислияет признаки для предложений 
        :returns: Список списков формата [have_quote, sent_len, sent_pos]
        """
        
        have_quote = [
            re.search(self.__QUOTE_PAT, s) is not None for s in text_sents
        ]
        
        sents_count = len(text_sents)
        sents_pos = [
            i/sents_count for i in range(1, sents_count+1)
        ]
        
        text_len = sum([len(s) for s in text_sents])
        sents_len = [
            len(s)/text_len for s in text_sents
        ]
        
        return [
            [quote, text_len, text_pos]
            for quote, text_len, text_pos in zip(have_quote, sents_len, sents_pos)
        ]
        
    def convert_text_to_df(self, text: str) -> pd.DataFrame:    
        """Преобразует текст в датафрейм со столбцами [text, have_quote, sent_len, sent_pos]"""
        sentences = self.__get_sentences(text)
        sents_df = pd.DataFrame(sentences, columns=["text"])
        
        features = self.__sents_to_features_vec(sentences)
        features_df = pd.DataFrame(features, columns=['have_quote', 'sent_len', 'sent_pos'])
        
        return pd.concat([sents_df, features_df], axis=1)
