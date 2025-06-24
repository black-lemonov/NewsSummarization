import json
from abc import ABC, abstractmethod

from src.parsers.news_parser import NewsParser


class ParsersFactory(ABC):
    @abstractmethod
    def load_parsers(self) -> list[NewsParser]:
        ...


class JSONParsersFactory(ParsersFactory):
    def load_parsers(self) -> list[NewsParser]:
        with open("config.json") as conf_file:
            conf_dict = json.load(conf_file)
            return [
                NewsParser(**conf)
                for conf in conf_dict["parsers_configs"]
                if conf["site_url"] in conf_dict["selected_sites_urls"]
            ]
