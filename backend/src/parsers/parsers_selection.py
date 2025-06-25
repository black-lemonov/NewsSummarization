import json

from src.dependencies import get_logger


def get_parsers_sites_urls() -> list[str]:
    with open("config.json") as f:
        conf = json.load(f)
        return conf["selected_sites_urls"]


def add_new_parser(parser_config: dict) -> None:
    logger = get_logger()
    logger.debug(parser_config)
    with open("config.json") as f:
        conf = json.load(f)
        conf["parsers_configs"].append(parser_config)
        conf["selected_sites_urls"].append(parser_config["site_url"])

    with open("config.json", "w") as f:
        json.dump(conf, f)


def remove_parser(site_url: str) -> None:
    with open("config.json") as f:
        conf = json.load(f)
        if site_url in conf["selected_sites_urls"]:
            conf["selected_sites_urls"].remove(site_url)

    with open("config.json", 'w') as f:
        json.dump(conf, f)
