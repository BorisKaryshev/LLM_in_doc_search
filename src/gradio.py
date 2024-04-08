from search import Searcher
from logger import setup_logger, setup_default_logger

import gradio as gr
from pathlib import Path
from json import load
import logging


logger = logging.getLogger()


HELP_MESSAGE = (
    "This chat bot supports commands:\n"
    "\thelp - view this message\n"
    "\tchange_config <config name> - change config for chat bot\n"
)


class GradioLLMSearcher:
    def __init__(self, config: dict) -> None:
        self.__config = config
        self.__searcher = Searcher(config.get("datasheet_searcher"))

    def __change_searcher(self, searcher_config_name: str):
        if searcher_config_name not in self.__config:
            logger.error(f"Changing searcher config on {searcher_config_name} failed - config not found")
            return "Config not found"
        self.__searcher = Searcher(self.__config[searcher_config_name])
        logger.info(f"Successfully changed searcher config on {searcher_config_name}")
        return "Config changed successfully"

    def __call__(self, query: str) -> str:
        command = query.strip().split()[0].lower()

        if command == "change_config":
            config_name = query.strip().split()[1]
            return self.__change_searcher(config_name)
        if command == "help":
            return HELP_MESSAGE

        return self.__searcher.ask_question(query)


DEFAULT_CONFIG_PATH = Path(
    "./datasheets.json"
)


def gradio_main():
    setup_default_logger()

    config = None
    with open(DEFAULT_CONFIG_PATH, 'r') as conf_file:
        config = load(conf_file)

    setup_logger(config.get("logging"))

    searcher = GradioLLMSearcher()
    demo = gr.ChatInterface(searcher)
    demo.launch()
