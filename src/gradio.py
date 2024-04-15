from .search import Searcher

import gradio as gr
import logging


logger = logging.getLogger()


HELP_MESSAGE = (
    "This chat bot supports commands:\n"
    "\thelp - view this message\n"
    "\tchange_config <config name> - change config for chat bot\n"
)


class SearcherForGradio:
    def __init__(self, config: dict, searcher: str):
        self.__config = config
        self.change_searcher(config)

    def change_searcher(self, searcher_name: str):
        try:
            searcher_config = self.__config[searcher_name]
        except Exception:
            logger.error(f"Could not load searcher config {searcher_name}")
            return

        self.__searcher =  Searcher(searcher_config)

    def ask_question(self, query: str) -> str:
        return self.__searcher.ask_question(query)
    
    def add_document(self, document_path: str) -> None:
        self.__searcher.add_document(document_path)


class GradioLLMSearcher:
    def __init__(self, config: dict, searcher_name: str) -> None:
        self.__searcher = SearcherForGradio(config, searcher_name)

    def __call__(self, query: str) -> str:
        command = query.strip().split()[0].lower()

        if command == "change_config":
            searcher_name = query.strip().split()[1]
            return self.__searcher.change_searcher(searcher_name)
        if command == "help":
            return HELP_MESSAGE

        return self.__searcher.ask_question(query)


def gradio_main(config: dict, searcher_name: str, publish_link_to_web: bool = False):
    searcher = GradioLLMSearcher(config, searcher_name)
    demo = gr.ChatInterface(searcher)
    demo.launch(share=publish_link_to_web)
