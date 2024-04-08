from .logger import setup_logger, setup_default_logger

import logging
import json
from typing import Optional, TextIO


setup_default_logger()
logger = logging.getLogger()


from .search import Searcher

def main(input_stream: Optional[TextIO] = input, output_stream: Optional[TextIO] = print) -> None:
    configs = None 

    try:
        with open("datasheets.json", 'r', encoding='utf8') as config_file:
            configs = json.load(config_file)
    except Exception as ex:
        logger.error(f"Failed to read configs: {ex}")
        exit(1)
    setup_logger(configs.get("logging"))
    
    searcher_config = configs.get("datasheet_searcher")
    if not searcher_config:
        logger.error("Could not load searcher config")
    
    searcher = Searcher(searcher_config)
    
    if searcher_config.get("questions_path"):
        questions = None
        try:
            with open(searcher_config.get("questions_path"), 'r') as f:
                questions = json.load(f)["questions"]             
        except Exception as ex:
            logger.error("Failed while loading questions. Reading from stdin.")
        else:
            for question in questions:
                res = searcher.ask_question(question)  
                output_stream(res)

    question = input_stream()
    while question.upper() != "EXIT":
        res = searcher.ask_question(question)
        output_stream(res)
        question = input_stream()

class GradioInterface:
    def __init__(self, config: dict):
        self.__config = config
        self.__searcher = self.__apply_configs(config)


    @staticmethod
    def __apply_configs(config: dict): 
        searcher_config = config.get("datasheet_searcher")
        if not searcher_config:
            logger.error("Could not load searcher config")
        return Searcher(searcher_config)

if __name__ == "__main__":
    main()
