from .logger import setup_logger, setup_default_logger

from typing import Callable
import logging
import argparse
import json


setup_default_logger()
logger = logging.getLogger()


from .search import Searcher


def ask_question_from_file(searcher: Searcher, path: str, outstream: Callable[[str], None] = print):
    questions = None
    try:
        with open(path, 'r') as f:
            questions = json.load(f)["questions"]             
    except Exception as ex:
        logger.error(f"Failed while loading questions. Reading from stdin: {ex}. ")
    else:
        for question in questions:
            res = searcher.ask_question(question)  
            outstream(res)


def main(config: dict, searcher_name: str) -> None:

    searcher_config = configs.get(searcher_name)
    if not searcher_config:
        logger.error("Could not load searcher config")
    
    searcher = Searcher(searcher_config)
    
    if searcher_config.get("questions_path"):
        ask_question_from_file(searcher, searcher_config.get("questions_path"))

    question = input()
    while question.upper() != "EXIT":
        res = searcher.ask_question(question)
        print(res)
        question = input()

if __name__ == "__main__":
    main()
