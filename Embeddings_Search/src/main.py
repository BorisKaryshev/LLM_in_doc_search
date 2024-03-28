from .logger import setup_logger, setup_default_logger

import logging
import json

setup_default_logger()
logger = logging.getLogger()


from .search import Searcher

def main() -> None:
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
                res = (
                    f'Question: {question}\n'
                    f'Answer: {res}\n'
                )                
            exit(0)

    question = input("Your request is: ")
    while question.upper() != "EXIT":
        res = searcher.ask_question(question)
        print(res)
        question = input("Your request is: ")
    

if __name__ == "__main__":
    main()
