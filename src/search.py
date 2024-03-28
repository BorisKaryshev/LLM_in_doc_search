from .GigaChatWrapper import ChatWrapper
from .PdfReader import load_pdfs
from .MyRetriever import MyRetriever

from pandas import read_csv, DataFrame
import logging
from typing import Optional
import os.path
from langchain.chat_models.gigachat import GigaChat
from langchain.prompts.chat import ChatPromptTemplate

logger = logging.getLogger()


DEFAULT_DATABASE_LOCATION = "./data.csv"


def try_deco(func):
    def inner(*args, **kwargs):
        try: 
            return func(*args, **kwargs)
        except Exception as ex:
            logger.error(f"Exception while calling {func.__name__} occurred")
            logger.error(f"{ex}")
            raise
    
    return inner


@try_deco
def create_chat_model(config: dict) -> ChatWrapper:
    chat = None
    model_name: Optional[str] = config.get("chat_model")

    if model_name is None or model_name.lower() == "gigachat":
        chat = GigaChat(credentials=config["credentials"], verify_ssl_certs=False)
    
    if chat is None:
        raise RuntimeError(f"Got unexpected chat model name {model_name}")  
    return ChatWrapper(chat, prompt_template=config.get("prompt_template"))


class Searcher:
    def __init__(self, config: dict):
        logging.info("Creating searcher")
        self.__chat_model = create_chat_model(config)
        self.__data = DataFrame()

        database_path = config.get("database_location")            
        if database_path:
            if os.path.exists(database_path):
                self.__data = read_csv(config["database_location"]) 
            else:
                logger.warning("Database file not exists")
                self.__data = DataFrame()

        if config.get("pdfs_location"):
            self.__update_database(config)
        
        if not config.get("pdfs_location") and not config.get("database_location"):
            logger.warning("Database or pdfs locations not given. Chat model will answer only based on it knowledge")

    def __update_database(self, config: dict):
        self.__data = load_pdfs(config["pdfs_location"], self.__data) 
        database_location = config.get("database_location") or DEFAULT_DATABASE_LOCATION
        self.__data.to_csv(database_location)
        
        retriever = MyRetriever(self.__data)
        self.__chat_model.set_retriever(retriever)
        
    def ask_question(self, question: str) -> str:
        try:
            return self.__chat_model.ask_question(question)
        except Exception as ex:
            logger.error(f"Asking question failed with: {ex}")
            return "Failed to answer question. See logs for details."    
