from .GigaChatWrapper import ChatWrapper
from .PdfReader import load_pdfs
from .MyRetriever import MyRetriever

from pandas import read_csv, DataFrame
from typing import Optional
from pathlib import Path
from langchain.chat_models.gigachat import GigaChat
from langchain.prompts.chat import ChatPromptTemplate
import logging
import os.path


logger = logging.getLogger(__name__)


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
        chat = GigaChat(model="GigaChat-Plus", credentials=config["credentials"], verify_ssl_certs=False)
    
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

        self.__retriever.set_num_of_relevant_chunks(config.get("num_of_relevant_chunks", 2))

    def __update_database(self, config: dict):
        if config.get("pdfs_location"):
            self.__data = load_pdfs(config["pdfs_location"], self.__data)
        database_location = config.get("database_location") or DEFAULT_DATABASE_LOCATION
        self.__data.to_csv(database_location)

        retriever = MyRetriever(self.__data,
                                config.get("embedder_name"),
                                config.get("max_num_of_tokens", 256))
        self.__retriever = retriever

    def add_document(self, path: Path) -> None:
        self.__retriever.add_document(path)

    def ask_question(self, question: str) -> str:
        try:
            logger.info(f"Asking question: {question}")
            context = self.__retriever.get_relevant_documents(question)
            result = self.__chat_model.ask_question(question, context)
            logger.info(f"Got result: {result}")
            return result 
        except Exception as ex:
            logger.error(f"Asking question failed with: {ex}")
            return "Failed to answer question. See logs for details."
