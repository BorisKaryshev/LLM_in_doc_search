from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from typing import Optional, Any
import logging


logger = logging.getLogger()


class ChatWrapper:
    def __init__(self, chat_model, retriever: Optional[BaseRetriever] = None, prompt_template: Optional[str] = None):
        logger.info("Creating ModelWrapper")
        self.__chat_model = chat_model
        self.__retriever = retriever
        
        if prompt_template:
            self.__chat_prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_template),
                ("human", "{text}"),
            ])
        else:
            self.__chat_prompt = ChatPromptTemplate.from_messages([
                ("human", "{text}"),
            ])      

    def set_retriever(self, retriever: BaseRetriever):
        self.__retriever = retriever

    def ask_question(self, question: str) -> str:
        context = ""
        if self.__retriever:
            context = self.__retriever.get_relevant_documents(question)
        
        prompt = self.__chat_prompt.format_messages(
            context = context,
            text = question      
        )
        logger.debug(f"Invoking chat model with prompt: {prompt}")
        return self.__chat_model.invoke(prompt).content
