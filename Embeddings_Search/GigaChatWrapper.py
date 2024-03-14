from MyRetriever import MyRetriever

from langchain.chat_models.gigachat import GigaChat
from langchain.prompts.chat import ChatPromptTemplate
from pandas import DataFrame


CREDENTIALS = "ZjkyMWJhMmUtMTA3OC00NzBmLTk1NTctZjkwZTFlM2EwMjEzOjBkZmNkMzcxLWM3Y2EtNGRmOS1hYzlmLWZjMjBmNGNlMDJlMg=="

TEMPLATE = (
    "Ты эксперт по поиску информации в тексте. Твоя задача максимально кратко, но в то же время полно ответить на вопрос по тексту.\n"
    "Отвечай на языке запроса! При необходимости переведи текст.\n"    
    "Далее приведен текст \"{context}\""
)


class GigaChatWrapper:
    def __init__(self, database: DataFrame):
        self.__retriever = MyRetriever(database)

        self.__chat_prompt = ChatPromptTemplate.from_messages([
            ("system", TEMPLATE),
            ("human", "{text}"),
        ])

        self.__chat = GigaChat(credentials=CREDENTIALS, verify_ssl_certs=False)

    def ask_question(self, question: str) -> str:
        prompt = self.__chat_prompt.format_messages(
            context=self.__retriever.get_relevant_documents(question),
            text=question      
        )
        return self.__chat.invoke(prompt).content
