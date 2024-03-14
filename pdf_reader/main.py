from PdfReader import load_pdfs
from pandas import read_csv, DataFrame

from langchain.chat_models.gigachat import GigaChat
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List


CREDENTIALS = "ZjkyMWJhMmUtMTA3OC00NzBmLTk1NTctZjkwZTFlM2EwMjEzOjBkZmNkMzcxLWM3Y2EtNGRmOS1hYzlmLWZjMjBmNGNlMDJlMg=="


#configure_logger('pdf_reader.log')

TEMPLATE = (
    "Ты эксперт по поиску информации в тексте. Твоя задача максимально кратко, но в то же время полно ответить на вопрос по тексту.\n"
    "Отвечай на языке запроса! При необходимости переведи текст.\n"    
    "Далее приведен текст \"{context}\""
)


PAPERS = [
    "14.3-bit Extended Counting ADC with Built-in Binning Function for Medical X-Ray Imagers.pdf",
    "A 9-V_Lux-s 5000-Framess 512x512 CMOS Sensor.pdf",
    "A Flexible 14-Bit Column-Parallel ADC Concept for Application in Wafer-Scale X-ray CMOS Imagers.pdf",
]


QUESTIONS_WITH_SOURCE = {
    "О чём статья?" : PAPERS[0],
    "С каким техпроцессом был изготовлен прототип?" : PAPERS[0], # 0.35 мкм

    "О чём статья?" : PAPERS[1],
    "Опиши архитектуру сенсора" : PAPERS[1],
    "С какой максимальной частотой сенсор корректно работает?" : PAPERS[1], # 5000 fps

    "О чём статья?" : PAPERS[2],
    "Какая рабочая частота смены кадров устройства?" : PAPERS[2], # 100 fps
    "Какая глубина конверсии?" : PAPERS[2], # 14 bit
    "Какое напряжение питания устройства?" : PAPERS[2], # 3.5V
}


class MyRetriever(BaseRetriever):

    def __init__(self, data: DataFrame): 
        super().__init__()
        self.__dict__['_data'] = data

    def __get_text_from_table(self, source_name: str) -> str:
        if source_name in self._data:
            raise AttributeError("Context not found")

        filtered_data = self._data[self._data["name"] == source_name]
        result = filtered_data.astype(str).values[0]
        return result[1]
        
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:        
        return [
            Document(page_content=self.__get_text_from_table(QUESTIONS_WITH_SOURCE[query]))
        ]


def main() -> None:
    # update text database
    database = 'data.csv'
    folder_name = 'C:\Files\Science\\3. Использование LLM при проектировании ЭРА\Обработка пдф-ок\Статьи\\'
    database = load_pdfs(folder_name, database, num_of_jobs=1)
    
    database.to_csv('data.csv', index=False)

    retriever = MyRetriever(database)

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", TEMPLATE),
        ("human", "{text}"),
    ])
    
    chat = GigaChat(credentials=CREDENTIALS, verify_ssl_certs=False)
    for question in QUESTIONS_WITH_SOURCE.keys():
        prompt = chat_prompt.format_messages(
            context=retriever.get_relevant_documents(question),
            text=question      
        )
        res = chat.invoke(prompt)
        print(
            f'Question: {question}\n'
            f'Answer: {res.content}\n'
        )

if __name__ == "__main__":
    main()
