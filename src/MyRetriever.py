from .embeddings import create_embeddings

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from numpy import dot, ndarray
from numpy.linalg import norm
from typing import List, Callable
from pandas import DataFrame, Series
import logging


logger = logging.getLogger()


def cosine_distances(a, b):
    return dot(a, b)/(norm(a)*norm(b))


class MyRetriever(BaseRetriever):

    def __init__(self, data: DataFrame):
        logger.info("Creating MyRetriever")
        super().__init__()
        
        self.__dict__['_num_of_relevant_chunks'] = 2
        self.__dict__['_embedder'] = HuggingFaceEmbeddings()
        self.__dict__['_data'] = create_embeddings(data, self._embedder, max_tokens=256)
        logger.info(f"MyRetriever created with data")

        
    def set_num_of_relevant_chunks(self, num: int) -> None:
        self._num_of_relevant_chunks = num

    def __get_text_from_table(self, relevant_chunk_index: int) -> str:
        if self._data.empty:
            return ""

        filename = self._data['name']
        filtered_data = self._data['text']
        
        filename = f"Filename: {filename.astype(str).values[relevant_chunk_index]}\n"
        
        relevant_chunk = filtered_data.astype(str).values[relevant_chunk_index]

        result = filename + relevant_chunk
        logger.info(f"Relevant document is: {result}")
        return result
    
    def __sort_data_by_relevancy(self, query: str) -> None:
        query_embedding = self._embedder.embed_query(query)
        self._data.sort_values(
            by=['embedding'],
            key=lambda series: series.apply(lambda x: cosine_distances(query_embedding, x)),
            inplace=True,
            ascending=False,
        )
    
    def get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        self.__sort_data_by_relevancy(query)
        
        indexes = self._data['chunk'].astype(str).values
        logger.debug(f"After searching for best documents got indexes: {indexes}")

        return "\n".join([
            self.__get_text_from_table(index)
            for index in range(self._num_of_relevant_chunks)
        ])
