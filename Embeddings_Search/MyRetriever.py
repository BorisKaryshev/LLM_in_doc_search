from embeddings import create_embeddings

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.embeddings import GPT4AllEmbeddings
from sklearn.metrics.pairwise import cosine_distances
from typing import List
from pandas import DataFrame


class MyRetriever(BaseRetriever):

    def __init__(self, data: DataFrame):
        super().__init__()
        
        print("============> Embeddings initializing: START")
        self.__dict__['_embedder'] = GPT4AllEmbeddings()
        self.__dict__['_data'] = create_embeddings(data, self._embedder, max_tokens=128)
        print("============> Embeddings initializing: END")

    def __get_text_from_table(self, query: str) -> str:
        query_embedding = self._embedder.embed_query(query)
        self._data.sort_values(
            by=['embedding'],
            key=lambda embds: cosine_distances(DataFrame(list(embds)), DataFrame(query_embedding).T).T[0]
        )

        filtered_data = self._data['text']
        result = filtered_data.astype(str).values[:2]
        # print('\n'.join(result))
        
        return result[0]
        
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:               
        return [
            Document(page_content=self.__get_text_from_table(query))
        ]
