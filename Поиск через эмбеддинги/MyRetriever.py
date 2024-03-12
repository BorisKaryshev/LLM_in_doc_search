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
        self.__dict__['_data'] = data
        # self._data = data
        if 'chunk' not in self._data:
            self.__dict__['_data'] = create_embeddings(self._data, max_tokens=128)
        # self.__dict__['_embedder'] = GigaChatEmbeddings(credentials=CREDENTIALS, verify_ssl_certs=False)

    def __get_text_from_table(self, query: str) -> str:
        # query_embedding = self._embedder.embed_documents(texts=[query])[0]
        # self._data.sort_values(
        #     by=['embedding'],
        #     key=lambda x: cosine_distances(x, query_embedding)
        # )

        filtered_data = self._data['text']
        result = '\n'.join(filtered_data.astype(str).values[:2])
        print(result)
        
        # exit(0)
        return result
        
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:               
        return [
            Document(page_content=self.__get_text_from_table(query))
        ]
