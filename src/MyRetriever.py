from .PdfReader import load_pdfs
from .embeddings import create_embeddings

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from numpy.linalg import norm, dot
from typing import List
from pandas import DataFrame, concat
from pathlib import Path
from uuid import uuid4
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

    def add_document(self, path: Path):
        logger.info(f"Adding document {path}")
        if path.is_dir():
            raise RuntimeError("Got folder while adding single document")
        if len(path.suffixes) > 1 and path.suffix != ".pdf":
            raise RuntimeError(f"Only supported format is pdf, got {path}")
        
        tmp_dir = Path(f"./{uuid4()}").mkdir()
        prev_path = path
        path.rename(tmp_dir / path.name)
        
        try:        
            data = load_pdfs(tmp_dir, DataFrame())
        except Exception:
            path.rename(prev_path)
            raise

        self._data = concat([self._data, data], ignore_index=True, sort=False)
        logger.info("Document added successfully")

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
