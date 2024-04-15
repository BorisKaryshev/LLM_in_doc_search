from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
from typing import Optional
import logging


logger = logging.getLogger()


class Embedder(ABC):
    @abstractmethod
    def embed_query(self, query: str) -> list:
        pass


class GPT4AllEmbedder(Embedder):
    def __init__(self):
        self.__embedder = GPT4AllEmbeddings()

    def embed_query(self, query: str) -> list:
        return self.__embedder.embed_query(query)
    
    
class E5Embedder(Embedder): 
    def __init__(self):
        self.__embedder = SentenceTransformer('intfloat/e5-large-v2')

    def embed_query(self, query: str) -> list:
        return self.__embedder.encode([query])[0]


def get_embedder(embedder_name: Optional[str] = None) -> Embedder:
    embedders = {
        "gpt4all" : GPT4AllEmbedder,
        "e5" : E5Embedder,
    }
    if embedder_name is None:
        logger.info("Creating embedder gpt4all")
        return GPT4AllEmbedder()
    
    embedder = embedders.get(embedder_name)
    if embedder is None:
        raise RuntimeError(f"Failed to create embedder, name not found {embedder_name}")
    logger.info(f"Created embedder {embedder_name}")
    return embedder
