#pylint: disable=W1514,W0603,W1203,C0114,C0116,C0115
from abc import ABC, abstractmethod

from gensim import models
from numpy import ndarray

class Embedder(ABC):

    @abstractmethod
    def embed(self, tokens:list) -> list[ndarray]:
        pass

class FastTextEmbedder(Embedder):
    def __init__(self, model_path: str):
        self.model = models.fasttext.load_facebook_vectors(model_path)

    def embed(self, tokens: list) -> list[ndarray]:
        return self.model[tokens]
    
    def similarity(self, token1: str, token2: str) -> float:
        """
        Compute the cosine similarity between two tokens.
        """
        return self.model.similarity(token1, token2)
