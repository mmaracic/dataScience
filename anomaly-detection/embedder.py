#pylint: disable=W1514,W0603,W1203,C0114,C0116,C0115
from abc import ABC, abstractmethod

from gensim import models

class Embedder(ABC):

    @abstractmethod
    def embed(self, valid_embeddings:list):
        pass

class FastTextEmbedder(Embedder):
    def __init__(self, model_path: str):
        self.model = models.fasttext.load_facebook_vectors(model_path)

    def embed(self, valid_embeddings: list) -> list:
        return self.model[valid_embeddings]
