#pylint: disable=W1514,W0603,W1203,C0114,C0116,C0115
from abc import ABC, abstractmethod
from sklearn.ensemble import IsolationForest

class Classifier(ABC):

    @abstractmethod
    def train(self, valid_embeddings:list):
        pass

    @abstractmethod
    def test(self, valid_embeddings:list) -> list:
        pass

    @abstractmethod
    def is_trained(self) -> bool:
        pass

class IsolationForestClassifier(Classifier):
    def __init__(self):
        self.classifier = IsolationForest(n_estimators=100, warm_start=False)

    def train(self, valid_embeddings: list):
        self.classifier.fit(valid_embeddings)


    def test(self, valid_embeddings: list) -> list:
        return self.classifier.predict(valid_embeddings)

    def is_trained(self) -> bool:
        return hasattr(self.classifier, 'estimators_') and len(self.classifier.estimators_) > 0
