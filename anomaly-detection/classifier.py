#pylint: disable=W1514,W0603,W1203,C0114,C0116,C0115,C2801
from abc import ABC, abstractmethod
import logging
import numpy as np
from numpy import ndarray

from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import OPTICS
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Classifier(ABC):

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __len__(self) -> int:
        return 1

    @abstractmethod
    def train(self, valid_embeddings:list, valid_words: list[list[str]], valid_labels: list[int]):
        pass

    @abstractmethod
    def test(self, valid_embeddings:list, valid_words: list[list[str]]) -> ndarray:
        pass

    @abstractmethod
    def is_trained(self) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

def get_classifier(classifier_name: str) -> Classifier:
    if classifier_name == "IsolationForest":
        return IsolationForestClassifier()
    elif classifier_name == "OneClassSVM":
        return OneClassSVMClassifier()
    elif classifier_name == "OpticsClusterClassifier":
        return OpticsClusterClassifier()
    elif classifier_name == "DecisionTreeClassifier":
        return DecTreeClassifier()
    elif classifier_name == "MultiClassifierIsoForestLocalOutlier":
        return MultiClassifierIsoForestLocalOutlier()
    elif classifier_name == "MultiClassifierIsoForestOpticsCluster":
        return MultiClassifierIsoForestOpticsCluster()
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

class MultiClassifier(Classifier):
    def __init__(self, classifiers: list[Classifier]):
        self.classifiers = classifiers

    def __len__(self) -> int:
        return len(self.classifiers)

    def __str__(self) -> str:
        return "MultiClassifier_" + "_".join([f" {str(classifier)}" for classifier in self.classifiers])

    def train(self, valid_embeddings: list, valid_words: list[list[str]], valid_labels: list[int]):
        for classifier in self.classifiers:
            classifier.train(valid_embeddings, valid_words, valid_labels)

    def test(self, valid_embeddings: list, valid_words: list[list[str]]) -> ndarray:
        results = np.zeros((len(self.classifiers), len(valid_embeddings)))
        for index, classifier in enumerate(self.classifiers):
            results[index] = classifier.test(valid_embeddings, valid_words)
        return results

    def is_trained(self) -> bool:
        return all(classifier.is_trained() for classifier in self.classifiers)
    
    def clear(self) -> None:
        for classifier in self.classifiers:
            classifier.clear()

class IsolationForestClassifier(Classifier):
    def __init__(self):
        self.classifier = IsolationForest(n_estimators=100, warm_start=False)

    def __str__(self) -> str:
        return "IsolationForestClassifier"

    def train(self, valid_embeddings: list, valid_words: list[list[str]], valid_labels: list[int]):
        self.classifier.fit(valid_embeddings)

    def test(self, valid_embeddings: list, valid_words: list[list[str]]) -> ndarray:
        return self.classifier.predict(valid_embeddings)

    def is_trained(self) -> bool:
        return hasattr(self.classifier, 'estimators_') and len(self.classifier.estimators_) > 0
    
    def clear(self) -> None:
        self.__init__()


class OneClassSVMClassifier(Classifier):
    def __init__(self):
        self.classifier = OneClassSVM(kernel='rbf', gamma='auto')

    def __str__(self) -> str:
        return "OneClassSVMClassifier"

    def train(self, valid_embeddings: list, valid_words: list[list[str]], valid_labels: list[int]):
        self.classifier.fit(valid_embeddings)

    def test(self, valid_embeddings: list, valid_words: list[list[str]]) -> ndarray:
        return self.classifier.predict(np.array(valid_embeddings))

    def is_trained(self) -> bool:
        return hasattr(self.classifier, 'support_vectors_') and len(self.classifier.support_vectors_) > 0

    def clear(self) -> None:
        self.__init__()

class LocalOutlierFactorClassifier(Classifier):
    def __init__(self):
        self.classifier = LocalOutlierFactor(n_neighbors=20, novelty=True)

    def __str__(self) -> str:
        return "LocalOutlierFactorClassifier"

    def train(self, valid_embeddings: list, valid_words: list[list[str]], valid_labels: list[int]):
        self.classifier.fit(np.array(valid_embeddings))

    def test(self, valid_embeddings: list, valid_words: list[list[str]]) -> ndarray:
        return self.classifier.predict(np.array(valid_embeddings))

    def is_trained(self) -> bool:
        return hasattr(self.classifier, 'negative_outlier_factor_') and len(self.classifier.negative_outlier_factor_) > 0

    def clear(self) -> None:
        self.__init__()

class OpticsClusterClassifier(Classifier):
    def __init__(self):
        self.classifier = OPTICS(min_samples=2, max_eps=np.inf)
        self.label_counts = {}
        self.max_cluster_size = 1000

    def __str__(self) -> str:
        return "OpticsClusterClassifier"

    def train(self, valid_embeddings: list, valid_words: list[list[str]], valid_labels: list[int]):
        self.classifier.fit(np.array(valid_embeddings))
        label_statistics = np.unique(self.classifier.labels_, return_counts=True)
        self.label_counts = dict(zip(label_statistics[0], label_statistics[1]))
        logging.debug(f"OpticsClusterClassifier label counts: {self.label_counts}")

    def test(self, valid_embeddings: list, valid_words: list[list[str]]) -> ndarray:
        labels = self.classifier.fit_predict(np.array(valid_embeddings))
        results = np.zeros(len(valid_embeddings))
        for index, label in enumerate(labels):
            if label == -1:
                results[index] = -1
            elif self.label_counts.get(label, 0) > self.max_cluster_size:
                results[index] = -1
            else:
                results[index] = 1
        return results

    def is_trained(self) -> bool:
        return hasattr(self.classifier, 'labels_') and len(self.classifier.labels_) > 0

    def clear(self) -> None:
        self.__init__()


class DecTreeClassifier(Classifier):
    
    def __init__(self, max_depth: int = 10):
        self.classifier = DecisionTreeClassifier(max_depth=max_depth)

    def __str__(self) -> str:
        return "DecisionTreeClassifier"

    def train(self, valid_embeddings: list, valid_words: list[list[str]], valid_labels: list[int]):
        self.classifier.fit(np.array(valid_embeddings), valid_labels)

    def test(self, valid_embeddings: list, valid_words: list[list[str]]) -> ndarray:
        return self.classifier.predict(np.array(valid_embeddings))

    def is_trained(self) -> bool:
        return hasattr(self.classifier, 'tree_') and self.classifier.tree_ is not None

    def clear(self) -> None:
        self.__init__()


class MultiClassifierIsoForestLocalOutlier(MultiClassifier):
    def __init__(self):
        super().__init__([IsolationForestClassifier(), LocalOutlierFactorClassifier()])

class MultiClassifierIsoForestOpticsCluster(MultiClassifier):
    def __init__(self):
        super().__init__([IsolationForestClassifier(), OpticsClusterClassifier()])
