# pylint: disable=W1514,W0603,W1203,C0114,C0116,C0115
"""
Analysis functions for log anomaly detection using FastText and visualization tools.
"""

import ast
import json
import re
from io import StringIO

import numpy as np
from gensim.models.fasttext import FastTextKeyedVectors
from matplotlib import pyplot as plt

def compute_distances(i: int, dictionary: list[str], distances: np.ndarray, n: int, fasttext: FastTextKeyedVectors) -> None:
    """
    Compute pairwise distances between dictionary entries using FastText and fill the symmetric distance matrix.
    """
    for j in range(i, n):
        dist = fasttext.distance(dictionary[i], dictionary[j])
        distances[i, j] = dist
        distances[j, i] = dist  # Symmetric

def create_matrix_from_string(s: str) -> np.ndarray:
    """
    Convert a string representation of a matrix into a numpy array.
    """
    return np.loadtxt(StringIO(re.sub(r'[\[\]]', '', s)))

def extract_matrix_from_json(file: str) -> np.ndarray:
    """
    Extract anomaly prediction matrices from a JSON file and return as a numpy array.
    """
    with open(file, "r") as f:
        data = json.load(f)
    detections = [create_matrix_from_string(item["anomaly_prediction"]) for item in data["results"]]
    matrix = np.zeros((len(detections), len(detections[0])))
    for i, detection in enumerate(detections):
        matrix[i, :] = detection
    return matrix

def extract_metadata_from_json(file: str) -> list[tuple[float, float]]:
    """
    Extract metadata tuples from a JSON file.
    """
    with open(file, "r") as f:
        data = json.load(f)
    return [ast.literal_eval(item["metadata"]) for item in data["results"]]

def plot_metadata(data: list[tuple[tuple[float, float], float]]) -> None:
    """
    Plot 2D metadata points, coloring by outlier/normal status.
    """
    x_normal = [item[0][0] for item in data if item[1] == 1.0]
    y_normal = [item[0][1] for item in data if item[1] == 1.0]

    x_outlier = [item[0][0] for item in data if item[1] == -1.0]
    y_outlier = [item[0][1] for item in data if item[1] == -1.0]

    plt.scatter(x_normal, y_normal, color='blue', label='Normal', alpha=0.5)
    plt.scatter(x_outlier, y_outlier, color='red', label='Outlier', alpha=0.5)
    plt.legend()
    plt.xlabel('Mean')
    plt.ylabel('Std deviation')
    plt.title('Metadata as 2D Points separated by Outlier Detection')
    plt.grid(True)
    plt.show()
