# pylint: disable=W1514,W0603,W1203,C0114,C0116,C0115
"""
Analysis functions for log anomaly detection using FastText and visualization tools.
"""

import ast
import json
import re
from io import StringIO

import numpy as np
from embedder import FastTextEmbedder
from matplotlib import pyplot as plt
from Levenshtein import distance as levenshtein_distance

def compute_distances(i: int, dictionary: list[str], distances: np.ndarray, n: int, fasttext: FastTextEmbedder) -> None:
    """
    Compute pairwise distances between dictionary entries using FastText and fill the symmetric distance matrix.
    """
    for j in range(i, n):
        dist = fasttext.similarity(dictionary[i], dictionary[j])
        distances[i, j] = dist
        distances[j, i] = dist  # Symmetric

def compute_levenshtein_distances(i: int, dictionary: list[str], distances: np.ndarray, n: int) -> None:
    """
    Compute pairwise Levenshtein distances between dictionary entries and fill the symmetric distance matrix.
    """
    for j in range(i, n):
        dist = levenshtein_distance(dictionary[i], dictionary[j])
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

def extract_database_from_json(file: str) -> str:
    """
    Extract lines, indices, and anomaly predictions from a JSON file.
    """
    with open(file, "r") as f:
        data = json.load(f)
    match data["database"]:
        case "WindowsLogSource": return "WindowsLog"
        case "WebServerLogSource": return "WebServerLog"
        case _:
            raise ValueError(f"Unknown database: {data['database']}")

def extract_lines_from_json(file: str, result_index : int = 0) -> list[tuple[int, str, float]]:
    """
    Extract lines, indices, and anomaly predictions from a JSON file.
    """
    with open(file, "r") as f:
        data = json.load(f)
    lines = []
    for item in data["results"]:
        detection = create_matrix_from_string(item["anomaly_prediction"])[result_index]
        line = item["line"]
        index = item["index"]
        lines.append((index, line, detection))
    return lines

def extract_result_map_from_json(file: str, result_index : int = 0) -> dict[int, float]:
    """
    Extract lines, indices, and anomaly predictions from a JSON file.
    """
    with open(file, "r") as f:
        data = json.load(f)
    results = {}
    for item in data["results"]:
        detection = create_matrix_from_string(item["anomaly_prediction"])[result_index]
        index = item["index"]
        results[index] = detection
    return results

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

def plot_histograms_json(results: np.ndarray, file:str):
    rows =  1 if (results.ndim == 1) else results.shape[0]
    for i in range(rows):
        plt.figure(figsize=(8, 5))
        sequence = results if (results.ndim == 1) else results[i]
        counts, bins, _ = plt.hist(sequence, bins=50, color='skyblue', edgecolor='black')
        # Add counts on top of columns
        for count, bin_left, bin_right in zip(counts, bins[:-1], bins[1:]):
            if count > 0:
                plt.text((bin_left + bin_right) / 2, count, str(int(count)),
                        ha='center', va='bottom', fontsize=8, rotation=90)
        plt.title(f'Histogram of Embedding Distances {file}')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

def plot_results_json(results: np.ndarray, file: str, x_start: int = 1000, x_end: int = 1100):
    rows, cols = results.shape
    x_start_limit = min(x_start, cols)
    x_end_limit = min(x_end, cols)
    for i in range(rows):
        plt.plot(results[i, x_start_limit:x_end_limit], label=f'Sequence {i+1}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'Results from {file}')
    plt.legend()
    plt.grid(True)
    plt.show()
