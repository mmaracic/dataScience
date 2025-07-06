# pylint: disable=W1514,W0603,W1203,C0114,C0116,C0115
"""
API and streaming logic for log anomaly detection using FastText embeddings and custom classifiers.
"""

import logging
import os
import json
from datetime import datetime
from typing import Optional, cast
import threading

import numpy as np
from numpy import ndarray
from fastapi import BackgroundTasks, FastAPI, Request

from database_processors import WindowsLogSource, WebServerLogSource, SampleSource
from classifier import Classifier, get_classifier
from embedder import Embedder, FastTextEmbedder

# Database and embedding model paths
DATABASES: dict[str, SampleSource] = {
    "WindowsLog": WindowsLogSource("/mnt/c/Projects/log-anomaly-detection/FastText/data/WindowsTop10000.log"),
    "WebServerLog": WebServerLogSource("/mnt/c/Projects/log-anomaly-detection/FastText/data/WebServerTop10000.log"),
}

EMBEDDINGS: dict[str, str] = {
    "FastText": "/mnt/c/Projects/log-anomaly-detection/FastText/data/crawl-300d-2M-subword.bin",
}

# Error/status messages
DATABASE_NOT_INITIALIZED = "Database is not initialized. Please set up the database first."
MODEL_NOT_FOUND = "Embedding model is not loaded."
CLASSIFIER_NOT_INITIALIZED = "Classifier is not initialized."
CLASSIFIER_NOT_TRAINED = "Classifier is not trained."

# FastAPI app and logging setup
app = FastAPI()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.database: Optional[SampleSource] = None
        self.embedder: Optional[Embedder] = None
        self.classifier: Optional[Classifier] = None
        self.sample_max_len: int = 0
        self.num_features: int = 0  # Number of features to select based on kurtosis 
        self.selected_features: list[int] = [] # Indices of selected features based on kurtosis
        self.training_data: list[tuple[list[str], ndarray|None]] = []
        self.training_logs: list[str] = []
        self.training_labels: list[int] = []
        self.max_training_logs: int = 1
        self.logs_since_training: int = 0
        self.max_logs_since_training: int = 1

app.state.app_state = AppState()

@app.get("/setup")
def setup_api(background_tasks: BackgroundTasks, request: Request, database_name: str, classifier_name: str,
              training_size: int = 1000, training_batch_size: int = 500, selected_features: int = 1000):
    """
    Set up the database, classifier, and training parameters.
    """
    state = request.app.state.app_state
    with state.lock:
        state.max_training_logs = training_size if training_size > 0 else 1
        state.max_logs_since_training = training_batch_size if training_batch_size > 0 else 1
        state.num_features = selected_features if selected_features > 0 else 1000
        state.database = DATABASES[database_name]
        state.classifier = get_classifier(classifier_name)
    logger.info("Classifier initialized")
    background_tasks.add_task(lambda: setup(request))
    return {"status": "Setup started"}

@app.get("/train")
def train_api(request: Request, start: int, end: int):
    """
    Train the classifier on a range of logs from the database.
    """
    state = request.app.state.app_state
    with state.lock:
        train_samples(start, end, state)
        train_model(state)
        logs_count = len(state.training_logs)
        db_str = str(state.database)
    return {"status": f"Finished training with {logs_count} logs from {start} to {end} in {db_str}"}

@app.get("/test")
def test_api(request: Request, start: int, end: int):
    """
    Test the classifier on a range of logs from the database.
    Returns predictions and metadata as JSON.
    """ 
    state = request.app.state.app_state
    with state.lock:
        status = check_system_ready(state, require_trained=True)
        if status is not None:
            return status
        testing_data, testing_logs = collect_testing_data(start, end, state)
        if not testing_data:
            return {"status": "No testing embeddings generated."}
        prediction, _, metadata = test_samples(testing_data, state)
        interpretation = result_json_output(start, end, prediction, testing_logs, metadata, state)
    return interpretation

@app.get("/test_stream")
def test_stream_api(request: Request):
    state: AppState = request.app.state.app_state
    with state.lock:
        if state.database is None:
            logger.error(DATABASE_NOT_INITIALIZED)
            return {"status": DATABASE_NOT_INITIALIZED}
        if state.embedder is None:
            logger.error(MODEL_NOT_FOUND)
            return {"status": MODEL_NOT_FOUND}
        if state.classifier is None:
            logger.error(CLASSIFIER_NOT_INITIALIZED)
            return {"status": CLASSIFIER_NOT_INITIALIZED}
        time = datetime.now()
        state.training_logs.clear()
        state.training_data.clear()
        state.classifier.clear()
        state.logs_since_training = 0
        logger.info("Starting streaming samples from database")
        index = 0
        for log in state.database.read_log():
            stream_sample(time, index, log, state)
            index += 1
        return {"status": f"Processed {index-1} logs from the database in streaming mode."}

def check_system_ready(state: AppState, require_trained: bool = False):
    """
    Helper to check if database, classifier, and embedder are initialized.
    Optionally checks if the classifier is trained.
    """
    if state.database is None:
        logger.error(DATABASE_NOT_INITIALIZED)
        return {"status": DATABASE_NOT_INITIALIZED}
    if state.classifier is None:
        logger.error(CLASSIFIER_NOT_INITIALIZED)
        return {"status": CLASSIFIER_NOT_INITIALIZED}
    if require_trained and not state.classifier.is_trained():
        logger.error(CLASSIFIER_NOT_TRAINED)
        return {"status": CLASSIFIER_NOT_TRAINED}
    if state.embedder is None:
        logger.error(MODEL_NOT_FOUND)
        return {"status": MODEL_NOT_FOUND}
    return None


def collect_testing_data(start: int, end: int, state: AppState) -> tuple[list[tuple[ndarray, list[str]]], list[str]]:
    """
    Helper to collect embeddings and tokens for testing logs.
    """
    testing_data = []
    testing_logs = []
    if state.database is not None:
        for line in state.database.read_range_of_logs(start, end):
            embedding, tokens = process_line(line, state)
            if embedding is not None:
                testing_data.append((embedding, tokens))
                testing_logs.append(line)
    return testing_data, testing_logs

def setup(request: Request):
    state: AppState = request.app.state.app_state
    logger.info("Loading embeddings")
    with state.lock:
        if state.embedder is None:
            state.embedder = FastTextEmbedder(EMBEDDINGS["FastText"])
            logger.info("Embedding loaded successfully")
        else:
            logger.info("Embedding already loaded")


def train_samples(start: int, end: int, state: AppState) -> int:
    logger.info(f"Training with database: {state.database}, start: {start}, end: {end}")
    if state.database is None:
        logger.error(DATABASE_NOT_INITIALIZED)
        return 0
    if state.embedder is None:
        logger.error(MODEL_NOT_FOUND)
        return 0
    state.training_logs.clear()
    state.training_data.clear()
    state.logs_since_training = 0
    for line in state.database.read_range_of_logs(start, end):
        embedding, tokens = process_line(line, state)
        state.training_logs.append(line)
        state.training_data.append((tokens, embedding))
    logger.info(f"Training logs collected, {len(state.training_logs)} logs")
    return len(state.training_logs)

def train_model(state:AppState) -> int:
    valid_training_embeddings = [tuple[1] for tuple in state.training_data if tuple[1] is not None]
    valid_training_tokens = [tuple[0] for tuple in state.training_data if tuple[1] is not None]
    if len(valid_training_embeddings) == 0:
        logger.error("No embeddings generated. Please check the database and range.")
        return 0
    state.sample_max_len = len(max(valid_training_embeddings, key=len))
    processed_valid_training_embeddings = process_embeddings(valid_training_embeddings, state)
    logger.info(f"Training embeddings created, {len(processed_valid_training_embeddings)} embeddings out of {len(state.training_logs)} logs")
    if state.classifier is None:
        logger.error(CLASSIFIER_NOT_INITIALIZED)
        return 0
    state.selected_features = select_features(processed_valid_training_embeddings, state)
    processed_valid_training_embeddings = filter_features(processed_valid_training_embeddings, state)
    state.classifier.train(processed_valid_training_embeddings, valid_training_tokens, state.training_labels)
    logger.info(f"Classifier trained successfully. Sample max length: {state.sample_max_len}")
    return len(processed_valid_training_embeddings)

def test_samples(testing_samples: list[tuple[ndarray, list[str]]], state:AppState) -> tuple[ndarray, list[list[str]], list[tuple[float, float]]]:
    valid_testing_samples = [smp for smp in testing_samples if smp[0] is not None]
    if len(valid_testing_samples) == 0:
        logger.error("No testing embeddings generated. Please check the database and range.")
        return np.array([]), [], []
    logger.info(f"Testing samples created, {len(valid_testing_samples)} valid samples  out of {len(testing_samples)} logs")
    valid_testing_embeddings = [smp[0] for smp in testing_samples]
    valid_testing_tokens = [smp[1] for smp in testing_samples]
    processed_valid_testing_embeddings = process_embeddings(valid_testing_embeddings, state)
    processed_valid_testing_embeddings = filter_features(processed_valid_testing_embeddings, state)
    valid_metadata = generate_test_metadata(processed_valid_testing_embeddings, state)
    logger.info("Testing metadata created")
    if state.classifier is None:
        logger.error(CLASSIFIER_NOT_INITIALIZED)
        return np.array([]), [], []
    prediction = state.classifier.test(processed_valid_testing_embeddings, valid_testing_tokens)
    return prediction, valid_testing_tokens, valid_metadata

def generate_test_metadata(valid_embeddings: list[ndarray], state:AppState) -> list[tuple[float, float]]:
    metadata = []
    processed_training_data = process_embeddings([emb for _, emb in state.training_data if emb is not None], state)
    for embedding in valid_embeddings:
        distances = [np.linalg.norm(item[1] - embedding) for item in processed_training_data if item[1] is not None]
        metadata.append((np.mean(distances), np.std(distances)))
    return metadata

def stream_sample(time: datetime, index: int, sample: str, state:AppState) -> ndarray:
    embedding, tokens = accumulate_log_and_train_model(sample, state)
    prediction_correction = np.zeros((1, len(cast(Classifier, state.classifier))))
    metadata = []
    if cast(Classifier, state.classifier).is_trained() is True:
        prediction, _, metadata = test_samples([(embedding, tokens)], state)
        prediction_correction = prediction if prediction.shape[0] > 0 else np.zeros((1, len(cast(Classifier, state.classifier))))
    result_txt_output(time, 0, index, prediction_correction, metadata, state)
    return prediction_correction


def get_filename(start_index: int, end_index: int, time: str, extension: str, state:AppState) -> str:
    return f"./output/results_{str(state.database)}_{str(state.classifier)}_{start_index}_{end_index}_{time}.{extension}"

def result_json_output(start_index: int, end_index: int, predictions: ndarray, lines: list[str], metadata: list[tuple[float, float]], state:AppState) -> list:
    time = datetime.now().isoformat()
    output = {}
    output["database"] = str(state.database)
    output["timestamp"] = time
    output["start_index"] = start_index
    output["end_index"] = end_index
    output["sample_max_len"] = state.sample_max_len
    transposed_predictions = predictions.transpose()
    results = []
    for i, line in enumerate(lines):
        result = {"index": start_index+i, "line": line, "anomaly_prediction": str(transposed_predictions[i]), "metadata": str(metadata[i])}
        results.append(result)
    output["results"] = results
    with open(get_filename(start_index, end_index, time, "json", state), "w") as f:
        f.write(json.dumps(output))
    return results

def result_txt_output(time: datetime, start_index: int, index: int, prediction: ndarray, metadata: list[tuple[float, float]], state:AppState) -> None:
    time_str = time.isoformat()
    end_index = 0
    filename = get_filename(start_index, end_index, time_str, "txt", state)
    if os.path.exists(filename):
        with open(filename, "a") as f:
            write_row(f, index, prediction, metadata)
    else:
        with open(filename, "w") as f:
            f.write(f"{str(state.database)}\n{time_str}\n{start_index}\n{end_index}\n{state.sample_max_len}\n")
            f.write("#Data#\n")
            write_row(f, index, prediction, metadata)

def write_row(f, index: int, prediction: ndarray, metadata: list[tuple[float, float]]) -> None:
    prediction_str = np.array2string(prediction.flatten(), separator=' ')[1:-1]
    f.write(f"{index}#{prediction_str}#{metadata}\n")

def process_line(line: str, state: AppState) -> tuple[Optional[ndarray], list[str]]:
    if state.embedder is None:
        logger.error(MODEL_NOT_FOUND)
        raise ValueError(MODEL_NOT_FOUND)
    tokens = cast(SampleSource, state.database).split_log(line)
    if (not tokens) or (len(tokens) == 0):
        logger.warning(f"Empty line encountered: {line}")
        return None, tokens
    else:
        embedding_matrix = state.embedder.embed(tokens)
        embedding_vector = np.concatenate(embedding_matrix, axis=0) if embedding_matrix is not None else None
    return embedding_vector, tokens

def process_embeddings(embeddings: list[ndarray], state:AppState) -> list[ndarray]:
    return [add_components(emb, state.sample_max_len) for emb in embeddings]

def add_components(l: ndarray, target_len: int) -> ndarray:
    if len(l) > target_len:
        logger.debug(f"Trimming list from {len(l)} to {target_len}")
        return l[:target_len]
    elif len(l) < target_len:
        extended = np.zeros((target_len), dtype=l.dtype)
        extended[:len(l)] = l
        return extended
    return l

def accumulate_log_and_train_model(log: str, state:AppState) -> tuple[ndarray, list[str]]:
    state.logs_since_training += 1
    embedding, tokens = process_line(log, state)
    if embedding is None:
        logger.warning(f"Skipping log due to empty embedding: {log}")
        return np.array([]), tokens
    state.training_logs.append(log)
    state.training_data.append((tokens, embedding))
    if len(state.training_logs) >= state.max_training_logs:
        state.training_logs.pop(0)
        state.training_data.pop(0)

    if state.logs_since_training >= state.max_logs_since_training:
        train_model(state)
        state.logs_since_training = 0
    return embedding, tokens

def select_features(embeddings: list[ndarray], state: AppState) -> list[int]:
    kurtosis = np.mean((embeddings - np.mean(embeddings, axis=0))**4, axis=0) / np.var(embeddings, axis=0)**2
    kurtosis = kurtosis.tolist()
    kurtosis_tuples = [(i, round(k, 2)) for i, k in enumerate(kurtosis)]
    kurtosis_tuples = sorted(kurtosis_tuples, key=lambda x: x[1], reverse=True)
    print(f"Will select {state.num_features} features out of {len(kurtosis_tuples)}: {kurtosis_tuples[0][1]}-{kurtosis_tuples[state.num_features][1]}")
    return [i for i, _ in kurtosis_tuples[:state.num_features]]

def filter_features(embeddings: list[ndarray], state: AppState) -> list[ndarray]:
    selected_features = state.selected_features
    if selected_features:
        filtered_embeddings = [embeddings[selected_features] for embeddings in embeddings]
        return filtered_embeddings

    return embeddings
