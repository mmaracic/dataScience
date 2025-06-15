#pylint: disable=W1514,W0603,W1203,C0114,C0116,C0115
import logging
from datetime import datetime
import json
import os
from typing import Optional

from fastapi import BackgroundTasks, FastAPI
from numpy import ndarray
from database_processors import WindowsLogSource, SampleSource
from classifier import Classifier, IsolationForestClassifier
from embedder import Embedder, FastTextEmbedder

DATABASES = {
    "WindowsLog": WindowsLogSource("/mnt/d/Data/NLP/WindowsTop10000.zip")
}

EMBEDDINGS = {
    "FastText":"/mnt/d/Data/NLP/crawl-300d-2M-subword.bin",
}

DATABASE_NOT_INITIALIZED = "Database is not initialized. Please set up the database first."
MODEL_NOT_FOUND = "Embedding model is not loaded."
CLASSIFIER_NOT_INITIALIZED = "Classifier is not initialized."
CLASSIFIER_NOT_TRAINED = "Classifier is not trained."

app = FastAPI()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
database: Optional[SampleSource] = None
embedder: Optional[Embedder] = None
classifier: Optional[Classifier] = None
sample_max_len = 0
training_logs = []
training_embeddings = []
max_training_logs = 1
logs_since_training = 0
max_logs_since_training = 1

@app.get("/setup")
def setup_api(background_tasks: BackgroundTasks, database_name: str, training_size: int = 1000, training_batch_size: int = 100):
    global max_training_logs
    global max_logs_since_training
    global database
    global classifier
    max_training_logs = training_size if training_size>0 else 1
    max_logs_since_training = training_batch_size if training_batch_size>0 else 1
    database = DATABASES[database_name]
    classifier = IsolationForestClassifier()
    logger.info("Classifier initialized")
    background_tasks.add_task(setup)
    return {"status": "Setup started"}

@app.get("/train")
def train_api(start: int, end: int):
    train_samples(start, end)
    train_model()
    return {"status": f"Finished training with {len(training_logs)} logs from {start} to {end} in {database}"}

@app.get("/test")
def test_api(start: int, end: int):
    if database is None:
        logger.error(DATABASE_NOT_INITIALIZED)
        return {"status": DATABASE_NOT_INITIALIZED}
    if classifier is None:
        logger.error(CLASSIFIER_NOT_INITIALIZED)
        return {"status": CLASSIFIER_NOT_INITIALIZED}
    if not classifier.is_trained():
        logger.error(CLASSIFIER_NOT_TRAINED)
        return {"status": CLASSIFIER_NOT_TRAINED}
    if embedder is None:
        logger.error(MODEL_NOT_FOUND)
        return {"status": MODEL_NOT_FOUND}
    testing_samples:list[tuple] = []
    for line in database.read_range_of_embeddings(start, end):
        embedding, line = process_line(line)
        testing_samples.append(tuple(embedding, line))
    prediction, valid_testing_logs = test_samples(testing_samples)
    if len(valid_testing_logs) == 0:
        return {"status": "No testing embeddings generated."}
    interpretation = result_json_output(start, end, prediction, valid_testing_logs)
    return interpretation

@app.get("/test_stream")
def test_stream_api():
    if database is None:
        logger.error(DATABASE_NOT_INITIALIZED)
        return {"status": DATABASE_NOT_INITIALIZED}
    if embedder is None:
        logger.error(MODEL_NOT_FOUND)
        return {"status": MODEL_NOT_FOUND}
    if classifier is None:
        logger.error(CLASSIFIER_NOT_INITIALIZED)
        return {"status": CLASSIFIER_NOT_INITIALIZED}
    if not classifier.is_trained():
        logger.error(CLASSIFIER_NOT_TRAINED)
        return {"status": CLASSIFIER_NOT_TRAINED}
    log = database.read_log()
    stream_sample(log)

def setup():
    global embedder
    logger.info("Loading embeddings")
    embedder = FastTextEmbedder(EMBEDDINGS["FastText"])
    logger.info("Embedding loaded successfully")

def train_samples(start: int, end: int):
    logger.info(f"Training with database: {database}, start: {start}, end: {end}")
    if database is None:
        logger.error(DATABASE_NOT_INITIALIZED)
        return 0
    if embedder is None:
        logger.error(MODEL_NOT_FOUND)
        return 0
    training_logs.clear()
    training_embeddings.clear()
    global logs_since_training
    logs_since_training = 0
    for line in database.read_range_of_embeddings(start, end):
        embedding, line = process_line(line)
        training_logs.append(line)
        training_embeddings.append(embedding)
    logger.info(f"Training logs collected, {len(training_logs)} logs")

def train_model():
    global sample_max_len
    valid_training_embeddings = [emb for emb in training_embeddings if emb is not None]
    if len(valid_training_embeddings) == 0:
        logger.error("No embeddings generated. Please check the database and range.")
        return 0
    sample_max_len = len(max(valid_training_embeddings, key=len))
    processed_valid_training_embeddings = process_embeddings(valid_training_embeddings)
    logger.info(f"Training embeddings created, {len(processed_valid_training_embeddings)} embeddings out of {len(training_logs)} logs")
    classifier.train(processed_valid_training_embeddings)
    logger.info(f"Classifier trained successfully. Sample max length: {sample_max_len}")
    return len(processed_valid_training_embeddings)

def test_samples(testing_samples: list[tuple]) -> tuple[ndarray, list]:
    valid_testing_samples = [emb for emb in testing_samples if emb[0] is not None]
    if len(valid_testing_samples) == 0:
        logger.error("No testing embeddings generated. Please check the database and range.")
        return tuple([], [])
    logger.info(f"Testing samples created, {len(valid_testing_samples)} valid samples  out of {len(testing_samples)} logs")
    valid_testing_embeddings = [smp[0] for smp in testing_samples]
    valid_testing_logs = [smp[1] for smp in testing_samples]
    processed_valid_testing_embeddings = process_embeddings(valid_testing_embeddings)
    prediction = classifier.test(processed_valid_testing_embeddings)
    return prediction, valid_testing_logs

def stream_sample(sample: str):
    embedding = accumulate_log_and_train_model(sample)
    prediction, _ = test_samples([(embedding, sample)])
    prediction_correction = prediction[0] if len(prediction) > 0 else 0
    result_txt_output(0, 0, prediction_correction)
    return prediction_correction


def get_filename(start_index: int, end_index: int, time: str, extension: str) -> str:
    return f"results_{start_index}_{end_index}_{time}.{extension}"

def result_json_output(start_index: int, end_index: int, predictions, lines) -> list:
    time = datetime.now().isoformat()
    output = {}
    output["database"] = database
    output["timestamp"] = time
    output["start_index"] = start_index
    output["end_index"] = end_index
    output["sample_max_len"] = sample_max_len
    results = []
    for i, line in enumerate(lines):
        result = {"index": start_index+i, "line": line, "anomaly_prediction": float(predictions[i])}
        results.append(result)
    output["results"] = results
    with open(get_filename(start_index, end_index, time, "json"), "w") as f:
        f.write(json.dumps(output))
    return results

def result_txt_output(start_index: int, index: int, prediction: float):
    time = datetime.now().isoformat()
    end_index = 0
    filename = get_filename(start_index, end_index, time, "txt")
    if os.path.exists(filename):
        with open(filename, "a") as f:
            f.write(f"{index}\t{prediction}\n")
    else:
        with open(filename, "w") as f:
            f.write(f"{database}\n{time}\n{start_index}\n{end_index}\n{sample_max_len}\n")
            f.write(f"{index}\t{prediction}\n")

def process_line(line: str):
    if embedder is None:
        logger.error(MODEL_NOT_FOUND)
        raise ValueError(MODEL_NOT_FOUND)
    words = database.split_log(line)
    if (not words) or (len(words) == 0):
        logger.warning(f"Empty line encountered: {line}")
        return None, line
    else:
        embedding_matrix = embedder.embed(words)
        embedding_vector = [item for sublist in embedding_matrix for item in sublist]
    return embedding_vector, line

def process_embeddings(embeddings: list)-> list:
    return [add_components(emb, sample_max_len) for emb in embeddings]

def add_components(l: list, target_len: int) -> list:
    if len(l) > target_len:
        logger.debug(f"Trimming list from {len(l)} to {target_len}")
        return l[:target_len]
    while len(l) < target_len:
        l.append(0)
    return l

def accumulate_log_and_train_model(log: str) -> list:
    global logs_since_training
    logs_since_training = logs_since_training + 1
    embedding, _ = process_line(log)
    if embedding is None:
        logger.warning(f"Skipping log due to empty embedding: {log}")
        return []
    training_logs.append(log)
    training_embeddings.append(embedding)
    if len(training_logs) >= max_training_logs:
        training_logs.pop(0)
        training_embeddings.pop(0)

    if logs_since_training >= max_logs_since_training:
        train_model()
        logs_since_training = 0
    return embedding
