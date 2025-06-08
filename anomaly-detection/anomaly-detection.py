#pylint: disable=W1514,W0603,W1203,C0114,C0116,C0115
import logging
from datetime import datetime
from io import TextIOWrapper
import json

from fastapi import BackgroundTasks, FastAPI
from gensim import models
from sklearn.ensemble import IsolationForest
from database_processors import WindowsLogSource, SampleSource

DATABASES = {
    "WindowsLog": WindowsLogSource("/mnt/d/Data/NLP/WindowsTop10000.zip")
}

EMBEDDINGS = {
    "FastText":"/mnt/d/Data/NLP/crawl-300d-2M-subword.bin",
}

MODEL_NOT_FOUND = "Embedding model is not loaded."
CLASSIFIER_NOT_INITIALIZED = "Classifier is not initialized."

app = FastAPI()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dictionary = None
classifier = None
sample_max_len = 0
training_logs = []
max_training_logs = 1
logs_since_training = 0
max_logs_since_training = 1

@app.get("/setup")
def setup_api(background_tasks: BackgroundTasks, training_size: int = 1000, training_batch_size: int = 100):
    global max_training_logs
    global max_logs_since_training
    max_training_logs = training_size if training_size>0 else 1
    max_logs_since_training = training_batch_size if training_batch_size>0 else 1
    background_tasks.add_task(setup)
    return {"status": "Setup started"}

@app.get("/train")
def train_api(database: str, start: int, end: int):
    train_embeddings_len = train(database, start, end)
    return {"status": f"Finished training with {train_embeddings_len} embeddings from {start} to {end} in {database}"}

@app.get("/test")
def test_api(database: str, start: int, end: int, filename: str = None):
    if classifier is None:
        logger.error(CLASSIFIER_NOT_INITIALIZED)
        return {"status": CLASSIFIER_NOT_INITIALIZED}
    if dictionary is None:
        logger.error(MODEL_NOT_FOUND)
        return {"status": MODEL_NOT_FOUND}
    testing_samples:list[tuple] = []
    for line in DATABASES[database].read_range_of_embeddings(start, end):
        embedding, line = process_line(database, line)
        testing_samples.append(tuple(embedding, line))
    valid_testing_samples = [emb for emb in testing_samples if emb[0] is not None]
    if len(valid_testing_samples) == 0:
        logger.error("No testing embeddings generated. Please check the database and range.")
        return {"status": "No testing embeddings generated."}
    logger.info(f"Testing samples created, {len(valid_testing_samples)} valid samples  out of {len(testing_samples)} logs")
    valid_testing_embeddings = [smp[0] for smp in testing_samples]
    valid_testing_logs = [smp[1] for smp in testing_samples]
    processed_valid_testing_embeddings = process_embeddings(valid_testing_embeddings)
    prediction = classifier.predict(processed_valid_testing_embeddings)
    f = open(filename, 'w') if filename is not None else None
    interpretation = result_output(database, start, prediction, valid_testing_logs, f)
    if f is not None:
        f.close()
    return interpretation

def stream_samples(sample: str):
    if dictionary is None:
        logger.error(MODEL_NOT_FOUND)
        return {"status": MODEL_NOT_FOUND}
    if classifier is None:
        logger.error(CLASSIFIER_NOT_INITIALIZED)
        return {"status": CLASSIFIER_NOT_INITIALIZED}

def setup():
    global dictionary
    logger.info("Loading embeddings")
    dictionary = models.fasttext.load_facebook_vectors(EMBEDDINGS["FastText"])
    logger.info("Embedding loaded successfully")

def train(database: str, start: int, end: int) -> int:
    global classifier
    global sample_max_len
    logger.info(f"Training with database: {database}, start: {start}, end: {end}")
    if dictionary is None:
        logger.error(MODEL_NOT_FOUND)
        return 0
    training_embeddings = []
    for line in DATABASES[database].read_range_of_embeddings(start, end):
        embedding, line = process_line(database, line)
        training_logs.append(line)
        training_embeddings.append(embedding)
    valid_training_embeddings = [emb for emb in training_embeddings if emb is not None]
    if len(valid_training_embeddings) == 0:
        logger.error("No embeddings generated. Please check the database and range.")
        return 0
    sample_max_len = len(max(valid_training_embeddings, key=len))
    processed_valid_training_embeddings = process_embeddings(valid_training_embeddings)
    logger.info(f"Training embeddings created, {len(processed_valid_training_embeddings)} embeddings out of {len(training_logs)} logs")
    classifier = IsolationForest(n_estimators=100, warm_start=True)
    logger.info("Classifier initialized")
    classifier.fit(processed_valid_training_embeddings)
    logger.info(f"Classifier trained successfully. Sample max length: {sample_max_len}")
    return len(processed_valid_training_embeddings)

def result_output(database: str, start_index: int, predictions, lines, f:TextIOWrapper):
    output = {}
    output["database"] = database
    output["timestamp"] = datetime.now().isoformat()
    results = []
    for i, line in enumerate(lines):
        result = {"index": start_index+i, "line": line, "anomaly_prediction": float(predictions[i])}
        results.append(result)
    output["results"] = results
    if f is not None:
        f.write(json.dumps(output))
    return results

def process_line(database: SampleSource, line: str):
    if dictionary is None:
        logger.error(MODEL_NOT_FOUND)
        raise ValueError(MODEL_NOT_FOUND)
    words = database.split_log(line)
    if (not words) or (len(words) == 0):
        logger.warning(f"Empty line encountered: {line}")
        return None, line
    else:
        embedding_matrix = dictionary[words]
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

def accumulate_log(log: str):
    global logs_since_training
    logs_since_training = logs_since_training + 1
    training_logs.append(log)
    if len(training_logs) >= max_training_logs:
        training_logs.pop(0)

    if logs_since_training >= max_logs_since_training:
        
        logs_since_training = 0
