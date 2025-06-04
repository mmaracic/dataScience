from io import TextIOWrapper
from fastapi import BackgroundTasks, FastAPI
from gensim import models
from sklearn.ensemble import IsolationForest
import logging


DATABASES = {
    "WindowsLog":"/mnt/d/Data/NLP/WindowsTop10000.log",
}

EMBEDDINGS = {
    "FastText":"/mnt/d/Data/NLP/crawl-300d-2M-subword.bin",
}

MODEL_NOT_FOUND = "Embedding model is not loaded."
CLASSIFIER_NOT_INITIALIZED = "Classifier is not initialized."


app = FastAPI()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

embedding = None
classifier = None
sample_max_len = 0

@app.get("/setup")
def setup_api(background_tasks: BackgroundTasks):
    background_tasks.add_task(setup)
    return {"status": "Setup started"}

@app.get("/train")
def train_api(database: str, start: int, end: int):
    train_embeddings_len = train(database, start, end)
    return {"status": f"Finished training with {train_embeddings_len} embeddings from {start} to {end} in {database}"}

@app.get("/test")
def test_api(database: str, start: int, end: int, filename: str = None):
    global sample_max_len
    global classifier
    if classifier is None:
        logger.error(CLASSIFIER_NOT_INITIALIZED)
        return {"status": CLASSIFIER_NOT_INITIALIZED}
    if embedding is None:
        logger.error(MODEL_NOT_FOUND)
        return {"status": MODEL_NOT_FOUND}
    testing_embeddings,lines = read_range_of_embeddings(database, start, end)
    if len(testing_embeddings) == 0:
        logger.error("No testing embeddings generated. Please check the database and range.")
        return {"status": "No testing embeddings generated."}
    testing_embeddings = process_embeddings(testing_embeddings, sample_max_len)
    prediction = classifier.predict(testing_embeddings)
    f = open(filename, 'w') if filename is not None else None
    return result_output(prediction, lines, f)

def setup():
    global embedding
    logger.info("Loading embeddings")
    embedding = models.fasttext.load_facebook_vectors(EMBEDDINGS["FastText"])
    logger.info("Embedding loaded successfully")

def train(database: str, start: int, end: int) -> int:
    global classifier
    global sample_max_len
    logger.info(f"Training with database: {database}, start: {start}, end: {end}")
    if embedding is None:
        logger.error(MODEL_NOT_FOUND)
        return 0
    training_embeddings,_ = read_range_of_embeddings(database, start, end)
    if len(training_embeddings) == 0:
        logger.error("No training embeddings generated. Please check the database and range.")
        return 0
    sample_max_len = len(max(training_embeddings, key=len))
    training_embeddings = process_embeddings(training_embeddings, sample_max_len)
    logger.info(f"Training embeddings loaded, count: {len(training_embeddings)}")
    classifier = IsolationForest(n_estimators=100, warm_start=True)
    logger.info("Classifier initialized")
    classifier.fit(training_embeddings)
    logger.info(f"Classifier trained successfully. Sample max length: {sample_max_len}")
    return len(training_embeddings)

def result_output(predictions, lines, f:TextIOWrapper):
    import json
    output = []
    for i, line in enumerate(lines):
        result = {"line": line, "anomaly_prediction": float(predictions[i])}
        output.append(result)
        if f is not None:
            f.write(f"{json.dumps(result)}\n")
    return output

def read_range_of_embeddings(database, start, end):
    import zipfile
    embeddings = []
    lines = []
    if zipfile.is_zipfile(DATABASES[database]):
        return read_range_of_embeddings_from_zip(database, start, end, embeddings, lines)
    else:
        return read_range_of_embeddings_from_text(database, start, end, embeddings, lines)

def read_range_of_embeddings_from_zip(database: str, start: int, end: int, embeddings: list, lines: list):
    import zipfile
    with zipfile.ZipFile(DATABASES[database]) as z:
        with z.open(...) as f:
            for _ in range(end):
                line = f.readline().decode('utf-8')
                if _ < start:
                    continue
                else:
                    embeddings, lines = process_line(line, embeddings, lines)
    return embeddings, lines

def read_range_of_embeddings_from_text(database: str, start: int, end: int, embeddings: list, lines: list):
    with open(DATABASES[database], 'r') as f:
        for _ in range(end):
            line = f.readline()
            if _ < start:
                continue
            else:
                embeddings, lines = process_line(line, embeddings, lines)
    return embeddings, lines

def process_line(line: str, embeddings: list, lines: list):
    if embedding is None:
        logger.error(MODEL_NOT_FOUND)
        return embeddings, lines
    words = split_log(line)
    if (not words) or (len(words) == 0):
        logger.warning(f"Empty line encountered: {line}")
    else:
        embedding_matrix = embedding[words]
        embedding_vector = [item for sublist in embedding_matrix for item in sublist]
        embeddings.append(embedding_vector)
        lines.append(line)
    return embeddings, lines

def process_embeddings(embeddings: list, sample_max_len: int)-> list:
    return [add_items(emb, sample_max_len) for emb in embeddings]

def add_items(l: list, target_len: int) -> list:
    if len(l) > target_len:
        logger.debug(f"Trimming list from {len(l)} to {target_len}")
        return l[:target_len]
    while len(l) < target_len:
        l.append(0)
    return l

def split_log(log: str) -> list:
    return log.split()