from fastapi import FastAPI
from gensim import models
from sklearn.ensemble import IsolationForest

DATABASES = {
    "WindowsLog":"../anaconda/notebooks/NLP/big-data/WindowsTop10000.log",
}

EMBEDDINGS = {
    "FastText":"../anaconda/notebooks/NLP/big-data/crawl-300d-2M-subword.bin",
}


app = FastAPI()
embedding = None
classifier = None

@app.get("/setup")
def setup(database, start, end):
    global embedding
    global classifier
    embedding = models.fasttext.load_facebook_vectors(EMBEDDINGS)
    training_embeddings,_ = read_range_of_embeddings(database, start, end)
    classifier = IsolationForest(n_estimators=100, warm_start=True)
    classifier.fit(training_embeddings)
    return {"status": "Setup complete"}

@app.get("/test")
def test(database, start, end):
    testing_embeddings,lines = read_range_of_embeddings(database, start, end)
    prediction = classifier.predict(testing_embeddings)
    return result_output(prediction, lines)

def result_output(predictions, lines):
    output = []
    for i, line in enumerate(lines):
        if predictions[i] == -1:
            output.append({"line": line, "anomaly": True})
        else:
            output.append({"line": line, "anomaly": False})
    return output

def read_range_of_embeddings(database, start, end):
    import zipfile
    embeddings = []
    lines = []
    with zipfile.ZipFile(DATABASES[database]) as z:
        with z.open(...) as f:
            for _ in range(end):
                line = f.readline().decode('utf-8')
                if _ < start:
                    continue
                else:
                    lines.append(line)
                    words = split_log(line)
                    embedding_vector = embedding[words]
                    embeddings.append(embedding_vector)
    return embeddings, lines

def split_log(log: str):
    return log.split()