import pickle
from src.config import MODEL_PATH

def save_model(model):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)