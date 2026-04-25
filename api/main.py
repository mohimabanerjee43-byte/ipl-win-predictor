from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import numpy as np
import os

from fastapi.middleware.cors import CORSMiddleware

# -------------------- APP INIT --------------------
app = FastAPI()

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- LOAD MODEL --------------------
model_path = os.path.join("models", "model.pkl")

try:
    model = pickle.load(open(model_path, "rb"))
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# -------------------- INPUT SCHEMA --------------------
class InputData(BaseModel):
    current_score: float
    balls_left: float
    wickets_left: float
    run_rate: float
    required_run_rate: float

# -------------------- ROOT --------------------
@app.get("/")
def home():
    return {"message": "IPL Win Predictor API is running 🚀"}

# -------------------- PREDICT --------------------
@app.post("/predict")
def predict(data: InputData):
    try:
        input_df = pd.DataFrame([data.dict()])
        input_array = input_df.values

        proba = model.predict_proba(input_array)

        win_prob = float(proba[0][1])

        return {
            "prediction": win_prob
        }

    except Exception as e:
        return {"error": str(e)
        }