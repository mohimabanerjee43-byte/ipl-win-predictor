from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# -------------------- APP INIT --------------------
app = FastAPI()

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        # -------- VALIDATION --------
        if (
            data.balls_left <= 0 or
            data.wickets_left < 0 or
            data.run_rate < 0 or
            data.required_run_rate < 0
        ):
            return {"error": "Invalid input values"}

        # -------- CALCULATE PRESSURE --------
        pressure = data.required_run_rate - data.run_rate

        # -------- LOGISTIC FUNCTION --------
        logit = (-0.9 * pressure) + (0.35 * data.wickets_left) + (0.01 * data.balls_left)

        win_prob = 1 / (1 + np.exp(-logit))

        # -------- CLAMP (extra safety) --------
        win_prob = max(0.0, min(1.0, float(win_prob)))

        print("📊 Prediction:", win_prob)

        return {
            "prediction": win_prob
        }

    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}