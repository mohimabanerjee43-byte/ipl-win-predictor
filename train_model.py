import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
rows = 3000

# -------- BASE DATA --------
df = pd.DataFrame({
    "target": np.random.randint(120, 200, rows),
    "balls_left": np.random.randint(1, 120, rows),
    "wickets_left": np.random.randint(1, 10, rows),
})

# simulate current score based on target progress
df["current_score"] = df["target"] - (
    np.random.randint(0, 120, rows)
)

# -------- FEATURES --------
df["run_rate"] = df["current_score"] / (120 - df["balls_left"] + 1)
df["required_run_rate"] = (
    (df["target"] - df["current_score"]) * 6 / df["balls_left"]
)

# -------- REALISTIC PROBABILITY --------
# pressure = required RR - current RR
pressure = df["required_run_rate"] - df["run_rate"]

prob = (
    -1.2 * pressure +           # high pressure → bad
    0.3 * df["wickets_left"]   # more wickets → good
)

# sigmoid for smooth probability
win_prob = 1 / (1 + np.exp(-prob))

# create label
df["result"] = (win_prob > 0.5).astype(int)

# -------- TRAIN --------
X = df[[
    "current_score",
    "balls_left",
    "wickets_left",
    "run_rate",
    "required_run_rate"
]]

y = df["result"]

model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

# -------- SAVE --------
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Stable model trained successfully")