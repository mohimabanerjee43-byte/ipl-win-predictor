import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("ipl_data.csv")

# Example columns expected:
# ['current_score', 'balls_left', 'wickets_left',
#  'run_rate', 'required_run_rate', 'result']

# -----------------------------
# 2. Features & Target
# -----------------------------
X = df[['current_score', 'balls_left', 'wickets_left', 'run_rate', 'required_run_rate']]
y = df['result']   # 1 = win, 0 = lose

# -----------------------------
# 3. Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Save model
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained & saved successfully ✅")