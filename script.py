"""
Churn prediction on an imbalanced dataset.
Produces:
  - data/X_train.csv, data/X_test.csv, data/y_train.csv, data/y_test.csv
  - models/model.pkl
  - metrics.txt
  - conf_matrix.png
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE

# ── 1. Generate a synthetic imbalanced churn dataset ────────────────────────
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    weights=[0.9, 0.1],  # 90 % non-churn, 10 % churn → imbalanced
    flip_y=0.02,
    random_state=42,
)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["churn"] = y

# ── 2. Train / test split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names], df["churn"], test_size=0.2, random_state=42, stratify=df["churn"]
)

# ── 3. Handle imbalance with SMOTE ──────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ── 4. Save prepared data ──────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# ── 5. Train model ─────────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# ── 6. Save model ──────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

# ── 7. Evaluate ────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ── 8. Write metrics.txt ───────────────────────────────────────────────────
with open("metrics.txt", "w") as f:
    f.write(f"accuracy  {accuracy:.4f}\n")
    f.write(f"precision {precision:.4f}\n")
    f.write(f"recall    {recall:.4f}\n")
    f.write(f"f1_score  {f1:.4f}\n")

print("Metrics written to metrics.txt")

# ── 9. Confusion matrix plot ───────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("conf_matrix.png", dpi=100)
print("Confusion matrix saved to conf_matrix.png")
