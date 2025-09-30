#!/usr/bin/env python3
"""
Quick train & save inference pipeline for fraud detection (Logistic Regression only)
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42

# 1. Load data
df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 2. Preprocessing
num_cols = X_train.columns.tolist()
preproc = ColumnTransformer([("scale", StandardScaler(), num_cols)], remainder="drop")

# 3. Fast Logistic Regression pipeline
pipe_logreg = Pipeline([
    ("preproc", preproc),
    ("clf", LogisticRegression(
        solver="liblinear",      # faster than saga
        class_weight="balanced",
        max_iter=500,            # reduced iterations
        random_state=SEED
    ))
])

# 4. Train & evaluate
print("Training Logistic Regression...")
pipe_logreg.fit(X_train, y_train)
y_proba = pipe_logreg.predict_proba(X_test)[:, 1]
ap = average_precision_score(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
print(f"Logreg -> AP={ap:.4f}, AUC={auc:.4f}")

# 5. Save final inference pipeline
os.makedirs("models/sklearn", exist_ok=True)
save_path = "models/sklearn/fraud_pipeline.joblib"
joblib.dump(pipe_logreg, save_path)
print(f"✅ Inference pipeline saved to {save_path}")
