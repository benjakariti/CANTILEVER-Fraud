#!/usr/bin/env python3
"""
Usage:
  python src/predict_fraud.py --file ../data/sample_to_predict.csv
  python src/predict_fraud.py --example "Time, V1, V2, ..., Amount"  (single example as comma-separated)
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "sklearn" / "fraud_pipeline.joblib"

def load_model():
    pipe = joblib.load(MODEL_PATH)
    return pipe

def predict_file(pipe, filepath):
    df = pd.read_csv(filepath)
    probs = pipe.predict_proba(df)[:,1]
    preds = (probs >= 0.5).astype(int)
    out = df.copy()
    out["fraud_prob"] = probs
    out["fraud_pred"] = preds
    print(out[["fraud_prob","fraud_pred"]].head())
    return out

def predict_example(pipe, example_str):
    # expects comma-separated values in order of features (Time, V1..V28, Amount)
    vals = [float(x.strip()) for x in example_str.split(",")]
    df = pd.DataFrame([vals], columns=pipe.named_steps['preproc'].transformers[0][2])
    # If ColumnTransformer above used names differently, you may need to provide columns explicitly.
    probs = pipe.predict_proba(df)[:,1]
    print("prob, pred:", probs[0], int(probs[0] >= 0.5))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="CSV file with same columns as training data")
    parser.add_argument("--example", type=str, help="Single example: numeric comma-separated values")
    args = parser.parse_args()

    pipe = load_model()
    if args.file:
        out = predict_file(pipe, args.file)
        out.to_csv("predictions_out.csv", index=False)
        print("Saved predictions_out.csv")
    elif args.example:
        predict_example(pipe, args.example)
    else:
        print("Provide --file or --example")

if __name__ == "__main__":
    main()
