"""Baseline Dummy model (01_dummy) for Bobathon ESILV.

Trains a dummy mean regressor, evaluates it via skore, writes predictions to
submissions/01_dummy.csv, and publishes the report to Skore Hub.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyRegressor
from skore import Project, evaluate, login

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from parkinson.hub import load_skore_credentials


def main() -> None:
    print("Loading datasets...", flush=True)
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    visits = X_train.merge(y_train, on="Index")

    feature_cols = [
        "sexM",
        "age_at_diagnosis",
        "age",
        "ledd",
        "time_since_intake_on",
        "time_since_intake_off",
        "on",
        "off",
    ]
    X = visits[feature_cols]
    y = visits["target"]

    print("Evaluating DummyRegressor(strategy='mean') with skore.evaluate...", flush=True)
    dummy = DummyRegressor(strategy="mean")
    report = evaluate(dummy, X, y)

    try:
        rmse = report.metrics.rmse()
        print(f"Dummy model evaluated. RMSE: {rmse}", flush=True)
    except Exception as exc:
        print(f"Could not compute rmse via report.metrics: {exc}", flush=True)

    # Generate predictions on X_test
    print("Fitting dummy model on full train data and generating predictions...", flush=True)
    dummy.fit(X, y)
    preds = dummy.predict(X_test[feature_cols])

    SUBMISSIONS_DIR = REPO_ROOT / "submissions"
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSIONS_DIR / "01_dummy.csv"
    submission_df = pd.DataFrame({"Index": X_test["Index"], "target": preds})
    submission_df.to_csv(submission_path, index=False)
    print(f"Saved submission file to {submission_path} ({len(submission_df)} rows)", flush=True)

    # Push to Skore Hub
    print("Connecting to Skore Hub...", flush=True)
    config = load_skore_credentials(REPO_ROOT)
    workspace = config.get("workspace")
    print(f"Using workspace: {workspace}", flush=True)

    login(mode="hub")
    project = Project(name="bobathon-esilv", mode="hub", workspace=workspace)
    print("Uploading report '01_dummy' to Skore Hub...", flush=True)
    project.put("01_dummy", report)
    print("Upload complete!", flush=True)


if __name__ == "__main__":
    main()
