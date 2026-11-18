"""Run and track the multiclass injury-model experiments."""

import argparse
import json
import sqlite3
import subprocess
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from models.evaluation import (
    CLASSES, evaluate_model, feature_importance, fit_calibrated, load_data, ordinal_predictions,
    score_predictions, temperature_softmax,
)


PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "experiments" / "results.sqlite3"
SEED = 42
EXPERIMENTS = {
    "random_forest": (RandomForestClassifier, {
        "n_estimators": 100, "random_state": SEED, "n_jobs": 4,
    }),
    "catboost": (CatBoostClassifier, {
        "iterations": 500, "loss_function": "MultiClass", "auto_class_weights": "Balanced",
        "random_seed": SEED, "thread_count": 4, "verbose": False, "allow_writing_files": False,
    }),
    "xgboost": (XGBClassifier, {
        "objective": "multi:softprob", "eval_metric": "mlogloss", "tree_method": "hist",
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
        "random_state": SEED, "n_jobs": 4,
    }),
}


def build_model(name, **overrides):
    estimator, params = EXPERIMENTS[name]
    return estimator(**(params | overrides))


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def connect():
    RESULTS.parent.mkdir(exist_ok=True)
    connection = sqlite3.connect(RESULTS)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            name TEXT NOT NULL,
            evaluated_on TEXT NOT NULL,
            model TEXT NOT NULL,
            params TEXT NOT NULL,
            notes TEXT NOT NULL,
            git_commit TEXT,
            metrics TEXT NOT NULL
        )
    """)
    return connection


def save_result(name, model, split, metrics, notes):
    with closing(connect()) as connection, connection:
        cursor = connection.execute(
            """INSERT INTO runs (timestamp, name, evaluated_on, model, params, notes, git_commit, metrics)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"), name, split,
                type(model).__name__, serialized_params(model),
                notes, git_commit(), json.dumps(metrics, allow_nan=False),
            ),
        )
        return cursor.lastrowid


def serialized_params(model):
    params = {
        key: repr(value) if isinstance(value, float) and not np.isfinite(value) else value
        for key, value in model.get_params().items()
    }
    return json.dumps(params, sort_keys=True, allow_nan=False)


def cross_validate(name):
    X, _, y, _ = load_data(include_test=False)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    raw = np.empty((len(y), len(CLASSES)))
    calibrated = np.empty_like(raw)
    predictions = np.empty(len(y), dtype=int)
    calibration = []
    importance = []
    for fit_rows, check_rows in folds.split(X, y):
        model = build_model(name)
        temperature, cutoffs = fit_calibrated(model, X.iloc[fit_rows], y.iloc[fit_rows])
        importance.append(feature_importance(model, X.columns))
        raw[check_rows] = model.predict_proba(X.iloc[check_rows])
        calibrated[check_rows] = temperature_softmax(raw[check_rows], temperature)
        predictions[check_rows] = ordinal_predictions(calibrated[check_rows], cutoffs)
        calibration.append({"temperature": float(temperature), "cutoffs": list(map(float, cutoffs))})
    result = score_predictions(y, raw, calibrated, predictions)
    result["data"] = {"feature_count": X.shape[1], "evaluated_rows": len(y)}
    result["calibration_by_fold"] = calibration
    result["feature_importance"] = pd.DataFrame(importance).mean().to_dict()
    return result


def run_test(name, model, *, notes="", plot=False):
    result = evaluate_model(model, plot=plot)
    run_id = save_result(name, model, "test", result, notes)
    return run_id, result


def show(history=False, best=False):
    with closing(connect()) as connection:
        rows = connection.execute("SELECT id, timestamp, name, evaluated_on, metrics FROM runs ORDER BY id")
        entries = []
        for row in rows:
            metrics = json.loads(row["metrics"])
            entries.append({
                "id": row["id"], "timestamp": row["timestamp"], "name": row["name"],
                "split": row["evaluated_on"], "accuracy": metrics["accuracy"],
                "macro_f1": metrics["threshold_metrics"]["Macro average"]["F1"],
                "brier": metrics["calibrated_brier"],
                "roc_auc": metrics["probability_metrics"]["Macro average"]["ROC AUC"],
                "pr_auc": metrics["probability_metrics"]["Macro average"]["PR AUC"],
                "combined": metrics["combined_score"],
            })
    if not entries:
        print("No experiment runs recorded yet.")
        return
    table = pd.DataFrame(entries)
    if best:
        table = table.sort_values(["split", "combined", "id"], ascending=[True, False, False])
        table = table.drop_duplicates("split", keep="first")
    elif not history:
        table = table.drop_duplicates(["name", "split"], keep="last")
        table = table.sort_values(["split", "combined"], ascending=[True, False])
    print(table.round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--only", choices=EXPERIMENTS, help="cross-validate one experiment")
    action.add_argument("--final", choices=EXPERIMENTS, help="evaluate one experiment on the test split")
    action.add_argument("--show", action="store_true", help="show the latest result per experiment and split")
    action.add_argument("--history", action="store_true", help="show all runs")
    action.add_argument("--best", action="store_true", help="show the best historical run per split by combined score")
    action.add_argument("--run", type=int, metavar="ID", help="show full settings and metrics for one run")
    action.add_argument("--rerun", action="store_true", help="cross-validate every experiment again")
    parser.add_argument("--notes", default="", help="annotation to save with new runs")
    args = parser.parse_args()

    if args.show or args.history or args.best:
        show(history=args.history, best=args.best)
        return
    if args.run is not None:
        with closing(connect()) as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (args.run,)).fetchone()
        if row is None:
            parser.error(f"no run with ID {args.run}")
        record = dict(row)
        for column in ("params", "metrics"):
            record[column] = json.loads(record[column])
        print(json.dumps(record, indent=2))
        return
    if args.final:
        run_id, result = run_test(args.final, build_model(args.final), notes=args.notes)
        print(f"Saved test run {run_id} for {args.final} (combined score {result['combined_score']:.4f})")
        return

    with closing(connect()) as connection:
        recorded = {
            (row["name"], row["params"])
            for row in connection.execute("SELECT name, params FROM runs WHERE evaluated_on = 'cv'")
        }
    for name in [args.only] if args.only else EXPERIMENTS:
        model = build_model(name)
        if not (args.only or args.rerun) and (name, serialized_params(model)) in recorded:
            continue
        result = cross_validate(name)
        run_id = save_result(name, model, "cv", result, args.notes)
        print(f"Saved CV run {run_id} for {name} (combined score {result['combined_score']:.4f})")
    show()


if __name__ == "__main__":
    main()
