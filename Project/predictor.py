"""Restore a recorded experiment configuration for local inference."""

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from threading import Lock

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from experiments import EXPERIMENTS, PROJECT
from models.evaluation import fit_calibrated, ordinal_predictions, temperature_softmax


MODEL_NAMES = {estimator.__name__: name for name, (estimator, _) in EXPERIMENTS.items()}
FIT_LOCK = Lock()


def model_option(run):
    data = run["metrics"].get("data", {})
    if (data.get("source") != "RawData.csv" or not data.get("features")
            or not data.get("target_classes") or run["model"] not in MODEL_NAMES):
        return None
    return {
        "id": run["id"], "model": run["model"], "name": run["name"],
        "evaluation": run["evaluated_on"], "classes": data["target_classes"],
        "feature_count": len(data["features"]),
        "combined_score": run["metrics"]["combined_score"],
    }


@dataclass
class LoadedModel:
    option: dict
    pipeline: object
    temperature: float
    cutoffs: tuple
    fields: list

    def schema(self):
        return self.option | {"fields": self.fields, "cutoffs": list(map(float, self.cutoffs))}

    def predict(self, values):
        fields = [field["name"] for field in self.fields]
        if not isinstance(values, dict) or set(values).difference(fields):
            raise ValueError("Enter values using the loaded model’s feature fields.")
        row = {}
        for name in fields:
            value = values.get(name)
            if value is None:
                row[name] = np.nan
            elif type(value) in (int, float) and isfinite(value) and abs(value) <= np.finfo(np.float32).max:
                row[name] = float(value)
            else:
                raise ValueError(f"{name} must be a finite number within the supported numeric range.")
        if all(np.isnan(value) for value in row.values()):
            raise ValueError("Enter at least one feature value before predicting.")
        raw = self.pipeline.predict_proba(pd.DataFrame([row], columns=fields))
        probabilities = temperature_softmax(raw, self.temperature)
        selected = ordinal_predictions(probabilities, self.cutoffs)[0]
        classes = self.option["classes"]
        return {
            "run_id": self.option["id"],
            "prediction": classes[selected],
            "probabilities": dict(zip(classes, probabilities[0].tolist(), strict=True)),
            "imputed_features": [name for name in fields if np.isnan(row[name])],
        }


@lru_cache(maxsize=6)
def cached_model(run_id):
    # Imported here so the history view can still run without loading ML libraries.
    from experiment_view import read_runs

    run = next((row for row in read_runs(run_id)), None)
    option = model_option(run) if run else None
    if option is None:
        raise ValueError("This run has no saved feature configuration to load.")
    data = run["metrics"]["data"]
    columns = data["features"]
    raw = pd.read_csv(PROJECT / "src" / "Dataset" / "RawData.csv")
    original_target = raw["injury_occurred"]
    X, _, y, _ = train_test_split(
        raw[columns], original_target, test_size=data["holdout_fraction"],
        stratify=original_target, random_state=data["split_seed"],
    )
    if len(option["classes"]) == 2:
        y = (y == 2).astype(int)

    estimator, _ = EXPERIMENTS[MODEL_NAMES[run["model"]]]
    params = {
        key: float(value) if isinstance(value, str) and value in {"nan", "inf", "-inf"} else value
        for key, value in run["params"].items()
    }
    pipeline = make_pipeline(SimpleImputer(strategy="median"), estimator(**params))
    temperature, cutoffs = fit_calibrated(pipeline, X, y, calibration_size=data["calibration_fraction"])
    fields = [
        {"name": name, "median": float(median), "minimum": float(X[name].min()), "maximum": float(X[name].max())}
        for name, median in zip(columns, pipeline.steps[0][1].statistics_, strict=True)
    ]
    return LoadedModel(option, pipeline, temperature, cutoffs, fields)


def load_model(run_id):
    with FIT_LOCK:
        return cached_model(run_id)
