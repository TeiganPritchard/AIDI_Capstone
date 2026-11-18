"""Compare injury prediction features and settings and record every run."""

import argparse
import json
from contextlib import closing
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline

from experiments import PROJECT, SEED, build_model, connect, save_result
from models.evaluation import (
    CLASS_NAMES, feature_importance, fit_calibrated, ordinal_predictions, score_predictions,
    temperature_softmax,
)


CORE4 = ["recovery_score", "sleep_quality", "fatigue_index", "training_load"]
CORE6 = CORE4 + ["stress_level", "hydration_level"]
FEATURES = {
    "core4": CORE4,
    "core6": CORE6,
    "physiology": [
        "heart_rate", "body_temperature", "hydration_level", "sleep_quality",
        "recovery_score", "stress_level",
    ],
    "workload": ["training_intensity", "training_duration", "training_load", "fatigue_index"],
    "physiology_workload": [
        *CORE6, "heart_rate", "body_temperature", "training_intensity", "training_duration",
    ],
    "biomechanics": [
        "muscle_activity", "joint_angles", "gait_speed", "cadence", "step_count",
        "jump_height", "ground_reaction_force", "range_of_motion",
    ],
    "all_numeric": None,
}
SETTINGS = {
    "random_forest": [
        ("regularized", {"n_estimators": 250, "min_samples_leaf": 10}),
        ("balanced_regularized", {"n_estimators": 250, "min_samples_leaf": 10, "class_weight": "balanced"}),
    ],
    "catboost": [
        ("unweighted", {"auto_class_weights": None}),
        ("shallow_unweighted", {"depth": 3, "auto_class_weights": None}),
    ],
    "xgboost": [
        ("shallow", {"max_depth": 2}),
        ("regularized", {"max_depth": 3, "min_child_weight": 10, "subsample": 0.8}),
    ],
}
SWEEP_BASE = FEATURES["physiology_workload"]
SWEEP_FEATURES = {
    "physiology_workload": SWEEP_BASE,
    "core6": CORE6,
    "all_numeric": None,
    "recovery_stress": ["recovery_score", "stress_level"],
    "recovery_stress_fatigue": ["recovery_score", "stress_level", "fatigue_index"],
    **{
        f"without_{feature}": [name for name in SWEEP_BASE if name != feature]
        for feature in ("recovery_score", "stress_level", "fatigue_index", "sleep_quality", "hydration_level")
    },
}
SWEEP_SETTINGS = [
    ("sweep_depth1", {"max_depth": 1, "n_estimators": 200, "learning_rate": 0.08}, 0.10),
    ("sweep_depth2_fast", {"max_depth": 2, "n_estimators": 180, "learning_rate": 0.08}, 0.10),
    ("sweep_depth2_l2", {"max_depth": 2, "n_estimators": 220, "reg_lambda": 8}, 0.20),
    ("sweep_depth3", {"max_depth": 3, "n_estimators": 160, "learning_rate": 0.08}, 0.10),
    ("sweep_depth3_sampled", {"max_depth": 3, "n_estimators": 200, "min_child_weight": 8,
                              "subsample": 0.8, "colsample_bytree": 0.8}, 0.15),
]


@dataclass(frozen=True)
class Candidate:
    model: str
    features: str
    variant: str = "baseline"
    params: dict = field(default_factory=dict)
    calibration_size: float = 0.2

    @property
    def name(self):
        return f"study/{self.model}/{self.features}/{self.variant}/cal{self.calibration_size:.0%}"


def evaluate(candidate, X, y, splits, stage, class_names, X_test=None, y_test=None):
    columns = FEATURES[candidate.features] or X.columns.tolist()
    X = X[columns]
    y_eval = y if y_test is None else y_test
    raw = np.empty((len(y_eval), len(class_names)))
    calibrated = np.empty_like(raw)
    predictions = np.empty(len(y_eval), dtype=int)
    calibration = []
    importance = []
    for fit_rows, check_rows in splits:
        params = candidate.params.copy()
        if len(class_names) == 2:
            if candidate.model == "xgboost":
                params.update(objective="binary:logistic", eval_metric="logloss")
            elif candidate.model == "catboost":
                params.update(loss_function="Logloss")
        estimator = build_model(candidate.model, **params)
        pipeline = make_pipeline(SimpleImputer(strategy="median"), estimator)
        temperature, cutoffs = fit_calibrated(
            pipeline, X.iloc[fit_rows], y.iloc[fit_rows], calibration_size=candidate.calibration_size
        )
        importance.append(feature_importance(estimator, columns))
        X_check = X.iloc[check_rows] if X_test is None else X_test[columns]
        raw[check_rows] = pipeline.predict_proba(X_check)
        calibrated[check_rows] = temperature_softmax(raw[check_rows], temperature)
        predictions[check_rows] = ordinal_predictions(calibrated[check_rows], cutoffs)
        calibration.append({"temperature": float(temperature), "cutoffs": list(map(float, cutoffs))})

    metrics = score_predictions(y_eval, raw, calibrated, predictions, class_names=class_names)
    metrics["calibration_by_fold"] = calibration
    metrics["feature_importance"] = pd.DataFrame(importance).mean().to_dict()
    metrics["data"] = {
        "feature_count": len(columns), "features": columns, "evaluated_rows": len(y_eval),
        "feature_set": candidate.features, "calibration_fraction": candidate.calibration_size,
        "holdout_fraction": 0.3, "split_seed": SEED, "imputation": "fit-fold median",
        "source": "RawData.csv", "cv_folds": len(splits) if X_test is None else None,
        "target_classes": class_names,
    }
    run_id = save_result(candidate.name, estimator, stage, metrics, "Feature and split study")
    print(
        f"{stage} #{run_id} {candidate.name}: "
        f"combined={metrics['combined_score']:.4f} "
        f"macro F1={metrics['threshold_metrics']['Macro average']['F1']:.4f}",
        flush=True,
    )
    return metrics["combined_score"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", action="store_true", help="merge Healthy and Low Risk into Low Risk")
    replay = parser.add_mutually_exclusive_group()
    replay.add_argument("--sweep", action="store_true", help="run 50 XGBoost variations for one target")
    replay.add_argument(
        "--athlete-check", type=int, metavar="RUN_ID",
        help="recheck a five-fold finalist with athlete-disjoint validation folds",
    )
    replay.add_argument(
        "--retest", type=int, metavar="RUN_ID",
        help="rerun the saved test configuration and record feature importance",
    )
    args = parser.parse_args()
    prefix = "binary" if args.binary else "study"
    class_names = ["Low Risk", "High Risk"] if args.binary else CLASS_NAMES

    dataset = pd.read_csv(PROJECT / "src/Dataset/RawData.csv")
    y = dataset["injury_occurred"]
    X = dataset.drop(columns=["injury_occurred", "athlete_id", "session_id", "sport_type", "gender"])
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    if args.binary:
        y, y_test = (y == 2).astype(int), (y_test == 2).astype(int)
    if args.athlete_check is not None or args.retest is not None:
        run_id, stage = (
            (args.retest, f"{prefix}_test") if args.retest is not None
            else (args.athlete_check, f"{prefix}_cv5")
        )
        with closing(connect()) as connection:
            row = connection.execute(
                "SELECT name, metrics FROM runs WHERE id = ? AND evaluated_on = ?",
                (run_id, stage),
            ).fetchone()
        if row is None:
            parser.error(f"run {run_id} must be a {stage} result")
        _, model, feature_set, variant, _ = row["name"].split("/")
        params = dict(SETTINGS[model]).get(variant, {})
        fraction = json.loads(row["metrics"])["data"]["calibration_fraction"]
        candidate = Candidate(model, feature_set, variant, params, fraction)
        if args.retest is not None:
            evaluate(
                candidate, X, y, [(np.arange(len(y)), np.arange(len(y_test)))],
                stage, class_names, X_test=X_test, y_test=y_test,
            )
            return
        groups = dataset.loc[X.index, "athlete_id"]
        folds = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
        evaluate(candidate, X, y, list(folds.split(X, y, groups)), f"{prefix}_athlete_cv5", class_names)
        return
    screen = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, y))
    if args.sweep:
        FEATURES.update(SWEEP_FEATURES)
        stage = f"{prefix}_sweep_cv3"
        with closing(connect()) as connection:
            saved = {
                row["name"] for row in connection.execute(
                    "SELECT name FROM runs WHERE evaluated_on = ?", (stage,)
                )
            }
        for feature_set in SWEEP_FEATURES:
            for variant, params, fraction in SWEEP_SETTINGS:
                candidate = Candidate("xgboost", feature_set, variant, params, fraction)
                if candidate.name not in saved:
                    evaluate(candidate, X, y, screen, stage, class_names)
        return
    confirm = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))

    finalists = []
    for model, variants in SETTINGS.items():
        ranked = []
        for feature_set in FEATURES:
            candidate = Candidate(model, feature_set)
            ranked.append((evaluate(candidate, X, y, screen, f"{prefix}_cv3", class_names), candidate))
        ranked.sort(reverse=True, key=lambda entry: entry[0])
        best, second = [entry[1] for entry in ranked[:2]]
        finalists.extend([best, second])
        finalists.extend(replace(best, variant=label, params=params) for label, params in variants)

    ranked = [
        (evaluate(candidate, X, y, confirm, f"{prefix}_cv5", class_names), candidate)
        for candidate in finalists
    ]
    # Compare calibration sizes for the leading model/feature/settings combination.
    _, leader = max(ranked, key=lambda entry: entry[0])
    for fraction in (0.1, 0.3, 0.4):
        candidate = replace(leader, calibration_size=fraction)
        ranked.append((evaluate(candidate, X, y, confirm, f"{prefix}_cv5", class_names), candidate))

    _, winner = max(ranked, key=lambda entry: entry[0])
    print(f"Selected by five-fold CV: {winner.name}", flush=True)
    evaluate(
        winner, X, y, [(np.arange(len(y)), np.arange(len(y_test)))],
        f"{prefix}_test", class_names, X_test=X_test, y_test=y_test,
    )


if __name__ == "__main__":
    main()
