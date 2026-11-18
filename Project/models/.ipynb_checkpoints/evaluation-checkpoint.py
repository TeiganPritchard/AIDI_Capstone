"""Shared multiclass evaluation for the injury model notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score, log_loss,
    precision_recall_curve, roc_curve,
)
from sklearn.model_selection import train_test_split


CLASS_NAMES = ["Healthy", "Low Risk", "Injured"]
CLASSES = np.arange(len(CLASS_NAMES))


def load_data(*, include_test=True):
    dataset = Path(__file__).resolve().parent.parent / "src" / "Dataset"
    X_train = pd.read_csv(dataset / "X_train.csv")
    y_train = pd.read_csv(dataset / "y_train.csv")["injury_occurred"]
    if not include_test:
        return X_train, None, y_train, None
    X_test = pd.read_csv(dataset / "X_test.csv")
    y_test = pd.read_csv(dataset / "y_test.csv")["injury_occurred"]
    return X_train, X_test, y_train, y_test


def fit_calibrated(model, X, y):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_fit, y_fit)
    return select_calibration_and_thresholds(y_val, model.predict_proba(X_val))


def temperature_softmax(probabilities, temperature):
    # Log probabilities act as logits up to an additive constant. This also
    # works with Random Forest's averaged class probabilities.
    logits = np.log(np.clip(probabilities.astype(float), 1e-15, 1)) / temperature
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def ordinal_predictions(probabilities, cutoffs):
    risk = probabilities @ CLASSES
    return np.searchsorted(cutoffs, risk, side="right")


def select_calibration_and_thresholds(y_val, probabilities):
    temperature = min(
        np.geomspace(0.25, 4, 81),
        key=lambda value: log_loss(
            y_val, temperature_softmax(probabilities, value), labels=CLASSES
        ),
    )
    calibrated = temperature_softmax(probabilities, temperature)
    # Use validation risk quantiles to cover the range where predictions change.
    quantiles = np.quantile(calibrated @ CLASSES, np.linspace(0, 1, 31))
    candidates = np.unique(np.r_[0, quantiles, len(CLASSES) - 1])
    cutoffs = max(
        ((lower, upper) for i, lower in enumerate(candidates) for upper in candidates[i + 1:]),
        key=lambda pair: f1_score(
            y_val, ordinal_predictions(calibrated, pair), labels=CLASSES,
            average="macro", zero_division=0,
        ),
    )
    return temperature, cutoffs


def threshold_report(y_true, predictions):
    matrix = confusion_matrix(y_true, predictions, labels=CLASSES)
    tp = matrix.diagonal()
    fn = matrix.sum(axis=1) - tp
    fp = matrix.sum(axis=0) - tp
    tn = matrix.sum() - tp - fn - fp
    report = pd.DataFrame(
        {
            "Accuracy (OvR)": (tp + tn) / matrix.sum(),
            "Precision": tp / np.maximum(tp + fp, 1),
            "Recall": tp / np.maximum(tp + fn, 1),
            "F1": 2 * tp / np.maximum(2 * tp + fp + fn, 1),
            "TPR": tp / np.maximum(tp + fn, 1),
            "FPR": fp / np.maximum(fp + tn, 1),
        },
        index=CLASS_NAMES,
    )
    report.loc["Macro average"] = report.mean()
    confusion = pd.DataFrame(
        matrix,
        index=pd.Index(CLASS_NAMES, name="Actual"),
        columns=pd.Index(CLASS_NAMES, name="Predicted"),
    )
    return report, confusion


def probability_report(y_true, raw, calibrated, *, plot=False):
    actual = np.asarray(y_true)[:, None] == CLASSES
    if plot:
        figure, (roc_ax, calibration_ax, pr_ax) = plt.subplots(1, 3, figsize=(16, 5))
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        observed = actual[:, i]
        scores = calibrated[:, i]
        fpr, tpr, _ = roc_curve(observed, scores)
        precision, recall, _ = precision_recall_curve(observed, scores)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        rows.append({
            "Raw Brier": np.mean((raw[:, i] - observed) ** 2),
            "Calibrated Brier": np.mean((scores - observed) ** 2),
            "ROC AUC": roc_auc,
            "PR AUC": pr_auc,
        })
        if not plot:
            continue
        roc_line, = roc_ax.plot(fpr, tpr, label=f"{name} ({roc_auc:.3f})")
        color = roc_line.get_color()
        pr_ax.plot(recall, precision, color=color, label=f"{name} ({pr_auc:.3f})")
        pr_ax.axhline(observed.mean(), color=color, linestyle=":", alpha=0.5)
        for values, style, label in [
            (raw[:, i], "--", "raw"),
            (scores, "-", "calibrated"),
        ]:
            fraction_positive, mean_prediction = calibration_curve(
                observed, values, n_bins=10, strategy="quantile"
            )
            calibration_ax.plot(
                mean_prediction, fraction_positive, style, marker=".",
                color=color, label=f"{name} ({label})",
            )
    if plot:
        roc_ax.plot([0, 1], [0, 1], "k:", label="Chance")
        calibration_ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
        roc_ax.set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
        calibration_ax.set(
            title="Calibration curves", xlabel="Mean predicted probability", ylabel="Observed frequency"
        )
        pr_ax.set(title="Precision–recall curves", xlabel="Recall", ylabel="Precision")
        for axis in (roc_ax, calibration_ax, pr_ax):
            axis.set(xlim=(0, 1), ylim=(0, 1))
            axis.legend(fontsize="small")
        figure.tight_layout()

    report = pd.DataFrame(rows, index=CLASS_NAMES)
    report.loc["Macro average"] = report.mean()
    multiclass_brier = report.loc[CLASS_NAMES, "Calibrated Brier"].sum()
    macro_roc_auc = report.loc["Macro average", "ROC AUC"]
    # Convert Brier loss (range 0–2) to a higher-is-better score before combining.
    weighted_score = 0.5 * (1 - multiclass_brier / 2) + 0.5 * macro_roc_auc
    return report, multiclass_brier, weighted_score


def score_predictions(y, raw, calibrated, predictions, *, plot=False):
    threshold, confusion = threshold_report(y, predictions)
    probability, brier, weighted_score = probability_report(y, raw, calibrated, plot=plot)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "raw_brier": float(probability.loc[CLASS_NAMES, "Raw Brier"].sum()),
        "calibrated_brier": float(brier),
        "log_loss": float(log_loss(y, calibrated, labels=CLASSES)),
        "combined_score": float(weighted_score),
        "threshold_metrics": threshold.to_dict(orient="index"),
        "probability_metrics": probability.to_dict(orient="index"),
        "confusion_matrix": confusion.to_dict(orient="index"),
    }


def evaluate_model(model, *, plot=True):
    X_train, X_test, y_train, y_test = load_data()
    temperature, cutoffs = fit_calibrated(model, X_train, y_train)
    raw = model.predict_proba(X_test)
    calibrated = temperature_softmax(raw, temperature)
    predictions = ordinal_predictions(calibrated, cutoffs)

    result = score_predictions(y_test, raw, calibrated, predictions, plot=plot)
    result["data"] = {"feature_count": X_train.shape[1], "evaluated_rows": len(y_test)}
    result["calibration"] = {"temperature": float(temperature), "cutoffs": list(map(float, cutoffs))}
    return result
