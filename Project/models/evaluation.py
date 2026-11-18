"""Shared multiclass evaluation for the injury model notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, log_loss,
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


def fit_calibrated(model, X, y, *, calibration_size=0.2):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=calibration_size, stratify=y, random_state=42
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
    risk = probabilities @ np.arange(probabilities.shape[1])
    return np.searchsorted(cutoffs, risk, side="right")


def select_calibration_and_thresholds(y_val, probabilities):
    classes = np.arange(probabilities.shape[1])
    temperature = min(
        np.geomspace(0.25, 4, 81),
        key=lambda value: log_loss(
            y_val, temperature_softmax(probabilities, value), labels=classes
        ),
    )
    calibrated = temperature_softmax(probabilities, temperature)
    # Use validation risk quantiles to cover the range where predictions change.
    risk = calibrated @ classes
    quantiles = np.quantile(risk, np.linspace(0, 1, 31))
    candidates = np.unique(np.r_[0, quantiles, len(classes) - 1])
    choices = (
        ((cutoff,) for cutoff in candidates)
        if len(classes) == 2
        else ((lower, upper) for i, lower in enumerate(candidates) for upper in candidates[i + 1:])
    )
    actual = np.asarray(y_val, dtype=int)
    n_classes = len(classes)
    actual_counts = np.bincount(actual, minlength=n_classes)

    def macro_f1(thresholds):
        predicted = np.searchsorted(thresholds, risk, side="right")
        correct_counts = np.bincount(actual[actual == predicted], minlength=n_classes)
        predicted_counts = np.bincount(predicted, minlength=n_classes)
        return np.mean(2 * correct_counts / np.maximum(actual_counts + predicted_counts, 1))

    cutoffs = max(choices, key=macro_f1)
    return temperature, cutoffs


def threshold_report(y_true, predictions, class_names=CLASS_NAMES):
    matrix = confusion_matrix(y_true, predictions, labels=np.arange(len(class_names)))
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
        index=class_names,
    )
    report.loc["Macro average"] = report.mean()
    confusion = pd.DataFrame(
        matrix,
        index=pd.Index(class_names, name="Actual"),
        columns=pd.Index(class_names, name="Predicted"),
    )
    return report, confusion


def probability_report(y_true, raw, calibrated, *, plot=False, class_names=CLASS_NAMES):
    actual = np.asarray(y_true)[:, None] == np.arange(len(class_names))
    if plot:
        figure, (roc_ax, calibration_ax, pr_ax) = plt.subplots(1, 3, figsize=(16, 5))
    rows = []
    for i, name in enumerate(class_names):
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

    report = pd.DataFrame(rows, index=class_names)
    report.loc["Macro average"] = report.mean()
    multiclass_brier = report.loc[class_names, "Calibrated Brier"].sum()
    macro_roc_auc = report.loc["Macro average", "ROC AUC"]
    # Convert Brier loss (range 0–2) to a higher-is-better score before combining.
    weighted_score = 0.5 * (1 - multiclass_brier / 2) + 0.5 * macro_roc_auc
    return report, multiclass_brier, weighted_score


def score_predictions(y, raw, calibrated, predictions, *, plot=False, class_names=CLASS_NAMES):
    threshold, confusion = threshold_report(y, predictions, class_names)
    probability, brier, weighted_score = probability_report(
        y, raw, calibrated, plot=plot, class_names=class_names
    )
    # For two labels, the per-class Brier losses are equal. Use the standard 0–1 score.
    brier_scale = 2 if len(class_names) == 2 else 1
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "raw_brier": float(probability.loc[class_names, "Raw Brier"].sum() / brier_scale),
        "calibrated_brier": float(brier / brier_scale),
        "log_loss": float(log_loss(y, calibrated, labels=np.arange(len(class_names)))),
        "combined_score": float(weighted_score),
        "threshold_metrics": threshold.to_dict(orient="index"),
        "probability_metrics": probability.to_dict(orient="index"),
        "confusion_matrix": confusion.to_dict(orient="index"),
    }


def feature_importance(model, columns):
    return {str(name): float(value) for name, value in zip(columns, model.feature_importances_, strict=True)}


def evaluate_model(model, *, plot=True):
    X_train, X_test, y_train, y_test = load_data()
    temperature, cutoffs = fit_calibrated(model, X_train, y_train)
    raw = model.predict_proba(X_test)
    calibrated = temperature_softmax(raw, temperature)
    predictions = ordinal_predictions(calibrated, cutoffs)

    result = score_predictions(y_test, raw, calibrated, predictions, plot=plot)
    result["data"] = {"feature_count": X_train.shape[1], "evaluated_rows": len(y_test)}
    result["feature_importance"] = feature_importance(model, X_train.columns)
    result["calibration"] = {"temperature": float(temperature), "cutoffs": list(map(float, cutoffs))}
    return result
