
from features import load_manifest, get_features
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import json
import joblib
import matplotlib
matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xgboost_model.joblib"
METRICS_DIR = ROOT / "outputs" / "metrics"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
CLASSES = [1, 2, 3, 4, 5, 6]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run python src/train.py first.")

    data = load_manifest()
    X = get_features(
        data, rebuild="--rebuild-features" in __import__("sys").argv)

    y = data["label"].to_numpy()
    test_idx = np.flatnonzero(data["split"].to_numpy() == "test")

    model = joblib.load(MODEL_PATH)

    pred = model.predict(X[test_idx]).astype(int) + 1
    prob = model.predict_proba(X[test_idx])

    report = classification_report(
        y[test_idx],
        pred,
        labels=CLASSES,
        digits=4,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y[test_idx], pred)
        ),
        "macro_f1": float(
            f1_score(
                y[test_idx], pred,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y[test_idx], pred,
                labels=CLASSES,
                average="weighted",
                zero_division=0,
            )
        ),
        "mean_max_probability": float(prob.max(axis=1).mean()),
        "test_samples": int(len(test_idx)),
        "classification_report": classification_report(
            y[test_idx], pred,
            labels=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    results = data.iloc[test_idx][
        ["sample_id", "filepath", "label", "split"]
    ].copy()
    results["predicted_label"] = pred
    results["confidence"] = prob.max(axis=1)

    for j, label in enumerate(CLASSES):
        results[f"probability_class_{label}"] = prob[:, j]

    results.to_csv(
        PREDICTIONS_DIR / "test_predictions.csv",
        index=False,
    )

    with open(METRICS_DIR / "test_metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    with open(
        METRICS_DIR / "test_classification_report.txt", "w"
    ) as file:
        file.write(report)

    cm = confusion_matrix(y[test_idx], pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_title("XGBoost Test Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(6), CLASSES)
    ax.set_yticks(range(6), CLASSES)

    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(
        METRICS_DIR / "test_confusion_matrix.png",
        dpi=160,
    )
    plt.close(fig)

    print("XGBoost test results")
    print("--------------------")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Mean max probability: {metrics['mean_max_probability']:.4f}")
    print()
    print(report)


if __name__ == "__main__":
    main()
