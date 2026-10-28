
from pathlib import Path
import csv
import json
import shutil
import sys
import time
from datetime import datetime

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from features import load_manifest, get_features


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xgboost_model.joblib"
OUTPUT = ROOT / "outputs" / "metrics"

# Stop starting new configurations after this time.
# An individual model.fit() already in progress is not interrupted.
TIME_BUDGET_SECONDS = 24 * 60

# Existing best validation macro F1 from the previous training run.
# The loaded baseline model's score will be measured again below.
PREVIOUS_BASELINE_F1 = 0.7724

# Three targeted candidates. Keep the same train/validation split.
CONFIGS = [
    {
        "name": "A_simpler_trees",
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 1,
    },
    {
        "name": "B_stronger_regularization",
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 3,
    },
    {
        "name": "C_slower_learning",
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 3,
    },
]


def calculate_metrics(y_true, predictions):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=[1, 2, 3, 4, 5, 6],
                average="macro",
                zero_division=0,
            )
        ),
    }


def save_results(results, path):
    # Rewrite the small results file after every completed run.
    # This means completed experiments remain recorded if interrupted.
    columns = [
        "name",
        "status",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "elapsed_seconds",
        "error",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)


def main():
    started = time.monotonic()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_path = OUTPUT / f"tuning_results_{timestamp}.csv"
    summary_path = OUTPUT / f"tuning_summary_{timestamp}.json"
    candidate_path = ROOT / "models" / "xgboost_tuned_candidate.joblib"

    # Preserve the current model before any tuning.
    baseline_model = None
    baseline_backup = ROOT / "models" / (
        f"xgboost_baseline_backup_{timestamp}.joblib"
    )

    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, baseline_backup)
        baseline_model = joblib.load(MODEL_PATH)
        print(f"Baseline backup saved to: {baseline_backup}")
    else:
        print("WARNING: No existing model found to back up.")

    print("Loading cached features...")
    data = load_manifest()
    X = get_features(
        data,
        rebuild="--rebuild-features" in sys.argv,
    )

    y = data["label"].to_numpy()
    split = data["split"].to_numpy()

    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "val")

    y_encoded = y - 1

    # Weights are computed from training data only.
    weights = compute_sample_weight(
        class_weight="balanced",
        y=y[train_idx],
    )

    # Score the existing baseline on the same validation split.
    baseline_f1 = PREVIOUS_BASELINE_F1
    if baseline_model is not None:
        baseline_predictions = (
            baseline_model.predict(X[val_idx]).astype(int) + 1
        )
        baseline_metrics = calculate_metrics(
            y[val_idx], baseline_predictions
        )
        baseline_f1 = baseline_metrics["macro_f1"]

        print("\nExisting baseline validation results:")
        print(json.dumps(baseline_metrics, indent=2))

    best_model = None
    best_config = None
    best_metrics = None
    best_score = -1.0
    results = []

    for number, config in enumerate(CONFIGS, start=1):
        elapsed = time.monotonic() - started
        remaining = TIME_BUDGET_SECONDS - elapsed

        # Do not start another potentially long model fit near the limit.
        if remaining < 8 * 60:
            print(
                f"\nTime budget nearly exhausted "
                f"({remaining / 60:.1f} minutes remain)."
            )
            print("Not starting another configuration.")
            break

        params = {
            key: value
            for key, value in config.items()
            if key != "name"
        }

        print(f"\n{'=' * 55}")
        print(f"Configuration {number}/{len(CONFIGS)}: {config['name']}")
        print(params)
        print(f"Approximate time remaining: {remaining / 60:.1f} minutes")

        result = {
            "name": config["name"],
            "status": "running",
            **params,
            "accuracy": "",
            "balanced_accuracy": "",
            "macro_f1": "",
            "elapsed_seconds": "",
            "error": "",
        }
        results.append(result)
        save_results(results, results_path)

        fit_started = time.monotonic()

        try:
            model = XGBClassifier(
                objective="multi:softprob",
                num_class=6,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                **params,
            )

            model.fit(
                X[train_idx],
                y_encoded[train_idx],
                sample_weight=weights,
                verbose=False,
            )

            predictions = model.predict(X[val_idx]).astype(int) + 1
            metrics = calculate_metrics(y[val_idx], predictions)
            fit_seconds = time.monotonic() - fit_started

            result.update({
                "status": "completed",
                **metrics,
                "elapsed_seconds": round(fit_seconds, 1),
            })

            print(f"Validation accuracy: {metrics['accuracy']:.4f}")
            print(
                "Validation balanced accuracy: "
                f"{metrics['balanced_accuracy']:.4f}"
            )
            print(f"Validation macro F1: {metrics['macro_f1']:.4f}")
            print(f"Training time: {fit_seconds / 60:.1f} minutes")

            if metrics["macro_f1"] > best_score:
                best_score = metrics["macro_f1"]
                best_model = model
                best_config = params.copy()
                best_metrics = metrics.copy()

                # Save the best candidate so far immediately.
                joblib.dump(best_model, candidate_path)
                print(f"Best candidate checkpoint saved: {candidate_path}")

        except Exception as exc:
            result.update({
                "status": "failed",
                "error": repr(exc),
                "elapsed_seconds": round(
                    time.monotonic() - fit_started, 1
                ),
            })
            print(f"Configuration failed: {exc}")

        # Persist the result immediately after each completed/failed run.
        save_results(results, results_path)
        print(f"Results saved: {results_path}")

    # Never replace the baseline with a worse candidate.
    promoted = False

    if best_model is not None and best_score > baseline_f1:
        shutil.copy2(candidate_path, MODEL_PATH)
        promoted = True
        print("\nTuned candidate beats the baseline and was promoted.")
    else:
        print(
            "\nNo candidate beat the baseline validation macro F1. "
            "The original model remains unchanged."
        )

    summary = {
        "baseline_validation_macro_f1": float(baseline_f1),
        "best_tuned_validation_macro_f1": (
            float(best_score) if best_model is not None else None
        ),
        "best_tuned_parameters": best_config,
        "best_tuned_metrics": best_metrics,
        "tuned_model_promoted": promoted,
        "completed_configurations": sum(
            row["status"] == "completed" for row in results
        ),
        "total_configurations": len(CONFIGS),
        "elapsed_minutes": round(
            (time.monotonic() - started) / 60, 2
        ),
        "results_csv": str(results_path),
        "candidate_model": (
            str(candidate_path) if best_model is not None else None
        ),
        "baseline_backup": (
            str(baseline_backup) if baseline_backup.exists() else None
        ),
        "note": (
            "Used the fixed validation split for tuning; "
            "the test set was not used for model selection. "
            "No early stopping or cross-validation was used."
        ),
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nFinal tuning summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nResults CSV: {results_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Active model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
