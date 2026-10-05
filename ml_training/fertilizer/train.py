"""Train a clean held-out NEUTRA-BOOST export from the supplied CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from backend.app.ml.fertilizer_model import (
    BASE_LR,
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    MAX_DEPTH,
    MIN_SPECIALIST_SAMPLES,
    MARGIN_THRESHOLD,
    N_ESTIMATORS,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    SPECIALIST_CONF,
    STABILITY_WEIGHT,
    TARGET_COLUMN,
    TAU,
    NeutraBoost,
)


def read_dataset(path: Path) -> tuple[list[dict[str, str]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Fertilizer dataset not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        expected_columns = FEATURE_ORDER + [TARGET_COLUMN]
        if reader.fieldnames != expected_columns:
            raise ValueError(
                "Fertilizer CSV columns/order do not match the supplied notebook contract. "
                f"Expected {expected_columns}; got {reader.fieldnames}."
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Fertilizer dataset has no records.")
    for row_number, row in enumerate(rows, start=2):
        if any(not (row.get(name) or "").strip() for name in expected_columns):
            raise ValueError(f"Missing value in CSV row {row_number}.")
        for name in NUMERICAL_FEATURES:
            try:
                value = float(row[name])
            except ValueError as error:
                raise ValueError(
                    f"Expected a numeric {name} value in CSV row {row_number}."
                ) from error
            if not np.isfinite(value):
                raise ValueError(f"Non-finite {name} value in CSV row {row_number}.")
    return rows, digest


def raw_matrix(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.empty((len(rows), len(FEATURE_ORDER)), dtype=object)
    for column_index, name in enumerate(FEATURE_ORDER):
        if name in CATEGORICAL_FEATURES:
            matrix[:, column_index] = [row[name] for row in rows]
        else:
            matrix[:, column_index] = [float(row[name]) for row in rows]
    labels = np.asarray([row[TARGET_COLUMN] for row in rows], dtype=str)
    return matrix, labels


def fit_feature_encoders(
    x_train: np.ndarray,
) -> tuple[dict[str, LabelEncoder], np.ndarray, StandardScaler]:
    encoded = x_train.copy()
    encoders: dict[str, LabelEncoder] = {}
    for column_index, name in enumerate(FEATURE_ORDER):
        if name in CATEGORICAL_FEATURES:
            encoder = LabelEncoder()
            encoded[:, column_index] = encoder.fit_transform(x_train[:, column_index].astype(str))
            encoders[name] = encoder
        else:
            encoded[:, column_index] = x_train[:, column_index].astype(np.float64)
    encoded = encoded.astype(np.float64)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(encoded).astype(np.float32)
    return encoders, scaled, scaler


def transform_features(
    x: np.ndarray,
    encoders: dict[str, LabelEncoder],
    scaler: StandardScaler,
) -> np.ndarray:
    encoded = x.copy()
    for column_index, name in enumerate(FEATURE_ORDER):
        if name in CATEGORICAL_FEATURES:
            encoder = encoders[name]
            values = x[:, column_index].astype(str)
            unseen = sorted(set(values) - set(encoder.classes_.tolist()))
            if unseen:
                raise ValueError(f"Unseen {name} categories in evaluation data: {unseen}")
            encoded[:, column_index] = encoder.transform(values)
        else:
            encoded[:, column_index] = x[:, column_index].astype(np.float64)
    return scaler.transform(encoded.astype(np.float64)).astype(np.float32)


def train(dataset_path: Path, artifact_dir: Path) -> dict:
    rows, dataset_sha256 = read_dataset(dataset_path)
    x, labels = raw_matrix(rows)
    raw_class_counts = {
        name: int(np.count_nonzero(labels == name)) for name in sorted(set(labels))
    }
    if len(raw_class_counts) < 3:
        raise ValueError("NEUTRA-BOOST requires at least three fertilizer classes.")

    x_train, x_test, labels_train, labels_test = train_test_split(
        x,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(labels_train)
    classes = label_encoder.classes_.tolist()
    if set(classes) != set(labels.tolist()):
        raise ValueError("The stratified training split did not retain every class.")
    y_test = label_encoder.transform(labels_test)

    categorical_encoders, x_train_scaled, scaler = fit_feature_encoders(x_train)
    x_test_scaled = transform_features(x_test, categorical_encoders, scaler)

    model = NeutraBoost(
        n_estimators=N_ESTIMATORS,
        base_lr=BASE_LR,
        max_depth=MAX_DEPTH,
        tau=TAU,
        margin_threshold=MARGIN_THRESHOLD,
        stability_weight=STABILITY_WEIGHT,
        min_spec_samples=MIN_SPECIALIST_SAMPLES,
        specialist_conf=SPECIALIST_CONF,
        use_class_weights=True,
        use_amadr=True,
    )
    train_started = time.perf_counter()
    model.fit(x_train_scaled, y_train)
    training_seconds = time.perf_counter() - train_started

    # The held-out test split is evaluated once. Notebook settings are fixed;
    # this phase does not tune parameters or sensitivity thresholds.
    result = model.predict_details(x_test_scaled)
    y_pred = result["predictions"]
    probabilities = np.asarray([detail["probabilities"] for detail in result["details"]])
    top3 = np.argsort(-probabilities, axis=1, kind="stable")[:, :3]
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "top_1_accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "top_3_accuracy": float(
            np.mean(np.any(top3 == y_test[:, None], axis=1))
        ),
        "class_wise": classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(classes)),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=np.arange(len(classes))
        ).tolist(),
        "confusion_matrix_labels": classes,
        "specialist_path_used_count": int(
            sum(detail["used_specialist"] for detail in result["details"])
        ),
        "confusion_gate_triggered_count": int(
            sum(detail["confusion_gate_triggered"] for detail in result["details"])
        ),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "neutra_boost.joblib", compress=3)
    joblib.dump(scaler, artifact_dir / "scaler.joblib", compress=3)
    category_mappings = {
        name: categorical_encoders[name].classes_.tolist()
        for name in CATEGORICAL_FEATURES
    }
    (artifact_dir / "category_mappings.json").write_text(
        json.dumps(category_mappings, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "label_mapping.json").write_text(
        json.dumps(classes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    train_class_counts = {
        name: int(np.count_nonzero(labels_train == name)) for name in classes
    }
    test_class_counts = {
        name: int(np.count_nonzero(labels_test == name)) for name in classes
    }
    specialist_pairs = [
        {
            "classes": [classes[a], classes[b]],
            "encoded_pair": [a, b],
            "training_samples": count,
        }
        for (a, b), count in sorted(model.specialist_sample_counts_.items())
    ]
    metadata = {
        "model_name": "NEUTRA-BOOST",
        "implementation_source": "supplied final-notebook (1).ipynb; custom NeutraBoost class",
        "dataset_filename": dataset_path.name,
        "dataset_sha256": dataset_sha256,
        "samples": len(rows),
        "target_column": TARGET_COLUMN,
        "feature_order": FEATURE_ORDER,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "classes": classes,
        "class_counts": raw_class_counts,
        "split": {
            "method": "stratified train_test_split",
            "test_size": 0.20,
            "random_state": RANDOM_STATE,
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "train_class_counts": train_class_counts,
            "test_class_counts": test_class_counts,
            "preprocessing_fit_on": "training split only",
        },
        "preprocessing": {
            "categorical_encoding": "one sklearn LabelEncoder per feature, fit only on training split",
            "categorical_feature_order": CATEGORICAL_FEATURES,
            "numeric_feature_conversion": "float64",
            "scaler": "StandardScaler fit on encoded training feature matrix; transforms all 19 features",
            "label_mapping": "LabelEncoder fit only on training target labels",
        },
        "model_parameters": {
            "n_estimators": N_ESTIMATORS,
            "base_lr": BASE_LR,
            "max_depth": MAX_DEPTH,
            "tau": TAU,
            "margin_threshold_gamma": MARGIN_THRESHOLD,
            "stability_weight_lambda": STABILITY_WEIGHT,
            "min_specialist_samples": MIN_SPECIALIST_SAMPLES,
            "specialist_confidence_delta": SPECIALIST_CONF,
            "class_weighting": "mean-normalized log(N / class_count)",
            "weak_learner": "DecisionTreeRegressor, min_samples_leaf=30, fixed random_state=42",
            "specialist_learner": "DecisionTreeClassifier, max_depth=3, min_samples_leaf=40, fixed random_state=42",
            "specialist_pairs": specialist_pairs,
        },
        "evaluation_method": "single stratified 20% held-out test; notebook hyperparameters fixed, no test-set tuning",
        "training_seconds": training_seconds,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "numpy_version": np.__version__,
            "scikit_learn_version": __import__("sklearn").__version__,
        },
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (artifact_dir / "evaluation.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"metadata": metadata, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/app/ml/fertilizer_artifacts"),
    )
    args = parser.parse_args()
    result = train(args.dataset.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
