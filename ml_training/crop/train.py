"""Train the SAARTHI crop ANN from the explicitly supplied CSV dataset."""

import argparse
import csv
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backend.app.ml.crop_model import CropANN, FEATURE_ORDER


TARGET_COLUMN = "label"
SEED = 42
HIDDEN_SIZES = (128, 64)
DROPOUT = 0.1
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 500
PATIENCE = 40
MIN_DELTA = 1e-5


def read_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Crop dataset not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        expected = FEATURE_ORDER + [TARGET_COLUMN]
        if columns is None or set(columns) != set(expected) or len(columns) != len(expected):
            raise ValueError(f"Expected exactly these CSV columns: {expected}; found: {columns}")
        rows = list(reader)

    if not rows:
        raise ValueError("Crop dataset has no records.")
    for line, row in enumerate(rows, start=2):
        if any(not (row.get(column) or "").strip() for column in expected):
            raise ValueError(f"Missing value in CSV row {line}.")

    try:
        features = np.asarray(
            [[float(row[name]) for name in FEATURE_ORDER] for row in rows],
            dtype=np.float32,
        )
    except ValueError as error:
        raise ValueError("All seven crop features must contain numeric values.") from error
    if not np.isfinite(features).all():
        raise ValueError("Crop feature values must all be finite.")

    labels = np.asarray([row[TARGET_COLUMN].strip() for row in rows], dtype=str)
    if any(not label for label in labels):
        raise ValueError("Crop labels must not be blank.")
    return features, labels, digest


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def evaluate_loss(model: CropANN, features: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.inference_mode():
        loss = nn.functional.cross_entropy(model(features), labels)
    return float(loss.item())


def train_model(
    model: CropANN,
    train_loader: DataLoader,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> tuple[int, float, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            batch_size = batch_labels.size(0)
            total_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_loss = total_loss / seen
        val_loss = evaluate_loss(model, x_val, y_val)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best_loss - MIN_DELTA:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} validation_loss={val_loss:.6f}", flush=True)
        if epochs_without_improvement >= PATIENCE:
            print(f"early stopping at epoch {epoch}; best validation epoch {best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid validation checkpoint.")
    model.load_state_dict(best_state)
    return best_epoch, best_loss, history


def train(dataset_path: Path, artifact_dir: Path, seed: int = SEED) -> dict:
    generator = seed_everything(seed)
    features, labels, dataset_sha256 = read_dataset(dataset_path)
    class_counts = {name: int((labels == name).sum()) for name in sorted(set(labels))}
    if len(class_counts) < 3:
        raise ValueError("At least three crop classes are required for Top-3 recommendations.")

    x_train_val, x_test, y_train_val, y_test_text = train_test_split(
        features,
        labels,
        test_size=0.15,
        random_state=seed,
        stratify=labels,
    )
    x_train, x_val, y_train_text, y_val_text = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=y_train_val,
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    classes = label_encoder.classes_.tolist()
    if set(classes) != set(labels.tolist()):
        raise ValueError("Stratified training split did not retain every dataset crop class.")
    y_val = label_encoder.transform(y_val_text)
    y_test = label_encoder.transform(y_test_text)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_val_scaled = scaler.transform(x_val).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(x_train_scaled),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    x_val_tensor = torch.from_numpy(x_val_scaled)
    y_val_tensor = torch.from_numpy(y_val.astype(np.int64))
    model = CropANN(len(FEATURE_ORDER), len(classes), HIDDEN_SIZES, DROPOUT)

    training_started = time.perf_counter()
    best_epoch, best_validation_loss, history = train_model(
        model, train_loader, x_val_tensor, y_val_tensor
    )
    training_seconds = time.perf_counter() - training_started

    # The test set is consulted once after early stopping; it is not used for model selection.
    model.eval()
    with torch.inference_mode():
        test_logits = model(torch.from_numpy(x_test_scaled))
        probabilities = torch.softmax(test_logits, dim=1).cpu().numpy()
    y_pred = probabilities.argmax(axis=1)
    top3_indices = np.argsort(-probabilities, axis=1, kind="stable")[:, :3]
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "top_1_accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "top_3_accuracy": float(np.mean(np.any(top3_indices == y_test[:, None], axis=1))),
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
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "crop_model.pt")
    joblib.dump(scaler, artifact_dir / "scaler.joblib")
    (artifact_dir / "label_mapping.json").write_text(
        json.dumps(classes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    split_counts = {"train": len(x_train), "validation": len(x_val), "test": len(x_test)}
    per_split_class_counts = {
        "train": {name: int((y_train_text == name).sum()) for name in classes},
        "validation": {name: int((y_val_text == name).sum()) for name in classes},
        "test": {name: int((y_test_text == name).sum()) for name in classes},
    }
    metadata = {
        "model_name": "SAARTHI crop feed-forward ANN",
        "dataset_filename": dataset_path.name,
        "dataset_sha256": dataset_sha256,
        "samples": int(len(labels)),
        "input_features": int(len(FEATURE_ORDER)),
        "feature_order": FEATURE_ORDER,
        "target_column": TARGET_COLUMN,
        "classes": classes,
        "class_counts": class_counts,
        "split_counts": split_counts,
        "split_class_counts": per_split_class_counts,
        "seed": seed,
        "architecture": {
            "input_size": len(FEATURE_ORDER),
            "hidden_sizes": list(HIDDEN_SIZES),
            "activation": "ReLU",
            "dropout": DROPOUT,
            "output_size": len(classes),
            "output_for_inference": "softmax probabilities",
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "cross_entropy_with_logits",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping": "validation loss; restore best validation checkpoint",
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "training_seconds": training_seconds,
            "device": "cpu",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "scaler": "StandardScaler fitted only on the training split",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (artifact_dir / "evaluation.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (artifact_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )

    report = {
        "metadata": metadata,
        "metrics": metrics,
        "final_epoch": len(history),
        "dataset_path": str(dataset_path),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Path to supplied crop CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/app/ml/artifacts"),
        help="Directory for trained model and evaluation artifacts",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    result = train(args.dataset.resolve(), args.output_dir.resolve(), args.seed)
    print(json.dumps(result["metadata"], indent=2), flush=True)
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
