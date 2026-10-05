"""Production loader and prediction adapter for saved NEUTRA-BOOST artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .fertilizer_model import (
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    NeutraBoost,
)


class FertilizerModelUnavailable(RuntimeError):
    """Raised when NEUTRA-BOOST artifacts are missing or inconsistent."""


class FertilizerModelService:
    def __init__(self, artifact_dir: Path | str) -> None:
        self.artifact_dir = Path(artifact_dir)
        files = {
            "metadata": self.artifact_dir / "metadata.json",
            "categories": self.artifact_dir / "category_mappings.json",
            "labels": self.artifact_dir / "label_mapping.json",
            "scaler": self.artifact_dir / "scaler.joblib",
            "model": self.artifact_dir / "neutra_boost.joblib",
        }
        missing = [name for name, path in files.items() if not path.is_file()]
        if missing:
            raise FertilizerModelUnavailable(
                f"Missing NEUTRA-BOOST artifacts: {', '.join(missing)}"
            )

        self.metadata: dict[str, Any] = json.loads(files["metadata"].read_text(encoding="utf-8"))
        self.categories: dict[str, list[str]] = json.loads(
            files["categories"].read_text(encoding="utf-8")
        )
        self.classes: list[str] = json.loads(files["labels"].read_text(encoding="utf-8"))
        if self.metadata.get("feature_order") != FEATURE_ORDER:
            raise FertilizerModelUnavailable("Fertilizer model feature order does not match API schema.")
        if self.metadata.get("classes") != self.classes or len(self.classes) < 3:
            raise FertilizerModelUnavailable("Fertilizer class mapping is missing or inconsistent.")
        if set(self.categories) != set(CATEGORICAL_FEATURES):
            raise FertilizerModelUnavailable("Categorical feature mappings are incomplete.")
        if any(not isinstance(values, list) or not values for values in self.categories.values()):
            raise FertilizerModelUnavailable("A categorical mapping is empty or invalid.")
        if any(values != sorted(set(values)) for values in self.categories.values()):
            raise FertilizerModelUnavailable("Categorical mappings are not in LabelEncoder order.")
        self.category_indices = {
            name: {category: index for index, category in enumerate(values)}
            for name, values in self.categories.items()
        }

        self.scaler = joblib.load(files["scaler"])
        self.model: NeutraBoost = joblib.load(files["model"])
        if self.scaler.n_features_in_ != len(FEATURE_ORDER):
            raise FertilizerModelUnavailable("Fertilizer scaler input width is invalid.")
        if self.model.K_ != len(self.classes):
            raise FertilizerModelUnavailable("Fertilizer model output width is invalid.")

    def schema(self) -> dict[str, Any]:
        fields = []
        for name in FEATURE_ORDER:
            if name in CATEGORICAL_FEATURES:
                fields.append({
                    "name": name,
                    "type": "categorical",
                    "options": self.categories[name],
                })
            else:
                fields.append({"name": name, "type": "number"})
        return {"input_fields": fields, "target": "Recommended_Fertilizer", "classes": self.classes}

    def _transform(self, values: dict[str, Any]) -> np.ndarray:
        encoded: list[float] = []
        for name in FEATURE_ORDER:
            value = values[name]
            if name in CATEGORICAL_FEATURES:
                encoded.append(float(self.category_indices[name][value]))
            else:
                encoded.append(float(value))
        return self.scaler.transform(np.asarray([encoded], dtype=np.float64)).astype(np.float32)

    def predict(self, values: dict[str, Any]) -> dict[str, Any]:
        features = self._transform(values)
        result = self.model.predict_details(features)
        detail = result["details"][0]
        predicted_index = detail["predicted_index"]
        probabilities = detail["probabilities"]
        ranked_indices = sorted(
            range(len(self.classes)), key=lambda index: (-probabilities[index], index)
        )
        return {
            "fertilizer": self.classes[predicted_index],
            "confidence": detail["confidence"],
            "confidence_source": detail["confidence_source"],
            "used_specialist": detail["used_specialist"],
            "confusion_gate_triggered": detail["confusion_gate_triggered"],
            "specialist_pair": (
                [self.classes[index] for index in detail["specialist_pair"]]
                if detail["specialist_pair"] is not None
                else None
            ),
            "ranked_probabilities": [
                {"fertilizer": self.classes[index], "probability": probabilities[index]}
                for index in ranked_indices
            ],
            "top_3": [
                {"fertilizer": self.classes[index], "probability": probabilities[index]}
                for index in ranked_indices[:3]
            ],
            "caution": (
                "NEUTRA-BOOST predictions and scores reflect the supplied dataset and are not "
                "a fertilizer dosage or a guarantee of agricultural outcome."
            ),
        }
