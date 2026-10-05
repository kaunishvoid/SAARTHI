import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn


FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
REQUEST_TO_FEATURE = {
    "nitrogen": "N",
    "phosphorus": "P",
    "potassium": "K",
    "temperature": "temperature",
    "humidity": "humidity",
    "ph": "ph",
    "rainfall": "rainfall",
}


class CropANN(nn.Module):
    """Tabular feed-forward ANN. It returns logits; inference applies softmax."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_sizes: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        first, second = hidden_sizes
        self.network = nn.Sequential(
            nn.Linear(input_size, first),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class CropModelUnavailable(RuntimeError):
    """Raised when the trained crop artifacts are missing or inconsistent."""


class CropModelService:
    def __init__(self, artifact_dir: Path | str) -> None:
        self.artifact_dir = Path(artifact_dir)
        artifact_dir = self.artifact_dir
        metadata_path = artifact_dir / "metadata.json"
        labels_path = artifact_dir / "label_mapping.json"
        scaler_path = artifact_dir / "scaler.joblib"
        weights_path = artifact_dir / "crop_model.pt"
        required = [metadata_path, labels_path, scaler_path, weights_path]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise CropModelUnavailable(f"Missing crop model artifacts: {', '.join(missing)}")

        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.classes: list[str] = json.loads(labels_path.read_text(encoding="utf-8"))
        if self.metadata.get("feature_order") != FEATURE_ORDER:
            raise CropModelUnavailable("Crop artifact feature order does not match the API contract.")
        if self.metadata.get("classes") != self.classes or len(self.classes) < 3:
            raise CropModelUnavailable("Crop artifact class mapping is missing or inconsistent.")

        self.scaler = joblib.load(scaler_path)
        architecture = self.metadata["architecture"]
        self.model = CropANN(
            input_size=len(FEATURE_ORDER),
            num_classes=len(self.classes),
            hidden_sizes=tuple(architecture["hidden_sizes"]),
            dropout=float(architecture["dropout"]),
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, request_values: dict[str, float]) -> dict[str, Any]:
        by_feature = {
            feature: float(request_values[request_key])
            for request_key, feature in REQUEST_TO_FEATURE.items()
        }
        ordered = np.asarray([[by_feature[name] for name in FEATURE_ORDER]], dtype=np.float32)
        scaled = self.scaler.transform(ordered).astype(np.float32, copy=False)
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(scaled))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Stable descending ordering keeps class-map order as the tie-breaker.
        ranked = sorted(range(len(self.classes)), key=lambda i: (-float(probabilities[i]), i))[:3]
        top_3 = [
            {"crop": self.classes[i], "probability": float(probabilities[i])}
            for i in ranked
        ]
        return {
            "predicted_crop": top_3[0]["crop"],
            "confidence": top_3[0]["probability"],
            "top_3": top_3,
            "caution": (
                "These are model predictions based on the supplied training dataset; "
                "they are not guaranteed agricultural outcomes."
            ),
        }
