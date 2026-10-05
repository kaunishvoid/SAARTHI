"""NEUTRA-BOOST as implemented in the supplied notebook, with inference details."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import softmax
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


FEATURE_ORDER = [
    "Soil_Type",
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Nitrogen_Level",
    "Phosphorus_Level",
    "Potassium_Level",
    "Temperature",
    "Humidity",
    "Rainfall",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Previous_Crop",
    "Region",
    "Fertilizer_Used_Last_Season",
    "Yield_Last_Season",
]
CATEGORICAL_FEATURES = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Previous_Crop",
    "Region",
]
NUMERICAL_FEATURES = [name for name in FEATURE_ORDER if name not in CATEGORICAL_FEATURES]
TARGET_COLUMN = "Recommended_Fertilizer"

# Values are fixed from the supplied notebook's final sensitivity-selected configuration.
N_ESTIMATORS = 60
BASE_LR = 0.15
MAX_DEPTH = 3
TAU = 0.25
MARGIN_THRESHOLD = 0.035
STABILITY_WEIGHT = 0.0
MIN_SPECIALIST_SAMPLES = 100
SPECIALIST_CONF = 0.55
RANDOM_STATE = 42


class NeutraBoost:
    """Newton-style multiclass boosting plus AMADR-gated pair specialists."""

    def __init__(
        self,
        n_estimators: int = N_ESTIMATORS,
        base_lr: float = BASE_LR,
        max_depth: int = MAX_DEPTH,
        tau: float = TAU,
        margin_threshold: float = MARGIN_THRESHOLD,
        stability_weight: float = STABILITY_WEIGHT,
        min_spec_samples: int = MIN_SPECIALIST_SAMPLES,
        specialist_conf: float = SPECIALIST_CONF,
        use_class_weights: bool = True,
        use_amadr: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.base_lr = base_lr
        self.max_depth = max_depth
        self.tau = tau
        self.margin_threshold = margin_threshold
        self.stability_weight = stability_weight
        self.min_spec_samples = min_spec_samples
        self.specialist_conf = specialist_conf
        self.use_class_weights = use_class_weights
        self.use_amadr = use_amadr
        self.models_: list[list[DecisionTreeRegressor]] = []
        self.K_: int | None = None
        self.stability_: np.ndarray | None = None
        self.specialists_: dict[tuple[int, int], DecisionTreeClassifier] = {}
        self.specialist_sample_counts_: dict[tuple[int, int], int] = {}
        self.class_weights_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeutraBoost":
        n = X.shape[0]
        self.K_ = len(np.unique(y))
        logits = np.zeros((n, self.K_), dtype=np.float64)
        self.models_ = []

        # Notebook Contribution 2: mean-normalized log-frequency weights.
        counts = np.bincount(y, minlength=self.K_).astype(float)
        raw_weights = np.log(n / counts)
        self.class_weights_ = raw_weights / raw_weights.mean()
        sample_weights = (
            self.class_weights_[y] if self.use_class_weights else np.ones(n)
        )

        # Notebook Contribution 1: margin-adaptive diagonal-Newton tree updates.
        for _ in range(self.n_estimators):
            probabilities = softmax(logits, axis=1)
            top2 = np.partition(probabilities, -2, axis=1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]
            margin_weights = np.clip(np.exp(-margin / self.tau), 0.05, 1.0)
            combined_weights = margin_weights * sample_weights

            gradients = probabilities.copy()
            gradients[np.arange(n), y] -= 1
            gradients *= combined_weights[:, None]
            hessians = np.clip(probabilities * (1 - probabilities), 1e-3, None)

            round_trees: list[DecisionTreeRegressor] = []
            for class_index in range(self.K_):
                target = -gradients[:, class_index] / hessians[:, class_index]
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=30,
                    random_state=RANDOM_STATE,
                )
                tree.fit(X, target)
                logits[:, class_index] += self.base_lr * tree.predict(X)
                round_trees.append(tree)
            self.models_.append(round_trees)

        # Notebook Contribution 3: rescaled class stability values.
        probabilities_train = softmax(logits, axis=1)
        top2_train = np.partition(probabilities_train, -2, axis=1)[:, -2:]
        train_margins = top2_train[:, 1] - top2_train[:, 0]
        stability = np.zeros(self.K_, dtype=float)
        stability_counts = np.zeros(self.K_, dtype=float)
        for row_index, class_index in enumerate(y):
            stability[class_index] += train_margins[row_index]
            stability_counts[class_index] += 1
        stability /= stability_counts + 1e-9
        stability = (
            (stability - stability.min())
            / (stability.max() - stability.min() + 1e-9)
            * 0.2
        )
        self.stability_ = stability

        # Notebook AMADR specialists: unordered top-two pairs are row-sorted.
        top2_indices = np.argsort(probabilities_train, axis=1)[:, -2:]
        sorted_pairs = np.sort(top2_indices, axis=1)
        self.specialists_ = {}
        self.specialist_sample_counts_ = {}
        for pair_values in np.unique(sorted_pairs, axis=0):
            a, b = (int(pair_values[0]), int(pair_values[1]))
            indices = np.where(
                (sorted_pairs[:, 0] == a) & (sorted_pairs[:, 1] == b)
            )[0]
            if len(indices) < self.min_spec_samples:
                continue
            # Binary label is 1 when the true class is a; other labels remain 0,
            # exactly as in the source notebook's specialist training mechanism.
            binary_target = (y[indices] == a).astype(int)
            specialist = DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=40,
                random_state=RANDOM_STATE,
            )
            specialist.fit(X[indices], binary_target)
            self.specialists_[(a, b)] = specialist
            self.specialist_sample_counts_[(a, b)] = int(len(indices))
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        if self.K_ is None:
            raise RuntimeError("NEUTRA-BOOST has not been fitted.")
        logits = np.zeros((X.shape[0], self.K_), dtype=np.float64)
        for round_trees in self.models_:
            for class_index, tree in enumerate(round_trees):
                logits[:, class_index] += self.base_lr * tree.predict(X)
        return logits

    def predict_details(self, X: np.ndarray) -> dict[str, Any]:
        probabilities = softmax(self.predict_logits(X), axis=1)
        rank = np.argsort(probabilities, axis=1)
        top1 = rank[:, -1]
        top2 = rank[:, -2]
        top3 = rank[:, -3]
        final = top1.copy()
        details: list[dict[str, Any]] = []

        for row_index, row_probabilities in enumerate(probabilities):
            margin = float(row_probabilities[top1[row_index]] - row_probabilities[top2[row_index]])
            gate_triggered = bool(self.use_amadr and margin < self.margin_threshold)
            used_specialist = False
            specialist_confidence: float | None = None
            specialist_pair: tuple[int, int] | None = None

            if gate_triggered:
                candidates = [int(top1[row_index]), int(top2[row_index]), int(top3[row_index])]
                scores = [
                    float(row_probabilities[candidate])
                    + self.stability_weight * float(self.stability_[candidate])
                    for candidate in candidates
                ]
                chosen = candidates[int(np.argmax(scores))]
                specialist_pair = tuple(sorted((int(top1[row_index]), int(top2[row_index]))))
                if specialist_pair in self.specialists_:
                    specialist_probabilities = self.specialists_[specialist_pair].predict_proba(
                        X[row_index].reshape(1, -1)
                    )[0]
                    specialist_confidence = float(np.max(specialist_probabilities))
                    if specialist_confidence > self.specialist_conf:
                        decision = int(np.argmax(specialist_probabilities))
                        chosen = specialist_pair[1 - decision]
                        used_specialist = True
                final[row_index] = chosen

            predicted_index = int(final[row_index])
            details.append({
                "predicted_index": predicted_index,
                "confidence": (
                    specialist_confidence
                    if used_specialist
                    else float(row_probabilities[predicted_index])
                ),
                "confidence_source": (
                    "confusion-pair specialist"
                    if used_specialist
                    else "base softmax probability"
                ),
                "specialist_confidence": specialist_confidence,
                "used_specialist": used_specialist,
                "confusion_gate_triggered": gate_triggered,
                "specialist_pair": list(specialist_pair) if specialist_pair else None,
                "probabilities": [float(value) for value in row_probabilities],
            })
        return {"predictions": final, "details": details}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_details(X)["predictions"]

    def predict_topk(self, X: np.ndarray, k: int = 3) -> np.ndarray:
        probabilities = softmax(self.predict_logits(X), axis=1)
        return np.argsort(probabilities, axis=1)[:, -k:]
