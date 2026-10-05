"""Validated NEUTRA-BOOST API endpoints."""

import math
from numbers import Real

from flask import Blueprint, current_app, jsonify, request

from ..ml.fertilizer_model import CATEGORICAL_FEATURES, FEATURE_ORDER
from ..ml.fertilizer_service import FertilizerModelService
from ..auth import authenticated_user, login_required
from ..models import FertilizerPrediction, as_utc_iso
from ..persistence import PersistenceUnavailable, persist_prediction


fertilizer_bp = Blueprint("fertilizer", __name__, url_prefix="/api/fertilizer")


def _service() -> FertilizerModelService | None:
    service = current_app.extensions.get("fertilizer_model_service")
    return service if isinstance(service, FertilizerModelService) else None


def _unavailable():
    return jsonify({
        "error": "model_unavailable",
        "message": "The NEUTRA-BOOST fertilizer model is not available.",
    }), 503


@fertilizer_bp.get("/schema")
def fertilizer_schema():
    service = _service()
    if service is None:
        return _unavailable()
    return jsonify(service.schema()), 200


@fertilizer_bp.post("/predict")
def predict_fertilizer():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({
            "error": "validation_error",
            "message": "Request body must be a JSON object with all 19 model inputs.",
            "details": {"body": "must be a JSON object"},
        }), 422

    errors: dict[str, str] = {}
    required = set(FEATURE_ORDER)
    missing = sorted(required - set(payload))
    extra = sorted(set(payload) - required)
    for name in missing:
        errors[name] = "is required"
    if extra:
        errors["body"] = f"contains unsupported fields: {', '.join(extra)}"

    service = _service()
    for name in FEATURE_ORDER:
        if name not in payload:
            continue
        value = payload[name]
        if name in CATEGORICAL_FEATURES:
            if not isinstance(value, str):
                errors[name] = "must be a string category"
            elif service is not None and value not in service.categories[name]:
                errors[name] = "must match one of the categories returned by /api/fertilizer/schema"
        elif isinstance(value, bool) or not isinstance(value, Real):
            errors[name] = "must be a JSON number"
        elif not math.isfinite(float(value)):
            errors[name] = "must be finite"

    if errors:
        return jsonify({
            "error": "validation_error",
            "message": "Invalid fertilizer prediction input.",
            "details": errors,
        }), 422
    if service is None:
        return _unavailable()

    try:
        result = service.predict(payload)
    except Exception:
        current_app.logger.exception("NEUTRA-BOOST prediction failed")
        return jsonify({
            "error": "prediction_failed",
            "message": "The fertilizer prediction could not be completed.",
        }), 500
    try:
        prediction_id = persist_prediction("fertilizer", payload, result)
    except PersistenceUnavailable:
        current_app.logger.exception("Fertilizer prediction could not be persisted")
        return jsonify({"error": "persistence_unavailable", "message": "Sign-in prediction could not be saved; retry after database recovery."}), 503
    if prediction_id is not None:
        result["prediction_id"] = prediction_id
    return jsonify(result), 200


@fertilizer_bp.get("/predictions")
@login_required
def fertilizer_predictions():
    user = authenticated_user()
    limit = request.args.get("limit", default=50, type=int)
    limit = min(max(limit or 50, 1), 100)
    rows = FertilizerPrediction.query.filter_by(user_id=user.id).order_by(FertilizerPrediction.created_at.desc()).limit(limit).all()
    return jsonify({"items": [{"id": row.id, "inputs": row.inputs, "result": row.result, "created_at": as_utc_iso(row.created_at)} for row in rows]}), 200


@fertilizer_bp.get("/predictions/<int:prediction_id>")
@login_required
def fertilizer_prediction_detail(prediction_id: int):
    user = authenticated_user()
    row = FertilizerPrediction.query.filter_by(id=prediction_id, user_id=user.id).first()
    if row is None:
        return jsonify({"error": "not_found", "message": "Prediction not found."}), 404
    return jsonify({"id": row.id, "inputs": row.inputs, "result": row.result, "created_at": as_utc_iso(row.created_at)}), 200
