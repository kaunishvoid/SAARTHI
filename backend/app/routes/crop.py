import math
from numbers import Real

from flask import Blueprint, current_app, jsonify, request

from ..ml.crop_model import CropModelService
from ..auth import authenticated_user, login_required
from ..extensions import db
from ..models import CropPrediction, as_utc_iso
from ..persistence import PersistenceUnavailable, persist_prediction


crop_bp = Blueprint("crop", __name__, url_prefix="/api/crop")
REQUIRED_FIELDS = (
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
)


def _validate(payload: object) -> tuple[dict[str, float] | None, dict[str, str]]:
    if not isinstance(payload, dict):
        return None, {"body": "must be a JSON object"}

    errors: dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors[field] = "is required"
        elif isinstance(payload[field], bool) or not isinstance(payload[field], Real):
            errors[field] = "must be a JSON number"
        elif not math.isfinite(float(payload[field])):
            errors[field] = "must be finite"

    extra_fields = sorted(set(payload) - set(REQUIRED_FIELDS))
    if extra_fields:
        errors["body"] = f"contains unsupported fields: {', '.join(extra_fields)}"

    if errors:
        return None, errors

    values = {field: float(payload[field]) for field in REQUIRED_FIELDS}
    for field in ("nitrogen", "phosphorus", "potassium"):
        if values[field] < 0:
            errors[field] = "must be non-negative"
    if not 0 <= values["ph"] <= 14:
        errors["ph"] = "must be between 0 and 14"
    if not 0 <= values["humidity"] <= 100:
        errors["humidity"] = "must be between 0 and 100"
    if values["rainfall"] < 0:
        errors["rainfall"] = "must be non-negative"
    return (None, errors) if errors else (values, {})


@crop_bp.post("/predict")
def predict_crop():
    values, errors = _validate(request.get_json(silent=True))
    if errors:
        return jsonify({
            "error": "validation_error",
            "message": "Invalid crop prediction input.",
            "details": errors,
        }), 422

    service = current_app.extensions.get("crop_model_service")
    if not isinstance(service, CropModelService):
        return jsonify({
            "error": "model_unavailable",
            "message": "The crop recommendation model is not available.",
        }), 503

    try:
        result = service.predict(values)
    except Exception:
        current_app.logger.exception("Crop prediction failed")
        return jsonify({
            "error": "prediction_failed",
            "message": "The crop prediction could not be completed.",
        }), 500
    try:
        prediction_id = persist_prediction("crop", values, result)
    except PersistenceUnavailable:
        current_app.logger.exception("Crop prediction could not be persisted")
        return jsonify({"error": "persistence_unavailable", "message": "Sign-in prediction could not be saved; retry after database recovery."}), 503
    if prediction_id is not None:
        result["prediction_id"] = prediction_id
    return jsonify(result), 200


@crop_bp.get("/predictions")
@login_required
def crop_predictions():
    user = authenticated_user()
    limit = request.args.get("limit", default=50, type=int)
    limit = min(max(limit or 50, 1), 100)
    rows = CropPrediction.query.filter_by(user_id=user.id).order_by(CropPrediction.created_at.desc()).limit(limit).all()
    return jsonify({"items": [{"id": row.id, "inputs": row.inputs, "result": row.result, "created_at": as_utc_iso(row.created_at)} for row in rows]}), 200


@crop_bp.get("/predictions/<int:prediction_id>")
@login_required
def crop_prediction_detail(prediction_id: int):
    user = authenticated_user()
    row = CropPrediction.query.filter_by(id=prediction_id, user_id=user.id).first()
    if row is None:
        return jsonify({"error": "not_found", "message": "Prediction not found."}), 404
    return jsonify({"id": row.id, "inputs": row.inputs, "result": row.result, "created_at": as_utc_iso(row.created_at)}), 200
