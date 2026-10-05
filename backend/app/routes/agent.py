"""Validated chat endpoint for the SAARTHI consultancy orchestrator."""

import re
import uuid
import math
from numbers import Real

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy.exc import SQLAlchemyError

from ..agent.service import ConsultancyService
from ..auth import authenticated_user
from ..extensions import db
from ..ml.crop_model import CropModelService
from ..ml.fertilizer_model import CATEGORICAL_FEATURES, FEATURE_ORDER
from ..ml.fertilizer_service import FertilizerModelService
from ..models import Conversation, Message, utc_now
from .crop import _validate as validate_crop_inputs


agent_bp = Blueprint("agent", __name__, url_prefix="/api/agent")
_CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


@agent_bp.post("/chat")
def chat():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "validation_error", "message": "Request body must be a JSON object."}), 422
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip() or len(message) > 2000:
        return jsonify({
            "error": "validation_error",
            "message": "message must be a non-empty string of at most 2000 characters.",
            "details": {"message": "invalid message"},
        }), 422
    conversation_id = payload.get("conversation_id")
    if conversation_id is None:
        conversation_id = uuid.uuid4().hex
    elif not isinstance(conversation_id, str) or not _CONVERSATION_ID.fullmatch(conversation_id):
        return jsonify({"error": "validation_error", "message": "conversation_id has an invalid format."}), 422
    extra = set(payload) - {"message", "conversation_id", "context"}
    if extra:
        return jsonify({
            "error": "validation_error",
            "message": "Request contains unsupported fields.",
            "details": {"fields": sorted(extra)},
        }), 422

    service = current_app.extensions.get("consultancy_service")
    if not isinstance(service, ConsultancyService):
        return jsonify({"error": "service_unavailable", "message": "Consultancy service is unavailable."}), 503
    supplied_context = payload.get("context")
    trusted_context = {}
    if supplied_context is not None and not isinstance(supplied_context, dict):
        return jsonify({"error": "validation_error", "message": "context must be a JSON object."}), 422
    supplied_context = supplied_context or {}
    unknown_context = set(supplied_context) - {"crop_inputs", "fertilizer_inputs"}
    if unknown_context:
        return jsonify({
            "error": "validation_error",
            "message": "Context contains unsupported fields.",
            "details": {"fields": sorted(unknown_context)},
        }), 422

    if "crop_inputs" in supplied_context:
        crop_values, errors = validate_crop_inputs(supplied_context["crop_inputs"])
        model = current_app.extensions.get("crop_model_service")
        if errors:
            return jsonify({"error": "validation_error", "message": "Invalid crop context.", "details": errors}), 422
        if isinstance(model, CropModelService):
            try:
                trusted_context["crop_prediction"] = model.predict(crop_values)
            except Exception:
                current_app.logger.exception("Agent crop context prediction failed")
                return jsonify({"error": "prediction_failed", "message": "Crop context could not be evaluated."}), 500

    if "fertilizer_inputs" in supplied_context:
        values = supplied_context["fertilizer_inputs"]
        model = current_app.extensions.get("fertilizer_model_service")
        errors = {}
        if not isinstance(values, dict):
            errors["fertilizer_inputs"] = "must be a JSON object"
        else:
            missing = set(FEATURE_ORDER) - set(values)
            extra = set(values) - set(FEATURE_ORDER)
            for name in missing:
                errors[name] = "is required"
            if extra:
                errors["fertilizer_inputs"] = f"unsupported fields: {', '.join(sorted(extra))}"
            for name in set(FEATURE_ORDER) & set(values):
                value = values[name]
                if name in CATEGORICAL_FEATURES:
                    if not isinstance(value, str) or (isinstance(model, FertilizerModelService) and value not in model.categories[name]):
                        errors[name] = "must match a category from the fertilizer schema"
                elif isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                    errors[name] = "must be a finite JSON number"
        if errors:
            return jsonify({"error": "validation_error", "message": "Invalid fertilizer context.", "details": errors}), 422
        if isinstance(model, FertilizerModelService):
            try:
                trusted_context["fertilizer_prediction"] = model.predict(values)
            except Exception:
                current_app.logger.exception("Agent fertilizer context prediction failed")
                return jsonify({"error": "prediction_failed", "message": "Fertilizer context could not be evaluated."}), 500

    user = authenticated_user()
    conversation = None
    if user is not None:
        conversation = db.session.get(Conversation, conversation_id)
        if conversation is not None and conversation.user_id != user.id:
            return jsonify({"error": "not_found", "message": "Conversation not found."}), 404
        if conversation is None:
            conversation = Conversation(id=conversation_id, user_id=user.id)
            db.session.add(conversation)
        conversation.updated_at = utc_now()
    result = service.respond(message.strip(), conversation_id, trusted_context)
    result["conversation_persisted"] = user is not None
    if user is not None:
        db.session.add(Message(conversation_id=conversation_id, role="user", content=message.strip(), metadata_json={}))
        db.session.add(Message(
            conversation_id=conversation_id,
            role="assistant",
            content=result["response"],
            metadata_json={"intent": result["intent"], "sources": result["sources"]},
        ))
        try:
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
            current_app.logger.exception("Consultancy conversation could not be persisted")
            return jsonify({"error": "persistence_unavailable", "message": "The assistant response could not be saved."}), 503
    return jsonify(result), 200
