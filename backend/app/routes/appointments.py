"""Farmer consultation requests and officer response workflow."""

from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from sqlalchemy import and_, or_

from ..auth import authenticated_user, roles_required
from ..extensions import db
from ..models import Appointment


appointments_bp = Blueprint("appointments", __name__, url_prefix="/api/appointments")


def _parse_requested_for(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    parsed = parsed.astimezone(timezone.utc)
    if parsed <= datetime.now(timezone.utc):
        return None
    return parsed.replace(tzinfo=None)  # stored as UTC; SQLite does not preserve tzinfo


@appointments_bp.post("")
@roles_required("farmer")
def create_appointment():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or set(payload) - {"requested_for", "topic", "notes"}:
        return jsonify({"error": "validation_error", "message": "Provide requested_for, topic, and optional notes."}), 422
    requested_for = _parse_requested_for(payload.get("requested_for"))
    topic = payload.get("topic")
    notes = payload.get("notes", "")
    errors = {}
    if requested_for is None:
        errors["requested_for"] = "must be a future ISO-8601 datetime with timezone"
    if not isinstance(topic, str) or not 3 <= len(topic.strip()) <= 120:
        errors["topic"] = "must contain 3 to 120 characters"
    if not isinstance(notes, str) or len(notes) > 1000:
        errors["notes"] = "must be a string of at most 1000 characters"
    if errors:
        return jsonify({"error": "validation_error", "message": "Invalid appointment request.", "details": errors}), 422
    user = authenticated_user()
    appointment = Appointment(farmer_id=user.id, requested_for=requested_for, topic=topic.strip(), notes=notes.strip(), status="requested", is_demo=user.is_demo)
    db.session.add(appointment)
    db.session.commit()
    return jsonify({
        "appointment": appointment.as_dict(),
        "notice": "Request recorded; no officer or availability was represented as confirmed.",
    }), 201


@appointments_bp.get("")
def list_appointments():
    user = authenticated_user()
    if user is None:
        return jsonify({"error": "authentication_required", "message": "Please sign in to view appointments."}), 401
    if user.role == "farmer":
        rows = Appointment.query.filter_by(farmer_id=user.id).order_by(Appointment.created_at.desc()).limit(100).all()
    elif user.role == "officer":
        rows = Appointment.query.filter(or_(Appointment.status == "requested", Appointment.officer_id == user.id)).order_by(Appointment.requested_for.asc()).limit(100).all()
    else:
        return jsonify({"error": "forbidden", "message": "This account role cannot view appointments."}), 403
    return jsonify({"items": [row.as_dict() for row in rows]}), 200


@appointments_bp.patch("/<int:appointment_id>")
def update_appointment(appointment_id: int):
    user = authenticated_user()
    if user is None:
        return jsonify({"error": "authentication_required", "message": "Please sign in to update appointments."}), 401
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or set(payload) != {"action"}:
        return jsonify({"error": "validation_error", "message": "Provide one action."}), 422
    appointment = db.session.get(Appointment, appointment_id)
    if appointment is None:
        return jsonify({"error": "not_found", "message": "Appointment request not found."}), 404
    action = payload["action"]
    if user.role == "farmer" and appointment.farmer_id == user.id and action == "cancel" and appointment.status == "requested":
        appointment.status = "cancelled"
    elif user.role == "officer" and appointment.status == "requested" and action in {"accept", "decline"}:
        if action == "accept":
            conflict = Appointment.query.filter(
                Appointment.officer_id == user.id,
                Appointment.status == "accepted",
                Appointment.requested_for == appointment.requested_for,
            ).first()
            if conflict:
                return jsonify({"error": "appointment_conflict", "message": "You already accepted a request for this time."}), 409
            appointment.status = "accepted"
            appointment.officer_id = user.id
        else:
            appointment.status = "declined"
            appointment.officer_id = user.id
    else:
        return jsonify({"error": "forbidden", "message": "This action is not allowed for your role or the current status."}), 403
    db.session.commit()
    return jsonify({"appointment": appointment.as_dict()}), 200
