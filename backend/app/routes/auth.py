"""Farmer authentication with signed, HTTP-only Flask sessions."""

import re
import secrets

from flask import Blueprint, current_app, jsonify, request, session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from werkzeug.security import check_password_hash, generate_password_hash

from ..auth import authenticated_user
from ..extensions import db
from ..models import User


auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")
users_bp = Blueprint("users", __name__, url_prefix="/api/users")
_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _rotate_csrf() -> str:
    token = secrets.token_urlsafe(32)
    session["csrf_token"] = token
    return token


def _user_response(user: User, csrf_token: str | None = None):
    body = {"user": user.public_dict()}
    if csrf_token:
        body["csrf_token"] = csrf_token
    return jsonify(body), 200


@auth_bp.get("/csrf")
def csrf_token():
    token = session.get("csrf_token")
    if not isinstance(token, str):
        token = _rotate_csrf()
    return jsonify({"csrf_token": token}), 200


@auth_bp.post("/register")
def register():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "validation_error", "message": "Request body must be a JSON object."}), 422
    extra = set(payload) - {"full_name", "email", "password"}
    if extra:
        return jsonify({"error": "validation_error", "message": "Unsupported registration fields.", "details": {"fields": sorted(extra)}}), 422
    full_name = payload.get("full_name")
    email = payload.get("email")
    password = payload.get("password")
    errors = {}
    if not isinstance(full_name, str) or not 2 <= len(full_name.strip()) <= 80 or any(ord(c) < 32 for c in full_name):
        errors["full_name"] = "must contain 2 to 80 printable characters"
    if not isinstance(email, str) or len(email) > 254 or not _EMAIL.fullmatch(email.strip()):
        errors["email"] = "must be a valid email address"
    if not isinstance(password, str) or not 10 <= len(password) <= 256:
        errors["password"] = "must contain 10 to 256 characters"
    if errors:
        return jsonify({"error": "validation_error", "message": "Invalid registration details.", "details": errors}), 422

    user = User(
        full_name=full_name.strip(),
        email=email.strip().lower(),
        password_hash=generate_password_hash(password, method="scrypt"),
        role="farmer",
        is_demo=bool(current_app.config.get("TESTING")),
    )
    try:
        db.session.add(user)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "email_conflict", "message": "An account with that email already exists."}), 409
    except SQLAlchemyError:
        db.session.rollback()
        current_app.logger.exception("Farmer account registration could not be persisted")
        return jsonify({"error": "database_unavailable", "message": "Account creation could not be completed."}), 503
    session.clear()
    session.permanent = True
    session["user_id"] = user.id
    return _user_response(user, _rotate_csrf())


@auth_bp.post("/login")
def login():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or set(payload) != {"email", "password"}:
        return jsonify({"error": "validation_error", "message": "Provide email and password."}), 422
    email, password = payload.get("email"), payload.get("password")
    if not isinstance(email, str) or not isinstance(password, str):
        return jsonify({"error": "validation_error", "message": "Email and password must be strings."}), 422
    user = User.query.filter_by(email=email.strip().lower()).first()
    if user is None or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "invalid_credentials", "message": "Email or password is incorrect."}), 401
    session.clear()
    session.permanent = True
    session["user_id"] = user.id
    return _user_response(user, _rotate_csrf())


@auth_bp.post("/logout")
def logout():
    session.clear()
    token = _rotate_csrf()
    return jsonify({"status": "signed_out", "csrf_token": token}), 200


@users_bp.get("/me")
def me():
    user = authenticated_user()
    if user is None:
        return jsonify({"authenticated": False}), 200
    return jsonify({"authenticated": True, "user": user.public_dict()}), 200
