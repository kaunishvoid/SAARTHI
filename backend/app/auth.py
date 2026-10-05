"""Session authentication and role helpers."""

from functools import wraps

from flask import jsonify, session

from .extensions import db
from .models import User


def authenticated_user() -> User | None:
    user_id = session.get("user_id")
    if not isinstance(user_id, int):
        return None
    user = db.session.get(User, user_id)
    if user is None:
        session.clear()
    return user


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = authenticated_user()
        if user is None:
            return jsonify({"error": "authentication_required", "message": "Please sign in to continue."}), 401
        return view(*args, **kwargs)
    return wrapped


def roles_required(*roles):
    def decorate(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = authenticated_user()
            if user is None:
                return jsonify({"error": "authentication_required", "message": "Please sign in to continue."}), 401
            if user.role not in roles:
                return jsonify({"error": "forbidden", "message": "Your account role cannot perform this action."}), 403
            return view(*args, **kwargs)
        return wrapped
    return decorate
