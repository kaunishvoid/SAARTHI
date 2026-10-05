from flask import Blueprint, current_app, jsonify
from sqlalchemy.exc import SQLAlchemyError

from ..extensions import db


health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health():
    try:
        with db.engine.connect() as connection:
            connection.execute(db.text("SELECT 1"))
        return jsonify({"status": "ok", "database": "connected"}), 200
    except SQLAlchemyError as error:
        current_app.logger.warning("Database health check failed (%s)", type(error).__name__)
        return jsonify({
            "status": "unavailable",
            "database": "disconnected",
            "message": "Database connection is unavailable.",
        }), 503
