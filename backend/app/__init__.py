import os
import hmac

from flask import Flask, jsonify, request, session
from sqlalchemy.exc import SQLAlchemyError

from .config import Config
from .extensions import db
from .routes.health import health_bp
from .routes.crop import crop_bp
from .routes.fertilizer import fertilizer_bp
from .routes.agent import agent_bp
from .routes.auth import auth_bp, users_bp
from .routes.schemes import schemes_bp
from .routes.appointments import appointments_bp
from .routes.history import history_bp
from .agent.service import ConsultancyService
from . import models  # noqa: F401 - register SQLAlchemy models before create_all
from .ml.crop_model import CropModelService
from .ml.fertilizer_service import FertilizerModelService


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    # Flask-SQLAlchemy resolves relative SQLite URLs under this directory.
    os.makedirs(app.instance_path, exist_ok=True)
    db.init_app(app)
    app.config.setdefault("SESSION_COOKIE_NAME", "saarthi_session")
    app.register_blueprint(health_bp)
    app.register_blueprint(crop_bp)
    app.register_blueprint(fertilizer_bp)
    app.register_blueprint(agent_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(users_bp)
    app.register_blueprint(schemes_bp)
    app.register_blueprint(appointments_bp)
    app.register_blueprint(history_bp)
    app.extensions["consultancy_service"] = ConsultancyService()
    @app.before_request
    def csrf_protection():
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            expected = session.get("csrf_token")
            supplied = request.headers.get("X-CSRF-Token", "")
            if not isinstance(expected, str) or not hmac.compare_digest(expected, supplied):
                return jsonify({
                    "error": "csrf_failed",
                    "message": "Refresh the page and retry the request.",
                }), 403

    with app.app_context():
        try:
            db.create_all()
        except SQLAlchemyError as error:
            db.session.rollback()
            app.logger.warning("Database schema initialization deferred (%s)", type(error).__name__)

    @app.errorhandler(SQLAlchemyError)
    def database_error(error):
        db.session.rollback()
        app.logger.warning("Database request failed (%s)", type(error).__name__)
        return jsonify({"error": "database_unavailable", "message": "The database request could not be completed."}), 503

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify({"error": "upload_too_large", "message": "Request exceeds the 2 MiB upload limit."}), 413
    try:
        app.extensions["crop_model_service"] = CropModelService(app.config["CROP_MODEL_ARTIFACT_DIR"])
        app.logger.info("Loaded crop ANN artifacts from %s", app.config["CROP_MODEL_ARTIFACT_DIR"])
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        app.extensions["crop_model_service"] = None
        app.logger.warning("Crop ANN unavailable: %s", error)
    try:
        app.extensions["fertilizer_model_service"] = FertilizerModelService(
            app.config["FERTILIZER_MODEL_ARTIFACT_DIR"]
        )
        app.logger.info(
            "Loaded NEUTRA-BOOST artifacts from %s",
            app.config["FERTILIZER_MODEL_ARTIFACT_DIR"],
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        app.extensions["fertilizer_model_service"] = None
        app.logger.warning("NEUTRA-BOOST unavailable: %s", error)

    return app
