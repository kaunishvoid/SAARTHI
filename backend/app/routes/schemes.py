"""Account-owned text document intake; verified scheme matching is unavailable."""

import hashlib

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from ..auth import authenticated_user, login_required, roles_required
from ..extensions import db
from ..models import DocumentUpload, as_utc_iso


schemes_bp = Blueprint("schemes", __name__, url_prefix="/api/schemes")
MAX_TEXT_BYTES = 64 * 1024


@schemes_bp.get("/status")
def scheme_status():
    return jsonify({
        "verified_information_available": False,
        "verified_scheme_recommendations_available": False,
        "scheme_source_count": 0,
        "source": None,
        "document_upload_available": True,
        "supported_upload": "UTF-8 plain-text .txt only",
        "text_extraction_available": True,
        "ocr_available": False,
        "message": "No project-owned scheme source or eligibility rules are configured. Uploaded text remains user-provided and unverified.",
    }), 200


def _document_dict(record: DocumentUpload, include_text: bool = True) -> dict:
    result = {
        "id": record.id,
        "filename": record.filename,
        "sha256": record.sha256,
        "source_type": record.source_type,
        "created_at": as_utc_iso(record.created_at),
        "verified_scheme_source": False,
    }
    if include_text:
        result["extracted_text"] = record.extracted_text
    return result


@schemes_bp.post("/documents")
@roles_required("farmer")
def upload_document():
    file = request.files.get("document")
    if file is None or not file.filename:
        return jsonify({"error": "validation_error", "message": "Select a text document to upload."}), 422
    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".txt") or file.mimetype not in {"text/plain", "application/octet-stream"}:
        return jsonify({"error": "unsupported_document", "message": "Only UTF-8 plain-text .txt documents are supported. PDF/image OCR is unavailable."}), 422
    data = file.stream.read(MAX_TEXT_BYTES + 1)
    if not data or len(data) > MAX_TEXT_BYTES:
        return jsonify({"error": "validation_error", "message": "Document must contain between 1 byte and 64 KiB."}), 422
    try:
        text = data.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        return jsonify({"error": "invalid_document", "message": "Text document must use UTF-8 encoding."}), 422
    if not text or "\x00" in text:
        return jsonify({"error": "invalid_document", "message": "Document contains no readable UTF-8 text."}), 422
    user = authenticated_user()
    record = DocumentUpload(
        user_id=user.id,
        filename=filename[:180],
        sha256=hashlib.sha256(data).hexdigest(),
        extracted_text=text,
        source_type="user_provided_unverified",
    )
    db.session.add(record)
    db.session.commit()
    return jsonify({
        "document": _document_dict(record),
        "scheme_recommendations_available": False,
        "message": "Text was extracted from your supplied file. It is not a verified scheme source, and no eligibility determination was made.",
    }), 201


@schemes_bp.get("/documents")
@roles_required("farmer")
def list_documents():
    user = authenticated_user()
    rows = DocumentUpload.query.filter_by(user_id=user.id).order_by(DocumentUpload.created_at.desc()).limit(50).all()
    return jsonify({"items": [_document_dict(row) for row in rows]}), 200


@schemes_bp.post("/analyze")
@roles_required("farmer")
def unavailable_scheme_analysis():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or set(payload) - {"document_id"}:
        return jsonify({"error": "validation_error", "message": "Provide only a document_id."}), 422
    document_id = payload.get("document_id")
    user = authenticated_user()
    document = DocumentUpload.query.filter_by(id=document_id, user_id=user.id).first() if isinstance(document_id, int) else None
    if document is None:
        return jsonify({"error": "not_found", "message": "Document not found for this account."}), 404
    return jsonify({
        "status": "verified_scheme_sources_unavailable",
        "scheme_recommendations": [],
        "document_id": document.id,
        "message": "No scheme was matched. This version has no verified scheme database or eligibility rules.",
    }), 200
