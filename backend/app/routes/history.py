"""Authenticated prediction and conversation history."""

from flask import Blueprint, jsonify, request

from ..auth import authenticated_user, login_required
from ..extensions import db
from ..models import Conversation, Message, as_utc_iso


history_bp = Blueprint("history", __name__, url_prefix="/api")


def _limit() -> int:
    try:
        return min(max(int(request.args.get("limit", "50")), 1), 100)
    except ValueError:
        return 50


@history_bp.get("/agent/history")
@login_required
def conversation_history():
    user = authenticated_user()
    rows = Conversation.query.filter_by(user_id=user.id).order_by(Conversation.updated_at.desc()).limit(_limit()).all()
    items = []
    for conversation in rows:
        messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.created_at.asc()).limit(200).all()
        items.append({
            "conversation_id": conversation.id,
            "created_at": as_utc_iso(conversation.created_at),
            "messages": [{"role": message.role, "content": message.content, "metadata": message.metadata_json} for message in messages],
        })
    return jsonify({"items": items}), 200
