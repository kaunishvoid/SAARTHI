"""Relational records for authenticated SAARTHI workflows."""

from datetime import datetime, timezone

from .extensions import db


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(254), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(16), nullable=False, default="farmer", index=True)
    is_demo = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)

    def public_dict(self) -> dict:
        return {"id": self.id, "full_name": self.full_name, "email": self.email, "role": self.role, "demo_account": self.is_demo}


class OfficerProfile(db.Model):
    __tablename__ = "officer_profiles"

    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    display_name = db.Column(db.String(80), nullable=False)
    organization = db.Column(db.String(120), nullable=True)
    is_demo = db.Column(db.Boolean, nullable=False, default=False)


class CropPrediction(db.Model):
    __tablename__ = "crop_predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    inputs = db.Column(db.JSON, nullable=False)
    result = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, index=True)


class FertilizerPrediction(db.Model):
    __tablename__ = "fertilizer_predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    inputs = db.Column(db.JSON, nullable=False)
    result = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, index=True)


class Conversation(db.Model):
    __tablename__ = "conversations"

    id = db.Column(db.String(80), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)


class Message(db.Model):
    __tablename__ = "messages"

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(80), db.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = db.Column(db.String(16), nullable=False)
    content = db.Column(db.Text, nullable=False)
    metadata_json = db.Column(db.JSON, nullable=False, default=dict)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, index=True)


class DocumentUpload(db.Model):
    __tablename__ = "document_uploads"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = db.Column(db.String(180), nullable=False)
    sha256 = db.Column(db.String(64), nullable=False)
    extracted_text = db.Column(db.Text, nullable=False)
    source_type = db.Column(db.String(40), nullable=False, default="user_provided_unverified")
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, index=True)


class Appointment(db.Model):
    __tablename__ = "appointments"

    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    officer_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    requested_for = db.Column(db.DateTime(timezone=True), nullable=False, index=True)
    topic = db.Column(db.String(120), nullable=False)
    notes = db.Column(db.String(1000), nullable=False, default="")
    status = db.Column(db.String(20), nullable=False, default="requested", index=True)
    is_demo = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)

    def as_dict(self) -> dict:
        officer = db.session.get(User, self.officer_id) if self.officer_id else None
        return {
            "id": self.id,
            "farmer_id": self.farmer_id,
            "officer_id": self.officer_id,
            "officer_name": officer.full_name if officer else None,
            "requested_for": as_utc_iso(self.requested_for),
            "topic": self.topic,
            "notes": self.notes,
            "status": self.status,
            "demo_record": self.is_demo,
            "created_at": as_utc_iso(self.created_at),
        }
