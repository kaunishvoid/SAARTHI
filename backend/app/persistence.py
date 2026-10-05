"""Optional authenticated prediction persistence."""

from sqlalchemy.exc import SQLAlchemyError

from .auth import authenticated_user
from .extensions import db
from .models import CropPrediction, FertilizerPrediction


class PersistenceUnavailable(RuntimeError):
    """Authenticated result could not be committed to the configured database."""


def persist_prediction(module: str, inputs: dict, result: dict) -> int | None:
    user = authenticated_user()
    if user is None:
        return None
    model = CropPrediction if module == "crop" else FertilizerPrediction
    record = model(user_id=user.id, inputs=inputs, result=result)
    try:
        db.session.add(record)
        db.session.commit()
    except SQLAlchemyError as error:
        db.session.rollback()
        raise PersistenceUnavailable from error
    return record.id
