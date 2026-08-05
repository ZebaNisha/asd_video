from datetime import datetime
from ..extensions.db import db

class Job(db.Model):
    __tablename__ = 'jobs'
    id = db.Column(db.String(36), primary_key=True)  # UUID string
    video_path = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='queued')  # queued, processing, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Result fields populated after inference
    prediction_label = db.Column(db.String(10))
    asd_probability = db.Column(db.Float)
    td_probability = db.Column(db.Float)
    confidence_score = db.Column(db.Float)
    processing_time = db.Column(db.Float)
    model_version = db.Column(db.String(20))
    raw_classification = db.Column(db.String(50))
