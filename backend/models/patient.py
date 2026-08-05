from datetime import datetime

from ..extensions.db import db


class Patient(db.Model):
    __tablename__ = "patients"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    guardian_name = db.Column(db.String(120), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    date_of_birth = db.Column(db.Date, nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    videos = db.relationship('Video', backref='patient', lazy=True)

    def __repr__(self):
        return f"<Patient {self.name}>"
