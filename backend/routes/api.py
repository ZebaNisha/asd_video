from flask import Blueprint, jsonify, request
from ..models.job import Job
from ..settings import get_config, update_config

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/predictions', methods=['GET'])
def get_predictions():
    """Return JSON list of prediction records.
    Reuses the Job model to provide data for the frontend.
    """
    jobs = Job.query.order_by(Job.created_at.desc()).all()
    records = []
    for job in jobs:
        records.append({
            "id": job.id,
            "status": job.status,
            "predictionLabel": job.prediction_label or "",
            "confidenceScore": job.confidence_score or 0,
            "processingTime": job.processing_time or 0,
            "modelVersion": job.model_version or "",
            "createdAt": job.created_at.isoformat() if job.created_at else "",
        })
    return jsonify(records)

@api_bp.route('/reports', methods=['GET'])
def get_reports():
    """Return JSON list of completed reports."""
    jobs = Job.query.filter_by(status='completed').order_by(Job.created_at.desc()).all()
    records = []
    for job in jobs:
        records.append({
            "id": job.id,
            "status": job.status,
            "predictionLabel": job.prediction_label or "",
            "confidenceScore": job.confidence_score or 0,
            "processingTime": job.processing_time or 0,
            "modelVersion": job.model_version or "",
            "createdAt": job.created_at.isoformat() if job.created_at else "",
        })
    return jsonify(records)

@api_bp.route('/settings', methods=['GET'])
def get_settings():
    """Return current mutable settings."""
    return jsonify(get_config())

@api_bp.route('/settings', methods=['POST'])
def update_settings():
    """Update mutable SETTINGS with posted JSON payload."""
    data = request.get_json() or {}
    updated = update_config(data)
    return jsonify({"status": "ok", "updated": updated})
