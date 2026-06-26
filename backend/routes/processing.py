from flask import Blueprint, render_template, jsonify, current_app
from ..models.job import Job
from ..extensions.db import db

processing_bp = Blueprint('processing', __name__)

@processing_bp.route('/processing/<job_id>')
def processing_view(job_id):
    """Render the processing timeline page for a given job.
    The template will poll the job status via the /api/job/<id> endpoint.
    """
    job = Job.query.get_or_404(job_id)
    return render_template('processing.html', job_id=job.id)

@processing_bp.route('/api/job/<job_id>')
def job_status_api(job_id):
    """Return JSON with current job status and results (if completed)."""
    job = Job.query.get_or_404(job_id)
    data = {
        'id': job.id,
        'status': job.status,
        'prediction_label': job.prediction_label,
        'asd_probability': job.asd_probability,
        'td_probability': job.td_probability,
        'confidence_score': job.confidence_score,
        'processing_time': job.processing_time,
        'model_version': job.model_version,
        'raw_classification': job.raw_classification,
    }
    return jsonify(data)

def start_inference(job_id, video_path):
    """Helper to start inference in a background thread (used by upload route)."""
    from ..services.inference_service import predict
    import threading

    def _run():
        try:
            result = predict(video_path)
            job = Job.query.get(job_id)
            if job:
                job.status = 'completed'
                job.prediction_label = result.get('prediction_label')
                job.asd_probability = result.get('asd_probability')
                job.td_probability = result.get('td_probability')
                job.confidence_score = result.get('confidence_score')
                job.processing_time = result.get('processing_time')
                job.model_version = result.get('model_version')
                job.raw_classification = result.get('raw_classification')
                db.session.commit()
        except Exception as e:
            job = Job.query.get(job_id)
            if job:
                job.status = 'failed'
                db.session.commit()
            current_app.logger.error(f'Inference failed for job {job_id}: {e}')

    threading.Thread(target=_run, daemon=True).start()
