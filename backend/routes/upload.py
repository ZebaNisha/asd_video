from flask import Blueprint, request, redirect, url_for, flash, current_app, render_template, jsonify
import os
import uuid
import threading
from werkzeug.utils import secure_filename
from ..extensions.db import db
from ..models.job import Job
from ..services.inference_service import predict

# Blueprint for upload functionality
upload_bp = Blueprint('upload', __name__)

# Allowed extensions (you can extend as needed)
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_prediction(app, job_id, video_path):
    """Background thread that runs inference and updates the Job record."""
    with app.app_context():
        try:
            # Set job status to processing
            job = db.session.get(Job, job_id) if hasattr(db.session, 'get') else Job.query.get(job_id)
            if job:
                job.status = 'processing'
                db.session.commit()
            
            result = predict(video_path)
            
            # Update job fields with results
            job = db.session.get(Job, job_id) if hasattr(db.session, 'get') else Job.query.get(job_id)
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
            # Mark job as failed
            try:
                job = db.session.get(Job, job_id) if hasattr(db.session, 'get') else Job.query.get(job_id)
                if job:
                    job.status = 'failed'
                    db.session.commit()
            except Exception as db_err:
                app.logger.error(f"Failed to update job status to failed: {db_err}")
            app.logger.error(f"Prediction failed for job {job_id}: {e}")

@upload_bp.route('/upload', methods=['GET', 'POST'], endpoint='upload')
def upload_video():
    """Handle video upload and start background inference."""
    if request.method == "POST":
        # Ensure a video file is provided
        if "video" not in request.files:
            flash("No video file part")
            return redirect(request.url)
        file = request.files["video"]
        # Validate filename
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)
        # Validate allowed extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Create a unique job id (UUID) and subfolder for the upload
            job_id = str(uuid.uuid4())
            upload_dir = os.path.join(current_app.root_path, "..", "static", "uploads", job_id)
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            # Insert job record into the database
            job = Job(id=job_id, video_path=file_path, status="queued")
            db.session.add(job)
            db.session.commit()
            # Start background inference thread
            app = current_app._get_current_object()
            thread = threading.Thread(target=run_prediction, args=(app, job_id, file_path), daemon=True)
            thread.start()
            # Return JSON response with job information
            return jsonify({"id": job_id, "status": "queued"})
        else:
            flash("File type not allowed")
            return redirect(request.url)
    # GET request – simple health check
    return jsonify({"message": "Upload endpoint ready"})
