from flask import Blueprint, render_template, abort, redirect, url_for
from ..models.job import Job

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/prediction/<job_id>')
def prediction_page(job_id):
    job = Job.query.get(job_id)
    if not job:
        abort(404)
    if job.status != 'completed':
        return redirect(url_for('processing.processing_view', job_id=job_id))
    return render_template('prediction.html', job=job)
