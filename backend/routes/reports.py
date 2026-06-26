from flask import Blueprint, render_template, abort, jsonify, make_response, url_for
from ..models.job import Job
import csv
import io

reports_bp = Blueprint('reports', __name__)

@reports_bp.route('/reports/<job_id>')
def reports(job_id):
    """Render a detailed report page for a completed job."""
    job = Job.query.get_or_404(job_id)
    return render_template('reports.html', job=job)

@reports_bp.route('/reports/<job_id>/download/<fmt>')
def download_report(job_id, fmt):
    """Download the job report as CSV or JSON.

    URL pattern: /reports/<job_id>/download/csv  or  /reports/<job_id>/download/json
    """
    job = Job.query.get_or_404(job_id)
    data = {
        'id': job.id,
        'label': job.prediction_label,
        'asd_probability': job.asd_probability,
        'td_probability': job.td_probability,
        'confidence_score': job.confidence_score,
        'processing_time': job.processing_time,
        'model_version': job.model_version,
        'raw_classification': job.raw_classification,
        'created_at': job.created_at.isoformat() if job.created_at else None,
        'updated_at': job.updated_at.isoformat() if job.updated_at else None,
    }
    fmt = fmt.lower()
    if fmt == 'json':
        return jsonify(data)
    elif fmt == 'csv':
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
        csv_bytes = output.getvalue().encode('utf-8')
        response = make_response(csv_bytes)
        response.headers['Content-Type'] = 'text/csv'
        filename = f"report_{job.id}.csv"
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        return response
    else:
        abort(400, description='Unsupported format')
