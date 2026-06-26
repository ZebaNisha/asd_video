from flask import Blueprint, render_template, jsonify

# Blueprint for the AI dashboard

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard', endpoint='dashboard')
def dashboard():
    return render_template('dashboard.html')

@dashboard_bp.route('/api/dashboard/stats')
def dashboard_stats():
    """Return JSON metrics and recent predictions for the dashboard."""
    from ..models.patient import Patient
    from ..models.job import Job

    total_patients = Patient.query.count()
    all_jobs = Job.query.all()
    
    videos_uploaded = len(all_jobs)
    completed_analyses = len([j for j in all_jobs if j.status == 'completed'])
    processing_queue = len([j for j in all_jobs if j.status in ['queued', 'processing']])

    # Get recent predictions
    recent_jobs = Job.query.order_by(Job.created_at.desc()).limit(5).all()
    recent = []
    for j in recent_jobs:
        recent.append({
            'id': j.id,
            'status': j.status,
            'prediction_label': j.prediction_label or '-',
            'confidence_score': j.confidence_score or 0.0,
            'processing_time': j.processing_time or 0.0,
            'created_at': j.created_at.isoformat() if j.created_at else None
        })

    return jsonify({
        'total_patients': total_patients,
        'videos_uploaded': videos_uploaded,
        'completed_analyses': completed_analyses,
        'processing_queue': processing_queue,
        'recent_predictions': recent
    })
