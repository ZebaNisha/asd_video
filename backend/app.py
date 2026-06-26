import os
from flask import Flask, redirect, url_for
from .config import DevelopmentConfig, ProductionConfig
from .extensions.db import db


def create_app(test_config=None):
    """Flask application factory."""
    app = Flask(__name__, instance_relative_config=True)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    env = os.getenv('FLASK_ENV', 'development')
    if test_config is None:
        if env == 'production':
            app.config.from_object(ProductionConfig)
        else:
            app.config.from_object(DevelopmentConfig)
    else:
        app.config.update(test_config)

    # ------------------------------------------------------------------
    # Extensions
    # ------------------------------------------------------------------
    db.init_app(app)

    from .extensions.bcrypt import bcrypt
    from .extensions.jwt_manager import jwt
    bcrypt.init_app(app)
    jwt.init_app(app)

    # ------------------------------------------------------------------
    # Models  (must be imported so SQLAlchemy registers table metadata)
    # ------------------------------------------------------------------
    from .models import user, patient, job, video, report, prediction  # noqa: F401

    # ------------------------------------------------------------------
    # Blueprints
    # ------------------------------------------------------------------
    from .routes.patient import patient_bp
    app.register_blueprint(patient_bp)

    from .routes.upload import upload_bp
    app.register_blueprint(upload_bp)

    from .routes.processing import processing_bp
    app.register_blueprint(processing_bp)

    from .routes.prediction import prediction_bp
    app.register_blueprint(prediction_bp)

    from .routes.reports import reports_bp
    app.register_blueprint(reports_bp)

    from .routes.dashboard import dashboard_bp
    app.register_blueprint(dashboard_bp)

    from .routes.auth import auth_bp
    app.register_blueprint(auth_bp)

    # Register new API Blueprint
    from .routes.api import api_bp
    app.register_blueprint(api_bp)

    # ------------------------------------------------------------------
    # Root redirect  →  /dashboard
    # ------------------------------------------------------------------
    @app.route('/')
    def root():
        return redirect(url_for('dashboard.dashboard'))

    # Simple page routes that have no dedicated blueprint
    from flask import render_template as _rt

    @app.route('/predictions')
    def predictions():
        from .models.job import Job
        jobs = Job.query.order_by(Job.created_at.desc()).all()
        return _rt('predictions.html', jobs=jobs)

    @app.route('/reports')
    def reports_list():
        from .models.job import Job
        jobs = Job.query.filter_by(status='completed').order_by(Job.created_at.desc()).all()
        return _rt('reports_list.html', jobs=jobs)

    @app.route('/settings')
    def settings():
        return _rt('settings.html')

    # ------------------------------------------------------------------
    # Database bootstrap
    # ------------------------------------------------------------------
    with app.app_context():
        db.create_all()

    return app
