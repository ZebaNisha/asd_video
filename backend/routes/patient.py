from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from ..extensions.db import db
from ..models.patient import Patient

# Blueprint for patient-related endpoints
patient_bp = Blueprint('patient', __name__)

@patient_bp.route('/patients', methods=['GET'], endpoint='patients')
def list_patients():
    """Return JSON list of patients or render HTML page."""
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        patients = Patient.query.all()
        data = [
            {
                'id': p.id,
                'name': p.name,
                'age': p.age,
                'gender': p.gender,
                'guardian_name': p.guardian_name,
                'created_at': p.created_at.isoformat() if p.created_at else None
            } for p in patients
        ]
        return jsonify(data)
    return render_template('patients.html')

@patient_bp.route('/patient/<int:patient_id>', methods=['GET'], endpoint='patient_detail')
def patient_detail(patient_id):
    """Return JSON details of a patient or render HTML page."""
    patient = Patient.query.get_or_404(patient_id)
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify({
            'id': patient.id,
            'name': patient.name,
            'age': patient.age,
            'gender': patient.gender,
            'guardian_name': patient.guardian_name,
            'notes': patient.notes,
            'created_at': patient.created_at.isoformat() if patient.created_at else None
        })
    return render_template('patient_detail.html', patient=patient)

@patient_bp.route('/patients', methods=['POST'])
def save_patient():
    """Create a new patient or update an existing one."""
    # Handle JSON or Form URL encoded data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    patient_id = data.get('id')
    if patient_id:
        patient = Patient.query.get_or_404(int(patient_id))
    else:
        from ..models.user import User
        # Ensure at least one default user exists for foreign key constraint
        user = User.query.first()
        if not user:
            user = User(username='doctor', email='doctor@asdvision.com', password_hash='dummy_hash')
            db.session.add(user)
            db.session.commit()
        patient = Patient(user_id=user.id)
        db.session.add(patient)

    patient.name = data.get('name')
    patient.age = int(data.get('age')) if data.get('age') else None
    patient.gender = data.get('gender')
    patient.guardian_name = data.get('guardian_name')
    patient.notes = data.get('notes')

    db.session.commit()

    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            'id': patient.id,
            'name': patient.name,
            'age': patient.age,
            'gender': patient.gender,
            'guardian_name': patient.guardian_name,
            'notes': patient.notes
        }), 200
    
    return redirect(url_for('patient.patients'))

@patient_bp.route('/patient/<int:patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    """Delete a patient from the database."""
    patient = Patient.query.get_or_404(patient_id)
    db.session.delete(patient)
    db.session.commit()
    return jsonify({'msg': 'Patient deleted successfully'}), 200
