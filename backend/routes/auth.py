from flask import Blueprint, request, jsonify, current_app
try:
    from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
except ImportError:
    # Dummy implementations when flask_jwt_extended is not installed
    def create_access_token(identity, **kwargs):
        return f"dummy-token-{identity}"
    def jwt_required(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    def get_jwt_identity():
        return None
    def get_jwt():
        return {}

from werkzeug.security import generate_password_hash, check_password_hash
from ..extensions.db import db
from ..extensions.bcrypt import bcrypt
from ..models.user import User
import datetime

auth_bp = Blueprint('auth', __name__)

# In-memory token blocklist (for demo; replace with Redis in prod)
_token_blocklist = set()

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or not all(k in data for k in ('email', 'password', 'role')):
        return jsonify({'msg': 'Missing required fields'}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'msg': 'User already exists'}), 409
    password_hash = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    username = data['email'].split('@')[0]
    user = User(username=username, email=data['email'], password_hash=password_hash)
    db.session.add(user)
    db.session.commit()
    return jsonify({'msg': 'User created'}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not all(k in data for k in ('email', 'password')):
        return jsonify({'msg': 'Missing email or password'}), 400
    user = User.query.filter_by(email=data['email']).first()
    if not user or not bcrypt.check_password_hash(user.password_hash, data['password']):
        return jsonify({'msg': 'Bad credentials'}), 401
    access_token = create_access_token(identity={'id': user.id, 'role': 'doctor'})
    return jsonify(access_token=access_token), 200

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    jti = get_jwt()['jti']
    _token_blocklist.add(jti)
    return jsonify({'msg': 'Token revoked'}), 200

@jwt_required()
def check_if_token_revoked(jwt_header, jwt_payload):
    return jwt_payload['jti'] in _token_blocklist
