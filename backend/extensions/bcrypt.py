try:
    from flask_bcrypt import Bcrypt
except ImportError:
    # Fallback using Werkzeug's security utilities
    from werkzeug.security import generate_password_hash, check_password_hash
    class _DummyBcrypt:
        def __init__(self):
            pass
        def init_app(self, app):
            # No initialization needed for dummy
            pass
        def generate_password_hash(self, password, rounds=None):
            return generate_password_hash(password).encode('utf-8')
        def check_password_hash(self, pw_hash, password):
            return check_password_hash(pw_hash.decode('utf-8'), password)
    Bcrypt = _DummyBcrypt

bcrypt = Bcrypt()