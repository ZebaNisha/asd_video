try:
    from flask_jwt_extended import JWTManager
except ImportError:
    # Fallback dummy JWT manager when flask_jwt_extended is not installed
    from functools import wraps
    class _DummyJWTManager:
        def __init__(self, app=None):
            self.app = app
        def init_app(self, app):
            self.app = app
        def create_access_token(self, identity, **kwargs):
            # Simple placeholder token
            return f"dummy-token-{identity}"
        def jwt_required(self, *args, **kwargs):
            # Decorator that does nothing (allows access without token)
            def decorator(fn):
                @wraps(fn)
                def wrapper(*f_args, **f_kwargs):
                    return fn(*f_args, **f_kwargs)
                return wrapper
            return decorator
    JWTManager = _DummyJWTManager

jwt = JWTManager()
