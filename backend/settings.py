# backend/settings.py
"""Mutable in‑memory configuration shared across the backend.
The Settings are loaded by the Flask API endpoints and the inference
service. Updating them via the `/api/settings` POST endpoint changes the
runtime behavior without needing to restart the server.
"""

# Default configuration values
CONFIG = {
    "modelType": "vgg16_bilstm_v2",  # default model version
    "hiddenDim": 128,
    "dropout": 0.3,
    "serverEndpoint": "/api",
    "darkMode": False,
}

def get_config():
    """Return a shallow copy of the current configuration."""
    return CONFIG.copy()

def update_config(new_vals: dict):
    """Update allowed keys in CONFIG with values from ``new_vals``.
    Only existing keys are updated; unexpected keys are ignored.
    """
    for k, v in new_vals.items():
        if k in CONFIG:
            CONFIG[k] = v
    return get_config()
