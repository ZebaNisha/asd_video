#!/usr/bin/env python
"""
Backend Entry Point for ASD Video Diagnostics API
"""
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import create_app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask backend API server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
