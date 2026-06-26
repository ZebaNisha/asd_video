import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.app import create_app
from backend.extensions.db import db

app = create_app()

with app.app_context():
    db.create_all()
    print('Database tables created successfully.')
