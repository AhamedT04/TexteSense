import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Base de données dédiée aux tests : définie AVANT tout import de `app`, car
# le moteur SQLAlchemy est créé à l'import. Une variable d'environnement est
# prioritaire sur le fichier .env, donc la vraie base n'est jamais touchée.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="textsense-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TEST_DB_DIR) / 'test.db'}"


@pytest.fixture(scope="session", autouse=True)
def _test_database():
    from app.database import engine, init_db

    init_db()
    yield
    engine.dispose()
