import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import MagicMock
import os

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["SUPABASE_URL"] = "https://mock.supabase.co"
os.environ["SUPABASE_KEY"] = "mock-key-needs-to-be-longer-to-pass-validation-test-mock-key"
os.environ["JWT_SECRET_KEY"] = "mock-secret"

# We must mock supabase client before app imports since supabase checks key format
import sys
from unittest.mock import MagicMock
mock_supabase = MagicMock()
sys.modules['supabase'] = MagicMock()
sys.modules['supabase'].create_client = MagicMock(return_value=mock_supabase)
sys.modules['supabase'].Client = MagicMock

# Import app components before tests
from app.main import app
from app.database import Base, get_db
from app.dependencies.auth import get_current_user
from app.supabase_client import supabase
from app.models.user import User

# --- Database Mock ---
# Use in-memory SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}, pool_pre_ping=True
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    Base.metadata.create_all(bind=engine)
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="function")
def db_session():
    """Create a new database session for a test."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="module")
def client():
    """Test client for FastAPI."""
    with TestClient(app) as c:
        yield c

# --- Supabase Storage Mock ---
class MockBucket:
    def __init__(self):
        self.files = {}
        
    def upload(self, path, file, file_options=None):
        self.files[path] = file

    def download(self, path):
        return self.files.get(path, b"col1,col2\n1,2")

class MockStorage:
    def __init__(self):
        self._buckets = {}
        
    def from_(self, name):
        if name not in self._buckets:
            self._buckets[name] = MockBucket()
        return self._buckets[name]

@pytest.fixture(autouse=True)
def mock_supabase_storage(monkeypatch):
    """Automatically mock the Supabase storage client for all tests."""
    storage_mock = MockStorage()
    
    # Check if supabase object is a MagicMock, if so, just attach the property directly
    if isinstance(supabase, MagicMock):
        supabase.storage = storage_mock
    else:
        monkeypatch.setattr(type(supabase), "storage", property(lambda self: storage_mock))
    
    return storage_mock
