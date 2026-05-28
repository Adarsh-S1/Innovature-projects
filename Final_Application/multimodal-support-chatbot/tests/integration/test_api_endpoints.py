"""
API endpoint integration tests using FastAPI TestClient.

Tests cover:
- Health endpoint
- Chat endpoint (mocked pipeline)
- Document listing endpoint
- Input validation
- Error handling
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI TestClient with mocked infrastructure dependencies."""
    # Mock Redis and Milvus before importing the app
    with patch("app.db.redis_client.redis_manager") as mock_redis, \
         patch("app.db.milvus_client.milvus_manager") as mock_milvus, \
         patch("app.db.minio_client.minio_manager") as mock_minio:

        # Configure Redis mocks
        mock_redis.connect = AsyncMock()
        mock_redis.disconnect = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.get_session = AsyncMock(return_value=None)
        mock_redis.save_session = AsyncMock()
        mock_redis.get_all_documents = AsyncMock(return_value=[])
        mock_redis.delete_document = AsyncMock()

        # Configure Milvus mocks
        mock_milvus.connect = MagicMock()
        mock_milvus.disconnect = MagicMock()
        mock_milvus.text_collection = None
        mock_milvus.image_collection = None

        # Configure MinIO mocks
        mock_minio.connect = AsyncMock()
        mock_minio.disconnect = AsyncMock()

        from app.main import create_app
        app = create_app()
        yield TestClient(app)


class TestHealthEndpoint:
    """Tests for the /api/v1/health endpoint."""

    def test_health_returns_200(self, client):
        """Health check should return 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response should have required fields."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data


class TestRootEndpoint:
    """Tests for the root / endpoint."""

    def test_root_returns_200(self, client):
        """Root endpoint should return 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_app_info(self, client):
        """Root endpoint should include app name."""
        response = client.get("/")
        data = response.json()
        assert "app" in data


class TestChatEndpoint:
    """Tests for the /api/v1/chat endpoint."""

    def test_chat_rejects_empty_query(self, client):
        """Should reject empty query with 422."""
        response = client.post("/api/v1/chat", json={"query": ""})
        assert response.status_code == 422

    def test_chat_rejects_too_long_query(self, client):
        """Should reject queries over 2000 chars."""
        response = client.post("/api/v1/chat", json={"query": "x" * 2001})
        assert response.status_code == 422

    def test_chat_accepts_valid_query(self, client):
        """Valid query should be accepted (pipeline mocked)."""
        with patch("app.api.v1.chat.compiled_graph") as mock_graph:
            mock_graph.invoke.return_value = {
                "final_answer": "Test answer",
                "cited_sources": ["source.pdf, Pages 1-2"],
                "quality_score": 0.9,
                "confidence_level": "high",
                "response_images": [],
                "query_type": "general",
                "retrieved_chunks": [],
                "retrieved_images": [],
            }

            response = client.post(
                "/api/v1/chat",
                json={"query": "How to install RAM?"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "session_id" in data


class TestDocumentsEndpoint:
    """Tests for the /api/v1/documents endpoint."""

    def test_list_documents_returns_200(self, client):
        """Document listing should return 200."""
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_delete_document_returns_200(self, client):
        """Document deletion should return 200."""
        with patch("app.api.v1.documents.milvus_manager") as mock_milvus:
            mock_milvus.text_collection = None
            mock_milvus.image_collection = None

            response = client.delete("/api/v1/documents/test-doc-123")
            assert response.status_code == 200
            data = response.json()
            assert data["deleted"] is True
