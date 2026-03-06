import pytest
import io

def test_upload_csv_success(client, db_session):
    # Register and login to get token
    client.post("/auth/register", json={"email": "test@example.com", "password": "password"})
    login_res = client.post("/auth/login", data={"username": "test@example.com", "password": "password"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    csv_content = b"col1,col2\n1,2\n3,4"
    files = {"file": ("test.csv", io.BytesIO(csv_content), "text/csv")}

    response = client.post("/files/upload", headers=headers, files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["file_name"] == "test.csv"
    assert data["total_rows"] == 2
    assert data["total_columns"] == 2
    assert data["column_names"] == ["col1", "col2"]

def test_upload_invalid_extension(client, db_session):
    client.post("/auth/register", json={"email": "test@example.com", "password": "password"})
    login_res = client.post("/auth/login", data={"username": "test@example.com", "password": "password"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    files = {"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")}

    response = client.post("/files/upload", headers=headers, files=files)
    assert response.status_code == 400
    assert response.json()["detail"] == "Only CSV files are allowed."


# ---------------------------------------------------------------------------
# Tests for GET /files/list
# ---------------------------------------------------------------------------
def test_list_files_empty(client, db_session):
    """An authenticated user with no uploads should get an empty list."""
    client.post("/auth/register", json={"email": "list_empty@example.com", "password": "password"})
    login_res = client.post("/auth/login", data={"username": "list_empty@example.com", "password": "password"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/files/list", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total_files"] == 0
    assert data["files"] == []


def test_list_files_after_upload(client, db_session):
    """After uploading a file, it should appear in the file list."""
    client.post("/auth/register", json={"email": "list_upload@example.com", "password": "password"})
    login_res = client.post("/auth/login", data={"username": "list_upload@example.com", "password": "password"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    csv_content = b"a,b\n1,2\n3,4"
    files = {"file": ("data.csv", io.BytesIO(csv_content), "text/csv")}
    client.post("/files/upload", headers=headers, files=files)

    response = client.get("/files/list", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total_files"] == 1
    assert data["files"][0]["file_name"] == "data.csv"


# ---------------------------------------------------------------------------
# Tests for GET /files/download/{file_name}
# ---------------------------------------------------------------------------
def test_download_file_success(client, db_session):
    """Upload a file, then download it and verify the content."""
    client.post("/auth/register", json={"email": "dl_ok@example.com", "password": "password"})
    login_res = client.post("/auth/login", data={"username": "dl_ok@example.com", "password": "password"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    csv_content = b"x,y\n10,20"
    files = {"file": ("download_me.csv", io.BytesIO(csv_content), "text/csv")}
    client.post("/files/upload", headers=headers, files=files)

    response = client.get("/files/download/download_me.csv", headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert "download_me.csv" in response.headers.get("content-disposition", "")
    assert b"x,y" in response.content


def test_download_file_not_found(client, db_session):
    """Attempting to download a non-existent file should return 404."""
    client.post("/auth/register", json={"email": "dl_404@example.com", "password": "password"})
    login_res = client.post("/auth/login", data={"username": "dl_404@example.com", "password": "password"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/files/download/nonexistent.csv", headers=headers)
    assert response.status_code == 404

