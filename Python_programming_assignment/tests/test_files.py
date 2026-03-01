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
