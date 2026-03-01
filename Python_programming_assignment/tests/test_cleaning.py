import pytest
import io
import pandas as pd

def test_clean_invalid_id(client, db_session):
    client.post("/auth/register", json={"email": "test@example.com", "password": "password"})
    login_res = client.post("/auth/login", data={"username": "test@example.com", "password": "password"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    response = client.post(
        "/files/clean",
        headers=headers,
        json={"file_name": "test.csv", "clean": 99} # Invalid ID
    )
    assert response.status_code == 422 # Pydantic validation error

def test_clean_dropna(client, db_session, mock_supabase_storage):
    # Register, login, upload file with missing value
    client.post("/auth/register", json={"email": "test@example.com", "password": "password"})
    login_res = client.post("/auth/login", data={"username": "test@example.com", "password": "password"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    csv_content = b"col1,col2\n1,2\n3,\n5,6" # Row 2 has missing col2
    files = {"file": ("test.csv", io.BytesIO(csv_content), "text/csv")}
    client.post("/files/upload", headers=headers, files=files)

    mock_supabase_storage_client = mock_supabase_storage.from_("csv-uploads")
    mock_supabase_storage_client.download = lambda path: csv_content

    response = client.post(
        "/files/clean",
        headers=headers,
        json={"file_name": "test.csv", "clean": 1}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["original_file_name"] == "test.csv"
    assert data["rows_before"] == 3
    assert data["rows_after"] == 2 # Dropped the row with missing value
    assert data["nan_filled"] == 0
    assert data["records_removed_or_modified"] == 1

def test_clean_fill_mean(client, db_session, mock_supabase_storage):
    # Register, login, upload file with missing value
    client.post("/auth/register", json={"email": "fill_mean@example.com", "password": "password"})
    login_res = client.post("/auth/login", data={"username": "fill_mean@example.com", "password": "password"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    csv_content = b"col1,col2\n1,2\n3,\n5,4" # Mean of col2 (2, 4) is 3
    files = {"file": ("test_mean.csv", io.BytesIO(csv_content), "text/csv")}
    client.post("/files/upload", headers=headers, files=files)

    mock_supabase_storage_client = mock_supabase_storage.from_("csv-uploads")
    mock_supabase_storage_client.download = lambda path: csv_content

    response = client.post(
        "/files/clean",
        headers=headers,
        json={"file_name": "test_mean.csv", "clean": 2}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["rows_before"] == 3
    assert data["rows_after"] == 3 # Rows are not dropped, just filled
    assert data["nan_filled"] == 1
    assert data["records_removed_or_modified"] == 0

def test_clean_ffill(client, db_session, mock_supabase_storage):
    client.post("/auth/register", json={"email": "ffill@example.com", "password": "password"})
    login_res = client.post("/auth/login", data={"username": "ffill@example.com", "password": "password"})
    headers = {"Authorization": f"Bearer {login_res.json()['access_token']}"}

    csv_content = b"col1,col2\n1,2\n3,\n5,6" # Row 2 has missing col2
    files = {"file": ("test_ffill.csv", io.BytesIO(csv_content), "text/csv")}
    client.post("/files/upload", headers=headers, files=files)

    mock_supabase_storage.from_("csv-uploads").download = lambda path: csv_content

    response = client.post("/files/clean", headers=headers, json={"file_name": "test_ffill.csv", "clean": 3})
    
    assert response.status_code == 200
    data = response.json()
    assert data["rows_before"] == 3
    assert data["rows_after"] == 3
    assert data["nan_filled"] == 1
    assert data["records_removed_or_modified"] == 0

def test_clean_bfill(client, db_session, mock_supabase_storage):
    client.post("/auth/register", json={"email": "bfill@example.com", "password": "password"})
    login_res = client.post("/auth/login", data={"username": "bfill@example.com", "password": "password"})
    headers = {"Authorization": f"Bearer {login_res.json()['access_token']}"}

    csv_content = b"col1,col2\n1,\n3,4\n5,6" # Row 1 has missing col2
    files = {"file": ("test_bfill.csv", io.BytesIO(csv_content), "text/csv")}
    client.post("/files/upload", headers=headers, files=files)

    mock_supabase_storage.from_("csv-uploads").download = lambda path: csv_content

    response = client.post("/files/clean", headers=headers, json={"file_name": "test_bfill.csv", "clean": 4})
    
    assert response.status_code == 200
    data = response.json()
    assert data["rows_before"] == 3
    assert data["rows_after"] == 3
    assert data["nan_filled"] == 1
    assert data["records_removed_or_modified"] == 0

def test_clean_drop_duplicates(client, db_session, mock_supabase_storage):
    client.post("/auth/register", json={"email": "drop_dup@example.com", "password": "password"})
    login_res = client.post("/auth/login", data={"username": "drop_dup@example.com", "password": "password"})
    headers = {"Authorization": f"Bearer {login_res.json()['access_token']}"}

    csv_content = b"col1,col2\n1,2\n3,4\n1,2\n5,6" # Row 0 and Row 2 are duplicates
    files = {"file": ("test_dup.csv", io.BytesIO(csv_content), "text/csv")}
    client.post("/files/upload", headers=headers, files=files)

    mock_supabase_storage.from_("csv-uploads").download = lambda path: csv_content

    response = client.post("/files/clean", headers=headers, json={"file_name": "test_dup.csv", "clean": 5})
    
    assert response.status_code == 200
    data = response.json()
    assert data["rows_before"] == 4
    assert data["rows_after"] == 3
    assert data["nan_filled"] == 0
    assert data["records_removed_or_modified"] == 1
