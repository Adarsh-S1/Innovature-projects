import pytest

def test_register_success(client, db_session):
    response = client.post(
        "/auth/register",
        json={"email": "test@example.com", "password": "password123"}
    )
    assert response.status_code == 201
    assert response.json()["email"] == "test@example.com"
    assert "id" in response.json()

def test_register_duplicate(client, db_session):
    client.post(
        "/auth/register",
        json={"email": "test@example.com", "password": "password123"}
    )
    response = client.post(
        "/auth/register",
        json={"email": "test@example.com", "password": "password123"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "A user with this email already exists."

def test_login_success(client, db_session):
    client.post(
        "/auth/register",
        json={"email": "test@example.com", "password": "password123"}
    )
    response = client.post(
        "/auth/login",
        data={"username": "test@example.com", "password": "password123"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_login_invalid(client, db_session):
    response = client.post(
        "/auth/login",
        data={"username": "nonexistent@example.com", "password": "wrongpassword"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."


def test_get_current_user_success(client, db_session):
    # Register and login
    client.post("/auth/register", json={"email": "me@example.com", "password": "password123"})
    login_resp = client.post("/auth/login", data={"username": "me@example.com", "password": "password123"})
    token = login_resp.json()["access_token"]

    # Get current user
    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "me@example.com"


def test_get_current_user_unauthorized(client):
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_logout_success(client, db_session):
    client.post("/auth/register", json={"email": "logout@example.com", "password": "password123"})
    login_resp = client.post("/auth/login", data={"username": "logout@example.com", "password": "password123"})
    token = login_resp.json()["access_token"]

    # Logout
    logout_resp = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout_resp.status_code == 200
    assert logout_resp.json()["message"] == "Successfully logged out."

    # Try to access protected route with blocklisted token
    me_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 401


def test_forget_password_success(client, db_session):
    client.post("/auth/register", json={"email": "forget@example.com", "password": "password123"})
    
    response = client.post("/auth/forgot-password", json={"email": "forget@example.com"})
    assert response.status_code == 200
    assert "Reset token generated:" in response.json()["message"]


def test_reset_password_success(client, db_session):
    client.post("/auth/register", json={"email": "reset@example.com", "password": "password123"})
    
    forgot_resp = client.post("/auth/forgot-password", json={"email": "reset@example.com"})
    token_msg = forgot_resp.json()["message"]
    reset_token = token_msg.split("Reset token generated: ")[1]

    # Reset password
    reset_resp = client.post("/auth/reset-password", json={
        "token": reset_token,
        "new_password": "newpassword123"
    })
    assert reset_resp.status_code == 200
    assert reset_resp.json()["message"] == "Password has been successfully reset."

    # Verify old password fails
    login_old = client.post("/auth/login", data={"username": "reset@example.com", "password": "password123"})
    assert login_old.status_code == 401

    # Verify new password works
    login_new = client.post("/auth/login", data={"username": "reset@example.com", "password": "newpassword123"})
    assert login_new.status_code == 200

    # Verify reset token cannot be reused
    reset_resp_2 = client.post("/auth/reset-password", json={
        "token": reset_token,
        "new_password": "anotherpassword123"
    })
    assert reset_resp_2.status_code == 400
    assert reset_resp_2.json()["detail"] == "Token has already been used."


def test_reset_password_invalid_token(client, db_session):
    response = client.post("/auth/reset-password", json={
        "token": "invalid_token_string",
        "new_password": "newpassword123"
    })
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or expired reset token."
