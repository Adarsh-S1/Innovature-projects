# FastAPI CSV Manager

A production-ready FastAPI application for user authentication, CSV file uploads, and data cleaning operations with Supabase integration.

## Features

- **User Authentication** — Registration, login, logout, password reset, and user profile with `bcrypt` password hashing, JWT tokens, and token blocklisting.
- **CSV File Upload** — Upload CSV files to Supabase Storage with file validation (size and type restrictions).
- **Data Cleaning** — Apply predefined data cleaning operations to uploaded CSV files using `pandas`:
  - Drop missing values
  - Fill missing values with column mean
  - Forward fill / Backward fill
  - Remove duplicate rows
- **Automated Testing** — Comprehensive test suite using `pytest` and `httpx` with a mock Supabase storage fixture.
- **CI/CD Pipeline** — GitHub Actions integration for automated testing on push and pull requests to `main`.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | FastAPI, Uvicorn |
| **Database** | Supabase PostgreSQL (via SQLAlchemy) |
| **Storage** | Supabase Storage Buckets |
| **Auth** | PyJWT, passlib[bcrypt] |
| **Data Processing** | pandas |
| **Testing** | pytest, httpx, pytest-asyncio |

## Setup

1. **Install dependencies:**
   It is recommended to use a virtual environment or Conda environment.
   ```bash
   conda activate fastapi
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   Create a `.env` file in the root directory and add the following:
   ```env
   DATABASE_URL="postgresql://postgres:<password>@<project-ref>.pooler.supabase.com:6543/postgres"
   SUPABASE_URL="https://<project-ref>.supabase.co"
   SUPABASE_KEY="<your-anon-or-service-key>"
   JWT_SECRET_KEY="<your-secret-key>"
   ```

3. **Supabase Configuration:**
   Ensure you have a corresponding database schema and a Supabase Storage bucket created (e.g., `csv-uploads`).

4. **Run the application:**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Open Swagger docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

## API Endpoints

### Authentication
| Method | Endpoint | Auth Required | Description |
|--------|----------|---------------|-------------|
| POST | `/auth/register` | ✗ | Register a new user |
| POST | `/auth/login` | ✗ | Login and obtain a JWT access token |
| GET | `/auth/me` | ✓ | Get current authenticated user details |
| POST | `/auth/logout` | ✓ | Logout and blocklist the active JWT token |
| POST | `/auth/forgot-password` | ✗ | Request a password reset logic (returns token for testing) |
| POST | `/auth/reset-password` | ✗ | Reset password using the reset token |

### File Management & Cleaning
| Method | Endpoint | Auth Required | Description |
|--------|----------|---------------|-------------|
| POST | `/files/upload` | ✓ | Upload a CSV file up to 5MB to Supabase Storage |
| POST | `/cleaning/clean` | ✓ | Perform a cleaning operation on an uploaded CSV file |

## Cleaning Operations

When calling `/cleaning/clean`, provide the `file_name` and the `operation_id`:

| ID | Operation |
|----|-----------|
| 1 | Drop rows with any missing values (`dropna`) |
| 2 | Fill missing values with column mean (`fillna(mean)`) |
| 3 | Forward fill missing values (`ffill`) |
| 4 | Backward fill missing values (`bfill`) |
| 5 | Remove duplicate rows (`drop_duplicates`) |

## Running Tests

To run the full test suite locally:

```bash
pytest -v tests/
```

The database connection and Supabase storage calls are aggressively mocked during testing to prevent polluting external services.
