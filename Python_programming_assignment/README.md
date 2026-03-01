# FastAPI CSV Manager

A production-ready FastAPI application for user authentication, CSV file uploads, and data cleaning operations with Supabase integration.

## Features

- **User Authentication** — Registration & login with bcrypt password hashing and JWT tokens.
- **CSV File Upload** — Upload CSV files to Supabase Storage with file validation (type + 5 MB limit) and automatic analysis.
- **Data Cleaning** — Apply predefined cleaning operations (dropna, mean fill, ffill, bfill, deduplicate) to uploaded CSVs.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI |
| Database | Supabase PostgreSQL (via SQLAlchemy) |
| Storage | Supabase Storage Buckets |
| Auth | PyJWT + passlib[bcrypt] |
| Data Processing | pandas |

## Setup

1. **Install dependencies:**
   ```bash
   conda activate fastapi
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   Edit `.env` with your Supabase credentials and JWT secret.

3. **Create a Supabase Storage bucket** named `csv-uploads` (or whatever you set in `.env`).

4. **Run the application:**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Open Swagger docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/register` | ✗ | Register a new user |
| POST | `/auth/login` | ✗ | Login and get JWT token |
| POST | `/files/upload` | ✓ | Upload a CSV file |
| POST | `/files/clean` | ✓ | Clean a previously uploaded CSV |
| GET | `/` | ✗ | Health check |

## Cleaning Operations

| ID | Operation |
|----|-----------|
| 1 | Drop rows with any missing values |
| 2 | Fill missing values with column mean |
| 3 | Forward fill missing values |
| 4 | Backward fill missing values |
| 5 | Remove duplicate rows |
