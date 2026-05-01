# Face Recognition Backend API

A FastAPI-based face recognition and attendance system that supports multi-tenant architecture with company-specific databases.

## Features

- 🎯 **Face Detection & Recognition** using InsightFace
- 👥 **Multi-tenant Support** - Separate databases per company
- ⏰ **Attendance Cooldown** - Prevents duplicate attendance entries
- 🔄 **External API Integration** - Syncs attendance data with external systems
- 📊 **SQLite Database** - Separate database per client code
- 🔒 **CORS Enabled** - Ready for frontend integration

---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** (Python 3.9 or 3.10 recommended)
- **pip** (Python package manager)
- **Git** (for version control)

---

## Installation & Setup

### 1. Clone the Repository (if not already)

```bash
git clone <your-repo-url>
cd face_recognition_backend
```

### 2. Create Virtual Environment

Create a Python virtual environment to isolate project dependencies:

```bash
# Create virtual environment
python3 -m venv venv
```

### 3. Activate Virtual Environment

Activate the virtual environment based on your operating system:

**On macOS/Linux:**

```bash
source venv/bin/activate
```

**On Windows:**

```bash
venv\Scripts\activate
```

> **Note:** After activation, you should see `(venv)` prefix in your terminal prompt.

### 4. Install Dependencies

Install all required Python packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Installation Note:** `insightface` and `onnxruntime` may take a few minutes to install as they include large model files.

---

## Running the Application

### 1. Start the FastAPI Server

With the virtual environment activated, run:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Command Breakdown:**

- `app:app` - References the FastAPI app instance in `app.py`
- `--reload` - Auto-restart server on code changes (development mode)
- `--host 0.0.0.0` - Makes server accessible from other devices
- `--port 8000` - Server runs on port 8000

**Alternative (Production Mode):**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2. Verify Server is Running

Once started, you should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using StatReload
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 3. Access API Documentation

Open your browser and navigate to:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## API Endpoints

### 1. Enroll Employee

```http
POST /enroll
Content-Type: multipart/form-data

Fields:
- client_code: string (Company identifier, e.g., "COMP_A")
- employee_id: string (Unique employee ID)
- name: string (Employee name)
- image: file (Face image)
```

### 2. Recognize Face & Mark Attendance

```http
POST /recognize
Content-Type: multipart/form-data

Fields:
- client_code: string (Company identifier)
- device_id: string (Device identifier)
- ur: string (Base URL for sync API)
- image: file (Face image for recognition)
- min_sim: float (Optional, default: 0.70)
- punch_type: string (Punch type ID)
```

### 3. Get All Employees

```http
GET /employees/{client_code}

Path Parameter:
- client_code: string (Company identifier)
```

### 4. Reset Cooldown (Admin)

```http
POST /reset-cooldown/{client_code}/{employee_id}

Path Parameters:
- client_code: string (Company identifier)
- employee_id: string (Employee ID)
```

---

## Project Structure

```
face_recognition_backend/
│
├── app.py                 # Main FastAPI application
├── db.py                  # Database configuration & session management
├── models.py              # SQLAlchemy database models
├── face_engine.py         # Face detection & recognition engine
├── requirements.txt       # Python dependencies
│
├── embeddings/            # SQLite databases (auto-created)
│   └── company_*.db       # Separate DB per client_code
│
├── venv/                  # Virtual environment (created by you)
└── README.md              # This file
```

---

## Database Schema

### Person Table

- `id` - Primary Key
- `employee_id` - Unique employee identifier
- `name` - Employee name
- `last_attendance_time` - Latest attendance timestamp (for cooldown)

### Embedding Table

- `id` - Primary Key
- `person_id` - Foreign key to Person
- `vector` - Face embedding (binary)
- `l2norm` - L2 normalization value

---

## Configuration

### Cooldown Settings

Edit `app.py` to adjust the attendance cooldown period:

```python
COOLDOWN_SECONDS = 60  # 1 minute cooldown
```

### Detection Settings

Modify face detection size in `app.py`:

```python
engine = FaceEngine(det_size=(640, 640))
```

---

## Testing the API

### Using cURL

**Enroll Employee:**

```bash
curl -X POST "http://localhost:8000/enroll" \
  -F "client_code=COMP_A" \
  -F "employee_id=EMP001" \
  -F "name=John Doe" \
  -F "image=@/path/to/face_image.jpg"
```

**Recognize Face:**

```bash
curl -X POST "http://localhost:8000/recognize" \
  -F "client_code=COMP_A" \
  -F "device_id=DEVICE001" \
  -F "ur=https://your-api-url.com" \
  -F "punch_type=1" \
  -F "min_sim=0.70" \
  -F "image=@/path/to/face_image.jpg"
```

**Get Employees:**

```bash
curl -X GET "http://localhost:8000/employees/COMP_A"
```

---

## Troubleshooting

### Virtual Environment Issues

**Problem:** `venv` command not found

```bash
# Install venv module
sudo apt-get install python3-venv  # Ubuntu/Debian
```

**Problem:** Cannot activate virtual environment

```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
```

### Dependency Installation Issues

**Problem:** `insightface` installation fails

```bash
# Install system dependencies first (Ubuntu/Debian)
sudo apt-get install build-essential cmake
pip install --upgrade pip setuptools wheel
pip install insightface
```

**Problem:** `onnxruntime` conflicts

```bash
# Reinstall with specific version
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime
```

### Port Already in Use

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn app:app --reload --port 8080
```

---

## Stopping the Server

To stop the running server:

1. Press `CTRL + C` in the terminal where the server is running
2. Deactivate the virtual environment:
   ```bash
   deactivate
   ```

---

## Development Workflow

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Start development server
uvicorn app:app --reload --port 8000

# 3. Make code changes (server auto-reloads)

# 4. Stop server (Ctrl+C)

# 5. Deactivate virtual environment
deactivate
```

---

## Production Deployment

For production deployment, consider:

1. **Use Gunicorn with Uvicorn workers:**

   ```bash
   pip install gunicorn
   gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

2. **Use environment variables** for sensitive configuration

3. **Set up proper CORS origins** in production:

   ```python
   allow_origins=["https://yourdomain.com"]
   ```

4. **Use PostgreSQL/MySQL** instead of SQLite for production databases

5. **Configure proper logging** and error monitoring

---

## Dependencies

- **fastapi** - Modern web framework
- **uvicorn** - ASGI server
- **python-multipart** - File upload support
- **sqlalchemy** - ORM for database operations
- **requests** - HTTP client for external API calls
- **insightface** - Face recognition models
- **onnxruntime** - Inference runtime for models
- **opencv-python** - Image processing
- **numpy** - Numerical operations

---

## License

This project is proprietary software. All rights reserved.

---

## Support

For issues or questions, please contact your system administrator or development team.
