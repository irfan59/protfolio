# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import cv2
from sqlalchemy.orm import Session
from db import get_session
from models import Person, Embedding
from face_engine import FaceEngine
import requests
from datetime import datetime, timedelta  # ✅ ADD THIS

app = FastAPI(title="Face Attendance API - Multi-Tenant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"],
)

engine = FaceEngine(det_size=(640, 640))

# ✅ ADD CONFIGURATION
COOLDOWN_SECONDS = 60  # 1 minute cooldown

# ---------- Better Response Models ----------
class MatchResult(BaseModel):
    person_id: Optional[int]
    employee_id: Optional[str] = None
    name: str
    similarity: float
    bbox: list
    status: str = "success"  # ✅ NEW: success, cooldown, unknown
    message: Optional[str] = None  # ✅ NEW: for cooldown messages


class RecognizeResponse(BaseModel):
    matches: List[MatchResult]

class EmployeeInfo(BaseModel):
    id: int
    employee_id: str
    name: str
    embeddings_count: int
    last_attendance: Optional[datetime] = None  # ✅ NEW
    
    class Config:
        from_attributes = True

class EmployeeListResponse(BaseModel):
    client_code: str
    total_employees: int
    employees: List[EmployeeInfo]


# ---------- Error Handler ----------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"❌ Validation Error Details: {exc.errors()}")
    print(f"❌ Request Body: {exc.body}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "message": "Invalid request format. Check your form-data fields."
        }
    )

# ---------- helpers ----------
def read_image_to_bgr(file_bytes: bytes):
    img = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)

def emb_to_bytes(emb: np.ndarray) -> bytes:
    return emb.tobytes(order="C")

def bytes_to_emb(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)

# ✅ NEW HELPER FUNCTION
def check_cooldown(person: Person, cooldown_seconds: int = COOLDOWN_SECONDS) -> tuple[bool, Optional[int]]:
    """
    Check if employee is in cooldown period
    
    Returns:
        (is_in_cooldown: bool, remaining_seconds: int or None)
    """
    if person.last_attendance_time is None:
        return False, None
    
    now = datetime.utcnow()
    time_diff = now - person.last_attendance_time
    
    if time_diff < timedelta(seconds=cooldown_seconds):
        remaining = cooldown_seconds - int(time_diff.total_seconds())
        return True, remaining
    
    return False, None

# ---------- endpoints ----------
@app.post("/enroll")
async def enroll(
    client_code: str = Form(...),
    employee_id: str = Form(...), 
    name: str = Form(...), 
    image: UploadFile = File(...)
):
    """
    Enroll employee for specific company
    - client_code: Unique company identifier (e.g., "COMP_A")
    """
    if not client_code or not client_code.strip():
        raise HTTPException(400, "client_code is required")
    
    db: Session = get_session(client_code)
    print(f"Enrolling: {employee_id} ({name}) for company: {client_code}")
    
    try:
        bgr = read_image_to_bgr(await image.read())
        faces = engine.detect_and_embed(bgr)
        
        if len(faces) != 1:
            return {"ok": False, "msg": f"Expected 1 face, got {len(faces)}"}
        
        emb = faces[0]["embedding"]
        
        person = db.query(Person).filter_by(employee_id=employee_id).first()
        if not person:
            person = Person(employee_id=employee_id, name=name)
            db.add(person)
            db.flush()
        
        db.add(Embedding(person_id=person.id, vector=emb_to_bytes(emb), l2norm=1.0))
        db.commit()
        
        return {
            "ok": True, 
            "client_code": client_code,
            "employee_id": employee_id, 
            "name": person.name
        }
    finally:
        db.close()

@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    client_code: str = Form(...),
    image: UploadFile = File(...), 
    device_id: str = Form(...),
    ur: str = Form(...),
    min_sim: float = Form(0.70),
    punch_type: str = Form(...),
):
    """
    Recognize faces for specific company with cooldown protection
    - client_code: Company identifier
    - device_id: Device identifier
    - ur: Base URL for sync API
    - min_sim: Cosine similarity threshold (0..1)
    """
    print(f"📥 Received: client_code={client_code}, device_id={device_id}, ur={ur}, min_sim={min_sim}")
    
    if not client_code or not client_code.strip():
        raise HTTPException(400, "client_code is required")
    
    db: Session = get_session(client_code)
    print(f"Recognizing for company: {client_code}")
    
    try:
        bgr = read_image_to_bgr(await image.read())
        faces = engine.detect_and_embed(bgr)
        results = []
        
        if not faces:
            return {"matches": []}
        
        db_embs = db.query(Embedding).all()
        print(f"Searching against {len(db_embs)} embeddings for {client_code}")
        
        for f in faces:
            probe = f["embedding"]
            best_score, best_row = -1.0, None
            
            for row in db_embs:
                sim = float(np.dot(bytes_to_emb(row.vector), probe))
                if sim > best_score:
                    best_score, best_row = sim, row
            
            if best_row and best_score >= min_sim:
                person = db.query(Person).get(best_row.person_id)
                
                # ✅ CHECK COOLDOWN BEFORE API CALL
                is_cooldown, remaining_seconds = check_cooldown(person)
                
                if is_cooldown:
                    print(f"⏳ Cooldown active for {person.employee_id}: {remaining_seconds}s remaining")
                    results.append({
                        "person_id": person.id,
                        "employee_id": person.employee_id,
                        "name": person.name,
                        "similarity": round(best_score, 4),
                        "bbox": f["bbox"],
                        "status": "cooldown",
                        "message": f"Please wait {remaining_seconds} seconds before next attendance"
                    })
                    continue  # ✅ SKIP API CALL
                
                # Call external API only if NOT in cooldown
                try:
                    api_url = f"{ur}/api/sync/insertNewmatrixrawsync"
                    payload = {
                        "MatrixRawSync": {
                            "EmployeeId": str(person.employee_id),
                            "PunchTypeId": punch_type
                        },
                        "ClientCode": client_code,
                        "DeviceId": device_id
                    }
                    
                    print(f"🔄 Calling API: {api_url} with payload: {payload}")
                    response = requests.post(api_url, json=payload, timeout=100)
                    response_data = response.json()
                    print(f"{response_data}")
                    
                    # Check if status is 'success'
                    if response.status_code == 200 and response_data.get("status") == "success":
                        print(f"✓ Successfully synced attendance for employee {person.employee_id}")
                        
                        # ✅ UPDATE LAST ATTENDANCE TIME
                        person.last_attendance_time = datetime.utcnow()
                        db.commit()
                        
                        results.append({
                            "person_id": person.id,
                            "employee_id": person.employee_id,
                            "name": person.name,
                            "similarity": round(best_score, 4),
                            "bbox": f["bbox"],
                            "status": "success",
                            "message": "Attendance marked successfully"
                        })
                    else:
                        # Return error with message from API response
                        error_message = response_data.get("message", f"API sync failed: Status {response.status_code}")
                        print(f"✗ {error_message}")
                        raise HTTPException(status_code=400, detail=error_message)
                        
                except requests.exceptions.Timeout:
                    error_message = f"API sync timeout for employee {person.employee_id}"
                    print(f"✗ {error_message}")
                    raise HTTPException(status_code=408, detail=error_message)
                except requests.exceptions.ConnectionError:
                    error_message = f"API connection error for employee {person.employee_id}"
                    print(f"✗ {error_message}")
                    raise HTTPException(status_code=503, detail=error_message)
                except HTTPException:
                    raise
                except Exception as api_error:
                    error_message = f"API error: {str(api_error)}"
                    print(f"✗ {error_message}")
                    raise HTTPException(status_code=500, detail=error_message)
                
            else:
                results.append({
                    "person_id": None,
                    "employee_id": None,
                    "name": "Unknown", 
                    "similarity": round(best_score, 4), 
                    "bbox": f["bbox"],
                    "status": "unknown",
                    "message": "Face not recognized"
                })
        
        return {"matches": results}
    finally:
        db.close()

@app.get("/employees/{client_code}", response_model=EmployeeListResponse)
async def get_employees(client_code: str):
    """
    Get list of all employees for a specific company
    
    Args:
        client_code: Company identifier (e.g., "COMP_A")
    
    Returns:
        List of employees with their details
    """
    if not client_code or not client_code.strip():
        raise HTTPException(400, "client_code is required")
    
    db: Session = get_session(client_code)
    
    try:
        persons = db.query(Person).all()
        
        employees_data = []
        for person in persons:
            emb_count = db.query(Embedding).filter_by(person_id=person.id).count()
            
            employees_data.append({
                "id": person.id,
                "employee_id": person.employee_id,
                "name": person.name,
                "embeddings_count": emb_count,
                "last_attendance": person.last_attendance_time  # ✅ NEW
            })
        
        return {
            "client_code": client_code,
            "total_employees": len(employees_data),
            "employees": employees_data
        }
    
    finally:
        db.close()

# ✅ OPTIONAL: Add endpoint to manually reset cooldown (for testing/admin)
@app.post("/reset-cooldown/{client_code}/{employee_id}")
async def reset_cooldown(client_code: str, employee_id: str):
    """Reset cooldown for specific employee (admin function)"""
    db: Session = get_session(client_code)
    try:
        person = db.query(Person).filter_by(employee_id=employee_id).first()
        if not person:
            raise HTTPException(404, "Employee not found")
        
        person.last_attendance_time = None
        db.commit()
        
        return {
            "ok": True,
            "message": f"Cooldown reset for {employee_id}"
        }
    finally:
        db.close()