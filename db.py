# db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

DB_FOLDER = "embeddings"
os.makedirs(DB_FOLDER, exist_ok=True)

def get_engine(client_code: str):
    """Get or create SQLAlchemy engine for specific client"""
    db_path = os.path.join(DB_FOLDER, f"company_{client_code}.db")
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False}
    )
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    return engine

def get_session(client_code: str):
    """Get database session for specific client"""
    engine = get_engine(client_code)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal()