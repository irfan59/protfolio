# models.py
from sqlalchemy import Column, Integer, String, LargeBinary, Float, DateTime, create_engine, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True)
    employee_id = Column(String, unique=True, index=True)
    name = Column(String, index=True)
    last_attendance_time = Column(DateTime, nullable=True, index=True)  # ✅ NEW FIELD

class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"))
    vector = Column(LargeBinary, nullable=False)
    l2norm = Column(Float, default=1.0)
    person = relationship("Person")