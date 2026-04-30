"""
Database models for MedTex clinical NER system.
Includes the HITL verification + learning flywheel schema.
"""
from sqlalchemy import (
    Column, Integer, Text, JSON, String, DateTime,
    Float, Boolean, func, Index, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
import os

load_dotenv()

Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medtex.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Dependency for getting database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


class Extraction(Base):
    """
    Stores each extraction attempt with full audit trail.
    Columns:
        raw_text        — original text / OCR output
        ai_extracted_json — what the NER pipeline extracted
        verified_json   — human-corrected version (nullable until verified)
        status          — 'pending' | 'verified' | 'flagged' | 'rejected'
        source          — 'text' | 'pdf' | 'image_printed' | 'image_handwritten'
        ocr_method      — which OCR pipeline was used (groq/easyocr/tesseract/…)
        ocr_confidence  — mean OCR confidence score (0-1)
        phi_detected    — whether PHI was found in the raw text
        document_type   — 'prescription' | 'clinical_note' | 'lab_report' | 'unknown'
    """
    __tablename__ = "extractions"

    id                = Column(Integer, primary_key=True, index=True)
    raw_text          = Column(Text, nullable=False)
    ai_extracted_json = Column(JSON, nullable=False)
    verified_json     = Column(JSON, nullable=True)
    status            = Column(String(20), default="pending", nullable=False, index=True)
    source            = Column(String(30), default="text", nullable=True)
    ocr_method        = Column(String(40), nullable=True)
    ocr_confidence    = Column(Float, nullable=True)
    phi_detected      = Column(Boolean, default=False, nullable=True)
    document_type     = Column(String(30), default="unknown", nullable=True)
    created_at        = Column(DateTime, server_default=func.now(), index=True)
    updated_at        = Column(DateTime, onupdate=func.now())

    __table_args__ = (
        Index("ix_extractions_status_created", "status", "created_at"),
    )


class MedicalKnowledge(Base):
    """
    Stores term-level corrections learned from human verification.
    The learning flywheel: each verify call populates this table;
    MedTexEngine.load_knowledge_from_db() reads it on startup + after each verify.

    Columns:
        original_term   — raw / OCR-corrupted term (lower-cased)
        corrected_term  — verified correct form
        category        — 'drug' | 'disease' | 'symptom' | 'anatomy' |
                          'dosage' | 'frequency' | 'duration' | 'route' | 'form' | 'entity'
                          For category='entity', corrected_term format: "term::LABEL"
        confidence      — count of times this correction was verified (higher = more trusted)
        source_extractions — comma-separated extraction IDs that confirmed this correction
    """
    __tablename__ = "medical_knowledge"

    id                  = Column(Integer, primary_key=True, index=True)
    original_term       = Column(String(200), nullable=False, unique=True, index=True)
    corrected_term      = Column(String(200), nullable=False)
    category            = Column(String(30), nullable=True, index=True)
    confidence          = Column(Integer, default=1)
    source_extractions  = Column(Text, nullable=True)   # e.g. "12,45,78"
    created_at          = Column(DateTime, server_default=func.now())
    updated_at          = Column(DateTime, onupdate=func.now())

    __table_args__ = (
        Index("ix_knowledge_category_confidence", "category", "confidence"),
    )