from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

class PrescriptionExtraction(BaseModel):
    """
    Data model for tracking prescription extractions with Human-in-the-Loop verification.
    Using Pydantic for validation without requiring database connection.
    """
    id: Optional[int] = None
    raw_ocr_text: str
    ai_extracted_json: Dict[str, Any]  # What the AI thought
    verified_json: Optional[Dict[str, Any]] = None  # What the Human corrected
    status: str = 'pending'  # 'pending', 'verified', 'flagged'
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
