"""Pydantic response models for MedTex API."""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel


class Entity(BaseModel):
    text: str
    label: str           # DRUG | DOSAGE | STRENGTH | FORM | FREQUENCY | DURATION |
                         # ROUTE | DISEASE | SYMPTOM | ANATOMY | PROCEDURE | LAB |
                         # CHEMICAL | CANCER
    start: int
    end: int
    color: str
    score: Optional[float] = None
    is_fuzzy_match: Optional[bool] = None
    suggested_correction: Optional[str] = None
    normalized_name: Optional[str] = None
    rxnorm_cui: Optional[str] = None


class Medication(BaseModel):
    drug: str
    strength: Optional[str] = None
    dosage: Optional[str] = None
    form: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    route: Optional[str] = None
    instructions: Optional[str] = None


class NERResponse(BaseModel):
    original_text: str
    entities: list[Entity]
    count: int
    medications: list[Medication] = []
    medication_count: int = 0
    standardized_text: str = ""

    # Optional enrichment fields
    resolved_terms: list[dict[str, Any]] = []
    corrected_text: Optional[str] = None
    filename: Optional[str] = None
    extraction_id: Optional[int] = None

    # OCR / PHI metadata
    phi_analysis: Optional[dict[str, Any]] = None
    ocr_analysis: Optional[dict[str, Any]] = None
    vlm_structured_data: Optional[dict[str, Any]] = None

    error: Optional[str] = None