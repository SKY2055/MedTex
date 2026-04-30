"""
Extract API — Core NER extraction endpoint.

Supports:
  • POST /extract          — raw text
  • POST /upload           — file upload (PDF, image, plain text)
  • POST /batch            — multiple texts concurrently
  • POST /export/fhir/{id} — FHIR R4 bundle export
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.core.audit import audit_logger
from backend.app.core.auth import User, get_current_active_user
from backend.app.core.cache import cache_service
from backend.app.core.database import get_db
from backend.app.core.rate_limit import check_rate_limit
from backend.app.core.database import Extraction
from backend.app.models.request import TextRequest
from backend.app.models.response import NERResponse
from backend.app.services.fhir_service import fhir_service
from backend.app.services.file_processor import extract_text_from_file
from backend.app.services.nlp_engine import medtex_engine
from backend.app.services.phi_service import phi_service
import re

logger = logging.getLogger(__name__)
router = APIRouter()

ENABLE_PHI = os.getenv("ENABLE_PHI_DETECTION", "true").lower() == "true"


# ---------------------------------------------------------------------------
# Safety net: Strip entity labels from text before NLP
# ---------------------------------------------------------------------------

def _strip_entity_labels(text: str) -> str:
    """
    Safety net to strip VLM-embedded entity labels before NLP processing.
    This is a backup in case clean_ocr_text in file_processor.py misses any.
    """
    if not text:
        return ""
    
    ENTITY_LABELS = [
        "DRUG", "FORM", "STRENGTH", "DOSAGE", "FREQUENCY",
        "DURATION", "ROUTE", "DISEASE", "SYMPTOM", "ANATOMY",
        "PROCEDURE", "LAB", "CHEMICAL", "CANCER",
        "Drug", "Form", "Strength", "Dosage", "Frequency",
        "Duration", "Route", "Disease", "Symptom", "Anatomy",
        "Dosage Form",
    ]
    for label in ENTITY_LABELS:
        text = re.sub(r'\s*\b' + re.escape(label) + r'\b\s*', ' ', text)
    
    return re.sub(r' {2,}', ' ', text).strip()


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

class BatchTextRequest(BaseModel):
    texts: list[str]
    clinical_summary: bool = True


# ---------------------------------------------------------------------------
# Shared extraction logic
# ---------------------------------------------------------------------------

def _detect_phi(text: str) -> Optional[dict]:
    if not ENABLE_PHI:
        return None
    phi_data = phi_service.detect_phi(text)
    if phi_data:
        return {
            "detected_phi": phi_service.get_phi_summary(phi_data),
            "phi_details": phi_data,
            "phi_free": phi_service.is_phi_free(text),
        }
    return None


def _run_ner(text: str, clinical_summary: bool = True) -> dict:
    # Safety net: strip entity labels before NLP
    text = _strip_entity_labels(text)
    
    if clinical_summary:
        return medtex_engine.extract_clinical_summary(text)
    entities = medtex_engine.extract_entities(text)
    return {
        "entities": entities,
        "medications": [],
        "medication_count": 0,
        "standardized_text": "",
    }


def _save_extraction(db: Session, raw_text: str, result: dict,
                     source: str = "text", ocr_method: str | None = None,
                     ocr_confidence: float | None = None,
                     phi_detected: bool = False) -> Optional[int]:
    try:
        ext = Extraction(
            raw_text=raw_text,
            ai_extracted_json={
                "entities": result["entities"],
                "medications": result["medications"],
                "standardized_text": result["standardized_text"],
            },
            status="pending",
            source=source,
            ocr_method=ocr_method,
            ocr_confidence=ocr_confidence,
            phi_detected=phi_detected,
        )
        db.add(ext)
        db.commit()
        db.refresh(ext)
        return ext.id
    except Exception as exc:
        logger.error(f"Failed to save extraction record: {exc}")
        db.rollback()
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/extract", response_model=NERResponse)
async def extract_clinical_entities(
    payload: TextRequest,
    clinical_summary: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    _rate: dict = Depends(check_rate_limit),
):
    """
    Extract clinical entities from plain text.

    Returns all entity types: DRUG, DOSAGE, STRENGTH, FORM, FREQUENCY, DURATION,
    ROUTE, DISEASE, SYMPTOM, ANATOMY, PROCEDURE, LAB, CHEMICAL.
    """
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    # Cache check
    cached = cache_service.get(payload.text)
    if cached:
        return cached

    phi_analysis = _detect_phi(payload.text)

    try:
        result = _run_ner(payload.text, clinical_summary)
    except Exception as exc:
        logger.error(f"NER failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {exc}")

    phi_flag = bool(phi_analysis and not phi_analysis.get("phi_free", True))
    extraction_id = _save_extraction(
        db, payload.text, result,
        source="text", phi_detected=phi_flag
    )

    audit_logger.log_extraction(
        user=current_user.username,
        extraction_id=extraction_id or 0,
        text_length=len(payload.text),
        entity_count=len(result["entities"]),
        phi_detected=phi_flag,
    )

    response = {
        "original_text": payload.text,
        "entities": result["entities"],
        "medications": result["medications"],
        "medication_count": result["medication_count"],
        "standardized_text": result["standardized_text"],
        "phi_analysis": phi_analysis,
        "extraction_id": extraction_id,
        "count": len(result["entities"]),
    }
    cache_service.set(payload.text, response)
    return response


@router.post("/upload")
@router.post("/upload-prescription")
async def upload_document(
    file: UploadFile = File(...),
    detect_phi: bool = True,
    clinical_summary: bool = True,
    db: Session = Depends(get_db),
):
    """
    Upload a document (PDF, PNG, JPG, JPEG, TIFF, plain text) and extract
    clinical entities.

    The OCR pipeline is chosen automatically:
      handwritten image  → Groq VLM (Llama-4-Scout) → EasyOCR → Tesseract
      printed image/PDF  → PyMuPDF → EasyOCR → Tesseract
      plain text         → direct NER
    """
    logger.info(f"Upload: filename={file.filename}, content_type={file.content_type}")
    content = await file.read()

    # --- Step 1: Extract text ---
    try:
        raw_text, metadata = extract_text_from_file(
            content, file.content_type, detect_phi=detect_phi
        )
    except Exception as exc:
        logger.error(f"File extraction failed: {exc}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Could not extract text: {exc}")

    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    # --- Step 2: OCR metadata ---
    ocr_meta   = (metadata or {}).get("ocr_analysis", {})
    phi_meta   = (metadata or {}).get("phi_analysis", {})
    ocr_method = ocr_meta.get("extraction_method")
    ocr_conf   = ocr_meta.get("best_confidence")

    # --- Step 3: NER ---
    try:
        result = _run_ner(raw_text, clinical_summary)
    except Exception as exc:
        logger.error(f"NER failed on upload: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {exc}")

    # --- Step 4: Resolve OCR-corrupted medical terms ---
    resolved_terms = medtex_engine.resolve_medical_terms(raw_text)

    # --- Step 5: Persist ---
    # Infer source from content type
    ct = file.content_type or ""
    if "pdf" in ct:
        source = "pdf"
    elif "image" in ct:
        source = "image_handwritten" if (ocr_conf or 1.0) < 0.75 else "image_printed"
    else:
        source = "text"

    phi_flag = bool(phi_meta and not phi_meta.get("phi_free", True))
    extraction_id = _save_extraction(
        db, raw_text, result,
        source=source, ocr_method=ocr_method,
        ocr_confidence=ocr_conf, phi_detected=phi_flag,
    )

    response: dict[str, Any] = {
        "filename": file.filename,
        "original_text": raw_text,
        "entities": result["entities"],
        "medications": result["medications"],
        "medication_count": result["medication_count"],
        "standardized_text": result["standardized_text"],
        "count": len(result["entities"]),
        "extraction_id": extraction_id,
        "resolved_terms": resolved_terms,
    }
    if ocr_meta:
        response["ocr_analysis"] = ocr_meta
    if phi_meta:
        response["phi_analysis"] = phi_meta
    if ocr_meta.get("vlm_structured_data"):
        response["vlm_structured_data"] = ocr_meta["vlm_structured_data"]

    return response


@router.post("/batch")
async def extract_batch(
    payload: BatchTextRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Process multiple texts concurrently and return a list of results."""

    async def _process_one(index: int, text: str) -> dict:
        if not text.strip():
            return {"index": index, "success": False, "error": "Empty text"}
        try:
            cached = cache_service.get(text)
            if cached:
                return {"index": index, "success": True, "result": cached}

            phi_analysis = _detect_phi(text)
            result = _run_ner(text, payload.clinical_summary)
            phi_flag = bool(phi_analysis and not phi_analysis.get("phi_free", True))
            extraction_id = _save_extraction(db, text, result, phi_detected=phi_flag)
            data = {
                "original_text": text,
                "entities": result["entities"],
                "medications": result["medications"],
                "medication_count": result["medication_count"],
                "standardized_text": result["standardized_text"],
                "phi_analysis": phi_analysis,
                "extraction_id": extraction_id,
                "count": len(result["entities"]),
            }
            cache_service.set(text, data)
            return {"index": index, "success": True, "result": data}
        except Exception as exc:
            logger.error(f"Batch item {index} failed: {exc}")
            return {"index": index, "success": False, "error": str(exc)}

    tasks = [_process_one(i, t) for i, t in enumerate(payload.texts)]
    return await asyncio.gather(*tasks)


@router.post("/export/fhir/{extraction_id}")
async def export_fhir(
    extraction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Export an extraction as a FHIR R4 Bundle."""
    extraction = db.query(Extraction).filter(Extraction.id == extraction_id).first()
    if not extraction:
        raise HTTPException(status_code=404, detail="Extraction not found.")
    try:
        bundle = fhir_service.convert_to_fhir(extraction.ai_extracted_json)
        audit_logger.log_action(
            user=current_user.username,
            action="EXPORT_FHIR",
            resource=f"extraction_{extraction_id}",
            details={"format": "FHIR R4 Bundle"},
        )
        return bundle
    except Exception as exc:
        logger.error(f"FHIR export failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"FHIR export failed: {exc}")