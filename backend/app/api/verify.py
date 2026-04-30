"""
Verification API — Human-in-the-Loop correction endpoint.

When a clinician edits the AI's extraction and submits it:
  1. The verified JSON is stored in the Extraction record.
  2. Every corrected entity (all categories, not just drugs) is diffed
     against the AI's original output and written to MedicalKnowledge.
  3. The NER engine's in-memory vocabulary is refreshed immediately so
     that the very next extraction benefits from the correction.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.database import Extraction, MedicalKnowledge

logger = logging.getLogger(__name__)

router = APIRouter()

# Category lookup by entity label
LABEL_TO_CATEGORY: dict[str, str] = {
    "DRUG":      "drug",
    "DISEASE":   "disease",
    "SYMPTOM":   "symptom",
    "ANATOMY":   "anatomy",
    "DOSAGE":    "dosage",
    "STRENGTH":  "dosage",
    "FREQUENCY": "frequency",
    "DURATION":  "duration",
    "ROUTE":     "route",
    "FORM":      "form",
    "PROCEDURE": "procedure",
    "LAB":       "lab",
    "CHEMICAL":  "drug",
    "CANCER":    "disease",
}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    """
    verified_json structure expected:
    {
      "original_text": "...",
      "entities": [{"text": "...", "label": "DRUG", ...}, ...],
      "medications": [{"drug": "...", "strength": "...", ...}, ...],
      "standardized_text": "..."
    }
    """
    verified_json: dict[str, Any]
    status: Optional[str] = "verified"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.put("/extractions/{extraction_id}/verify")
async def verify_extraction(
    extraction_id: int,
    request: VerifyRequest,
    db: Session = Depends(get_db),
):
    """
    Save a clinician's corrections and propagate them to the learning flywheel.

    Returns a summary of what was learned.
    """
    logger.info(f"Verify request — extraction_id={extraction_id}, status={request.status}")

    try:
        # ----------------------------------------------------------------
        # 1. Fetch or create Extraction record
        # ----------------------------------------------------------------
        extraction = db.query(Extraction).filter(Extraction.id == extraction_id).first()

        if not extraction:
            # Allow verifying an extraction that wasn't stored (e.g. cache-only path)
            extraction = Extraction(
                raw_text=request.verified_json.get("original_text", ""),
                ai_extracted_json=request.verified_json.get("ai_output", {}),
                verified_json=request.verified_json,
                status=request.status,
            )
            db.add(extraction)
        else:
            extraction.verified_json = request.verified_json
            extraction.status = request.status

        db.commit()
        db.refresh(extraction)

        # ----------------------------------------------------------------
        # 2. Diff AI output vs human corrections to find what changed
        # ----------------------------------------------------------------
        ai_entities: list[dict] = (
            (extraction.ai_extracted_json or {}).get("entities", [])
        )
        human_entities: list[dict] = request.verified_json.get("entities", [])

        # Build lookup: (start, end) → entity for the AI output
        ai_lookup: dict[tuple[int, int], dict] = {
            (e.get("start", -1), e.get("end", -1)): e
            for e in ai_entities
        }

        learned: list[dict] = []
        corrections_count = 0

        for h_ent in human_entities:
            h_text  = h_ent.get("text", "").strip()
            h_label = h_ent.get("label", "").upper()
            h_start = h_ent.get("start", -1)
            h_end   = h_ent.get("end", -1)

            if not h_text or not h_label:
                continue

            ai_ent = ai_lookup.get((h_start, h_end))

            # Case A: entity existed in AI output but text or label was corrected
            if ai_ent:
                ai_text  = (ai_ent.get("text") or "").strip()
                ai_label = (ai_ent.get("label") or "").upper()
                if ai_text.lower() == h_text.lower() and ai_label == h_label:
                    continue  # no change — skip learning
                original_term = ai_text.lower() or h_text.lower()
            else:
                # Case B: new entity added by human (AI missed it entirely)
                original_term = h_text.lower()

            category = LABEL_TO_CATEGORY.get(h_label, "entity")

            # For non-drug entities, store as "term::LABEL" so the engine
            # can inject them back into extractions via learned_entities
            if category == "entity":
                corrected_value = f"{h_text}::{h_label}"
            else:
                corrected_value = h_text

            corrections_count += _upsert_knowledge(
                db=db,
                original_term=original_term,
                corrected_term=corrected_value,
                category=category,
                extraction_id=extraction_id,
            )
            learned.append({"original": original_term, "corrected": corrected_value, "category": category})

        # ----------------------------------------------------------------
        # 3. Also learn from verified medication objects (structured fields)
        # ----------------------------------------------------------------
        for med in request.verified_json.get("medications", []):
            _learn_medication_fields(db, med, extraction_id)

        db.commit()

        # ----------------------------------------------------------------
        # 4. Refresh NER engine's in-memory knowledge immediately
        # ----------------------------------------------------------------
        try:
            from backend.app.services.nlp_engine import medtex_engine
            medtex_engine.load_knowledge_from_db(db)
            logger.info("NER engine knowledge refreshed after verification.")
        except Exception as exc:
            logger.warning(f"NER engine refresh failed (non-fatal): {exc}")

        # ----------------------------------------------------------------
        # 5. Return summary
        # ----------------------------------------------------------------
        return {
            "message": "Extraction verified successfully",
            "extraction_id": extraction_id,
            "status": request.status,
            "corrections_learned": corrections_count,
            "learned_details": learned,
        }

    except Exception as exc:
        db.rollback()
        logger.error(f"Verification failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _upsert_knowledge(
    db: Session,
    original_term: str,
    corrected_term: str,
    category: str,
    extraction_id: int,
) -> int:
    """
    Insert or update a MedicalKnowledge record.
    Returns 1 if a new record was created, 0 if updated.
    """
    original_term = original_term.lower().strip()[:200]
    corrected_term = corrected_term.strip()[:200]

    if not original_term:
        return 0

    existing = (
        db.query(MedicalKnowledge)
        .filter(MedicalKnowledge.original_term == original_term)
        .first()
    )

    if existing:
        existing.corrected_term = corrected_term
        existing.category = category
        existing.confidence += 1
        existing.updated_at = datetime.utcnow()
        # Append extraction ID to source list (deduplicated)
        sources = set((existing.source_extractions or "").split(","))
        sources.discard("")
        sources.add(str(extraction_id))
        existing.source_extractions = ",".join(sorted(sources))
        return 0
    else:
        record = MedicalKnowledge(
            original_term=original_term,
            corrected_term=corrected_term,
            category=category,
            confidence=1,
            source_extractions=str(extraction_id),
        )
        db.add(record)
        return 1


def _learn_medication_fields(db: Session, med: dict, extraction_id: int) -> None:
    """
    Learn all non-null medication fields as individual knowledge entries.
    This ensures frequency patterns, routes, forms, durations typed by
    the clinician are recorded, not just drug names.
    """
    field_to_category = {
        "drug":      "drug",
        "strength":  "dosage",
        "dosage":    "dosage",
        "form":      "form",
        "frequency": "frequency",
        "duration":  "duration",
        "route":     "route",
    }
    for field, category in field_to_category.items():
        value = (med.get(field) or "").strip()
        if value and len(value) >= 2:
            # For drug names, learn brand → verified name mapping
            if category == "drug":
                _upsert_knowledge(db, value.lower(), value, category, extraction_id)
            else:
                # Learn normalised form of the phrase
                _upsert_knowledge(db, value.lower(), value, category, extraction_id)