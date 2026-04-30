"""
Active Learning API Endpoints
Provides endpoints for sampling uncertain predictions and managing active learning
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from backend.app.services.active_learning import active_learning_service
from backend.app.core.database import get_db
from backend.app.core.auth import get_current_active_user, User
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

class ActiveLearningRequest(BaseModel):
    """Request model for active learning sampling"""
    sample_type: str = "uncertain"  # uncertain, diverse, low_confidence
    limit: int = 10
    min_confidence: float = 0.0
    max_confidence: float = 0.7

@router.get("/samples")
async def get_active_learning_samples(
    sample_type: str = "uncertain",
    limit: int = 10,
    min_confidence: float = 0.0,
    max_confidence: float = 0.7,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get samples for active learning review.
    
    Args:
        sample_type: Type of sampling (uncertain, diverse, low_confidence)
        limit: Maximum number of samples
        min_confidence: Minimum confidence threshold
        max_confidence: Maximum confidence threshold
        db: Database session
        current_user: Authenticated user
        
    Returns:
        List of samples for review
    """
    try:
        if sample_type == "uncertain":
            samples = active_learning_service.get_uncertain_predictions(
                db, limit, min_confidence, max_confidence
            )
        elif sample_type == "diverse":
            samples = active_learning_service.get_diverse_samples(db, limit)
        elif sample_type == "low_confidence":
            samples = active_learning_service.get_low_confidence_entities(db, limit)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sample_type: {sample_type}. Must be one of: uncertain, diverse, low_confidence"
            )
        
        return {
            "sample_type": sample_type,
            "count": len(samples),
            "samples": samples
        }
        
    except Exception as e:
        logger.error(f"Error getting active learning samples: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_learning_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get active learning statistics.
    
    Args:
        db: Database session
        current_user: Authenticated user
        
    Returns:
        Learning statistics
    """
    try:
        stats = active_learning_service.get_learning_statistics(db)
        return stats
    except Exception as e:
        logger.error(f"Error getting learning statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
