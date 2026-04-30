"""
Active Learning Pipeline Service
Identifies uncertain predictions for human review and model improvement
"""

import logging
import os
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.app.core.database import Extraction, MedicalKnowledge
import random

logger = logging.getLogger(__name__)

# Active learning configuration
UNCERTAINTY_THRESHOLD = float(os.getenv("UNCERTAINTY_THRESHOLD", 0.7))  # Confidence threshold
SAMPLE_SIZE = int(os.getenv("ACTIVE_LEARNING_SAMPLE_SIZE", 10))  # Number of samples to return

class ActiveLearningService:
    """
    Service for active learning pipeline.
    Identifies uncertain predictions and samples for human review.
    """
    
    def __init__(self):
        self.enabled = os.getenv("ENABLE_ACTIVE_LEARNING", "true").lower() == "true"
    
    def get_uncertain_predictions(
        self,
        db: Session,
        limit: int = SAMPLE_SIZE,
        min_confidence: float = 0.0,
        max_confidence: float = UNCERTAINTY_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Get extractions with uncertain predictions for human review.
        
        Args:
            db: Database session
            limit: Maximum number of samples to return
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold (uncertain range)
            
        Returns:
            List of uncertain extractions with metadata
        """
        if not self.enabled:
            return []
        
        try:
            # Get pending extractions that haven't been verified
            extractions = db.query(Extraction).filter(
                Extraction.status == 'pending'
            ).order_by(Extraction.created_at.desc()).limit(limit * 2).all()
            
            uncertain_samples = []
            
            for extraction in extractions:
                # Calculate average confidence from entities
                entities = extraction.ai_extracted_json.get("entities", [])
                if not entities:
                    continue
                
                confidences = [e.get("confidence", 0.0) for e in entities if e.get("confidence") is not None]
                
                if not confidences:
                    # If no confidence scores, consider it uncertain
                    avg_confidence = 0.0
                else:
                    avg_confidence = sum(confidences) / len(confidences)
                
                # Check if confidence is in uncertain range
                if min_confidence <= avg_confidence <= max_confidence:
                    uncertain_samples.append({
                        "extraction_id": extraction.id,
                        "raw_text": extraction.raw_text,
                        "entities": entities,
                        "medications": extraction.ai_extracted_json.get("medications", []),
                        "avg_confidence": avg_confidence,
                        "entity_count": len(entities),
                        "created_at": extraction.created_at.isoformat()
                    })
                
                if len(uncertain_samples) >= limit:
                    break
            
            # Sort by confidence (lowest first - most uncertain)
            uncertain_samples.sort(key=lambda x: x["avg_confidence"])
            
            logger.info(f"Found {len(uncertain_samples)} uncertain samples for review")
            return uncertain_samples
            
        except Exception as e:
            logger.error(f"Error getting uncertain predictions: {e}")
            return []
    
    def get_diverse_samples(
        self,
        db: Session,
        limit: int = SAMPLE_SIZE
    ) -> List[Dict[str, Any]]:
        """
        Get diverse samples for review using stratified sampling.
        
        Args:
            db: Database session
            limit: Maximum number of samples to return
            
        Returns:
            List of diverse extractions
        """
        if not self.enabled:
            return []
        
        try:
            # Get all pending extractions
            extractions = db.query(Extraction).filter(
                Extraction.status == 'pending'
            ).order_by(Extraction.created_at.desc()).all()
            
            if not extractions:
                return []
            
            # Stratify by entity count
            samples_by_count = {
                "low": [],    # 1-3 entities
                "medium": [], # 4-7 entities
                "high": []    # 8+ entities
            }
            
            for extraction in extractions:
                entity_count = len(extraction.ai_extracted_json.get("entities", []))
                
                if entity_count <= 3:
                    samples_by_count["low"].append(extraction)
                elif entity_count <= 7:
                    samples_by_count["medium"].append(extraction)
                else:
                    samples_by_count["high"].append(extraction)
            
            # Sample from each stratum
            diverse_samples = []
            samples_per_stratum = max(1, limit // 3)
            
            for stratum in ["low", "medium", "high"]:
                available = samples_by_count[stratum]
                if available:
                    selected = random.sample(
                        available,
                        min(samples_per_stratum, len(available))
                    )
                    for extraction in selected:
                        diverse_samples.append({
                            "extraction_id": extraction.id,
                            "raw_text": extraction.raw_text,
                            "entities": extraction.ai_extracted_json.get("entities", []),
                            "medications": extraction.ai_extracted_json.get("medications", []),
                            "entity_count": len(extraction.ai_extracted_json.get("entities", [])),
                            "created_at": extraction.created_at.isoformat()
                        })
            
            # Shuffle to mix strata
            random.shuffle(diverse_samples)
            
            logger.info(f"Selected {len(diverse_samples)} diverse samples for review")
            return diverse_samples[:limit]
            
        except Exception as e:
            logger.error(f"Error getting diverse samples: {e}")
            return []
    
    def get_low_confidence_entities(
        self,
        db: Session,
        limit: int = SAMPLE_SIZE
    ) -> List[Dict[str, Any]]:
        """
        Get specific entities with low confidence scores.
        
        Args:
            db: Database session
            limit: Maximum number of entities to return
            
        Returns:
            List of low-confidence entities
        """
        if not self.enabled:
            return []
        
        try:
            # Get pending extractions
            extractions = db.query(Extraction).filter(
                Extraction.status == 'pending'
            ).limit(limit * 2).all()
            
            low_confidence_entities = []
            
            for extraction in extractions:
                entities = extraction.ai_extracted_json.get("entities", [])
                
                for entity in entities:
                    confidence = entity.get("confidence", 1.0)
                    
                    if confidence < UNCERTAINTY_THRESHOLD:
                        low_confidence_entities.append({
                            "extraction_id": extraction.id,
                            "text": entity.get("text"),
                            "label": entity.get("label"),
                            "confidence": confidence,
                            "start": entity.get("start"),
                            "end": entity.get("end"),
                            "context": extraction.raw_text[max(0, entity.get("start", 0) - 50):entity.get("end", 0) + 50]
                        })
                
                if len(low_confidence_entities) >= limit:
                    break
            
            # Sort by confidence (lowest first)
            low_confidence_entities.sort(key=lambda x: x["confidence"])
            
            logger.info(f"Found {len(low_confidence_entities)} low-confidence entities")
            return low_confidence_entities[:limit]
            
        except Exception as e:
            logger.error(f"Error getting low-confidence entities: {e}")
            return []
    
    def get_learning_statistics(self, db: Session) -> Dict[str, Any]:
        """
        Get statistics about the active learning pipeline.
        
        Args:
            db: Database session
            
        Returns:
            Learning statistics
        """
        try:
            total_extractions = db.query(Extraction).count()
            pending_extractions = db.query(Extraction).filter(Extraction.status == 'pending').count()
            verified_extractions = db.query(Extraction).filter(Extraction.status == 'verified').count()
            
            total_corrections = db.query(MedicalKnowledge).count()
            
            # Calculate verification rate
            verification_rate = 0.0
            if total_extractions > 0:
                verification_rate = verified_extractions / total_extractions
            
            return {
                "total_extractions": total_extractions,
                "pending_extractions": pending_extractions,
                "verified_extractions": verified_extractions,
                "total_corrections": total_corrections,
                "verification_rate": round(verification_rate * 100, 2),
                "active_learning_enabled": self.enabled
            }
            
        except Exception as e:
            logger.error(f"Error getting learning statistics: {e}")
            return {"error": str(e)}


# Global singleton instance
active_learning_service = ActiveLearningService()
