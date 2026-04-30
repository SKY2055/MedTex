"""
Celery Background Tasks
Defines asynchronous tasks for long-running operations
"""

import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from backend.app.core.celery_app import celery_app
from backend.app.core.database import SessionLocal
from backend.app.core.database import Extraction
from backend.app.core.cache import cache_service

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def process_extraction_async(self, extraction_id: int, text: str):
    """
    Process extraction asynchronously.
    
    Args:
        extraction_id: ID of the extraction to process
        text: Clinical text to process
    """
    from backend.app.services.nlp_engine import medtex_engine
    
    try:
        logger.info(f"Processing extraction {extraction_id} asynchronously")
        
        # Process the text
        result = medtex_engine.extract_clinical_summary(text)
        
        # Update database
        db = SessionLocal()
        try:
            extraction = db.query(Extraction).filter(Extraction.id == extraction_id).first()
            if extraction:
                extraction.ai_extracted_json = {
                    "entities": result["entities"],
                    "medications": result["medications"],
                    "standardized_text": result["standardized_text"]
                }
                extraction.status = 'completed'
                extraction.processed_at = datetime.utcnow()
                db.commit()
                logger.info(f"Extraction {extraction_id} completed successfully")
            else:
                logger.error(f"Extraction {extraction_id} not found")
        finally:
            db.close()
        
        return {"status": "completed", "extraction_id": extraction_id}
        
    except Exception as e:
        logger.error(f"Error processing extraction {extraction_id}: {e}")
        
        # Update status to failed
        db = SessionLocal()
        try:
            extraction = db.query(Extraction).filter(Extraction.id == extraction_id).first()
            if extraction:
                extraction.status = 'failed'
                extraction.error_message = str(e)
                db.commit()
        finally:
            db.close()
        
        raise

@celery_app.task
def cleanup_old_extractions():
    """
    Clean up old extractions from database.
    Runs daily via Celery Beat.
    """
    try:
        db = SessionLocal()
        try:
            # Delete extractions older than 30 days
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            
            old_extractions = db.query(Extraction).filter(
                Extraction.created_at < cutoff_date,
                Extraction.status == 'verified'  # Only delete verified extractions
            ).all()
            
            count = len(old_extractions)
            
            for extraction in old_extractions:
                db.delete(extraction)
            
            db.commit()
            
            logger.info(f"Cleaned up {count} old extractions")
            return {"cleaned": count}
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error cleaning up old extractions: {e}")
        raise

@celery_app.task
def collect_cache_stats():
    """
    Collect and log cache statistics.
    Runs every 30 minutes via Celery Beat.
    """
    try:
        stats = cache_service.get_stats()
        logger.info(f"Cache statistics: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Error collecting cache stats: {e}")
        return {"error": str(e)}

@celery_app.task(bind=True)
def process_batch_extractions(self, extraction_ids: list):
    """
    Process multiple extractions in batch.
    
    Args:
        extraction_ids: List of extraction IDs to process
    """
    from backend.app.services.nlp_engine import medtex_engine
    
    results = []
    
    for extraction_id in extraction_ids:
        try:
            db = SessionLocal()
            try:
                extraction = db.query(Extraction).filter(Extraction.id == extraction_id).first()
                if not extraction:
                    results.append({"extraction_id": extraction_id, "status": "not_found"})
                    continue
                
                # Process the text
                result = medtex_engine.extract_clinical_summary(extraction.raw_text)
                
                # Update database
                extraction.ai_extracted_json = {
                    "entities": result["entities"],
                    "medications": result["medications"],
                    "standardized_text": result["standardized_text"]
                }
                extraction.status = 'completed'
                extraction.processed_at = datetime.utcnow()
                db.commit()
                
                results.append({"extraction_id": extraction_id, "status": "completed"})
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error processing extraction {extraction_id}: {e}")
            results.append({"extraction_id": extraction_id, "status": "failed", "error": str(e)})
    
    logger.info(f"Batch processing complete: {len(results)} extractions")
    return results
