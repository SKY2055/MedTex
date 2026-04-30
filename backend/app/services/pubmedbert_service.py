"""
PubMedBERT Service for Clinical Entity Classification
Uses Microsoft's PubMedBERT model for superior clinical NER performance
"""

import logging
import os
import threading
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

logger = logging.getLogger(__name__)

# Environment variable for enabling PubMedBERT
ENABLE_PUBMEDBERT = os.getenv("ENABLE_PUBMEDBERT", "false").lower() == "true"

# Model configuration
PUBMEDBERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# Alternative: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" for faster inference

class PubMedBERTService:
    """
    Singleton service for PubMedBERT-based clinical entity classification.
    Provides confidence scores and superior accuracy for clinical text.
    """
    _instance = None
    _initialized = False
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(PubMedBERTService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if PubMedBERTService._initialized:
            return
        
        self.model = None
        self.tokenizer = None
        self.ner_pipeline = None
        self._loaded = False
        self._load_lock = threading.Lock()
        
        # Clinical entity labels (will be loaded from model)
        self.label_map = {
            'DRUG': 'DRUG',
            'DOSAGE': 'DOSAGE',
            'DURATION': 'DURATION',
            'FORM': 'FORM',
            'FREQUENCY': 'FREQUENCY',
            'ROUTE': 'ROUTE',
            'STRENGTH': 'STRENGTH',
            'DISEASE': 'DISEASE',
            'SYMPTOM': 'SYMPTOM',
            'ANATOMY': 'ANATOMY',
            'PROCEDURE': 'PROCEDURE',
            'LAB': 'LAB'
        }
        
        PubMedBERTService._initialized = True
    
    def _load_model(self):
        """Lazy load PubMedBERT model on first use"""
        if self._loaded:
            return
        
        with self._load_lock:
            if self._loaded:
                return
            
            try:
                logger.info(f"Loading PubMedBERT model: {PUBMEDBERT_MODEL}")
                
                # Check if CUDA is available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Using device: {device}")
                
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(PUBMEDBERT_MODEL)
                self.model = AutoModelForTokenClassification.from_pretrained(PUBMEDBERT_MODEL)
                self.model.to(device)
                
                # Create NER pipeline
                self.ner_pipeline = pipeline(
                    "token-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple",  # Merge sub-word tokens
                    device=0 if torch.cuda.is_available() else -1
                )
                
                self._loaded = True
                logger.info("PubMedBERT model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load PubMedBERT model: {e}")
                self._loaded = False
    
    def extract_entities(self, text: str, return_confidence: bool = True) -> List[Dict]:
        """
        Extract clinical entities using PubMedBERT with confidence scores.
        
        Args:
            text: Clinical text to process
            return_confidence: Whether to include confidence scores
            
        Returns:
            List of entities with text, label, start, end, and confidence score
        """
        if not ENABLE_PUBMEDBERT:
            logger.warning("PubMedBERT not enabled. Set ENABLE_PUBMEDBERT=true in .env")
            return []
        
        self._load_model()
        
        if not self._loaded:
            return []
        
        try:
            # Run NER pipeline
            results = self.ner_pipeline(text)
            
            entities = []
            for result in results:
                entity = {
                    "text": result["word"],
                    "label": result["entity_group"],
                    "start": result["start"],
                    "end": result["end"],
                }
                
                if return_confidence:
                    entity["confidence"] = result["score"]
                
                entities.append(entity)
            
            logger.info(f"PubMedBERT extracted {len(entities)} entities")
            return entities
            
        except Exception as e:
            logger.error(f"PubMedBERT extraction failed: {e}")
            return []
    
    def get_confidence_score(self, text: str, entity_label: str) -> float:
        """
        Get confidence score for a specific entity prediction.
        
        Args:
            text: Text containing the entity
            entity_label: Predicted entity label
            
        Returns:
            Confidence score (0-1)
        """
        entities = self.extract_entities(text, return_confidence=True)
        for entity in entities:
            if entity["label"] == entity_label:
                return entity.get("confidence", 0.0)
        return 0.0
    
    def is_available(self) -> bool:
        """Check if PubMedBERT is available and loaded"""
        return ENABLE_PUBMEDBERT and self._loaded


# Global singleton instance
pubmedbert_service = PubMedBERTService()
