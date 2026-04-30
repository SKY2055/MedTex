"""
Model Versioning and A/B Testing Service
Manages multiple model versions and enables A/B testing
"""

import logging
import os
import random
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Model versioning configuration
ENABLE_MODEL_VERSIONING = os.getenv("ENABLE_MODEL_VERSIONING", "true").lower() == "true"
DEFAULT_MODEL_VERSION = os.getenv("DEFAULT_MODEL_VERSION", "pubmedbert")
ENABLE_AB_TESTING = os.getenv("ENABLE_AB_TESTING", "false").lower() == "true"

class ModelVersion:
    """Represents a model version with metadata"""
    
    def __init__(self, name: str, description: str, is_active: bool = True):
        self.name = name
        self.description = description
        self.is_active = is_active
        self.created_at = datetime.utcnow()
        self.performance_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "latency_ms": 0.0
        }
        self.usage_count = 0

class ModelVersioningService:
    """
    Service for managing model versions and A/B testing.
    """
    
    def __init__(self):
        self.enabled = ENABLE_MODEL_VERSIONING
        self.ab_testing_enabled = ENABLE_AB_TESTING
        self.models: Dict[str, ModelVersion] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available model versions"""
        if not self.enabled:
            return
        
        # PubMedBERT model
        self.models["pubmedbert"] = ModelVersion(
            name="pubmedbert",
            description="Microsoft PubMedBERT for clinical entity classification",
            is_active=True
        )
        
        # Ensemble model (Med7 + SciSpaCy + BC5CDR)
        self.models["ensemble"] = ModelVersion(
            name="ensemble",
            description="Ensemble of Med7, SciSpaCy, and BC5CDR models",
            is_active=True
        )
        
        # Set default model
        self.default_model = DEFAULT_MODEL_VERSION
        
        logger.info(f"Model versioning initialized with {len(self.models)} models")
    
    def get_model(self, model_name: Optional[str] = None) -> str:
        """
        Get the model to use for extraction.
        
        Args:
            model_name: Specific model name (if None, uses default or A/B testing)
            
        Returns:
            Model name to use
        """
        if not self.enabled:
            return "ensemble"
        
        # If specific model requested
        if model_name and model_name in self.models:
            self.models[model_name].usage_count += 1
            return model_name
        
        # A/B testing mode
        if self.ab_testing_enabled:
            # Randomly select between active models
            active_models = [name for name, model in self.models.items() if model.is_active]
            if active_models:
                selected = random.choice(active_models)
                self.models[selected].usage_count += 1
                logger.info(f"A/B testing selected model: {selected}")
                return selected
        
        # Use default model
        if self.default_model in self.models:
            self.models[self.default_model].usage_count += 1
            return self.default_model
        
        # Fallback to ensemble
        return "ensemble"
    
    def register_model(self, name: str, description: str, is_active: bool = True) -> bool:
        """
        Register a new model version.
        
        Args:
            name: Model name
            description: Model description
            is_active: Whether the model is active
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        if name in self.models:
            logger.warning(f"Model {name} already exists")
            return False
        
        self.models[name] = ModelVersion(name, description, is_active)
        logger.info(f"Registered new model: {name}")
        return True
    
    def update_model_metrics(
        self,
        model_name: str,
        accuracy: float = None,
        precision: float = None,
        recall: float = None,
        f1_score: float = None,
        latency_ms: float = None
    ) -> bool:
        """
        Update performance metrics for a model.
        
        Args:
            model_name: Model name
            accuracy: Accuracy metric
            precision: Precision metric
            recall: Recall metric
            f1_score: F1 score
            latency_ms: Latency in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or model_name not in self.models:
            return False
        
        model = self.models[model_name]
        
        if accuracy is not None:
            model.performance_metrics["accuracy"] = accuracy
        if precision is not None:
            model.performance_metrics["precision"] = precision
        if recall is not None:
            model.performance_metrics["recall"] = recall
        if f1_score is not None:
            model.performance_metrics["f1_score"] = f1_score
        if latency_ms is not None:
            model.performance_metrics["latency_ms"] = latency_ms
        
        logger.info(f"Updated metrics for model {model_name}")
        return True
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Model name
            
        Returns:
            Model information or None if not found
        """
        if not self.enabled or model_name not in self.models:
            return None
        
        model = self.models[model_name]
        return {
            "name": model.name,
            "description": model.description,
            "is_active": model.is_active,
            "created_at": model.created_at.isoformat(),
            "performance_metrics": model.performance_metrics,
            "usage_count": model.usage_count
        }
    
    def get_all_models(self) -> Dict[str, Any]:
        """
        Get information about all models.
        
        Returns:
            Dictionary of all models with their information
        """
        if not self.enabled:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "default_model": self.default_model,
            "ab_testing_enabled": self.ab_testing_enabled,
            "models": {
                name: self.get_model_info(name)
                for name in self.models.keys()
            }
        }
    
    def set_default_model(self, model_name: str) -> bool:
        """
        Set the default model.
        
        Args:
            model_name: Model name to set as default
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or model_name not in self.models:
            return False
        
        self.default_model = model_name
        logger.info(f"Default model set to: {model_name}")
        return True
    
    def toggle_model(self, model_name: str, is_active: bool) -> bool:
        """
        Toggle a model's active status.
        
        Args:
            model_name: Model name
            is_active: Active status
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or model_name not in self.models:
            return False
        
        self.models[model_name].is_active = is_active
        logger.info(f"Model {model_name} set to active={is_active}")
        return True


# Global singleton instance
model_versioning_service = ModelVersioningService()
