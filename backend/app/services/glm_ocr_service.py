"""
GLM-OCR Service for Handwritten Prescription Recognition
Integrates with Ollama for local, privacy-preserving handwritten text extraction
"""

import logging
import os
import base64
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GLM_OCR_MODEL = os.getenv("GLM_OCR_MODEL", "glm-ocr")

class GLMOCRService:
    """
    Service for using GLM-OCR model via Ollama for handwritten text recognition.
    Provides superior OCR for handwritten prescriptions with local deployment.
    """
    
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = GLM_OCR_MODEL
        self.enabled = os.getenv("ENABLE_GLM_OCR", "false").lower() == "true"
        self._available_cache = None
        self._last_check = 0
        self._cache_ttl = 60  # Cache availability for 60 seconds
    
    async def is_available(self) -> bool:
        """Check if GLM-OCR service is available via Ollama (with caching)"""
        if not self.enabled:
            return False
        
        # Return cached result if still valid
        if self._available_cache is not None:
            import time
            if time.time() - self._last_check < self._cache_ttl:
                return self._available_cache
        
        # Perform actual check
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                self._available_cache = response.status_code == 200
                self._last_check = time.time()
                return self._available_cache
        except Exception as e:
            logger.error(f"GLM-OCR service unavailable: {e}")
            self._available_cache = False
            self._last_check = time.time()
            return False
    
    async def extract_text_from_image(
        self, 
        image_bytes: bytes,
        prompt: str = "Extract text from this handwritten prescription.\nRules:\n- Keep medicine names separate (DO NOT merge words)\n- Preserve dosage like: 500 mg, 1-0-1\n- Preserve duration like: 5 days, 1 week\n- DO NOT combine unrelated words\n- Ignore non-medical items (toothbrush, paint, etc.)\nReturn clean readable text only."
    ) -> Optional[str]:
        """
        Extract text from handwritten prescription using GLM-OCR.
        
        Args:
            image_bytes: Image file bytes
            prompt: Custom prompt for the model
            
        Returns:
            Extracted text or None if failed
        """
        if not self.enabled:
            logger.warning("GLM-OCR not enabled")
            return None
        
        if not await self.is_available():
            logger.warning("GLM-OCR service not available")
            return None
        
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Prepare Ollama API request
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                extracted_text = result.get("response", "").strip()
                
                if extracted_text:
                    logger.info(f"GLM-OCR extracted {len(extracted_text)} characters")
                    return extracted_text
                else:
                    logger.warning("GLM-OCR returned empty text")
                    return None
                
        except Exception as e:
            logger.error(f"GLM-OCR extraction failed: {e}")
            return None
    
    async def extract_structured_prescription(
        self,
        image_bytes: bytes
    ) -> Optional[Dict[str, Any]]:
        """
        Extract structured prescription data using GLM-OCR with structured output.
        
        Args:
            image_bytes: Image file bytes
            
        Returns:
            Structured prescription data or None if failed
        """
        prompt = """
        Extract prescription information from this handwritten image and return as JSON:
        {
            "patient_name": "...",
            "date": "...",
            "medications": [
                {
                    "name": "...",
                    "dosage": "...",
                    "frequency": "...",
                    "duration": "..."
                }
            ],
            "doctor_name": "...",
            "raw_text": "..."
        }
        Return only valid JSON, no other text.
        """
        
        extracted_text = await self.extract_text_from_image(image_bytes, prompt)
        
        if not extracted_text:
            return None
        
        # Try to parse as JSON with repair mechanism
        try:
            import json
            import re
            structured_data = json.loads(extracted_text)
            logger.info("GLM-OCR extracted structured prescription data")
            return structured_data
        except json.JSONDecodeError:
            # Attempt JSON repair
            logger.warning("GLM-OCR response not valid JSON, attempting repair")
            try:
                # Fix common JSON issues
                repaired = extracted_text
                # Fix unquoted keys
                repaired = re.sub(r'(\w+):', r'"\1":', repaired)
                # Fix single quotes to double quotes
                repaired = repaired.replace("'", '"')
                # Remove trailing commas
                repaired = re.sub(r',\s*}', '}', repaired)
                repaired = re.sub(r',\s*]', ']', repaired)
                
                structured_data = json.loads(repaired)
                logger.info("GLM-OCR JSON repair successful")
                return structured_data
            except json.JSONDecodeError:
                logger.warning("GLM-OCR JSON repair failed, returning raw text")
                return {"raw_text": extracted_text}
    
    def get_confidence_score(self, text: str) -> float:
        """
        Get confidence score for GLM-OCR based on text characteristics.
        
        Args:
            text: Extracted text to evaluate
            
        Returns:
            Confidence score (0-1)
        """
        if not text:
            return 0.0
        
        # Length score (prefer longer, more complete extractions)
        length_score = min(len(text) / 200, 1.0)
        
        # Word count score
        word_count = len(text.split())
        word_score = min(word_count / 50, 1.0)
        
        # Medical keyword presence
        medical_keywords = ["mg", "tab", "cap", "dose", "take", "days", "week", "rx", "prescription"]
        keyword_score = sum(1 for kw in medical_keywords if kw.lower() in text.lower()) / len(medical_keywords)
        
        # Combined score
        combined = (length_score * 0.4) + (word_score * 0.3) + (keyword_score * 0.3)
        
        return min(combined, 1.0)


# Global singleton instance
glm_ocr_service = GLMOCRService()
