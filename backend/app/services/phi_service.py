"""
HIPAA PHI Detection and Redaction Service
Detects and redacts Protected Health Information for HIPAA compliance
"""

import logging
import re
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class PHIService:
    """
    Service for detecting and redacting Protected Health Information (PHI)
    in clinical text to ensure HIPAA compliance.
    """
    
    def __init__(self):
        # PHI patterns for detection
        self.phi_patterns = {
            "SSN": r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
            "PHONE": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "DATE": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            "MRN": r'\b(MRN|Medical Record #|MR#)\s*:?\s*\d+\b',
            "DOB": r'\b(DOB|Date of Birth|Birth Date)\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            "ADDRESS": r'\b\d+\s+[A-Za-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
            "ZIP": r'\b\d{5}(-\d{4})?\b',
        }
        
        # Common names that might be PHI (first + last name pattern)
        self.name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        
        # Medical identifiers that might be PHI
        self.medical_id_pattern = r'\b(ID|Patient ID|Account #)\s*:?\s*\d+\b'
    
    def detect_phi(self, text: str) -> Dict[str, List[Dict]]:
        """
        Detect PHI in text and return locations and types.
        
        Args:
            text: Clinical text to analyze
            
        Returns:
            Dictionary with PHI types as keys and lists of matches as values:
            {
                "SSN": [{"text": "123-45-6789", "start": 10, "end": 22}],
                "PHONE": [{"text": "555-123-4567", "start": 30, "end": 42}],
                ...
            }
        """
        phi_detected = {}
        
        for phi_type, pattern in self.phi_patterns.items():
            matches = []
            for match in re.finditer(pattern, text):
                matches.append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
            
            if matches:
                phi_detected[phi_type] = matches
                logger.warning(f"Detected {len(matches)} {phi_type} patterns")
        
        # Detect potential names (heuristic-based)
        name_matches = []
        for match in re.finditer(self.name_pattern, text):
            # Filter out common medical terms that look like names
            text_lower = match.group().lower()
            if text_lower not in ["heart attack", "blood pressure", "chest pain", "high blood"]:
                name_matches.append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.5  # Lower confidence for names
                })
        
        if name_matches:
            phi_detected["NAME"] = name_matches
            logger.warning(f"Detected {len(name_matches)} potential NAME patterns")
        
        # Detect medical IDs
        medical_id_matches = []
        for match in re.finditer(self.medical_id_pattern, text):
            medical_id_matches.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end()
            })
        
        if medical_id_matches:
            phi_detected["MEDICAL_ID"] = medical_id_matches
            logger.warning(f"Detected {len(medical_id_matches)} MEDICAL_ID patterns")
        
        return phi_detected
    
    def redact_phi(self, text: str, phi_data: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        Redact PHI from text by replacing with placeholders.
        
        Args:
            text: Clinical text to redact
            phi_data: Optional pre-computed PHI data (if None, will detect)
            
        Returns:
            Tuple of (redacted_text, phi_data)
        """
        if phi_data is None:
            phi_data = self.detect_phi(text)
        
        redacted_text = text
        phi_count = sum(len(matches) for matches in phi_data.values())
        
        if phi_count == 0:
            return text, phi_data
        
        # Sort matches by position in reverse order to preserve indices
        all_matches = []
        for phi_type, matches in phi_data.items():
            for match in matches:
                all_matches.append({
                    "type": phi_type,
                    "text": match["text"],
                    "start": match["start"],
                    "end": match["end"]
                })
        
        all_matches.sort(key=lambda x: x["start"], reverse=True)
        
        # Redact each match
        for match in all_matches:
            placeholder = f"[{match['type']}_REDACTED]"
            redacted_text = redacted_text[:match["start"]] + placeholder + redacted_text[match["end"]:]
        
        logger.info(f"Redacted {phi_count} PHI instances")
        return redacted_text, phi_data
    
    def get_phi_summary(self, phi_data: Dict) -> Dict:
        """
        Get summary statistics of detected PHI.
        
        Args:
            phi_data: PHI detection results
            
        Returns:
            Summary statistics
        """
        summary = {
            "total_phi_count": sum(len(matches) for matches in phi_data.values()),
            "phi_types": list(phi_data.keys()),
            "phi_by_type": {phi_type: len(matches) for phi_type, matches in phi_data.items()}
        }
        
        return summary
    
    def is_phi_free(self, text: str) -> bool:
        """
        Check if text is free of detectable PHI.
        
        Args:
            text: Clinical text to check
            
        Returns:
            True if no PHI detected, False otherwise
        """
        phi_data = self.detect_phi(text)
        return len(phi_data) == 0


# Global singleton instance
phi_service = PHIService()
