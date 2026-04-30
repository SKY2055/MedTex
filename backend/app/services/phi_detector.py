import re
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class PHIFinding:
    type: str
    text: str
    start: int
    end: int
    confidence: float = 1.0

class PHIDetector:
    """
    Detects Protected Health Information (PHI) in clinical text.
    Implements HIPAA-compliant entity identification for compliance workflows.
    """
    
    # Common PHI patterns
    PATTERNS = {
        "SSN": {
            "pattern": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
            "description": "Social Security Number"
        },
        "PHONE": {
            "pattern": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "description": "Phone Number"
        },
        "EMAIL": {
            "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "description": "Email Address"
        },
        "MRN": {
            "pattern": r"\b(?:MRN|Medical Record Number|Patient ID|ID Number)[\s:#]*(\d{3,10})\b",
            "description": "Medical Record Number"
        },
        "DATE_OF_BIRTH": {
            "pattern": r"\b(?:DOB|Date of Birth|Birth Date)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
            "description": "Date of Birth"
        },
        "ADDRESS": {
            "pattern": r"\b\d+\s+[A-Za-z0-9\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way)\b",
            "description": "Street Address"
        },
        "FULL_NAME": {
            "pattern": r"\b(?:Dr|Mr|Mrs|Ms|Miss)?\s*[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+(?:Jr|Sr|MD|PhD))?\b",
            "description": "Patient/Provider Name"
        }
    }
    
    def __init__(self, redact: bool = False):
        """
        Args:
            redact: If True, return text with PHI replaced by [REDACTED]
        """
        self.redact = redact
        self.compiled_patterns = {
            name: re.compile(info["pattern"], re.IGNORECASE)
            for name, info in self.PATTERNS.items()
        }
    
    def detect_phi(self, text: str) -> List[PHIFinding]:
        """
        Detect all PHI patterns in text.
        
        Returns:
            List of PHIFinding objects with type, text, position, and confidence
        """
        findings = []
        
        for phi_type, pattern in self.compiled_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                # Calculate confidence based on pattern specificity
                confidence = self._calculate_confidence(phi_type, match.group())
                
                findings.append(PHIFinding(
                    type=phi_type,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence
                ))
        
        # Sort by position
        findings.sort(key=lambda x: x.start)
        
        # Remove overlapping findings (keep higher confidence)
        findings = self._remove_overlaps(findings)
        
        return findings
    
    def redact_phi(self, text: str, findings: Optional[List[PHIFinding]] = None) -> str:
        """
        Redact detected PHI from text.
        
        Returns:
            Text with PHI replaced by [REDACTED-<TYPE>]
        """
        if findings is None:
            findings = self.detect_phi(text)
        
        # Replace from end to start to preserve positions
        redacted_text = text
        for finding in sorted(findings, key=lambda x: x.start, reverse=True):
            placeholder = f"[REDACTED-{finding.type}]"
            redacted_text = (
                redacted_text[:finding.start] + 
                placeholder + 
                redacted_text[finding.end:]
            )
        
        return redacted_text
    
    def get_phi_report(self, text: str) -> Dict:
        """
        Generate a comprehensive PHI detection report.
        
        Returns:
            Dict with findings count, types detected, and risk assessment
        """
        findings = self.detect_phi(text)
        
        phi_types = {}
        for finding in findings:
            phi_types[finding.type] = phi_types.get(finding.type, 0) + 1
        
        # Risk assessment
        risk_level = "LOW"
        if len(findings) >= 5:
            risk_level = "HIGH"
        elif len(findings) >= 2:
            risk_level = "MEDIUM"
        
        return {
            "phi_detected": len(findings) > 0,
            "total_findings": len(findings),
            "phi_types": phi_types,
            "risk_level": risk_level,
            "findings": [
                {
                    "type": f.type,
                    "text": f"{f.text[:3]}..." if len(f.text) > 6 else f.text,  # Partial masking
                    "confidence": round(f.confidence, 2)
                }
                for f in findings
            ],
            "recommendation": "Review and redact before sharing" if findings else "No PHI detected"
        }
    
    def _calculate_confidence(self, phi_type: str, text: str) -> float:
        """Calculate confidence score based on pattern specificity."""
        base_confidence = {
            "SSN": 0.95,
            "PHONE": 0.90,
            "EMAIL": 0.98,
            "MRN": 0.85,
            "DATE_OF_BIRTH": 0.80,
            "ADDRESS": 0.75,
            "FULL_NAME": 0.70
        }
        
        confidence = base_confidence.get(phi_type, 0.70)
        
        # Adjust based on text length (longer matches = more specific)
        if len(text) > 20:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _remove_overlaps(self, findings: List[PHIFinding]) -> List[PHIFinding]:
        """Remove overlapping findings, keeping higher confidence ones."""
        if not findings:
            return findings
        
        filtered = []
        last_end = -1
        
        for finding in findings:
            if finding.start >= last_end:
                filtered.append(finding)
                last_end = finding.end
            else:
                # Overlapping - compare with last kept finding
                if filtered and finding.confidence > filtered[-1].confidence:
                    filtered[-1] = finding
                    last_end = finding.end
        
        return filtered


# Global detector instance
phi_detector = PHIDetector()
