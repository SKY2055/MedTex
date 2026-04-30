"""
FHIR Export Service for Clinical Interoperability
Converts extraction results to FHIR (Fast Healthcare Interoperability Resources) format
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class FHIRService:
    """
    Service for converting clinical extraction results to FHIR format.
    Supports MedicationRequest, Condition, and Observation resources.
    """
    
    def __init__(self):
        self.fhir_base_url = "https://hapi.fhir.org/baseR4"  # Public FHIR server for reference
    
    def convert_to_fhir(self, extraction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert extraction result to FHIR Bundle.
        
        Args:
            extraction_data: Extraction result with entities and medications
            
        Returns:
            FHIR Bundle containing MedicationRequest, Condition, and Observation resources
        """
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "entry": []
        }
        
        # Convert medications to MedicationRequest
        medications = extraction_data.get("medications", [])
        for med in medications:
            medication_request = self._create_medication_request(med)
            if medication_request:
                bundle["entry"].append({
                    "resource": medication_request,
                    "request": {
                        "method": "POST",
                        "url": "MedicationRequest"
                    }
                })
        
        # Convert disease entities to Condition resources
        entities = extraction_data.get("entities", [])
        diseases = [e for e in entities if e.get("label") == "DISEASE"]
        for disease in diseases:
            condition = self._create_condition(disease)
            if condition:
                bundle["entry"].append({
                    "resource": condition,
                    "request": {
                        "method": "POST",
                        "url": "Condition"
                    }
                })
        
        # Convert lab entities to Observation resources
        labs = [e for e in entities if e.get("label") == "LAB"]
        for lab in labs:
            observation = self._create_observation(lab)
            if observation:
                bundle["entry"].append({
                    "resource": observation,
                    "request": {
                        "method": "POST",
                        "url": "Observation"
                    }
                })
        
        logger.info(f"Converted to FHIR Bundle with {len(bundle['entry'])} resources")
        return bundle
    
    def _create_medication_request(self, medication: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create FHIR MedicationRequest from medication data.
        
        Args:
            medication: Medication data from extraction
            
        Returns:
            FHIR MedicationRequest resource
        """
        try:
            medication_request = {
                "resourceType": "MedicationRequest",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {
                    "coding": [{
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": medication.get("rxnorm_cui", ""),
                        "display": medication.get("drug", "")
                    }],
                    "text": medication.get("drug", "")
                },
                "dosageInstruction": []
            }
            
            # Add dosage if available
            if medication.get("dosage"):
                medication_request["dosageInstruction"].append({
                    "text": medication.get("dosage"),
                    "doseAndRate": [{
                        "doseQuantity": {
                            "value": self._extract_dose_value(medication.get("dosage", "")),
                            "unit": self._extract_dose_unit(medication.get("dosage", "")),
                            "system": "http://unitsofmeasure.org",
                            "code": self._extract_dose_unit(medication.get("dosage", ""))
                        }
                    }]
                })
            
            # Add frequency if available
            if medication.get("frequency"):
                if medication_request["dosageInstruction"]:
                    medication_request["dosageInstruction"][0]["timing"] = {
                        "code": {
                            "coding": [{
                                "system": "http://terminology.hl7.org/CodeSystem/v3-GTSAbbreviation",
                                "code": self._map_frequency_to_code(medication.get("frequency", "")),
                                "display": medication.get("frequency")
                            }]
                        }
                    }
            
            # Add route if available
            if medication.get("route"):
                if medication_request["dosageInstruction"]:
                    medication_request["dosageInstruction"][0]["route"] = {
                        "coding": [{
                            "system": "http://snomed.info/sct",
                            "code": self._map_route_to_code(medication.get("route", "")),
                            "display": medication.get("route")
                        }]
                    }
            
            return medication_request
            
        except Exception as e:
            logger.error(f"Error creating MedicationRequest: {e}")
            return None
    
    def _create_condition(self, disease: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create FHIR Condition from disease entity.
        
        Args:
            disease: Disease entity from extraction
            
        Returns:
            FHIR Condition resource
        """
        try:
            condition = {
                "resourceType": "Condition",
                "clinicalStatus": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                        "display": "Active"
                    }]
                },
                "verificationStatus": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "confirmed",
                        "display": "Confirmed"
                    }]
                },
                "code": {
                    "text": disease.get("text", "")
                }
            }
            
            return condition
            
        except Exception as e:
            logger.error(f"Error creating Condition: {e}")
            return None
    
    def _create_observation(self, lab: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create FHIR Observation from lab entity.
        
        Args:
            lab: Lab entity from extraction
            
        Returns:
            FHIR Observation resource
        """
        try:
            observation = {
                "resourceType": "Observation",
                "status": "final",
                "code": {
                    "text": lab.get("text", "")
                },
                "valueString": lab.get("text", "")
            }
            
            return observation
            
        except Exception as e:
            logger.error(f"Error creating Observation: {e}")
            return None
    
    def _extract_dose_value(self, dosage: str) -> float:
        """Extract numeric dose value from dosage string"""
        import re
        match = re.search(r'(\d+\.?\d*)', dosage)
        return float(match.group(1)) if match else 0.0
    
    def _extract_dose_unit(self, dosage: str) -> str:
        """Extract dose unit from dosage string"""
        import re
        match = re.search(r'(\d+\.?\d*)\s*([a-zA-Z]+)', dosage)
        return match.group(2) if match else "mg"
    
    def _map_frequency_to_code(self, frequency: str) -> str:
        """Map frequency text to FHIR timing code"""
        freq_map = {
            "daily": "BID",
            "twice daily": "BID",
            "three times daily": "TID",
            "four times daily": "QID",
            "as needed": "PRN",
            "once daily": "QD"
        }
        freq_lower = frequency.lower()
        for key, code in freq_map.items():
            if key in freq_lower:
                return code
        return "PRN"  # Default to as needed
    
    def _map_route_to_code(self, route: str) -> str:
        """Map route text to SNOMED CT code"""
        route_map = {
            "oral": "26643006",
            "intravenous": "47625008",
            "topical": "421526004",
            "subcutaneous": "34206005",
            "inhaled": "410942007"
        }
        route_lower = route.lower()
        for key, code in route_map.items():
            if key in route_lower:
                return code
        return "26643006"  # Default to oral
    
    def export_to_json(self, fhir_bundle: Dict[str, Any]) -> str:
        """
        Export FHIR bundle to JSON string.
        
        Args:
            fhir_bundle: FHIR Bundle
            
        Returns:
            JSON string
        """
        import json
        return json.dumps(fhir_bundle, indent=2)


# Global singleton instance
fhir_service = FHIRService()
