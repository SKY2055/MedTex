"""
RxNorm API Service for Drug Normalization
Integrates with NIH's RxNorm API for FDA-approved drug normalization
"""

import logging
import os
import requests
from typing import Optional, Dict, List
import json

logger = logging.getLogger(__name__)

# RxNorm API endpoints
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
RXNORM_API_KEY = os.getenv("RXNORM_API_KEY", "")  # Optional API key for higher rate limits

class RxNormService:
    """
    Service for interacting with RxNorm API to normalize drug names.
    RxNorm is the official FDA drug terminology standard.
    """
    
    def __init__(self):
        self.base_url = RXNORM_BASE_URL
        self.api_key = RXNORM_API_KEY
        self.cache = {}  # Simple in-memory cache for drug lookups
    
    def normalize_drug(self, drug_name: str) -> Optional[Dict]:
        """
        Normalize a drug name using RxNorm API.
        
        Args:
            drug_name: Raw drug name from prescription
            
        Returns:
            Dictionary with normalized drug information:
            {
                "rxcui": "Concept Unique Identifier",
                "name": "Normalized drug name",
                "synonym": "Closest matching synonym",
                "tty": "Term Type (e.g., SCD, SBD, PIN)",
                "rxnav_url": "Link to RxNorm entry"
            }
        """
        # Check cache first
        cache_key = drug_name.lower()
        if cache_key in self.cache:
            logger.info(f"RxNorm cache hit for: {drug_name}")
            return self.cache[cache_key]
        
        try:
            # Try exact match first
            exact_match = self._get_exact_match(drug_name)
            if exact_match:
                self.cache[cache_key] = exact_match
                return exact_match
            
            # Try approximate match
            approx_match = self._get_approximate_match(drug_name)
            if approx_match:
                self.cache[cache_key] = approx_match
                return approx_match
            
            logger.warning(f"No RxNorm match found for: {drug_name}")
            return None
            
        except Exception as e:
            logger.error(f"RxNorm API error for {drug_name}: {e}")
            return None
    
    def _get_exact_match(self, drug_name: str) -> Optional[Dict]:
        """Get exact match from RxNorm API"""
        try:
            # Use the getDrugs endpoint for exact matching
            url = f"{self.base_url}/rxcui.json?name={drug_name}"
            if self.api_key:
                url += f"&apikey={self.api_key}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get("idGroup") and data["idGroup"].get("rxnormId"):
                rxcui = data["idGroup"]["rxnormId"][0]
                
                # Get drug properties
                properties = self._get_drug_properties(rxcui)
                
                return {
                    "rxcui": rxcui,
                    "name": drug_name,
                    "normalized_name": properties.get("name", drug_name),
                    "tty": properties.get("tty", ""),
                    "rxnav_url": f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Exact match error: {e}")
            return None
    
    def _get_approximate_match(self, drug_name: str) -> Optional[Dict]:
        """Get approximate match using RxNorm spelling suggestions"""
        try:
            # Use the spellingsuggestions endpoint
            url = f"{self.base_url}/spellingsuggestions.json?name={drug_name}"
            if self.api_key:
                url += f"&apikey={self.api_key}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get("suggestionGroup") and data["suggestionGroup"].get("suggestionList"):
                suggestions = data["suggestionGroup"]["suggestionList"]["suggestion"]
                if suggestions:
                    # Use the first suggestion
                    best_match = suggestions[0]
                    rxcui = self._get_rxcui_from_name(best_match)
                    
                    if rxcui:
                        properties = self._get_drug_properties(rxcui)
                        
                        return {
                            "rxcui": rxcui,
                            "name": drug_name,
                            "normalized_name": best_match,
                            "synonym": best_match,
                            "tty": properties.get("tty", ""),
                            "rxnav_url": f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}"
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Approximate match error: {e}")
            return None
    
    def _get_rxcui_from_name(self, drug_name: str) -> Optional[str]:
        """Get RxCUI from drug name"""
        try:
            url = f"{self.base_url}/rxcui.json?name={drug_name}"
            if self.api_key:
                url += f"&apikey={self.api_key}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get("idGroup") and data["idGroup"].get("rxnormId"):
                return data["idGroup"]["rxnormId"][0]
            
            return None
            
        except Exception as e:
            logger.error(f"Get RxCUI error: {e}")
            return None
    
    def _get_drug_properties(self, rxcui: str) -> Dict:
        """Get drug properties from RxCUI"""
        try:
            url = f"{self.base_url}/rxcui/{rxcui}/properties.json"
            if self.api_key:
                url += f"?apikey={self.api_key}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get("propConceptGroup") and data["propConceptGroup"].get("propConcept"):
                props = data["propConceptGroup"]["propConcept"]
                return {prop["propName"]: prop["propValue"] for prop in props}
            
            return {}
            
        except Exception as e:
            logger.error(f"Get drug properties error: {e}")
            return {}
    
    def get_drug_interactions(self, rxcuis: List[str]) -> Optional[List[Dict]]:
        """
        Get drug-drug interactions for a list of RxCUIs.
        
        Args:
            rxcuis: List of RxNorm Concept Unique Identifiers
            
        Returns:
            List of drug interaction information
        """
        if len(rxcuis) < 2:
            return None
        
        try:
            rxcui_str = "+".join(rxcuis)
            url = f"{self.base_url}/interaction/interaction.json?rxcuis={rxcui_str}"
            if self.api_key:
                url += f"&apikey={self.api_key}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get("interactionTypeGroup"):
                interactions = []
                for group in data["interactionTypeGroup"]:
                    for interaction in group.get("interactionTypePair", []):
                        interactions.append({
                            "severity": interaction.get("severity", ""),
                            "description": interaction.get("interactionConcept", [{}])[0].get("name", "")
                        })
                return interactions
            
            return None
            
        except Exception as e:
            logger.error(f"Drug interactions error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if RxNorm API is accessible"""
        try:
            response = requests.get(f"{self.base_url}/version.json", timeout=5)
            return response.status_code == 200
        except:
            return False


# Global singleton instance
rxnorm_service = RxNormService()
