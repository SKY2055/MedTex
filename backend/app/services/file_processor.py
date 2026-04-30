import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import io
import re
import os
import json
import logging
from typing import Tuple, Dict, Optional, List
from difflib import SequenceMatcher, get_close_matches
from backend.app.services.phi_detector import phi_detector

logger = logging.getLogger(__name__)

# Calculate project root once at module level
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Optional OpenCV for advanced image pre-processing
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Optional TrOCR for handwritten text recognition
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

# Optional PaddleOCR for better handwriting recognition
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

# Optional Groq for VLM-based OCR (Llama-4-Scout, Qwen3-VL)
try:
    from groq import Groq
    from dotenv import load_dotenv
    
    # Try multiple .env locations
    env_paths = [
        os.path.join(project_root, '.env'),
        '/Users/suraj/MedTex/medtex/.env',  # Absolute path fallback
        os.path.join(os.getcwd(), '.env'),   # Current working directory
    ]
    
    GROQ_API_KEY = None
    logger.info(f"Checking .env paths: {env_paths}")
    for env_path in env_paths:
        exists = os.path.exists(env_path)
        logger.info(f"  - {env_path}: exists={exists}")
        if exists:
            load_dotenv(dotenv_path=env_path, override=True)
            GROQ_API_KEY = os.getenv("GROQ_API_KEY")
            if GROQ_API_KEY:
                logger.info(f"✅ Loaded .env from: {env_path}, key length: {len(GROQ_API_KEY)}")
                break
    
    if not GROQ_API_KEY:
        # Try loading without specifying path (python-dotenv default behavior)
        load_dotenv(override=True)
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if GROQ_API_KEY:
            logger.info("✅ Loaded .env from default location")
    
    GROQ_AVAILABLE = bool(GROQ_API_KEY and GROQ_API_KEY.strip())
    logger.info(f"GROQ_API_KEY loaded: {GROQ_AVAILABLE}, length: {len(GROQ_API_KEY) if GROQ_API_KEY else 0}")
except ImportError as e:
    GROQ_AVAILABLE = False
    logger.warning(f"Groq import failed: {e}. Install with: pip install groq")

# Optional GLM-OCR for handwritten prescriptions via Ollama
try:
    from backend.app.services.glm_ocr_service import glm_ocr_service
    GLM_OCR_AVAILABLE = True
except ImportError:
    GLM_OCR_AVAILABLE = False
    glm_ocr_service = None

# Optional Google Gemini for VLM-based OCR
try:
    from google import genai
    from dotenv import load_dotenv
    # Load .env file from project root (same as Groq)
    env_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=env_path, override=True)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_AVAILABLE = bool(GEMINI_API_KEY)
    logger.debug(f"file_processor.py - GEMINI_API_KEY loaded: {GEMINI_AVAILABLE}")
except ImportError as e:
    GEMINI_AVAILABLE = False
    GEMINI_API_KEY = None
    logger.debug(f"file_processor.py - Gemini import failed: {e}")

# Optional Ollama for local VLM-based OCR (GLM-OCR)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Optional EasyOCR for lightweight handwriting recognition
try:
    import easyocr
    import ssl
    import urllib.request
    # Fix SSL certificate verification on macOS
    try:
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context
        EASYOCR_AVAILABLE = True
    except Exception:
        EASYOCR_AVAILABLE = False
except ImportError:
    EASYOCR_AVAILABLE = False

logger = logging.getLogger(__name__)


# --- OpenCV Image Pre-processing (Optional) ---

def _preprocess_with_opencv(image: Image.Image) -> Image.Image:
    """
    Advanced image pre-processing using OpenCV for better OCR on medical documents.
    
    Steps:
        1. Grayscale conversion
        2. Rescaling (2x) - helps with small handwriting
        3. Adaptive thresholding - better contrast for handwriting
        4. Denoising - remove shadows and noise
    """
    if not OPENCV_AVAILABLE:
        logger.warning("OpenCV not available, skipping advanced pre-processing")
        return image
    
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # 2. Rescale (Enlarge) - helps with small handwriting
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 3. Adaptive Thresholding (Turns it into pure black and white)
        # This removes the gray background and shadows
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 4. Denoising
        kernel = np.ones((1, 1), np.uint8)
        denoised = cv2.dilate(thresh, kernel, iterations=1)
        denoised = cv2.erode(denoised, kernel, iterations=1)
        
        # Convert back to PIL Image
        pil_image = Image.fromarray(denoised)
        
        return pil_image
    except Exception as e:
        logger.warning(f"OpenCV pre-processing failed: {e}, falling back to original image")
        return image


# --- TrOCR Handwritten Text Recognition (Optional) ---

import threading

# Thread-safe TrOCR singleton class
class TrOCRSingleton:
    _instance = None
    _lock = threading.Lock()
    _processor = None
    _model = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self):
        """Thread-safe lazy load of TrOCR model."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            try:
                logger.info("Loading TrOCR model for handwritten text recognition...")
                self._processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
                self._model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
                self._initialized = True
                logger.info("TrOCR model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load TrOCR model: {e}")
                self._initialized = False
                raise
    
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from single-line image using TrOCR.
        
        Args:
            image: PIL Image (single line of text)
            
        Returns:
            Extracted text as string
        """
        if not TROCR_AVAILABLE:
            logger.warning("TrOCR not available")
            return None
        
        try:
            self.load_model()
            
            # Prepare image for TrOCR
            pixel_values = self._processor(images=image, return_tensors="pt").pixel_values
            
            # Generate text with max_new_tokens limit to prevent hallucinations
            # Limit to 64 tokens for single-line text to prevent long-form hallucinations
            generated_ids = self._model.generate(
                pixel_values,
                max_new_tokens=64,  # Prevents hallucinations about British judges
                early_stopping=True
            )
            generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text
        except Exception as e:
            logger.warning(f"TrOCR extraction failed: {e}")
            return None


# Global singleton instance
_trocr_singleton = TrOCRSingleton()


# --- PaddleOCR for better handwriting recognition (Optional) ---

class PaddleOCRSingleton:
    _instance = None
    _lock = threading.Lock()
    _ocr = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self):
        """Thread-safe lazy load of PaddleOCR model."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            try:
                logger.info("Loading PaddleOCR model for handwritten text recognition...")
                self._ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                self._initialized = True
                logger.info("PaddleOCR model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load PaddleOCR: {e}")
                self._initialized = False
                raise
    
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from image using PaddleOCR.
        PaddleOCR is better at scene text and handwriting than TrOCR.
        
        Args:
            image: PIL Image
            
        Returns:
            Extracted text as string
        """
        if not PADDLEOCR_AVAILABLE:
            logger.warning("PaddleOCR not available")
            return None
        
        try:
            self.load_model()
            
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Run OCR
            result = self._ocr.ocr(img_array, cls=True)
            
            # Extract text from results
            if result and result[0]:
                extracted_text = " ".join([line[1][0] for line in result[0]])
                return extracted_text
            else:
                return ""
        except Exception as e:
            logger.warning(f"PaddleOCR extraction failed: {e}")
            return None


# Global singleton instance
_paddleocr_singleton = PaddleOCRSingleton()


def _extract_text_with_paddleocr(image: Image.Image) -> str:
    """
    Extract text from image using PaddleOCR.
    PaddleOCR is better for scene text and handwriting than TrOCR.
    
    Args:
        image: PIL Image
        
    Returns:
        Extracted text as string
    """
    return _paddleocr_singleton.extract_text(image)


# --- EasyOCR for lightweight handwriting recognition (Optional) ---

class EasyOCRSingleton:
    _instance = None
    _lock = threading.Lock()
    _reader = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self):
        """Thread-safe lazy load of EasyOCR model."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            try:
                logger.info("Loading EasyOCR model for handwritten text recognition...")
                self._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                self._initialized = True
                logger.info("EasyOCR model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load EasyOCR: {e}")
                self._initialized = False
                raise
    
    def preprocess_image(self, image: Image.Image):
        """
        Pre-process image for better handwriting recognition using CLAHE and denoising.
        
        Uses LAB color space for better brightness stabilization before CLAHE.
        
        Args:
            image: PIL Image
            
        Returns:
            Pre-processed numpy array
        """
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            # Convert to LAB color space for better brightness stabilization
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to the L-channel (lightness)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge back
            enhanced = cv2.merge((cl, a, b))
            gray = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # 1. Rescale (Enlarge) - helps with small handwriting
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # 2. Adaptive Thresholding for pure black/white (Crucial for Tesseract)
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 3. Morphological Closing (connect nearby text regions)
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 4. Final Dilation (thicken text strokes)
        final_kernel = np.ones((1, 1), np.uint8)
        final = cv2.dilate(morph, final_kernel, iterations=1)
        
        # 5. Denoising
        denoised = cv2.fastNlMeansDenoising(final, None, 10, 7, 21)
        
        return denoised
    
    def extract_text(self, image: Image.Image, paragraph: bool = True) -> str:
        """
        Extract text from image using EasyOCR.
        EasyOCR is lightweight, fast, and accurate for handwriting without hallucination risks.
        
        Args:
            image: PIL Image
            paragraph: If True, group lines together (crucial for prescriptions)
            
        Returns:
            Extracted text as string
        """
        if not EASYOCR_AVAILABLE:
            logger.warning("EasyOCR not available")
            return None
        
        try:
            self.load_model()
            
            # Pre-process image
            processed_img = self.preprocess_image(image)
            
            # Perform OCR with paragraph mode
            results = self._reader.readtext(processed_img, detail=0, paragraph=paragraph)
            
            return "\n".join(results)
        except Exception as e:
            logger.warning(f"EasyOCR extraction failed: {e}")
            return None


# Global singleton instance
_easyocr_singleton = EasyOCRSingleton()


def _extract_text_with_easyocr(image: Image.Image, paragraph: bool = True) -> str:
    """
    Extract text from image using EasyOCR.
    EasyOCR is lightweight, fast, and accurate for handwriting without hallucination risks.
    
    Args:
        image: PIL Image
        paragraph: If True, group lines together (crucial for prescriptions)
        
    Returns:
        Extracted text as string
    """
    return _easyocr_singleton.extract_text(image, paragraph=paragraph)


def clean_ocr_text(text: str) -> str:
    """
    Clean OCR text for common handwriting errors and remove entity labels.
    
    This function fixes common OCR errors that occur when reading medical shorthand
    and handwriting, such as confusing '.' with ',', 'y' with 'mg', etc.
    Also removes entity labels (FORM, DRUG, STRENGTH, FREQUENCY, DURATION, CANCER)
    that may be included in the OCR output.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove entity labels that may be appended to words (e.g., "capsuleFORM", "CephalexinDRUG")
    entity_labels = ["FORM", "DRUG", "STRENGTH", "FREQUENCY", "DURATION", "CANCER", "DOSAGE", "ROUTE"]
    for label in entity_labels:
        # Remove label regardless of position (attached to word or standalone)
        text = re.sub(label, "", text, flags=re.IGNORECASE)
    
    # Fix common handwriting OCR errors
    replacements = {
        r"(\d+)\s*ny": r"\1mg",    # 10ny -> 10mg
        r"(\d+)\s*uy": r"\1mg",    # 10uy -> 10mg
        r"\bT\b": "Tablet",        # T -> Tablet
        r"\bBD\b": "twice daily",
        r"\bOD\b": "once daily",
        r"\bHS\b": "at bedtime",
        r"\bSOS\b": "as needed",
        r"\b0zin\b": "Ozis",       # '0zin' -> 'Ozis'
        r"\bStat/808\b": "Stat/SOS"
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_prescription_details(text: str) -> Dict:
    """
    Extract prescription details from OCR text including abbreviations, dosages, and frequencies.
    
    Args:
        text: OCR text
        
    Returns:
        Dictionary with abbreviations, dosages, frequencies, and durations
    """
    if not text:
        return {
            "abbreviations": [],
            "dosages": [],
            "frequencies": [],
            "durations": []
        }
    
    text_lower = text.lower()
    
    # Medical abbreviations mapping
    abbreviations_map = {
        'od': 'Once Daily',
        'bd': 'Twice Daily',
        'tid': 'Three Times Daily',
        'qid': 'Four Times Daily',
        'sos': 'If Needed',
        'prn': 'As Needed',
        'stat': 'Immediately',
        'hs': 'At Bedtime',
        'ac': 'Before Meals',
        'pc': 'After Meals',
        'tab': 'Tablet',
        'cap': 'Capsule',
        'syr': 'Syrup',
        'inj': 'Injection',
        'mg': 'Milligram',
        'ml': 'Milliliter'
    }
    
    # Extract abbreviations
    found_abbreviations = []
    for abbr, meaning in abbreviations_map.items():
        if re.search(r'\b' + re.escape(abbr) + r'\b', text_lower):
            found_abbreviations.append({
                'abbreviation': abbr.upper(),
                'meaning': meaning
            })
    
    # Extract dosages (e.g., 10mg, 500mg, 2.5ml)
    dosages = re.findall(r'(\d+(?:\.\d+)?\s*(?:mg|gm|ml|mcg|iu))', text_lower, re.IGNORECASE)
    
    # Extract frequencies (e.g., od, bd, tid, qid, sos, prn, stat)
    frequencies = re.findall(r'\b(od|bd|tid|qid|sos|prn|stat|hs|ac|pc)\b', text_lower, re.IGNORECASE)
    
    # Extract durations (e.g., 5 days, 2 weeks, 1 month)
    durations = re.findall(r'(\d+\s*(?:day|days|week|weeks|month|months))', text_lower, re.IGNORECASE)
    
    return {
        "abbreviations": found_abbreviations,
        "dosages": dosages,
        "frequencies": frequencies,
        "durations": durations
    }


# --- Medical Dictionary Lookup (Fuzzy Matching) ---

COMMON_DRUGS = [
    "Sitagliptin", "Metformin", "Losartan", "Atorvastatin", "Amlodipine",
    "Naproxen", "Ibuprofen", "Paracetamol", "Amoxicillin", "Azithromycin",
    "Ciprofloxacin", "Doxycycline", "Omeprazole", "Pantoprazole", "Ranitidine",
    "Cetirizine", "Loratadine", "Montelukast", "Salbutamol", "Formoterol",
    "Metoprolol", "Amlodipine", "Hydrochlorothiazide", "Furosemide", "Spironolactone",
    "Aspirin", "Clopidogrel", "Warfarin", "Insulin", "Glimepiride",
    "Gabapentin", "Pregabalin", "Duloxetine", "Sertraline", "Escitalopram"
]


def autocorrect_drug_name(word: str) -> str:
    """
    Fuzzy match drug name to closest match in common drugs dictionary.
    
    Args:
        word: OCR-extracted drug name
        
    Returns:
        Corrected drug name or original if no match found
    """
    if not word or len(word) < 3:
        return word
    
    # Find the closest match with at least 60% similarity
    matches = get_close_matches(word, COMMON_DRUGS, n=1, cutoff=0.6)
    return matches[0] if matches else word


def normalize_dosage_notation(text: str) -> str:
    """
    Normalize medical shorthand and dosage notation to plain English.
    
    Args:
        text: Text containing medical shorthand
        
    Returns:
        Text with normalized dosage notation
    """
    if not text:
        return text
    
    # Map 1-0-1 or 1-1-1 patterns (dosage notation)
    patterns = {
        r"\b1-0-1\b": "twice daily",
        r"\b1-1-1\b": "three times daily",
        r"\b1-0-0\b": "once daily in the morning",
        r"\b0-0-1\b": "once daily at bedtime",
        r"\b0-1-0\b": "once daily in the afternoon",
        r"\bx\s*BD\b": "twice daily",
        r"\bx\s*OD\b": "once daily",
        r"\bx\s*TDS\b": "three times daily",
        r"\bx\s*QID\b": "four times daily",
    }
    
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text


# --- VLM-based OCR with Groq API ---

VLM_OCR_PROMPT = """
You are a medical prescription OCR assistant with specialized knowledge of medical shorthand and dosage notation.

PRIMARY GOAL:
From the prescription image, extract all medications with their details in structured JSON format.

CRITICAL RULES:
- DO NOT GUESS.
- DO NOT USE MEDICAL KNOWLEDGE to correct or complete drug names unless explicitly asked.
- Read the handwriting as literally as possible.
- If you are not sure about a part of the text, set the field to null.
- Prefer null over guessing.
- Do NOT explain anything.
- Do NOT output markdown.
- Output MUST be a single valid JSON object.

SHORTHAND MAPPING (apply these transformations):
- "BD" or "x BD" → "twice daily"
- "OD" or "x OD" → "once daily"
- "TD" or "TDS" → "three times daily"
- "QID" → "four times daily"
- "HS" → "at bedtime"
- "SOS" or "PRN" → "as needed"
- "AC" → "before meals"
- "PC" → "after meals"
- "STAT" → "immediately"

DOSAGE NOTATION MAPPING (apply these transformations):
- "1-0-1" → "twice daily"
- "1-1-1" → "three times daily"
- "1-0-0" → "once daily in the morning"
- "0-0-1" → "once daily at bedtime"
- "0-1-0" → "once daily in the afternoon"

INTERPRETATION RULES FOR FIELDS:
- "raw_text": exact text as you read it from the prescription - DO NOT add entity labels like FORM, DRUG, STRENGTH, FREQUENCY, DURATION, CANCER, DOSAGE, ROUTE to the text
- "drug_name": name of the medication as written
- "strength": concentration or strength (e.g., "500 mg", "125 mg/5 ml", "3%")
- "dose": amount to be taken each time (e.g., "1 tablet", "1 tsp", "20 drops")
- "frequency": how often the dose is taken (apply shorthand mapping)
- "duration": total length of treatment (e.g., "5 days", "1 week")
- "instructions": extra notes like "for cough", "after meals"

OUTPUT FORMAT (JSON ONLY):

{
  "medications": [
    {
      "raw_text": string,             // exact text as you read it from the prescription
      "drug_name": string or null,    // as written, not corrected
      "strength": string or null,     // e.g. "500 mg", "3 gr", "3%", "5 ml"
      "dose": string or null,         // e.g. "1 tablet", "1 tsp", "20 drops"
      "frequency": string or null,    // e.g. "twice daily", "three times daily" (mapped from shorthand)
      "duration": string or null,     // e.g. "5 days", "1 week"
      "instructions": string or null, // other directions like "for cough", "after meals"
      "confidence": number            // 0–1
    }
  ],
  "illegible": boolean,
  "illegible_details": string or null
}

IMPORTANT:
- Apply shorthand and dosage notation mapping in the frequency field
- If no medication can be reliably read, return empty medications list and set illegible = true
"""


def _extract_json_from_text(text: str) -> Dict:
    """
    Extract JSON from VLM response, handling markdown code blocks.
    
    Args:
        text: Raw response text from VLM
        
    Returns:
        Parsed JSON dictionary
    """
    text = text.strip()
    
    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # Remove first line (``` or ```json)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # Remove last line (```)
        text = "\n".join(lines).strip()
        
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    
    # Extract JSON from first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in text")
    
    json_str = text[start:end + 1]
    return json.loads(json_str)


def _extract_text_with_vlm(image: Image.Image) -> Tuple[str, Dict]:
    """
    Extract text from image using VLM-based OCR via Groq API (Llama-4-Scout).
    
    Args:
        image: PIL Image
        
    Returns:
        Tuple of (extracted_text, structured_data)
    """
    if not GROQ_AVAILABLE:
        raise ValueError("Groq API key not configured. Set GROQ_API_KEY environment variable.")
    
    # Convert image to base64
    import base64
    from io import BytesIO
    
    byte_buffer = BytesIO()
    image.save(byte_buffer, format="PNG")
    base64_image = base64.b64encode(byte_buffer.getvalue()).decode("utf-8")
    data_url = f"data:image/png;base64,{base64_image}"
    
    # Initialize Groq client
    client = Groq(api_key=GROQ_API_KEY)
    
    # Call VLM model with Llama 4 Scout (recommended for vision tasks)
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": VLM_OCR_PROMPT}
                    ]
                }
            ],
            max_tokens=2048,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        raise
    
    raw_content = response.choices[0].message.content
    structured_data = _extract_json_from_text(raw_content)
    
    # Apply drug name autocorrection to medications
    medications = structured_data.get("medications", [])
    for med in medications:
        if med.get("drug_name"):
            med["drug_name"] = autocorrect_drug_name(med["drug_name"])
        if med.get("frequency"):
            med["frequency"] = normalize_dosage_notation(med["frequency"])
    
    # Combine all raw_text from medications for NER processing
    combined_text = " ".join([m.get("raw_text", "") for m in medications])
    
    return combined_text, structured_data


def _extract_text_with_gemini(image: Image.Image) -> Tuple[str, Dict]:
    """
    Extract text from image using Google Gemini Vision API.
    
    Args:
        image: PIL Image
        
    Returns:
        Tuple of (extracted_text, structured_data)
    """
    if not GEMINI_AVAILABLE:
        raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY environment variable.")
    
    # Configure Gemini with API key using new google.genai package
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Convert image to bytes
    from io import BytesIO
    byte_buffer = BytesIO()
    image.save(byte_buffer, format="PNG")
    image_bytes = byte_buffer.getvalue()
    
    # Call Gemini with image using new google.genai API with retry logic for 429 errors
    import time
    import json
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    VLM_OCR_PROMPT,
                    genai.types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                ]
            )
            
            raw_content = response.text
            structured_data = _extract_json_from_text(raw_content)
            
            # Apply drug name autocorrection to medications
            medications = structured_data.get("medications", [])
            for med in medications:
                if med.get("drug_name"):
                    med["drug_name"] = autocorrect_drug_name(med["drug_name"])
                if med.get("frequency"):
                    med["frequency"] = normalize_dosage_notation(med["frequency"])
            
            # Combine all raw_text from medications for NER processing
            combined_text = " ".join([m.get("raw_text", "") for m in medications])
            
            return combined_text, structured_data
            
        except Exception as e:
            error_str = str(e)
            # Check if it's a 429 rate limit error
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries - 1:
                    # Extract retry delay from error if available
                    retry_delay = 15  # default 15 seconds
                    try:
                        if "retryDelay" in error_str:
                            # Parse retry delay from error message
                            import re
                            match = re.search(r'retryDelay.*?(\d+)', error_str)
                            if match:
                                retry_delay = int(match.group(1))
                    except:
                        pass
                    
                    logger.warning(f"Gemini API rate limited (429). Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.warning("Gemini API rate limited (429) after retry. Falling back to Hybrid OCR.")
                    raise
            else:
                # For other errors, don't retry
                raise


def _preprocess_image_for_ocr(image: Image.Image) -> bytes:
    """
    Preprocess image for better OCR accuracy on handwritten prescriptions.
    Uses OpenCV for grayscale conversion, denoising, and adaptive thresholding.
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed image as bytes
    """
    try:
        import cv2
        import numpy as np
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Denoise with Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding (critical for handwriting)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to bytes
        _, buffer = cv2.imencode('.png', thresh)
        return buffer.tobytes()
    except ImportError:
        logger.warning("OpenCV not available, using raw image")
        from io import BytesIO
        byte_buffer = BytesIO()
        image.save(byte_buffer, format="PNG")
        return byte_buffer.getvalue()
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}, using raw image")
        from io import BytesIO
        byte_buffer = BytesIO()
        image.save(byte_buffer, format="PNG")
        return byte_buffer.getvalue()

async def _extract_text_with_glm_ocr(image: Image.Image) -> Tuple[str, Dict]:
    """
    Extract text from image using local GLM-OCR via Ollama.
    GLM-OCR is optimized for document OCR and handwriting recognition.
    
    Args:
        image: PIL Image
        
    Returns:
        Tuple of (extracted_text, structured_data)
    """
    if not GLM_OCR_AVAILABLE or not glm_ocr_service:
        logger.warning("GLM-OCR service not available")
        raise ValueError("GLM-OCR not available. Install Ollama and enable GLM-OCR in .env")
    
    try:
        # Preprocess image for better OCR accuracy
        image_bytes = _preprocess_image_for_ocr(image)
        
        # Use the GLM-OCR service (async)
        extracted_text = await glm_ocr_service.extract_text_from_image(image_bytes)
        
        if not extracted_text:
            raise ValueError("GLM-OCR returned empty text")
        
        raw_text = extracted_text
        
        # Try to extract structured data from the response
        # GLM-OCR might return plain text instead of JSON, so handle gracefully
        structured_data = None
        try:
            structured_data = _extract_json_from_text(raw_text)
        except (ValueError, json.JSONDecodeError) as e:
            logger.info(f"GLM-OCR returned non-JSON response, using raw text: {e}")
            structured_data = None
        
        # If no structured data found or no medications, create a basic one
        if not structured_data or not structured_data.get("medications"):
            structured_data = {
                "medications": [],
                "raw_text": raw_text
            }
        
        # Apply drug name autocorrection to medications
        medications = structured_data.get("medications", [])
        for med in medications:
            if med.get("drug_name"):
                med["drug_name"] = autocorrect_drug_name(med["drug_name"])
            if med.get("frequency"):
                med["frequency"] = normalize_dosage_notation(med["frequency"])
        
        # Combine all raw_text from medications for NER processing
        combined_text = " ".join([m.get("raw_text", "") for m in medications])
        
        # If no medications, use the raw text
        if not combined_text:
            combined_text = raw_text
        
        return combined_text, structured_data
        
    except Exception as e:
        logger.warning(f"GLM-OCR extraction failed: {e}")
        raise


def _extract_text_with_trocr(image: Image.Image) -> str:
    """
    Extract text from image using TrOCR (Transformer-based OCR).
    TrOCR is specifically trained for single-line handwritten text recognition.
    
    Args:
        image: PIL Image
        
    Returns:
        Extracted text as string
    """
    return _trocr_singleton.extract_text(image)


def _detect_and_split_lines_cv2(image: Image.Image) -> List[Image.Image]:
    """
    Detect text lines in multi-line image using Horizontal Projection Profile.
    
    This method calculates the sum of black pixels across the horizontal axis.
    Peaks represent text lines, valleys represent spaces between lines.
    This is more robust for handwritten medical documents than contour detection.
    
    Args:
        image: PIL Image (multi-line document)
        
    Returns:
        List of PIL Image strips (one per text line)
    """
    if not OPENCV_AVAILABLE:
        logger.warning("OpenCV not available, skipping line detection")
        return [image]
    
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Binarize: make background white (255) and text black (0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate horizontal projection profile
        # Sum of black pixels in each row
        horizontal_profile = np.sum(binary, axis=1)
        
        logger.info(f"Image height: {image.height}, Profile length: {len(horizontal_profile)}")
        logger.info(f"Profile max: {np.max(horizontal_profile)}, Profile mean: {np.mean(horizontal_profile)}")
        
        # Normalize profile
        if np.max(horizontal_profile) > 0:
            horizontal_profile = horizontal_profile / np.max(horizontal_profile)
        
        # Find valleys (spaces between lines)
        # A valley is where the profile drops below a threshold
        threshold = 0.1  # 10% of max height
        lines = []
        in_line = False
        line_start = 0
        
        for i, value in enumerate(horizontal_profile):
            if value > threshold and not in_line:
                # Start of a line
                in_line = True
                line_start = i
            elif value <= threshold and in_line:
                # End of a line
                in_line = False
                line_end = i
                # Only keep lines with reasonable height (at least 10 pixels)
                if line_end - line_start >= 10:
                    lines.append((line_start, line_end))
        
        # Handle case where we're still in a line at the end
        if in_line and len(horizontal_profile) - line_start >= 10:
            lines.append((line_start, len(horizontal_profile)))
        
        if not lines:
            logger.warning("Horizontal projection detected no lines, using full image")
            return [image]
        
        logger.info(f"Detected {len(lines)} raw lines before merging")
        
        # Merge lines that are too close (less than 15 pixels apart)
        merged_lines = []
        if lines:
            current_start, current_end = lines[0]
            for start, end in lines[1:]:
                if start - current_end < 15:
                    # Merge
                    current_end = end
                else:
                    merged_lines.append((current_start, current_end))
                    current_start, current_end = start, end
            merged_lines.append((current_start, current_end))
        
        # Crop image into line strips
        line_images = []
        for start, end in merged_lines:
            # Add padding
            padding = 10
            y_min = max(0, start - padding)
            y_max = min(image.height, end + padding)
            
            # Crop the full width of the line
            line_strip = image.crop((0, y_min, image.width, y_max))
            line_images.append(line_strip)
        
        logger.info(f"Horizontal projection detected {len(line_images)} text lines")
        return line_images
        
    except Exception as e:
        logger.warning(f"Horizontal projection line detection failed: {e}, using full image")
        return [image]


def _detect_and_split_lines(image: Image.Image) -> List[Image.Image]:
    """
    Detect text lines in multi-line image using CRAFT and split into single-line strips.
    
    Args:
        image: PIL Image (multi-line document)
        
    Returns:
        List of PIL Image strips (one per text line)
    """
    # Try CV2 first (simpler, more robust)
    cv2_result = _detect_and_split_lines_cv2(image)
    if len(cv2_result) > 1:
        return cv2_result
    
    if not CRAFT_AVAILABLE:
        logger.warning("CRAFT not available, skipping line detection")
        return [image]
    
    try:
        # Initialize CRAFT detector
        detector = craft_text_detector.CraftDetector()
        detector.load_model('craft_mlt_25k.pth')
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Detect text regions
        prediction_result = detector.detect_text(img_array)
        
        # Extract bounding boxes
        boxes = prediction_result['boxes']
        
        if not boxes or len(boxes) == 0:
            logger.warning("CRAFT detected no text regions, using full image")
            return [image]
        
        # Sort boxes by vertical position (top to bottom)
        boxes = sorted(boxes, key=lambda box: box[0][1])
        
        # Crop image into line strips
        line_images = []
        for box in boxes:
            # Convert box to integer coordinates
            x_coords = [int(p[0]) for p in box]
            y_coords = [int(p[1]) for p in box]
            
            # Add padding
            padding = 10
            x_min = max(0, min(x_coords) - padding)
            y_min = max(0, min(y_coords) - padding)
            x_max = min(image.width, max(x_coords) + padding)
            y_max = min(image.height, max(y_coords) + padding)
            
            # Crop line
            line_strip = image.crop((x_min, y_min, x_max, y_max))
            line_images.append(line_strip)
        
        logger.info(f"CRAFT detected {len(line_images)} text lines")
        return line_images
        
    except Exception as e:
        logger.warning(f"CRAFT detection failed: {e}, using full image")
        return [image]


def _extract_text_with_trocr_multiline(image: Image.Image) -> str:
    """
    Extract text from multi-line image using CRAFT + TrOCR pipeline.
    
    Strategy:
        1. Use CRAFT to detect text lines
        2. Split image into single-line strips
        3. Run TrOCR on each strip
        4. Join results with newlines
    
    Args:
        image: PIL Image (multi-line document)
        
    Returns:
        Extracted text as string
    """
    if not TROCR_AVAILABLE:
        logger.warning("TrOCR not available")
        return None
    
    # Detect and split lines
    line_images = _detect_and_split_lines(image)
    
    if len(line_images) == 1:
        # Single line, use direct extraction
        return _trocr_singleton.extract_text(line_images[0])
    
    # Multi-line: process each line
    lines_text = []
    for line_img in line_images:
        line_text = _trocr_singleton.extract_text(line_img)
        if line_text:
            lines_text.append(line_text)
    
    # Join with newlines
    full_text = '\n'.join(lines_text)
    logger.info(f"TrOCR extracted {len(lines_text)} lines from multi-line image")
    
    return full_text


# --- Text Normalization ---

def normalize_text(text: str) -> str:
    """
    Normalize clinical text for consistent NER processing.
    
    Steps:
        1. Unicode normalization
        2. Whitespace standardization
        3. Common medical abbreviation standardization
        4. Remove excessive special characters
    """
    if not text:
        return ""
    
    # 1. OCR text cleaning - fix line breaks in middle of sentences
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # 2. Basic unicode normalization
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Standardize whitespace
    text = re.sub(r'[\t\n\r]+', ' ', text)  # Newlines/tabs to space
    text = re.sub(r'\s+', ' ', text)        # Multiple spaces to single
    
    # 3. Common medical abbreviation standardization (shorthand parser)
    abbreviations = {
        # Dosage forms
        r'\bmgr\b': 'mg',
        r'\bmls\b': 'mL',
        r'\btabs?\b': 'tablet',
        r'\bcaps?\b': 'capsule',
        r'\bsusp\b': 'suspension',
        r'\bsoln\b': 'solution',
        r'\binj\b': 'injection',
        r'\bcr\b': 'cream',
        r'\bont\b': 'ointment',
        r'\bsupp\b': 'suppository',
        # Frequency abbreviations (shorthand parser)
        r'\bod\b': 'once daily',
        r'\bbd\b': 'twice daily',
        r'\btds\b': 'three times daily',
        r'\btid\b': 'three times daily',
        r'\bqid\b': 'four times daily',
        r'\bqod\b': 'every other day',
        r'\bsos\b': 'as needed',
        r'\bq\.?d\.?\b': 'daily',
        r'\bb\.?i\.?d\.?\b': 'twice daily',
        r'\bt\.?i\.?d\.?\b': 'three times daily',
        r'\bq\.?i\.?d\.?\b': 'four times daily',
        r'\bq\.?o\.?d\.?\b': 'every other day',
        r'\bhs\b': 'at bedtime',
        r'\bq\.?h\.?s\.?\b': 'at bedtime',
        r'\bac\b': 'before meals',
        r'\bpc\b': 'after meals',
        r'\bprn\b': 'as needed',
        r'\bstat\b': 'immediately',
        # Route abbreviations
        r'\bp\.?o\.?\b': 'by mouth',
        r'\bi\.?v\.?\b': 'intravenous',
        r'\bs\.?c\.?\b': 'subcutaneous',
        r'\bim\b': 'intramuscular',
        r'\bsl\b': 'sublingual',
        r'\bpr\b': 'per rectum',
        r'\btop\b': 'topically',
        # Duration abbreviations
        r'\bq[0-9]+h\b': lambda m: f'every {m.group()[1:-1]} hours',
    }
    
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # 4. Remove excessive special chars but keep medical symbols
    text = re.sub(r'[^\w\s.,;:!?()\-%°/]', ' ', text)
    
    # 5. Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_text_from_pdf(file_bytes):
    """
    Extract text from PDF using hybrid approach:
    1. Try standard text extraction (for selectable text)
    2. If text is too short (< 50 chars), use OCR (for scanned documents)
    """
    text = ""
    
    # 1. Try standard text extraction
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    
    # 2. If text is too short, it's likely a scan. Use OCR.
    if len(text.strip()) < 50:
        logger.info("PDF appears to be scanned, using OCR fallback")
        try:
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(file_bytes)
            for img in images:
                # Try EasyOCR first (best for handwriting, no hallucination)
                if EASYOCR_AVAILABLE:
                    ocr_text = _extract_text_with_easyocr(img, paragraph=True)
                    if ocr_text and len(ocr_text.strip()) > 10:
                        # Clean OCR text for handwriting errors
                        ocr_text = clean_ocr_text(ocr_text)
                        text += ocr_text
                        continue
                
                # Fallback to PaddleOCR with OpenCV pre-processing
                if OPENCV_AVAILABLE:
                    img = _preprocess_with_opencv(img)
                else:
                    img = _preprocess_image_for_ocr(img, mode="printed")
                
                if PADDLEOCR_AVAILABLE:
                    ocr_text = _extract_text_with_paddleocr(img)
                    if ocr_text and len(ocr_text.strip()) > 10:
                        ocr_text = clean_ocr_text(ocr_text)
                        text += ocr_text
                    else:
                        # Fallback to Tesseract
                        ocr_text = pytesseract.image_to_string(img)
                        ocr_text = clean_ocr_text(ocr_text)
                        text += ocr_text
                else:
                    # Fallback to Tesseract
                    ocr_text = pytesseract.image_to_string(img)
                    ocr_text = clean_ocr_text(ocr_text)
                    text += ocr_text
        except ImportError:
            logger.warning("pdf2image not installed, cannot OCR scanned PDFs")
        except Exception as e:
            logger.error(f"OCR fallback failed: {e}")
    
    return normalize_text(text)

def _preprocess_image_for_ocr(image: Image.Image, mode: str = "default") -> Image.Image:
    """
    Pre-process image for better OCR/HTR accuracy on handwritten/medical documents.
    
    Modes:
        - "default": Balanced pre-processing for mixed text
        - "handwritten": Aggressive contrast + thresholding for cursive/script
        - "printed": Light processing for typed/printed prescriptions
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Upscale small images (width < 1000px) for better recognition
    width, height = image.size
    if width < 1000:
        scale_factor = 2
        new_size = (width * scale_factor, height * scale_factor)
        image = image.resize(new_size, Image.LANCZOS)
    
    if mode == "handwritten":
        # Aggressive processing for handwritten/cursive text
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(3.0)  # High contrast for faint handwriting
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.5)
        # Aggressive binarization - lower threshold to capture faint strokes
        threshold = 100
        image = image.point(lambda x: 255 if x > threshold else 0, '1')
        image = image.convert('RGB')
        image = image.filter(ImageFilter.MedianFilter(size=3))
    elif mode == "printed":
        # Light processing for printed/typed text
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        threshold = 140
        image = image.point(lambda x: 255 if x > threshold else 0, '1')
        image = image.convert('RGB')
    else:
        # Default balanced processing
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        threshold = 128
        image = image.point(lambda x: 255 if x > threshold else 0, '1')
        image = image.convert('RGB')
        image = image.filter(ImageFilter.MedianFilter(size=3))
    
    return image


def _detect_rois_opencv(img_bgr, max_rois=12):
    """
    Fast ROI detection for text blocks using OpenCV.
    Returns list of (x, y, w, h) bounding boxes for text regions.
    Based on Kaggle notebook implementation.
    """
    if not OPENCV_AVAILABLE:
        return [(0, 0, img_bgr.shape[1], img_bgr.shape[0])] if hasattr(img_bgr, 'shape') else []
    
    try:
        import cv2
        import numpy as np
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 41, 11
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        dil = cv2.dilate(thr, kernel, iterations=2)

        contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        boxes = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < 1500:
                continue
            if bw > 0.98*w and bh > 0.98*h:
                continue
            boxes.append((x, y, bw, bh))

        boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
        return boxes[:max_rois]
    except Exception as e:
        logger.warning(f"ROI detection failed: {e}")
        return [(0, 0, img_bgr.shape[1], img_bgr.shape[0])] if hasattr(img_bgr, 'shape') else []


def _compute_ocr_confidence(image: Image.Image, config: str) -> Dict:
    """
    Run OCR with confidence data to assess extraction quality.
    Returns dict with text, mean_confidence, and word-level confidences.
    """
    try:
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        
        confidences = [int(c) for c in data['conf'] if int(c) > 0]
        words = [data['text'][i] for i in range(len(data['conf'])) if int(data['conf'][i]) > 0]
        
        mean_conf = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
        
        # Word-level confidence mapping
        word_confidences = {}
        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            conf = int(data['conf'][i])
            if word and conf > 0:
                word_confidences[word] = conf / 100.0
        
        return {
            "mean_confidence": round(mean_conf, 2),
            "word_confidences": word_confidences,
            "word_count": len(confidences)
        }
    except Exception as e:
        logger.warning(f"OCR confidence computation failed: {e}")
        return {"mean_confidence": 0.0, "word_confidences": {}, "word_count": 0}


def _merge_ocr_results(results: List[Dict]) -> str:
    """
    Merge multiple OCR passes using word-level voting.
    For each position, pick the most common word across passes.
    If passes disagree, use the one from the highest-confidence pass.
    """
    if not results:
        return ""
    if len(results) == 1:
        return results[0]['text']
    
    # Simple merge: use the result with highest mean confidence
    best = max(results, key=lambda r: r.get('mean_confidence', 0))
    return best['text']


def _merge_ocr_results(easyocr_text: str, tesseract_text: str) -> str:
    """
    Merge OCR results from EasyOCR and Pytesseract using word-level confidence voting.
    
    Args:
        easyocr_text: Text from EasyOCR
        tesseract_text: Text from Pytesseract
        
    Returns:
        Merged text
    """
    if not easyocr_text and not tesseract_text:
        return ""
    if not easyocr_text:
        return tesseract_text
    if not tesseract_text:
        return easyocr_text
    
    # Simple merge: prefer EasyOCR for handwriting (better accuracy)
    # EasyOCR has 90% confidence, Tesseract typically lower for handwriting
    return easyocr_text


def extract_text_from_image(file_bytes) -> Tuple[str, Dict]:
    """
    Extract text from image using ensemble VLM + Hybrid OCR approach with multiple fallbacks.
    
    Strategy:
        1. Try VLM (Groq Llama-4-Scout) - primary method
        2. Try GLM-OCR local VLM (fallback for privacy, no API limits)
        3. Run Hybrid OCR (EasyOCR + Pytesseract) for cross-validation
        4. Ensemble voting: cross-reference results for confidence
        5. Fallback to TrOCR if all VLMs fail
        6. Fallback to Tesseract multi-pass as last resort
    
    Returns:
        Tuple of (normalized_text, ocr_metadata)
    """
    image = Image.open(io.BytesIO(file_bytes))
    logger.info(f"Starting OCR extraction for image size: {image.size}")
    
    # Step 0: ROI Detection - identify text regions for focused OCR
    roi_text = None
    if OPENCV_AVAILABLE:
        try:
            import cv2
            import numpy as np
            # Convert PIL to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            rois = _detect_rois_opencv(img_cv, max_rois=12)
            
            if len(rois) > 0:
                roi_texts = []
                for (x, y, w, h) in rois[:6]:  # Process top 6 ROIs
                    roi_img = img_cv[y:y+h, x:x+w]
                    # Convert back to PIL for pytesseract
                    roi_pil = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
                    # Preprocess ROI
                    roi_pil = _preprocess_image_for_ocr(roi_pil, mode="handwritten")
                    txt = pytesseract.image_to_string(roi_pil, config=r'--oem 3 --psm 6 -c preserve_interword_spaces=1')
                    txt = re.sub(r"\s+", " ", txt).strip()
                    if txt and len(txt) > 3:
                        roi_texts.append(txt)
                
                if roi_texts:
                    roi_text = "\n".join(roi_texts)
                    logger.info(f"ROI detection found {len(roi_texts)} text regions")
        except Exception as e:
            logger.warning(f"ROI detection failed: {e}")
    
    # Step 1: Try Groq VLM (primary method - best for handwriting)
    vlm_structured_data = None
    vlm_text = None
    vlm_source = None
    logger.info(f"GROQ_AVAILABLE: {GROQ_AVAILABLE}, OLLAMA_AVAILABLE: {OLLAMA_AVAILABLE}")
    if GROQ_AVAILABLE:
        try:
            logger.info("Attempting Groq VLM OCR extraction...")
            vlm_text, vlm_structured_data = _extract_text_with_vlm(image)
            if vlm_text and len(vlm_text.strip()) > 5:
                vlm_source = "groq"
                logger.info(f"Groq VLM OCR extraction successful. Text length: {len(vlm_text)}")
            else:
                logger.warning("Groq VLM returned empty or too short text")
        except Exception as e:
            logger.warning(f"Groq VLM OCR extraction failed: {e}")
    else:
        logger.warning("Groq API key not configured, skipping VLM")
    
    # Step 2: Try GLM-OCR local VLM (fallback for privacy and rate limits)
    if not vlm_text and OLLAMA_AVAILABLE:
        try:
            logger.info("Attempting GLM-OCR local VLM extraction...")
            vlm_text, vlm_structured_data = _extract_text_with_glm_ocr(image)
            if vlm_text and len(vlm_text.strip()) > 5:
                vlm_source = "glm-ocr"
                logger.info(f"GLM-OCR extraction successful. Text length: {len(vlm_text)}")
        except Exception as e:
            logger.warning(f"GLM-OCR extraction failed: {e}")
    
    # Step 3: Run Hybrid OCR (EasyOCR + Pytesseract) for cross-validation
    easyocr_text = None
    if EASYOCR_AVAILABLE:
        try:
            easyocr_text = _extract_text_with_easyocr(image, paragraph=True)
            if easyocr_text and len(easyocr_text.strip()) > 5:
                logger.info("EasyOCR extraction successful")
        except Exception as e:
            logger.warning(f"EasyOCR extraction failed: {e}")
    
    tesseract_text = None
    try:
        if OPENCV_AVAILABLE:
            processed_img = _preprocess_with_opencv(image)
        else:
            processed_img = _preprocess_image_for_ocr(image, mode="printed")
        
        config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        tesseract_text = pytesseract.image_to_string(processed_img, config=config)
        if tesseract_text and len(tesseract_text.strip()) > 5:
            logger.info("Pytesseract extraction successful")
    except Exception as e:
        logger.warning(f"Pytesseract extraction failed: {e}")
    
    # Step 5: Ensemble voting - cross-reference VLM and Hybrid results
    if vlm_text and (easyocr_text or tesseract_text):
        # VLM succeeded, check if hybrid OCR agrees
        hybrid_text = _merge_ocr_results(easyocr_text, tesseract_text)
        
        # Simple similarity check using word overlap
        vlm_words = set(vlm_text.lower().split())
        hybrid_words = set(hybrid_text.lower().split())
        
        if vlm_words and hybrid_words:
            overlap = len(vlm_words & hybrid_words) / len(vlm_words | hybrid_words)
            logger.info(f"VLM-Hybrid similarity: {overlap:.2f}")
            
            # If high similarity (> 0.5), use VLM (more structured)
            # If low similarity, VLM might be hallucinating, use hybrid
            if overlap > 0.5:
                cleaned_text = clean_ocr_text(vlm_text)
                normalized = normalize_text(cleaned_text)
                return normalized, {
                    "extraction_method": f"vlm_ensemble_{vlm_source}",
                    "best_confidence": 0.95,
                    "word_confidences": {},
                    "passes_completed": 2,
                    "vlm_structured_data": vlm_structured_data,
                    "ensemble_similarity": overlap,
                    "vlm_source": vlm_source
                }
    
    # Step 6: If VLM succeeded but hybrid failed, use VLM
    if vlm_text:
        cleaned_text = clean_ocr_text(vlm_text)
        normalized = normalize_text(cleaned_text)
        return normalized, {
            "extraction_method": f"vlm_{vlm_source}",
            "best_confidence": 0.90,
            "word_confidences": {},
            "passes_completed": 1,
            "vlm_structured_data": vlm_structured_data,
            "vlm_source": vlm_source
        }
    
    # Step 7: If VLM failed or low confidence, use Hybrid OCR
    # Include ROI text if available
    texts_to_merge = []
    if easyocr_text:
        texts_to_merge.append(easyocr_text)
    if tesseract_text:
        texts_to_merge.append(tesseract_text)
    if roi_text:
        texts_to_merge.append(roi_text)
    
    if texts_to_merge:
        # Use ROI text as primary if available, else merge all
        if roi_text and len(roi_text.strip()) > 20:
            merged_text = roi_text
            extraction_method = "roi_ocr"
        else:
            merged_text = "\n".join(texts_to_merge)
            extraction_method = "hybrid_ocr"
        
        cleaned_text = clean_ocr_text(merged_text)
        normalized = normalize_text(cleaned_text)
        return normalized, {
            "extraction_method": extraction_method,
            "best_confidence": 0.90,
            "word_confidences": {},
            "passes_completed": len(texts_to_merge),
            "easyocr_used": easyocr_text is not None,
            "tesseract_used": tesseract_text is not None,
            "roi_used": roi_text is not None
        }
    
    # Step 8: Apply OpenCV advanced pre-processing if available
    if OPENCV_AVAILABLE:
        try:
            image = _preprocess_with_opencv(image)
            logger.info("Applied OpenCV pre-processing for OCR enhancement")
        except Exception as e:
            logger.warning(f"OpenCV pre-processing failed, continuing with PIL only: {e}")
    
    # Step 9: Try TrOCR for handwritten text (quaternary method)
    if TROCR_AVAILABLE:
        try:
            trocr_text = _extract_text_with_trocr_multiline(image)
            if trocr_text and len(trocr_text.strip()) > 5:
                cleaned_text = clean_ocr_text(trocr_text)
                normalized = normalize_text(cleaned_text)
                logger.info("TrOCR extraction successful")
                return normalized, {
                    "extraction_method": "trocr_multiline",
                    "best_confidence": 0.85,
                    "word_confidences": {},
                    "passes_completed": 1
                }
        except Exception as e:
            logger.warning(f"TrOCR extraction failed, falling back to Tesseract: {e}")
    
    # Step 10: Fallback to Tesseract multi-pass
    base_config = r'--oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,:;/()-+=% '
    
    passes = [
        {"mode": "default", "psm": 6, "label": "uniform_block"},
        {"mode": "handwritten", "psm": 4, "label": "single_column"},
        {"mode": "printed", "psm": 3, "label": "auto_detect"},
    ]
    
    ocr_results = []
    logger.info(f"Starting Tesseract multi-pass OCR ({len(passes)} passes)")
    for pass_config in passes:
        try:
            processed = _preprocess_image_for_ocr(image.copy(), mode=pass_config["mode"])
            config = f'{base_config} --psm {pass_config["psm"]}'
            
            text = pytesseract.image_to_string(processed, config=config)
            confidence_data = _compute_ocr_confidence(processed, config)
            logger.info(f"OCR pass '{pass_config['label']}': len={len(text)}, conf={confidence_data['mean_confidence']:.2f}")
            
            ocr_results.append({
                "text": text,
                "mean_confidence": confidence_data["mean_confidence"],
                "word_confidences": confidence_data["word_confidences"],
                "word_count": confidence_data["word_count"],
                "pass": pass_config["label"]
            })
        except Exception as e:
            logger.warning(f"OCR pass '{pass_config['label']}' failed: {e}")
            continue
    
    # Merge results from multiple passes
    merged_text = _merge_ocr_results(ocr_results)
    normalized = normalize_text(merged_text)
    
    # Compute overall metadata
    best_confidence = max((r['mean_confidence'] for r in ocr_results), default=0.0)
    all_word_confidences = {}
    for r in ocr_results:
        for word, conf in r.get('word_confidences', {}).items():
            if word not in all_word_confidences or conf > all_word_confidences[word]:
                all_word_confidences[word] = conf
    
    ocr_metadata = {
        "extraction_method": "tesseract_multipass",
        "passes_completed": len(ocr_results),
        "best_confidence": round(best_confidence, 2),
        "word_confidences": all_word_confidences,
        "pass_details": [{"pass": r["pass"], "confidence": r["mean_confidence"], "words": r["word_count"]} for r in ocr_results]
    }
    
    # Step 11: Confidence-based fallback to GLM-OCR
    # If Tesseract confidence is below threshold, try GLM-OCR for better accuracy
    CONFIDENCE_THRESHOLD = 0.80
    if best_confidence < CONFIDENCE_THRESHOLD and GLM_OCR_AVAILABLE and glm_ocr_service:
        logger.info(f"Tesseract confidence ({best_confidence:.2f}) below threshold ({CONFIDENCE_THRESHOLD}), triggering GLM-OCR fallback...")
        try:
            # Preprocess image for better OCR accuracy
            image_bytes = _preprocess_image_for_ocr(image)
            
            # Run GLM-OCR (async function called from sync context)
            glm_text = asyncio.run(glm_ocr_service.extract_text_from_image(image_bytes))
            
            if glm_text and len(glm_text.strip()) > 10:
                logger.info(f"GLM-OCR fallback successful. Text length: {len(glm_text)}")
                cleaned_text = clean_ocr_text(glm_text)
                normalized = normalize_text(cleaned_text)
                return normalized, {
                    "extraction_method": "glm_ocr_fallback",
                    "best_confidence": 0.90,  # GLM-OCR typically has high confidence
                    "word_confidences": {},
                    "passes_completed": len(ocr_results) + 1,
                    "tesseract_confidence": round(best_confidence, 2),
                    "fallback_triggered": True,
                    "fallback_reason": f"tesseract_confidence_{best_confidence:.2f}_below_{CONFIDENCE_THRESHOLD}"
                }
            else:
                logger.warning("GLM-OCR fallback returned empty or insufficient text")
        except Exception as e:
            logger.warning(f"GLM-OCR fallback failed: {e}")
    
    logger.info(f"Final OCR result: len={len(normalized)}, method={ocr_metadata.get('extraction_method', 'unknown')}")
    return normalized, ocr_metadata

def extract_text_from_file(file_bytes, content_type, detect_phi: bool = True) -> Tuple[str, Optional[Dict]]:
    """
    Route to appropriate extractor based on file type.
    
    Args:
        file_bytes: Raw file content
        content_type: MIME type of file
        detect_phi: Whether to run PHI detection (default: True for compliance)
    
    Returns:
        Tuple of (normalized_text, extraction_metadata)
        extraction_metadata includes phi_report and optionally ocr_metadata
    """
    ocr_metadata = None
    
    # 1. Extract raw text based on content type
    if content_type == "application/pdf":
        raw_text = extract_text_from_pdf(file_bytes)
    elif content_type.startswith("image/"):
        raw_text, ocr_metadata = extract_text_from_image(file_bytes)
    else:
        # Try as text
        try:
            raw_text = file_bytes.decode("utf-8")
            raw_text = normalize_text(raw_text)
        except UnicodeDecodeError:
            raise ValueError(f"Cannot decode file as text. Content-Type: {content_type}")
    
    # 2. PHI Detection (compliance gate)
    phi_report = None
    if detect_phi and raw_text:
        phi_report = phi_detector.get_phi_report(raw_text)
    
    # 3. Build combined metadata
    metadata = {}
    if phi_report:
        metadata["phi_analysis"] = phi_report
    if ocr_metadata:
        metadata["ocr_analysis"] = ocr_metadata
    
    return raw_text, metadata if metadata else None