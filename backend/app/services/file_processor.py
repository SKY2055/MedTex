import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io

def extract_text_from_pdf(file_bytes):
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_image(file_bytes):
    """Extract text from image using OCR (Tesseract)"""
    image = Image.open(io.BytesIO(file_bytes))
    # Convert to RGB if necessary (for PNG with transparency)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    text = pytesseract.image_to_string(image)
    return text.strip()

def extract_text_from_file(file_bytes, content_type):
    """Route to appropriate extractor based on file type"""
    if content_type == "application/pdf":
        return extract_text_from_pdf(file_bytes)
    elif content_type.startswith("image/"):
        return extract_text_from_image(file_bytes)
    else:
        # Try as text
        return file_bytes.decode("utf-8")