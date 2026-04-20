from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.services.file_processor import extract_text_from_file
from backend.app.services.nlp_engine import medtex_engine
from backend.app.models.request import TextRequest
from backend.app.models.response import NERResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/extract", response_model=NERResponse)
async def extract_clinical_entities(payload: TextRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        entities = medtex_engine.extract_entities(payload.text)
    except Exception as e:
        logger.error(f"NER extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")
    
    return {
        "original_text": payload.text,
        "entities": entities,
        "count": len(entities)
    }

@router.post("/upload-prescription")
async def upload_prescription(file: UploadFile = File(...)):
    # 1. Read file
    file_content = await file.read()
    
    # 2. Extract text (PDF, image, or text file)
    try:
        raw_text = extract_text_from_file(file_content, file.content_type)
    except Exception as e:
        logger.error(f"File extraction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Could not extract text from file: {str(e)}")
    
    # 3. Run NER
    entities = medtex_engine.extract_entities(raw_text)
    
    return {
        "filename": file.filename,
        "original_text": raw_text,
        "entities": entities,
        "count": len(entities)
    }