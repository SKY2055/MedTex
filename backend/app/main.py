"""
MedTex API — FastAPI application entry point.
"""

from __future__ import annotations

import logging
import os
import warnings

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import active_learning, auth, extract, verify
from backend.app.core.config import settings
from backend.app.core.database import init_db, SessionLocal

warnings.filterwarnings("ignore", message=".*Model.*was trained with spaCy.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# Load .env relative to this file
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=_env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Clinical NER API — extracts drugs, diseases, symptoms, dosages and more.",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    # 1. Create DB tables if they don't exist
    init_db()
    logger.info("Database initialised.")

    # 2. Warm up the NER engine (loads spaCy models)
    from backend.app.services.nlp_engine import medtex_engine  # noqa: F401
    logger.info("NER engine warmed up.")

    # 3. Load learned knowledge from DB → NER in-memory vocabulary
    try:
        db = SessionLocal()
        try:
            medtex_engine.load_knowledge_from_db(db)
            logger.info("Knowledge flywheel loaded into NER engine.")
        finally:
            db.close()
    except Exception as exc:
        logger.warning(f"Knowledge flywheel load failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(extract.router, prefix=settings.API_V1_STR, tags=["Extraction"])
app.include_router(verify.router, prefix=settings.API_V1_STR, tags=["Verification"])
app.include_router(auth.router, prefix=settings.API_V1_STR + "/auth", tags=["Authentication"])
app.include_router(
    active_learning.router,
    prefix=settings.API_V1_STR + "/active-learning",
    tags=["Active Learning"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "MedTex API running", "model": settings.MODEL_NAME}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": settings.VERSION}