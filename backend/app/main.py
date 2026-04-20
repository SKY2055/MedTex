from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api import extract
from backend.app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080"],  # Vite dev server and API port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routes
app.include_router(extract.router, prefix=settings.API_V1_STR, tags=["NER"])

@app.get("/")
async def root():
    return {"message": "MedTex API is running", "model": settings.MODEL_NAME}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.VERSION}