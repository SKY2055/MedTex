# MedTex - Market-Ready Clinical Named Entity Recognition System

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat&logo=spacy&logoColor=white)](https://spacy.io/)
[![PubMedBERT](https://img.shields.io/badge/PubMedBERT-0078D4?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/microsoft/)
[![HIPAA](https://img.shields.io/badge/HIPAA-Compliant-green)](https://www.hhs.gov/hipaa/)

MedTex is a **production-ready, industry-standard Clinical NER (Named Entity Recognition)** system that extracts medical entities from clinical text, PDFs, and images. It combines **PubMedBERT** for superior clinical entity classification, **RxNorm** for FDA-approved drug normalization, and includes comprehensive HIPAA compliance features.

## Features

### Core NLP Capabilities
- **PubMedBERT Integration**: State-of-the-art clinical entity classification with confidence scores
- **RxNorm Drug Normalization**: FDA-approved drug name normalization with CUI mapping
- **Ensemble NER Engine**: Med7 (medication) + SciSpacy (anatomy/chemicals) + BC5CDR (diseases)
- **GLM-OCR Support**: Handwritten prescription recognition via Ollama (privacy-preserving)
- **Confidence Scoring**: Entity-level confidence scores for quality assessment

### HIPAA Compliance
- **PHI Detection**: Automated detection of Protected Health Information (SSN, MRN, emails, addresses, etc.)
- **PHI Redaction**: Automatic redaction of detected PHI
- **Audit Logging**: HIPAA-compliant audit trails for all system actions
- **Role-Based Access Control**: OAuth2/JWT authentication with clinician/admin/reviewer roles

### Performance & Scalability
- **Redis Caching**: Performance optimization with configurable TTL
- **Batch Processing**: Concurrent text processing endpoint for bulk operations
- **Celery Background Tasks**: Asynchronous processing for long-running operations
- **API Rate Limiting**: Redis-backed sliding window rate limiting

### Clinical Interoperability
- **FHIR Export**: Convert extraction results to FHIR Bundle format (MedicationRequest, Condition, Observation)
- **Standardized Output**: Industry-standard clinical data formats

### Advanced Features
- **Active Learning Pipeline**: Identify uncertain predictions for human review and model improvement
- **Model Versioning**: Support for multiple model versions with A/B testing
- **Prometheus Monitoring**: Comprehensive metrics for observability
- **Human-in-the-Loop**: Verification workflow for continuous improvement

### Multi-Format Support
- **Text**: Direct clinical text input
- **PDF**: Document extraction with OCR support
- **Image**: OCR for scanned prescriptions and handwritten notes
- **Batch**: Large-scale document processing

## Tech Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **PubMedBERT** - Microsoft's clinical BERT model for entity classification
- **RxNorm API** - NIH drug normalization service
- **spaCy** - Industrial-strength Natural Language Processing
- **Med7** - Clinical NLP model for medication extraction
- **SciSpacy** - Biomedical NLP for anatomy and chemical detection
- **Celery** - Distributed task queue for background processing
- **Redis** - Caching and rate limiting
- **PostgreSQL** - Database for extractions and learned corrections
- **Prometheus** - Metrics and monitoring

### Frontend
- **React 18** - Component-based UI library
- **Vite** - Next-generation frontend build tool
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide Icons** - Beautiful, consistent icon set

## Project Structure

```
medtex/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── extract.py          # API endpoints for NER
│   │   │   ├── verify.py           # Human-in-the-loop verification
│   │   │   ├── auth.py             # Authentication endpoints
│   │   │   └── active_learning.py  # Active learning sampling
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py          # Application configuration
│   │   │   ├── database.py        # Database models
│   │   │   ├── auth.py            # JWT authentication
│   │   │   ├── audit.py           # Audit logging
│   │   │   ├── cache.py           # Redis caching
│   │   │   ├── rate_limit.py      # API rate limiting
│   │   │   ├── celery_app.py      # Celery configuration
│   │   │   ├── model_versioning.py # Model versioning
│   │   │   └── prometheus_metrics.py # Monitoring
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── request.py         # Pydantic request models
│   │   │   ├── response.py        # Pydantic response models
│   │   │   └── database.py        # SQLAlchemy models
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── file_processor.py  # PDF/Image text extraction & normalization
│   │       ├── nlp_engine.py      # Ensemble NER engine
│   │       ├── phi_service.py     # HIPAA PHI detection
│   │       ├── pubmedbert_service.py # PubMedBERT integration
│   │       ├── rxnorm_service.py  # RxNorm drug normalization
│   │       ├── glm_ocr_service.py # GLM-OCR for handwriting
│   │       ├── fhir_service.py    # FHIR export
│   │       └── active_learning.py # Active learning pipeline
│   ├── tasks/
│   │   └── celery_tasks.py        # Background tasks
│   └── jobs/
│       └── batch_processor.py     # PySpark batch processing
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx     # Main dashboard component
│   │   │   ├── EntityTable.jsx   # Entity display table
│   │   │   ├── HighlightedText.jsx  # Color-coded text display
│   │   │   ├── NoteInput.jsx     # Text input & file upload
│   │   │   └── MedicationSummary.jsx # Medication grouping
│   │   ├── utils/
│   │   │   └── colors.js         # Entity color mappings
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
├── requirements.txt               # Python dependencies
├── .env                           # Environment configuration
├── .env.example                   # Environment template
├── test.csv                     # Sample data for testing
└── README.md                      # This file
```

## Prerequisites

- **Python 3.12** (recommended - spacy compatible)
- **Node.js 18+**
- **PostgreSQL 16+** (for database persistence)
- **Redis 7+** (for caching and Celery broker)
- **Java 17+** (for PySpark batch processing - optional)
- **Tesseract OCR** (for image text extraction)
- **Ollama** (for GLM-OCR handwritten recognition - optional)
- **macOS/Linux/Windows**

## Installation

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd medtex

# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install System Dependencies

**PostgreSQL (macOS):**
```bash
brew install postgresql@16
brew services start postgresql@16
createdb medtex
```

**PostgreSQL (Ubuntu/Debian):**
```bash
sudo apt-get install postgresql postgresql-contrib
sudo -u postgres createdb medtex
```

**Redis (macOS):**
```bash
brew install redis
brew services start redis
```

**Redis (Ubuntu/Debian):**
```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

**Ollama (optional, for GLM-OCR):**
```bash
brew install ollama
ollama pull glm-ocr
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install NLP Models

```bash
# Install Med7 model
pip install --no-deps en_core_med7_lg-0.0.1-py3-none-any.whl

# Install SciSpacy BioNLP model
pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bionlp13cg_md-0.5.1.tar.gz

# Install BC5CDR model (for disease detection)
pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
```

### 5. Install Optional Java 17 (for batch processing)

```bash
# macOS
brew install openjdk@17
sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk

# Add to ~/.zshrc or ~/.bashrc
echo 'export JAVA_HOME="/opt/homebrew/opt/openjdk@17"' >> ~/.zshrc
echo 'export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### 6. Install Tesseract OCR (for images)

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

### 7. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Set DATABASE_URL, SECRET_KEY, and other settings
```

### 8. Initialize Database

```bash
# Database tables are auto-created on startup
# Or manually:
python -c "from backend.app.core.database import init_db; init_db()"
```

### 9. Install Frontend Dependencies

```bash
cd frontend
npm install
```

**Note:** If `npm install` hangs with SSL errors:
```bash
npm config set strict-ssl false
npm install
```

## How to Run

### Start the Backend

```bash
# From project root
source .venv/bin/activate
uvicorn backend.app.main:app --reload --port 8000
```

The API will be available at: **http://localhost:8000**

**API Documentation:** http://localhost:8000/docs

### Start Celery Worker (for background tasks)

```bash
# In a new terminal
source .venv/bin/activate
celery -A backend.app.core.celery_app worker --loglevel=info
```

### Start Celery Beat (for scheduled tasks)

```bash
# In a new terminal
source .venv/bin/activate
celery -A backend.app.core.celery_app beat --loglevel=info
```

### Start the Frontend

```bash
# In a new terminal
cd frontend
npm run dev
```

The frontend will be available at: http://localhost:5173

### Run Batch Processing

```bash
# Process CSV file with clinical notes
python backend/jobs/batch_processor.py test.csv output_data note
```

## API Endpoints

### Authentication

**POST** `http://localhost:8000/api/v1/auth/login`

Authenticate user and receive JWT token.

**Request:**
```json
{
  "username": "clinician",
  "password": "clinician123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "username": "clinician",
    "email": "clinician@medtex.com",
    "role": "clinician"
  }
}
```

**GET** `http://localhost:8000/api/v1/auth/me`

Get current authenticated user information.

**Headers:** `Authorization: Bearer <token>`

### 1. Text Extraction

**POST** `http://localhost:8000/api/v1/extract`

Extract entities from clinical text (requires authentication).

**Headers:** `Authorization: Bearer <token>`

**Request:**
```json
{
  "text": "Patient was prescribed Ibuprofen 200mg twice daily for 5 days"
}
```

**Response:**
```json
{
  "original_text": "Patient was prescribed Ibuprofen 200mg twice daily for 5 days",
  "entities": [
    {
      "text": "Ibuprofen",
      "label": "DRUG",
      "start": 23,
      "end": 32,
      "color": "#ffcfcc",
      "confidence": 0.95,
      "normalized_name": "Ibuprofen",
      "rxnorm_cui": "5640"
    },
    {
      "text": "200mg",
      "label": "STRENGTH",
      "start": 33,
      "end": 38,
      "color": "#c5f6fa",
      "confidence": 0.92
    }
  ],
  "medications": [...],
  "phi_analysis": {
    "phi_free": true,
    "detected_phi": {...}
  },
  "extraction_id": 1,
  "count": 2
}
```

### 2. Batch Processing

**POST** `http://localhost:8000/api/v1/batch`

Process multiple texts concurrently (requires authentication).

**Headers:** `Authorization: Bearer <token>`

**Request:**
```json
{
  "texts": [
    "Patient prescribed Metformin 500mg daily",
    "Patient prescribed Lisinopril 10mg twice daily"
  ],
  "clinical_summary": true
}
```

**Response:**
```json
[
  {
    "index": 0,
    "success": true,
    "result": {...}
  },
  {
    "index": 1,
    "success": true,
    "result": {...}
  }
]
```

### 3. FHIR Export

**POST** `http://localhost:8000/api/v1/export/fhir/{extraction_id}`

Export extraction result to FHIR Bundle format (requires authentication).

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "MedicationRequest",
        "medicationCodeableConcept": {
          "coding": [{
            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
            "code": "5640",
            "display": "Ibuprofen"
          }]
        }
      }
    }
  ]
}
```

### 4. Active Learning

**GET** `http://localhost:8000/api/v1/active-learning/samples`

Get uncertain predictions for human review (requires authentication).

**Headers:** `Authorization: Bearer <token>`

**Query Parameters:**
- `sample_type`: uncertain, diverse, or low_confidence
- `limit`: Number of samples (default: 10)

**Response:**
```json
{
  "sample_type": "uncertain",
  "count": 10,
  "samples": [
    {
      "extraction_id": 123,
      "raw_text": "...",
      "entities": [...],
      "avg_confidence": 0.65
    }
  ]
}
```

### 5. File Upload (PDF/Image)

**POST** `http://localhost:8000/api/v1/upload-prescription`

Upload PDF or image files for OCR and entity extraction (requires authentication).

**Headers:** `Authorization: Bearer <token>`

**Request:** Multipart form data with `file` field

**Supported formats:** PDF, PNG, JPG, JPEG, GIF, BMP, TIFF

**Response:** Same format as text extraction with `filename` field

### 6. Health Check

**GET** `http://localhost:8000/health`

Check API status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## Supported Entity Types

### PubMedBERT Entities (Clinical)
| Label | Description | Example |
|-------|-------------|---------|
| DRUG | Medication name | Ibuprofen, Amoxicillin, Metformin |
| DISEASE | Medical conditions | Diabetes, Hypertension, Pneumonia |
| SYMPTOM | Clinical symptoms | Chest pain, Headache, Fever |
| ANATOMY | Body parts | Left Ventricle, Respiratory System |
| LAB | Laboratory values | Glucose, Sodium Chloride |
| PROCEDURE | Medical procedures | MRI, CT Scan, Surgery |

### Med7 Entities (Medication)
| Label | Description | Example |
|-------|-------------|---------|
| DRUG | Medication name | Ibuprofen, Amoxicillin |
| DOSAGE | Amount to take | 200mg, 500mg |
| DURATION | How long to take | 5 days, 2 weeks |
| FORM | Medication form | tablet, capsule |
| FREQUENCY | How often | twice daily, three times a day |
| ROUTE | How to administer | orally, intravenously |
| STRENGTH | Drug strength | 1000mg, 10mg/ml |

### SciSpacy Entities (Anatomy/Chemicals)
| Label | Description | Example |
|-------|-------------|---------|
| ANATOMY | Body parts | Left Ventricle, Respiratory System |
| CHEMICAL | Chemical substances | Glucose, Sodium Chloride |
| ORGANISM | Biological organisms | E. coli, HIV |

### BC5CDR Entities (Diseases)
| Label | Description | Example |
|-------|-------------|---------|
| DISEASE | Disease names | Diabetes, Cancer, Pneumonia |

## PHI Detection & HIPAA Compliance

MedTex includes an intelligent PHI (Protected Health Information) detection system that serves as the first gate for compliance-ready workflows.

### Detected PHI Types

| Type | Pattern | Confidence |
|------|---------|------------|
| SSN | Social Security Numbers | 95% |
| PHONE | Phone numbers | 90% |
| EMAIL | Email addresses | 98% |
| MRN | Medical Record Numbers | 85% |
| DATE_OF_BIRTH | Birth dates | 80% |
| ADDRESS | Street addresses | 75% |
| FULL_NAME | Patient/provider names | 70% |

### API Usage

PHI detection is enabled by default on file uploads:

```bash
curl -X POST http://localhost:8080/api/v1/upload-prescription \
  -F "file=@prescription.pdf" \
  -F "detect_phi=true"
```

**Response includes PHI analysis:**

```json
{
  "filename": "prescription.pdf",
  "original_text": "Patient John Doe, MRN: 123456...",
  "entities": [...],
  "phi_analysis": {
    "phi_detected": true,
    "total_findings": 3,
    "phi_types": {
      "FULL_NAME": 1,
      "MRN": 1
    },
    "risk_level": "MEDIUM",
    "findings": [
      {"type": "FULL_NAME", "text": "Joh...", "confidence": 0.85}
    ],
    "recommendation": "Review and redact before sharing"
  }
}
```

### Text Normalization

Medical abbreviations are automatically standardized:

| Abbreviation | Normalized |
|--------------|------------|
| q.d. / qd | daily |
| b.i.d. / bid | twice daily |
| t.i.d. / tid | three times daily |
| p.o. / po | by mouth |
| i.v. / iv | intravenous |
| tabs | tablet |
| caps | capsule |

## Architecture

### Backend Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   FastAPI       │────▶│   NER Engine    │────▶│   PubMedBERT    │
│   (REST API)    │     │   (MedTexEngine)│     │   (Transformers)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ├────────────────┬────────────────┐
                               ▼                ▼                ▼
                        ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
                        │   Med7 Model    │ │   SciSpacy      │ │   BC5CDR        │
                        │   (spaCy)       │ │   (BioNLP)      │ │   (Disease)     │
                        └─────────────────┘ └─────────────────┘ └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │   RxNorm API   │
                        │   (Normalization)│
                        └─────────────────┘
```

**Key Components:**
- **MedTexEngine**: Ensemble processor that merges results from PubMedBERT, Med7, SciSpacy, and BC5CDR
- **PubMedBERT Service**: State-of-the-art clinical entity classification with confidence scores
- **RxNorm Service**: FDA-approved drug normalization with CUI mapping
- **PHI Service**: HIPAA-compliant PHI identification and redaction
- **Cache Service**: Redis-backed caching for performance optimization
- **Rate Limiter**: Redis-backed sliding window rate limiting
- **Audit Logger**: HIPAA-compliant audit trails
- **Active Learning**: Uncertain prediction sampling for model improvement
- **Celery Tasks**: Background processing for long-running operations

### Frontend Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Dashboard     │────▶│   NoteInput     │────▶│   API (FastAPI) │
│   (Container)   │     │   (Text/Upload) │     │   (Auth + NER)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │
        ├────────────────┬────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Highlighted  │ │ EntityTable  │ │ Medication   │
│ Text         │ │              │ │ Summary      │
└──────────────┘ └──────────────┘ └──────────────┘
```

**Key Components:**
- **Dashboard**: Main container managing state and API calls with authentication
- **NoteInput**: Handles text input and file uploads (PDF, images)
- **HighlightedText**: Renders color-coded entities with confidence scores
- **EntityTable**: Displays extracted entities in tabular format with RxNorm CUIs
- **MedicationSummary**: Groups medications and supports human verification

## Color Coding

Entities are color-coded in the UI for quick visual identification:

| Entity Type | Color | Hex Code |
|-------------|-------|----------|
| DRUG | Soft Red | `#ffcfcc` |
| DOSAGE | Soft Green | `#d3f9d8` |
| DURATION | Soft Yellow | `#fff3bf` |
| FORM | Soft Purple | `#eebefa` |
| FREQUENCY | Soft Blue | `#d0ebff` |
| ROUTE | Soft Orange | `#fff4e6` |
| ANATOMY | Soft Cyan | `#c5f6fa` |
| CHEMICAL | Soft Yellow | `#ffec99` |

## Troubleshooting

### Backend Issues

**Issue:** `ModuleNotFoundError: No module named 'backend'`
**Solution:** Ensure `__init__.py` files exist in all backend subdirectories.

**Issue:** `OSError: [E050] Can't find model 'en_core_med7_lg'`
**Solution:** Install the Med7 model: `pip install --no-deps en_core_med7_lg-0.0.1-py3-none-any.whl`

**Issue:** `JAVA_GATEWAY_EXITED` with PySpark
**Solution:** Ensure Java 17 is installed and `JAVA_HOME` is set correctly.

### Frontend Issues

**Issue:** `npm install` hangs or fails with SSL errors
**Solution:** 
```bash
npm config set strict-ssl false
npm install
```

**Issue:** CORS errors in browser console
**Solution:** Ensure backend CORS is configured for `http://localhost:5173` (frontend) and API uses port `8080`

### OCR Issues

**Issue:** Image upload returns empty text
**Solution:** Install Tesseract OCR: `brew install tesseract` (macOS) or `apt-get install tesseract-ocr` (Linux)

## Dependencies

### Python Packages
- `fastapi>=0.104.0` - Web framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `spacy==3.7.0` - NLP library
- `transformers>=4.30.0` - HuggingFace transformers for PubMedBERT
- `torch>=2.0.0` - PyTorch for deep learning models
- `celery>=5.3.0` - Background task queue
- `redis>=4.5.0` - Redis for caching and Celery broker
- `sqlalchemy>=2.0.0` - ORM for database
- `psycopg2-binary>=2.9.0` - PostgreSQL adapter
- `python-jose[cryptography]>=3.3.0` - JWT token handling
- `passlib[bcrypt]>=1.7.4` - Password hashing
- `prometheus_client>=0.25.0` - Prometheus metrics
- `pyspark>=3.5.0` - Distributed processing (optional)
- `pyarrow>=15.0.0` - Data serialization
- `pymupdf>=1.23.0` - PDF processing
- `pillow>=10.0.0` - Image processing
- `pytesseract>=0.3.10` - OCR
- `ollama>=0.1.0` - Local VLM OCR (optional)

### Node.js Packages
- `react@^18.2.0` - UI library
- `vite@^5.0.8` - Build tool
- `tailwindcss@^3.4.0` - CSS framework

### System Requirements
- PostgreSQL 16+ - Database
- Redis 7+ - Caching and message broker
- Ollama (optional) - Local OCR for handwriting
- Tesseract OCR - Image text extraction
- Java 17+ (optional) - For PySpark batch processing
- **Note**: Use Python 3.12 for spacy compatibility (Python 3.14 not supported)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- **Med7 Model**: [kormilitzin/med7](https://github.com/kormilitzin/med7) - Clinical NLP for medication extraction
- **SciSpacy**: [allenai/scispacy](https://github.com/allenai/scispacy) - Biomedical NLP models
- **spaCy**: [explosion/spaCy](https://github.com/explosion/spaCy) - Industrial-strength NLP
