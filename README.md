# MedTex - Clinical Named Entity Recognition System

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat&logo=spacy&logoColor=white)](https://spacy.io/)
[![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=flat&logo=apachespark&logoColor=white)](https://spark.apache.org/)

MedTex is a production-ready Clinical NER (Named Entity Recognition) system that extracts medical entities from clinical text, PDFs, and images. It combines **Med7** for medication details and **SciSpacy** for anatomy and chemical detection.

## Features

- **Dual-Model NER Engine**: Combines Med7 (medication) + SciSpacy (anatomy/chemicals)
- **Multi-Format Support**: Text, PDF, and image (OCR) extraction
- **Real-time Processing**: FastAPI backend with React frontend
- **Batch Processing**: PySpark support for large-scale document processing
- **Visual Entity Highlighting**: Color-coded entities in the UI
- **Professional UI**: Drag-and-drop file upload with prescription analysis

## Tech Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **spaCy** - Industrial-strength Natural Language Processing
- **Med7** - Clinical NLP model for medication extraction
- **SciSpacy** - Biomedical NLP for anatomy and chemical detection
- **PySpark** - Large-scale data processing for batch jobs
- **PyMuPDF** - PDF text extraction
- **Pytesseract** - OCR for image text extraction
- **Pydantic** - Data validation using Python type hints

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
│   │   │   └── extract.py          # API endpoints for NER
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── config.py          # Application configuration
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── request.py         # Pydantic request models
│   │   │   └── response.py        # Pydantic response models
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── file_processor.py  # PDF/Image text extraction
│   │       └── nlp_engine.py      # Dual-model NER engine
│   └── jobs/
│       └── batch_processor.py     # PySpark batch processing
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx     # Main dashboard component
│   │   │   ├── EntityTable.jsx   # Entity display table
│   │   │   ├── HighlightedText.jsx  # Color-coded text display
│   │   │   └── NoteInput.jsx     # Text input & file upload
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
├── test.csv                     # Sample data for testing
└── README.md                      # This file
```

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Java 17+** (for PySpark batch processing)
- **Tesseract OCR** (for image text extraction)
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

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Med7 Model

```bash
pip install --no-deps en_core_med7_lg-0.0.1-py3-none-any.whl
```

### 4. Install SciSpacy Model

```bash
pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bionlp13cg_md-0.5.1.tar.gz
```

### 5. Install Java 17 (for batch processing)

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

### 7. Install Frontend Dependencies

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
uvicorn backend.app.main:app --reload --port 8080
```

The API will be available at: **http://localhost:8080**

**API Documentation:** http://localhost:8080/docs

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

### 1. Text Extraction

**POST** `http://localhost:8080/api/v1/extract`

Extract entities from clinical text.

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
      "color": "#ffcfcc"
    },
    {
      "text": "200mg",
      "label": "STRENGTH",
      "start": 33,
      "end": 38,
      "color": "#c5f6fa"
    }
  ],
  "count": 2
}
```

### 2. File Upload (PDF/Image)

**POST** `http://localhost:8080/api/v1/upload-prescription`

Upload PDF or image files for OCR and entity extraction.

**Request:** Multipart form data with `file` field

**Supported formats:** PDF, PNG, JPG, JPEG, GIF, BMP, TIFF

**Response:** Same format as text extraction with `filename` field

### 3. Health Check

**GET** `http://localhost:8080/health`

Check API status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## Supported Entity Types

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

## Architecture

### Backend Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   FastAPI       │────▶│   NER Engine    │────▶│   Med7 Model    │
│   (REST API)    │     │   (MedTexEngine)│     │   (spaCy)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │   SciSpacy      │
                        │   (BioNLP)      │
                        └─────────────────┘
```

**Key Components:**
- **MedTexEngine**: Dual-model processor that merges results from Med7 and SciSpacy
- **File Processor**: Routes files to appropriate extractor (PDF, image, or text)
- **Batch Processor**: PySpark-based distributed processing for large datasets

### Frontend Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Dashboard     │────▶│   NoteInput     │────▶│   API (FastAPI) │
│   (Container)   │     │   (Text/Upload) │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │
        ├────────────────┬────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Highlighted  │ │ EntityTable  │ │ Error Display│
│ Text         │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
```

**Key Components:**
- **Dashboard**: Main container managing state and API calls
- **NoteInput**: Handles text input and file uploads (PDF, images)
- **HighlightedText**: Renders color-coded entities
- **EntityTable**: Displays extracted entities in tabular format

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
- `pyspark>=3.5.0` - Distributed processing
- `pyarrow>=15.0.0` - Data serialization
- `pymupdf>=1.23.0` - PDF processing
- `pillow>=10.0.0` - Image processing
- `pytesseract>=0.3.10` - OCR

### Node.js Packages
- `react@^18.2.0` - UI library
- `vite@^5.0.8` - Build tool
- `tailwindcss@^3.4.0` - CSS framework

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- **Med7 Model**: [kormilitzin/med7](https://github.com/kormilitzin/med7) - Clinical NLP for medication extraction
- **SciSpacy**: [allenai/scispacy](https://github.com/allenai/scispacy) - Biomedical NLP models
- **spaCy**: [explosion/spaCy](https://github.com/explosion/spaCy) - Industrial-strength NLP
