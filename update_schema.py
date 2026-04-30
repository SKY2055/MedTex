"""
Script to add missing columns to the extractions table
"""
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sky:sky123@localhost:5432/medtex_db")

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    # Check if columns exist
    result = conn.execute(text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'extractions'
    """))
    existing_columns = {row[0] for row in result}
    print(f"Existing columns: {existing_columns}")
    
    # Add missing columns
    if 'source' not in existing_columns:
        conn.execute(text("ALTER TABLE extractions ADD COLUMN source VARCHAR(30) DEFAULT 'text'"))
        print("Added 'source' column")
    
    if 'ocr_method' not in existing_columns:
        conn.execute(text("ALTER TABLE extractions ADD COLUMN ocr_method VARCHAR(40)"))
        print("Added 'ocr_method' column")
    
    if 'ocr_confidence' not in existing_columns:
        conn.execute(text("ALTER TABLE extractions ADD COLUMN ocr_confidence FLOAT"))
        print("Added 'ocr_confidence' column")
    
    if 'phi_detected' not in existing_columns:
        conn.execute(text("ALTER TABLE extractions ADD COLUMN phi_detected BOOLEAN DEFAULT FALSE"))
        print("Added 'phi_detected' column")
    
    if 'document_type' not in existing_columns:
        conn.execute(text("ALTER TABLE extractions ADD COLUMN document_type VARCHAR(30) DEFAULT 'unknown'"))
        print("Added 'document_type' column")
    
    # Add indexes if they don't exist
    if 'ix_extractions_status' not in existing_columns:
        conn.execute(text("CREATE INDEX ix_extractions_status ON extractions(status)"))
        print("Added 'ix_extractions_status' index")
    
    if 'ix_extractions_created_at' not in existing_columns:
        conn.execute(text("CREATE INDEX ix_extractions_created_at ON extractions(created_at)"))
        print("Added 'ix_extractions_created_at' index")
    
    if 'ix_extractions_status_created' not in existing_columns:
        conn.execute(text("CREATE INDEX ix_extractions_status_created ON extractions(status, created_at)"))
        print("Added 'ix_extractions_status_created' index")
    
    conn.commit()
    print("Schema update complete")
