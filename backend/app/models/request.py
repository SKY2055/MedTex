from pydantic import BaseModel, Field

class TextRequest(BaseModel):
    text: str = Field(..., max_length=50000, description="Clinical text to analyze (max 50KB)")