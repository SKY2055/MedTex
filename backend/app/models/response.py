from pydantic import BaseModel
from typing import List

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    color: str

class NERResponse(BaseModel):
    original_text: str
    entities: List[Entity]
    count: int