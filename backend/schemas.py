from pydantic import BaseModel
from typing import List, Dict, Any

class TextRequest(BaseModel):
    text: str

class FileRequest(BaseModel):
    conversations: List[Dict[str, Any]]

class AnalysisResponse(BaseModel):
    result: Dict[str, Any]
