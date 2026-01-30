from pydantic import BaseModel, field_validator
from typing import List, Dict, Any
import re

class TextRequest(BaseModel):
    text: str
    
    @field_validator('text')
    @classmethod
    def validate_arabic_text(cls, v):
        """التحقق من أن النص يحتوي على حروف عربية (يقبل نصوص مختلطة)"""
        if not v or not v.strip():
            raise ValueError('النص لا يمكن أن يكون فارغاً')
        
        # التحقق من وجود حروف عربية
        has_arabic = bool(re.search(r'[\u0621-\u064A\u0671-\u06D3]', v))
        if not has_arabic:
            raise ValueError('النص يجب أن يحتوي على حروف عربية')
        
        # حساب نسبة الحروف العربية
        arabic_chars = len(re.findall(r'[\u0621-\u064A\u0671-\u06D3]', v))
        total_chars = len(re.findall(r'[\w]', v))
        
        if total_chars > 0 and (arabic_chars / total_chars) < 0.3:
            raise ValueError('النص يجب أن يحتوي على نسبة كافية من الحروف العربية')
        
        return v

class FileRequest(BaseModel):
    conversations: List[Dict[str, Any]]
    
    @field_validator('conversations')
    @classmethod
    def validate_conversations(cls, v):
        """التحقق من أن البيانات تحتوي على محادثات صالحة"""
        if not v or len(v) == 0:
            raise ValueError('يجب أن تحتوي البيانات على محادثة واحدة على الأقل')
        return v

class AnalysisResponse(BaseModel):
    result: Dict[str, Any]
