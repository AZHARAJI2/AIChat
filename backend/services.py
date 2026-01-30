from models.profile_generator import ProfileGenerator
import re

generator = ProfileGenerator()

def is_valid_arabic_text(text: str) -> tuple:
    """
    التحقق من صحة النص العربي
    يقبل النصوص المختلطة (عربي مع أرقام أو إنجليزي)
    
    Returns:
        (is_valid: bool, error_message: str)
    """
    if not text or not text.strip():
        return False, "النص فارغ"
    
    # التحقق من وجود حروف عربية
    has_arabic = bool(re.search(r'[\u0621-\u064A\u0671-\u06D3]', text))
    if not has_arabic:
        return False, "النص يجب أن يحتوي على حروف عربية"
    
    # حساب نسبة الحروف العربية في النص
    arabic_chars = len(re.findall(r'[\u0621-\u064A\u0671-\u06D3]', text))
    total_chars = len(re.findall(r'[\w]', text))
    
    # يجب أن يكون على الأقل 30% من الأحرف عربية
    if total_chars > 0 and (arabic_chars / total_chars) < 0.3:
        return False, "النص يجب أن يحتوي على نسبة كافية من الحروف العربية (30% على الأقل)"
    
    return True, ""

def analyze_single_text(text: str):
    """تحليل نص واحد مع التحقق من الصحة"""
    is_valid, error_msg = is_valid_arabic_text(text)
    if not is_valid:
        raise ValueError(error_msg)
    
    return generator.analyze_single(text)

def analyze_batch(conversations: list):
    """
    تحليل مجموعة من المحادثات
    يقبل البيانات المختلطة (عربي وإنجليزي) ويستخرج فقط النصوص العربية الصالحة
    """
    # استخراج النصوص من المحادثات
    texts = []
    for conv in conversations:
        if isinstance(conv, str):
            texts.append(conv)
        elif isinstance(conv, dict):
            prompts = conv.get('prompts_ar', conv.get('prompts', conv.get('text', conv.get('content', []))))
            if isinstance(prompts, str):
                texts.append(prompts)
            elif isinstance(prompts, list):
                texts.extend([p for p in prompts if isinstance(p, str)])
    
    if not texts:
        raise ValueError("لم يتم العثور على نصوص في البيانات المرسلة")
    
    # استخراج فقط النصوص العربية الصالحة (تجاهل الإنجليزية والأرقام)
    valid_texts = []
    for text in texts:
        is_valid, _ = is_valid_arabic_text(text)
        if is_valid and len(text.strip()) > 3:
            valid_texts.append(text)
    
    # يجب أن يكون هناك 3 نصوص عربية على الأقل
    if len(valid_texts) < 3:
        raise ValueError(f"تم العثور على {len(valid_texts)} نص عربي صالح فقط من إجمالي {len(texts)} نص. يجب أن تحتوي البيانات على 3 محادثات عربية صالحة على الأقل (بدون أرقام أو حروف إنجليزية)")
    
    return generator.analyze_batch(valid_texts)
