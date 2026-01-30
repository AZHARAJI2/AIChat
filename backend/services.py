from models.profile_generator import ProfileGenerator

generator = ProfileGenerator()

def analyze_single_text(text: str):
    return generator.analyze_single(text)

def analyze_batch(conversations: list):
    # استخدام الاسم الصحيح للدالة analyze_batch
    return generator.analyze_batch(conversations)
