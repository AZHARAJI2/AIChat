from models.profile_generator import ProfileGenerator

generator = ProfileGenerator()

def analyze_single_text(text: str):
    return generator.analyze_single(text)

def analyze_batch(conversations: list):
    return generator.analyze_batch(conversations)
