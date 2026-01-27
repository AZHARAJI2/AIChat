INTEREST_KEYWORDS = {
    "AI & Programming": ["python", "model", "ai", "ml", "deep", "algorithm"],
    "Academic": ["research", "paper", "study", "exam"],
    "Business": ["startup", "market", "project", "money"],
    "Health": ["health", "diet", "exercise", "sleep"],
    "Creative": ["design", "write", "story", "art"]
}

def classify_interest(tokens):
    scores = {k: 0 for k in INTEREST_KEYWORDS}
    for word in tokens:
        for interest, keywords in INTEREST_KEYWORDS.items():
            if word in keywords:
                scores[interest] += 1

    return max(scores, key=scores.get)
