PERSONA_RULES = {
    "Learner": ["what", "how", "explain", "learn"],
    "Problem Solver": ["error", "fix", "issue", "bug"],
    "Explorer": ["trend", "future", "new", "latest"]
}

def detect_persona(text: str):
    text = text.lower()
    scores = {p: 0 for p in PERSONA_RULES}

    for persona, keywords in PERSONA_RULES.items():
        for kw in keywords:
            if kw in text:
                scores[persona] += 1

    return max(scores, key=scores.get)
