import re
from collections import Counter

STOPWORDS = {
    "a",
    "ai",
    "au",
    "aux",
    "avec",
    "beaucoup",
    "car",
    "ce",
    "ces",
    "cet",
    "cette",
    "comme",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "est",
    "et",
    "il",
    "je",
    "j",
    "la",
    "le",
    "les",
    "leur",
    "mais",
    "me",
    "mes",
    "mon",
    "ne",
    "nos",
    "notre",
    "ou",
    "par",
    "pas",
    "plus",
    "pour",
    "que",
    "qui",
    "se",
    "ses",
    "son",
    "sur",
    "te",
    "tes",
    "ton",
    "tres",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "y",
}


def extract_keywords(text: str, limit: int = 5) -> list[str]:
    tokens = re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower())
    filtered = [
        token.strip("'")
        for token in tokens
        if len(token.strip("'")) > 2 and token.strip("'") not in STOPWORDS
    ]
    if not filtered:
        return []

    ranked = Counter(filtered).most_common(limit)
    return [word for word, _ in ranked]
