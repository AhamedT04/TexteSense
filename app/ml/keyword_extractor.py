import re
from collections import Counter

# Stopwords français complets (articles, pronoms, prépositions, conjonctions,
# auxiliaires, adverbes courants et formes conjuguées fréquentes)
STOPWORDS: frozenset[str] = frozenset({
    # Articles
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "d",
    # Pronoms personnels
    "je", "j", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "me", "m", "te", "t", "se", "s", "lui", "leur", "y", "en",
    "moi", "toi", "soi",
    # Pronoms démonstratifs / indéfinis
    "ce", "cet", "cette", "ces", "ceci", "cela", "ca", "ça",
    "on", "quelque", "chaque", "tout", "tous", "toute", "toutes",
    "autre", "autres", "même", "meme",
    # Prépositions
    "a", "à", "au", "aux", "de", "du", "des", "en", "par", "pour",
    "sur", "sous", "dans", "avec", "sans", "entre", "vers", "chez",
    "depuis", "pendant", "avant", "après", "apres", "contre",
    "parmi", "selon",
    # Conjonctions
    "et", "ou", "mais", "donc", "or", "ni", "car", "que", "qui",
    "quoi", "dont", "où", "ou", "si", "lorsque", "quand", "comme",
    "parce", "puisque", "bien", "alors", "ainsi",
    # Auxiliaires et formes verbales courantes
    "est", "sont", "être", "etre", "avoir", "ai", "as", "avons",
    "avez", "ont", "été", "ete", "avoir", "sera", "seront", "était",
    "etait", "etaient", "était", "avait", "avaient", "fait", "faire",
    "dit", "dire", "va", "vais", "allons", "allez", "vont",
    # Déterminants possessifs
    "mon", "ton", "son", "ma", "ta", "sa", "nos", "vos", "mes",
    "tes", "ses", "leur", "leurs",
    # Adverbes courants
    "ne", "pas", "plus", "très", "tres", "bien", "mal", "aussi",
    "beaucoup", "peu", "trop", "assez", "très", "si", "non", "oui",
    "jamais", "toujours", "souvent", "parfois", "encore", "déjà",
    "deja", "maintenant", "ici", "là", "la", "voici", "voilà",
    "voila", "alors", "donc", "ensuite", "enfin", "puis",
    # Divers
    "dont", "dès", "des", "lors", "cependant", "pourtant",
    "malgré", "malgre", "quant", "sinon", "voire",
})


def extract_keywords(text: str, limit: int = 5) -> list[str]:
    # Tokenisation : lettres + accents, longueur minimale 3
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]{3,}", text.lower())

    filtered = [
        token
        for token in tokens
        if token not in STOPWORDS and not token.isdigit()
    ]

    if not filtered:
        return []

    ranked = Counter(filtered).most_common(limit)
    return [word for word, _ in ranked]
