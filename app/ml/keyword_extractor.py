import re
from collections import Counter

# Stopwords français (articles, pronoms, prépositions, conjonctions,
# auxiliaires conjugués, adverbes courants). Les mots de moins de 3 lettres
# sont déjà écartés par la tokenisation, ils ne sont donc pas listés ici.
STOPWORDS: frozenset[str] = frozenset({
    # Articles, déterminants
    "les", "une", "des", "aux", "ces", "cet", "cette", "mon", "ton", "son",
    "nos", "vos", "mes", "tes", "ses", "leur", "leurs", "quel", "quelle",
    "quels", "quelles", "chaque", "quelque", "quelques", "plusieurs",
    "aucun", "aucune",
    # Pronoms
    "nous", "vous", "ils", "elle", "elles", "lui", "moi", "toi", "soi",
    "eux", "ceci", "cela", "celui", "celle", "ceux", "celles", "quoi",
    "dont", "lequel", "laquelle", "lesquels", "lesquelles", "rien",
    "personne", "tout", "tous", "toute", "toutes", "autre", "autres",
    "même", "meme", "mêmes",
    # Prépositions
    "par", "pour", "sur", "sous", "dans", "avec", "sans", "entre", "vers",
    "chez", "depuis", "pendant", "avant", "après", "apres", "contre",
    "parmi", "selon", "jusqu", "jusque", "durant", "via",
    # Conjonctions
    "mais", "donc", "car", "que", "qui", "lorsque", "quand", "comme",
    "parce", "puisque", "alors", "ainsi", "sinon", "voire", "tandis",
    # Être
    "être", "etre", "suis", "est", "sommes", "êtes", "etes", "sont",
    "étais", "etais", "était", "etait", "étions", "etions", "étiez",
    "etiez", "étaient", "etaient", "été", "ete", "serai", "seras", "sera",
    "serons", "serez", "seront", "serais", "serait", "serions", "seriez",
    "seraient", "sois", "soit", "soyons", "soyez", "soient", "fus", "fut",
    "étant", "etant",
    # Avoir
    "avoir", "avons", "avez", "ont", "avais", "avait", "avions", "aviez",
    "avaient", "eue", "eus", "aurai", "auras", "aura", "aurons", "aurez",
    "auront", "aurais", "aurait", "aurions", "auriez", "auraient", "aie",
    "aies", "ait", "ayons", "ayez", "aient", "ayant",
    # Aller, faire, dire, pouvoir, devoir (formes fréquentes)
    "aller", "vais", "vas", "allons", "allez", "vont", "allait", "faire",
    "fait", "fais", "faisons", "faites", "font", "faisait", "dire", "dit",
    "dis", "peux", "peut", "pouvons", "pouvez", "peuvent", "pouvait",
    "dois", "doit", "devons", "devez", "doivent", "devait", "faut",
    # Adverbes courants
    "pas", "plus", "très", "tres", "bien", "mal", "aussi", "beaucoup",
    "peu", "trop", "assez", "non", "oui", "jamais", "toujours", "souvent",
    "parfois", "encore", "déjà", "deja", "maintenant", "ici", "voici",
    "voilà", "voila", "ensuite", "enfin", "puis", "vraiment", "tellement",
    "complètement", "completement", "totalement", "absolument", "juste",
    "plutôt", "plutot", "presque", "surtout", "seulement", "aujourd", "hui",
    # Divers
    "dès", "lors", "cependant", "pourtant", "malgré", "malgre", "quant",
    "chose", "choses",
})

# Suites d'au moins 3 lettres Unicode (gère é, è, ç, œ, æ… ; exclut chiffres et « _ »).
_TOKEN_RE = re.compile(r"[^\W\d_]{3,}")


def extract_keywords(text: str, limit: int = 5) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    filtered = [token for token in tokens if token not in STOPWORDS]
    if not filtered:
        return []
    ranked = Counter(filtered).most_common(limit)
    return [word for word, _ in ranked]
