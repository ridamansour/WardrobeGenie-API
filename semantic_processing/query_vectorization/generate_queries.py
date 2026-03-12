import random

# =========================================================
# KEYBOARD NEIGHBORS (for realistic typos)
# =========================================================

keyboard_neighbors = {
    "a": "sqwz", "b": "vghn", "c": "xdfv", "d": "serfcx",
    "e": "wsdr", "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb",
    "i": "ujko", "j": "huikmn", "k": "jiolm", "l": "kop",
    "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol",
    "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc",
    "y": "tghu", "z": "asx"
}

# Common realistic misspellings
common_misspellings = {
    "jacket": "jakcet",
    "sneakers": "snikers",
    "weather": "wether",
    "women": "womans",
    "comfortable": "comfy",
    "business": "buisness",
    "casual": "casul",
    "colour": "color",
    "color": "colour"
}

# =========================================================
# EUROPEAN ↔ AMERICAN SPELLINGS
# =========================================================

spelling_variants = {
    "color": ["color", "colour"],
    "colors": ["colors", "colours"],
    "gray": ["gray", "grey"],
    "favorite": ["favorite", "favourite"],
    "center": ["center", "centre"],
    "fiber": ["fiber", "fibre"]
}


def apply_spelling_variant(text):
    words = text.split()
    for i, w in enumerate(words):
        if w in spelling_variants:
            words[i] = random.choice(spelling_variants[w])
    return " ".join(words)


# =========================================================
# TYPO ENGINE
# =========================================================

def introduce_typo(word):
    if len(word) < 4:
        return word

    typo_type = random.choice([
        "delete", "swap", "double", "neighbor", "phonetic", "none"
    ])

    if typo_type == "delete":
        i = random.randrange(len(word))
        return word[:i] + word[i+1:]

    elif typo_type == "swap":
        i = random.randrange(len(word)-1)
        return word[:i] + word[i+1] + word[i] + word[i+2:]

    elif typo_type == "double":
        i = random.randrange(len(word))
        return word[:i] + word[i]*2 + word[i+1:]

    elif typo_type == "neighbor":
        i = random.randrange(len(word))
        c = word[i]
        if c in keyboard_neighbors:
            return word[:i] + random.choice(keyboard_neighbors[c]) + word[i+1:]

    elif typo_type == "phonetic" and word in common_misspellings:
        return common_misspellings[word]

    return word


def inject_typos(query, probability=0.25):
    words = query.split()

    for i in range(len(words)):
        if random.random() < probability:
            words[i] = introduce_typo(words[i])

    # drop space
    if random.random() < 0.08 and len(words) > 1:
        i = random.randrange(len(words)-1)
        words[i] += words[i+1]
        del words[i+1]

    # extra space
    if random.random() < 0.08:
        words.insert(random.randrange(len(words)), "")

    return " ".join(words).strip()


# =========================================================
# QUERY GENERATOR
# =========================================================

def generate_queries(num_samples=50000, output_file="fashion_queries_realworld.txt"):

    garments = [
        "jacket", "coat", "dress", "jeans", "hoodie",
        "sneakers", "boots", "suit", "skirt", "blazer",
        "sweater", "t-shirt", "activewear"
    ]

    colors = ["black", "white", "beige", "navy", "brown", "red", "neutral", "dark"]

    occasions = [
        "wedding guest", "date night", "office", "job interview",
        "gym", "travel", "graduation", "weekend brunch"
    ]

    weather = [
        "winter", "summer", "rainy day", "cold weather",
        "fall", "spring", "humid climate"
    ]

    styles = [
        "streetwear", "minimalist", "elegant", "casual",
        "sporty", "smart casual", "trendy", "vintage"
    ]

    gender_fit = [
        "", "men", "women", "unisex",
        "plus size", "petite", "tall fit", "oversized"
    ]

    materials = [
        "", "leather", "denim", "cotton", "wool",
        "linen", "waterproof", "fleece"
    ]

    price_intent = [
        "", "cheap", "affordable", "budget",
        "premium", "luxury", "under 100"
    ]

    brands = [
        "", "nike", "adidas", "uniqlo",
        "zara", "h&m", "north face"
    ]

    templates = [

        # short search fragments
        "{color} {garment} {weather}",
        "{garment} for {weather}",
        "{style} {garment}",
        "{material} {garment}",

        # intent + context
        "{garment} for {occasion}",
        "{style} {garment} for {occasion}",
        "{color} {material} {garment}",

        # shopping & decision intent
        "best {garment} for {weather}",
        "{price_intent} {garment}",
        "{brand} {garment}",

        # messy real-world phrasing
        "{garment} men winter",
        "{style} outfit women fall",
        "comfortable {garment} for walking all day",

        # conversational
        "what to wear to {occasion}",
        "outfit ideas for {occasion}",
        "how to style {garment}"
    ]

    queries = set()

    while len(queries) < num_samples:

        q = random.choice(templates).format(
            garment=random.choice(garments),
            color=random.choice(colors),
            occasion=random.choice(occasions),
            weather=random.choice(weather),
            style=random.choice(styles),
            gender_fit=random.choice(gender_fit),
            material=random.choice(materials),
            price_intent=random.choice(price_intent),
            brand=random.choice(brands),
        )

        q = " ".join(q.split()).lower().strip()

        # apply EU/US spelling variants
        if random.random() < 0.35:
            q = apply_spelling_variant(q)

        # inject typos
        if random.random() < 0.20:
            q = inject_typos(q)

        queries.add(q)

    with open(output_file, "w") as f:
        for q in queries:
            f.write(q + "\n")

    print(f"Generated {len(queries)} realistic queries → {output_file}")
    return list(queries)


if __name__ == "__main__":
    generate_queries()