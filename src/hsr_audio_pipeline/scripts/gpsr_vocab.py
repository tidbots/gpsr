# gpsr_vocab.py
# GPSR コマンドジェネレータと合わせるための語彙定義

NAMES = [
    "Adel", "Angel", "Axel", "Charlie", "Jane",
    "Jules", "Morgan", "Paris", "Robin", "Simone",
]

LOCATIONS = [
    "bed", "bedside table", "shelf", "trashbin", "dishwasher",
    "potted plant", "kitchen table", "chairs", "pantry",
    "refrigerator", "sink", "cabinet", "coatrack", "desk",
    "armchair", "desk lamp", "waste basket", "tv stand",
    "storage rack", "lamp", "side tables", "sofa", "bookshelf",
    "entrance", "exit",
]

PLACEMENT_LOCATIONS = [
    "bed", "bedside table", "shelf", "dishwasher",
    "kitchen table", "pantry", "refrigerator", "sink",
    "cabinet", "desk", "tv stand", "storage rack",
    "side tables", "sofa", "bookshelf",
]

ROOMS = [
    "bedroom", "kitchen", "office", "living room", "bathroom",
]

OBJECTS = [
    "juice pack", "cola", "milk", "orange juice", "tropical juice",
    "red wine", "iced tea", "tennis ball", "rubiks cube", "baseball",
    "soccer ball", "dice", "orange", "pear", "peach", "strawberry",
    "apple", "lemon", "banana", "plum", "cornflakes", "pringles",
    "cheezit", "cup", "bowl", "fork", "plate", "knife", "spoon",
    "chocolate jello", "coffee grounds", "mustard", "tomato soup",
    "tuna", "strawberry jello", "spam", "sugar", "cleanser", "sponge",
]

CATEGORIES = [
    ("drink", "drinks"),
    ("toy", "toys"),
    ("fruit", "fruits"),
    ("snack", "snacks"),
    ("dish", "dishes"),
    ("food", "food"),
    ("cleaning supply", "cleaning supplies"),
]


def get_full_vocab(include_single_words: bool = True):
    """
    GPSR で出てくる可能性が高い語彙を lower-case セットで返す。

    include_single_words=True のとき:
      - "living room" -> "living room", "living", "room" も含む
    """
    vocab_phrases = []

    vocab_phrases.extend(NAMES)
    vocab_phrases.extend(LOCATIONS)
    vocab_phrases.extend(PLACEMENT_LOCATIONS)
    vocab_phrases.extend(ROOMS)
    vocab_phrases.extend(OBJECTS)

    # カテゴリ: 単数・複数両方
    vocab_phrases.extend([c[0] for c in CATEGORIES])
    vocab_phrases.extend([c[1] for c in CATEGORIES])

    vocab_set = set()

    for phrase in vocab_phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
        vocab_set.add(phrase.lower())

        if include_single_words:
            # "living room" → "living", "room" も語彙に加える
            tokens = phrase.lower().split()
            for t in tokens:
                if t:
                    vocab_set.add(t)

    return vocab_set
