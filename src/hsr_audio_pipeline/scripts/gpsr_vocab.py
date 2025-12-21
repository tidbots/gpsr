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

# gpsr_vocab.py に追記（既存があれば末尾に追加）

from pathlib import Path
import re

def _read(p: str) -> str:
    return Path(p).read_text(encoding="utf-8", errors="ignore")

def load_from_md(names_md: str, rooms_md: str, locations_md: str, objects_md: str, test_objects_md: str = ""):
    # --- names ---
    names = re.findall(r"\|\s*([A-Za-z]+)\s*\|", _read(names_md))
    names = [x.strip() for x in names][1:] if names else []

    # --- rooms ---
    rooms = re.findall(r"\|\s*(\w+ \w*)\s*\|", _read(rooms_md))
    rooms = [x.strip() for x in rooms][1:] if rooms else []

    # --- locations (+ placement) ---
    loc_pairs = re.findall(r"\|\s*([0-9]+)\s*\|\s*([A-Za-z,\s\(\)]+)\|", _read(locations_md))
    locs = [b.strip() for (_, b) in loc_pairs]
    placement = [x.replace("(p)", "").strip() for x in locs if x.strip().endswith("(p)")]
    locs = [x.replace("(p)", "").strip() for x in locs]

    # --- objects + categories ---
    md_obj = _read(objects_md)
    obj_names = re.findall(r"\|\s*(\w+)\s*\|", md_obj)
    obj_names = [o for o in obj_names if o != "Objectname"]
    obj_names = [o.replace("_", " ").strip() for o in obj_names if o.strip()]

    cats = re.findall(r"# Class \s*([\w,\s\(\)]+)\s*", md_obj)
    cats = [c.strip().replace("(", "").replace(")", "") for c in cats]
    cat_plur, cat_sing = [], []
    for c in cats:
        parts = c.split()
        if len(parts) >= 2:
            cat_plur.append(parts[0].replace("_", " "))
            cat_sing.append(parts[1].replace("_", " "))

    # test_objects は「hotwords増量」用途で object_names に足す（パースには必須ではない）
    if test_objects_md:
        md_test = _read(test_objects_md)
        test_objs = re.findall(r"\|\s*(\w+)\s*\|", md_test)
        test_objs = [o for o in test_objs if o != "Objectname"]
        test_objs = [o.replace("_", " ").strip() for o in test_objs if o.strip()]
        obj_names = sorted(set(obj_names + test_objs), key=lambda s: (len(s), s.lower()))

    return dict(
        person_names=names,
        room_names=rooms,
        location_names=locs,
        placement_location_names=sorted(set(placement)),
        object_names=sorted(set(obj_names)),
        object_categories_plural=sorted(set(cat_plur)),
        object_categories_singular=sorted(set(cat_sing)),
    )
