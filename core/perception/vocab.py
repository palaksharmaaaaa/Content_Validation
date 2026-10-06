"""core.perception.vocab: the closed vocabularies the zero-shot recognizer chooses between.

Each entry is ``label`` or ``(label, prompt)``. A closed vocabulary always returns *something*, so callers must apply a
confidence threshold and treat a low top score as "not sure". Add or remove entries freely: no retraining is needed.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# (label, is_indoor). Used for "where was this taken" and the indoor/outdoor decision.
SCENES: List[Tuple[str, bool]] = [
    ("living room", True), ("bedroom", True), ("kitchen", True), ("bathroom", True), ("dining room", True), ("office", True),
    ("classroom", True), ("lecture hall", True), ("library", True), ("restaurant", True), ("cafe", True), ("shop interior", True),
    ("shopping mall", True), ("hospital", True), ("gym", True), ("hotel room", True), ("auditorium or stage", True),
    ("wedding hall", True), ("temple interior", True), ("car interior", True), ("train or bus interior", True), ("corridor", True),
    ("staircase", True), ("hostel room", True), ("laboratory", True),
    ("city street", False), ("market", False), ("highway or road", False), ("railway station", False), ("airport", False),
    ("temple courtyard", False), ("palace or fort", False), ("historic monument", False), ("garden", False), ("park", False),
    ("lawn", False), ("playground", False), ("sports field", False), ("beach", False), ("lake or river", False), ("waterfall", False),
    ("forest", False), ("mountains", False), ("desert", False), ("farm or field", False), ("village", False), ("campus building", False),
    ("rooftop or terrace", False), ("balcony", False), ("parking lot", False), ("bridge", False), ("construction site", False),
    ("stadium", False), ("amusement park", False), ("snow landscape", False), ("sunset sky", False), ("night city", False),
]

TIME_OF_DAY: List[Tuple[str, str]] = [
    ("bright daylight", "a photo taken in bright daylight"),
    ("overcast daylight", "a photo taken on an overcast day"),
    ("golden hour or sunset", "a photo taken at sunset in warm golden light"),
    ("night", "a photo taken at night in the dark"),
    ("indoor artificial light", "a photo taken indoors under artificial lamp light"),
    ("studio lighting", "a photo taken with professional studio lighting"),
]

GENRE: List[Tuple[str, str]] = [
    ("selfie", "a selfie taken at arm's length"),
    ("portrait", "a portrait photo of one person"),
    ("group photo", "a group photo of several people posing"),
    ("full-body photo", "a full-body photo of a person standing"),
    ("candid street photo", "a candid street photograph"),
    ("landscape", "a landscape photograph"),
    ("food", "a photo of food"),
    ("product", "a product photo"),
    ("pet or animal", "a photo of an animal"),
    ("event or ceremony", "a photo from an event or ceremony"),
    ("document or screenshot", "a screenshot or a photo of a document"),
    ("artwork or illustration", "a drawing, painting or digital illustration"),
]

ANIMALS: List[str] = [
    "dog", "puppy", "labrador retriever", "german shepherd", "golden retriever", "pug", "beagle", "bulldog", "husky", "street dog", "pomeranian",
    "cat", "kitten", "tabby cat", "black cat", "persian cat", "horse", "pony", "donkey", "camel", "cow", "buffalo", "bull", "goat", "sheep",
    "pig", "elephant", "monkey", "langur", "squirrel", "rabbit", "deer", "tiger", "lion", "leopard", "bear", "fox", "wolf", "zebra", "giraffe",
    "peacock", "parrot", "pigeon", "crow", "sparrow", "eagle", "owl", "duck", "swan", "hen or rooster", "kingfisher", "heron", "seagull",
    "fish", "turtle", "crocodile", "snake", "lizard", "frog", "butterfly", "bee", "spider", "cockroach", "mosquito",
]

VEHICLES: List[str] = [
    "hatchback car", "sedan car", "SUV", "pickup truck", "minivan", "sports car", "taxi", "police car", "ambulance", "jeep",
    "auto rickshaw", "e-rickshaw", "motorcycle", "scooter", "moped", "bicycle", "electric scooter", "city bus", "school bus", "coach bus",
    "truck", "tipper truck", "tanker truck", "tractor", "bulldozer", "train", "metro train", "tram", "airplane", "helicopter",
    "fishing boat", "ship", "speedboat", "cart or carriage", "toy car",
]

# Age-group descriptions for the second, independent opinion in core.perception.age (the first labels are the minor ones).
AGE_GROUPS: List[Tuple[str, str]] = [
    ("baby", "a photo of a baby."), ("toddler", "a photo of a toddler."), ("child", "a photo of a young child."),
    ("teenager", "a photo of a teenager."), ("young adult", "a photo of a young adult in their twenties."),
    ("adult", "a photo of an adult in their thirties or forties."), ("middle-aged", "a photo of a middle-aged person."),
    ("elderly", "a photo of an elderly person."),
]

VOCABS: Dict[str, List[Tuple[str, str]]] = {}


VOCABS["scene"] = [(label, f"a photo of a {label}.") for label, _indoor in SCENES]
VOCABS["time_of_day"] = TIME_OF_DAY
VOCABS["genre"] = GENRE
VOCABS["animal"] = [(a, f"a photo of a {a}.") for a in ANIMALS]
VOCABS["vehicle"] = [(v, f"a photo of a {v}.") for v in VEHICLES]
VOCABS["age_group"] = AGE_GROUPS

INDOOR = {label: indoor for label, indoor in SCENES}
