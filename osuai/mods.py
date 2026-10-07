"""The osu!stable mod bit flags that change how a map plays."""

NO_FAIL = 1
EASY = 2
HIDDEN = 8
HARD_ROCK = 16
SUDDEN_DEATH = 32
DOUBLE_TIME = 64
RELAX = 128
HALF_TIME = 256
NIGHTCORE = 512  # always set together with DOUBLE_TIME
FLASHLIGHT = 1024
AUTOPLAY = 2048
SPUN_OUT = 4096
AUTOPILOT = 8192
PERFECT = 16384

# mods that change object positions, sizes, windows or timing
GAMEPLAY = EASY | HARD_ROCK | DOUBLE_TIME | HALF_TIME

_NAMES = {
    "nf": NO_FAIL, "ez": EASY, "hd": HIDDEN, "hr": HARD_ROCK, "sd": SUDDEN_DEATH,
    "dt": DOUBLE_TIME, "rx": RELAX, "ht": HALF_TIME, "nc": NIGHTCORE | DOUBLE_TIME,
    "fl": FLASHLIGHT, "so": SPUN_OUT, "ap": AUTOPILOT, "pf": PERFECT,
}


def rate(mods: int) -> float:
    """How much faster than normal the song plays."""
    if mods & DOUBLE_TIME:
        return 1.5
    if mods & HALF_TIME:
        return 0.75
    return 1.0


def parse(text: str) -> int:
    """'hdhr' or 'HD,HR' -> bit flags."""
    text = text.lower().replace(",", "").replace("+", "").replace(" ", "")
    if len(text) % 2:
        raise ValueError(f"can't read mods {text!r}")
    flags = 0
    for i in range(0, len(text), 2):
        flags |= _NAMES[text[i:i + 2]]
    return flags
