"""Where data and models live. Set OSUAI_DATA to keep the data somewhere else."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("OSUAI_DATA", os.path.join(ROOT, "data"))
MODELS = os.path.join(ROOT, "models")
SPLITS = os.path.join(ROOT, "splits")
RESULTS = os.path.join(ROOT, "results")


def data(relative: str) -> str:
    return os.path.join(DATA, relative)
