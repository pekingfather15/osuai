"""Generate a play for a map: cursor positions and key states on the frame timeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import beatmap as bm, decoding, features
from ..models import KeyModel


@dataclass
class Play:
    md5: str
    mods: int
    rate: float
    start_time: int        # real ms of frame 0
    cursor: np.ndarray     # [frames, 2] osu!px
    keys: np.ndarray       # [frames, 2] 0/1, all 0 for cursor-only plays
    cursor_only: bool


def generate(map_path: str, md5: str, mods: int, cursor_model, key_model: KeyModel | None,
             merge_gap_ms: float | None = 36, cursor_only: bool = False, seed: int | None = None,
             spacing_ratio: float | None = 0.4) -> Play:
    """Presses closer than merge_gap_ms, or than spacing_ratio times the local note spacing
    (whichever is smaller, chosen on the validation set), are merged into one."""
    beatmap = bm.load(map_path, mods)
    if not beatmap.objects:
        raise ValueError("the map has no hit objects")
    x = features.map_features(beatmap)
    n = len(features.frame_times(beatmap))
    cursor = features.to_playfield(cursor_model.generate(x, seed=seed).reshape(-1, 2))[:n]
    if cursor_only or key_model is None:
        keys = np.zeros((n, 2), dtype=np.float32)
    else:
        probs = key_model.hold_probabilities(x).reshape(-1, 2)
        if merge_gap_ms:
            spacing = features.note_spacing(beatmap, len(probs)) if spacing_ratio else None
            keys = decoding.merge_presses(probs, merge_gap_ms / features.FRAME_MS,
                                          note_spacing=spacing, spacing_ratio=spacing_ratio)
        else:
            keys = decoding.threshold(probs)
        keys = keys[:n]
    return Play(md5, beatmap.mods, beatmap.rate, beatmap.start_time, cursor, keys, cursor_only or key_model is None)
