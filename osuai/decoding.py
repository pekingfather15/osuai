"""Turn per-frame key hold probabilities into key presses.

Thresholding each key on its own often presses both keys within a frame or two, because
the model can't tell which key a player would use. In the game the extra press hits the
next object early, or reaches a slider head first and lets go of it. So presses that start
closer together than `min_gap_frames` become one press, held until the last of them is
released, and every press gets a key that is not already held.
"""

import numpy as np


def _runs(held: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every run of True."""
    edges = np.diff(np.concatenate([[0], held.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def threshold(probs: np.ndarray, level: float = 0.5) -> np.ndarray:
    return (probs > level).astype(np.float32)


def merge_presses(probs: np.ndarray, min_gap_frames: int, level: float = 0.5) -> np.ndarray:
    """probs: [frames, 2]. Returns [frames, 2] key states (0/1)."""
    held = probs > level
    presses = sorted([(s, e, k) for k in (0, 1) for s, e in _runs(held[:, k])])

    merged: list[list[int]] = []  # [start, end, key]
    for start, end, key in presses:
        if merged and start - merged[-1][0] < min_gap_frames:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, key])

    latest: list[list[int] | None] = [None, None]  # the most recent press on each key
    for press in merged:
        start, key = press[0], press[2]

        def busy(k):
            return latest[k] is not None and latest[k][1] > start

        if busy(key):
            if not busy(1 - key):
                key = 1 - key
            else:
                key = 0 if latest[0][0] < latest[1][0] else 1  # re-press the key held longer
        previous = latest[key]
        if previous is not None and previous[1] >= start:
            previous[1] = start - 1  # one frame up so the game sees a new press
        press[2] = key
        latest[key] = press

    out = np.zeros((len(probs), 2), dtype=np.float32)
    for start, end, key in merged:
        if end > start:
            out[start:end, key] = 1
    return out
