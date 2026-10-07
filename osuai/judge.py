"""Score a play (a cursor path and key states on the frame timeline) the way the game would.

Scores computed for generated plays and for human replays alike:
  relax       each circle / slider head judged as if the game pressed the keys: the moment
              within its 50 window when the cursor is inside the circle and closest in time
  keys        judged with the key presses: every press of either key can hit one object, an
              object takes the earliest unused press in its 50 window while the cursor is
              inside, and |press time - object time| decides 300 / 100 / 50
  tails       like lazer, only the key that hit a slider head keeps tracking it: that key has
              to stay down, with the cursor in the follow circle, until 36 ms before the end
  spin        rotations around the playfield centre during spinners
  distance    cursor distance from each object at its time
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH
from .beatmap import Beatmap, Slider, Spinner
from .features import FRAME_MS

TAIL_LENIENCY_MS = 36
FOLLOW_RADIUS = 2.4  # follow circle, in circle radii
MIN_SPIN_RADIUS = 10  # cursor closer to the centre than this isn't spinning
CENTRE = np.array([PLAYFIELD_WIDTH / 2, PLAYFIELD_HEIGHT / 2])


@dataclass
class Score:
    relax: dict = field(default_factory=lambda: {300: 0, 100: 0, 50: 0, 0: 0})
    keys: dict | None = None
    tails_kept: int = 0
    tails: int = 0
    presses: int = 0
    distances: list = field(default_factory=list)
    inside: list = field(default_factory=list)  # cursor inside the circle at the object's time
    spin_turns: float = 0.0
    spin_minutes: float = 0.0


def accuracy(counts: dict) -> float:
    n = sum(counts.values())
    return (300 * counts[300] + 100 * counts[100] + 50 * counts[50]) / (300 * n) if n else 0.0


class Timeline:
    """Cursor positions (osu!px) and key states on the frame timeline of a map."""

    def __init__(self, beatmap: Beatmap, cursor: np.ndarray, keys: np.ndarray | None = None):
        self.start = beatmap.start_time
        self.cursor = cursor
        self.keys = None if keys is None else keys > 0.5
        self._index = np.arange(len(cursor))

    def frame_of(self, time: float) -> float:
        return (time - self.start) / FRAME_MS

    def cursor_at(self, times) -> np.ndarray:
        """Linearly interpolated cursor position at one or more times."""
        f = self.frame_of(np.atleast_1d(np.asarray(times, dtype=np.float64)))
        x = np.interp(f, self._index, self.cursor[:, 0])
        y = np.interp(f, self._index, self.cursor[:, 1])
        return np.stack([x, y], axis=1)

    def presses(self) -> list[tuple[float, int, int]]:
        """(time, frame, key) of every key going down, each key on its own."""
        down = self.keys
        before = np.vstack([np.zeros((1, 2), dtype=bool), down[:-1]])
        frames, which = np.nonzero(down & ~before)
        return [(self.start + f * FRAME_MS, int(f), int(k)) for f, k in zip(frames, which)]


def _windows(beatmap: Beatmap):
    w300, w100, w50 = beatmap.hit_windows()
    return w300, w100, w50


def _grade(dt: float, windows) -> int:
    w300, w100, _ = windows
    return 300 if dt <= w300 else 100 if dt <= w100 else 50


def score(beatmap: Beatmap, cursor: np.ndarray, keys: np.ndarray | None = None) -> Score:
    """cursor: [frames, 2] in osu!px, keys: [frames, 2] key states (0/1) or None."""
    tl = Timeline(beatmap, cursor, keys)
    radius = beatmap.circle_radius
    windows = _windows(beatmap)
    w50 = windows[2]
    s = Score()

    hit_objects = []
    for obj in beatmap.objects:
        if isinstance(obj, Spinner):
            a = max(int(tl.frame_of(obj.time)), 0)
            b = min(int(tl.frame_of(obj.end_time)), len(cursor))
            seg = cursor[a:b] - CENTRE
            if len(seg) >= 2:
                angles = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
                far = np.linalg.norm(seg, axis=1)[1:] > MIN_SPIN_RADIUS
                s.spin_turns += float(np.abs(np.diff(angles))[far].sum() / (2 * math.pi))
                s.spin_minutes += (obj.end_time - obj.time) / 60000
            continue
        hit_objects.append(obj)
        target = np.array([obj.x, obj.y])
        distance = float(np.linalg.norm(tl.cursor_at(obj.time)[0] - target))
        s.distances.append(distance)
        s.inside.append(distance <= radius)

        # relax: best moment inside the circle within the 50 window, every 2 ms
        offsets = np.arange(-w50, w50 + 1, 2.0)
        inside = np.linalg.norm(tl.cursor_at(obj.time + offsets) - target, axis=1) <= radius
        s.relax[_grade(np.abs(offsets[inside]).min(), windows) if inside.any() else 0] += 1

    if keys is None:
        return s

    presses = tl.presses()
    s.presses = len(presses)
    s.keys = {300: 0, 100: 0, 50: 0, 0: 0}
    next_press = 0
    for obj in hit_objects:
        while next_press < len(presses) and presses[next_press][0] < obj.time - w50:
            next_press += 1  # too early for this object, and so for every later one
        hit = None
        for q in range(next_press, len(presses)):
            t = presses[q][0]
            if t > obj.time + w50:
                break
            if np.linalg.norm(tl.cursor_at(t)[0] - (obj.x, obj.y)) <= radius:
                hit = presses[q]
                next_press = q + 1
                break
        s.keys[_grade(abs(hit[0] - obj.time), windows) if hit else 0] += 1

        if isinstance(obj, Slider):
            s.tails += 1
            if hit is not None and _tail_kept(tl, obj, hit, radius):
                s.tails_kept += 1
    return s


def _tail_kept(tl: Timeline, slider: Slider, hit, radius: float) -> bool:
    _, frame, key = hit
    tail_time = max(slider.end_time - TAIL_LENIENCY_MS, slider.time)
    tail_frame = int(tl.frame_of(tail_time))
    if not tl.keys[frame:tail_frame + 1, key].all():
        return False
    ball = np.array(slider.position_at(tail_time))
    return float(np.linalg.norm(tl.cursor_at(tail_time)[0] - ball)) <= FOLLOW_RADIUS * radius


def summarize(scores: list[Score]) -> dict:
    """Totals over many plays."""
    relax = {j: sum(s.relax[j] for s in scores) for j in (300, 100, 50, 0)}
    dist = np.concatenate([np.asarray(s.distances) for s in scores]) if scores else np.zeros(0)
    out = {
        "plays": len(scores),
        "relax_accuracy": accuracy(relax),
        "relax": {str(k): v for k, v in relax.items()},
        "hit_rate": float(np.mean(np.concatenate([np.asarray(s.inside) for s in scores]))) if scores else None,
        "mean_distance": float(dist.mean()) if len(dist) else None,
        "median_distance": float(np.median(dist)) if len(dist) else None,
    }
    minutes = sum(s.spin_minutes for s in scores)
    out["spinner_rpm"] = sum(s.spin_turns for s in scores) / minutes if minutes else None
    if scores and all(s.keys is not None for s in scores):
        keys = {j: sum(s.keys[j] for s in scores) for j in (300, 100, 50, 0)}
        tails = sum(s.tails for s in scores)
        objects = sum(sum(s.keys.values()) for s in scores)
        out.update({
            "full_accuracy": accuracy(keys),
            "keys": {str(k): v for k, v in keys.items()},
            "slider_tail_rate": sum(s.tails_kept for s in scores) / tails if tails else None,
            "presses_per_object": sum(s.presses for s in scores) / objects if objects else None,
        })
    return out
