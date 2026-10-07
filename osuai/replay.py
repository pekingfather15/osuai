"""Read .osr replay files (osu!standard).

The cursor and key state is a step function of time: each recorded frame holds
until the next one. Times are in real milliseconds, like Beatmap times.
"""

from __future__ import annotations

import io
import lzma
import struct
from dataclasses import dataclass

import numpy as np

from . import mods as mod_flags

# key bits in a replay frame: mouse buttons and the two keyboard keys
M1, M2, K1, K2 = 1, 2, 4, 8
# the last frame of a replay stores the RNG seed, not a position
_SEED_FRAME = -12345


def _uleb128(f) -> int:
    value, shift = 0, 0
    while True:
        byte = f.read(1)[0]
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value
        shift += 7


def _string(f) -> str:
    marker = f.read(1)[0]
    if marker == 0x00:
        return ""
    if marker != 0x0B:
        raise ValueError("not a replay file")
    return f.read(_uleb128(f)).decode("utf-8")


@dataclass
class Replay:
    mode: int
    version: int
    beatmap_md5: str
    player: str
    counts: tuple[int, int, int, int]  # 300s, 100s, 50s, misses
    max_combo: int
    mods: int
    times: np.ndarray   # real ms, sorted
    xs: np.ndarray      # osu!px
    ys: np.ndarray
    keys: np.ndarray    # key bits

    @property
    def accuracy(self) -> float:
        n300, n100, n50, miss = self.counts
        total = n300 + n100 + n50 + miss
        return (300 * n300 + 100 * n100 + 50 * n50) / (300 * total) if total else 0.0

    def state_at(self, time: float) -> tuple[float, float, bool, bool]:
        """Cursor position and whether key 1 / key 2 is down at this real time."""
        i = int(np.searchsorted(self.times, time, side="right")) - 1
        if i < 0:
            return 0.0, 0.0, False, False
        k = int(self.keys[i])
        return float(self.xs[i]), float(self.ys[i]), bool(k & (K1 | M1)), bool(k & (K2 | M2))

    def states_at(self, times: np.ndarray) -> np.ndarray:
        """state_at for many times at once: [n, 4] of x, y, key 1, key 2."""
        i = np.searchsorted(self.times, times, side="right") - 1
        out = np.zeros((len(times), 4), dtype=np.float64)
        ok = i >= 0
        j = i[ok]
        out[ok, 0] = self.xs[j]
        out[ok, 1] = self.ys[j]
        out[ok, 2] = (self.keys[j] & (K1 | M1)) != 0
        out[ok, 3] = (self.keys[j] & (K2 | M2)) != 0
        return out


def parse(data: bytes) -> Replay:
    f = io.BytesIO(data)
    mode, version = struct.unpack("<bi", f.read(5))
    if mode != 0:
        raise ValueError("not an osu!standard replay")
    beatmap_md5 = _string(f)
    player = _string(f)
    _string(f)  # replay md5
    n300, n100, n50, _geki, _katu, miss = struct.unpack("<6H", f.read(12))
    _score, max_combo, _perfect, mods = struct.unpack("<iHBi", f.read(11))
    _string(f)  # life bar graph
    _timestamp, length = struct.unpack("<qi", f.read(12))
    frames_text = lzma.decompress(f.read(length)).decode("ascii")

    rate = mod_flags.rate(mods)
    times, xs, ys, keys = [], [], [], []
    clock = 0
    for frame in frames_text.split(","):
        if not frame:
            continue
        w, x, y, k = frame.split("|")
        w = int(w)
        if w == _SEED_FRAME:
            continue
        clock += w
        times.append(clock / rate)
        xs.append(float(x))
        ys.append(float(y))
        keys.append(int(k))

    order = np.argsort(np.asarray(times), kind="stable")
    return Replay(mode, version, beatmap_md5, player, (n300, n100, n50, miss), max_combo, mods,
                  np.asarray(times, dtype=np.float64)[order], np.asarray(xs)[order],
                  np.asarray(ys)[order], np.asarray(keys, dtype=np.int64)[order])


def load(path: str) -> Replay:
    with open(path, "rb") as f:
        return parse(f.read())
