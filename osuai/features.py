"""Turn a map into the model's input sequence, and a replay into its target sequence.

The timeline starts when the first object appears and has one frame every 12 ms. It is
cut into chunks of 2048 frames (24.6 s); the last chunk is padded.

Input features per frame describe the object the player should be dealing with:
the slider being followed, else the spinner being spun, else the next object to appear.
  x, y              where that object (or the slider ball) is, scaled to [-0.5, 0.5]
  time_until_click  seconds until the object's time, clipped to [0, 2]
  is_slider, is_spinner, is_note
  cs                circle size / 10
  slider_speed      osu!px per ms of the slider, at most 10 (0 for other objects)
  slider_len        slider length / 600 (0 for other objects)

These are the features the V3 models were trained with, so their quirks are kept:
frames with no object get cs 0.4 instead of the map's, and while a spinner is being
spun time_until_click is 0.1 * sin(ms since it started) / 1000 clipped at 0.
"""

from __future__ import annotations

import bisect
import math

import numpy as np

from . import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH
from .beatmap import Beatmap, Slider, Spinner
from .replay import Replay

FRAME_MS = 12
CHUNK_FRAMES = 2048
INPUT_FEATURES = ["x", "y", "time_until_click", "is_slider", "is_spinner", "is_note",
                  "cs", "slider_speed", "slider_len"]
N_INPUT = len(INPUT_FEATURES)
TARGET_FEATURES = ["x", "y", "k1", "k2"]

MAX_TIME_UNTIL_CLICK = 2.0
NO_OBJECT_CS = 0.4
MAX_SLIDER_SPEED = 10.0
# sliders shorter than this are played like circles
MIN_SLIDER_MS = 0.005
# padding after the end of the map
PAD_FRAME = np.array([0.0, 0.0, 9999.0, 0, 0, 0, NO_OBJECT_CS, 0, 0], dtype=np.float32)


def frame_times(beatmap: Beatmap) -> np.ndarray:
    """Real time of every frame, in ms."""
    return np.arange(beatmap.start_time, beatmap.end_time, FRAME_MS, dtype=np.int64)


def _to_chunks(frames: np.ndarray, pad: np.ndarray) -> np.ndarray:
    n_chunks = max(1, math.ceil(len(frames) / CHUNK_FRAMES))
    out = np.tile(pad, (n_chunks * CHUNK_FRAMES, 1)).astype(np.float32)
    out[:len(frames)] = frames
    return out.reshape(n_chunks, CHUNK_FRAMES, -1)


def map_features(beatmap: Beatmap) -> np.ndarray:
    """[chunks, 2048, 9] model input for the whole map."""
    times = frame_times(beatmap)
    objects = beatmap.objects
    starts = [o.time for o in objects]
    preempt = beatmap.preempt
    cs = beatmap.cs / 10.0

    def is_followed(o):
        return isinstance(o, Slider) and o.duration >= MIN_SLIDER_MS

    # objects that are held for a while: a slider being followed wins over a spinner
    held_groups = []
    for group in ([o for o in objects if is_followed(o)], [o for o in objects if isinstance(o, Spinner)]):
        group.sort(key=lambda o: o.time)
        held_groups.append((group, np.array([o.time for o in group]), np.array([o.end_time for o in group])))

    rows = np.empty((len(times), N_INPUT), dtype=np.float32)
    for n, t in enumerate(times):
        obj = None
        for group, g_start, g_end in held_groups:
            active = np.nonzero((g_start <= t) & (t <= g_end))[0]
            if len(active):
                obj = group[active[0]]  # the one that started first
                break
        if obj is None:
            i = bisect.bisect_left(starts, t)
            if i < len(objects) and objects[i].time <= t + preempt:
                obj = objects[i]

        if obj is None:
            rows[n] = (0.0, 0.0, MAX_TIME_UNTIL_CLICK, 0, 0, 0, NO_OBJECT_CS, 0, 0)
            continue

        followed = is_followed(obj)
        x, y = obj.position_at(t) if followed else (obj.x, obj.y)
        if isinstance(obj, Spinner) and t > obj.time:
            until = 0.1 * math.sin(t - obj.time)
        else:
            until = obj.time - t
        speed = length = 0.0
        if followed:
            speed = obj.pixel_length / obj.duration
            speed = MAX_SLIDER_SPEED if math.isnan(speed) else min(speed, MAX_SLIDER_SPEED)
            length = obj.pixel_length / 600.0
        rows[n] = (
            min(max(x / PLAYFIELD_WIDTH, 0.0), 1.0) - 0.5,
            min(max(y / PLAYFIELD_HEIGHT, 0.0), 1.0) - 0.5,
            min(max(until / 1000.0, 0.0), MAX_TIME_UNTIL_CLICK),
            float(followed), float(isinstance(obj, Spinner)),
            float(not followed and not isinstance(obj, Spinner)),
            cs, speed, length,
        )
    return _to_chunks(rows, PAD_FRAME)


def replay_targets(beatmap: Beatmap, replay: Replay) -> np.ndarray:
    """[chunks, 2048, 4] cursor (scaled to [-0.5, 0.5]) and key state of a replay."""
    states = replay.states_at(frame_times(beatmap).astype(np.float64))
    states[:, 0] = np.clip(states[:, 0] / PLAYFIELD_WIDTH, 0, 1) - 0.5
    states[:, 1] = np.clip(states[:, 1] / PLAYFIELD_HEIGHT, 0, 1) - 0.5
    return _to_chunks(states, np.zeros(4, dtype=np.float32))


def to_playfield(frames: np.ndarray) -> np.ndarray:
    """Model or target x, y in [-0.5, 0.5] -> osu!px, for [n, >=2] frames."""
    return np.stack([(frames[:, 0] + 0.5) * PLAYFIELD_WIDTH,
                     (frames[:, 1] + 0.5) * PLAYFIELD_HEIGHT], axis=1).astype(np.float64)
