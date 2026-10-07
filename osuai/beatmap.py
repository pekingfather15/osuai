"""Read .osu beatmap files (osu!standard).

A Beatmap is loaded once per set of gameplay mods. Every time it exposes is in real
milliseconds, so DT and HT maps are already sped up or slowed down, and Hard Rock
maps are already flipped.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from . import PLAYFIELD_HEIGHT, mods as mod_flags
from .slider_path import SliderPath


@dataclass
class TimingPoint:
    time: float        # ms, song time
    beat_length: float  # ms per beat if uninherited, else -100 / slider velocity multiplier
    uninherited: bool


@dataclass
class HitObject:
    x: float
    y: float
    time: float  # real ms

    @property
    def end_time(self) -> float:
        return self.time


@dataclass
class Circle(HitObject):
    pass


@dataclass
class Slider(HitObject):
    curve_type: str = "B"
    control_points: list = field(default_factory=list)  # including the head
    repeats: int = 1  # number of times the ball crosses the path
    pixel_length: float = 0.0
    duration: float = 0.0  # real ms, filled in by the Beatmap
    _path: SliderPath | None = None

    @property
    def end_time(self) -> float:
        return self.time + self.duration

    @property
    def path(self) -> SliderPath:
        if self._path is None:
            self._path = SliderPath(self.curve_type, self.control_points, self.pixel_length)
        return self._path

    def position_at(self, time: float) -> tuple[float, float]:
        """Where the slider ball is at this real time."""
        if time <= self.time or self.duration <= 0:
            return self.x, self.y
        progress = min((time - self.time) / self.duration, 1.0) * self.repeats
        span = min(int(progress), self.repeats - 1)
        along = progress - span
        if span % 2 == 1:
            along = 1.0 - along  # going back
        return self.path.position(along * self.pixel_length)


@dataclass
class Spinner(HitObject):
    spin_end: float = 0.0  # real ms

    @property
    def end_time(self) -> float:
        return self.spin_end


def _sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def _key_values(lines: list[str]) -> dict[str, str]:
    values = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


class Beatmap:
    def __init__(self, text: str, mods: int = 0):
        sections = _sections(text.lstrip("﻿"))
        self.metadata = _key_values(sections.get("Metadata", []))
        self.general = _key_values(sections.get("General", []))
        difficulty = _key_values(sections.get("Difficulty", []))

        self.mods = mods & mod_flags.GAMEPLAY
        self.rate = mod_flags.rate(self.mods)

        od = float(difficulty.get("OverallDifficulty", 5))
        cs = float(difficulty.get("CircleSize", 5))
        # maps from before approach rate existed use the overall difficulty for it
        ar = float(difficulty.get("ApproachRate", od))
        self.base_od, self.base_cs, self.base_ar = od, cs, ar
        self.slider_multiplier = float(difficulty.get("SliderMultiplier", 1.4))

        if self.mods & mod_flags.HARD_ROCK:
            od, cs, ar = min(od * 1.4, 10), min(cs * 1.3, 10), min(ar * 1.4, 10)
        elif self.mods & mod_flags.EASY:
            od, cs, ar = od / 2, cs / 2, ar / 2
        self.od, self.cs, self.ar = od, cs, ar

        self.timing_points = self._read_timing(sections.get("TimingPoints", []))
        self._timing_times = [tp.time for tp in self.timing_points]
        self.objects = self._read_objects(sections.get("HitObjects", []))

    # ---- metadata ----
    def title(self) -> str:
        return self.metadata.get("Title", "")

    def artist(self) -> str:
        return self.metadata.get("Artist", "")

    def version(self) -> str:
        return self.metadata.get("Version", "")

    # ---- difficulty, in real time ----
    @property
    def preempt(self) -> float:
        """How long before its time an object appears (ms, real time)."""
        ar = self.ar
        song_ms = 1200 + 600 * (5 - ar) / 5 if ar <= 5 else 1200 - 750 * (ar - 5) / 5
        return song_ms / self.rate

    @property
    def circle_radius(self) -> float:
        return 54.4 - 4.48 * self.cs

    def hit_windows(self) -> tuple[float, float, float]:
        """Largest |press time - object time| for a 300, 100 and 50, in real ms."""
        od = self.od
        return tuple(w / self.rate for w in (80 - 6 * od, 140 - 8 * od, 200 - 10 * od))

    @property
    def start_time(self) -> int:
        """When the first object appears: where the frame timeline starts."""
        return int(self.objects[0].time - self.preempt) if self.objects else 0

    @property
    def end_time(self) -> int:
        return int(self.objects[-1].end_time) if self.objects else 0

    # ---- timing ----
    def beat_length(self, song_time: float) -> float:
        """Milliseconds per beat at this song time, scaled by the slider velocity, so
        that slider duration = beat_length * pixel_length / (100 * multiplier)."""
        i = bisect.bisect_right(self._timing_times, song_time) - 1
        current = self.timing_points[max(i, 0)]
        base = None
        for tp in reversed(self.timing_points[:max(i, 0) + 1]):
            if tp.uninherited:
                base = tp.beat_length
                break
        if base is None:
            base = next((tp.beat_length for tp in self.timing_points if tp.uninherited), 500.0)
        if current.uninherited:
            return base
        return base * min(max(-current.beat_length, 10.0), 1000.0) / 100

    @staticmethod
    def _read_timing(lines: list[str]) -> list[TimingPoint]:
        points = []
        for line in lines:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            beat_length = float(parts[1])
            uninherited = bool(int(parts[6])) if len(parts) > 6 else beat_length > 0
            # a negative beat length always means a slider velocity change
            points.append(TimingPoint(float(parts[0]), beat_length, uninherited and beat_length > 0))
        points.sort(key=lambda tp: tp.time)
        return points or [TimingPoint(0.0, 500.0, True)]

    def _read_objects(self, lines: list[str]) -> list[HitObject]:
        flip = bool(self.mods & mod_flags.HARD_ROCK)
        objects: list[HitObject] = []
        for line in lines:
            parts = line.split(",")
            x, y, song_time, kind = float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
            if flip:
                y = PLAYFIELD_HEIGHT - y
            time = song_time / self.rate
            if kind & 2:
                curve = parts[5].split("|")
                points = [(x, y)]
                for p in curve[1:]:
                    px, py = p.split(":")
                    points.append((float(px), PLAYFIELD_HEIGHT - float(py) if flip else float(py)))
                slider = Slider(x, y, time, curve_type=curve[0], control_points=points,
                                repeats=max(int(parts[6]), 1), pixel_length=float(parts[7]))
                beat = self.beat_length(song_time)
                song_duration = beat * slider.pixel_length / (100 * self.slider_multiplier) * slider.repeats
                slider.duration = song_duration / self.rate
                objects.append(slider)
            elif kind & 8:
                objects.append(Spinner(x, y, time, spin_end=float(parts[5]) / self.rate))
            elif kind & 1:
                objects.append(Circle(x, y, time))
            # osu!mania hold notes (kind & 128) don't exist in osu!standard
        objects.sort(key=lambda o: o.time)
        return objects


def load(path: str, mods: int = 0) -> Beatmap:
    with open(path, encoding="utf-8", errors="replace") as f:
        return Beatmap(f.read(), mods)
