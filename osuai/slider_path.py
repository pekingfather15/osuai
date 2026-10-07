"""Slider paths: the curve a slider ball follows, from the control points in the map.

Each curve type is sampled into a polyline, the polyline is cut (or extended in a
straight line) to the slider's length from the map, and positions are looked up by
distance along it. This follows the curve types of the .osu format:
  L  straight segments through the points
  P  an arc of the circle through three points
  B  Bezier curves, a new one starting at every repeated control point
  C  Catmull-Rom spline through the points
"""

import math

import numpy as np

# points per curve piece; slider lengths are a few hundred osu!px at most
_SAMPLES = 64


def _bezier(points: np.ndarray) -> np.ndarray:
    # evaluate with de Casteljau's algorithm at evenly spaced parameters
    t = np.linspace(0.0, 1.0, max(_SAMPLES, 8 * len(points)))[:, None, None]
    pts = np.broadcast_to(points, (t.shape[0],) + points.shape).copy()
    for _ in range(len(points) - 1):
        pts = pts[:, :-1] * (1 - t) + pts[:, 1:] * t
    return pts[:, 0]


def _bezier_path(points: np.ndarray) -> np.ndarray:
    pieces = []
    start = 0
    for i in range(1, len(points)):
        # a repeated point ends one Bezier curve and starts the next
        if i == len(points) - 1 or np.array_equal(points[i], points[i + 1]):
            piece = points[start:i + 1]
            if len(piece) > 1:
                pieces.append(_bezier(piece))
            start = i + 1 if i < len(points) - 1 else i
    return np.concatenate(pieces) if pieces else points.copy()


def _arc_path(points: np.ndarray) -> np.ndarray | None:
    a, b, c = points
    # circle through three points: solve for the centre
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-6:
        return None  # the points are on a line
    sq = (a ** 2).sum(), (b ** 2).sum(), (c ** 2).sum()
    centre = np.array([
        (sq[0] * (b[1] - c[1]) + sq[1] * (c[1] - a[1]) + sq[2] * (a[1] - b[1])) / d,
        (sq[0] * (c[0] - b[0]) + sq[1] * (a[0] - c[0]) + sq[2] * (b[0] - a[0])) / d,
    ])
    radius = np.linalg.norm(a - centre)
    start = math.atan2(a[1] - centre[1], a[0] - centre[0])
    end = math.atan2(c[1] - centre[1], c[0] - centre[0])
    # go around the way that passes through b: b is on the left of a->c for one direction
    clockwise = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0
    if clockwise:
        while end > start:
            end -= 2 * math.pi
    else:
        while end < start:
            end += 2 * math.pi
    n = max(_SAMPLES, int(abs(end - start) * radius / 4))
    angles = np.linspace(start, end, n)
    return centre + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _catmull_path(points: np.ndarray) -> np.ndarray:
    padded = np.concatenate([points[:1], points, points[-1:]])
    t = np.linspace(0.0, 1.0, _SAMPLES)[:, None]
    pieces = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1:i + 3]
        pieces.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                             + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    return np.concatenate(pieces)


def _linear_path(points: np.ndarray) -> np.ndarray:
    return points.copy()


class SliderPath:
    def __init__(self, curve_type: str, points: list[tuple[float, float]], length: float):
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 2:
            polyline = np.repeat(pts, 2, axis=0)
        elif curve_type == "P" and len(pts) == 3:
            polyline = _arc_path(pts)
            if polyline is None:
                polyline = _linear_path(pts)
        elif curve_type == "L":
            polyline = _linear_path(pts)
        elif curve_type == "C":
            polyline = _catmull_path(pts)
        else:  # B, and P with more than three points
            polyline = _bezier_path(pts)

        self.length = float(length)
        self.points, self.distances = self._fit_length(polyline, self.length)

    @staticmethod
    def _fit_length(polyline: np.ndarray, length: float):
        seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        dist = np.concatenate([[0.0], np.cumsum(seg)])
        if length <= 0:
            return polyline[:1].repeat(2, axis=0), np.array([0.0, 0.0])
        if dist[-1] >= length:
            # cut the curve where it reaches the slider's length
            end = np.searchsorted(dist, length)
            frac = (length - dist[end - 1]) / max(seg[end - 1], 1e-9)
            last = polyline[end - 1] + (polyline[end] - polyline[end - 1]) * frac
            return np.vstack([polyline[:end], last]), np.concatenate([dist[:end], [length]])
        # too short: continue straight in the direction of the last segment
        moving = np.nonzero(seg > 1e-9)[0]
        if len(moving) == 0:
            return np.vstack([polyline, polyline[-1:]]), np.concatenate([dist, [length]])
        i = moving[-1]
        direction = (polyline[i + 1] - polyline[i]) / seg[i]
        last = polyline[-1] + direction * (length - dist[-1])
        return np.vstack([polyline, last]), np.concatenate([dist, [length]])

    def position(self, distance: float) -> tuple[float, float]:
        """Point at this distance from the slider head along the curve."""
        d = min(max(distance, 0.0), self.distances[-1])
        i = int(np.searchsorted(self.distances, d, side="right")) - 1
        i = min(max(i, 0), len(self.points) - 2)
        span = self.distances[i + 1] - self.distances[i]
        frac = 0.0 if span <= 0 else (d - self.distances[i]) / span
        p = self.points[i] + (self.points[i + 1] - self.points[i]) * frac
        return float(p[0]), float(p[1])
