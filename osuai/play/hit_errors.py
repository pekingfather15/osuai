"""Record how early or late every hit of one play was, to calibrate the playback offset.

    python -m osuai.play.hit_errors

Start it before the play, then play one map. tosu's precise endpoint lists the timing
error of every hit so far (positive = late, in ms). When the play ends the errors are
saved to results/ with their mean, median and spread. If the median is m ms late, raise
the offset by about m.
"""

from __future__ import annotations

import datetime
import json
import os
import statistics
import time
import urllib.request

from .. import paths
from .tosu import HOST, PLAYING


def fetch(path: str) -> dict:
    with urllib.request.urlopen(f"http://{HOST}{path}", timeout=2) as response:
        return json.load(response)


def main():
    print("waiting for a play to start...")
    errors: list = []
    beatmap = None
    while True:
        try:
            state, precise = fetch("/json/v2"), fetch("/json/v2/precise")
        except Exception:
            time.sleep(0.5)
            continue
        if state.get("state", {}).get("number") == PLAYING:
            if beatmap is None:
                beatmap = state.get("beatmap", {})
                print(f"recording {beatmap.get('artist')} - {beatmap.get('title')} [{beatmap.get('version')}]")
            current = precise.get("hitErrors") or []
            # the list starts over on a retry: keep the latest attempt
            if len(current) >= len(errors) or len(current) < len(errors) // 2:
                errors = list(current)
        elif beatmap is not None:
            break
        time.sleep(0.1)

    if not errors:
        print("no hits were recorded")
        return
    sd = statistics.pstdev(errors)
    summary = {
        "recorded": datetime.datetime.now().isoformat(timespec="seconds"),
        "beatmap": {k: beatmap.get(k) for k in ("artist", "title", "version", "checksum")},
        "hits": len(errors),
        "mean_ms": statistics.mean(errors),
        "median_ms": statistics.median(errors),
        "stdev_ms": sd,
        "unstable_rate": sd * 10,
        "early_share": sum(e < 0 for e in errors) / len(errors),
        "errors_ms": errors,
    }
    os.makedirs(paths.RESULTS, exist_ok=True)
    out = os.path.join(paths.RESULTS, f"hit_errors_{datetime.datetime.now():%m-%d_%H-%M-%S}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{len(errors)} hits: median {summary['median_ms']:+.1f} ms, UR {sd * 10:.0f}, "
          f"{summary['early_share']:.0%} early. saved {out}")


if __name__ == "__main__":
    main()
