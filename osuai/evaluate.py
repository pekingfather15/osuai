"""Score the models and the human replays on the held-out maps.

    python -m osuai.evaluate                                   # test split, V3 models
    python -m osuai.evaluate --split val --gaps none 24 36 36@0.6   # compare key decodings
"""

from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np

from . import beatmap as bm, decoding, features, judge, paths, replay as rp
from .models import KeyModel, load_cursor


def load_split(name: str) -> list[dict]:
    with open(os.path.join(paths.SPLITS, f"{name}.json")) as f:
        return json.load(f)["replays"]


def parse_decoding(spec: str):
    """'none' -> no merging, '36' -> merge presses closer than 36 ms, '36@0.6' -> closer than
    36 ms or 0.6 times the local note spacing, whichever is smaller."""
    if spec == "none":
        return None
    gap, _, ratio = spec.partition("@")
    return float(gap), float(ratio) if ratio else None


def decoding_name(d) -> str:
    if d is None:
        return "threshold"
    gap, ratio = d
    return f"merge {gap:g} ms" + (f" or {ratio:g}x spacing" if ratio else "")


def decode(probs: np.ndarray, d, beatmap) -> np.ndarray:
    flat = probs.reshape(-1, 2)
    if d is None:
        return decoding.threshold(flat)
    gap, ratio = d
    spacing = features.note_spacing(beatmap, len(flat)) if ratio else None
    return decoding.merge_presses(flat, gap / features.FRAME_MS, note_spacing=spacing, spacing_ratio=ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--cursor", default=os.path.join(paths.MODELS, "v3", "cursor_tcn_wgan.pt"))
    ap.add_argument("--keys", default=os.path.join(paths.MODELS, "v3", "keys_lstm.pt"))
    ap.add_argument("--gaps", nargs="*", default=["36"], help="press merging: 'none', a gap in ms ('36'), or a gap and a fraction of the note spacing ('36@0.6')")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", help="file with replay paths to restrict to (one per line)")
    args = ap.parse_args()

    cursor_model = load_cursor(args.cursor)
    key_model = KeyModel.load(args.keys)
    gaps = [parse_decoding(g) for g in args.gaps]
    cases = load_split(args.split)
    if args.only:
        with open(args.only) as f:
            keep = {line.strip() for line in f if line.strip()}
        cases = [c for c in cases if c["replay"] in keep]

    sources = ["human"] + [f"model, {decoding_name(g)}" for g in gaps]
    scores = {s: [] for s in sources}
    used = []
    for i, case in enumerate(cases):
        try:
            replay = rp.load(paths.data(case["replay"]))
            beatmap = bm.load(paths.data(case["map"]), replay.mods)
        except Exception as e:  # unreadable files are skipped, and reported
            print(f"skipping {case['replay']}: {e}")
            continue
        if not beatmap.objects:
            continue
        x = features.map_features(beatmap)
        y = features.replay_targets(beatmap, replay).reshape(-1, 4)
        scores["human"].append(judge.score(beatmap, features.to_playfield(y), y[:, 2:]))

        cursor = features.to_playfield(cursor_model.generate(x, seed=args.seed + i).reshape(-1, 2))
        probs = key_model.hold_probabilities(x)
        for g, name in zip(gaps, sources[1:]):
            scores[name].append(judge.score(beatmap, cursor, decode(probs, g, beatmap)))
        used.append(case["replay"])
        print(f"[{i + 1}/{len(cases)}] {beatmap.artist()} - {beatmap.title()} [{beatmap.version()}]")

    results = {name: judge.summarize(s) for name, s in scores.items()}
    print(f"\n{args.split}: {len(used)} plays")
    print(f"{'source':<36}{'full acc':>9}{'300/100/50/miss':>22}{'tails':>8}{'press/obj':>10}"
          f"{'relax':>9}{'hit rate':>9}{'mean px':>8}{'RPM':>6}")
    for name, r in results.items():
        k = r["keys"]
        print(f"{name:<36}{r['full_accuracy']:>9.2%}{'/'.join(str(k[j]) for j in ('300', '100', '50', '0')):>22}"
              f"{r['slider_tail_rate']:>8.1%}{r['presses_per_object']:>10.2f}{r['relax_accuracy']:>9.2%}"
              f"{r['hit_rate']:>9.1%}{r['mean_distance']:>8.1f}{r['spinner_rpm'] or 0:>6.0f}")

    os.makedirs(paths.RESULTS, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(paths.RESULTS, f"eval_{args.split}_{stamp}.json")
    with open(out, "w") as f:
        json.dump({"split": args.split, "cursor": args.cursor, "keys": args.keys, "seed": args.seed,
                   "plays": used, "results": results}, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
