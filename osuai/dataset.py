"""Build a training set from every replay in the data folder, leaving out the maps of the
test and validation splits.

    python -m osuai.dataset --name v3

Replays live in data/replays-<source>/<id>.osr, each with a <id>.meta JSON file naming
its map: {"map": "songs/<set> <artist> - <title>/<file>.osu"} (paths relative to data/).
Replays renamed to .meta.excluded are skipped. The output is datasets/<name>/ with
xs_<k>.npy ([chunks, 2048, 9] inputs) and ys_<k>.npy ([chunks, 2048, 4] targets).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np

from . import beatmap as bm, features, paths, replay as rp


def all_replays() -> list[tuple[str, str]]:
    """(map, replay) paths relative to the data folder."""
    pairs = []
    for meta in sorted(glob.glob(os.path.join(paths.DATA, "replays-*", "*.meta"))):
        replay = meta[:-len(".meta")] + ".osr"
        with open(meta) as f:
            map_path = json.load(f)["map"].replace("\\", "/").removeprefix(".data/")
        if os.path.exists(replay) and os.path.exists(paths.data(map_path)):
            pairs.append((map_path, os.path.relpath(replay, paths.DATA).replace("\\", "/")))
    return pairs


def file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def held_out_maps(splits=("test", "val")) -> set[str]:
    """MD5 of every held-out map file. The same map can be stored in different folders, so
    the file contents are compared, not the paths."""
    hashes = set()
    for name in splits:
        with open(os.path.join(paths.SPLITS, f"{name}.json")) as f:
            hashes |= {file_md5(paths.data(r["map"])) for r in json.load(f)["replays"]}
    return hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="output folder name under datasets/")
    ap.add_argument("--shard-size", type=int, default=300, help="replays per file, to limit memory use")
    ap.add_argument("--limit", type=int, help="only use this many replays (for quick tests)")
    args = ap.parse_args()

    held_out = held_out_maps()
    md5_cache: dict[str, str] = {}
    pairs = []
    for map_path, replay_path in all_replays():
        if map_path not in md5_cache:
            md5_cache[map_path] = file_md5(paths.data(map_path))
        if md5_cache[map_path] not in held_out:
            pairs.append((map_path, replay_path))
    pairs = pairs[:args.limit] if args.limit else pairs
    print(f"{len(pairs)} training replays")

    out_dir = os.path.join(paths.ROOT, "datasets", args.name)
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for k in range(0, len(pairs), args.shard_size):
        xs, ys = [], []
        for map_path, replay_path in pairs[k:k + args.shard_size]:
            try:
                replay = rp.load(paths.data(replay_path))
                beatmap = bm.load(paths.data(map_path), replay.mods)
                if not beatmap.objects:
                    continue
                xs.append(features.map_features(beatmap))
                ys.append(features.replay_targets(beatmap, replay))
            except Exception as e:
                print(f"skipping {replay_path}: {e}")
        shard = k // args.shard_size
        np.save(os.path.join(out_dir, f"xs_{shard}.npy"), np.concatenate(xs))
        np.save(os.path.join(out_dir, f"ys_{shard}.npy"), np.concatenate(ys))
        total += len(xs)
        print(f"shard {shard}: {len(xs)} replays, {sum(len(x) for x in xs)} chunks")
    print(f"done: {total} replays in {out_dir}")


if __name__ == "__main__":
    main()
