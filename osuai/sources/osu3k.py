"""Import the strongest plays of the osu!3k Kaggle dataset (MIT licence).

    pip install kaggle   # or: uv run --with kaggle ...
    python -m osuai.sources.osu3k --count 1000

https://www.kaggle.com/datasets/ilymeow/osu3k holds ~83 GB, almost all of it rendered
videos. Only the replay files and the scores table are downloaded. Needs your own Kaggle
API token (see Kaggle's documentation). The dataset's `mapid` column is the beatmapset id.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile

import pandas as pd

from .. import paths, replay as rp
from .mirror import find_by_md5, mapset_folder

DATASET = "ilymeow/osu3k"
REPLAYS = "replays-osu3k"
MAPS = "osu3k/maps"
CSV = "osu3k/data.csv"

# Relax, Autopilot and Spun Out change how the game is played, and Easy and Half Time
# are rare enough to leave out
ALLOWED_MODS = ["SV2", "NM", "NF", "HD", "HR", "DT", "NC", "SD", "PF", "FL", "TD"]


def parse_mods(mods) -> list[str] | None:
    text = "NM" if pd.isna(mods) else mods
    found = []
    while text:
        for mod in ALLOWED_MODS:
            if text.startswith(mod):
                found.append(mod)
                text = text[len(mod):]
                break
        else:
            return None
    return found


def candidates(min_acc: float, min_star: float) -> pd.DataFrame:
    """Usable plays, strongest players first."""
    df = pd.read_csv(paths.data(CSV))
    df = df[df["mods"].apply(parse_mods).notna() & (df["player"] != "Guest")]
    df = df[(df["map_acc"] >= min_acc) & (df["star"] >= min_star)]
    # some plays were rendered twice; if the copies disagree on the mods, one was edited
    # and there's no telling which, so both go
    play = ["player", "mapid", "map_acc"]
    conflicting = df.groupby(play)["mods"].transform("nunique") > 1
    return df[~conflicting].drop_duplicates(subset=play).sort_values("pp", ascending=False)


def kaggle_file(api, remote: str, local_dir: str) -> str:
    api.dataset_download_file(DATASET, remote, path=local_dir, quiet=True)
    local = os.path.join(local_dir, os.path.basename(remote))
    if not os.path.exists(local) and os.path.exists(local + ".zip"):  # sometimes zipped
        with zipfile.ZipFile(local + ".zip") as z:
            z.extractall(local_dir)
        os.remove(local + ".zip")
    return local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--min-acc", type=float, default=93.0)
    ap.add_argument("--min-star", type=float, default=5.0)
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(paths.data(CSV)):
        os.makedirs(os.path.dirname(paths.data(CSV)), exist_ok=True)
        kaggle_file(api, "dataset/data.csv", os.path.dirname(paths.data(CSV)))
    plays = candidates(args.min_acc, args.min_star)
    print(f"{len(plays)} usable plays, importing the strongest {args.count}")

    replay_dir = paths.data(REPLAYS)
    os.makedirs(replay_dir, exist_ok=True)
    imported, skipped, ids = 0, {}, []
    for tried, row in enumerate(plays.itertuples(), 1):
        if imported >= args.count:
            break
        render_id, set_id = int(row.renderid), int(row.mapid)
        osr = os.path.join(replay_dir, f"{render_id}.osr")
        meta = os.path.join(replay_dir, f"{render_id}.meta")
        if os.path.exists(meta):
            imported += 1
            ids.append(render_id)
            continue
        try:
            if not os.path.exists(osr):
                kaggle_file(api, f"dataset/replays/{render_id}.osr", replay_dir)
            replay = rp.load(osr)
            folder = mapset_folder(set_id, MAPS)
            map_path = find_by_md5(folder, replay.beatmap_md5) if folder else None
            if map_path is None:
                # unavailable, or the map was updated after the play and no longer fits
                reason = "map unavailable" if folder is None else "map changed since the play"
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            with open(meta, "w") as f:
                json.dump({"map": map_path}, f)
            imported += 1
            ids.append(render_id)
        except Exception as e:
            skipped[type(e).__name__] = skipped.get(type(e).__name__, 0) + 1
        if tried % 25 == 0:
            print(f"[{tried} tried] imported {imported}, skipped {skipped}")

    selection = plays[plays["renderid"].isin(ids)]
    os.makedirs(paths.RESULTS, exist_ok=True)
    selection.to_csv(os.path.join(paths.RESULTS, "osu3k_selection.csv"), index=False)
    print(f"imported {imported} plays from {selection['player'].nunique()} players, skipped {skipped}")


if __name__ == "__main__":
    main()
