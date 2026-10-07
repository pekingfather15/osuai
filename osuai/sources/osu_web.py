"""Download a player's replays from the osu! website.

    python -m osuai.sources.osu_web --user 7562902

Score lists are public, but downloading a replay needs a logged-in session: put your own
osu_session cookie in a .env file in the project folder as OSU_SESSION=... (the .env file
is ignored by git; never share it). Downloads are slowed down to respect the site.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import requests

from .. import paths, replay as rp
from .mirror import find_by_md5, mapset_folder

SITE = "https://osu.ppy.sh"
AGENT = "Mozilla/5.0 (osuai research downloader)"


def session_cookie() -> str:
    value = os.environ.get("OSU_SESSION")
    env_file = os.path.join(paths.ROOT, ".env")
    if not value and os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.strip().startswith("OSU_SESSION="):
                    value = line.split("=", 1)[1].strip().strip('"')
    if not value:
        raise SystemExit("set OSU_SESSION in .env to download replays")
    return value


def get(url: str, headers=None, tries: int = 5) -> requests.Response | None:
    for attempt in range(tries):
        response = requests.get(url, headers={"User-Agent": AGENT, **(headers or {})}, timeout=60)
        if response.status_code == 404:
            return None
        if response.ok:
            return response
        # rate limited or a passing error: wait longer each time
        time.sleep(300 if response.status_code == 429 else 4 * (attempt + 1))
    return None


def scores(user: int, kind: str, limit: int):
    for offset in range(0, limit, 100):
        response = get(f"{SITE}/users/{user}/scores/{kind}?mode=osu&limit=100&offset={offset}")
        page = response.json() if response else []
        if not page:
            return
        yield from page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--kinds", nargs="*", default=["best", "firsts", "recent"])
    ap.add_argument("--limit", type=int, default=1000, help="scores to look at per kind")
    ap.add_argument("--only-s", action="store_true", help="only S ranks and better")
    args = ap.parse_args()

    cookie = session_cookie()
    folder = paths.data(f"replays-{args.user}")
    os.makedirs(folder, exist_ok=True)
    saved = 0
    for kind in args.kinds:
        for score in scores(args.user, kind, args.limit):
            if not score.get("has_replay") or (args.only_s and not score["rank"].startswith("S")):
                continue
            score_id = score["id"]
            meta = os.path.join(folder, f"{score_id}.meta")
            if os.path.exists(meta):
                continue
            map_md5, set_id = score["beatmap"]["checksum"], score["beatmap"]["beatmapset_id"]
            set_folder = mapset_folder(set_id)
            map_path = find_by_md5(set_folder, map_md5) if set_folder else None
            if map_path is None:
                continue
            time.sleep(1.5)
            response = get(f"{SITE}/scores/{score_id}/download",
                           {"Cookie": f"osu_session={cookie}", "Referer": f"{SITE}/scores/{score_id}"})
            if response is None:
                continue
            if rp.parse(response.content).beatmap_md5 != map_md5:
                continue
            with open(os.path.join(folder, f"{score_id}.osr"), "wb") as f:
                f.write(response.content)
            with open(meta, "w") as f:
                json.dump({"map": map_path}, f)
            saved += 1
            print(f"saved {score_id} ({kind}), {saved} so far")
    print(f"done: {saved} new replays in {folder}")


if __name__ == "__main__":
    main()
