"""Download the .osu files of a beatmapset from a mirror."""

from __future__ import annotations

import hashlib
import io
import os
import time
import zipfile

import requests

from .. import paths

MAPS = "maps"  # data/maps/<beatmapset id>/*.osu

# sayobot's "mini" download has no video or audio, beatconnect is the fallback
MIRRORS = (
    "https://dl.sayobot.cn/beatmaps/download/mini/{}",
    "https://txy1.sayobot.cn/beatmaps/download/mini/{}",
    "https://beatconnect.io/b/{}/",
)


def mapset_folder(set_id: int, subfolder: str = MAPS) -> str | None:
    """Folder with the set's .osu files, downloading them if needed. None if unavailable."""
    folder = paths.data(f"{subfolder}/{set_id}")
    if os.path.isdir(folder) and os.listdir(folder):
        return folder
    for attempt in range(5):
        try:
            response = requests.get(MIRRORS[attempt % len(MIRRORS)].format(set_id), timeout=120,
                                    headers={"User-Agent": "Mozilla/5.0"})
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if response.status_code == 404:
            return None
        if response.status_code == 200 and response.content[:2] == b"PK":
            os.makedirs(folder, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for name in z.namelist():
                    if name.lower().endswith(".osu"):
                        with open(os.path.join(folder, os.path.basename(name)), "wb") as f:
                            f.write(z.read(name))
            return folder
        time.sleep(5 * (attempt + 1))
    return None


def find_by_md5(folder: str, md5: str) -> str | None:
    """The .osu file a replay was played on (replays store its MD5), relative to data/."""
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        with open(path, "rb") as f:
            if hashlib.md5(f.read()).hexdigest() == md5:
                return os.path.relpath(path, paths.DATA).replace("\\", "/")
    return None
