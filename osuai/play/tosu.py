"""Game state from tosu (https://github.com/tosuapp/tosu), which reads osu!lazer and
osu!stable and serves the state on a local websocket."""

from __future__ import annotations

import json
import os
import threading
import time

import websocket

HOST = os.environ.get("TOSU_HOST", "127.0.0.1:24050")

# tosu's state numbers (the same as osu!stable's)
PLAYING = 2
SONG_SELECT = 5
RESULTS = 7


class SongClock:
    """The song time, smoothed between tosu's updates.

    tosu sends the song time every 10-20 ms (set POLL_RATE in tosu's config), which is
    coarser than the 12 ms frames. Between updates the time runs on with the local clock
    at the song's rate. If it hasn't changed for a while the game is paused. Extrapolating
    and then receiving the real time can step back a little, which would replay key presses,
    so small steps back are ignored; big ones are a retry or a seek."""

    PAUSED_AFTER_S = 0.25
    MAX_IGNORED_STEP_BACK_MS = 50

    def __init__(self):
        self._reported = None
        self._reported_at = 0.0
        self._last = None

    def update(self, reported: float | None, rate: float) -> float | None:
        if reported is None:
            self._last = None
            return None
        now = time.perf_counter()
        if reported != self._reported:
            self._reported, self._reported_at = reported, now
            estimate = reported
        elif now - self._reported_at > self.PAUSED_AFTER_S:
            estimate = reported
        else:
            estimate = reported + (now - self._reported_at) * 1000 * rate
        if self._last is not None and 0 < self._last - estimate <= self.MAX_IGNORED_STEP_BACK_MS:
            return self._last
        self._last = estimate
        return estimate


class Tosu:
    def __init__(self, host: str = HOST):
        self.host = host
        self._state: dict = {}
        self._precise_time = None
        self._lock = threading.Lock()
        self.clock = SongClock()
        for path, handler in (("/websocket/v2", self._on_state), ("/websocket/v2/precise", self._on_precise)):
            threading.Thread(target=self._listen, args=(path, handler), daemon=True).start()

    def _listen(self, path, handler):
        url = f"ws://{self.host}{path}"
        while True:
            ws = websocket.WebSocketApp(url, on_message=lambda _ws, msg: handler(msg))
            ws.run_forever()
            time.sleep(2)  # tosu isn't running yet, or restarted

    def _on_state(self, message: str):
        state = json.loads(message)
        with self._lock:
            self._state = state

    def _on_precise(self, message: str):
        self._precise_time = json.loads(message).get("currentTime")

    def get(self, *keys):
        with self._lock:
            node = self._state
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node

    @property
    def connected(self) -> bool:
        return bool(self._state)

    # ---- what the player needs ----
    def status(self) -> int | None:
        return self.get("state", "number")

    def mods(self) -> int:
        """Legacy mod bits of the current play (or of the selected mods in song select)."""
        return self.get("play", "mods", "number") or 0

    def map_md5(self) -> str | None:
        return self.get("beatmap", "checksum")

    def map_title(self) -> str:
        b = self.get("beatmap") or {}
        return f"{b.get('artist', '')} - {b.get('title', '')} [{b.get('version', '')}]"

    def map_file(self) -> str | None:
        """Path of the .osu file. osu!lazer keeps files under hashed names, which tosu
        gives relative to the files folder that it reports as folders.songs."""
        relative = self.get("directPath", "beatmapFile")
        folder = self.get("folders", "songs")
        if relative is None:
            return None
        return relative if folder is None or os.path.isabs(relative) else os.path.join(folder, relative)

    def song_time(self, rate: float) -> float | None:
        """Song time in ms (song time, not real time)."""
        if self._precise_time is not None:
            return self._precise_time
        return self.clock.update(self.get("beatmap", "time", "live"), rate)

    def is_guest(self) -> bool:
        """True only when tosu says nobody is logged in: osu!lazer shows the guest as
        "Guest" with id 0. Logged-in plays are submitted online, so the player refuses to
        play unless this is True. Unknown counts as logged in."""
        name, user_id = self.get("profile", "name"), self.get("profile", "id")
        if name is None and user_id is None:
            return False
        return (name or "").lower() in ("", "guest") and (user_id or 0) <= 1
