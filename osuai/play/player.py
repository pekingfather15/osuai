"""The playback loop: follows the song time from tosu and moves the mouse and presses
the keys of the generated play."""

from __future__ import annotations

import threading
import time
from typing import Callable

import keyboard

from ..features import FRAME_MS
from .generate import Play
from .screen import Playfield, move_mouse
from .tosu import PLAYING, Tosu

KEYS = ("z", "x")
# when the loop falls behind, key changes of up to this many skipped frames are still
# played in order, so short taps aren't lost; bigger jumps are seeks or retries
MAX_REPLAYED_FRAMES = 10


class Player:
    def __init__(self, tosu: Tosu, log: Callable[[str], None] = print):
        self.tosu = tosu
        self.log = log
        self.play: Play | None = None
        self.active = False
        self.offset_ms = 29.0  # play this much earlier, to make up for the input latency
        self.storyboard_shift = False
        self._down = [False, False]
        self._reset = threading.Event()
        self._last_frame = None
        self._playfield = None
        self._playfield_shift = False
        threading.Thread(target=self._run, daemon=True).start()

    # ---- keys ----
    def _set_key(self, k: int, down: bool):
        if down != self._down[k]:
            (keyboard.press if down else keyboard.release)(KEYS[k])
            self._down[k] = down

    def release_keys(self):
        for k in (0, 1):
            self._set_key(k, False)

    def reset_keys(self):
        """Release both keys and forget the key state (R hotkey)."""
        self._reset.set()

    # ---- loop ----
    def _run(self):
        warned_logged_in = False
        while True:
            if self._reset.is_set():
                self._reset.clear()
                self.release_keys()
                self._last_frame = None

            play = self.play
            if not (self.active and play and self.tosu.status() == PLAYING and self.tosu.map_md5() == play.md5):
                self.release_keys()
                self._last_frame = None
                time.sleep(0.01)
                continue
            if not self.tosu.is_guest():
                if not warned_logged_in:
                    self.log("logged in to osu!: playback is off. Log out to test locally")
                    warned_logged_in = True
                self.release_keys()
                time.sleep(0.5)
                continue
            warned_logged_in = False

            song_time = self.tosu.song_time(play.rate)
            if song_time is None:
                time.sleep(0.005)
                continue
            self._step(play, song_time / play.rate + self.offset_ms)
            time.sleep(0.001)

    def _step(self, play: Play, real_time: float):
        pos = (real_time - play.start_time) / FRAME_MS
        n = len(play.cursor)
        if pos < 0 or pos >= n - 1:
            self.release_keys()
            return
        i = int(pos)
        frac = pos - i
        x, y = play.cursor[i] * (1 - frac) + play.cursor[i + 1] * frac
        if self._playfield is None or self._playfield_shift != self.storyboard_shift:
            self._playfield, self._playfield_shift = Playfield(self.storyboard_shift), self.storyboard_shift
        move_mouse(*self._playfield.to_screen(x, y))

        last = self._last_frame
        if last is not None and last < i - 1 <= last + MAX_REPLAYED_FRAMES:
            for j in range(last + 1, i):
                for k in (0, 1):
                    self._set_key(k, bool(play.keys[j, k]))
        if last != i:
            for k in (0, 1):
                self._set_key(k, bool(play.keys[i, k]))
            self._last_frame = i
