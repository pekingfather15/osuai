"""Window for playing generated plays back in osu!lazer.

    python -m osuai.play

Start tosu, log out of osu!lazer and turn off "High precision mouse". Pick a map in the
game; the play is generated when the map changes. Press Q to start, W to stop and R to
release the keys. Playback only runs while nobody is logged in.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import keyboard

from .. import mods as mod_flags, paths
from ..models import KeyModel, load_cursor
from .generate import generate
from .player import Player
from .tosu import Tosu

DEFAULT_CURSOR = os.path.join(paths.MODELS, "v3", "cursor_tcn_wgan.pt")
DEFAULT_KEYS = os.path.join(paths.MODELS, "v3", "keys_lstm.pt")


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("osu! AI")
        root.resizable(False, False)

        self.tosu = Tosu()
        self.player = Player(self.tosu, log=self.log)
        self.cursor_model = None
        self.key_model = None
        self._generated_for = None
        self._generating = False

        self.status = tk.StringVar(value="waiting for tosu")
        self.map_text = tk.StringVar(value="-")
        self.models_text = tk.StringVar(value="-")
        self.active = tk.BooleanVar(value=False)
        self.cursor_only = tk.BooleanVar(value=False)
        self.storyboard = tk.BooleanVar(value=False)
        self.offset = tk.DoubleVar(value=self.player.offset_ms)
        self.gap = tk.IntVar(value=36)

        pad = {"padx": 8, "pady": 3, "sticky": "w"}
        frame = ttk.Frame(root, padding=8)
        frame.grid()
        row = 0
        for label, var in (("Status", self.status), ("Map", self.map_text), ("Models", self.models_text)):
            ttk.Label(frame, text=label).grid(row=row, column=0, **pad)
            ttk.Label(frame, textvariable=var, width=60).grid(row=row, column=1, columnspan=3, **pad)
            row += 1

        ttk.Button(frame, text="Load cursor model", command=self.pick_cursor).grid(row=row, column=0, **pad)
        ttk.Button(frame, text="Load key model", command=self.pick_keys).grid(row=row, column=1, **pad)
        ttk.Button(frame, text="Generate again", command=self.regenerate).grid(row=row, column=2, **pad)
        row += 1

        ttk.Checkbutton(frame, text="Active (Q / W)", variable=self.active,
                        command=self.apply_settings).grid(row=row, column=0, **pad)
        ttk.Checkbutton(frame, text="Cursor only, for Relax", variable=self.cursor_only,
                        command=self.regenerate).grid(row=row, column=1, **pad)
        ttk.Checkbutton(frame, text="Map has a storyboard", variable=self.storyboard,
                        command=self.apply_settings).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(frame, text="Offset (ms)").grid(row=row, column=0, **pad)
        ttk.Scale(frame, from_=-50, to=100, variable=self.offset, length=260,
                  command=lambda _v: self.apply_settings()).grid(row=row, column=1, columnspan=2, **pad)
        self.offset_label = ttk.Label(frame, text="")
        self.offset_label.grid(row=row, column=3, **pad)
        row += 1

        ttk.Label(frame, text="Merge presses closer than (ms)").grid(row=row, column=0, **pad)
        ttk.Spinbox(frame, from_=0, to=100, increment=6, textvariable=self.gap, width=6,
                    command=self.regenerate).grid(row=row, column=1, **pad)
        row += 1

        self.log_box = tk.Text(frame, height=10, width=80, state="disabled")
        self.log_box.grid(row=row, column=0, columnspan=4, pady=6)

        keyboard.add_hotkey("q", lambda: self.root.after(0, self.set_active, True))
        keyboard.add_hotkey("w", lambda: self.root.after(0, self.set_active, False))
        keyboard.add_hotkey("r", lambda: self.root.after(0, self.reset_keys))

        self.load_models(DEFAULT_CURSOR, DEFAULT_KEYS)
        self.apply_settings()
        self.refresh()

    # ---- ui actions ----
    def log(self, text: str):
        def append():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", text + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.root.after(0, append)

    def set_active(self, on: bool):
        self.active.set(on)
        self.apply_settings()
        self.log("playback on" if on else "playback off")

    def reset_keys(self):
        self.player.reset_keys()
        self.log("keys released")

    def apply_settings(self):
        self.player.active = self.active.get()
        self.player.offset_ms = round(self.offset.get())
        self.player.storyboard_shift = self.storyboard.get()
        self.offset_label.configure(text=f"{self.player.offset_ms:+.0f}")
        if not self.player.active:
            self.player.release_keys()

    def load_models(self, cursor_path: str | None, keys_path: str | None):
        try:
            if cursor_path:
                self.cursor_model = load_cursor(cursor_path)
            if keys_path:
                self.key_model = KeyModel.load(keys_path)
        except Exception as e:
            self.log(f"could not load the model: {e}")
        names = [os.path.basename(p) for p in (cursor_path, keys_path) if p]
        self.models_text.set(", ".join(names))
        self.regenerate()

    def pick_cursor(self):
        path = filedialog.askopenfilename(initialdir=paths.MODELS, filetypes=[("model", "*.pt")])
        if path:
            self.load_models(path, None)

    def pick_keys(self):
        path = filedialog.askopenfilename(initialdir=paths.MODELS, filetypes=[("model", "*.pt")])
        if path:
            self.load_models(None, path)

    def regenerate(self):
        self._generated_for = None  # the refresh loop generates again

    # ---- background ----
    def refresh(self):
        tosu = self.tosu
        if not tosu.connected:
            self.status.set("waiting for tosu (is it running?)")
        else:
            who = "guest" if tosu.is_guest() else "LOGGED IN, playback is off"
            self.status.set(f"{who} | {'active' if self.player.active else 'stopped'}")
            mods = tosu.mods()
            self.map_text.set(f"{tosu.map_title()} {self._mod_names(mods)}")
            key = (tosu.map_md5(), mods & mod_flags.GAMEPLAY, self.cursor_only.get(), self.gap.get())
            if key[0] and key != self._generated_for and not self._generating and self.cursor_model:
                self._start_generation(key)
        self.root.after(200, self.refresh)

    @staticmethod
    def _mod_names(mods: int) -> str:
        names = [name for name, bit in (("EZ", 2), ("HD", 8), ("HR", 16), ("DT", 64), ("HT", 256),
                                        ("RX", 128), ("FL", 1024)) if mods & bit]
        return "+" + "".join(names) if names else ""

    def _start_generation(self, key):
        md5, mods, cursor_only, gap = key
        path = self.tosu.map_file()
        if not path or not os.path.exists(path):
            return
        self._generating = True

        def work():
            try:
                play = generate(path, md5, mods, self.cursor_model, self.key_model,
                                merge_gap_ms=gap or None, cursor_only=cursor_only)
                self.player.play = play
                what = "cursor only (turn on Relax)" if play.cursor_only else f"cursor + keys, merge {gap} ms"
                self.log(f"generated: {self.tosu.map_title()} ({what})")
            except Exception as e:
                self.log(f"could not generate a play: {e}")
            finally:
                self._generated_for = key
                self._generating = False

        threading.Thread(target=work, daemon=True).start()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
