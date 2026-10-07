"""Where the playfield is on screen in osu!lazer, and moving the mouse there (Windows).

osu!lazer in fullscreen scales the 512x384 playfield by screen height / 480 and centres it.
Maps with a storyboard move it 8 osu!px down (OsuPlayfieldAdjustmentContainer in ppy/osu).
"High precision mouse" must be off in osu!lazer, or it ignores the absolute cursor position.
"""

import ctypes

from .. import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH

_user32 = ctypes.windll.user32
try:
    # report real pixels, not pixels scaled by the Windows display scaling
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    _user32.SetProcessDPIAware()


def screen_size() -> tuple[int, int]:
    return _user32.GetSystemMetrics(0), _user32.GetSystemMetrics(1)


class Playfield:
    def __init__(self, storyboard_shift: bool = False):
        w, h = screen_size()
        self.scale = h / 480
        self.left = (w - PLAYFIELD_WIDTH * self.scale) / 2
        self.top = (h - PLAYFIELD_HEIGHT * self.scale) / 2 + (8 * self.scale if storyboard_shift else 0)

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        return int(self.left + x * self.scale), int(self.top + y * self.scale)


def move_mouse(x: int, y: int):
    _user32.SetCursorPos(x, y)
