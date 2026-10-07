"""Draw the paper's figures from the results and the V3 models.

    python paper/figures/make_figures.py                  # all figures
    python paper/figures/make_figures.py hit_errors       # only some

Writes PDFs next to this script and PNGs to docs/images/.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from osuai import beatmap as bm, decoding, features, paths, replay as rp
from osuai.beatmap import Spinner
from osuai.models import KeyModel, load_cursor

OUT = os.path.dirname(os.path.abspath(__file__))
PNG = os.path.join(ROOT, "docs", "images")
RESULTS = os.path.join(ROOT, "results")
V3 = os.path.join(paths.MODELS, "v3")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COLORS = {"human": "#2f6db5", "lstm": "#d9822b", "tcn": "#2e9e5b", "red": "#c0392b"}
# the map of the in-game runs at 7.88 stars, as osu!lazer stores it
KEY_FIGURE_MAP = os.path.expandvars(r"%APPDATA%\osu\files\5\50\5053a5d6b8844ad627e415e73ce5ff29b02c2b8b5da039dba73eedd84f8c005b")


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    os.makedirs(PNG, exist_ok=True)
    fig.savefig(os.path.join(PNG, f"{name}.png"), dpi=150)
    plt.close(fig)
    print("wrote", name)


def trajectories():
    with open(os.path.join(paths.SPLITS, "test.json")) as f:
        cases = json.load(f)["replays"]
    # the first test map with a spinner of at least 2 seconds
    for case in cases:
        replay = rp.load(paths.data(case["replay"]))
        beatmap = bm.load(paths.data(case["map"]), replay.mods)
        spinners = [o for o in beatmap.objects if isinstance(o, Spinner) and o.end_time - o.time >= 2000]
        if spinners:
            break
    spinner = spinners[0]
    print("trajectory map:", beatmap.title(), beatmap.version())

    x = features.map_features(beatmap)
    human = features.to_playfield(features.replay_targets(beatmap, replay).reshape(-1, 4))
    plays = {
        "Human": (human, COLORS["human"]),
        "V3 LSTM": (features.to_playfield(load_cursor(os.path.join(V3, "cursor_lstm.pt")).generate(x).reshape(-1, 2)),
                    COLORS["lstm"]),
        "V3 TCN-WGAN": (features.to_playfield(load_cursor(os.path.join(V3, "cursor_tcn_wgan.pt"))
                                              .generate(x, seed=0).reshape(-1, 2)), COLORS["tcn"]),
    }
    circles = [o for o in beatmap.objects if not isinstance(o, Spinner) and o.time < spinner.time]
    times = np.array([o.time for o in circles])
    busiest = max(range(len(times)), key=lambda i: ((times >= times[i]) & (times < times[i] + 1500)).sum())
    windows = [(times[busiest] - 100, times[busiest] + 1500), (spinner.time, spinner.time + 2000)]

    def frames(cursor, window):
        a, b = ((w - beatmap.start_time) / features.FRAME_MS for w in window)
        return cursor[max(int(a), 0):int(b)]

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.5))
    for col, (name, (cursor, color)) in enumerate(plays.items()):
        for row, window in enumerate(windows):
            ax = axes[row, col]
            if row == 0:
                for o in circles:
                    if window[0] <= o.time < window[1]:
                        ax.add_patch(plt.Circle((o.x, o.y), beatmap.circle_radius, fill=False, color="0.6", lw=0.8))
                ax.set_title(name)
            else:
                ax.plot(256, 192, "+", color="0.5")
            seg = frames(cursor, window)
            ax.plot(seg[:, 0], seg[:, 1], color=color, lw=1.0)
            ax.set_xlim(0, 512)
            ax.set_ylim(384, 0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color("0.8")
    axes[0, 0].set_ylabel("aim (1.5 s)")
    axes[1, 0].set_ylabel("spinner (2 s)")
    save(fig, "trajectories")


def checkpoints():
    with open(os.path.join(RESULTS, "history", "v3_checkpoint_selection_val.json")) as f:
        r = json.load(f)["results"]["test"]
    pts = sorted((int(k.split("_ep")[1]), v["relax_accuracy"] * 100) for k, v in r.items() if "_ep" in k)
    eps, acc = zip(*pts)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(eps, acc, "o-", color=COLORS["tcn"], ms=4)
    ax.axvline(60, color="0.6", ls="--", lw=0.8)
    ax.text(61, min(acc) + 2, "adversarial\ntraining", fontsize=7, color="0.4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val. Relax accuracy (%)")
    save(fig, "checkpoints")


def gap_sweep():
    with open(os.path.join(RESULTS, "key_decoding_val.json")) as f:
        r = json.load(f)["results"]
    rows = []
    for name, v in r.items():
        if name.startswith("model"):
            gap = 0.0 if "threshold" in name else float(name.split("merge ")[1].split(" ms")[0])
            rows.append((gap, v["full_accuracy"] * 100, v["presses_per_object"]))
    gaps, acc, ppo = map(np.array, zip(*sorted(rows)))
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.plot(gaps, acc, "o-", color=COLORS["tcn"], ms=4, label="full accuracy")
    ax.axhline(r["human"]["full_accuracy"] * 100, color=COLORS["human"], ls="--", lw=0.8, label="human")
    ax.set_xlabel("minimum press gap (ms), 0 = no merging")
    ax.set_ylabel("full accuracy (%)")
    ax2 = ax.twinx()
    ax2.plot(gaps, ppo, "s:", color="0.45", ms=3, label="presses / object")
    ax2.set_ylabel("presses per object")
    ax2.spines["right"].set_visible(True)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="lower left", frameon=False)
    save(fig, "gap_sweep")


def hit_errors():
    runs = [("hit_errors_10-07_19-29-37.json", "5.70*, offset 15"),
            ("hit_errors_10-07_21-13-07.json", "7.88*, offset 29"),
            ("hit_errors_10-07_22-25-03.json", "7.11* HDHR, offset 29, merged")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0))
    for ax, (name, label) in zip(axes, runs):
        with open(os.path.join(RESULTS, "ingame", name)) as f:
            e = np.array(json.load(f)["errors_ms"])
        ax.hist(e, bins=np.arange(-110, 60, 5), color=COLORS["tcn"], alpha=0.85)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.axvline(np.median(e), color=COLORS["red"], ls="--", lw=0.8)
        ax.set_title(f"{label}\nmedian {np.median(e):+.1f} ms, UR {e.std() * 10:.0f}", fontsize=8)
        ax.set_xlabel("hit error (ms)")
    axes[0].set_ylabel("hits")
    save(fig, "hit_errors")


def key_decoding():
    if not os.path.exists(KEY_FIGURE_MAP):
        print("skipping key_decoding, the example map is not installed")
        return
    beatmap = bm.load(KEY_FIGURE_MAP)
    probs = KeyModel.load(os.path.join(V3, "keys_lstm.pt")).hold_probabilities(features.map_features(beatmap)).reshape(-1, 2)
    raw = decoding.threshold(probs)
    merged = decoding.merge_presses(probs, 3)

    # the 800 ms window with the most double presses (both keys going down together)
    edges = raw * (1 - np.vstack([np.zeros((1, 2)), raw[:-1]]))
    win = 66
    start = int(np.argmax(np.convolve((edges.sum(axis=1) == 2).astype(float), np.ones(win), "valid")))
    t = beatmap.start_time + np.arange(start, start + win) * features.FRAME_MS
    objects = [o.time for o in beatmap.objects if t[0] <= o.time <= t[-1]]
    rel = t - t[0]

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 3.0), sharex=True)
    axes[0].plot(rel, probs[start:start + win, 0], color=COLORS["human"], label="P(K1 held)")
    axes[0].plot(rel, probs[start:start + win, 1], color=COLORS["lstm"], label="P(K2 held)")
    axes[0].axhline(0.5, color="0.6", ls=":", lw=0.8)
    axes[0].set_ylabel("probability", fontsize=7)
    axes[0].legend(fontsize=7, ncol=2, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.95))
    for ax, keys, title in ((axes[1], raw, "threshold 0.5"), (axes[2], merged, "merged, 36 ms")):
        for k, y in ((0, 1.0), (1, 0.0)):
            ax.fill_between(rel, y, y + 0.8 * keys[start:start + win, k], step="post",
                            color=(COLORS["human"], COLORS["lstm"])[k], alpha=0.8)
        ax.set_yticks([0.4, 1.4])
        ax.set_yticklabels(["K2", "K1"])
        ax.set_ylim(-0.1, 1.9)
        ax.set_ylabel(title, fontsize=7)
    for ax in axes:
        for o in objects:
            ax.axvline(o - t[0], color="0.5", lw=0.6, ls="--")
    axes[2].set_xlabel("time (ms); dashed lines: hit objects")
    save(fig, "key_decoding")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["checkpoints", "gap_sweep", "hit_errors", "key_decoding", "trajectories"]:
        globals()[name]()
