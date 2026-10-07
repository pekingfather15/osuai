# osu! AI: playing like a human

Neural networks that learn from top players' replays how to play the rhythm game
[osu!](https://osu.ppy.sh), and play the result back in osu!lazer. The goal is not a
perfect autoplayer but play that *looks human*: smooth, imperfect cursor paths, real
spinning and natural key timing.

[中文说明](README.zh-CN.md) · Paper: [English (PDF)](paper/paper_en.pdf) · [中文 (PDF)](paper/paper_zh.pdf) · Experiment log: [results/RESULTS.md](results/RESULTS.md)

![Cursor paths of a human, an LSTM and the TCN-WGAN on an unseen map](docs/images/trajectories.png)

*On an unseen map, the LSTM aims as well as the GAN but sits still on spinners (bottom).
The TCN-WGAN spins like the human.*

## Results

Trained on 1,992 replays and tested on 50 maps that no model trained on:

| | Relax accuracy | Full accuracy (with keys) | Spinner speed |
|---|---|---|---|
| Human replays | 99.98% | 96.04% | 371 RPM |
| LSTM baseline | 99.92% | 95.74% | 3 RPM |
| **TCN-WGAN** | **99.89%** | **95.45%** | **394 RPM** |

Full accuracy uses the key model with press merging (below); without merging the TCN-WGAN
scores 93.55%. Relax accuracy assumes perfect key timing and measures the cursor alone.

In osu!lazer, played logged out:

| Map | Stars | Mods | Accuracy | 300 / 100 / 50 / miss |
|---|---|---|---|---|
| Ado - AiAiA [Kiss me Insane] | 4.07 | | 98.38% | 668 / 12 / 2 / 3 |
| Ado - AiAiA [Be loved Expert] | 5.70 | | 95.85% | 820 / 39 / 15 / 3 |
| Kenshi Yonezu - IRIS OUT [Darling!!!] | 7.11 | HD HR | **96.63%** | 654 / 28 / 0 / 5 |

On the last map the hit timing had a standard deviation of 8.1 ms (UR 81).

## What this project found

Most of the gains came from fixing training objectives and evaluation, not from new
architectures. The paper explains each finding from first principles:

1. **The GAN critic never learned.** A finite-difference gradient penalty measures the
   gradient along one random direction, which in a 4,096-dimensional trajectory is about
   1/64 of its size. The exact penalty on a convolutional critic fixed it.
2. **The GAN could not aim** because its learning rate was too low. Pretraining the
   generator at a higher rate took Relax accuracy from 56.5% to 96.7%.
3. **Regression models cannot spin.** They learn the average path, and the average of every
   player's spinner circle is the centre. A separate polar output for spinners and a loss
   on spin speed and radius made the GAN spin.
4. **The evaluation used half the real circle radius.** After fixing it, offline predictions
   match the game within about one percentage point.
5. **Double key presses lose accuracy in game.** The key model often presses both keys at
   once. The extra press hits the next note early or steals a slider head and lets go.
   Merging presses closer than 36 ms removed every 50 on the 7.11-star map.

![Key model output before and after merging presses](docs/images/key_decoding.png)

## How it works

```
.osu map ──► features (12 ms frames) ──► TCN-WGAN generator + noise ──► cursor path
                                     └─► key LSTM ──► press merging ──► key presses
                                                                            │
osu!lazer ◄── mouse and keyboard ◄── player loop ◄── song time from tosu ◄──┘
```

| Module | What it does |
|---|---|
| `osuai/beatmap.py`, `slider_path.py` | Read .osu files, slider curves, mods (HR, DT, HT, EZ) |
| `osuai/replay.py` | Read .osr replays |
| `osuai/features.py` | Map features and replay targets on the 12 ms frame timeline |
| `osuai/models.py` | TCN-WGAN generator and critic, LSTM baseline, key model |
| `osuai/train.py` | Training: pretraining, exact gradient penalty, spin loss |
| `osuai/decoding.py` | Press merging |
| `osuai/judge.py`, `evaluate.py` | Scoring plays the way the game does, for models and humans |
| `osuai/dataset.py`, `sources/` | Building the training set, importing replays and maps |
| `osuai/play/` | osu!lazer playback through tosu |

## Running it

Tested on Windows 11 with an NVIDIA GPU, using [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
```

**Evaluate** the included V3 models on the test maps (needs the replays and maps in `data/`):

```bash
uv run python -m osuai.evaluate --split test --gaps none 36
```

**Play back in osu!lazer.** Start [tosu](https://github.com/tosuapp/tosu) (set `POLL_RATE=10`
in its config), log out of osu!lazer and turn off "High precision mouse". Open the window,
pick a map in the game, then press **Q** to start, **W** to stop and **R** to release the
keys. Tick "Cursor only" to let the Relax mod press the keys. Playback only runs while
nobody is logged in.

```bash
uv run python -m osuai.play
```

**Get data.** Import plays from the [osu!3k](https://www.kaggle.com/datasets/ilymeow/osu3k)
dataset (needs your own Kaggle API token), or download a player's replays from the osu!
website (needs your own `OSU_SESSION` cookie in a `.env` file, which git ignores):

```bash
uv run --extra osu3k python -m osuai.sources.osu3k --count 1000
uv run python -m osuai.sources.osu_web --user 7562902
```

**Train:**

```bash
uv run python -m osuai.dataset --name v4
uv run python -m osuai.train cursor --data v4 --out models/v4 --epochs 90 --pretrain 60
uv run python -m osuai.train keys --data v4 --out models/v4 --epochs 30
```

## Responsible use

This is a research project about imitating human behaviour. All in-game testing was done
logged out, so no score was ever submitted online. Using generated play on an online
account breaks the osu! rules and is unfair to other players, and it gets accounts banned.
The playback has no anti-detection features.

## Background and credits

The idea and the model families (LSTM, VAE, WGAN, TCN-WGAN) come from
[niooii/osu](https://github.com/niooii/osu), itself inspired by
[OsuLearn](https://github.com/GuiBrandt/OsuLearn). This project's first version was built
on niooii's code and trained the V3 models; all code in this repository was then written
from scratch for this project, and the V3 models were converted to it with identical
outputs. Data: the [osu!3k](https://www.kaggle.com/datasets/ilymeow/osu3k) dataset (MIT
licence). Playback: [tosu](https://github.com/tosuapp/tosu) and [osu!lazer](https://github.com/ppy/osu).

The code, experiments and paper were developed with Claude (Anthropic) as a coding assistant.
Replays, maps and music are not included.
