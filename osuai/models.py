"""Neural networks: the TCN-WGAN cursor model and the key model.

Checkpoints are dicts saved with torch.save:
  {"kind": "cursor_tcn_wgan" | "keys_lstm", "config": {...}, <weights>...}
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import INPUT_FEATURES, N_INPUT

SPINNER_FEATURE = INPUT_FEATURES.index("is_spinner")


class PaddedConv1d(nn.Conv1d):
    """Length-preserving 1D convolution whose last tap is `ahead` frames in the future
    (0 = causal: it sees only the present and the past). The shift is in frames, not in
    dilated steps, so a dilated layer still looks only `ahead` frames ahead."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, ahead: int):
        super().__init__(in_ch, out_ch, kernel, dilation=dilation)
        span = (kernel - 1) * dilation
        self.pad = (max(span - ahead, 0), ahead)

    def forward(self, x):
        return super().forward(F.pad(x, self.pad))


@dataclass
class TCNConfig:
    """Channels of the hidden layers. The first `future_layers` layers (dilations 1, 2,
    4, ...) each look `kernel - 1` frames ahead, or one frame when centred=True; the rest
    are causal and restart at dilation 1."""
    channels: list = field(default_factory=list)
    future_layers: int = 0
    centred: bool = False
    kernel: int = 3


class TCN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, config: TCNConfig):
        super().__init__()
        sizes = [in_ch, *config.channels, out_ch]
        k = config.kernel
        layers = []
        for i in range(len(sizes) - 1):
            if i < config.future_layers:
                dilation, look = 2 ** i, 1 if config.centred else k - 1
            else:
                dilation, look = 2 ** (i - config.future_layers), 0
            layers.append(PaddedConv1d(sizes[i], sizes[i + 1], k, dilation, look))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):  # [B, T, C] -> [B, T, out]
        h = x.transpose(1, 2)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h.transpose(1, 2)


GENERATOR_CONFIG = TCNConfig([64, 128, 128, 128, 128, 96, 96, 96, 32], future_layers=6)
CRITIC_CONFIG = TCNConfig([64, 96, 96, 96, 96, 96, 32], future_layers=7, centred=True)


class CursorGenerator(nn.Module):
    """Map features + a noise vector per chunk -> cursor x, y in [-0.5, 0.5].

    With the polar spinner head, spinner frames are drawn on a circle around the
    playfield centre instead: the network gives a radius and an angular speed per frame,
    and the noise picks the direction and the start angle. A position output can't learn
    to spin, because the average of every player's circle is the centre."""

    def __init__(self, noise_dim: int = 8, spinner_head: bool = True, config: TCNConfig = GENERATOR_CONFIG):
        super().__init__()
        self.noise_dim = noise_dim
        self.spinner_head = spinner_head
        self.net = TCN(N_INPUT + noise_dim, 4 if spinner_head else 2, config)

    def forward(self, features, noise):
        b, t, _ = features.shape
        out = self.net(torch.cat([features, noise[:, None, :].expand(b, t, self.noise_dim)], dim=-1))
        xy = out[..., :2]
        if not self.spinner_head:
            return xy
        radius = 20 + 30 * F.softplus(out[..., 2])        # osu!px
        speed = 0.3 * F.softplus(out[..., 3])             # radians per frame
        spinning = (features[..., SPINNER_FEATURE] > 0.5).float()
        direction = noise[:, :1].sign().clamp(-1, 1)
        angle = noise[:, 1:2] * math.pi + direction * torch.cumsum(speed * spinning, dim=1)
        circle = torch.stack([radius * torch.cos(angle) / 512, radius * torch.sin(angle) / 384], dim=-1)
        w = spinning[..., None]
        return xy * (1 - w) + circle * w


class CursorCritic(nn.Module):
    """Scores how real a cursor path looks for a map (higher = more real)."""

    def __init__(self, config: TCNConfig = CRITIC_CONFIG):
        super().__init__()
        self.net = TCN(N_INPUT + 2, 1, config)

    def forward(self, features, cursor):
        return self.net(torch.cat([features, cursor], dim=-1)).mean(dim=(1, 2))


class KeyLSTM(nn.Module):
    """Map features -> logits that key 1 / key 2 are held, per frame."""

    def __init__(self, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(N_INPUT, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.hidden = nn.Linear(hidden, 32)
        self.out = nn.Linear(32, 2)

    def forward(self, features):
        h, _ = self.lstm(features)
        return self.out(F.relu(self.hidden(h)))


class CursorLSTM(nn.Module):
    """Baseline: a deterministic regressor from map features to the cursor position."""

    def __init__(self, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(N_INPUT, hidden, num_layers=layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, features):
        h, _ = self.lstm(features)
        return self.head(h)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CursorModel:
    """Generator (and critic, for training) of the TCN-WGAN."""

    kind = "cursor_tcn_wgan"

    def __init__(self, noise_dim=8, spinner_head=True, gen_config=GENERATOR_CONFIG,
                 critic_config=CRITIC_CONFIG, device=None):
        self.device = device or default_device()
        self.config = {"noise_dim": noise_dim, "spinner_head": spinner_head,
                       "generator": asdict(gen_config), "critic": asdict(critic_config)}
        self.generator = CursorGenerator(noise_dim, spinner_head, gen_config).to(self.device)
        self.critic = CursorCritic(critic_config).to(self.device)

    @torch.no_grad()
    def generate(self, features: np.ndarray, seed: int | None = None) -> np.ndarray:
        """[chunks, 2048, 9] -> [chunks, 2048, 2] cursor in [-0.5, 0.5]. Each chunk gets its own noise."""
        self.generator.eval()
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        else:
            gen.seed()
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        noise = torch.randn(x.shape[0], self.config["noise_dim"], generator=gen).to(self.device)
        return self.generator(x, noise).cpu().numpy()

    def save(self, path: str):
        torch.save({"kind": self.kind, "config": self.config,
                    "generator": self.generator.state_dict(), "critic": self.critic.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device=None) -> "CursorModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("kind") != cls.kind:
            raise ValueError(f"{path} is not a {cls.kind} checkpoint")
        c = ckpt["config"]
        model = cls(c["noise_dim"], c["spinner_head"], TCNConfig(**c["generator"]), TCNConfig(**c["critic"]), device)
        model.generator.load_state_dict(ckpt["generator"])
        model.critic.load_state_dict(ckpt["critic"])
        return model


class KeyModel:
    kind = "keys_lstm"

    def __init__(self, hidden=64, layers=2, device=None):
        self.device = device or default_device()
        self.config = {"hidden": hidden, "layers": layers}
        self.net = KeyLSTM(hidden, layers).to(self.device)

    @torch.no_grad()
    def hold_probabilities(self, features: np.ndarray) -> np.ndarray:
        """[chunks, 2048, 9] -> [chunks, 2048, 2] probability that each key is held."""
        self.net.eval()
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        return torch.sigmoid(self.net(x)).cpu().numpy()

    def save(self, path: str):
        torch.save({"kind": self.kind, "config": self.config, "net": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device=None) -> "KeyModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("kind") != cls.kind:
            raise ValueError(f"{path} is not a {cls.kind} checkpoint")
        model = cls(**ckpt["config"], device=device)
        model.net.load_state_dict(ckpt["net"])
        return model


class CursorLSTMModel:
    """The LSTM baseline. It has no noise input, so every play of a map is the same."""

    kind = "cursor_lstm"

    def __init__(self, hidden=128, layers=2, device=None):
        self.device = device or default_device()
        self.config = {"hidden": hidden, "layers": layers}
        self.net = CursorLSTM(hidden, layers).to(self.device)

    @torch.no_grad()
    def generate(self, features: np.ndarray, seed: int | None = None) -> np.ndarray:
        self.net.eval()
        return self.net(torch.as_tensor(features, dtype=torch.float32, device=self.device)).cpu().numpy()

    def save(self, path: str):
        torch.save({"kind": self.kind, "config": self.config, "net": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device=None) -> "CursorLSTMModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("kind") != cls.kind:
            raise ValueError(f"{path} is not a {cls.kind} checkpoint")
        model = cls(**ckpt["config"], device=device)
        model.net.load_state_dict(ckpt["net"])
        return model


def load_cursor(path: str, device=None):
    """A cursor model of either kind."""
    kind = torch.load(path, map_location="cpu", weights_only=True).get("kind")
    return {m.kind: m for m in (CursorModel, CursorLSTMModel)}[kind].load(path, device)
