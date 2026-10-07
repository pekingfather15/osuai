"""Train the cursor model (TCN-WGAN) or the key model.

    python -m osuai.train cursor --data v3 --out models/v4 --epochs 90 --pretrain 60
    python -m osuai.train keys --data v3 --out models/v4 --epochs 30

The cursor model is trained in two phases:
  1. pretraining: the generator alone, on the position error, at a high learning rate.
     The adversarial learning rate is too low to learn precise aiming.
  2. adversarial: a critic learns to tell real paths from generated ones, with the exact
     gradient penalty of WGAN-GP, and the generator is pushed to fool it. Its weight ramps
     up slowly and stays small, so it adds realism without pulling the aim off the notes.
Spinner frames are left out of the position error (its minimum is the spinner centre) and
a spin loss asks for the players' spinning speed and radius instead.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import paths
from .features import INPUT_FEATURES
from .models import CursorModel, KeyModel, SPINNER_FEATURE

UNTIL_CLICK = INPUT_FEATURES.index("time_until_click")
PADDING_UNTIL_CLICK = 9999.0


def load_dataset(name: str):
    folder = os.path.join(paths.ROOT, "datasets", name)
    xs = [np.load(p) for p in sorted(glob.glob(os.path.join(folder, "xs_*.npy")))]
    ys = [np.load(p.replace("xs_", "ys_")) for p in sorted(glob.glob(os.path.join(folder, "xs_*.npy")))]
    if not xs:
        raise SystemExit(f"no dataset in {folder}, build one with python -m osuai.dataset --name {name}")
    return np.concatenate(xs), np.concatenate(ys)


def loaders(xs, ys, batch: int, holdout: float = 0.2, seed: int = 42):
    """Train and held-out loaders of (features, targets, real-frame mask) chunks."""
    order = np.random.default_rng(seed).permutation(len(xs))
    cut = int(len(xs) * (1 - holdout))

    def make(idx, shuffle):
        x = torch.from_numpy(np.ascontiguousarray(xs[idx], dtype=np.float32))
        y = torch.from_numpy(np.ascontiguousarray(ys[idx], dtype=np.float32))
        mask = x[..., UNTIL_CLICK] != PADDING_UNTIL_CLICK
        return DataLoader(TensorDataset(x, y, mask), batch_size=batch, shuffle=shuffle)

    return make(order[:cut], True), make(order[cut:], False)


# ---------------- cursor ----------------

def spin_stats(pos, spinning, min_radius_px=5.0):
    """Mean angular speed (radians per frame) and radius (osu!px) around the centre over
    spinner frames, in either direction."""
    px = torch.stack([pos[..., 0] * 512, pos[..., 1] * 384], dim=-1)
    p, step = px[:, :-1], px[:, 1:] - px[:, :-1]
    r2 = (p ** 2).sum(-1).clamp(min=min_radius_px ** 2)
    omega = (p[..., 0] * step[..., 1] - p[..., 1] * step[..., 0]).abs() / r2
    both = spinning[:, 1:] & spinning[:, :-1]
    n = both.sum().clamp(min=1)
    return (omega * both).sum() / n, (r2.sqrt() * both).sum() / n, bool(both.any())


def spin_loss(real, fake, spinning):
    """At least the players' spinning speed, and a similar radius. Unlike a position loss
    it doesn't care where on the circle the cursor is."""
    real_speed, real_radius, any_spin = spin_stats(real, spinning)
    if not any_spin:
        return fake.sum() * 0.0
    fake_speed, fake_radius, _ = spin_stats(fake, spinning)
    return F.relu(real_speed.detach() - fake_speed) + ((fake_radius - real_radius.detach()) / 100) ** 2


def gradient_penalty(critic, features, real, fake):
    """Exact WGAN-GP penalty: the critic's gradient norm should be 1 between real and fake."""
    a = torch.rand(real.shape[0], 1, 1, device=real.device)
    mix = (a * real + (1 - a) * fake).requires_grad_(True)
    grad, = torch.autograd.grad(critic(features, mix).sum(), mix, create_graph=True)
    return ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()


def train_cursor(args):
    xs, ys = load_dataset(args.data)
    train, held = loaders(xs, ys[..., :2], args.batch)
    model = CursorModel()
    g, c, dev = model.generator, model.critic, model.device
    opt_pre = torch.optim.AdamW(g.parameters(), lr=args.lr_pretrain)
    opt_g = torch.optim.AdamW(g.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_c = torch.optim.AdamW(c.parameters(), lr=args.lr, betas=(0.5, 0.9))
    os.makedirs(args.out, exist_ok=True)
    noise_dim = model.config["noise_dim"]

    for epoch in range(args.epochs):
        g.train(), c.train()
        adversarial = epoch >= args.pretrain
        adv_epoch = epoch - args.pretrain
        adv_weight = args.adv_max * min(max((adv_epoch - 10) / 60, 0.0), 1.0) if adversarial else 0.0
        spin_weight = args.spin if epoch >= args.pretrain // 2 else 0.0
        totals = {"pos": 0.0, "spin": 0.0, "wass": 0.0}
        for x, y, mask in train:
            x, y, mask = x.to(dev), y.to(dev), mask.to(dev)
            spinning = mask & (x[..., SPINNER_FEATURE] > 0.5)
            aim = mask & ~spinning

            if adversarial:
                for _ in range(args.critic_steps):
                    with torch.no_grad():
                        fake = g(x, torch.randn(len(x), noise_dim, device=dev))
                    real_score, fake_score = c(x, y), c(x, fake)
                    loss_c = fake_score.mean() - real_score.mean() + args.gp * gradient_penalty(c, x, y, fake)
                    opt_c.zero_grad()
                    loss_c.backward()
                    torch.nn.utils.clip_grad_norm_(c.parameters(), 1.0)
                    opt_c.step()
                    totals["wass"] += float(real_score.mean() - fake_score.mean()) / args.critic_steps

            fake = g(x, torch.randn(len(x), noise_dim, device=dev))
            pos = F.smooth_l1_loss(fake[aim], y[aim])
            spin = spin_weight * spin_loss(y, fake, spinning) if spin_weight else torch.zeros((), device=dev)
            loss = pos + spin
            if adversarial:
                loss = loss - adv_weight * c(x, fake).mean()
            opt = opt_g if adversarial else opt_pre
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
            opt.step()
            totals["pos"] += float(pos)
            totals["spin"] += float(spin)

        held_pos = evaluate_positions(g, held, noise_dim, dev)
        n = len(train)
        print(f"epoch {epoch}: pos {totals['pos'] / n:.5f} held-out pos {held_pos:.5f} "
              f"spin {totals['spin'] / n:.5f} wass {totals['wass'] / n:.3f} adv weight {adv_weight:.6f}")
        if args.save_every and (epoch + 1) % args.save_every == 0:
            model.save(os.path.join(args.out, f"cursor_tcn_wgan_ep{epoch + 1}.pt"))
    model.save(os.path.join(args.out, "cursor_tcn_wgan.pt"))


@torch.no_grad()
def evaluate_positions(g, loader, noise_dim, dev) -> float:
    g.eval()
    total, n = 0.0, 0
    for x, y, mask in loader:
        x, y, mask = x.to(dev), y.to(dev), mask.to(dev)
        aim = mask & ~(x[..., SPINNER_FEATURE] > 0.5)
        total += float(F.smooth_l1_loss(g(x, torch.randn(len(x), noise_dim, device=dev))[aim], y[aim]))
        n += 1
    return total / max(n, 1)


# ---------------- keys ----------------

def main_key_first(keys: torch.Tensor) -> torch.Tensor:
    """Swap the keys of chunks where key 2 is pressed more often, so key 1 is always the
    player's main key and the model doesn't have to guess which key a player prefers."""
    before = torch.cat([torch.zeros_like(keys[:, :1]), keys[:, :-1]], dim=1)
    presses = ((keys > 0.5) & (before < 0.5)).sum(dim=1)
    swap = presses[:, 1] > presses[:, 0]
    keys = keys.clone()
    keys[swap] = keys[swap].flip(-1)
    return keys


def train_keys(args):
    xs, ys = load_dataset(args.data)
    train, held = loaders(xs, ys[..., 2:], args.batch)
    model = KeyModel()
    net, dev = model.net, model.device
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr_keys, weight_decay=0.001)
    os.makedirs(args.out, exist_ok=True)

    def loss_of(x, y, mask):
        x, y, mask = x.to(dev), main_key_first(y.to(dev)), mask.to(dev)
        return F.binary_cross_entropy_with_logits(net(x)[mask], y[mask])

    for epoch in range(args.epochs):
        net.train()
        total = 0.0
        for x, y, mask in train:
            loss = loss_of(x, y, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(loss)
        net.eval()
        with torch.no_grad():
            held_loss = sum(float(loss_of(*b)) for b in held) / max(len(held), 1)
        print(f"epoch {epoch}: loss {total / len(train):.4f} held-out {held_loss:.4f}")
    model.save(os.path.join(args.out, "keys_lstm.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cursor", "keys"])
    ap.add_argument("--data", required=True, help="dataset name under datasets/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pretrain", type=int, default=60, help="cursor: epochs of position-only pretraining")
    ap.add_argument("--lr-pretrain", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-4, help="cursor: adversarial learning rate")
    ap.add_argument("--lr-keys", type=float, default=0.01)
    ap.add_argument("--critic-steps", type=int, default=3)
    ap.add_argument("--gp", type=float, default=8.0, help="gradient penalty weight")
    ap.add_argument("--adv-max", type=float, default=2e-4, help="largest adversarial loss weight")
    ap.add_argument("--spin", type=float, default=0.02, help="spin loss weight")
    ap.add_argument("--save-every", type=int, default=5)
    args = ap.parse_args()
    (train_cursor if args.what == "cursor" else train_keys)(args)


if __name__ == "__main__":
    main()
