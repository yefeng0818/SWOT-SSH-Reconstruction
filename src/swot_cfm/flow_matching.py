"""Conditional Flow Matching (CFM) training, checkpoint loading, and Heun ODE ensemble sampling."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import LLC4320Dataset, normalization
from .model import build_cfm_model
from .utils import emit, obs_bins, resolve, work_path


def _accelerator(cfg: dict):
    from accelerate import Accelerator

    t = cfg["training"]
    return Accelerator(
        mixed_precision=t["mixed_precision"],
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 1)),
    )


def in_channels(cfg: dict) -> int:
    bins = obs_bins(cfg)
    return 5 + 1 + 2 * bins + 3 + 1


def _ckpt_dir(cfg: dict) -> Path:
    path = work_path(cfg, "checkpoints", "x")
    return path.parent


def _save(path: Path, model, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), **extra}, tmp)
    tmp.replace(path)


def load_cfm(cfg: dict, device, checkpoint_path: str | Path | None = None):
    model = build_cfm_model(cfg, in_channels(cfg))
    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
    else:
        ckpt = _ckpt_dir(cfg) / "best.pt"
        if not ckpt.exists():
            ckpt = resolve(cfg, "weights/best.pt")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def _loader(ds, batch: int, cfg: dict, shuffle: bool):
    t = cfg["training"]
    workers = int(t["num_workers"])
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(t["pin_memory"]),
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def _cosine(optimizer, total_steps: int, warmup: int):
    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_cfm(cfg: dict, resume: bool = True) -> None:
    """Train the Conditional Flow Matching velocity network on normalized residual r = (truth - DUACS) / sigma."""
    acc = _accelerator(cfg)
    s = cfg["training"]
    norm = normalization(cfg)
    train = LLC4320Dataset(cfg, "train", train=True, norm=norm)
    val = LLC4320Dataset(cfg, "validation", train=False, norm=norm, max_days=s.get("validation_days"))
    model = build_cfm_model(cfg, in_channels(cfg))
    opt = torch.optim.AdamW(model.parameters(), lr=s["learning_rate"], weight_decay=s["weight_decay"])
    loader = _loader(train, s["batch_size"], cfg, True)
    sched = _cosine(opt, s["epochs"] * max(1, len(loader)), s["warmup_steps"])
    model, opt, loader, sched = acc.prepare(model, opt, loader, sched)
    out = _ckpt_dir(cfg)
    start_epoch, best = 0, float("inf")
    if resume and (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        acc.unwrap_model(model).load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_epoch, best = state["epoch"] + 1, state["best"]
    if acc.is_main_process:
        emit(
            "cfm_train_start",
            parameters=sum(p.numel() for p in acc.unwrap_model(model).parameters()),
            train_days=len(train),
            steps_per_epoch=len(loader),
        )

    def cfm_loss(net, x, y, generator=None):
        noise = torch.randn(y.shape, device=y.device, generator=generator)
        t = torch.rand(y.shape[0], device=y.device, generator=generator)
        xt = (1 - t)[:, None, None, None] * noise + t[:, None, None, None] * y
        v = net(torch.cat([xt, x], 1), t).float()
        return (v - (y - noise)).square().mean()

    for epoch in range(start_epoch, s["epochs"]):
        model.train()
        losses, t0 = [], time.time()
        for step, batch in enumerate(loader):
            with acc.accumulate(model):
                loss = cfm_loss(model, batch["x"], batch["y"])
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), s["gradient_clip_norm"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(loss))
            if step % s["log_every"] == 0 and acc.is_main_process:
                emit("train_step", epoch=epoch, step=step, loss=float(loss))
        score = None
        if (epoch + 1) % s["validate_every"] == 0 or epoch + 1 == s["epochs"]:
            net = acc.unwrap_model(model).eval()
            g = torch.Generator(device=acc.device).manual_seed(1234)
            with torch.no_grad():
                vals = [
                    float(cfm_loss(net, val[i]["x"][None].to(acc.device), val[i]["y"][None].to(acc.device), g))
                    for i in range(len(val))
                    for _ in range(2)
                ]
            score = float(np.mean(vals))
            if acc.is_main_process and score < best:
                best = score
                _save(out / "best.pt", net, {"epoch": epoch, "score": score})
        if acc.is_main_process:
            _save(
                out / "last.pt",
                acc.unwrap_model(model),
                {"epoch": epoch, "best": best, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()},
            )
            emit("epoch_done", epoch=epoch, seconds=time.time() - t0, train_loss=float(np.mean(losses)), validation=score, best=best)
        acc.wait_for_everyone()


def observe(fields: torch.Tensor, corners: torch.Tensor, wspace: torch.Tensor, wtime: torch.Tensor) -> torch.Tensor:
    """Interpolate bracketing daily fields (2, H, W) onto native SWOT swath observations."""
    flat = fields.reshape(2, -1)
    v0 = (flat[0, corners] * wspace).sum(-1)
    v1 = (flat[1, corners] * wspace).sum(-1)
    return wtime[:, 0] * v0 + wtime[:, 1] * v1


def project_out(r: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
    """Remove per-pass mean and along-track / cross-track linear plane."""
    coef = torch.linalg.lstsq(design, r[:, None]).solution[:, 0]
    return r - design @ coef


@torch.no_grad()
def sample_cfm_ensemble(
    model,
    x: torch.Tensor,
    duacs: torch.Tensor,
    norm: dict,
    cfg: dict,
    device,
    members: int = 16,
    seed: int = 0,
    steps: int | None = None,
    solver: str | None = None,
    member_batch: int = 8,
) -> torch.Tensor:
    """Integrate the learned ODE from N(0, I) at tau=0 to tau=1; returns (members, N, H, W) in metres."""
    x = x.to(device)
    steps = int(steps or cfg["cfm"].get("steps", 32))
    solver = solver or cfg["cfm"].get("solver", "heun")

    n = x.shape[0]
    outs = []
    for m0 in range(0, members, member_batch):
        ms = list(range(m0, min(members, m0 + member_batch)))
        noise = [
            torch.randn(
                1,
                1,
                *x.shape[-2:],
                device=device,
                generator=torch.Generator(device=device).manual_seed(seed * 1000 + m),
            ).expand(n, -1, -1, -1)
            for m in ms
        ]
        state = torch.cat(noise, 0).clone()
        xb = x.repeat(len(ms), 1, 1, 1)
        h = 1.0 / steps

        def vel(st, t_value):
            t = torch.full((st.shape[0],), float(t_value), device=device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                return model(torch.cat([st, xb], 1), t).float()

        for s in range(steps):
            if solver == "heun":
                t0 = s * h
                v0 = vel(state, t0)
                state = state + 0.5 * h * (v0 + vel(state + h * v0, t0 + h))
            else:
                state = state + h * vel(state, (s + 0.5) * h)
        res = state[:, 0].reshape(len(ms), n, *x.shape[-2:])
        outs.append(duacs.to(device)[None] + res * norm["residual_scale_m"])
    return torch.cat(outs, 0)
