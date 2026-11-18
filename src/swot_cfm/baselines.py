"""Benchmark baseline models for LLC4320 OSSE comparison (SI Text S3.1, Figure S3):
  - Deterministic: U-Net, Fourier Neural Operator (FNO), Vision Transformer (ViT), 4DVarNet
  - Generative: Elucidated Diffusion Model (EDM), Denoising Diffusion Probabilistic Model (DDPM)
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .dataset import LLC4320Dataset, normalization
from .flow_matching import _accelerator, _cosine, _loader, _save, in_channels
from .model import ZeroMeanResidual, build_cfm_model, build_unet_baseline
from .utils import emit, obs_bins, work_path

PIXEL_KM = 1.29
DETERMINISTIC = ("unet", "fno", "vit", "fourdvarnet")
GENERATIVE = ("edm", "ddpm")

_KERNELS: dict = {}


# -------------------------------------------------------------------------------------- Deterministic Loss
def _gauss_kernel(sigma: float, device, dtype) -> torch.Tensor:
    key = (round(float(sigma), 4), str(device), dtype)
    if key not in _KERNELS:
        radius = max(1, int(math.ceil(3 * sigma)))
        x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        _KERNELS[key] = (k / k.sum()).to(dtype)
    return _KERNELS[key]


def _odd_pad(x: torch.Tensor, r: int, dim: int) -> torch.Tensor:
    if r == 0:
        return x
    first = x.narrow(dim, 0, 1)
    last = x.narrow(dim, x.shape[dim] - 1, 1)
    head = 2 * first - x.narrow(dim, 1, r).flip(dim)
    tail = 2 * last - x.narrow(dim, x.shape[dim] - 1 - r, r).flip(dim)
    return torch.cat([head, x, tail], dim)


def _gaussian_blur(x: torch.Tensor, sigma_px: float) -> torch.Tensor:
    k = _gauss_kernel(sigma_px, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    r_h, r_w = min(r, x.shape[-2] - 1), min(r, x.shape[-1] - 1)
    kh = k[(r - r_h) : (r + r_h + 1)]
    kw = k[(r - r_w) : (r + r_w + 1)]
    kh, kw = kh / kh.sum(), kw / kw.sum()
    y = F.conv2d(_odd_pad(x, r_h, -2), kh.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(_odd_pad(y, r_w, -1), kw.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)


def dense_loss(pred: torch.Tensor, target: torch.Tensor, cfg_loss: dict, pixel_km: float = PIXEL_KM) -> tuple[torch.Tensor, dict]:
    """Scale-aware deterministic loss: mean-removed MSE + spatial gradient MSE + 30-150 km band-pass MSE."""
    e = (pred - target).float()
    e_zm = e - e.mean(dim=(-2, -1), keepdim=True)
    mse = e_zm.square().mean()
    gx = e[..., :, 1:] - e[..., :, :-1]
    gy = e[..., 1:, :] - e[..., :-1, :]
    grad = 0.5 * (gx.square().mean() + gy.square().mean())
    s_lo = cfg_loss["band_km"][0] / (2 * math.pi * pixel_km)
    s_hi = cfg_loss["band_km"][1] / (2 * math.pi * pixel_km)
    band = (_gaussian_blur(e, s_lo) - _gaussian_blur(e, s_hi)).square().mean()
    total = mse + cfg_loss["gradient_weight"] * grad / cfg_loss["gradient_reference"] + cfg_loss["band_weight"] * band
    return total, {"mse": float(mse), "gradient": float(grad), "band": float(band)}


# ----------------------------------------------------------------------------------------------------- FNO
class SpectralConv2d(nn.Module):
    def __init__(self, cin: int, cout: int, modes1: int, modes2: int):
        super().__init__()
        self.cout, self.m1, self.m2 = cout, modes1, modes2
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.randn(cin, cout, modes1, modes2, 2))
        self.w2 = nn.Parameter(scale * torch.randn(cin, cout, modes1, modes2, 2))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            b, _, h, w = x.shape
            xf = torch.fft.rfft2(x, norm="ortho")
            out = torch.zeros(b, self.cout, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
            m1, m2 = self.m1, self.m2
            out[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m1, :m2], torch.view_as_complex(self.w1))
            out[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m1:, :m2], torch.view_as_complex(self.w2))
            return torch.fft.irfft2(out, s=(h, w), norm="ortho")


class FNO2d(nn.Module):
    """Fourier Neural Operator (Li et al., 2020) with appended normalized grid coordinates."""

    def __init__(self, in_channels: int, width: int = 32, modes=(16, 20), layers: int = 4, pad: int = 16, proj: int = 128):
        super().__init__()
        self.pad = pad
        self.lift = nn.Conv2d(in_channels + 2, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, *modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, proj, 1), nn.GELU(), nn.Conv2d(proj, 1, 1))

    def forward(self, x):
        b, _, h, w = x.shape
        gy = torch.linspace(0, 1, h, device=x.device, dtype=x.dtype)[None, None, :, None].expand(b, 1, h, w)
        gx = torch.linspace(0, 1, w, device=x.device, dtype=x.dtype)[None, None, None, :].expand(b, 1, h, w)
        z = self.lift(torch.cat([x, gy, gx], 1))
        z = F.pad(z, (0, self.pad, 0, self.pad))
        for i, (s, p) in enumerate(zip(self.spec, self.pw)):
            z = s(z).to(z.dtype) + p(z)
            if i < len(self.spec) - 1:
                z = F.gelu(z)
        return self.proj(z[..., :h, :w])


# ----------------------------------------------------------------------------------------------------- ViT
class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + self.drop(self.out(a))
        return x + self.mlp(self.n2(x))


class ViT(nn.Module):
    """Dense-regression Vision Transformer (Dosovitskiy et al., 2020) with convolutional refinement head."""

    def __init__(
        self,
        in_channels: int,
        shape=(384, 480),
        patch: int = 8,
        dim: int = 512,
        depth: int = 12,
        heads: int = 8,
        mlp: int = 4,
        dropout: float = 0.0,
        decoder_channels: int = 16,
    ):
        super().__init__()
        self.patch, self.grid = patch, (shape[0] // patch, shape[1] // patch)
        self.embed = nn.Conv2d(in_channels, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.grid[0] * self.grid[1], dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_TransformerBlock(dim, heads, mlp, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.dec = int(decoder_channels)
        self.head = nn.Linear(dim, patch * patch * max(1, self.dec))
        self.refine = (
            nn.Sequential(
                nn.Conv2d(self.dec, 2 * self.dec, 3, padding=1, padding_mode="reflect"),
                nn.GELU(),
                nn.Conv2d(2 * self.dec, 1, 3, padding=1, padding_mode="reflect"),
            )
            if self.dec
            else None
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        b = x.shape[0]
        z = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            z = blk(z)
        z = self.head(self.norm(z))
        gh, gw, p, c = self.grid[0], self.grid[1], self.patch, max(1, self.dec)
        z = z.reshape(b, gh, gw, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, gh * p, gw * p)
        return self.refine(z) if self.refine is not None else z


# ------------------------------------------------------------------------------------------------ 4DVarNet
class BilinAEPrior(nn.Module):
    """4DVarNet bilinear auto-encoder prior Phi(state | condition)."""

    def __init__(self, dim_state: int, dim_cond: int, hidden: int = 64, kernel: int = 3, downsamp: int = 2):
        super().__init__()
        pad = kernel // 2
        self.down = nn.AvgPool2d(downsamp)
        self.up = nn.Upsample(scale_factor=downsamp, mode="bilinear", align_corners=False)
        self.conv_in = nn.Conv2d(dim_state + dim_cond, hidden, kernel, padding=pad)
        self.conv_hidden = nn.Conv2d(hidden, hidden, kernel, padding=pad)
        self.bilin_1 = nn.Conv2d(hidden, hidden, kernel, padding=pad)
        self.bilin_21 = nn.Conv2d(hidden, hidden, kernel, padding=pad)
        self.bilin_22 = nn.Conv2d(hidden, hidden, kernel, padding=pad)
        self.conv_out = nn.Conv2d(2 * hidden, dim_state, kernel, padding=pad)

    def forward(self, state, cond):
        z = self.conv_in(self.down(torch.cat([state, cond], 1)))
        z = self.conv_hidden(F.relu(z))
        z = self.conv_out(torch.cat([self.bilin_1(z), self.bilin_21(z) * self.bilin_22(z)], 1))
        return self.up(z)


class ConvLSTMGrad(nn.Module):
    """4DVarNet ConvLSTM gradient update module."""

    def __init__(self, dim_state: int, dim_cond: int, hidden: int = 64, kernel: int = 3, downsamp: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden, self.ds = hidden, downsamp
        self.down = nn.AvgPool2d(downsamp)
        self.up = nn.Upsample(scale_factor=downsamp, mode="bilinear", align_corners=False)
        self.gates = nn.Conv2d(dim_state + dim_cond + hidden, 4 * hidden, kernel, padding=kernel // 2)
        self.conv_out = nn.Conv2d(hidden, dim_state, kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)

    def reset(self, state):
        b, _, h, w = state.shape
        size = (b, self.hidden, h // self.ds, w // self.ds)
        self._h = torch.zeros(size, device=state.device)
        self._c = torch.zeros(size, device=state.device)
        self._norm = None

    def forward(self, grad, cond_lr):
        if self._norm is None:
            self._norm = grad.square().mean().sqrt().clamp_min(1e-12)
        g = self.down(self.drop(grad / self._norm))
        i, f, o, c = self.gates(torch.cat([g, cond_lr, self._h], 1)).chunk(4, 1)
        self._c = torch.sigmoid(f) * self._c + torch.sigmoid(i) * torch.tanh(c)
        self._h = torch.sigmoid(o) * torch.tanh(self._c)
        return self.up(self.conv_out(self._h))


class FourDVarNet(nn.Module):
    """4DVarNet variational assimilation baseline (Beauchamp et al., 2023)."""

    def __init__(
        self,
        in_channels: int,
        bins: int,
        innov_to_residual: float,
        hidden: int = 64,
        cond: int = 16,
        downsamp: int = 2,
        n_steps: int = 10,
        lr_grad: float = 0.2,
        grad_hidden: int = 96,
    ):
        super().__init__()
        self.bins, self.ratio, self.n_steps, self.lr_grad = bins, float(innov_to_residual), n_steps, lr_grad
        self.centre = bins // 2
        self.cond = nn.Sequential(nn.Conv2d(in_channels, 2 * cond, 3, padding=1), nn.SiLU(), nn.Conv2d(2 * cond, cond, 1))
        self.prior = BilinAEPrior(bins, cond, hidden, downsamp=downsamp)
        self.grad_mod = ConvLSTMGrad(bins, cond, grad_hidden, downsamp=downsamp)
        self.cond_down = nn.AvgPool2d(downsamp)
        self.log_a_obs = nn.Parameter(torch.zeros(()))
        self.log_a_prior = nn.Parameter(torch.zeros(()))
        self.last_prior_cost = None

    def observations(self, x):
        y = x[:, 6 : 6 + self.bins] * self.ratio
        w = x[:, 6 + self.bins : 6 + 2 * self.bins].clamp(0, 1)
        return y, w

    def cost(self, s, y, w, c):
        obs = (w * (s - y).square()).mean()
        prior = (s - self.prior(s, c)).square().mean()
        return self.log_a_obs.exp() * obs + self.log_a_prior.exp() * prior

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False), torch.enable_grad():
            x = x.float()
            y, w = self.observations(x)
            c = self.cond(x)
            c_lr = self.cond_down(c)
            s = torch.zeros_like(y).requires_grad_(True)
            self.grad_mod.reset(s)
            for k in range(self.n_steps):
                u = self.cost(s, y, w, c)
                grad = torch.autograd.grad(u, s, create_graph=self.training)[0]
                s = s - (self.grad_mod(grad, c_lr) / (k + 1) + self.lr_grad * (k + 1) / self.n_steps * grad)
                if not self.training:
                    s = s.detach().requires_grad_(True)
            self.last_prior_cost = (s - self.prior(s, c)).square().mean()
            out = s[:, self.centre : self.centre + 1]
            return out if self.training else out.detach()


# ----------------------------------------------------------------------------------------------------- EDM
class EDMDenoiser(nn.Module):
    """Elucidated Diffusion Model (Karras et al., 2022) wrapping the same U-Net backbone as CFM."""

    def __init__(self, net: nn.Module, sigma_data: float = 1.0):
        super().__init__()
        self.net, self.sd = net, float(sigma_data)

    def forward(self, r_noisy, sigma, cond):
        s = sigma.reshape(-1, 1, 1, 1)
        sd2 = self.sd**2
        c_skip = sd2 / (s**2 + sd2)
        c_out = s * self.sd / (s**2 + sd2).sqrt()
        c_in = 1.0 / (s**2 + sd2).sqrt()
        f = self.net(torch.cat([(c_in * r_noisy).to(cond.dtype), cond], 1), sigma.log().flatten() / 4).float()
        return c_skip * r_noisy + c_out * f


def edm_loss(model: EDMDenoiser, x, y, p_mean: float = -1.2, p_std: float = 1.2, generator=None):
    sigma = (p_mean + p_std * torch.randn(y.shape[0], device=y.device, generator=generator)).exp()
    n = torch.randn(y.shape, device=y.device, generator=generator) * sigma[:, None, None, None]
    d = model(y + n, sigma, x)
    sd = getattr(model, "module", model).sd
    weight = (sigma**2 + sd**2) / (sigma * sd) ** 2
    return (weight[:, None, None, None] * (d - y).square()).mean()


def edm_sigmas(steps: int, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0, device=None):
    i = torch.arange(steps, dtype=torch.float64, device=device)
    t = (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([t, torch.zeros(1, dtype=torch.float64, device=device)])


@torch.no_grad()
def sample_edm_ensemble(
    model: EDMDenoiser,
    x: torch.Tensor,
    duacs: torch.Tensor,
    norm: dict,
    device,
    members: int = 16,
    seed: int = 0,
    steps: int = 30,
    member_batch: int = 4,
) -> torch.Tensor:
    """EDM deterministic Heun ODE sampler (2*steps - 1 function evaluations)."""
    x = x.to(device)
    n = x.shape[0]
    t = edm_sigmas(steps, device=device)
    outs = []
    for m0 in range(0, members, member_batch):
        ms = list(range(m0, min(members, m0 + member_batch)))
        noise = [
            torch.randn(1, 1, *x.shape[-2:], device=device, generator=torch.Generator(device=device).manual_seed(seed * 1000 + m)).expand(
                n, -1, -1, -1
            )
            for m in ms
        ]
        xb = x.repeat(len(ms), 1, 1, 1)
        state = torch.cat(noise, 0).clone() * float(t[0])

        def den(st, sig):
            sigma = torch.full((st.shape[0],), float(sig), device=device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                return model(st, sigma, xb).float()

        for i in range(steps):
            t0, t1 = float(t[i]), float(t[i + 1])
            d0 = (state - den(state, t0)) / t0
            nxt = state + (t1 - t0) * d0
            if t1 > 0:
                d1 = (nxt - den(nxt, t1)) / t1
                nxt = state + (t1 - t0) * 0.5 * (d0 + d1)
            state = nxt
        res = state[:, 0].reshape(len(ms), n, *x.shape[-2:])
        outs.append(duacs.to(device)[None] + res * norm["residual_scale_m"])
    return torch.cat(outs, 0)


# ---------------------------------------------------------------------------------------------------- DDPM
def cosine_alpha_bar(steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.arange(steps + 1, dtype=torch.float64) / steps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0)


class DDPM(nn.Module):
    """Epsilon-prediction DDPM with cosine noise schedule wrapping the same U-Net backbone."""

    def __init__(self, net: nn.Module, steps: int = 1000):
        super().__init__()
        self.net, self.T = net, int(steps)
        ab = cosine_alpha_bar(self.T)
        self.register_buffer("alpha_bar", ab.float(), persistent=False)
        ab_prev = torch.cat([torch.ones(1, dtype=torch.float64), ab[:-1]])
        self.register_buffer("alpha_bar_prev", ab_prev.float(), persistent=False)

    def forward(self, x_t, t, cond):
        return self.net(torch.cat([x_t.to(cond.dtype), cond], 1), t.float() / self.T).float()


def ddpm_loss(model: DDPM, x, y, generator=None):
    m = getattr(model, "module", model)
    t = torch.randint(0, m.T, (y.shape[0],), device=y.device, generator=generator)
    eps = torch.randn(y.shape, device=y.device, generator=generator)
    ab = m.alpha_bar[t][:, None, None, None]
    return (model(ab.sqrt() * y + (1 - ab).sqrt() * eps, t, x) - eps).square().mean()


# --------------------------------------------------------------------------------------------------- Build
def build_baseline(cfg: dict, name: str, norm: dict | None = None) -> nn.Module:
    c = in_channels(cfg)
    b = cfg.get("baselines", {}).get(name, {}).get("model", {})
    if name == "unet":
        return build_unet_baseline(cfg, c)
    if name == "fno":
        return ZeroMeanResidual(FNO2d(c, b.get("width", 32), tuple(b.get("modes", (16, 20))), b.get("layers", 4), b.get("pad", 16)))
    if name == "vit":
        return ZeroMeanResidual(
            ViT(
                c,
                tuple(b.get("shape", (384, 480))),
                b.get("patch", 8),
                b.get("dim", 512),
                b.get("depth", 12),
                b.get("heads", 8),
                b.get("mlp", 4),
                b.get("dropout", 0.0),
                b.get("decoder_channels", 16),
            )
        )
    if name == "fourdvarnet":
        norm = norm or normalization(cfg)
        ratio = norm["innovation_scale_m"] / norm["residual_scale_m"]
        return ZeroMeanResidual(
            FourDVarNet(
                c,
                obs_bins(cfg),
                ratio,
                b.get("hidden", 64),
                b.get("cond", 16),
                b.get("downsamp", 2),
                b.get("n_steps", 10),
                b.get("lr_grad", 0.2),
                b.get("grad_hidden", 96),
            )
        )
    if name == "edm":
        return EDMDenoiser(build_cfm_model(cfg, c), b.get("sigma_data", 1.0))
    if name == "ddpm":
        return DDPM(build_cfm_model(cfg, c), b.get("steps", 1000))
    raise ValueError(f"Unknown baseline model: {name}")


def load_baseline(cfg: dict, name: str, checkpoint_path: str | Path, device, norm: dict | None = None) -> nn.Module:
    model = build_baseline(cfg, name, norm)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


# --------------------------------------------------------------------------------------------------- Train
def train_baseline(cfg: dict, name: str, resume: bool = True) -> None:
    """Train a benchmark baseline model on the exact same LLC4320 OSSE splits and inputs as CFM."""
    s = cfg["baselines"][name]["training"]
    if "gradient_accumulation_steps" in s:
        cfg = dict(cfg, training=dict(cfg["training"], gradient_accumulation_steps=s["gradient_accumulation_steps"]))
    acc = _accelerator(cfg)
    gen = name in GENERATIVE
    gen_loss = {"edm": edm_loss, "ddpm": ddpm_loss}.get(name)
    norm = normalization(cfg)
    torch.manual_seed(int(cfg["project"]["seed"]) + acc.process_index)
    train = LLC4320Dataset(cfg, "train", train=True, norm=norm)
    val = LLC4320Dataset(cfg, "validation", train=False, norm=norm, max_days=s.get("validation_days"))
    model = build_baseline(cfg, name, norm)
    opt = torch.optim.AdamW(model.parameters(), lr=s["learning_rate"], weight_decay=s["weight_decay"])
    loader = _loader(train, s["batch_size"], cfg, True)
    sched_per_epoch = max(1, len(loader)) // acc.gradient_accumulation_steps
    sched = _cosine(opt, s["epochs"] * sched_per_epoch, s["warmup_steps"])
    model, opt, loader, sched = acc.prepare(model, opt, loader, sched)
    stage = f"baseline-{name}"
    out = work_path(cfg, "checkpoints", stage)
    start_epoch, best = 0, float("inf")
    if resume and (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        acc.unwrap_model(model).load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_epoch, best = state["epoch"] + 1, state["best"]
    prior_w = float(cfg["baselines"][name].get("prior_weight", 0.0))
    if acc.is_main_process:
        emit(
            "baseline_start",
            stage=stage,
            parameters=sum(p.numel() for p in acc.unwrap_model(model).parameters()),
            train_days=len(train),
            steps_per_epoch=len(loader),
        )

    def det_loss(net, batch):
        pred = net(batch["x"]).float()
        loss, parts = dense_loss(pred, batch["y"], cfg["loss"], PIXEL_KM)
        inner = getattr(acc.unwrap_model(net), "net", None)
        if prior_w > 0 and inner is not None and getattr(inner, "last_prior_cost", None) is not None:
            loss = loss + prior_w * inner.last_prior_cost
            parts["prior"] = float(inner.last_prior_cost)
        return loss, parts

    for epoch in range(start_epoch, s["epochs"]):
        model.train()
        began, losses = time.time(), []
        for step, batch in enumerate(loader):
            with acc.accumulate(model):
                if gen:
                    loss, parts = gen_loss(model, batch["x"], batch["y"]), {}
                else:
                    loss, parts = det_loss(model, batch)
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(loss))
            if step % cfg["training"]["log_every"] == 0 and acc.is_main_process:
                emit("train_step", stage=stage, epoch=epoch, step=step, loss=float(loss), **parts)
        score = None
        if (epoch + 1) % s["validate_every"] == 0 or epoch + 1 == s["epochs"]:
            net = acc.unwrap_model(model).eval()
            vals = []
            g = torch.Generator(device=acc.device).manual_seed(1234)
            for i in range(len(val)):
                b = val[i]
                xb, yb = b["x"][None].to(acc.device), b["y"][None].to(acc.device)
                if gen:
                    with torch.no_grad():
                        vals += [float(gen_loss(net, xb, yb, generator=g)) for _ in range(2)]
                else:
                    with torch.no_grad(), torch.autocast(device_type=acc.device.type, enabled=acc.device.type == "cuda"):
                        pred = net(xb).float()
                    vals.append(float(dense_loss(pred, yb, cfg["loss"], PIXEL_KM)[0]))
            score = float(np.mean(vals))
            if acc.is_main_process and score < best:
                best = score
                _save(out / "best.pt", net, {"epoch": epoch, "score": score, "stage": stage, "normalization": norm})
        if acc.is_main_process:
            _save(
                out / "last.pt",
                acc.unwrap_model(model),
                {"epoch": epoch, "best": best, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "stage": stage},
            )
            emit("epoch_done", stage=stage, epoch=epoch, seconds=time.time() - began, train_loss=float(np.mean(losses)), validation=score, best=best)
        acc.wait_for_everyone()
