"""Four-scale Attention U-Net velocity network for Conditional Flow Matching."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int, groups: int) -> int:
    g = min(groups, channels)
    while channels % g:
        g -= 1
    return g


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, groups: int, dropout: float = 0.0, emb: int | None = None):
        super().__init__()
        self.n1 = nn.GroupNorm(_groups(cin, groups), cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, padding_mode="reflect")
        self.n2 = nn.GroupNorm(_groups(cout, groups), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, padding_mode="reflect")
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.film = nn.Linear(emb, 2 * cout) if emb else None

    def forward(self, x, e=None):
        h = self.c1(F.silu(self.n1(x)))
        h = self.n2(h)
        if self.film is not None and e is not None:
            scale, shift = self.film(e)[:, :, None, None].chunk(2, 1)
            h = h * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return h + self.skip(x)


class Adapter(nn.Module):
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        hidden = max(8, channels // ratio)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.up = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.silu(self.down(x)))


class Attention(nn.Module):
    def __init__(self, channels: int, heads: int, groups: int):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels, groups), channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        t = self.norm(x).flatten(2).transpose(1, 2)
        out, _ = self.attn(t, t, t, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], 1)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base: int = 64,
        mults=(1, 2, 4, 8),
        blocks: int = 2,
        groups: int = 8,
        heads: int = 4,
        dropout: float = 0.08,
        adapters: bool = True,
        time_dim: int | None = 128,
    ):
        super().__init__()
        self.time_dim = time_dim
        emb = None
        if time_dim:
            emb = 4 * time_dim
            self.temb = nn.Sequential(nn.Linear(time_dim, emb), nn.SiLU(), nn.Linear(emb, emb))
        chans = [base * m for m in mults]
        self.stem = nn.Conv2d(in_channels, chans[0], 3, padding=1, padding_mode="reflect")
        self.down_blocks, self.downsample, self.down_adapters = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        skips = []
        c = chans[0]
        for level, ch in enumerate(chans):
            layer = nn.ModuleList()
            for _ in range(blocks):
                layer.append(ResBlock(c, ch, groups, dropout, emb))
                c = ch
            self.down_blocks.append(layer)
            self.down_adapters.append(Adapter(c) if adapters else nn.Identity())
            skips.append(c)
            if level < len(chans) - 1:
                self.downsample.append(nn.Conv2d(c, c, 3, stride=2, padding=1, padding_mode="reflect"))
        self.mid1 = ResBlock(c, c, groups, dropout, emb)
        self.mid_attn = Attention(c, heads, groups) if heads else nn.Identity()
        self.mid2 = ResBlock(c, c, groups, dropout, emb)
        self.mid_adapter = Adapter(c) if adapters else nn.Identity()
        self.up_blocks, self.upsample, self.up_adapters = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for level in reversed(range(len(chans))):
            ch = chans[level]
            layer = nn.ModuleList()
            for b in range(blocks):
                layer.append(ResBlock(c + (skips[level] if b == 0 else 0), ch, groups, dropout, emb))
                c = ch
            self.up_blocks.append(layer)
            self.up_adapters.append(Adapter(c) if adapters else nn.Identity())
            if level > 0:
                self.upsample.append(nn.Conv2d(c, c, 3, padding=1, padding_mode="reflect"))
        self.out_norm = nn.GroupNorm(_groups(c, groups), c)
        self.head = nn.Conv2d(c, out_channels, 3, padding=1, padding_mode="reflect")
        self.levels = len(chans)

    def forward(self, x, t=None):
        e = self.temb(time_embedding(t, self.time_dim)) if self.time_dim else None
        factor = 2 ** (self.levels - 1)
        h0, w0 = x.shape[-2:]
        pad_h, pad_w = (-h0) % factor, (-w0) % factor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        h = self.stem(x)
        skips = []
        for level, layer in enumerate(self.down_blocks):
            for block in layer:
                h = block(h, e)
            h = self.down_adapters[level](h)
            skips.append(h)
            if level < len(chans := self.down_blocks) - 1:
                h = self.downsample[level](h)
        h = self.mid_adapter(self.mid2(self.mid_attn(self.mid1(h, e)), e))
        for i, layer in enumerate(self.up_blocks):
            level = self.levels - 1 - i
            for b, block in enumerate(layer):
                h = block(torch.cat([h, skips[level]], 1) if b == 0 else h, e)
            h = self.up_adapters[i](h)
            if level > 0:
                h = self.upsample[i](F.interpolate(h, scale_factor=2, mode="nearest"))
        out = self.head(F.silu(self.out_norm(h)))
        return out[..., :h0, :w0]


class ZeroMeanResidual(nn.Module):
    """Enforce zero spatial mean residual per sample so domain mean comes from DUACS."""

    def __init__(self, net: UNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out - out.mean(dim=(-2, -1), keepdim=True)


def build_unet_baseline(cfg: dict, in_channels: int = 20) -> nn.Module:
    m = cfg["baselines"]["unet"]["model"]
    net = UNet(
        in_channels,
        1,
        m["base_channels"],
        tuple(m["channel_mults"]),
        m["blocks_per_level"],
        m["groups"],
        m["attention_heads"],
        m.get("dropout", 0.0),
        m.get("adapters", True),
        time_dim=None,
    )
    return ZeroMeanResidual(net) if m.get("zero_mean_residual", True) else net


def build_cfm_model(cfg: dict, cond_channels: int = 20) -> UNet:
    f = cfg["cfm"]
    return UNet(
        cond_channels + 1,
        1,
        f["base_channels"],
        tuple(f["channel_mults"]),
        f["blocks_per_level"],
        f["groups"],
        f["attention_heads"],
        f.get("dropout", 0.08),
        f.get("adapters", True),
        time_dim=f["time_dim"],
    )
