"""DUACS observation simulator: virtual 9-altimeter constellation and spatiotemporal OI (L=150 km, T=4 days)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .utils import DAY_NS, emit

A_DUACS = 3.337


@dataclass(frozen=True)
class Mission:
    name: str
    inclination_deg: float
    repeat_days: float
    revolutions: float
    noise_m: float


DEFAULT_MISSIONS = (
    Mission("s6a", 66.04, 9.9156, 127, 0.020),
    Mission("j3n", 66.04, 9.9156, 127, 0.030),
    Mission("s3a", 98.65, 27.0, 385, 0.020),
    Mission("s3b", 98.65, 27.0, 385, 0.020),
    Mission("al", 98.54, 35.0, 501.37, 0.025),
    Mission("c2n", 92.0, 369.0, 5344, 0.025),
    Mission("h2b", 99.34, 14.0, 193, 0.030),
    Mission("h2c", 66.0, 10.0, 128, 0.030),
    Mission("swon", 77.6, 20.86, 292, 0.020),
)


def missions_from_config(items) -> tuple[Mission, ...]:
    return tuple(Mission(**m) for m in items) if items else DEFAULT_MISSIONS


def _rectilinear(lat2: np.ndarray, lon2: np.ndarray):
    y, x = lat2[:, 0], lon2[0]
    iy, ix = np.argsort(y), np.argsort(x)
    return y[iy], x[ix], iy, ix


def _stencil(lat_axis, lon_axis, iy, ix, nx, lat, lon):
    i = np.clip(np.searchsorted(lat_axis, lat) - 1, 0, len(lat_axis) - 2)
    j = np.clip(np.searchsorted(lon_axis, lon) - 1, 0, len(lon_axis) - 2)
    fy = ((lat - lat_axis[i]) / (lat_axis[i + 1] - lat_axis[i])).clip(0, 1)
    fx = ((lon - lon_axis[j]) / (lon_axis[j + 1] - lon_axis[j])).clip(0, 1)
    corners = np.stack([iy[i] * nx + ix[j], iy[i] * nx + ix[j + 1], iy[i + 1] * nx + ix[j], iy[i + 1] * nx + ix[j + 1]], 1)
    weights = np.stack([(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx], 1)
    return corners.astype(np.int32), weights.astype(np.float32)


def upsample_coarse(coarse: np.ndarray, shape: tuple[int, int], stride: int) -> np.ndarray:
    c = torch.as_tensor(np.asarray(coarse, np.float32))
    lead = c.shape[:-2]
    h, w = c.shape[-2:]
    c = c.reshape(-1, 1, h, w)
    up = F.interpolate(c, size=((h - 1) * stride + 1, (w - 1) * stride + 1), mode="bilinear", align_corners=True)
    up = F.pad(up, (0, shape[1] - up.shape[-1], 0, shape[0] - up.shape[-2]), mode="replicate")
    return up.reshape(*lead, *shape).numpy()


def ground_track(m: Mission, t_s: np.ndarray, u0: float, lon0_deg: float) -> tuple[np.ndarray, np.ndarray]:
    period = m.repeat_days * 86400.0 / m.revolutions
    nodal_rate = 2 * np.pi * round(m.repeat_days) / (m.repeat_days * 86400.0)
    inc = np.deg2rad(m.inclination_deg)
    u = u0 + 2 * np.pi * t_s / period
    lat = np.arcsin(np.sin(inc) * np.sin(u))
    lon = np.deg2rad(lon0_deg) + np.arctan2(np.cos(inc) * np.sin(u), np.cos(u)) - nodal_rate * t_s
    return np.rad2deg(lat), np.mod(np.rad2deg(lon), 360.0)


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def _gaussian_kernel(filter_km: float, spacing_km: float) -> np.ndarray:
    sigma = filter_km * np.sqrt(np.log(2) / 2) / np.pi / spacing_km
    half = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    return k / k.sum()


class NadirSimulator:
    def __init__(self, lat: np.ndarray, lon: np.ndarray, cfg: dict) -> None:
        self.lat, self.lon = lat, lon
        self.axes = _rectilinear(lat, lon)
        self.nx = lat.shape[1]
        self.box = (float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max()))
        self.cfg = cfg
        self.missions = missions_from_config(cfg.get("missions"))
        self.ref = (float(lat.mean()), float(lon.mean()))

    def draw_phases(self, rng: np.random.Generator) -> list[tuple[float, float]]:
        return [(float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0, 360))) for _ in self.missions]

    def passes(self, t0_ns: int, t1_ns: int, phases, epoch_ns: int) -> list[dict]:
        la, lo, iy, ix = self.axes
        lat0, lat1, lon0, lon1 = self.box
        min_pts = int(self.cfg.get("min_pass_points", 30))
        out = []
        for day0 in range(int(t0_ns), int(t1_ns), DAY_NS):
            t_s = (day0 - epoch_ns) / 1e9 + np.arange(0, min(DAY_NS, int(t1_ns) - day0) // 10**9, dtype=np.float64)
            for mi, (m, (u0, l0)) in enumerate(zip(self.missions, phases)):
                lat, lon = ground_track(m, t_s, u0, l0)
                inside = (lat > lat0) & (lat < lat1) & (lon > lon0) & (lon < lon1)
                for a, b in _segments(inside):
                    if b - a < min_pts:
                        continue
                    corners, w = _stencil(la, lo, iy, ix, self.nx, lat[a:b], lon[a:b])
                    out.append(
                        {
                            "mission": mi,
                            "time_ns": int(epoch_ns + round(t_s[(a + b) // 2] * 1e9)),
                            "lat": lat[a:b],
                            "lon": lon[a:b],
                            "corners": corners,
                            "w": w,
                        }
                    )
        return sorted(out, key=lambda p: p["time_ns"])

    def observe(self, passes: list[dict], sampler, rng: np.random.Generator) -> dict:
        c = self.cfg
        filter_km = float(c.get("filter_km", 65.0))
        step = int(c.get("decimation", 2))
        noise_scale = float(c.get("noise_scale", 1.0))
        kx = 111.2 * np.cos(np.deg2rad(self.ref[0]))
        cols = {k: [] for k in ("x", "y", "t", "value", "var", "pass", "mission")}
        for p_id, p in enumerate(passes):
            m = self.missions[p["mission"]]
            field = sampler.at(p["time_ns"])
            value = np.sum(field[p["corners"].astype(np.int64)] * p["w"], 1).astype(np.float64)
            n = len(value)
            dist = np.r_[0, np.cumsum(np.hypot(np.diff(p["lon"]) * kx, np.diff(p["lat"]) * 111.2))]
            kernel = _gaussian_kernel(filter_km, float(np.median(np.diff(dist))))
            noise_gain = float(np.sum(kernel**2))
            lwe = rng.normal(0, c.get("pass_bias_m", 0.01)) + rng.normal(0, c.get("pass_tilt_m_per_1000km", 0.01)) * (
                dist - dist.mean()
            ) / 1000
            noisy = value + lwe + rng.normal(0, m.noise_m * noise_scale, n)
            pad = len(kernel) // 2
            smooth = np.convolve(np.pad(noisy, pad, mode="reflect"), kernel, mode="valid")[:n]
            keep = np.arange(pad // 2, n - pad // 2, step) if n > pad else np.arange(0, n, step)
            cols["x"].append((p["lon"][keep] - self.ref[1]) * kx)
            cols["y"].append((p["lat"][keep] - self.ref[0]) * 111.2)
            cols["t"].append(np.full(len(keep), p["time_ns"], np.int64))
            cols["value"].append(smooth[keep])
            cols["var"].append(np.full(len(keep), (m.noise_m * noise_scale) ** 2 * noise_gain))
            cols["pass"].append(np.full(len(keep), p_id, np.int64))
            cols["mission"].append(np.full(len(keep), p["mission"], np.int64))
        if not cols["x"]:
            return {k: np.zeros(0) for k in cols}
        return {k: np.concatenate(v) for k, v in cols.items()}

    def state(self, phases) -> dict:
        return {"missions": [asdict(m) for m in self.missions], "phases_u0_lon0": phases, "config": self.cfg}


def duacs_correlation(r: torch.Tensor) -> torch.Tensor:
    ar = A_DUACS * r
    return (1 + ar + ar**2 / 6 - ar**3 / 6) * torch.exp(-ar)


class NadirOIMapper:
    """Daily DUACS-like maps from along-track observations via spatiotemporal OI."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, cfg: dict, signal_var_m2: float, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.L = float(cfg["length_km"])
        self.T = float(cfg["time_days"])
        self.window = float(cfg.get("window_days", 1.5 * self.T))
        self.var = float(signal_var_m2)
        self.pass_var = float(cfg.get("oi_pass_sigma_m", cfg.get("pass_bias_m", 0.01))) ** 2
        self.max_obs = int(cfg.get("max_obs", 7000))
        self.stride = int(cfg.get("grid_stride", 4))
        self.err_stride = int(cfg.get("err_stride", 16))
        ref = (float(lat.mean()), float(lon.mean()))
        kx = 111.2 * np.cos(np.deg2rad(ref[0]))
        self.shape = lat.shape

        def grid(stride):
            sub_lat, sub_lon = lat[::stride, ::stride], lon[::stride, ::stride]
            pos = np.stack([(sub_lon.ravel() - ref[1]) * kx, (sub_lat.ravel() - ref[0]) * 111.2], 1)
            return torch.tensor(pos, dtype=torch.float32, device=self.device), sub_lat.shape

        self.xg, self.gshape = grid(self.stride)
        self.xe, self.eshape = grid(self.err_stride)

    def _cov(self, xa, ta, xb, tb) -> torch.Tensor:
        r = torch.cdist(xa, xb) / self.L
        dt = (ta[:, None] - tb[None, :]) / self.T
        return self.var * duacs_correlation(r) * torch.exp(-(dt**2))

    def map_day(self, obs: dict, t_ns: int, rng: np.random.Generator | None = None):
        dt_days = (obs["t"] - int(t_ns)) / DAY_NS
        sel = np.flatnonzero(np.abs(dt_days) <= self.window)
        if len(sel) > self.max_obs:
            sel = sel[np.argsort(np.abs(dt_days[sel]), kind="stable")[: self.max_obs]]
        if len(sel) == 0:
            z = np.zeros(self.shape, np.float32)
            return z, np.full(self.shape, np.sqrt(self.var), np.float32), 0
        dev = self.device
        xo = torch.tensor(np.stack([obs["x"][sel], obs["y"][sel]], 1), dtype=torch.float32, device=dev)
        to = torch.tensor(dt_days[sel], dtype=torch.float32, device=dev)
        y = torch.tensor(obs["value"][sel], dtype=torch.float64, device=dev)
        p = torch.tensor(obs["pass"][sel], device=dev)
        c = self._cov(xo, to, xo, to).double()
        c += self.pass_var * (p[:, None] == p[None, :]).double()
        c.diagonal().add_(torch.tensor(obs["var"][sel], dtype=torch.float64, device=dev))
        chol = torch.linalg.cholesky(c)
        del c
        alpha = torch.cholesky_solve(y[:, None], chol)[:, 0].float()
        zero_t = torch.zeros(1, device=dev)
        field = (self._cov(self.xg, zero_t.expand(len(self.xg)), xo, to) @ alpha).cpu().numpy().reshape(self.gshape)
        k_et = self._cov(self.xe, zero_t.expand(len(self.xe)), xo, to).double().T
        v = torch.linalg.solve_triangular(chol, k_et, upper=False)
        err = torch.sqrt(torch.clamp(self.var - (v**2).sum(0), min=1e-8)).float().cpu().numpy().reshape(self.eshape)
        del chol, v, k_et
        return (
            upsample_coarse(field, self.shape, self.stride).astype(np.float32),
            upsample_coarse(err, self.shape, self.err_stride).astype(np.float32),
            len(sel),
        )


def emulate_split(
    sim: NadirSimulator,
    mapper: NadirOIMapper,
    sampler,
    times: np.ndarray,
    split_code: int,
    phases,
    epoch_ns: int,
    rng: np.random.Generator,
    progress: str = "",
) -> tuple[np.ndarray, np.ndarray, dict]:
    w = int(np.ceil(mapper.window * DAY_NS))
    t0 = max(int(times[0]) - w, int(sampler.p.hourly_time[0]))
    t1 = min(int(times[-1]) + w + DAY_NS, int(sampler.p.hourly_time[-1]) - 3_600_000_000_000)
    passes = [p for p in sim.passes(t0, t1, phases, epoch_ns) if sampler.split_code_at(p["time_ns"]) == split_code]
    obs = sim.observe(passes, sampler, rng)
    duacs = np.empty((len(times), *mapper.shape), np.float32)
    err = np.empty_like(duacs)
    counts = []
    for k, t in enumerate(times):
        duacs[k], err[k], n = mapper.map_day(obs, int(t), rng)
        counts.append(n)
        if k % 50 == 0:
            emit("nadir_emulator_progress", split=progress, day=k, days=len(times), obs=n)
    if mapper.device.type == "cuda":
        torch.cuda.empty_cache()
    info = {
        "passes": len(passes),
        "passes_per_day": len(passes) / max((t1 - t0) / DAY_NS, 1),
        "obs_per_map_mean": float(np.mean(counts)),
        "obs_per_map_max": int(np.max(counts)),
    }
    return duacs, err, info
