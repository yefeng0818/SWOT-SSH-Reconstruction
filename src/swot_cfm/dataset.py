"""Datasets for LLC4320 OSSE training/evaluation and real DUACS-SWOT zero-shot transfer.

Conditioning tensor c (20 channels):
  - 5 daily DUACS maps (d-2..d+2) / sla_std
  - 1 DUACS formal error: err_sla / err_mean - 1
  - 13 SWOT observation channels (5 innovation bins, 5 coverage masks, 3 nearest-SWOT features)
  - 1 Coriolis parameter f / 1e-4
Normalized residual target: r = (balanced_SSH - DUACS(d)) / residual_scale.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .observations import GranuleStore, ObservationEncoder, plane_design
from .utils import DAY_NS, REAL_ROLES, coriolis, work_path, write_json

INPUT_ROLES = {
    "train": ("train",),
    "validation_2025_h1": ("train", "validation_2025_h1"),
    "test_2025_h2": ("train", "validation_2025_h1", "test_2025_h2"),
    "heldout_cal": ("train", "validation_2025_h1", "heldout_cal"),
    "oneday_2023": ("oneday_2023",),
}
ROLE_CACHE = {"oneday_2023": "real_1day_cache.h5"}
PIXEL_KM = 1.29


def llc_exclusion(k: int, r: int, folds: int, train: bool) -> int:
    return (k + r) % folds if train else k % folds


def normalization(cfg: dict) -> dict:
    path = work_path(cfg, "normalization.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with h5py.File(work_path(cfg, "llc_cache.h5"), "r") as h:
        llc = json.loads(h.attrs["stats"])
    with h5py.File(work_path(cfg, "real_cache.h5"), "r") as h:
        real = json.loads(h.attrs["stats"])
    norm = {
        "sla_std_m": real["sla_std_m"],
        "innovation_scale_m": llc["innovation_std_m"],
        "residual_scale_m": llc["residual_std_m"],
        "err_mean_m": llc["err_mean_m"],
        "real_innovation_std_m": real["innovation_std_m"],
        "real_err_mean_m": real["err_mean_m"],
    }
    write_json(path, norm)
    return norm


def make_encoder(cfg: dict, shape, norm: dict) -> ObservationEncoder:
    o = cfg["observations"]
    return ObservationEncoder(
        shape,
        o["bin_edges_days"],
        o["window_days"],
        norm["innovation_scale_m"],
        taper_px=float(o.get("edge_taper_km", 0.0)) / PIXEL_KM,
        erode_max_px=float(o.get("edge_erode_max_km", 0.0)) / PIXEL_KM,
    )


def _contexts(times: np.ndarray, offsets=(-2, -1, 0, 1, 2)) -> np.ndarray:
    ok = []
    lookup = {int(t): i for i, t in enumerate(times)}
    for i, t in enumerate(times):
        if all(int(t) + o * DAY_NS in lookup for o in offsets):
            ok.append(i)
    return np.asarray(ok, np.int64)


class _Base(Dataset):
    def _inputs(self, k: int, granules: np.ndarray, duacs=None, err=None, store=None, rng=None):
        duacs = self.duacs if duacs is None else duacs
        err = self.err if err is None else err
        store = self.store if store is None else store
        t = int(self.times[k])
        idx = [self.lookup[t + o * DAY_NS] for o in (-2, -1, 0, 1, 2)]
        ctx = duacs[idx].astype(np.float32) / self.norm["sla_std_m"]
        err_k = err[k : k + 1].astype(np.float32) / self.norm["err_mean_m"] - 1.0
        obs = self.encoder.encode(store, t, granules, rng=rng)
        return np.concatenate([ctx, err_k, obs, self.coriolis[None]], 0).astype(np.float32)


class LLC4320Dataset(_Base):
    """LLC4320 OSSE dataset for Conditional Flow Matching training and evaluation."""

    def __init__(self, cfg: dict, split: str, train: bool = False, norm: dict | None = None, max_days: int | None = None):
        self.cfg, self.split, self.train = cfg, split, train
        self.norm = norm or normalization(cfg)
        with h5py.File(work_path(cfg, "llc_cache.h5"), "r") as h:
            g = h[split]
            self.times = np.asarray(g["daily/time"][:], np.int64)
            self.target = np.asarray(g["daily/target"][:], np.float32)
            names = sorted((k for k in g.keys() if k.startswith("r") and k[1:].isdigit()), key=lambda s: int(s[1:]))
            if not train:
                names = names[:1]
            self.realizations = [
                (
                    np.asarray(g[f"{n}/daily/duacs"][:]),
                    np.asarray(g[f"{n}/daily/err"][:]),
                    GranuleStore.from_h5(g[n]),
                )
                for n in names
            ]
            self.valid = [
                np.asarray(g[f"{n}/daily/valid"][:], bool) if f"{n}/daily/valid" in g else np.ones(len(self.times), bool)
                for n in names
            ]
            self.oneday = [r for r, n in enumerate(names) if g[n].attrs.get("kind", "") == "oneday"]
            self.lat = np.asarray(h["static/lat"][:], np.float64)
            self.lon = np.asarray(h["static/lon"][:], np.float64)
        self.duacs, self.err, self.store = self.realizations[0]
        self.lookup = {int(t): i for i, t in enumerate(self.times)}
        self.coriolis = (coriolis(self.lat) / 1e-4).astype(np.float32)
        self.encoder = make_encoder(cfg, self.lat.shape, self.norm)
        self.samples = _contexts(self.times)
        if max_days is not None:
            self.samples = self.samples[np.linspace(0, len(self.samples) - 1, min(max_days, len(self.samples))).astype(int)]
        self.regular = [r for r in range(len(self.realizations)) if r not in self.oneday]
        self.oneday_p = float(cfg["observations"].get("oneday_sample_probability", 0.5))
        self.drop_p = float(cfg["observations"]["fold_drop_probability"]) if train else 0.0
        self.flips = bool(cfg["observations"].get("augment_flips", False)) and train
        self.folds = int(cfg["observations"]["folds"])

    def __len__(self) -> int:
        return len(self.samples)

    def _draw_realization(self, k: int) -> int:
        oneday = [r for r in self.oneday if self.valid[r][k]]
        if oneday and torch.rand(()) < self.oneday_p:
            return oneday[int(torch.randint(len(oneday), ()))]
        return self.regular[int(torch.randint(len(self.regular), ()))]

    def granules_for(self, k: int, exclude_fold: int | None, store=None) -> np.ndarray:
        store = self.store if store is None else store
        allowed = np.ones(len(store.start), bool) if exclude_fold is None else store.fold != exclude_fold
        return self.encoder.select(store, int(self.times[k]), allowed)

    def __getitem__(self, i: int) -> dict:
        k = int(self.samples[i])
        r = self._draw_realization(k) if self.train else 0
        duacs, err, store = self.realizations[r]
        exclude = None
        if self.train and torch.rand(()) < self.drop_p:
            exclude = llc_exclusion(k, r, self.folds, True)
        elif not self.train:
            exclude = llc_exclusion(k, r, self.folds, False)
        rng = np.random.default_rng(int(torch.randint(2**31 - 1, ()))) if self.train else None
        x = self._inputs(k, self.granules_for(k, exclude, store), duacs, err, store, rng=rng)
        base = duacs[k].astype(np.float32)
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(((self.target[k] - base) / self.norm["residual_scale_m"])[None].astype(np.float32))
        if self.flips:
            for dim in (-1, -2):
                if torch.rand(()) < 0.5:
                    x, y = x.flip(dim), y.flip(dim)
        return {
            "x": x.contiguous(),
            "y": y.contiguous(),
            "index": torch.tensor(k),
            "realization": torch.tensor(r),
            "excluded_fold": torch.tensor(-1 if exclude is None else exclude),
        }


class SWOTPassDataset(_Base):
    """Real DUACS-SWOT dataset for zero-shot daily reconstruction and withheld-pass validation."""

    def __init__(self, cfg: dict, role: str, norm: dict | None = None, max_granules: int | None = None):
        if role not in REAL_ROLES:
            raise ValueError(role)
        self.cfg, self.role = cfg, role
        self.norm = norm or normalization(cfg)
        with h5py.File(work_path(cfg, ROLE_CACHE.get(role, "real_cache.h5")), "r") as h:
            self.times = np.asarray(h["daily/time"][:], np.int64)
            self.duacs = np.asarray(h["daily/duacs"][:], np.float32)
            self.err = np.asarray(h["daily/err"][:])
            self.store = GranuleStore.from_h5(h)
            self.lat = np.asarray(h["static/lat"][:], np.float64)
            self.lon = np.asarray(h["static/lon"][:], np.float64)
        self.lookup = {int(t): i for i, t in enumerate(self.times)}
        self.coriolis = (coriolis(self.lat) / 1e-4).astype(np.float32)
        self.encoder = make_encoder(cfg, self.lat.shape, self.norm)
        role_code = REAL_ROLES.index(role)
        self.input_role_codes = np.array([REAL_ROLES.index(r) for r in INPUT_ROLES[role]])
        context_ok = set(_contexts(self.times).tolist())
        targets = []
        for g in np.flatnonzero(self.store.role == role_code):
            node = self.store.node[self.store.slice(int(g))]
            n0 = int(node[:, 0].min())
            if np.all((node[:, 0] == n0) & (node[:, 1] == n0 + 1)) and n0 in context_ok and n0 + 1 in context_ok:
                targets.append((int(g), n0))
        if max_granules is not None:
            targets = targets[:max_granules]
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def input_granules(self, node_time: int, exclude_fold: int, exclude_granule: int) -> np.ndarray:
        allowed = np.isin(self.store.role, self.input_role_codes) & (self.store.fold != exclude_fold)
        allowed[exclude_granule] = False
        return self.encoder.select(self.store, node_time, allowed)

    def __getitem__(self, i: int) -> dict:
        g, n0 = self.targets[i]
        fold = int(self.store.fold[g])
        xs = [self._inputs(n, self.input_granules(int(self.times[n]), fold, g)) for n in (n0, n0 + 1)]
        sl = self.store.slice(g)
        line, pixel = self.store.line[sl], self.store.pixel[sl]
        t_obs = int(self.store.time[g])
        age_channels = self.encoder.encode(self.store, t_obs, self.input_granules(t_obs, fold, g))
        age_map = ObservationEncoder.age_map(age_channels, self.encoder.window).ravel()
        corners = self.store.corners[sl].astype(np.int64)
        wspace = self.store.wspace[sl]
        age = np.sum(age_map[corners] * wspace, 1)
        return {
            "x": torch.from_numpy(np.stack(xs)),
            "duacs": torch.from_numpy(self.duacs[[n0, n0 + 1]]),
            "corners": torch.from_numpy(corners),
            "wspace": torch.from_numpy(wspace.astype(np.float32)),
            "wtime": torch.from_numpy(self.store.wtime[sl].astype(np.float32)),
            "value": torch.from_numpy(self.store.value[sl].astype(np.float32)),
            "innov": torch.from_numpy(self.store.innov[sl].astype(np.float32)),
            "sigma": torch.from_numpy(self.store.sigma[sl].astype(np.float32)),
            "design": torch.from_numpy(plane_design(line, pixel).astype(np.float32)),
            "line": torch.from_numpy(line.astype(np.int64)),
            "pixel": torch.from_numpy(pixel.astype(np.int64)),
            "age": torch.from_numpy(age.astype(np.float32)),
            "granule": torch.tensor(g),
            "node0": torch.tensor(n0),
        }
