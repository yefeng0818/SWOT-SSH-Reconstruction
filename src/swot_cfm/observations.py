"""SWOT wide-swath observations: plane removal, gridding, and multi-bin temporal innovation encoding."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import DAY_NS, SWOT_CENTRE_PIXEL, SWOT_PIXEL_KM


def plane_design(line: np.ndarray, pixel: np.ndarray, quadratic_cross: bool = False) -> np.ndarray:
    along = (np.asarray(line, np.float64) - np.mean(line)) * SWOT_PIXEL_KM / 100.0
    cross = (np.asarray(pixel, np.float64) - SWOT_CENTRE_PIXEL) * SWOT_PIXEL_KM / 100.0
    columns = [np.ones_like(along), along, cross]
    if quadratic_cross:
        columns.append(cross * cross)
    return np.stack(columns, axis=1)


def remove_plane(values: np.ndarray, line: np.ndarray, pixel: np.ndarray, quadratic_cross: bool = False) -> np.ndarray:
    design = plane_design(line, pixel, quadratic_cross)
    coef, *_ = np.linalg.lstsq(design, values.astype(np.float64), rcond=None)
    return (values - design @ coef).astype(np.float32)


def sample_stencil(
    fields: np.ndarray, node: np.ndarray, wtime: np.ndarray, corners: np.ndarray, wspace: np.ndarray
) -> np.ndarray:
    flat = fields.reshape(fields.shape[0], -1)
    out = np.zeros(len(corners), np.float64)
    for t in range(2):
        out += wtime[:, t] * np.sum(flat[node[:, t][:, None], corners] * wspace, axis=1)
    return out


@dataclass
class GranuleStore:
    start: np.ndarray
    count: np.ndarray
    time: np.ndarray
    fold: np.ndarray
    role: np.ndarray
    corners: np.ndarray
    wspace: np.ndarray
    node: np.ndarray
    wtime: np.ndarray
    value: np.ndarray
    innov: np.ndarray
    line: np.ndarray
    pixel: np.ndarray
    sigma: np.ndarray

    def slice(self, g: int) -> slice:
        return slice(int(self.start[g]), int(self.start[g] + self.count[g]))

    @classmethod
    def from_h5(cls, group) -> "GranuleStore":
        keys = ("start", "count", "time", "fold", "role")
        obs = ("corners", "wspace", "node", "wtime", "value", "innov", "line", "pixel", "sigma")
        return cls(
            **{k: np.asarray(group[f"granules/{k}"][:]) for k in keys},
            **{k: np.asarray(group[f"obs/{k}"][:]) for k in obs},
        )


def write_store(group, store_arrays: dict) -> None:
    for key in ("start", "count", "time", "fold", "role"):
        group.create_dataset(f"granules/{key}", data=store_arrays[key])
    for key in ("corners", "wspace", "node", "wtime", "value", "innov", "line", "pixel", "sigma"):
        group.create_dataset(f"obs/{key}", data=store_arrays[key], chunks=True, compression="lzf")


class ObservationEncoder:
    """Encode plane-removed SWOT-DUACS innovations into time bins, coverage masks, and nearest-pass features."""

    def __init__(
        self,
        shape: tuple[int, int],
        bin_edges_days,
        window_days: float,
        innovation_scale_m: float,
        taper_px: float = 0.0,
        erode_max_px: float = 0.0,
    ) -> None:
        self.shape = tuple(shape)
        self.size = int(np.prod(shape))
        self.edges = np.asarray(bin_edges_days, np.float64)
        self.window = float(window_days)
        self.scale = float(innovation_scale_m)
        self.taper_px = float(taper_px)
        self.erode_max_px = float(erode_max_px)

    def _taper(self, covered: np.ndarray, rng) -> np.ndarray:
        from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt

        mask = binary_closing(np.pad(covered.reshape(self.shape), 2, mode="edge"), iterations=1)[2:-2, 2:-2]
        mask = binary_fill_holes(mask)
        r = int(np.ceil(self.taper_px + self.erode_max_px)) + 2
        dist = distance_transform_edt(np.pad(mask, r, mode="edge"))[r:-r, r:-r].ravel()
        erode = float(rng.uniform(0.0, self.erode_max_px)) if rng is not None and self.erode_max_px > 0 else 0.0
        return np.clip((dist - erode) / self.taper_px, 0.0, 1.0)

    @property
    def channels(self) -> int:
        return 2 * (len(self.edges) - 1) + 3

    def select(self, store: GranuleStore, node_time: int, allowed: np.ndarray) -> np.ndarray:
        dt = (store.time - int(node_time)) / DAY_NS
        return np.flatnonzero(allowed & (np.abs(dt) <= self.window))

    def encode(self, store: GranuleStore, node_time: int, granules: np.ndarray, rng=None, return_footprint: bool = False):
        bins = len(self.edges) - 1
        num = np.zeros((bins, self.size))
        den = np.zeros((bins, self.size))
        age = np.full(self.size, np.inf)
        nearest_val = np.zeros(self.size)
        nearest_dt = np.zeros(self.size)
        footprint = np.zeros(self.size, bool)
        dts = (store.time[granules] - int(node_time)) / DAY_NS
        for g, dt in sorted(zip(granules, dts), key=lambda item: -abs(item[1])):
            sl = store.slice(int(g))
            corners = store.corners[sl].astype(np.int64).ravel()
            w = store.wspace[sl].astype(np.float64)
            vals = (w * store.innov[sl][:, None].astype(np.float64)).ravel()
            g_num = np.bincount(corners, weights=vals, minlength=self.size)
            g_den = np.bincount(corners, weights=w.ravel(), minlength=self.size)
            covered = g_den > 1e-3
            with np.errstate(invalid="ignore", divide="ignore"):
                val = np.where(covered, g_num / np.maximum(g_den, 1e-12), 0.0)
            if self.taper_px > 0:
                t = self._taper(covered, rng)
                g_num, g_den = g_num * t, g_den * t
                covered = covered & (t > 0)
                cand_age = abs(dt) + (1.0 - t) * (self.window - abs(dt))
            else:
                t = covered.astype(np.float64)
                cand_age = np.full(self.size, abs(dt))
            footprint |= covered
            b = int(np.clip(np.searchsorted(self.edges, dt, side="right") - 1, 0, bins - 1))
            num[b] += g_num
            den[b] += g_den
            nearer = covered & (cand_age <= age)
            age[nearer] = cand_age[nearer]
            nearest_val[nearer] = val[nearer] * t[nearer]
            nearest_dt[nearer] = dt * t[nearer]
        with np.errstate(invalid="ignore", divide="ignore"):
            innov = np.where(den > 1e-3, num / np.maximum(den, 1e-12), 0.0) / self.scale
        coverage = np.clip(den, 0.0, 1.0)
        age_norm = np.where(np.isfinite(age), age, self.window) / self.window
        channels = np.concatenate(
            [
                innov,
                coverage,
                age_norm[None],
                (nearest_val / self.scale)[None],
                (nearest_dt / self.window)[None],
            ],
            axis=0,
        )
        channels = channels.reshape(self.channels, *self.shape).astype(np.float32)
        return (channels, footprint.reshape(self.shape)) if return_footprint else channels

    @staticmethod
    def age_map(channels: np.ndarray, window_days: float) -> np.ndarray:
        return channels[-3] * window_days
