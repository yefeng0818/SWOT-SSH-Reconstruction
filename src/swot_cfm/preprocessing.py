"""Ocean simulation preprocessing: tidal removal, large-scale surface filtering, and 25-hr balanced SSH target."""
from __future__ import annotations

import json
from dataclasses import dataclass

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

from .utils import DAY_NS, HOUR_NS, SPLIT_CODES, emit


def tidal_design(times_ns: np.ndarray, origin_ns: int, speeds_deg_per_hour) -> np.ndarray:
    hours = (np.asarray(times_ns, np.int64) - int(origin_ns)).astype(np.float64) / HOUR_NS
    phase = hours[:, None] * np.asarray(speeds_deg_per_hour, np.float64)[None] * np.pi / 180
    return np.concatenate((np.sin(phase), np.cos(phase)), axis=1)


def surface_design(lat: np.ndarray, lon: np.ndarray, order: int = 2) -> np.ndarray:
    y = (lat - lat.mean()) * 111.2 / 100.0
    x = (lon - lon.mean()) * 111.2 * np.cos(np.deg2rad(lat.mean())) / 100.0
    columns = [np.ones_like(x), x, y]
    if order >= 2:
        columns += [x * x, x * y, y * y]
    return np.stack([c.ravel() for c in columns], axis=1)


@dataclass
class BalancedProduct:
    split_code_hourly: np.ndarray
    hourly_time: np.ndarray
    surface_hf: np.ndarray
    surface_basis: np.ndarray
    tide_speeds: list
    tide_origin: int
    tide_coefficients: np.ndarray
    daily_time: dict
    daily_target: dict
    diagnostics: dict


def compute_balanced(
    llc_h5: str,
    basis_h5: str,
    *,
    cutoff_days: float = 20.0,
    order: int = 2,
    boxcar_hours: int = 25,
    block: int = 192,
) -> BalancedProduct:
    with h5py.File(basis_h5, "r") as b:
        diag = json.loads(b.attrs["tidal_diagnostics"])
        if diag["fit_split"] != "LLC_train_only":
            raise ValueError("Tidal coefficients must be fitted on LLC train only")
        speeds, origin = diag["constituent_speeds_deg_per_hour"], int(b.attrs["tidal_origin_ns"])
        coefficients = np.asarray(b["tidal_coefficients_m"][:], np.float32)
    with h5py.File(llc_h5, "r") as h:
        times = np.asarray(h["time"][:], np.int64)
        codes = np.asarray(h["split_code"][:])
        lat = np.asarray(h["yc_hr"][:], np.float64)
        lon = np.asarray(h["xc_hr"][:], np.float64)
        if np.any(np.diff(times) != HOUR_NS):
            raise ValueError("LLC hourly record must be contiguous")
        ny, nx = lat.shape
        basis = surface_design(lat, lon, order)
        sub = np.zeros((ny, nx), bool)
        sub[::4, ::4] = True
        projection = np.linalg.pinv(basis[sub.ravel()])
        coef_tide = coefficients.reshape(len(coefficients), -1)
        half = boxcar_hours // 2
        nodes = {}
        for name, code in SPLIT_CODES.items():
            midnights = times[(times % DAY_NS == 0) & (codes == code)]
            ok = []
            for t in midnights:
                i = int((t - times[0]) // HOUR_NS)
                lo, hi = i - half, i + half
                if lo >= 0 and hi < len(times) and np.all(codes[lo : hi + 1] == code):
                    ok.append(i)
            nodes[name] = np.asarray(ok, np.int64)
        sums = {name: np.zeros((len(v), ny * nx), np.float64) for name, v in nodes.items()}
        surface = np.empty((len(times), basis.shape[1]))
        for start in range(0, len(times), block):
            stop = min(start + block, len(times))
            raw = np.asarray(h["sla_hr"][start:stop], np.float32).reshape(stop - start, -1)
            tide = (tidal_design(times[start:stop], origin, speeds) @ coef_tide).astype(np.float32)
            detided = raw - tide
            surface[start:stop] = detided[:, sub.ravel()].astype(np.float64) @ projection.T
            for name, idx in nodes.items():
                for k, centre in enumerate(idx):
                    lo, hi = max(centre - half, start), min(centre + half, stop - 1)
                    if lo <= hi:
                        sums[name][k] += detided[lo - start : hi - start + 1].sum(0, dtype=np.float64)
            if start // block % 10 == 0:
                emit("balanced_progress", frames=stop, total=len(times))
    window = int(round(cutoff_days * 24)) | 1
    surface_hf = np.zeros_like(surface)
    edges = np.flatnonzero(np.diff(codes.astype(np.int64)) != 0) + 1
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(codes)]):
        seg = surface[lo:hi]
        surface_hf[lo:hi] = seg - uniform_filter1d(seg, size=min(window, max(1, (hi - lo) | 1)), axis=0, mode="nearest")
    daily_time, daily_target = {}, {}
    for name, idx in nodes.items():
        mean_hf = (
            np.stack([surface_hf[c - half : c + half + 1].mean(0) for c in idx])
            if len(idx)
            else np.zeros((0, basis.shape[1]))
        )
        fields = sums[name] / boxcar_hours - mean_hf @ basis.T
        daily_time[name] = times[idx]
        daily_target[name] = fields.reshape(len(idx), ny, nx).astype(np.float32)
    hf_field_rms = float(np.sqrt(np.mean((surface_hf @ basis[::97].T) ** 2)))
    diagnostics = {
        "cutoff_days": cutoff_days,
        "surface_order": order,
        "boxcar_hours": boxcar_hours,
        "daily_nodes": {k: int(len(v)) for k, v in nodes.items()},
        "removed_hf_surface_rms_m": hf_field_rms,
        "tide_definition": "frozen train-only 8-constituent coherent fit",
    }
    return BalancedProduct(
        codes,
        times,
        surface_hf.astype(np.float32),
        basis.astype(np.float32),
        speeds,
        origin,
        coefficients,
        daily_time,
        daily_target,
        diagnostics,
    )


class HourlyCorrectedSampler:
    """Instantaneous detided, large-scale-corrected LLC4320 SSH sampler."""

    def __init__(self, llc_h5: str, product: BalancedProduct) -> None:
        self.h = h5py.File(llc_h5, "r")
        self.p = product
        self.coef_tide = product.tide_coefficients.reshape(len(product.tide_coefficients), -1)
        self.cache: dict[int, np.ndarray] = {}

    def frame(self, index: int) -> np.ndarray:
        if index not in self.cache:
            if len(self.cache) > 16:
                self.cache.pop(next(iter(self.cache)))
            raw = np.asarray(self.h["sla_hr"][index], np.float32).ravel()
            t = self.p.hourly_time[index : index + 1]
            tide = (tidal_design(t, self.p.tide_origin, self.p.tide_speeds) @ self.coef_tide)[0]
            dac = self.p.surface_basis @ self.p.surface_hf[index]
            self.cache[index] = (raw - tide - dac).astype(np.float32)
        return self.cache[index]

    def at(self, time_ns: int) -> np.ndarray:
        i = int((int(time_ns) - int(self.p.hourly_time[0])) // HOUR_NS)
        if i < 0 or i + 1 >= len(self.p.hourly_time):
            raise ValueError("time outside LLC record")
        w = (int(time_ns) - int(self.p.hourly_time[i])) / HOUR_NS
        return (1 - w) * self.frame(i) + w * self.frame(i + 1)

    def split_code_at(self, time_ns: int) -> int:
        i = int((int(time_ns) - int(self.p.hourly_time[0])) // HOUR_NS)
        if i < 0 or i + 1 >= len(self.p.hourly_time):
            return -1
        a, b = self.p.split_code_hourly[i], self.p.split_code_hourly[i + 1]
        return int(a) if a == b else -1

    def close(self) -> None:
        self.h.close()
