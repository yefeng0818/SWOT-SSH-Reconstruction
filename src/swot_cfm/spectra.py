"""Along-track wavenumber spectral analysis on native SWOT swath arrays."""
from __future__ import annotations

import numpy as np
from scipy.signal import detrend
from scipy.signal.windows import hann

from .utils import SWOT_PIXEL_KM


def swath_arrays(line: np.ndarray, pixel: np.ndarray, *values: np.ndarray) -> tuple[np.ndarray, ...]:
    line = np.asarray(line, np.int64)
    pixel = np.asarray(pixel, np.int64)
    rows = line - line.min()
    shape = (int(rows.max()) + 1, int(pixel.max()) + 1)
    out = []
    for value in values:
        array = np.full(shape, np.nan)
        array[rows, pixel] = value
        out.append(array)
    return tuple(out)


def segment_spectra(fields: list[np.ndarray], mask: np.ndarray, n: int, step: int) -> list[np.ndarray]:
    window = hann(n, sym=False)
    pieces = [[] for _ in fields]
    for j in range(mask.shape[1]):
        edges = np.flatnonzero(np.diff(np.r_[0, mask[:, j].astype(np.int8), 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            for s in range(start, stop - n + 1, step):
                for k, field in enumerate(fields):
                    pieces[k].append(field[s : s + n, j])
    if not pieces[0]:
        return []
    return [np.fft.rfft(window * detrend(np.asarray(p), axis=1, type="linear"), axis=1) for p in pieces]


def accumulate(target: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    error = estimate - target
    return np.stack(
        [
            np.full(target.shape[1], target.shape[0], np.float64),
            (np.abs(target) ** 2).sum(0),
            (np.abs(estimate) ** 2).sum(0),
            (target * np.conj(estimate)).real.sum(0),
            (np.abs(error) ** 2).sum(0),
        ]
    )


def skill_coherence(sums: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore"):
        skill = 1 - sums[..., 4, :] / sums[..., 1, :]
        coherence = sums[..., 3, :] / np.sqrt(sums[..., 1, :] * sums[..., 2, :])
        ratio = np.sqrt(sums[..., 2, :] / sums[..., 1, :])
    return skill, coherence, ratio


def wavelengths(n: int, dx_km: float = SWOT_PIXEL_KM) -> np.ndarray:
    freq = np.fft.rfftfreq(n, d=dx_km)
    with np.errstate(divide="ignore"):
        return np.where(freq > 0, 1 / freq, np.inf)


def effective_resolution(skill: np.ndarray, lam: np.ndarray, threshold: float = 0.5) -> float | None:
    finite = np.isfinite(lam)
    order = np.argsort(-lam[finite])
    lam_s, sk = lam[finite][order], np.asarray(skill)[finite][order]
    for i in range(1, len(sk)):
        if sk[i] < threshold <= sk[i - 1]:
            return float(lam_s[i - 1] + (sk[i - 1] - threshold) / (sk[i - 1] - sk[i]) * (lam_s[i] - lam_s[i - 1]))
    if len(sk) and sk[0] < threshold:
        return None
    return float(lam_s[-1]) if len(sk) else None


def band_mean_skill(sums: np.ndarray, lam: np.ndarray, lo_km: float, hi_km: float) -> float:
    sel = (lam >= lo_km) & (lam < hi_km)
    if not sel.any():
        return float("nan")
    s = sums[..., sel].sum(-1)
    return float(1 - s[4] / s[1]) if s[1] > 0 else float("nan")


class SpectralAccumulator:
    """Accumulate along-track power spectra and spectral skill SK(k) = 1 - S_err(k)/S_SWOT(k)."""

    def __init__(self, names: list[str], n: int = 128, step: int = 64) -> None:
        self.names, self.n, self.step = list(names), int(n), int(step)
        self.lam = wavelengths(self.n)
        self.sums = {name: np.zeros((5, len(self.lam))) for name in self.names}

    def add(self, line, pixel, target, estimates: dict[str, np.ndarray]) -> int:
        arrays = swath_arrays(line, pixel, target, *[estimates[n] for n in self.names])
        mask = np.all([np.isfinite(a) for a in arrays], axis=0)
        spectra = segment_spectra(list(arrays), mask, self.n, self.step)
        if not spectra:
            return 0
        for name, est in zip(self.names, spectra[1:]):
            self.sums[name] += accumulate(spectra[0], est)
        return int(spectra[0].shape[0])

    def summary(self, bands=((50, 130), (30, 60), (60, 130), (130, 260))) -> dict:
        out = {}
        for name, sums in self.sums.items():
            if sums[0, 1] == 0:
                continue
            skill, coherence, ratio = skill_coherence(sums)
            out[name] = {
                "segments": int(sums[0, 1]),
                "effective_resolution_km": effective_resolution(skill, self.lam),
                **{f"skill_{lo}_{hi}km": band_mean_skill(sums, self.lam, lo, hi) for lo, hi in bands},
                "per_wavelength": [
                    {
                        "wavelength_km": float(self.lam[m]),
                        "skill": float(skill[m]),
                        "coherence": float(coherence[m]),
                        "amplitude_ratio": float(ratio[m]),
                    }
                    for m in range(1, len(self.lam))
                    if self.lam[m] >= 15
                ],
            }
        return out
