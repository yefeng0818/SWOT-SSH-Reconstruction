"""Physical, spectral, and Lagrangian diagnostics for SSH reconstructions."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal.windows import tukey

from .utils import GRAVITY, coriolis

SIGMA_PER_WAVELENGTH = np.sqrt(np.log(2.0) / 2.0) / np.pi


def grid_spacing_m(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    dy = np.gradient(lat, axis=0) * 111.2e3
    dx = np.gradient(lon, axis=1) * 111.2e3 * np.cos(np.deg2rad(lat))
    f = coriolis(lat)
    return dx, dy, f


def geostrophic_velocity(eta_m: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centered finite-difference geostrophic velocity components (U, V) [m/s]."""
    dx, dy, f = grid_spacing_m(lat, lon)
    eta = np.asarray(eta_m, dtype=np.float64)
    u = -GRAVITY / f * np.gradient(eta, axis=0) / dy
    v = GRAVITY / f * np.gradient(eta, axis=1) / dx
    return u, v


def normalized_vorticity(eta_m: np.ndarray, lat: np.ndarray, lon: np.ndarray, filter_km: float = 10.0) -> np.ndarray:
    """Normalized surface relative vorticity zeta/f = (dV/dx - dU/dy) / f."""
    dx, dy, f = grid_spacing_m(lat, lon)
    eta = np.asarray(eta_m, dtype=np.float64)
    if filter_km and filter_km > 0:
        dy_km = abs(float(np.mean(dy))) / 1e3
        dx_km = abs(float(np.mean(dx))) / 1e3
        sigma = (SIGMA_PER_WAVELENGTH * filter_km / dy_km, SIGMA_PER_WAVELENGTH * filter_km / dx_km)
        eta = gaussian_filter(eta, sigma=sigma, mode="nearest")
    u, v = geostrophic_velocity(eta, lat, lon)
    zeta = np.gradient(v, axis=1) / dx - np.gradient(u, axis=0) / dy
    return zeta / f


def geostrophic_ke_spectrum(eta_m: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2D isotropic geostrophic kinetic energy wavenumber spectrum E(k)."""
    dx, dy, _ = grid_spacing_m(lat, lon)
    ny, nx = lat.shape
    dy_km = abs(float(np.mean(dy))) / 1e3
    dx_km = abs(float(np.mean(dx))) / 1e3
    win = np.outer(tukey(ny, 0.2), tukey(nx, 0.2))
    w_norm = float(np.mean(win**2))
    ky = np.fft.fftfreq(ny, d=dy_km)
    kx = np.fft.fftfreq(nx, d=dx_km)
    kmag = np.hypot(*np.meshgrid(ky, kx, indexing="ij"))
    dk = 1.0 / min(ny * dy_km, nx * dx_km)
    nb = int(min(0.5 / dy_km, 0.5 / dx_km) / dk)
    kbin = np.minimum((kmag / dk).astype(int), nb).ravel()

    u, v = geostrophic_velocity(eta_m, lat, lon)
    uf = np.fft.fft2((u - u.mean()) * win)
    vf = np.fft.fft2((v - v.mean()) * win)
    e2d = 0.5 * (np.abs(uf) ** 2 + np.abs(vf) ** 2) * (dx_km * dy_km) / (ny * nx * w_norm)
    k_centers = (np.arange(nb) + 0.5) * dk
    ke_psd = np.bincount(kbin, weights=e2d.ravel(), minlength=nb + 1)[:nb] / dk
    return k_centers, ke_psd


def coarse_grained_ke_flux(
    eta_m: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    scales_km: tuple[float, ...] = (15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 160.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Coarse-grained kinetic energy flux Pi_ell (Aluie et al., 2018)."""
    dx, dy, _ = grid_spacing_m(lat, lon)
    dy_km = abs(float(np.mean(dy))) / 1e3
    dx_km = abs(float(np.mean(dx))) / 1e3
    u, v = geostrophic_velocity(eta_m, lat, lon)
    ny, nx = u.shape
    scales = np.asarray(scales_km, dtype=np.float64)
    fluxes = np.full(len(scales), np.nan, dtype=np.float64)
    for idx, ell in enumerate(scales):
        sy = SIGMA_PER_WAVELENGTH * ell / dy_km
        sx = SIGMA_PER_WAVELENGTH * ell / dx_km
        filt = lambda a: gaussian_filter(a, sigma=(sy, sx), mode="reflect")
        ub, vb = filt(u), filt(v)
        tau_xx = filt(u * u) - ub * ub
        tau_xy = filt(u * v) - ub * vb
        tau_yy = filt(v * v) - vb * vb
        du_dx = np.gradient(ub, axis=1) / dx
        du_dy = np.gradient(ub, axis=0) / dy
        dv_dx = np.gradient(vb, axis=1) / dx
        dv_dy = np.gradient(vb, axis=0) / dy
        pi = -(tau_xx * du_dx + tau_xy * (du_dy + dv_dx) + tau_yy * dv_dy)
        margin_y = max(2, int(round(0.5 * ell / dy_km)))
        margin_x = max(2, int(round(0.5 * ell / dx_km)))
        if 2 * margin_y < ny and 2 * margin_x < nx:
            fluxes[idx] = float(np.mean(pi[margin_y : ny - margin_y, margin_x : nx - margin_x]))
    return scales, fluxes


class DailyVelocityField:
    """Spatiotemporal bilinear interpolator for daily geostrophic velocity maps."""

    def __init__(self, times_ns: np.ndarray, lat_1d: np.ndarray, lon_1d: np.ndarray, u_mps: np.ndarray, v_mps: np.ndarray):
        self.t = np.asarray(times_ns, dtype=np.int64)
        self.lat = np.asarray(lat_1d, dtype=np.float64)
        self.lon = np.asarray(lon_1d, dtype=np.float64)
        self.u = np.asarray(u_mps, dtype=np.float64)
        self.v = np.asarray(v_mps, dtype=np.float64)

    def __call__(self, t_ns: float, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        if not (
            self.t[0] <= t_ns <= self.t[-1]
            and self.lat[0] <= lat_deg <= self.lat[-1]
            and self.lon[0] <= lon_deg <= self.lon[-1]
        ):
            return np.nan, np.nan
        k = min(int(np.searchsorted(self.t, t_ns, side="right")) - 1, len(self.t) - 2)
        wt = (t_ns - self.t[k]) / max(1.0, float(self.t[k + 1] - self.t[k]))
        iy = float(np.interp(lat_deg, self.lat, np.arange(len(self.lat))))
        ix = float(np.interp(lon_deg, self.lon, np.arange(len(self.lon))))
        y0 = min(int(iy), len(self.lat) - 2)
        x0 = min(int(ix), len(self.lon) - 2)
        fy, fx = iy - y0, ix - x0

        def _bilinear(field: np.ndarray) -> float:
            v00, v01 = field[y0, x0], field[y0, x0 + 1]
            v10, v11 = field[y0 + 1, x0], field[y0 + 1, x0 + 1]
            return float((v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy)

        u_val = (1.0 - wt) * _bilinear(self.u[k]) + wt * _bilinear(self.u[k + 1])
        v_val = (1.0 - wt) * _bilinear(self.v[k]) + wt * _bilinear(self.v[k + 1])
        return u_val, v_val


def advect_rk4(
    flow: DailyVelocityField,
    t0_ns: int,
    lat0: float,
    lon0: float,
    t1_ns: int,
    dt_seconds: float = 3600.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate virtual drifter trajectory dX/dt = U_g(X(t), t) via 4th-order Runge-Kutta."""
    step_ns = int(round(dt_seconds * 1e9))
    n_steps = int(np.ceil((int(t1_ns) - int(t0_ns)) / step_ns)) + 1
    ts = np.minimum(int(t0_ns) + np.arange(n_steps, dtype=np.int64) * step_ns, int(t1_ns))
    lats = np.full(n_steps, np.nan, dtype=np.float64)
    lons = np.full(n_steps, np.nan, dtype=np.float64)
    lats[0], lons[0] = float(lat0), float(lon0)

    def _rhs(t_val: float, lat_val: float, lon_val: float) -> np.ndarray:
        u_mps, v_mps = flow(t_val, lat_val, lon_val)
        dlat_dt = v_mps / 111.2e3
        dlon_dt = u_mps / (111.2e3 * np.cos(np.deg2rad(lat_val)))
        return np.array([dlat_dt, dlon_dt], dtype=np.float64)

    for i in range(n_steps - 1):
        pos = np.array([lats[i], lons[i]], dtype=np.float64)
        h_sec = float(ts[i + 1] - ts[i]) / 1e9
        t_cur = float(ts[i])
        k1 = _rhs(t_cur, pos[0], pos[1])
        k2 = _rhs(t_cur + 0.5 * h_sec * 1e9, *(pos + 0.5 * h_sec * k1))
        k3 = _rhs(t_cur + 0.5 * h_sec * 1e9, *(pos + 0.5 * h_sec * k2))
        k4 = _rhs(t_cur + h_sec * 1e9, *(pos + h_sec * k3))
        nxt = pos + (h_sec / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(nxt).all():
            break
        lats[i + 1], lons[i + 1] = nxt[0], nxt[1]
    return ts, lats, lons
