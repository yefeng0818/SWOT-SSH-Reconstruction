"""Configuration loader, physical constants, and shared utilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
REAL_ROLES = ("train", "validation_2025_h1", "test_2025_h2", "heldout_cal", "oneday_2023")
SWOT_PIXEL_KM = 2.0
SWOT_CENTRE_PIXEL = 34.0
OMEGA = 7.2921159e-5
GRAVITY = 9.81


def emit(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, default=_default), flush=True)


def _default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _load_yaml(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "extends" in cfg:
        cfg = _merge(_load_yaml((path.parent / cfg.pop("extends")).resolve()), cfg)
    return cfg


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = _load_yaml(path)
    root = Path(cfg.get("project", {}).get("root", path.parents[1]))
    cfg["_root"] = str(root if root.is_absolute() else (path.parent / root).resolve())
    cfg["_config_path"] = str(path)
    return cfg


def resolve(cfg: dict, value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else Path(cfg["_root"]) / value


CACHE_FILES = {
    "real_cache.h5",
    "real_1day_cache.h5",
    "llc_cache.h5",
    "normalization.json",
    "simulator_calibration.json",
}


def obs_bins(cfg: dict) -> int:
    return len(cfg["observations"]["bin_edges_days"]) - 1


def work_path(cfg: dict, *parts: str) -> Path:
    override = cfg["paths"].get("cache_files", {}).get(parts[0]) if len(parts) == 1 else None
    if override:
        path = resolve(cfg, override)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    base = cfg["paths"].get("cache_dir") if parts and parts[0] in CACHE_FILES else None
    path = resolve(cfg, base or cfg["paths"]["work_dir"]).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def fold_for_pass(cycle: int, number: int, folds: int, seed: int) -> int:
    token = f"{int(seed)}:{int(cycle)}:{int(number)}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % int(folds)


def coriolis(latitude_deg: np.ndarray) -> np.ndarray:
    return 2 * OMEGA * np.sin(np.deg2rad(latitude_deg))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False), encoding="utf-8")
