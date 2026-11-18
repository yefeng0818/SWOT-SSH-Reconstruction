#!/usr/bin/env python3
"""Evaluate CFM reconstructions on LLC4320 OSSE fields and withheld SWOT passes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swot_cfm.dataset import LLC4320Dataset, SWOTPassDataset, normalization
from swot_cfm.diagnostics import coarse_grained_ke_flux, geostrophic_ke_spectrum, geostrophic_velocity
from swot_cfm.flow_matching import load_cfm, observe, project_out, sample_cfm_ensemble
from swot_cfm.spectra import SpectralAccumulator
from swot_cfm.utils import load_config


def evaluate_osse(cfg, norm, model, device, steps: int, members: int) -> dict:
    ds = LLC4320Dataset(cfg, "test", train=False, norm=norm)
    idx = len(ds) // 2
    b = ds[idx]
    k = int(b["index"])
    day_str = str(np.datetime64(int(ds.times[k]), "ns"))[:10]
    truth = ds.target[k].astype(np.float64)
    duacs = ds.duacs[k].astype(np.float32)

    x_full = torch.from_numpy(ds._inputs(k, ds.granules_for(k, None)))[None]
    ens = (
        sample_cfm_ensemble(
            model,
            x_full,
            torch.from_numpy(duacs)[None],
            norm,
            cfg,
            device,
            members=members,
            seed=k,
            steps=steps,
            solver="heun",
            member_batch=min(4, members),
        )[:, 0]
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    cfm_mean = ens.mean(axis=0)
    cfm_member0 = ens[0]

    lat, lon = ds.lat, ds.lon
    u_true, v_true = geostrophic_velocity(truth, lat, lon)
    u_duacs, v_duacs = geostrophic_velocity(duacs, lat, lon)
    u_cfm, v_cfm = geostrophic_velocity(cfm_mean, lat, lon)

    scales_km, pi_true = coarse_grained_ke_flux(truth, lat, lon, scales_km=(30.0, 60.0, 100.0))
    _, pi_mem0 = coarse_grained_ke_flux(cfm_member0, lat, lon, scales_km=(30.0, 60.0, 100.0))
    k_cpm, ke_true = geostrophic_ke_spectrum(truth, lat, lon)
    _, ke_mem0 = geostrophic_ke_spectrum(cfm_member0, lat, lon)
    band = (1.0 / k_cpm >= 50.0) & (1.0 / k_cpm <= 100.0)
    ke_ratio = float(np.mean(ke_mem0[band] / np.maximum(ke_true[band], 1e-12)))

    return {
        "date": day_str,
        "ssh_mae_cm": {
            "duacs": round(float(100 * np.mean(np.abs((duacs - truth) - np.mean(duacs - truth)))), 3),
            "cfm_mean": round(float(100 * np.mean(np.abs((cfm_mean - truth) - np.mean(cfm_mean - truth)))), 3),
        },
        "velocity_mae_cm_s": {
            "duacs": round(float(50 * (np.mean(np.abs(u_duacs - u_true)) + np.mean(np.abs(v_duacs - v_true)))), 3),
            "cfm_mean": round(float(50 * (np.mean(np.abs(u_cfm - u_true)) + np.mean(np.abs(v_cfm - v_true)))), 3),
        },
        "ke_energy_ratio_50_100km": round(ke_ratio, 3),
        "coarse_grained_flux_1e9_m2_s3": {
            "scales_km": [float(s) for s in scales_km],
            "truth": [round(float(v * 1e9), 3) for v in pi_true],
            "cfm_member": [round(float(v * 1e9), 3) for v in pi_mem0],
        },
    }


def evaluate_withheld_swot(cfg, norm, model, device, steps: int, members: int) -> dict:
    ds = SWOTPassDataset(cfg, "oneday_2023", norm)
    t_target = int(np.datetime64("2023-04-22T23:05", "ns").astype(np.int64))
    pass_times = np.array([int(ds.store.time[g]) for g, _ in ds.targets])
    idx = int(np.argmin(np.abs(pass_times - t_target)))
    b = ds[idx]
    g = int(b["granule"])
    pass_str = str(np.datetime64(int(ds.store.time[g]), "ns"))[:16]

    corners, wspace, wtime = b["corners"].to(device), b["wspace"].to(device), b["wtime"].to(device)
    design = b["design"].to(device)
    swot_val = b["value"].numpy().astype(np.float64)

    duacs_obs = observe(b["duacs"].to(device), corners, wspace, wtime).cpu().numpy()
    ens = sample_cfm_ensemble(
        model,
        b["x"],
        b["duacs"],
        norm,
        cfg,
        device,
        members=members,
        seed=g,
        steps=steps,
        solver="heun",
        member_batch=min(4, members),
    )
    cfm_obs = observe(ens.mean(0).float(), corners, wspace, wtime).cpu().numpy()

    d_mat = design.cpu().numpy().astype(np.float64)
    aligned = {}
    for name, est in (("duacs", duacs_obs), ("cfm", cfm_obs)):
        coef, *_ = np.linalg.lstsq(d_mat, swot_val - est, rcond=None)
        aligned[name] = est + d_mat @ coef

    acc = SpectralAccumulator(["duacs", "cfm"], n=128, step=64)
    segments = acc.add(b["line"].numpy(), b["pixel"].numpy(), swot_val, aligned)
    spec = acc.summary(bands=((60, 100), (100, 200)))

    duacs_rmse = float(project_out(torch.tensor(duacs_obs - swot_val, device=device, dtype=torch.float32), design).square().mean().sqrt())
    cfm_rmse = float(project_out(torch.tensor(cfm_obs - swot_val, device=device, dtype=torch.float32), design).square().mean().sqrt())

    return {
        "withheld_pass_utc": pass_str,
        "segments": segments,
        "projected_rmse_cm": {"duacs": round(100 * duacs_rmse, 3), "cfm_mean": round(100 * cfm_rmse, 3)},
        "effective_resolution_km": {
            "duacs": round(float(spec["duacs"]["effective_resolution_km"]), 2),
            "cfm_mean": round(float(spec["cfm"]["effective_resolution_km"]), 2),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", default="weights/best.pt")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--members", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config(ROOT / args.config)
    device = torch.device(args.device)
    norm = normalization(cfg)
    model = load_cfm(cfg, device, ROOT / args.checkpoint)

    summary = {
        "osse": evaluate_osse(cfg, norm, model, device, args.steps, args.members),
        "withheld_swot": evaluate_withheld_swot(cfg, norm, model, device, args.steps, args.members),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
