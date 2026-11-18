#!/usr/bin/env python3
"""Reconstruct daily SWOT-scale SSH ensemble NetCDF maps from DUACS and SWOT observations."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swot_cfm.dataset import SWOTPassDataset, normalization
from swot_cfm.flow_matching import load_cfm, sample_cfm_ensemble
from swot_cfm.utils import DAY_NS, GRAVITY, REAL_ROLES, coriolis, load_config

ONEDAY_LAST = np.datetime64("2023-07-11")
SCIENCE_INPUT_ROLES = ("train", "validation_2025_h1", "test_2025_h2", "heldout_cal")


def day_ns(day: np.datetime64) -> int:
    return int(np.datetime64(day, "D").astype("datetime64[ns]").astype(np.int64))


def write_day(path: Path, day: np.datetime64, lat, lon, mdt, sla, sla_std, ugos, vgos, members) -> None:
    import netCDF4

    tmp = path.with_suffix(".nc.part")
    with netCDF4.Dataset(tmp, "w", format="NETCDF4") as nc:
        nc.createDimension("time", 1)
        nc.createDimension("member", members.shape[0])
        nc.createDimension("latitude", len(lat))
        nc.createDimension("longitude", len(lon))

        t = nc.createVariable("time", "f8", ("time",))
        t.units, t.calendar, t.standard_name, t.axis = "days since 1950-01-01 00:00:00", "standard", "time", "T"
        t[:] = (np.datetime64(day, "D") - np.datetime64("1950-01-01")).astype(int)
        m = nc.createVariable("member", "i2", ("member",))
        m[:] = np.arange(members.shape[0])
        y = nc.createVariable("latitude", "f4", ("latitude",))
        y.units, y.standard_name, y.axis = "degrees_north", "latitude", "Y"
        y[:] = lat
        x = nc.createVariable("longitude", "f4", ("longitude",))
        x.units, x.standard_name, x.axis = "degrees_east", "longitude", "X"
        x[:] = lon

        def packed(name, dims, data, units, long_name, standard_name=None, scale=1e-4):
            v = nc.createVariable(
                name,
                "i2",
                dims,
                zlib=True,
                complevel=4,
                shuffle=True,
                fill_value=np.int16(-32768),
                chunksizes=tuple(
                    1 if d in ("time", "member") else (len(lat) if d == "latitude" else len(lon)) for d in dims
                ),
            )
            v.scale_factor, v.add_offset = np.float32(scale), np.float32(0.0)
            v.units, v.long_name = units, long_name
            if standard_name:
                v.standard_name = standard_name
            v.set_auto_maskandscale(True)
            v[:] = data
            return v

        packed("sla_ens_mean", ("time", "latitude", "longitude"), sla[None], "m", "Sea level anomaly, CFM ensemble mean")
        packed("sla_ens_std", ("time", "latitude", "longitude"), sla_std[None], "m", "Sea level anomaly, ensemble spread")
        packed("ugos", ("time", "latitude", "longitude"), ugos[None], "m s-1", "Eastward absolute geostrophic velocity")
        packed("vgos", ("time", "latitude", "longitude"), vgos[None], "m s-1", "Northward absolute geostrophic velocity")
        packed("sla_member", ("time", "member", "latitude", "longitude"), members[None], "m", "CFM ensemble member SLA")
        packed("mdt", ("latitude", "longitude"), mdt, "m", "Mean dynamic topography CNES-CLS22")

        nc.Conventions = "CF-1.8"
        nc.title = "Daily SWOT-scale sea surface height ensemble (conditional flow matching)"
        nc.date_created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", default="weights/best.pt")
    p.add_argument("--mdt", default="data/cnes_cls22_mdt.npz")
    p.add_argument("--days", default="2023-04-22", help="Comma-separated YYYY-MM-DD dates")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--members", type=int, default=16)
    p.add_argument("--member-batch", type=int, default=4)
    p.add_argument("--out-dir", default="outputs/reconstructions")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cfg = load_config(ROOT / args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = normalization(cfg)
    model = load_cfm(cfg, dev, ROOT / args.checkpoint)
    oneday = SWOTPassDataset(cfg, "oneday_2023", norm)
    science = SWOTPassDataset(cfg, "test_2025_h2", norm)
    science.input_role_codes = np.array([REAL_ROLES.index(r) for r in SCIENCE_INPUT_ROLES])

    lat, lon = oneday.lat[:, 0].astype(np.float32), oneday.lon[0].astype(np.float32)
    lat2 = torch.as_tensor(oneday.lat, dtype=torch.float32, device=dev)
    f = torch.as_tensor(coriolis(oneday.lat), dtype=torch.float32, device=dev)
    dy = float(np.mean(np.diff(lat))) * 111.2e3
    dx = float(np.mean(np.diff(lon))) * 111.2e3 * torch.cos(torch.deg2rad(lat2))
    mdt_np = np.load(ROOT / args.mdt)["mdt"].astype(np.float32)
    mdt = torch.as_tensor(mdt_np, device=dev)

    days = [np.datetime64(s.strip(), "D") for s in args.days.split(",")]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for day in days:
        path = out_dir / f"SSH_ACC_{str(day).replace('-', '')}.nc"
        if path.exists() and not args.overwrite:
            continue
        ds = oneday if day <= ONEDAY_LAST else science
        d = day_ns(day)
        if not all(d + o * DAY_NS in ds.lookup for o in (-2, -1, 0, 1, 2)):
            continue
        t0 = time.time()
        k = ds.lookup[d]
        allowed = np.isin(ds.store.role, ds.input_role_codes)
        granules = ds.encoder.select(ds.store, d, allowed)
        x = torch.from_numpy(ds._inputs(k, granules)[None])
        duacs = torch.from_numpy(ds.duacs[k : k + 1])
        ens = sample_cfm_ensemble(
            model,
            x,
            duacs,
            norm,
            cfg,
            dev,
            members=args.members,
            seed=d // DAY_NS,
            steps=args.steps,
            solver="heun",
            member_batch=args.member_batch,
        )[:, 0].float()
        mean = ens.mean(0)
        spread = ens.std(0, unbiased=(args.members > 1))
        gy, gx = torch.gradient(mean + mdt, dim=(-2, -1))
        u, v = -GRAVITY / f * gy / dy, GRAVITY / f * gx / dx
        write_day(
            path,
            day,
            lat,
            lon,
            mdt_np,
            mean.cpu().numpy(),
            spread.cpu().numpy(),
            u.cpu().numpy(),
            v.cpu().numpy(),
            ens.cpu().numpy(),
        )
        print(json.dumps({"day": str(day), "file": str(path), "seconds": round(time.time() - t0, 2)}))


if __name__ == "__main__":
    main()
