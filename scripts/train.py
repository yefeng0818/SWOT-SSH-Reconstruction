#!/usr/bin/env python3
"""Train CFM or benchmark baseline models (U-Net, FNO, ViT, 4DVarNet, EDM, DDPM) on LLC4320 OSSE simulations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swot_cfm.baselines import DETERMINISTIC, GENERATIVE, train_baseline
from swot_cfm.flow_matching import train_cfm
from swot_cfm.utils import load_config


def main() -> None:
    choices = ("cfm",) + DETERMINISTIC + GENERATIVE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="cfm", choices=choices)
    p.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.model == "cfm":
        if args.epochs is not None:
            cfg["training"]["epochs"] = args.epochs
        if args.batch_size is not None:
            cfg["training"]["batch_size"] = args.batch_size
        train_cfm(cfg, resume=not args.no_resume)
    else:
        if args.epochs is not None:
            cfg["baselines"][args.model]["training"]["epochs"] = args.epochs
        if args.batch_size is not None:
            cfg["baselines"][args.model]["training"]["batch_size"] = args.batch_size
        train_baseline(cfg, args.model, resume=not args.no_resume)


if __name__ == "__main__":
    main()
