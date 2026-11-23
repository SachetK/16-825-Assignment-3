#!/usr/bin/env python3
"""Convenience entry point for 16-825 Assignment 3 experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "volume-rendering": ("nerf", ["volume_rendering_main.py", "--config-name=box"]),
    "optimize-box": ("nerf", ["volume_rendering_main.py", "--config-name=train_box"]),
    "nerf-lego": ("nerf", ["volume_rendering_main.py", "--config-name=nerf_lego"]),
    "nerf-materials": ("nerf", ["volume_rendering_main.py", "--config-name=nerf_materials"]),
    "volsdf": ("nerf", ["volume_rendering_main.py", "--config-name=volsdf_surface"]),
    "volsdf-ablation": ("nerf", ["ablation_volsdf.py"]),
    "gaussian-test": ("GS", ["unit_test_gaussians.py"]),
    "gaussian-render": ("GS", ["render.py"]),
    "gaussian-train": ("GS", ["train.py"]),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=COMMANDS)
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="additional arguments passed to the selected experiment",
    )
    parsed = parser.parse_args()

    directory, command = COMMANDS[parsed.experiment]
    completed = subprocess.run(
        [sys.executable, *command, *parsed.args],
        cwd=ROOT / directory,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
