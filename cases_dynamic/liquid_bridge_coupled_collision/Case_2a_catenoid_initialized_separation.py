"""Case 2a-sep: pre-bridged catenoid case with prescribed particle separation.

This is a visualization-oriented companion to `Case_2a_catenoid_initialized_bridge.py`.
It reuses the same equilibrium-catenoid initial bridge and the same mesh PNG
renderer, but drives the two particles apart so the output folder contains the
expected `mesh_iter****.png` sequence for a pre-bridged separation process.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases_dynamic.liquid_bridge_coupled_collision.src._core import OUT_ROOT, run_case
from cases_dynamic.liquid_bridge_coupled_collision.src._params import catenoid_initialized_case2_config


OUT_DIR = OUT_ROOT / "Case_2a_separation"


def separation_visual_config():
    return replace(
        catenoid_initialized_case2_config(),
        name="Case_2a_catenoid_separation",
        title="Case 2a-sep: Catenoid-initialized particles moving apart",
        prescribed_particle_motion=True,
        v_approach=-1.2e-2,
        n_steps=2500,
        record_every=50,
        mesh_snapshot_every=100,
        bridge_connection_distance=0.0,
    )


def main() -> None:
    run_case(
        separation_visual_config(),
        out_dir=OUT_DIR,
        save_fig=True,
        save_results=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
