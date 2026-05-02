"""Case 2a: pre-bridged coupled case seeded from the equilibrium catenoid."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases_dynamic.liquid_bridge_coupled_collision.src._core import OUT_ROOT, run_case
from cases_dynamic.liquid_bridge_coupled_collision.src._params import catenoid_initialized_case2_config


OUT_DIR = OUT_ROOT / "Case_2a"


def main() -> None:
    run_case(
        catenoid_initialized_case2_config(),
        out_dir=OUT_DIR,
        save_fig=True,
        save_results=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
