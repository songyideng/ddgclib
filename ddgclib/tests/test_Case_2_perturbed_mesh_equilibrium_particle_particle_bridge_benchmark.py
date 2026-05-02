from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases_dynamic.liquid_bridge_equilibrium import (
    Case_2_perturbed_mesh_equilibrium_particle_particle_bridge_benchmark as benchmark,
)


def test_perturbed_static_rows_have_expected_boundary_counts():
    rows = benchmark.run_static_benchmark()
    assert [int(row["n_boundary"]) for row in rows] == [8, 16, 32, 64]


def test_perturbed_static_rows_have_expected_active_vertex_counts():
    rows = benchmark.run_static_benchmark()
    assert [int(row["n_total"]) for row in rows] == [36, 136, 528, 2080]


def test_perturbed_dynamic_rows_are_finite():
    rows = benchmark.run_dynamic_benchmark()

    for row in rows:
        assert np.isfinite(row["capillary_force_error"])
        assert np.isfinite(row["integration_error"])


def test_perturbed_dynamic_rows_have_expected_boundary_counts():
    rows = benchmark.run_dynamic_benchmark()
    assert [int(row["n_boundary"]) for row in rows] == [8, 16, 32, 64]


def test_perturbed_dynamic_rows_have_expected_active_vertex_counts():
    rows = benchmark.run_dynamic_benchmark()
    assert [int(row["n_total"]) for row in rows] == [36, 136, 528, 2080]


def test_perturbed_dynamic_ddg_fd_rows_are_finite():
    rows = benchmark.run_dynamic_benchmark()
    for row in rows:
        assert np.isfinite(row["ddg_capillary_force_error"])
        assert np.isfinite(row["fd_capillary_force_error"])


def test_perturbed_relaxation_uses_case4_hold_time_settings():
    assert benchmark.PERTURBED_RELAX_DT == 2.0e-6
    assert benchmark.PERTURBED_RELAX_MAX_HOLD_TIME == 2.0e-4


def test_perturbed_benchmark_mesh_png_names():
    rows = benchmark.run_static_benchmark()
    paths = benchmark.render_perturbed_mesh_pngs(rows)

    expected = [
        benchmark.OUT_DIR / "perturbed_mesh_ref2_n8.png",
        benchmark.OUT_DIR / "perturbed_mesh_ref3_n16.png",
        benchmark.OUT_DIR / "perturbed_mesh_ref4_n32.png",
        benchmark.OUT_DIR / "perturbed_mesh_ref5_n64.png",
    ]

    assert paths == expected
    assert all(path.exists() for path in paths)
