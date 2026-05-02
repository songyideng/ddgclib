"""Regression checks for Case 5 volumetric stress equilibrium benchmark."""

from __future__ import annotations

import numpy as np

from cases_dynamic.liquid_bridge_equilibrium.Case_5_volumetric_stress_equilibrium_particle_particle_bridge_benchmark import (
    compute_dynamic_case,
    compute_static_case,
)


def test_static_case_5_stress_residual_is_zero_for_supported_refinements():
    rows = [compute_static_case(refinement) for refinement in (0, 1, 2, 3)]
    axial = np.abs([row["stress_axial_force_residual"] for row in rows])
    local = np.abs([row["stress_max_local_force_norm"] for row in rows])

    assert np.all(axial < 1.0e-12), axial
    assert np.all(local < 1.0e-12), local


def test_static_case_5_uses_expected_boundary_counts():
    rows = [compute_static_case(refinement) for refinement in (0, 1, 2, 3)]
    n_boundary = [int(row["n_boundary"]) for row in rows]
    assert n_boundary == [8, 16, 32, 64]


def test_static_case_5_uses_expected_active_vertex_counts():
    rows = [compute_static_case(refinement) for refinement in (0, 1, 2, 3)]
    n_total = [int(row["n_total"]) for row in rows]
    assert n_total == [81, 289, 1617, 8385]


def test_dynamic_case_5_stress_residual_stays_zero_after_hold():
    rows = [compute_dynamic_case(refinement) for refinement in (0, 1, 2, 3)]
    axial = np.abs([row["stress_axial_force_residual"] for row in rows])
    local = np.abs([row["stress_max_local_force_norm"] for row in rows])

    assert np.all(axial < 1.0e-12), axial
    assert np.all(local < 1.0e-12), local


def test_dynamic_case_5_preserves_expected_boundary_counts():
    rows = [compute_dynamic_case(refinement) for refinement in (0, 1, 2, 3)]
    n_boundary = [int(row["n_boundary"]) for row in rows]
    assert n_boundary == [8, 16, 32, 64]


def test_dynamic_case_5_preserves_expected_active_vertex_counts():
    rows = [compute_dynamic_case(refinement) for refinement in (0, 1, 2, 3)]
    n_total = [int(row["n_total"]) for row in rows]
    assert n_total == [81, 289, 1617, 8385]


def test_dynamic_case_5_operator_capillary_values_stay_finite_and_small():
    rows = [compute_dynamic_case(refinement) for refinement in (0, 1, 2, 3)]
    stress = np.abs([row["stress_capillary_force_error"] for row in rows])
    multiphase = np.abs([row["multiphase_stress_capillary_force_error"] for row in rows])
    ddg = np.abs([row["ddg_capillary_force_error"] for row in rows])
    fd = np.abs([row["fd_capillary_force_error"] for row in rows])

    assert np.all(np.isfinite(stress)), stress
    assert np.all(np.isfinite(multiphase)), multiphase
    assert np.all(np.isfinite(ddg)), ddg
    assert np.all(np.isfinite(fd)), fd
    assert np.all(stress < 1.0e-12), stress
    assert np.all(multiphase < 1.0e-12), multiphase
    assert np.all(ddg < 1.0e-12), ddg
    assert np.all(fd < 1.0e-9), fd


def test_dynamic_case_5_matches_static_equilibrium_metrics():
    static_rows = [compute_static_case(refinement) for refinement in (0, 1, 2, 3)]
    dynamic_rows = [compute_dynamic_case(refinement) for refinement in (0, 1, 2, 3)]

    for static_row, dynamic_row in zip(static_rows, dynamic_rows):
        for key in (
            "stress_axial_force_residual",
            "stress_max_local_force_norm",
            "stress_capillary_force_error",
            "surface_tension_capillary_force_error",
            "multiphase_stress_capillary_force_error",
            "ddg_capillary_force_error",
            "fd_capillary_force_error",
            "integration_error",
        ):
            np.testing.assert_allclose(
                dynamic_row[key],
                static_row[key],
                rtol=0.0,
                atol=1.0e-18,
            )
