"""Regression checks for the equilibrium particle-particle bridge benchmark."""

from __future__ import annotations

import numpy as np

from cases_dynamic.liquid_bridge_equilibrium.Case_1_equilibrium_particle_particle_bridge_benchmark import (
    DYNAMIC_DAMPING,
    ENDRES_2024_REFERENCE,
    run_dynamic_benchmark,
    run_static_benchmark,
)


def test_static_benchmark_uses_expected_boundary_counts():
    rows = run_static_benchmark()
    n_boundary = [int(row["n_boundary"]) for row in rows]
    assert n_boundary == [8, 16, 32, 64]


def test_static_benchmark_uses_expected_active_vertex_counts():
    rows = run_static_benchmark()
    n_total = [int(row["n_total"]) for row in rows]
    assert n_total == [36, 136, 528, 2080]


def test_static_benchmark_capillary_force_error_is_machine_scale_and_close_to_endres():
    rows = run_static_benchmark()
    errors = np.abs([row["capillary_force_error"] for row in rows])
    assert np.all(errors < 1e-12), errors
    np.testing.assert_allclose(
        errors,
        np.abs(ENDRES_2024_REFERENCE["capillary_force_error"]),
        rtol=0.0,
        atol=1e-15,
    )


def test_static_benchmark_matches_endres_2024_integration_error_sequence():
    rows = run_static_benchmark()
    integration_error = np.array([row["integration_error"] for row in rows])
    np.testing.assert_allclose(
        integration_error,
        ENDRES_2024_REFERENCE["integration_error"],
        rtol=0.0,
        atol=1e-12,
    )


def test_static_benchmark_supported_operator_force_errors_stay_machine_scale():
    rows = run_static_benchmark()
    stress = np.abs([row["stress_capillary_force_error"] for row in rows])
    multiphase = np.abs([row["multiphase_stress_capillary_force_error"] for row in rows])
    curvature = np.abs([row["ddg_capillary_force_error"] for row in rows])

    assert np.all(stress < 1e-12), stress
    assert np.all(multiphase < 1e-12), multiphase
    assert np.all(curvature < 1e-12), curvature


def test_dynamic_benchmark_stays_machine_scale_and_keeps_same_mesh_proxy():
    rows = run_dynamic_benchmark()
    errors = np.abs([row["capillary_force_error"] for row in rows])
    assert np.all(errors < 1e-10), errors

    integration_error = np.array([row["integration_error"] for row in rows])
    np.testing.assert_allclose(
        integration_error,
        ENDRES_2024_REFERENCE["integration_error"],
        rtol=0.0,
        atol=1e-12,
    )


def test_dynamic_benchmark_uses_damped_relaxation_and_keeps_supported_operator_force_errors_small():
    assert DYNAMIC_DAMPING > 0.0

    rows = run_dynamic_benchmark()
    stress = np.abs([row["stress_capillary_force_error"] for row in rows])
    multiphase = np.abs([row["multiphase_stress_capillary_force_error"] for row in rows])
    curvature = np.abs([row["ddg_capillary_force_error"] for row in rows])

    assert np.all(stress < 1e-8), stress
    assert np.all(multiphase < 1e-8), multiphase
    assert np.all(curvature < 1e-8), curvature
