"""Multiphase stress operators.

This is a lightweight local backport of the operator stack Stefan referenced.
For sharp-interface surface benchmarks, it gracefully falls back to the
curvature-based interface force when volumetric dual geometry is absent.
"""

from __future__ import annotations

import numpy as np

from ddgclib._curvatures_heron import hndA_i_interface
from ddgclib.operators.stress import dual_area_vector

# ---------------------------------------------------------------------------
# Benchmark add-on operators: multiphase tet-volume pressure closure
# ---------------------------------------------------------------------------

def multiphase_tet_volume_matrix(points: np.ndarray, tets: np.ndarray):
    """Tet volume-gradient operator for a multiphase simplicial mesh.

    This wrapper intentionally uses tet volumes, not HC local dual volumes, so
    a Delaunay flip does not directly create an HC-dual density/pressure jump.

    References: Chorin projection/continuity pressure solves; ddgclib
    retopology benchmark workaround for HC dual-volume jumps.
    """
    from ddgclib.operators.stress import tet_volume_matrix

    return tet_volume_matrix(points, tets)


def multiphase_tet_volume_lumped_masses(
    n_vertices: int,
    tets: np.ndarray,
    tet_volumes: np.ndarray,
    phases: np.ndarray,
    phase_densities: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return tet masses and barycentric nodal masses for phase-labelled tets.

    Each tet receives ``rho_phase V_t`` and each tet mass is lumped one quarter
    to its four vertices.
    """
    from ddgclib.operators.stress import tet_volume_lumped_nodal_values

    rho = np.asarray([float(phase_densities[int(phase)]) for phase in np.asarray(phases, dtype=int)], dtype=float)
    tet_masses = rho * np.asarray(tet_volumes, dtype=float)
    nodal_masses = tet_volume_lumped_nodal_values(n_vertices, tets, tet_masses)
    return tet_masses, nodal_masses


def multiphase_compressible_eos_pressure_correction(
    *,
    velocities: np.ndarray,
    nonpressure_forces: np.ndarray,
    inv_mass_dof: np.ndarray,
    tet_volumes: np.ndarray,
    target_tet_volumes: np.ndarray,
    reference_tet_volumes: np.ndarray,
    volume_matrix: np.ndarray,
    base_pressures: np.ndarray,
    bulk_by_tet: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiphase weak-compressible pressure correction.

    Solves the same EOS/continuity correction as the one-phase benchmark:

        (I + dt^2 D_K B M^-1 B^T) dp
          = -D_K (V - Vtar + dt B u*)

    where ``D_K = diag(K_t/V_t^0)`` may differ by phase.
    """
    from ddgclib.operators.stress import solve_regularized

    pressure_matrix = np.asarray(volume_matrix, dtype=float).T
    u_star = np.asarray(velocities, dtype=float) + float(dt) * (
        np.asarray(inv_mass_dof, dtype=float) * np.asarray(nonpressure_forces, dtype=float).reshape(-1)
    ).reshape(np.asarray(velocities).shape)
    compressibility = np.asarray(bulk_by_tet, dtype=float) / np.maximum(
        np.asarray(reference_tet_volumes, dtype=float),
        1.0e-300,
    )
    stiffness = (np.asarray(volume_matrix, dtype=float) * np.asarray(inv_mass_dof, dtype=float)[None, :]) @ pressure_matrix
    linear_error = (
        np.asarray(tet_volumes, dtype=float)
        - np.asarray(target_tet_volumes, dtype=float)
        + float(dt) * (np.asarray(volume_matrix, dtype=float) @ u_star.reshape(-1))
    )
    system = np.eye(np.asarray(tet_volumes).shape[0]) + (compressibility[:, None] * float(dt) * float(dt)) * stiffness
    rhs = -compressibility * linear_error
    pressure_delta = solve_regularized(system, rhs)
    velocity_vec = u_star.reshape(-1) + float(dt) * np.asarray(inv_mass_dof, dtype=float) * (pressure_matrix @ pressure_delta)
    return velocity_vec.reshape(np.asarray(velocities).shape), np.asarray(base_pressures, dtype=float) + pressure_delta


def multiphase_incompressible_volume_projection(
    *,
    velocities: np.ndarray,
    nonpressure_forces: np.ndarray,
    inv_mass_dof: np.ndarray,
    tet_volumes: np.ndarray,
    target_tet_volumes: np.ndarray,
    volume_matrix: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiphase local incompressible projection with B u = target_rates.

    Reference: Chorin/Temam projection method, applied to tet-volume
    continuity on the moving multiphase mesh.
    """
    from ddgclib.operators.stress import solve_regularized

    pressure_matrix = np.asarray(volume_matrix, dtype=float).T
    target_rates = -(np.asarray(tet_volumes, dtype=float) - np.asarray(target_tet_volumes, dtype=float)) / float(dt)
    velocity_vec = np.asarray(velocities, dtype=float).reshape(-1)
    nonpressure_vec = np.asarray(nonpressure_forces, dtype=float).reshape(-1)
    stiffness = (np.asarray(volume_matrix, dtype=float) * np.asarray(inv_mass_dof, dtype=float)[None, :]) @ pressure_matrix
    rhs = (
        (target_rates - np.asarray(volume_matrix, dtype=float) @ velocity_vec) / float(dt)
        - np.asarray(volume_matrix, dtype=float) @ (np.asarray(inv_mass_dof, dtype=float) * nonpressure_vec)
    )
    pressure_values = solve_regularized(stiffness, rhs)
    total_forces = np.asarray(nonpressure_forces, dtype=float) + (pressure_matrix @ pressure_values).reshape(np.asarray(velocities).shape)
    projected = np.asarray(velocities, dtype=float) + float(dt) * (
        np.asarray(inv_mass_dof, dtype=float) * total_forces.reshape(-1)
    ).reshape(np.asarray(velocities).shape)
    return projected, pressure_values


def multiphase_active_retopology_remap(
    points: np.ndarray,
    current_tets: np.ndarray,
    phases: np.ndarray,
    *,
    phase_densities: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Active retopology remap for phase-labelled tet meshes.

    The new Delaunay tet list is rebuilt, each new tet is assigned the phase
    of the nearest old tet center, and mass/target volume are recomputed from
    the new tet volume.  This is a controlled benchmark remap, not a fully
    conservative old/new-cell overlap remap.
    """
    from ddgclib.operators.stress import delaunay_retriangulate_points, tet_cell_volumes, tet_face_pairs

    pts = np.asarray(points, dtype=float)
    old_tets = np.asarray(current_tets, dtype=int)
    old_centers = np.mean(pts[old_tets], axis=1)
    new_tets = delaunay_retriangulate_points(pts, old_tets)
    new_centers = np.mean(pts[new_tets], axis=1)
    new_phases = np.zeros(new_tets.shape[0], dtype=int)
    for i, center in enumerate(new_centers):
        nearest = int(np.argmin(np.sum((old_centers - center) ** 2, axis=1)))
        new_phases[i] = int(np.asarray(phases, dtype=int)[nearest])
    new_volumes = tet_cell_volumes(pts, new_tets)
    rho = np.asarray([float(phase_densities[int(phase)]) for phase in new_phases], dtype=float)
    tet_masses = rho * new_volumes
    old_sorted = {tuple(sorted(int(v) for v in tet)) for tet in old_tets}
    new_sorted = {tuple(sorted(int(v) for v in tet)) for tet in new_tets}
    changed = len(old_sorted.symmetric_difference(new_sorted))
    stats = {
        "retriangulation_active": 1.0,
        "display_tet_count": float(new_tets.shape[0]),
        "display_internal_face_count": float(len(tet_face_pairs(new_tets))),
        "display_changed_tets": float(changed),
        "display_changed_fraction": float(changed) / max(float(len(old_sorted) + len(new_sorted)), 1.0),
    }
    return new_tets, new_phases, new_volumes, tet_masses, stats



def _resolve_pressure(v, pressure_model=None, HC=None, dim: int = 3) -> float:
    """Return a scalar pressure for vertex ``v``.

    ``pressure_model`` may update vertex state in-place.  The fallback is the
    scalar/array-like ``v.p`` field used by the single-phase stress operators.
    """
    if pressure_model is not None:
        return float(pressure_model(v, HC=HC, dim=dim))

    p_val = getattr(v, "p", 0.0)
    if np.ndim(p_val) == 0:
        return float(p_val)
    return float(np.asarray(p_val).ravel()[0])


def _interface_surface_tension(v, dim: int, mps) -> np.ndarray:
    interface_nbs = {nb for nb in v.nn if getattr(nb, "is_interface", False)}
    if len(interface_nbs) < 2:
        return np.zeros(dim)

    phases = getattr(v, "interface_phases", frozenset())
    if len(phases) < 2:
        return np.zeros(dim)

    phase_list = sorted(phases)
    gamma = mps.get_gamma_pair(phase_list[0], phase_list[1])
    if gamma == 0.0:
        return np.zeros(dim)

    HNdA, _ = hndA_i_interface(v, interface_nbs | {v})
    return -gamma * HNdA[:dim]


def multiphase_stress_force(
    v,
    dim: int = 3,
    mps=None,
    HC=None,
    pressure_model=None,
) -> np.ndarray:
    """Integrated force on a multiphase dual cell.

    When volumetric dual geometry is available, this mirrors the face-flux
    structure of ``stress_force``.  On pure surface/interface meshes it skips
    the pressure/viscous flux terms and still returns the sharp-interface
    curvature force, which is the relevant contribution for the catenoid
    equilibrium benchmark.
    """
    F = np.zeros(dim)

    can_use_flux_terms = HC is not None and hasattr(v, "vd")
    if can_use_flux_terms:
        p_i = _resolve_pressure(v, pressure_model, HC, dim)
        u_i = v.u[:dim]
        x_i = v.x_a[:dim]
        mu = mps.get_mu(v.phase) if mps is not None else 0.0

        for v_j in v.nn:
            try:
                A_ij = dual_area_vector(v, v_j, HC, dim)
            except (KeyError, IndexError, ValueError, RuntimeError, ZeroDivisionError):
                continue

            p_j = _resolve_pressure(v_j, pressure_model, HC, dim)
            F -= 0.5 * (p_i + p_j) * A_ij

            delta_u = v_j.u[:dim] - u_i
            d_ij = v_j.x_a[:dim] - x_i
            d_norm = np.linalg.norm(d_ij)
            if d_norm < 1e-30:
                continue
            d_hat = d_ij / d_norm
            F += (mu / d_norm) * delta_u * np.dot(d_hat, A_ij)

    if getattr(v, "is_interface", False) and mps is not None:
        F += _interface_surface_tension(v, dim, mps)

    return F


def multiphase_stress_acceleration(
    v,
    dim: int = 3,
    mps=None,
    HC=None,
    pressure_model=None,
) -> np.ndarray:
    F = multiphase_stress_force(v, dim=dim, mps=mps, HC=HC, pressure_model=pressure_model)
    if v.m < 1e-30:
        return np.zeros(dim)
    return F / v.m


multiphase_dudt_i = multiphase_stress_acceleration
