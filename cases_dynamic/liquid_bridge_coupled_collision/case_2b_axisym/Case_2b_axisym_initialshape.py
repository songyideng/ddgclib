"""Case 2b axisym initial equilibrium shape for the Pitois 2000 Fig. 5 setup.

This file exists only to build the static t=0 liquid-bridge shape before any
separation motion starts.

Design intent:
1. Reuse the same volumetric bridge mesh family as the equilibrium benchmarks.
2. Reuse the same volumetric stress operator path used by Case 2b separation.
3. Keep the two sphere-wetted cap patches fixed on the particle surfaces.
4. Relax only the free liquid surface.
5. Include the same hydrostatic pressure contribution used in the Fig. 5
   separation setup.
6. Apply a final scalar-pressure correction so the total discrete force on the
   liquid bridge plus sphere reactions closes to machine precision.

Important note:
The bridge profile is still initialized from the same axisymmetric geometric
guess used by the separation file, but this script then performs a dedicated
static relaxation and force-balance correction on that initial state.
"""

from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ddgclib-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.operators.multiphase_stress import multiphase_stress_force

from cases_dynamic.liquid_bridge_coupled_collision.case_2b_axisym import Case_2b_axisym_pitois2000_separation as sep


OUT_ROOT = Path(__file__).resolve().parent / "out" / "Case_2b_axisym_initialshape"

# ---------------------------------------------------------------------------
# USER CONTROLS
# ---------------------------------------------------------------------------
USER_CONTACT_LINE_NODES = 8
# Exact number of radial rings used near the contact line.
USER_CONTACT_LINE_RADIAL_RINGS = 2
# Additional outer-band rings near the contact line. These add visible radial
# node lines next to the CL; the bias ratio then packs them toward the sphere.
USER_EXTRA_CL_RADIAL_RINGS = 4
# 1 keeps all sidewall axial rings.  The neck has high curvature, so keep the
# full axial stack for the physical axisymmetric initial mesh.
USER_SIDEWALL_AXIAL_RING_STRIDE = 1
USER_NECK_EXTRA_AXIAL_LAYERS = sep.USER_NECK_EXTRA_AXIAL_LAYERS
USER_CL_EXTRA_AXIAL_LAYERS = sep.USER_CL_EXTRA_AXIAL_LAYERS
# Radial gap ratio. Larger r is finer; adjacent gap ratio is 1.1.
USER_CL_RADIAL_BIAS_RATIO = 1.1
USER_RELAX_DT_S = 2.0e-5
USER_TOTAL_ITERATIONS = 10
USER_RELAX_DAMPING = 20.0
USER_FORCE_TOL_N = 1.0e-10
USER_TOTAL_FORCE_TOL_N = 1.0e-14
USER_SPEED_TOL_MPS = 1.0e-9
USER_ACCEL_WORKERS = sep.USER_ACCEL_WORKERS
USER_STEP_MAX_RETRIES = 3
USER_STEP_RETRY_DT_SCALE = 0.5
USER_HISTORY_EVERY = 10
USER_MESH_SNAPSHOT_EVERY_ITERATIONS = 10
# Keep the equilibrium relaxer off the old radial volume projector by default.
# In this Case 2b initial-shape workflow that projector can collapse the
# interior rings while the contact line stays fixed, which destroys the bridge
# geometry before the force residual has a chance to relax.
USER_ENABLE_VOLUME_PROJECTION = False
USER_VOLUME_PROJECTION_MAX_ITERS = 8
USER_VOLUME_PROJECTION_REL_TOL = 1.0e-8
USER_UNPIN_CONTACT_LINE = True
USER_CONTACT_ANGLE_DEG = 0.0
USER_RENDER_INITIAL_GUESS = True
USER_RENDER_FINAL_STATE = True
USER_OPEN_INTERACTIVE_WINDOW = True
USER_INTERACTIVE_STEP = USER_TOTAL_ITERATIONS
USER_RENDER_DISPLAY_AXIAL_LAYERS = 17
USER_INITIAL_GUESS_NECK_FLOOR_RATIO = 0.05
USER_MAX_WATER_MESH_OVERLAP_FRACTION = 1.0e-4
# CL/neck refinement intentionally creates small positive tetrahedra. Treat
# only physically near-zero tet volume as degenerate, not merely small refined
# cells relative to the average volume.
USER_MIN_REL_TET_VOLUME_FRACTION = 1.0e-12
USER_INITIAL_GUESS_AXIAL_SCALE_MIN = 0.26
USER_INITIAL_GUESS_AXIAL_SCALE_MAX = 0.40
USER_INITIAL_GUESS_AXIAL_SCALE_SAMPLES = 15

# Broad Fig. 1-like meridian profile, extracted from the older bridge shape and
# normalized by the contact-line radius. We use this as a smooth reference
# family, then solve only its radial and axial scales so the water-filled tet
# volume matches the Fig. 5 target without collapsing the bridge neck.
_REFERENCE_FIG1_U = np.array(
    [0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0],
    dtype=float,
)
_REFERENCE_FIG1_R_RATIO = np.array(
    [
        0.4812155844,
        0.4957889610,
        0.5385564935,
        0.6066590909,
        0.6953305195,
        0.7978987013,
        0.9057857143,
        1.0085058442,
        1.0000000000,
    ],
    dtype=float,
)


def _separation_view_kwargs() -> dict[str, float]:
    """Always reuse the current separation-camera settings for initial-shape views."""
    return {
        "elev": float(sep.USER_INTERACTIVE_ELEV_DEG),
        "azim": float(sep.USER_INTERACTIVE_AZIM_DEG),
    }


@contextmanager
def _initialshape_mesh_render_style():
    """Render initialshape PNGs with liquid-air interface mesh emphasized."""
    overrides = {
        "USER_FILL_PARTICLE_CAP_SURFACES": False,
        "USER_CAP_EDGE_ALPHA": 0.0,
        "USER_SHOW_REAL_COMPUTE_TRIANGLES": True,
        "USER_SHOW_SIDE_TRIANGLE_DIAGONALS": True,
        "USER_SHOW_SURFACE_OVERLAY": True,
        "USER_SURFACE_OVERLAY_ALPHA": 0.50,
        "USER_SHOW_MESH_VERTICES": True,
        "USER_MESH_VERTEX_SIZE": 14.0,
    }
    old_values = {name: getattr(sep, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(sep, name, value)
        yield
    finally:
        for name, value in old_values.items():
            setattr(sep, name, value)


def _initialshape_display_state(state: sep.VolumetricPitoisState) -> sep.VolumetricPitoisState:
    """Keep computation dense, but render the legacy 17-layer display mesh."""
    target_layers = max(2, int(USER_RENDER_DISPLAY_AXIAL_LAYERS))
    n_layers = len(state.outer_rings)
    if n_layers <= target_layers:
        return state

    layer_idx = np.unique(np.rint(np.linspace(0, n_layers - 1, target_layers)).astype(int))
    return replace(
        state,
        outer_rings=[state.outer_rings[int(idx)] for idx in layer_idx],
        surface_export_side_tris=np.empty((0, 3), dtype=int),
    )


def _render_initialshape_mesh_snapshot(
    state: sep.VolumetricPitoisState,
    title: str,
    out_path: Path,
) -> Path:
    display_title = title.replace("Case 2b axisym:", "Case 2b:", 1)
    display_state = _initialshape_display_state(state)
    with _initialshape_mesh_render_style():
        return sep._render_mesh_snapshot(
            display_state,
            display_title,
            out_path,
            **_separation_view_kwargs(),
        )


def _radially_biased_liquid_air_half_samples(
    *,
    z_from_bvp: np.ndarray,
    r_from_bvp: np.ndarray,
    n_samples: int,
    contact_radius: float,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the liquid-air profile with fine radial spacing at the CL."""
    z_arr = np.asarray(z_from_bvp, dtype=float)
    r_arr = np.asarray(r_from_bvp, dtype=float)
    finite = np.isfinite(z_arr) & np.isfinite(r_arr)
    z_arr = z_arr[finite]
    r_arr = r_arr[finite]
    if z_arr.size < 2 or int(n_samples) < 2:
        return z_arr, r_arr

    order = np.argsort(z_arr)
    z_arr = z_arr[order]
    r_arr = np.maximum.accumulate(r_arr[order])
    r_arr[-1] = float(contact_radius)

    unique_r, unique_idx = np.unique(np.round(r_arr, decimals=15), return_index=True)
    if unique_r.size < 2:
        return (
            np.linspace(float(z_arr[0]), float(z_arr[-1]), int(n_samples), dtype=float),
            np.linspace(float(r_arr[0]), float(contact_radius), int(n_samples), dtype=float),
        )

    z_unique = z_arr[np.sort(unique_idx)]
    r_unique = r_arr[np.sort(unique_idx)]
    r_neck = float(r_unique[0])
    r_cl = float(contact_radius)
    if r_cl <= r_neck or abs(float(ratio) - 1.0) <= 1.0e-12:
        z_samples = np.linspace(float(z_unique[0]), float(z_unique[-1]), int(n_samples), dtype=float)
        return z_samples, np.interp(z_samples, z_unique, r_unique)

    n_gaps = int(n_samples) - 1
    gap_weights = float(ratio) ** np.arange(n_gaps - 1, -1, -1, dtype=float)
    radial_gaps = (r_cl - r_neck) * gap_weights / float(np.sum(gap_weights))
    r_samples = np.concatenate([[r_neck], r_neck + np.cumsum(radial_gaps)])
    r_samples[-1] = r_cl
    z_samples = np.interp(r_samples, r_unique, z_unique)
    z_samples[0] = float(z_unique[0])
    z_samples[-1] = float(z_unique[-1])
    return z_samples, r_samples


def _refinement_from_contact_line_nodes(n_nodes: int) -> int:
    n_nodes = int(n_nodes)
    if n_nodes < 4:
        raise ValueError("USER_CONTACT_LINE_NODES must be at least 4.")
    ratio = n_nodes / 4.0
    log2_ratio = np.log2(ratio)
    refinement = int(round(log2_ratio))
    if abs(log2_ratio - refinement) > 1.0e-12:
        raise ValueError("USER_CONTACT_LINE_NODES must be 4, 8, 16, 32, ... for the current mesh builder.")
    return refinement


def initialshape_config() -> sep.VolumetricPitoisConfig:
    return replace(
        sep.separation_config(),
        name="Case_2b_axisym_initialshape",
        title="Case 2b axisym: equilibrium initial shape",
        use_last_initialshape_msh=False,
        refinement=_refinement_from_contact_line_nodes(USER_CONTACT_LINE_NODES),
        contact_line_radial_rings=int(USER_CONTACT_LINE_RADIAL_RINGS),
        extra_cl_radial_rings=int(USER_EXTRA_CL_RADIAL_RINGS),
        axial_ring_stride=int(USER_SIDEWALL_AXIAL_RING_STRIDE),
        neck_extra_axial_layers=int(USER_NECK_EXTRA_AXIAL_LAYERS),
        cl_extra_axial_layers=int(USER_CL_EXTRA_AXIAL_LAYERS),
        contact_line_radial_bias_ratio=float(USER_CL_RADIAL_BIAS_RATIO),
        cap_speed=0.0,
        dt=USER_RELAX_DT_S,
        n_steps=0,
        damping=USER_RELAX_DAMPING,
        accel_workers=int(USER_ACCEL_WORKERS),
        integration_substeps=1,
        enable_initial_relaxation=False,
        enable_volume_projection=USER_ENABLE_VOLUME_PROJECTION,
        volume_projection_max_iters=USER_VOLUME_PROJECTION_MAX_ITERS,
        volume_projection_rel_tol=USER_VOLUME_PROJECTION_REL_TOL,
        allow_contact_line_growth=USER_UNPIN_CONTACT_LINE,
        contact_angle_deg=USER_CONTACT_ANGLE_DEG,
        record_every=1,
        mesh_snapshot_every=1,
    )


def _static_vertex_force(v, *, state: sep.VolumetricPitoisState) -> np.ndarray:
    return multiphase_stress_force(
        v,
        dim=3,
        mps=state.mps,
        HC=state.HC,
        pressure_model=lambda vv, HC=None, dim=3: sep._pressure_model(vv, HC=HC, dim=dim, state=state),
    )


def _free_force_metrics(state: sep.VolumetricPitoisState) -> dict[str, np.ndarray | float]:
    free_forces = [
        _static_vertex_force(v, state=state)
        for v in state.HC.V
        if v not in state.bV_caps
    ]
    if free_forces:
        free_arr = np.asarray(free_forces, dtype=float)
    else:
        free_arr = np.zeros((0, 3), dtype=float)

    sphere_forces = sep._sphere_total_forces(state)
    free_total = np.sum(free_arr, axis=0) if free_arr.size else np.zeros(3, dtype=float)
    sphere_total = sphere_forces["top_total"] + sphere_forces["bottom_total"]
    net_total = free_total + sphere_total

    return {
        "free_total": free_total,
        "sphere_total": sphere_total,
        "net_total": net_total,
        "max_free_force_norm": float(np.max(np.linalg.norm(free_arr, axis=1))) if free_arr.size else 0.0,
        "rms_free_force_norm": float(np.sqrt(np.mean(np.sum(free_arr * free_arr, axis=1)))) if free_arr.size else 0.0,
        "top_total": np.asarray(sphere_forces["top_total"], dtype=float),
        "bottom_total": np.asarray(sphere_forces["bottom_total"], dtype=float),
        "top_cap": np.asarray(sphere_forces["top_cap"], dtype=float),
        "bottom_cap": np.asarray(sphere_forces["bottom_cap"], dtype=float),
        "top_line": np.asarray(sphere_forces["top_line"], dtype=float),
        "bottom_line": np.asarray(sphere_forces["bottom_line"], dtype=float),
    }


def _solve_total_force_balance_pressure(
    state: sep.VolumetricPitoisState,
    *,
    tol: float = 1.0e-15,
    max_iters: int = 6,
) -> float:
    """Adjust the scalar pressure offset so the net axial force closes to zero."""

    pressure_scale = max(
        1.0,
        abs(float(state.pressure_scalar)),
        float(state.config.gamma) / max(float(state.config.target_cap_radius), 1.0e-30),
    )
    delta_p = max(1.0e-6 * pressure_scale, 1.0e-6)

    p = float(state.pressure_scalar)
    for _ in range(max_iters):
        state.pressure_scalar = p
        f0 = float(_free_force_metrics(state)["net_total"][2])
        if abs(f0) <= tol:
            break

        state.pressure_scalar = p + delta_p
        f1 = float(_free_force_metrics(state)["net_total"][2])
        slope = (f1 - f0) / delta_p
        if not np.isfinite(slope) or abs(slope) <= 1.0e-30:
            state.pressure_scalar = p
            break
        p = p - f0 / slope

    state.pressure_scalar = p
    return p


def _history_row(state: sep.VolumetricPitoisState, *, step: int, time_s: float | None = None) -> dict[str, float | int]:
    metrics = _free_force_metrics(state)
    water_diag = _water_tet_mesh_diagnostics(state)
    snapshot_volume_m3 = float(water_diag["water_volume_m3"])
    target_volume_m3 = float(state.config.initial_bridge_volume_m3)
    rel_error = (snapshot_volume_m3 - target_volume_m3) / max(target_volume_m3, 1.0e-30)
    return {
        "step": int(step),
        "time_s": float(step * state.config.dt if time_s is None else time_s),
        "gap_um": float(sep._gap(state) * 1.0e6),
        "pressure_scalar_pa": float(state.pressure_scalar),
        "snapshot_msh_volume_m3": snapshot_volume_m3,
        "snapshot_msh_volume_ul": float(1.0e9 * snapshot_volume_m3),
        "snapshot_msh_volume_rel_error": float(rel_error),
        "water_volume_m3": snapshot_volume_m3,
        "water_volume_ul": float(1.0e9 * snapshot_volume_m3),
        "water_volume_rel_error": float(rel_error),
        "tet_overlap_fraction": float(water_diag["overlap_fraction"]),
        "degenerate_tet_count": int(water_diag["degenerate_tet_count"]),
        "degenerate_tet_fraction": float(water_diag["degenerate_tet_fraction"]),
        "min_abs_tet_volume_m3": float(water_diag["min_abs_tet_volume_m3"]),
        "max_free_force_norm_n": float(metrics["max_free_force_norm"]),
        "rms_free_force_norm_n": float(metrics["rms_free_force_norm"]),
        "max_free_speed_mps": float(sep._max_free_speed(state)),
        "net_total_force": np.asarray(metrics["net_total"], dtype=float).tolist(),
        "net_total_force_norm_n": float(np.linalg.norm(metrics["net_total"])),
        "free_total_force": np.asarray(metrics["free_total"], dtype=float).tolist(),
        "sphere_total_force": np.asarray(metrics["sphere_total"], dtype=float).tolist(),
        "top_sphere_force": np.asarray(metrics["top_total"], dtype=float).tolist(),
        "bottom_sphere_force": np.asarray(metrics["bottom_total"], dtype=float).tolist(),
    }


def _format_snapshot_volume_status(
    row: dict[str, float | int],
    *,
    target_volume_m3: float,
) -> str:
    return (
        f"target_water_volume={1.0e9 * float(target_volume_m3):.6f} uL, "
        f"current_water_volume={float(row['snapshot_msh_volume_ul']):.6f} uL, "
        f"water_volume_error={100.0 * float(row['snapshot_msh_volume_rel_error']):+.6f} %"
    )


def _water_tet_mesh_diagnostics(state: sep.VolumetricPitoisState) -> dict[str, float | int]:
    points, tets, _node_ids = sep._snapshot_msh_indexed_mesh(state)
    pts = np.asarray(points, dtype=float)
    tet_idx = np.asarray(tets, dtype=int)
    if pts.size == 0 or tet_idx.size == 0:
        return {
            "water_volume_m3": 0.0,
            "signed_total_m3": 0.0,
            "overlap_fraction": 0.0,
            "degenerate_tet_count": 0,
            "degenerate_tet_fraction": 0.0,
            "min_abs_tet_volume_m3": 0.0,
            "tet_count": 0,
        }

    signed_volumes = np.empty(tet_idx.shape[0], dtype=float)
    for idx, (a, b, c, d) in enumerate(tet_idx):
        pa = pts[int(a)]
        pb = pts[int(b)]
        pc = pts[int(c)]
        pd = pts[int(d)]
        signed_volumes[idx] = float(np.dot(pa - pd, np.cross(pb - pd, pc - pd)) / 6.0)

    abs_volumes = np.abs(signed_volumes)
    water_volume = float(np.sum(abs_volumes))
    signed_total = float(abs(np.sum(signed_volumes)))
    overlap_fraction = max(0.0, water_volume - signed_total) / max(water_volume, 1.0e-30)
    avg_tet_volume = water_volume / max(int(abs_volumes.size), 1)
    degenerate_tol = max(
        float(USER_MIN_REL_TET_VOLUME_FRACTION) * avg_tet_volume,
        1.0e-30,
    )
    degenerate_mask = abs_volumes < degenerate_tol
    enclosed_boundary_volume = float(_boundary_enclosed_volume_m3(state))
    fill_mismatch = abs(water_volume - enclosed_boundary_volume) / max(water_volume, 1.0e-30)
    return {
        "water_volume_m3": water_volume,
        "signed_total_m3": signed_total,
        "overlap_fraction": float(overlap_fraction),
        "enclosed_boundary_volume_m3": enclosed_boundary_volume,
        "fill_mismatch_fraction": float(fill_mismatch),
        "degenerate_tet_count": int(np.count_nonzero(degenerate_mask)),
        "degenerate_tet_fraction": float(np.count_nonzero(degenerate_mask) / max(int(abs_volumes.size), 1)),
        "min_abs_tet_volume_m3": float(np.min(abs_volumes)),
        "tet_count": int(abs_volumes.size),
    }


def _assert_valid_water_tet_mesh(
    state: sep.VolumetricPitoisState,
    *,
    context: str,
) -> dict[str, float | int]:
    diag = _water_tet_mesh_diagnostics(state)
    if int(diag["degenerate_tet_count"]) > 0:
        raise ValueError(
            f"{context}: invalid tetra water mesh, degenerate_tets={int(diag['degenerate_tet_count'])}"
        )
    axis = np.asarray(state.top_sphere_center, dtype=float) - np.asarray(state.bottom_sphere_center, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if (not np.isfinite(axis_norm)) or axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    mid = 0.5 * (np.asarray(state.bottom_sphere_center, dtype=float) + np.asarray(state.top_sphere_center, dtype=float))
    layer_s = []
    for ring in state.outer_rings:
        center_pos = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in ring], axis=0)
        layer_s.append(float(np.dot(center_pos - mid, axis)))
    layer_s_arr = np.asarray(layer_s, dtype=float)
    interior_layer_s = layer_s_arr[1:-1] if layer_s_arr.size > 2 else layer_s_arr
    if interior_layer_s.size >= 2 and float(np.min(np.diff(interior_layer_s))) <= 1.0e-12:
        raise ValueError(f"{context}: invalid tetra water mesh, non-monotone axial layer order")
    for rings in state.layer_rings:
        mean_radii = []
        for ring in rings:
            coords = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
            center = np.mean(coords, axis=0)
            rel = coords - center[None, :]
            tangential = rel - np.outer(np.sum(rel * axis[None, :], axis=1), axis)
            mean_radii.append(float(np.mean(np.linalg.norm(tangential, axis=1))))
        if len(mean_radii) >= 2 and float(np.min(np.diff(np.asarray(mean_radii, dtype=float)))) <= 1.0e-12:
            raise ValueError(f"{context}: invalid tetra water mesh, collapsed radial ring order")
    return diag


def _axisym_scaled_positions(
    state: sep.VolumetricPitoisState,
    positions: np.ndarray,
    *,
    axial_scale: float,
    radial_scale: float = 1.0,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return np.empty((0, 3), dtype=float)
    if (not np.all(np.isfinite(positions))) or float(np.max(np.abs(positions))) > 1.0:
        raise ValueError("rejecting non-physical trial coordinates during initial mesh scaling")

    axis = np.asarray(state.top_sphere_center, dtype=float) - np.asarray(state.bottom_sphere_center, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if (not np.isfinite(axis_norm)) or axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    mid = 0.5 * (np.asarray(state.bottom_sphere_center, dtype=float) + np.asarray(state.top_sphere_center, dtype=float))
    if (not np.all(np.isfinite(mid))) or float(np.max(np.abs(mid))) > 1.0:
        raise ValueError("rejecting non-physical sphere centers during initial mesh scaling")

    rel = positions - mid[None, :]
    axial_mag = np.sum(rel * axis[None, :], axis=1)
    axial = axial_mag[:, None] * axis[None, :]
    radial = rel - axial
    return mid[None, :] + float(axial_scale) * axial + float(radial_scale) * radial


def _boundary_enclosed_volume_m3(state: sep.VolumetricPitoisState) -> float:
    points, triangles, _node_ids = sep._snapshot_surface_indexed_mesh(state)
    pts = np.asarray(points, dtype=float)
    tri_idx = np.asarray(triangles, dtype=int)
    if pts.size == 0 or tri_idx.size == 0:
        return 0.0

    total = 0.0
    for a, b, c in tri_idx:
        pa = pts[int(a)]
        pb = pts[int(b)]
        pc = pts[int(c)]
        total += float(np.dot(pa, np.cross(pb, pc)) / 6.0)
    return abs(total)


def _fig1_like_reference_outer_radii(
    state: sep.VolumetricPitoisState,
    *,
    template,
) -> np.ndarray:
    s_vals = np.array([float(item["s"]) for item in template[4]], dtype=float)
    half_span = float(np.max(np.abs(s_vals)))
    contact_radius = float(state.config.target_cap_radius)
    if half_span <= 1.0e-30:
        return np.full_like(s_vals, contact_radius, dtype=float)

    u = np.clip(np.abs(s_vals) / half_span, 0.0, 1.0)
    ref_ratio = np.interp(u, _REFERENCE_FIG1_U, _REFERENCE_FIG1_R_RATIO)
    return contact_radius * ref_ratio


def _apply_axisymmetric_analytic_initial_guess(
    state: sep.VolumetricPitoisState,
    *,
    template,
    outer_radii: np.ndarray,
    axial_scale: float,
    axial_offsets: np.ndarray | None = None,
) -> None:
    mid, axis, e1, e2, layer_data = template
    contact_radius = float(state.config.target_cap_radius)
    axial_offsets_arr = None if axial_offsets is None else np.asarray(axial_offsets, dtype=float)

    n_layers = len(layer_data)
    for layer_idx, (layer, outer_radius) in enumerate(zip(layer_data, np.asarray(outer_radii, dtype=float))):
        if axial_offsets_arr is not None:
            center_coord = mid + float(axial_offsets_arr[layer_idx]) * axis
        elif layer_idx in (0, n_layers - 1):
            center_coord = mid + float(layer["s"]) * axis
        else:
            center_coord = mid + float(axial_scale) * float(layer["s"]) * axis
        sep._move(layer["center_vertex"], tuple(center_coord), state.HC, state.bV_caps)
        for ordered_vertices, factor in layer["rings"]:
            ring_radius = max(float(factor) * float(outer_radius), 0.0)
            n_ring = max(1, len(ordered_vertices))
            targets = []
            for i in range(n_ring):
                phi = 2.0 * np.pi * float(i) / float(n_ring)
                target = center_coord + ring_radius * (np.cos(phi) * e1 + np.sin(phi) * e2)
                targets.append(tuple(target))
            sep._move_vertices_batch(ordered_vertices, targets, state.HC, state.bV_caps)

    sep._rebuild_cap_on_sphere(
        state,
        which="bottom",
        sphere_center=state.bottom_sphere_center,
        contact_radius=contact_radius,
        axis=axis,
        dt=None,
    )
    sep._rebuild_cap_on_sphere(
        state,
        which="top",
        sphere_center=state.top_sphere_center,
        contact_radius=contact_radius,
        axis=axis,
        dt=None,
    )
    sep._align_layer_azimuths(state)

def _build_young_laplace_bvp_initial_guess(state: sep.VolumetricPitoisState) -> bool:
    template = sep._initial_axisymmetric_template(state)
    s_vals = np.array([float(item["s"]) for item in template[4]], dtype=float)
    z_samples = np.sort(np.unique(np.abs(s_vals)))
    if z_samples.size < 2:
        return False

    z_cl = float(np.max(z_samples))
    contact_radius = float(state.config.target_cap_radius)
    rho = float(state.config.rho_f)
    gravity = float(state.config.gravity_mps2)
    gamma = float(state.config.gamma)
    target_half_volume = 0.5 * float(state.config.initial_bridge_volume_m3)

    xi = np.linspace(0.0, 1.0, 300, dtype=float)
    mid_radius_guess = 0.5e-3
    r_guess = mid_radius_guess + (contact_radius - mid_radius_guess) * np.power(xi, 1.4)
    z_guess = z_cl * xi
    psi_guess = np.linspace(np.pi / 2.0, 0.6, xi.size, dtype=float)
    vol_guess = np.pi * np.cumsum(
        np.concatenate(
            [
                [0.0],
                0.5 * (r_guess[1:] ** 2 + r_guess[:-1] ** 2) * np.diff(z_guess),
            ]
        )
    )
    y_guess = np.vstack([r_guess, z_guess, psi_guess, vol_guess])

    def _fun(xi_nodes: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        pressure_offset = float(p[0])
        arc_length_end = float(p[1])
        r = np.maximum(y[0], 1.0e-12)
        z = y[1]
        psi = y[2]
        two_gamma = (pressure_offset - rho * gravity * z) / gamma
        return np.vstack(
            [
                arc_length_end * np.cos(psi),
                arc_length_end * np.sin(psi),
                arc_length_end * (two_gamma - np.sin(psi) / r),
                arc_length_end * np.pi * r * r * np.sin(psi),
            ]
        )

    def _bc(ya: np.ndarray, yb: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.array(
            [
                ya[1],
                ya[2] - np.pi / 2.0,
                ya[3],
                yb[0] - contact_radius,
                yb[1] - z_cl,
                yb[3] - target_half_volume,
            ],
            dtype=float,
        )

    sol = solve_bvp(
        _fun,
        _bc,
        xi,
        y_guess,
        p=np.array([-35.0, 1.1e-3], dtype=float),
        max_nodes=50000,
        tol=1.0e-4,
    )
    if not bool(sol.success):
        return False

    z_half = np.asarray(sol.y[1], dtype=float)
    r_half = np.asarray(sol.y[0], dtype=float)
    finite = np.isfinite(z_half) & np.isfinite(r_half)
    z_half = z_half[finite]
    r_half = r_half[finite]
    order = np.argsort(z_half)
    z_sorted = z_half[order]
    r_sorted = r_half[order]
    z_keep = [float(z_sorted[0])]
    r_keep = [float(r_sorted[0])]
    for z_val, r_val in zip(z_sorted[1:], r_sorted[1:]):
        if float(z_val) > z_keep[-1] + 1.0e-12:
            z_keep.append(float(z_val))
            r_keep.append(float(r_val))
    z_keep_arr = np.asarray(z_keep, dtype=float)
    r_keep_arr = np.asarray(r_keep, dtype=float)
    if z_keep_arr.size < 2:
        return False
    r_keep_arr = np.clip(r_keep_arr, 1.0e-8 * contact_radius, contact_radius)
    r_keep_arr[-1] = contact_radius

    sampled_half = np.interp(z_samples, z_keep_arr, r_keep_arr)
    sampled_full = np.interp(np.abs(s_vals), z_samples, sampled_half)
    axial_offsets = np.array(s_vals, dtype=float, copy=True)

    # Extra liquid-air layers near each contact line use the requested radial
    # gap ratio: smallest dr at the CL, then each gap grows by 1.1 away from it.
    extra_total = max(0, int(state.config.cl_extra_axial_layers))
    lower_extra = (extra_total + 1) // 2
    upper_extra = extra_total // 2
    ratio = max(1.0, float(state.config.contact_line_radial_bias_ratio))

    def split_contact_segment(indices: list[int], *, cl_at_start: bool) -> None:
        if len(indices) < 2:
            return
        cl_to_anchor = list(indices) if cl_at_start else list(reversed(indices))
        gaps = len(cl_to_anchor) - 1
        weights = ratio ** np.arange(gaps, dtype=float)
        fractions = np.concatenate([[0.0], np.cumsum(weights / float(np.sum(weights)))])
        i_cl = cl_to_anchor[0]
        i_anchor = cl_to_anchor[-1]
        z_cl_abs = abs(float(s_vals[i_cl]))
        z_anchor_abs = abs(float(s_vals[i_anchor]))
        r_cl = contact_radius
        r_anchor = float(sampled_full[i_anchor])
        for idx, frac in zip(cl_to_anchor, fractions):
            axial_offsets[idx] = np.sign(s_vals[idx]) * (
                (1.0 - float(frac)) * z_cl_abs + float(frac) * z_anchor_abs
            )
            sampled_full[idx] = (1.0 - float(frac)) * r_cl + float(frac) * r_anchor

    if lower_extra > 0 and lower_extra + 1 < s_vals.size:
        split_contact_segment(list(range(0, lower_extra + 2)), cl_at_start=True)
    if upper_extra > 0 and s_vals.size - upper_extra - 2 >= 0:
        top_indices = list(range(s_vals.size - upper_extra - 2, s_vals.size))
        split_contact_segment(top_indices, cl_at_start=False)

    _apply_axisymmetric_analytic_initial_guess(
        state,
        template=template,
        outer_radii=sampled_full,
        axial_scale=1.0,
        axial_offsets=axial_offsets,
    )
    axial_span = float(axial_offsets[-1] - axial_offsets[0])
    if axial_span > 1.0e-30:
        state.layer_fractions = tuple(
            float(np.clip((float(offset) - float(axial_offsets[0])) / axial_span, 0.0, 1.0))
            for offset in axial_offsets
        )
    _enforce_target_msh_volume(state)
    sep._update_duals_and_masses(state)
    sep._update_pressure_scalar(state)
    _assert_valid_water_tet_mesh(state, context="Young-Laplace initial guess")
    return True


def _build_reference_scaled_initial_guess(state: sep.VolumetricPitoisState) -> None:
    template = sep._initial_axisymmetric_template(state)
    target_volume_m3 = float(state.config.initial_bridge_volume_m3)
    reference_outer_radii = _fig1_like_reference_outer_radii(state, template=template)
    best: tuple[float, float, float, np.ndarray, dict[str, float | int]] | None = None
    base_snapshot = _capture_state_snapshot(state)
    axial_candidates = np.linspace(
        float(USER_INITIAL_GUESS_AXIAL_SCALE_MIN),
        float(USER_INITIAL_GUESS_AXIAL_SCALE_MAX),
        max(2, int(USER_INITIAL_GUESS_AXIAL_SCALE_SAMPLES)),
        dtype=float,
    )

    def _diag_at(*, axial_scale: float, radial_scale: float) -> dict[str, float | int]:
        _restore_state_snapshot(state, base_snapshot)
        _apply_axisymmetric_analytic_initial_guess(
            state,
            template=template,
            outer_radii=reference_outer_radii * float(radial_scale),
            axial_scale=float(axial_scale),
        )
        return _water_tet_mesh_diagnostics(state)

    for axial_scale in axial_candidates:
        low = 0.45
        high = 0.85
        diag_low = _diag_at(axial_scale=float(axial_scale), radial_scale=low)
        diag_high = _diag_at(axial_scale=float(axial_scale), radial_scale=high)
        vol_low = float(diag_low["water_volume_m3"])
        vol_high = float(diag_high["water_volume_m3"])
        if not (min(vol_low, vol_high) <= target_volume_m3 <= max(vol_low, vol_high)):
            continue

        best_radial = 0.5 * (low + high)
        best_diag = diag_high
        for _ in range(24):
            mid = 0.5 * (low + high)
            diag_mid = _diag_at(axial_scale=float(axial_scale), radial_scale=mid)
            vol_mid = float(diag_mid["water_volume_m3"])
            best_radial = mid
            best_diag = diag_mid
            if abs(vol_mid - target_volume_m3) / max(target_volume_m3, 1.0e-30) <= 1.0e-8:
                break
            if vol_mid < target_volume_m3:
                low = mid
            else:
                high = mid

        if int(best_diag["degenerate_tet_count"]) > 0:
            continue

        enclosed = float(best_diag["enclosed_boundary_volume_m3"])
        water = float(best_diag["water_volume_m3"])
        mid_radius_mm = float(reference_outer_radii[len(reference_outer_radii) // 2] * best_radial * 1.0e3)
        score = (
            4.0 * abs(enclosed - target_volume_m3) / max(target_volume_m3, 1.0e-30)
            + 0.5 * abs(enclosed - water) / max(target_volume_m3, 1.0e-30)
            - 0.08 * mid_radius_mm
        )
        candidate = (
            float(score),
            float(axial_scale),
            float(best_radial),
            (reference_outer_radii * best_radial).copy(),
            best_diag,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        raise ValueError("Could not build a broad analytic initial tetra mesh matching the target water volume.")

    _score, best_axial, _best_radial, best_outer_radii, _best_diag = best
    _apply_axisymmetric_analytic_initial_guess(
        state,
        template=template,
        outer_radii=best_outer_radii,
        axial_scale=float(best_axial),
    )
    sep._update_duals_and_masses(state)
    _assert_valid_water_tet_mesh(state, context="analytic initial guess")


def _build_volume_matched_analytic_initial_guess(state: sep.VolumetricPitoisState) -> None:
    # Use the physical Young-Laplace bridge profile when the boundary-value
    # solve succeeds.  Fall back to the analytic Fig. 1-like profile only if the
    # BVP fails for the current coarse mesh controls.
    if _build_young_laplace_bvp_initial_guess(state):
        return
    _build_reference_scaled_initial_guess(state)


def _capture_state_snapshot(state: sep.VolumetricPitoisState) -> dict:
    vertices = list(state.HC.V)
    return {
        "vertices": vertices,
        "positions": [tuple(np.asarray(v.x_a[:3], dtype=float)) for v in vertices],
        "velocities": [np.asarray(v.u[:3], dtype=float).copy() for v in vertices],
        "bottom_sphere_center": np.asarray(state.bottom_sphere_center, dtype=float).copy(),
        "top_sphere_center": np.asarray(state.top_sphere_center, dtype=float).copy(),
        "pressure_scalar": float(state.pressure_scalar),
    }


def _restore_state_snapshot(state: sep.VolumetricPitoisState, snapshot: dict) -> None:
    sep._move_vertices_batch(snapshot["vertices"], snapshot["positions"], state.HC, state.bV_caps)
    for v, u in zip(snapshot["vertices"], snapshot["velocities"]):
        v.u = np.asarray(u, dtype=float).copy()
    state.bottom_sphere_center = np.asarray(snapshot["bottom_sphere_center"], dtype=float).copy()
    state.top_sphere_center = np.asarray(snapshot["top_sphere_center"], dtype=float).copy()
    state.pressure_scalar = float(snapshot["pressure_scalar"])


def _first_nonfinite_state_issue(state: sep.VolumetricPitoisState) -> str | None:
    for v in state.HC.V:
        pos = np.asarray(v.x_a[:3], dtype=float)
        if not np.all(np.isfinite(pos)):
            return f"non-finite vertex position at {tuple(pos.tolist())}"
        vel = np.asarray(v.u[:3], dtype=float)
        if not np.all(np.isfinite(vel)):
            return f"non-finite vertex velocity at {tuple(vel.tolist())}"
    if not np.all(np.isfinite(np.asarray(state.bottom_sphere_center, dtype=float))):
        return "non-finite bottom_sphere_center"
    if not np.all(np.isfinite(np.asarray(state.top_sphere_center, dtype=float))):
        return "non-finite top_sphere_center"
    if not np.isfinite(float(state.pressure_scalar)):
        return "non-finite pressure_scalar"
    return None


def _project_contact_line_vertices_to_spheres(state: sep.VolumetricPitoisState) -> None:
    """Constrain CL nodes to slide on the two spheres without rebuilding caps."""
    sphere_radius = float(state.config.particle_radius)
    if sphere_radius <= 0.0:
        return

    for ring, sphere_center in (
        (list(state.bottom_contact_ring), np.asarray(state.bottom_sphere_center, dtype=float)),
        (list(state.top_contact_ring), np.asarray(state.top_sphere_center, dtype=float)),
    ):
        if not ring:
            continue
        targets: list[tuple[float, float, float]] = []
        normals: list[np.ndarray] = []
        for vertex in ring:
            point = np.asarray(vertex.x_a[:3], dtype=float)
            rel = point - sphere_center
            rel_norm = float(np.linalg.norm(rel))
            if (not np.isfinite(rel_norm)) or rel_norm <= 1.0e-30:
                targets.append(tuple(point))
                normals.append(np.zeros(3, dtype=float))
                continue
            normal = rel / rel_norm
            targets.append(tuple(sphere_center + sphere_radius * normal))
            normals.append(normal)
        sep._move_vertices_batch(ring, targets, state.HC, state.bV_caps)
        for vertex, normal in zip(ring, normals):
            if float(np.linalg.norm(normal)) <= 1.0e-30:
                continue
            velocity = np.asarray(vertex.u[:3], dtype=float)
            vertex.u = velocity - float(np.dot(velocity, normal)) * normal


def _axisym_force_snapshot_volume_to_target_with_cl_resplit(
    state: sep.VolumetricPitoisState,
    *,
    rel_tol: float,
    max_iters: int,
) -> float:
    """Match water volume after imposing the liquid-air CL bias pattern."""
    target = float(state.config.initial_bridge_volume_m3)
    state.target_snapshot_volume_m3 = target
    _project_contact_line_vertices_to_spheres(state)
    sep._resplit_contact_axial_layers(state)
    current = float(sep._snapshot_msh_volume_m3(state))
    if abs(current - target) / max(target, 1.0e-30) <= float(rel_tol):
        return current

    vertices, base_positions = sep._snapshot_projectable_vertex_positions(state)
    if not vertices:
        return current

    def restore_base() -> None:
        sep._apply_free_vertex_positions(state, vertices, base_positions)

    def volume_at(scale: float) -> float:
        trial = sep._axisym_water_volume_scaled_positions(
            state,
            base_positions,
            meridional_scale=float(scale),
        )
        sep._apply_free_vertex_positions(state, vertices, trial)
        sep._resplit_contact_axial_layers(state)
        volume = float(sep._snapshot_msh_volume_m3(state))
        restore_base()
        return volume

    low = high = 1.0
    if current < target:
        high = 1.05
        vol_high = volume_at(high)
        while vol_high < target and high < 8.0:
            high *= 1.10
            vol_high = volume_at(high)
        if vol_high < target:
            restore_base()
            sep._resplit_contact_axial_layers(state)
            return float(sep._snapshot_msh_volume_m3(state))
    else:
        low = 0.95
        vol_low = volume_at(low)
        while vol_low > target and low > 0.02:
            low *= 0.90
            vol_low = volume_at(low)
        if vol_low > target:
            restore_base()
            sep._resplit_contact_axial_layers(state)
            return float(sep._snapshot_msh_volume_m3(state))
        high = 1.0

    final_scale = 1.0
    for _ in range(max(1, int(max_iters))):
        mid_scale = 0.5 * (low + high)
        vol_mid = volume_at(mid_scale)
        final_scale = mid_scale
        if abs(vol_mid - target) / max(target, 1.0e-30) <= float(rel_tol):
            break
        if vol_mid < target:
            low = mid_scale
        else:
            high = mid_scale

    final_positions = sep._axisym_water_volume_scaled_positions(
        state,
        base_positions,
        meridional_scale=float(final_scale),
    )
    sep._apply_free_vertex_positions(state, vertices, final_positions)
    sep._resplit_contact_axial_layers(state)
    _project_contact_line_vertices_to_spheres(state)
    return float(sep._snapshot_msh_volume_m3(state))


def _enforce_target_msh_volume(
    state: sep.VolumetricPitoisState,
    *,
    rel_tol: float = 1.0e-6,
    max_iters: int = 32,
) -> float:
    target = float(state.config.initial_bridge_volume_m3)
    state.target_snapshot_volume_m3 = target
    current = float(
        _axisym_force_snapshot_volume_to_target_with_cl_resplit(
            state,
            rel_tol=rel_tol,
            max_iters=max_iters,
        )
    )
    if abs(current - target) / max(target, 1.0e-30) > float(rel_tol):
        current = float(sep._axisym_force_snapshot_volume_to_target(state, rel_tol=rel_tol, max_iters=max_iters))
        sep._resplit_contact_axial_layers(state)
        current = float(
            _axisym_force_snapshot_volume_to_target_with_cl_resplit(
                state,
                rel_tol=rel_tol,
                max_iters=max_iters,
            )
        )
    _project_contact_line_vertices_to_spheres(state)
    sep._update_duals_and_masses(state)
    sep._update_pressure_scalar(state)
    return float(_assert_valid_water_tet_mesh(state, context="water-volume correction")["water_volume_m3"])


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _render_equilibrium_history_figures(history: list[dict], out_dir: Path, *, dt: float) -> list[Path]:
    if not history:
        return []

    fig_dir = out_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    steps = np.array([int(row["step"]) for row in history], dtype=float)
    time_ms = np.array(
        [1.0e3 * float(row.get("time_s", float(step) * float(dt))) for step, row in zip(steps, history)],
        dtype=float,
    )
    max_free_force = np.array([float(row["max_free_force_norm_n"]) for row in history], dtype=float)
    total_force = np.array([float(row["net_total_force_norm_n"]) for row in history], dtype=float)
    max_free_speed = np.array([float(row["max_free_speed_mps"]) for row in history], dtype=float)
    snapshot_volume_ul = np.array([float(row["snapshot_msh_volume_ul"]) for row in history], dtype=float)
    snapshot_volume_rel_frac = np.array([float(row["snapshot_msh_volume_rel_error"]) for row in history], dtype=float)
    target_volume_ul = float(snapshot_volume_ul[0] / (1.0 + snapshot_volume_rel_frac[0]))
    snapshot_volume_delta_pl = 1.0e6 * (snapshot_volume_ul - target_volume_ul)
    snapshot_volume_rel_ppb = 1.0e9 * snapshot_volume_rel_frac

    paths: list[Path] = []

    fig1, ax1 = plt.subplots(figsize=(7.0, 4.6))
    ax1.plot(time_ms, max_free_force, color="#bc4749", linewidth=1.8, marker="o", markersize=3.8)
    ax1.set_yscale("log")
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Max free-force norm [N]")
    ax1.set_title("Equilibrium relaxation: local force residual")
    ax1.grid(alpha=0.28, which="both")
    path1 = fig_dir / "equilibrium_max_free_force_vs_t.png"
    fig1.tight_layout()
    fig1.savefig(path1, dpi=180, bbox_inches="tight")
    plt.close(fig1)
    paths.append(path1)

    fig2, ax2 = plt.subplots(figsize=(7.0, 4.6))
    ax2.plot(time_ms, total_force, color="#2a6f97", linewidth=1.8, marker="o", markersize=3.8)
    ax2.set_yscale("log")
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("Total force norm [N]")
    ax2.set_title("Equilibrium relaxation: global force residual")
    ax2.grid(alpha=0.28, which="both")
    path2 = fig_dir / "equilibrium_total_force_vs_t.png"
    fig2.tight_layout()
    fig2.savefig(path2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    paths.append(path2)

    fig3, ax3 = plt.subplots(figsize=(7.0, 4.6))
    ax3.plot(time_ms, max_free_speed, color="#6a4c93", linewidth=1.8, marker="o", markersize=3.8)
    if np.any(max_free_speed > 0.0):
        ax3.set_yscale("log")
    ax3.set_xlabel("Time [ms]")
    ax3.set_ylabel("Max free speed [m/s]")
    ax3.set_title("Equilibrium relaxation: speed residual")
    ax3.grid(alpha=0.28, which="both")
    path3 = fig_dir / "equilibrium_max_free_speed_vs_t.png"
    fig3.tight_layout()
    fig3.savefig(path3, dpi=180, bbox_inches="tight")
    plt.close(fig3)
    paths.append(path3)

    fig4, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)
    axes[0].plot(time_ms, max_free_force, color="#bc4749", linewidth=1.8, marker="o", markersize=3.2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Max free-force [N]")
    axes[0].grid(alpha=0.28, which="both")
    axes[0].set_title("Equilibrium relaxation residuals vs time")

    axes[1].plot(time_ms, total_force, color="#2a6f97", linewidth=1.8, marker="o", markersize=3.2)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Total force [N]")
    axes[1].grid(alpha=0.28, which="both")

    axes[2].plot(time_ms, max_free_speed, color="#6a4c93", linewidth=1.8, marker="o", markersize=3.2)
    if np.any(max_free_speed > 0.0):
        axes[2].set_yscale("log")
    axes[2].set_ylabel("Max free speed [m/s]")
    axes[2].set_xlabel("Time [ms]")
    axes[2].grid(alpha=0.28, which="both")

    path4 = fig_dir / "equilibrium_residuals_vs_t.png"
    fig4.tight_layout()
    fig4.savefig(path4, dpi=180, bbox_inches="tight")
    plt.close(fig4)
    paths.append(path4)

    fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(7.2, 7.6), sharex=True)
    ax5a.plot(time_ms, snapshot_volume_delta_pl, color="#1f7a8c", linewidth=1.8, marker="o", markersize=3.2)
    ax5a.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax5a.set_ylabel("Delta volume [pL]")
    ax5a.set_title(f"Equilibrium relaxation: water-filled tetra volume vs target ({target_volume_ul:.6f} uL)")
    ax5a.grid(alpha=0.28)

    ax5b.plot(time_ms, snapshot_volume_rel_ppb, color="#bc4749", linewidth=1.8, marker="o", markersize=3.2)
    ax5b.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax5b.set_xlabel("Time [ms]")
    ax5b.set_ylabel("Rel. error [ppb]")
    ax5b.grid(alpha=0.28)

    path5 = fig_dir / "equilibrium_snapshot_volume_rel_error_vs_t.png"
    fig5.tight_layout()
    fig5.savefig(path5, dpi=180, bbox_inches="tight")
    plt.close(fig5)
    paths.append(path5)

    return paths


def _save_relaxation_snapshot(
    state: sep.VolumetricPitoisState,
    *,
    step: int,
    fig_dir: Path,
) -> Path:
    return _render_initialshape_mesh_snapshot(
        state,
        f"{state.config.title}: iteration {step} mesh",
        fig_dir / f"mesh_iter{step:04d}.png",
    )


def _show_interactive_snapshot(
    state: sep.VolumetricPitoisState,
    *,
    step: int,
) -> None:
    sep._show_state_interactive(
        state,
        step=int(step),
        **_separation_view_kwargs(),
    )


def _summary_payload(
    state: sep.VolumetricPitoisState,
    *,
    initial_metrics: dict,
    final_metrics: dict,
    history: list[dict],
) -> dict:
    snapshot_volume = float(_water_tet_mesh_diagnostics(state)["water_volume_m3"])
    dual_cell_sum = float(sep._mesh_volume_m3(state.HC))
    weight = np.array([0.0, 0.0, -float(state.config.rho_f) * snapshot_volume * float(state.config.gravity_mps2)], dtype=float)
    return {
        "config": {
            "refinement": int(state.config.refinement),
            "particle_radius_mm": float(state.config.particle_radius * 1.0e3),
            "target_cap_radius_mm": float(state.config.target_cap_radius * 1.0e3),
            "initial_gap_um": float(sep._gap(state) * 1.0e6),
            "volume_ul": float(snapshot_volume * 1.0e9),
            "dual_cell_sum_ul": float(dual_cell_sum * 1.0e9),
            "gamma_npm": float(state.config.gamma),
            "mu_pas": float(state.config.mu_f),
            "rho_f": float(state.config.rho_f),
            "gravity_mps2": float(state.config.gravity_mps2),
        },
        "initial_metrics": {
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in initial_metrics.items()},
        },
        "final_metrics": {
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in final_metrics.items()},
            "weight_force": weight.tolist(),
        },
        "history": history,
    }


def _relax_initial_shape(
    state: sep.VolumetricPitoisState,
    *,
    max_steps: int,
    force_tol: float,
    total_force_tol: float,
    speed_tol: float,
    fig_dir: Path,
) -> list[dict]:
    history: list[dict] = []
    run_wall_start = time.perf_counter()
    interactive_shown = False
    sim_time_s = 0.0
    working_dt = float(state.config.dt)
    target_volume_m3 = float(state.config.initial_bridge_volume_m3)

    for v in state.HC.V:
        v.u = np.zeros(3, dtype=float)

    sep._update_duals_and_masses(state)
    sep._update_pressure_scalar(state)
    if USER_UNPIN_CONTACT_LINE:
        _project_contact_line_vertices_to_spheres(state)
        _enforce_target_msh_volume(state)
        sep._update_duals_and_masses(state)
        sep._update_pressure_scalar(state)
    else:
        _enforce_target_msh_volume(state)
    _assert_valid_water_tet_mesh(state, context="equilibrium step 0")
    _solve_total_force_balance_pressure(state)
    row0 = _history_row(state, step=0, time_s=sim_time_s)
    history.append(row0)
    print(
        f"[equilibrium] step 0/{max_steps}: "
        f"step_wall=0.000000 s, elapsed=0.000000 s, "
        f"max_free_force={row0['max_free_force_norm_n']:.6e} N, "
        f"net_total={row0['net_total_force_norm_n']:.6e} N, "
        f"max_free_speed={row0['max_free_speed_mps']:.6e} m/s, "
        f"{_format_snapshot_volume_status(row0, target_volume_m3=target_volume_m3)}",
        flush=True,
    )
    if USER_OPEN_INTERACTIVE_WINDOW and int(USER_INTERACTIVE_STEP) == 0:
        _show_interactive_snapshot(state, step=0)
        interactive_shown = True

    for step in range(1, max_steps + 1):
        step_wall_start = time.perf_counter()
        snapshot = _capture_state_snapshot(state)
        dt_step = working_dt
        issue = None
        for attempt in range(max(1, int(USER_STEP_MAX_RETRIES)) + 1):
            if attempt > 0:
                _restore_state_snapshot(state, snapshot)
                for v in state.HC.V:
                    v.u = np.zeros(3, dtype=float)
                dt_step *= float(USER_STEP_RETRY_DT_SCALE)
                print(
                    f"[equilibrium] step {step}/{max_steps}: retry {attempt} with dt={dt_step:.3e} s",
                    flush=True,
                )

            symplectic_euler(
                state.HC,
                state.bV_caps,
                sep._vertex_acceleration,
                dt=dt_step,
                n_steps=1,
                dim=3,
                retopologize_fn=False,
                workers=max(1, int(USER_ACCEL_WORKERS)),
                state=state,
            )
            issue = _first_nonfinite_state_issue(state)
            if issue is None:
                try:
                    if USER_UNPIN_CONTACT_LINE:
                        _project_contact_line_vertices_to_spheres(state)
                        sep._axisymmetrize_layer_rings(state)
                        sep._regularize_layer_order(state)
                        _enforce_target_msh_volume(state)
                    else:
                        sep._axisymmetrize_layer_rings(state)
                        sep._regularize_layer_order(state)
                        _enforce_target_msh_volume(state)
                except ValueError as exc:
                    issue = str(exc)
                else:
                    issue = _first_nonfinite_state_issue(state)
            if issue is None:
                try:
                    _assert_valid_water_tet_mesh(state, context=f"equilibrium step {step} attempt {attempt}")
                except ValueError as exc:
                    issue = str(exc)
            if issue is None:
                break

        if issue is not None:
            _restore_state_snapshot(state, snapshot)
            for v in state.HC.V:
                v.u = np.zeros(3, dtype=float)
            print(
                f"[equilibrium] step {step}/{max_steps}: stopped on last valid state after {issue}",
                flush=True,
            )
            break

        working_dt = dt_step
        sim_time_s += dt_step
        sep._update_duals_and_masses(state)
        sep._update_pressure_scalar(state)
        _solve_total_force_balance_pressure(state)
        row = _history_row(state, step=step, time_s=sim_time_s)
        step_wall = time.perf_counter() - step_wall_start
        elapsed = time.perf_counter() - run_wall_start

        print(
            f"[equilibrium] step {step}/{max_steps}: "
            f"step_wall={step_wall:.6f} s, elapsed={elapsed:.6f} s, "
            f"dt={dt_step:.3e} s, "
            f"max_free_force={row['max_free_force_norm_n']:.6e} N, "
            f"net_total={row['net_total_force_norm_n']:.6e} N, "
            f"max_free_speed={row['max_free_speed_mps']:.6e} m/s, "
            f"{_format_snapshot_volume_status(row, target_volume_m3=target_volume_m3)}",
            flush=True,
        )

        snapshot_every = max(1, int(USER_MESH_SNAPSHOT_EVERY_ITERATIONS))
        if step % snapshot_every == 0:
            _save_relaxation_snapshot(state, step=step, fig_dir=fig_dir)
        if USER_OPEN_INTERACTIVE_WINDOW and not interactive_shown and int(USER_INTERACTIVE_STEP) == step:
            _show_interactive_snapshot(state, step=step)
            interactive_shown = True

        history.append(row)
        if (
            float(row["max_free_force_norm_n"]) <= force_tol
            and float(row["net_total_force_norm_n"]) <= total_force_tol
            and float(row["max_free_speed_mps"]) <= speed_tol
        ):
            break

    for v in state.HC.V:
        v.u = np.zeros(3, dtype=float)
    _enforce_target_msh_volume(state)
    sep._update_duals_and_masses(state)
    sep._update_pressure_scalar(state)
    _assert_valid_water_tet_mesh(state, context="equilibrium final correction")
    _solve_total_force_balance_pressure(state)
    final_row = _history_row(state, step=int(history[-1]["step"]) if history else 0, time_s=sim_time_s)
    if history and int(history[-1]["step"]) == int(final_row["step"]) and abs(float(history[-1]["time_s"]) - float(final_row["time_s"])) <= 1.0e-30:
        history[-1] = final_row
    else:
        history.append(final_row)
    print(
        f"[equilibrium] final correction: "
        f"elapsed={time.perf_counter() - run_wall_start:.6f} s, "
        f"max_free_force={final_row['max_free_force_norm_n']:.6e} N, "
        f"net_total={final_row['net_total_force_norm_n']:.6e} N, "
        f"max_free_speed={final_row['max_free_speed_mps']:.6e} m/s, "
        f"{_format_snapshot_volume_status(final_row, target_volume_m3=target_volume_m3)}",
        flush=True,
    )
    if USER_OPEN_INTERACTIVE_WINDOW and not interactive_shown and int(USER_INTERACTIVE_STEP) == int(final_row["step"]):
        _show_interactive_snapshot(state, step=int(final_row["step"]))
    return history


def main() -> None:
    config = initialshape_config()
    state = sep._prepare_state(config)
    _build_volume_matched_analytic_initial_guess(state)

    out_dir = OUT_ROOT
    fig_dir = out_dir / "fig"
    result_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    _enforce_target_msh_volume(state)
    sep._update_duals_and_masses(state)
    sep._update_pressure_scalar(state)
    if USER_UNPIN_CONTACT_LINE:
        _project_contact_line_vertices_to_spheres(state)
        _enforce_target_msh_volume(state)
        sep._update_duals_and_masses(state)
        sep._update_pressure_scalar(state)
    _assert_valid_water_tet_mesh(state, context="initial guess")
    _solve_total_force_balance_pressure(state)
    initial_metrics = _free_force_metrics(state)
    initial_row = _history_row(state, step=0, time_s=0.0)
    print(
        "[initial guess] "
        f"max_free_force={float(initial_metrics['max_free_force_norm']):.6e} N, "
        f"net_total={float(np.linalg.norm(initial_metrics['net_total'])):.6e} N, "
        f"{_format_snapshot_volume_status(initial_row, target_volume_m3=float(config.initial_bridge_volume_m3))}",
        flush=True,
    )

    if USER_RENDER_INITIAL_GUESS:
        _render_initialshape_mesh_snapshot(
            state,
            f"{config.title}: raw initial guess",
            fig_dir / "mesh_initial_guess.png",
        )

    history = _relax_initial_shape(
        state,
        max_steps=USER_TOTAL_ITERATIONS,
        force_tol=USER_FORCE_TOL_N,
        total_force_tol=USER_TOTAL_FORCE_TOL_N,
        speed_tol=USER_SPEED_TOL_MPS,
        fig_dir=fig_dir,
    )
    final_metrics = _free_force_metrics(state)

    if USER_RENDER_FINAL_STATE:
        _render_initialshape_mesh_snapshot(
            state,
            f"{config.title}: equilibrium initial mesh",
            fig_dir / "mesh_initial.png",
        )

    _write_json(result_dir / "equilibrium_history.json", history)
    _write_json(
        result_dir / "equilibrium_summary.json",
        _summary_payload(
            state,
            initial_metrics=initial_metrics,
            final_metrics=final_metrics,
            history=history,
        ),
    )
    history_fig_paths = _render_equilibrium_history_figures(history, out_dir, dt=float(config.dt))

    final_row = history[-1]
    total_cl_rings = int(config.contact_line_radial_rings) + int(config.extra_cl_radial_rings)
    print("=" * 72)
    print("  Case 2b axisym initial equilibrium shape")
    print("=" * 72)
    print(f"Refinement                = {config.refinement}")
    print(f"Contact-line nodes        = {len(state.outer_rings[0])}")
    print(f"CL radial rings           = {config.contact_line_radial_rings}")
    print(f"Extra CL outer rings      = {config.extra_cl_radial_rings}")
    print(f"CL radial bias ratio      = {config.contact_line_radial_bias_ratio}")
    effective_radial_factors = sep._radial_ring_factors_with_outer_refinement(
        config.contact_line_radial_rings,
        config.extra_cl_radial_rings,
        config.contact_line_radial_bias_ratio,
    )
    print(
        "CL radial factors         = "
        + ", ".join(f"{float(factor):.6f}" for factor in effective_radial_factors)
    )
    print(f"Sidewall axial stride     = {config.axial_ring_stride}")
    print(f"Sidewall axial layers     = {len(state.outer_rings)}")
    print(f"Neck extra axial layers   = {config.neck_extra_axial_layers}")
    print(f"CL extra axial layers     = {config.cl_extra_axial_layers}")
    if total_cl_rings <= 1 and abs(float(config.contact_line_radial_bias_ratio) - 1.0) > 1.0e-12:
        print("CL radial bias active?    = no (need at least 2 total CL rings)")
    print(f"Contact line              = {'unpinned' if USER_UNPIN_CONTACT_LINE else 'pinned'}")
    print(f"Contact angle             = {config.contact_angle_deg:.3f} deg")
    print(f"Particle radius           = {config.particle_radius * 1.0e3:.3f} mm")
    print(f"Bridge contact radius     = {config.target_cap_radius * 1.0e3:.3f} mm")
    print(f"Target bridge volume      = {config.initial_bridge_volume_m3 * 1.0e9:.6f} uL")
    print(f"Surface tension           = {config.gamma:.4f} N/m")
    print(f"Viscosity                 = {config.mu_f:.3e} Pa s")
    print(f"Density                   = {config.rho_f:.3f} kg/m^3")
    print(f"Gravity acceleration      = {config.gravity_mps2:.3f} m/s^2")
    print(f"Relax dt                  = {config.dt:.3e} s")
    print(f"Total iterations          = {USER_TOTAL_ITERATIONS}")
    print(f"Acceleration workers      = {USER_ACCEL_WORKERS}")
    print(f"Mesh snapshot every       = {USER_MESH_SNAPSHOT_EVERY_ITERATIONS}")
    print(f"Final pressure scalar     = {state.pressure_scalar:.12e} Pa")
    print(f"Final gap                 = {sep._gap(state) * 1.0e6:.6f} um")
    print(f"Final max free-force norm = {float(final_row['max_free_force_norm_n']):.12e} N")
    print(f"Final total-force norm    = {float(final_row['net_total_force_norm_n']):.12e} N")
    print(f"Final max free speed      = {float(final_row['max_free_speed_mps']):.12e} m/s")
    print(f"Saved summary             = {result_dir / 'equilibrium_summary.json'}")
    print(f"Saved history             = {result_dir / 'equilibrium_history.json'}")
    if USER_RENDER_FINAL_STATE:
        print(f"Saved mesh PNG/MSH        = {fig_dir / 'mesh_initial.png'}")
    if history_fig_paths:
        print("Saved history PNGs        =")
        for path in history_fig_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
