"""Case 2b initial equilibrium shape for the Pitois 2000 Fig. 5 setup.

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
from concurrent.futures import ThreadPoolExecutor
import json
import math
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.operators.multiphase_stress import multiphase_stress_force

from cases_dynamic.liquid_bridge_coupled_collision import Case_2b_pitois2000_separation as sep
from cases_dynamic.liquid_bridge_coupled_collision.case_2b_axisym import (
    Case_2b_axisym_initialshape as hyperct_axisym_initialshape,
)


OUT_ROOT = Path(__file__).resolve().parent / "out" / "Case_2b_initialshape"

PDF_PARTICLE_RADIUS_M = 4.0e-3
# Pitois Fig. 5 first black filled marker on the log x-axis: D/R ~= 0.01445.
PDF_INITIAL_D_OVER_R = 0.01445729462258473
PDF_INITIAL_SEPARATION_D_M = PDF_INITIAL_D_OVER_R * PDF_PARTICLE_RADIUS_M
PDF_BRIDGE_VOLUME_M3 = 1.10e-9
PDF_SURFACE_TENSION_NPM = 21.0e-3
PDF_LIQUID2_VISCOSITY_PAS = 100.0e-3
PDF_CONTACT_ANGLE_DEG = 10.0


def _contact_radius_from_pitois_eq6(*, R: float, D: float, V: float) -> float:
    """Pitois Eq. [6]: V = pi R/2 * ([D + b^2/R]^2 - D^2), solved for b = r_CL."""

    b2 = -float(R) * float(D) + math.sqrt((float(R) * float(D)) ** 2 + 2.0 * float(R) * float(V) / math.pi)
    if b2 <= 0.0:
        raise ValueError("Pitois Eq. [6] produced non-positive r_CL^2.")
    return math.sqrt(b2)


PDF_CONTACT_LINE_RADIUS_M = _contact_radius_from_pitois_eq6(
    R=PDF_PARTICLE_RADIUS_M,
    D=PDF_INITIAL_SEPARATION_D_M,
    V=PDF_BRIDGE_VOLUME_M3,
)

# ---------------------------------------------------------------------------
# USER CONTROLS
# ---------------------------------------------------------------------------
USER_CONTACT_LINE_NODES = 8
# Exact number of radial rings used near the contact line.
USER_CONTACT_LINE_RADIAL_RINGS = 3
# Additional outer-band rings near the contact line.
USER_EXTRA_CL_RADIAL_RINGS = 4
# Extra axial rings inserted next to the two contact lines.
USER_CL_EXTRA_AXIAL_LAYERS = 12
# 1 keeps all sidewall axial rings. 2 keeps every other axial ring, etc.
# This is the actual waist-density control: increase it when the neck band is
# too dark from too many axial interface layers.
USER_SIDEWALL_AXIAL_RING_STRIDE = 1
# 1.0 = equal radial spacing in the outer CL band. Values above 1.0 make the
# mesh slightly finer right next to the contact line.
USER_CL_RADIAL_BIAS_RATIO = 1.1
# Liquid-air interface meridian sampling from contact line to bridge center:
# 1.1 means each inward radial gap is 1.1x the previous gap, so the interface
# mesh is finer at the CL and gradually coarser toward the neck.
USER_INTERFACE_RADIAL_DISTRIBUTION_RATIO = 1.1
# Second meshing stage after the CL-to-neck 1.1 distribution.
# Keep this at zero unless the neck is visibly under-resolved; nonzero values
# create the dark dense band at the bridge waist.
USER_INTERFACE_NECK_REFINEMENT_GAPS = 0
USER_INTERFACE_NECK_EXTRA_RINGS_PER_GAP = 0
# Real HyperCT axial layers inserted around the liquid-bridge neck before tetra
# connectivity is built. This is actual .msh refinement, not a display overlay.
USER_INTERFACE_NECK_EXTRA_AXIAL_LAYERS = (
    2 * USER_INTERFACE_NECK_REFINEMENT_GAPS * USER_INTERFACE_NECK_EXTRA_RINGS_PER_GAP
)
# Recovered HC run mode for the saved mesh_iter0100.msh:
# run the 100-iteration HyperCT initial-shape relaxation and write fig/mesh_iter0100.msh.
USER_GENERATE_MESH_LEVEL_SETS = False
USER_MESH_LEVEL_ORDER = (
    "fine",
    "medium_fine",
    "medium",
    "medium_coarse",
    "coarse",
    "coarser",
)
USER_RELAX_DT_S = 2.0e-5
USER_TOTAL_ITERATIONS = 100
USER_RELAX_DAMPING = 20.0
USER_FORCE_TOL_N = 1.0e-10
USER_TOTAL_FORCE_TOL_N = 1.0e-14
USER_SPEED_TOL_MPS = 1.0e-9
# The separation module default is 0.44, which gives a visibly pinched
# Case-2b initial bridge. This HyperCT initial-shape case starts from the
# smoother Case-2b axisymmetric profile instead.
USER_INITIAL_NECK_RADIUS_RATIO = 0.6424
USER_ACCEL_WORKERS = 1
USER_FORCE_METRIC_WORKERS = USER_ACCEL_WORKERS
USER_STEP_MAX_RETRIES = 3
USER_STEP_RETRY_DT_SCALE = 0.5
USER_HISTORY_EVERY = 10
# Expensive global correction; the final correction still always runs.
USER_PRESSURE_BALANCE_EVERY = USER_TOTAL_ITERATIONS
USER_LOG_EVERY = USER_HISTORY_EVERY
USER_MESH_SNAPSHOT_EVERY_ITERATIONS = 10
# Build the t=0 geometry with the HyperCT Young-Laplace/volume matcher before
# relaxation. This keeps the .msh tetrahedral water volume equal to Case 2b's
# prescribed 1.10 uL without using the Gmsh path.
USER_MATCH_INITIAL_MSH_VOLUME = True
USER_VOLUME_MATCH_REL_TOL = 1.0e-6
USER_VOLUME_CORRECTION_EVERY = USER_MESH_SNAPSHOT_EVERY_ITERATIONS
# Fast path: avoid the old radial target-volume rescale entirely. Enable the
# continuity pressure only when stricter volume control is worth the extra
# force evaluations.
USER_ENABLE_CONTINUITY_PRESSURE = False
USER_UNPIN_CONTACT_LINE = False
USER_CONTACT_ANGLE_DEG = PDF_CONTACT_ANGLE_DEG
USER_RENDER_INITIAL_GUESS = True
USER_RENDER_FINAL_STATE = True
USER_OPEN_INTERACTIVE_WINDOW = False
USER_INTERACTIVE_STEP = USER_TOTAL_ITERATIONS


def _current_mesh_spec() -> dict[str, int | float | str]:
    return {
        "name": "medium",
        "contact_line_nodes": int(USER_CONTACT_LINE_NODES),
        "contact_line_radial_rings": int(USER_CONTACT_LINE_RADIAL_RINGS),
        "extra_cl_radial_rings": int(USER_EXTRA_CL_RADIAL_RINGS),
        "cl_extra_axial_layers": int(USER_CL_EXTRA_AXIAL_LAYERS),
        "axial_ring_stride": int(USER_SIDEWALL_AXIAL_RING_STRIDE),
        "cl_radial_bias_ratio": float(USER_CL_RADIAL_BIAS_RATIO),
        "interface_radial_distribution_ratio": float(USER_INTERFACE_RADIAL_DISTRIBUTION_RATIO),
        "interface_neck_refinement_gaps": int(USER_INTERFACE_NECK_REFINEMENT_GAPS),
        "interface_neck_extra_rings_per_gap": int(USER_INTERFACE_NECK_EXTRA_RINGS_PER_GAP),
        "interface_neck_extra_axial_layers": int(USER_INTERFACE_NECK_EXTRA_AXIAL_LAYERS),
    }


def _mesh_level_specs() -> dict[str, dict[str, int | float | str]]:
    medium = _current_mesh_spec()

    def spec(name: str, **changes) -> dict[str, int | float | str]:
        out = dict(medium)
        out["name"] = name
        out.update(changes)
        out["interface_neck_extra_axial_layers"] = int(
            changes.get(
                "interface_neck_extra_axial_layers",
                2
                * int(out["interface_neck_refinement_gaps"])
                * int(out["interface_neck_extra_rings_per_gap"]),
            )
        )
        return out

    return {
        "fine": spec(
            "fine",
            contact_line_nodes=16,
            axial_ring_stride=1,
            extra_cl_radial_rings=6,
            cl_extra_axial_layers=28,
            interface_neck_refinement_gaps=0,
            interface_neck_extra_rings_per_gap=0,
        ),
        "medium_fine": spec(
            "medium_fine",
            contact_line_nodes=8,
            axial_ring_stride=1,
            extra_cl_radial_rings=6,
            cl_extra_axial_layers=22,
            interface_neck_refinement_gaps=0,
            interface_neck_extra_rings_per_gap=0,
        ),
        "medium": medium,
        "medium_coarse": spec(
            "medium_coarse",
            contact_line_nodes=8,
            axial_ring_stride=1,
            extra_cl_radial_rings=3,
            cl_extra_axial_layers=8,
            interface_neck_refinement_gaps=0,
            interface_neck_extra_rings_per_gap=0,
        ),
        "coarse": spec(
            "coarse",
            contact_line_nodes=4,
            axial_ring_stride=1,
            extra_cl_radial_rings=2,
            cl_extra_axial_layers=6,
            interface_neck_refinement_gaps=0,
            interface_neck_extra_rings_per_gap=0,
        ),
        "coarser": spec(
            "coarser",
            contact_line_nodes=4,
            contact_line_radial_rings=2,
            axial_ring_stride=1,
            extra_cl_radial_rings=1,
            cl_extra_axial_layers=3,
            interface_neck_refinement_gaps=0,
            interface_neck_extra_rings_per_gap=0,
        ),
    }


def _mesh_spec_value(mesh_spec: dict | None, key: str, default):
    if mesh_spec is not None and key in mesh_spec:
        return mesh_spec[key]
    return default


def _separation_view_kwargs() -> dict[str, float]:
    """Always reuse the current separation-camera settings for initial-shape views."""
    return {
        "elev": float(sep.USER_INTERACTIVE_ELEV_DEG),
        "azim": float(sep.USER_INTERACTIVE_AZIM_DEG),
    }


_ORIGINAL_VOLUME_EXPORT_VERTEX_ORDER = sep._volume_export_vertex_order
_ORIGINAL_MOVE = sep._move
_ORIGINAL_AXISYM_MOVE = hyperct_axisym_initialshape.sep._move


def _initialshape_volume_export_vertex_order(state: sep.VolumetricPitoisState) -> list:
    """Keep restored tetrahedral export valid when the PDF r_CL changes the layer graph."""

    vertices = list(_ORIGINAL_VOLUME_EXPORT_VERTEX_ORDER(state))
    seen = {id(vertex) for vertex in vertices}
    for vertex in list(state.layer_centers):
        if id(vertex) not in seen:
            vertices.append(vertex)
            seen.add(id(vertex))
    for rings in list(state.layer_rings):
        for ring in list(rings):
            for vertex in list(ring):
                if id(vertex) not in seen:
                    vertices.append(vertex)
                    seen.add(id(vertex))
    return vertices


def _install_initialshape_export_order() -> None:
    sep._volume_export_vertex_order = _initialshape_volume_export_vertex_order


def _safe_hyperct_move(original_move, vertex, pos, HC, bV) -> None:
    # HyperCT's coordinate cache can miss coincident center vertices at PDF D = 0.
    # Reinsert the old coordinate before moving so the restored BVP path survives.
    old_key = tuple(vertex.x)
    if old_key not in HC.V.cache:
        HC.V.cache[old_key] = vertex
    original_move(vertex, pos, HC, bV)


def _initialshape_safe_move(vertex, pos, HC, bV) -> None:
    _safe_hyperct_move(_ORIGINAL_MOVE, vertex, pos, HC, bV)


def _axisym_initialshape_safe_move(vertex, pos, HC, bV) -> None:
    _safe_hyperct_move(_ORIGINAL_AXISYM_MOVE, vertex, pos, HC, bV)


def _install_initialshape_safe_move() -> None:
    sep._move = _initialshape_safe_move
    hyperct_axisym_initialshape.sep._move = _axisym_initialshape_safe_move


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


def _replace_supported_config(base: sep.VolumetricPitoisConfig, **changes) -> sep.VolumetricPitoisConfig:
    fields = getattr(base, "__dataclass_fields__", {})
    return replace(base, **{key: value for key, value in changes.items() if key in fields})


def initialshape_config(mesh_spec: dict | None = None) -> sep.VolumetricPitoisConfig:
    # Geometry is fixed from the Pitois 2000 PDF: R, D, V, and r_CL from Eq. [6].
    # Keep the restored relaxation/snapshot machinery, but override these values here.
    return _replace_supported_config(
        sep.separation_config(),
        name="Case_2b_initialshape",
        title="Case 2b: equilibrium initial shape",
        use_last_initialshape_msh=False,
        refinement=_refinement_from_contact_line_nodes(_mesh_spec_value(mesh_spec, "contact_line_nodes", USER_CONTACT_LINE_NODES)),
        contact_line_radial_rings=int(_mesh_spec_value(mesh_spec, "contact_line_radial_rings", USER_CONTACT_LINE_RADIAL_RINGS)),
        extra_cl_radial_rings=int(_mesh_spec_value(mesh_spec, "extra_cl_radial_rings", USER_EXTRA_CL_RADIAL_RINGS)),
        cl_extra_axial_layers=int(_mesh_spec_value(mesh_spec, "cl_extra_axial_layers", USER_CL_EXTRA_AXIAL_LAYERS)),
        axial_ring_stride=int(_mesh_spec_value(mesh_spec, "axial_ring_stride", USER_SIDEWALL_AXIAL_RING_STRIDE)),
        contact_line_radial_bias_ratio=float(_mesh_spec_value(mesh_spec, "cl_radial_bias_ratio", USER_CL_RADIAL_BIAS_RATIO)),
        particle_radius=float(PDF_PARTICLE_RADIUS_M),
        target_cap_radius=float(PDF_CONTACT_LINE_RADIUS_M),
        initial_d_over_r=float(PDF_INITIAL_SEPARATION_D_M / max(PDF_PARTICLE_RADIUS_M, 1.0e-30)),
        initial_bridge_volume_m3=float(PDF_BRIDGE_VOLUME_M3),
        cap_speed=0.0,
        dt=USER_RELAX_DT_S,
        n_steps=0,
        damping=USER_RELAX_DAMPING,
        integration_substeps=1,
        enable_initial_relaxation=False,
        enable_continuity_pressure=USER_ENABLE_CONTINUITY_PRESSURE,
        initial_neck_radius_ratio=float(USER_INITIAL_NECK_RADIUS_RATIO),
        gamma=float(PDF_SURFACE_TENSION_NPM),
        mu_f=float(PDF_LIQUID2_VISCOSITY_PAS),
        allow_contact_line_growth=USER_UNPIN_CONTACT_LINE,
        contact_angle_deg=USER_CONTACT_ANGLE_DEG,
        record_every=1,
        mesh_snapshot_every=1,
        match_initial_force_to_fig5=False,
        enable_ns_pressure_projection=False,
    )


def _prepare_initialshape_state(
    config: sep.VolumetricPitoisConfig,
    mesh_spec: dict | None = None,
) -> sep.VolumetricPitoisState:
    neck_extra_layers = max(
        0,
        int(
            _mesh_spec_value(
                mesh_spec,
                "interface_neck_extra_axial_layers",
                USER_INTERFACE_NECK_EXTRA_AXIAL_LAYERS,
            )
        ),
    )
    if neck_extra_layers <= 0:
        return sep._prepare_state(config)

    original_builder = sep._build_structured_volumetric_catenoid

    def build_with_neck_refinement(
        refinement,
        radial_ring_factors_override=None,
        axial_ring_stride_override: int = 1,
        neck_extra_axial_layers_override: int = 0,
        contact_extra_axial_layers_override: int = 0,
    ):
        # ENFORCE NECK REFINEMENT ON ACTUAL HYPERCT MESH:
        # insert real liquid-air interface layers before tetra connectivity exists.
        return original_builder(
            refinement,
            radial_ring_factors_override=radial_ring_factors_override,
            axial_ring_stride_override=axial_ring_stride_override,
            neck_extra_axial_layers_override=max(
                int(neck_extra_axial_layers_override),
                neck_extra_layers,
            ),
            contact_extra_axial_layers_override=contact_extra_axial_layers_override,
        )

    sep._build_structured_volumetric_catenoid = build_with_neck_refinement
    try:
        return sep._prepare_state(config)
    finally:
        sep._build_structured_volumetric_catenoid = original_builder


def _static_vertex_force(v, *, state: sep.VolumetricPitoisState) -> np.ndarray:
    return multiphase_stress_force(
        v,
        dim=3,
        mps=state.mps,
        HC=state.HC,
        pressure_model=lambda vv, HC=None, dim=3: sep._pressure_model(vv, HC=HC, dim=dim, state=state),
    )


def _maybe_update_continuity_pressure_scalar(state: sep.VolumetricPitoisState, *, dt: float) -> None:
    updater = getattr(sep, "_update_continuity_pressure_scalar", None)
    if updater is not None:
        updater(state, dt=float(dt))


def _free_force_metrics(state: sep.VolumetricPitoisState) -> dict[str, np.ndarray | float]:
    free_vertices = [v for v in state.HC.V if v not in state.bV_caps]
    workers = max(1, int(USER_FORCE_METRIC_WORKERS))
    if workers > 1 and len(free_vertices) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            free_forces = list(pool.map(lambda v: _static_vertex_force(v, state=state), free_vertices))
    else:
        free_forces = [_static_vertex_force(v, state=state) for v in free_vertices]
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
    max_iters: int = 2,
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


def _history_row_from_metrics(
    state: sep.VolumetricPitoisState,
    *,
    step: int,
    metrics: dict,
    time_s: float | None = None,
) -> dict[str, float | int]:
    return {
        "step": int(step),
        "time_s": float(step * state.config.dt if time_s is None else time_s),
        "gap_um": float(sep._gap(state) * 1.0e6),
        "pressure_scalar_pa": float(state.pressure_scalar),
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


def _history_row(state: sep.VolumetricPitoisState, *, step: int, time_s: float | None = None) -> dict[str, float | int]:
    return _history_row_from_metrics(
        state,
        step=step,
        time_s=time_s,
        metrics=_free_force_metrics(state),
    )


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


def _set_target_msh_volume_metadata(state: sep.VolumetricPitoisState) -> float:
    target = float(state.config.initial_bridge_volume_m3)
    state.target_snapshot_volume_m3 = target
    state.target_volume_m3 = target
    return float(sep._snapshot_msh_volume_m3(state))


def _pitois_eq3_half_span(config: sep.VolumetricPitoisConfig) -> float:
    R = float(config.particle_radius)
    D = float(config.initial_d_over_r) * R
    r_cl = float(config.target_cap_radius)
    cos_phi = math.sqrt(max(1.0 - (r_cl / max(R, 1.0e-30)) ** 2, 0.0))
    return 0.5 * D + R * (1.0 - cos_phi)


def _pitois_eq3_sphere_radius_sq(z: np.ndarray, config: sep.VolumetricPitoisConfig) -> np.ndarray:
    R = float(config.particle_radius)
    D = float(config.initial_d_over_r) * R
    top_center_z = 0.5 * D + R
    z = np.asarray(z, dtype=float)
    return np.maximum(R * R - (z - top_center_z) ** 2, 0.0)


def _solve_pitois_eq3_half_profile(
    config: sep.VolumetricPitoisConfig,
    z_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool | str]]:
    """Solve Pitois Eq. [3] for the symmetric half profile r(z)."""

    z_cl = _pitois_eq3_half_span(config)
    r_cl = float(config.target_cap_radius)
    target_half_volume = 0.5 * float(config.initial_bridge_volume_m3)
    if z_cl <= 0.0 or r_cl <= 0.0 or target_half_volume <= 0.0:
        raise RuntimeError("Pitois Eq. [3] needs positive z_CL, r_CL, and half-volume.")

    z_nodes = np.linspace(0.0, z_cl, 300, dtype=float)
    sphere_sq = _pitois_eq3_sphere_radius_sq(z_nodes, config)
    cap_half_volume = math.pi * float(np.trapz(sphere_sq, z_nodes))
    avg_r2 = max((target_half_volume + cap_half_volume) / (math.pi * z_cl), 1.0e-30)
    avg_r = math.sqrt(avg_r2)

    def ode(z: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        two_gamma = float(p[0])
        r = np.maximum(y[0], 1.0e-12)
        slope = y[1]
        one_plus = 1.0 + slope * slope
        sphere_sq_local = _pitois_eq3_sphere_radius_sq(z, config)
        # Eq. [3]: 2Γ = 1/(r sqrt(1+r'^2)) - r''/(1+r'^2)^(3/2).
        return np.vstack(
            [
                slope,
                one_plus / r - two_gamma * np.power(one_plus, 1.5),
                math.pi * (r * r - sphere_sq_local),
            ]
        )

    def bc(ya: np.ndarray, yb: np.ndarray, p: np.ndarray) -> np.ndarray:
        del p
        return np.array(
            [
                ya[1],
                ya[2],
                yb[0] - r_cl,
                yb[2] - target_half_volume,
            ],
            dtype=float,
        )

    best_sol = None
    guesses = [
        min(0.20e-3, 0.8 * r_cl),
        min(0.50e-3, 0.9 * r_cl),
        min(0.80e-3, 0.95 * r_cl),
        min(avg_r, 0.995 * r_cl),
        0.995 * r_cl,
    ]
    for neck_guess in guesses:
        xi = z_nodes / z_cl
        r_guess = float(neck_guess) + (r_cl - float(neck_guess)) * xi * xi
        slope_guess = 2.0 * (r_cl - float(neck_guess)) * xi / z_cl
        volume_integrand = math.pi * (r_guess * r_guess - sphere_sq)
        volume_guess = np.concatenate(
            [
                [0.0],
                np.cumsum(0.5 * (volume_integrand[1:] + volume_integrand[:-1]) * np.diff(z_nodes)),
            ]
        )
        y_guess = np.vstack([r_guess, slope_guess, volume_guess])
        sol = solve_bvp(
            ode,
            bc,
            z_nodes,
            y_guess,
            p=np.array([1.0 / max(avg_r, 1.0e-30)], dtype=float),
            tol=1.0e-5,
            max_nodes=50000,
        )
        if sol.success:
            best_sol = sol
            break

    if best_sol is None:
        raise RuntimeError("Pitois Eq. [3] BVP failed to converge for the initial bridge shape.")

    z_eval = np.asarray(z_eval, dtype=float)
    z_eval_used = np.clip(z_eval, 0.0, z_cl)
    ratio = float(USER_INTERFACE_RADIAL_DISTRIBUTION_RATIO)
    if ratio <= 0.0 or not np.isfinite(ratio):
        raise ValueError("USER_INTERFACE_RADIAL_DISTRIBUTION_RATIO must be positive and finite.")
    unique_original = np.asarray(sorted({round(float(value), 15) for value in z_eval_used}), dtype=float)
    if unique_original.size >= 2:
        # ENFORCE CL REFINEMENT ON ACTUAL HYPERCT MESH:
        # CL-to-neck liquid-air interface radial gaps follow USER_INTERFACE_RADIAL_DISTRIBUTION_RATIO.
        z_dense = np.linspace(0.0, z_cl, 5000, dtype=float)
        r_dense = np.asarray(best_sol.sol(z_dense)[0], dtype=float)
        r_dense = np.maximum.accumulate(np.clip(r_dense, 1.0e-8 * r_cl, r_cl))
        neck_radius = float(r_dense[0])
        radial_span = max(r_cl - neck_radius, 0.0)
        n_gaps = unique_original.size - 1
        gap_weights_from_cl = np.power(ratio, np.arange(n_gaps, dtype=float))
        radial_fraction_from_cl = np.concatenate(
            [[0.0], np.cumsum(gap_weights_from_cl) / float(np.sum(gap_weights_from_cl))]
        )
        target_r_from_cl = r_cl - radial_fraction_from_cl * radial_span
        target_r_center_to_cl = target_r_from_cl[::-1]
        target_r_center_to_cl[0] = neck_radius
        target_r_center_to_cl[-1] = r_cl
        unique_adjusted = np.interp(target_r_center_to_cl, r_dense, z_dense)
        unique_adjusted[0] = 0.0
        unique_adjusted[-1] = z_cl
        z_map = {round(float(old), 15): float(new) for old, new in zip(unique_original, unique_adjusted)}
        z_eval_used = np.asarray([z_map[round(float(value), 15)] for value in z_eval_used], dtype=float)

    r_eval = np.asarray(best_sol.sol(np.clip(z_eval_used, 0.0, z_cl))[0], dtype=float)
    r_eval = np.clip(r_eval, 1.0e-8 * r_cl, r_cl)
    r_eval[np.isclose(z_eval_used, z_cl, rtol=0.0, atol=1.0e-12)] = r_cl

    metadata = {
        "shape_equation": "Pitois Eq. [3], constant-mean-curvature Young-Laplace profile r(z)",
        "interface_radial_distribution_ratio": float(ratio),
        "eq3_two_gamma_1_per_m": float(best_sol.p[0]),
        "eq3_neck_radius_mm": float(best_sol.y[0, 0] * 1.0e3),
        "eq3_contact_slope_dr_dz": float(best_sol.y[1, -1]),
        "eq3_half_span_mm": float(z_cl * 1.0e3),
        "eq3_half_liquid_volume_ul": float(best_sol.y[2, -1] * 1.0e9),
        "eq3_cap_half_volume_ul": float(cap_half_volume * 1.0e9),
        "eq3_bvp_nodes": int(best_sol.x.size),
        "eq3_bvp_success": bool(best_sol.success),
        "eq3_bvp_message": str(best_sol.message),
    }
    return z_eval_used, r_eval, metadata


def _apply_pitois_eq3_initial_guess(state: sep.VolumetricPitoisState) -> dict[str, float | int | bool | str]:
    template = sep._initial_axisymmetric_template(state)
    mid, axis, e1, e2, layer_data = template
    s_vals = np.array([float(item["s"]) for item in layer_data], dtype=float)
    z_abs = np.abs(s_vals)
    z_abs_adjusted, outer_radii, metadata = _solve_pitois_eq3_half_profile(state.config, z_abs)
    s_adjusted = np.sign(s_vals) * z_abs_adjusted

    for layer, signed_z, outer_radius in zip(layer_data, s_adjusted, outer_radii):
        center_coord = mid + float(signed_z) * axis
        sep._move(layer["center_vertex"], tuple(center_coord), state.HC, state.bV_caps)
        for ordered_vertices, factor in layer["rings"]:
            ring_radius = max(float(factor) * float(outer_radius), 0.0)
            n_ring = max(1, len(ordered_vertices))
            targets = []
            for i in range(n_ring):
                phi = 2.0 * math.pi * float(i) / float(n_ring)
                target = center_coord + ring_radius * (math.cos(phi) * e1 + math.sin(phi) * e2)
                targets.append(tuple(target))
            sep._move_vertices_batch(ordered_vertices, targets, state.HC, state.bV_caps)

    contact_radius = float(state.config.target_cap_radius)
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
    axial_span = float(s_adjusted[-1] - s_adjusted[0])
    if axial_span > 1.0e-30:
        state.layer_fractions = tuple(
            float(np.clip((float(s_val) - float(s_adjusted[0])) / axial_span, 0.0, 1.0))
            for s_val in s_adjusted
        )
    state.pitois_eq3_initial_profile = metadata
    return metadata


def _interface_outer_radius(layer: dict) -> float:
    outer_vertices = list(layer["rings"][-1][0])
    center = np.asarray(layer["center_vertex"].x_a[:3], dtype=float)
    if not outer_vertices:
        return 0.0
    return float(np.mean([np.linalg.norm(np.asarray(vertex.x_a[:3], dtype=float) - center) for vertex in outer_vertices]))


def _geometric_interface_radii_from_center(neck_radius: float, contact_radius: float, n_layers: int) -> np.ndarray:
    n_layers = int(n_layers)
    if n_layers <= 1:
        return np.asarray([float(contact_radius)], dtype=float)
    n_gaps = n_layers - 1
    ratio = float(USER_INTERFACE_RADIAL_DISTRIBUTION_RATIO)
    if ratio <= 0.0 or not np.isfinite(ratio):
        raise ValueError("USER_INTERFACE_RADIAL_DISTRIBUTION_RATIO must be positive and finite.")
    weights_from_cl = np.power(ratio, np.arange(n_gaps, dtype=float))
    fractions_from_cl = np.concatenate([[0.0], np.cumsum(weights_from_cl) / float(np.sum(weights_from_cl))])
    from_cl = float(contact_radius) - fractions_from_cl * (float(contact_radius) - float(neck_radius))
    from_center = from_cl[::-1]
    from_center[0] = float(neck_radius)
    from_center[-1] = float(contact_radius)
    return from_center


def _apply_interface_radial_distribution_for_neck(
    state: sep.VolumetricPitoisState,
    *,
    neck_radius: float,
) -> None:
    template = sep._initial_axisymmetric_template(state)
    _mid, _axis, e1, e2, layer_data = template
    if len(layer_data) < 3:
        return
    center_idx = min(range(len(layer_data)), key=lambda idx: abs(float(layer_data[idx]["s"])))
    contact_radius = float(state.config.target_cap_radius)

    upper_indices = list(range(center_idx, len(layer_data)))
    lower_indices = list(range(center_idx, -1, -1))
    upper_radii = _geometric_interface_radii_from_center(neck_radius, contact_radius, len(upper_indices))
    lower_radii = _geometric_interface_radii_from_center(neck_radius, contact_radius, len(lower_indices))

    target_outer_by_idx: dict[int, float] = {}
    for idx, radius in zip(upper_indices, upper_radii):
        target_outer_by_idx[idx] = float(radius)
    for idx, radius in zip(lower_indices, lower_radii):
        target_outer_by_idx[idx] = float(radius)

    for idx, layer in enumerate(layer_data):
        center_coord = np.asarray(layer["center_vertex"].x_a[:3], dtype=float)
        outer_radius = target_outer_by_idx.get(idx)
        if outer_radius is None:
            continue
        for ordered_vertices, factor in layer["rings"]:
            ring_radius = max(float(factor) * float(outer_radius), 0.0)
            n_ring = max(1, len(ordered_vertices))
            targets = []
            for i in range(n_ring):
                phi = 2.0 * math.pi * float(i) / float(n_ring)
                target = center_coord + ring_radius * (math.cos(phi) * e1 + math.sin(phi) * e2)
                targets.append(tuple(target))
            sep._move_vertices_batch(ordered_vertices, targets, state.HC, state.bV_caps)
    sep._align_layer_azimuths(state)


def _enforce_interface_radial_distribution(
    state: sep.VolumetricPitoisState,
    *,
    target_volume_m3: float | None = None,
) -> float:
    template = sep._initial_axisymmetric_template(state)
    layer_data = template[4]
    center_idx = min(range(len(layer_data)), key=lambda idx: abs(float(layer_data[idx]["s"])))
    contact_radius = float(state.config.target_cap_radius)
    current_neck_radius = _interface_outer_radius(layer_data[center_idx])
    if target_volume_m3 is None:
        _apply_interface_radial_distribution_for_neck(state, neck_radius=current_neck_radius)
        return float(sep._snapshot_msh_volume_m3(state))

    target = float(target_volume_m3)
    base_snapshot = _capture_state_snapshot(state)

    def restore_base() -> None:
        _restore_state_snapshot(state, base_snapshot)

    def volume_for_neck(neck_radius: float) -> float:
        restore_base()
        _apply_interface_radial_distribution_for_neck(state, neck_radius=float(neck_radius))
        return float(sep._snapshot_msh_volume_m3(state))

    low = 1.0e-8 * contact_radius
    high = contact_radius
    v_low = volume_for_neck(low)
    v_high = volume_for_neck(high)
    if not (min(v_low, v_high) <= target <= max(v_low, v_high)):
        restore_base()
        chosen = low if abs(v_low - target) <= abs(v_high - target) else high
        _apply_interface_radial_distribution_for_neck(state, neck_radius=chosen)
        return float(sep._snapshot_msh_volume_m3(state))

    increasing = v_high >= v_low
    for _ in range(36):
        mid = 0.5 * (low + high)
        v_mid = volume_for_neck(mid)
        if (v_mid < target) == increasing:
            low = mid
        else:
            high = mid

    restore_base()
    _apply_interface_radial_distribution_for_neck(state, neck_radius=0.5 * (low + high))
    return float(sep._snapshot_msh_volume_m3(state))


def _match_initial_hyperct_msh_volume(state: sep.VolumetricPitoisState) -> float:
    target = float(state.config.initial_bridge_volume_m3)
    if not bool(USER_MATCH_INITIAL_MSH_VOLUME):
        return _set_target_msh_volume_metadata(state)

    _apply_pitois_eq3_initial_guess(state)
    sep._update_duals_and_masses(state)
    sep._update_pressure_scalar(state)
    _correct_current_hyperct_msh_volume(state)
    sep._update_duals_and_masses(state)
    sep._update_pressure_scalar(state)

    current = _set_target_msh_volume_metadata(state)
    rel_error = abs(current - target) / max(target, 1.0e-30)
    if rel_error > float(USER_VOLUME_MATCH_REL_TOL):
        raise RuntimeError(
            "HyperCT initial .msh volume mismatch: "
            f"target={1.0e9 * target:.9f} uL, current={1.0e9 * current:.9f} uL"
        )
    return current


def _correct_current_hyperct_msh_volume(state: sep.VolumetricPitoisState) -> float:
    target = float(state.config.initial_bridge_volume_m3)
    state.target_snapshot_volume_m3 = target
    state.target_volume_m3 = target
    if not bool(USER_MATCH_INITIAL_MSH_VOLUME):
        return float(sep._snapshot_msh_volume_m3(state))
    return float(
        hyperct_axisym_initialshape._enforce_target_msh_volume(
            state,
            rel_tol=float(USER_VOLUME_MATCH_REL_TOL),
            max_iters=32,
        )
    )


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _vertex_xyz(vertex) -> np.ndarray:
    return np.asarray(vertex.x_a[:3], dtype=float)


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-30 or not np.isfinite(norm):
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return arr / norm


def _orthonormal_basis(axis: np.ndarray, reference: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    axis = _unit_vector(axis)
    if reference is None:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    ref = np.asarray(reference, dtype=float)
    ref = ref - float(np.dot(ref, axis)) * axis
    if float(np.linalg.norm(ref)) <= 1.0e-30:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(fallback, axis))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        ref = fallback - float(np.dot(fallback, axis)) * axis
    e1 = _unit_vector(ref)
    e2 = _unit_vector(np.cross(axis, e1))
    return e1, e2


def _liquid_interface_vertex_rings(state: sep.VolumetricPitoisState) -> list[list]:
    return [list(ring) for ring in state.outer_rings]


def _ordered_ring_array(
    ring: list,
    *,
    axis: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> np.ndarray:
    coords = np.asarray([_vertex_xyz(vertex) for vertex in ring], dtype=float)
    if coords.size == 0:
        return coords.reshape((0, 3))
    center = np.mean(coords, axis=0)
    rel = coords - center[None, :]
    rel = rel - np.outer(rel @ axis, axis)
    angles = np.arctan2(rel @ e2, rel @ e1)
    return coords[np.argsort(angles)]


def _ring_radius_about_axis(ring: np.ndarray, center: np.ndarray, axis: np.ndarray) -> float:
    rel = np.asarray(ring, dtype=float) - np.asarray(center, dtype=float)[None, :]
    rel = rel - np.outer(rel @ axis, axis)
    radii = np.linalg.norm(rel, axis=1)
    if radii.size == 0:
        return 0.0
    return float(np.mean(radii))


def _monotone_radius_to_s_profile(radii: np.ndarray, s_abs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    profile_r: list[float] = []
    profile_s: list[float] = []
    running = -np.inf
    for rr, ss in zip(np.asarray(radii, dtype=float), np.asarray(s_abs, dtype=float)):
        running = max(running, float(rr))
        if not profile_r or running > profile_r[-1] + 1.0e-15:
            profile_r.append(running)
            profile_s.append(float(ss))
        else:
            profile_s[-1] = max(profile_s[-1], float(ss))
    if len(profile_r) == 1:
        profile_r.append(profile_r[0] + 1.0e-15)
        profile_s.append(profile_s[0])
    return np.asarray(profile_r, dtype=float), np.asarray(profile_s, dtype=float)


def _resample_half_interface_rings(
    *,
    n_layers: int,
    sign: float,
    profile_radii: np.ndarray,
    profile_s_abs: np.ndarray,
    neck_radius: float,
    contact_radius: float,
    mid: np.ndarray,
    axis: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    angles: np.ndarray,
) -> list[np.ndarray]:
    if n_layers <= 0:
        return []
    if n_layers == 1:
        target_radii = np.asarray([float(contact_radius)], dtype=float)
    else:
        target_radii = _geometric_interface_radii_from_center(neck_radius, contact_radius, n_layers)
    profile_r, profile_s = _monotone_radius_to_s_profile(profile_radii, profile_s_abs)
    target_s_abs = np.interp(target_radii, profile_r, profile_s)
    target_s_abs[0] = 0.0
    target_s_abs[-1] = float(profile_s_abs[-1])

    rings: list[np.ndarray] = []
    for radius, s_abs in zip(target_radii, target_s_abs):
        center = np.asarray(mid, dtype=float) + float(sign) * float(s_abs) * np.asarray(axis, dtype=float)
        ring = np.asarray(
            [
                center + float(radius) * (math.cos(float(phi)) * e1 + math.sin(float(phi)) * e2)
                for phi in angles
            ],
            dtype=float,
        )
        rings.append(ring)
    return rings


def _add_neck_refinement_rings(rings: list[np.ndarray], center_idx: int) -> list[np.ndarray]:
    extra_per_gap = max(0, int(USER_INTERFACE_NECK_EXTRA_RINGS_PER_GAP))
    span_gaps = max(0, int(USER_INTERFACE_NECK_REFINEMENT_GAPS))
    if extra_per_gap <= 0 or span_gaps <= 0 or len(rings) < 3:
        return rings

    refined: list[np.ndarray] = []
    for idx in range(len(rings) - 1):
        refined.append(np.asarray(rings[idx], dtype=float))
        gap_is_near_neck = abs(idx - int(center_idx)) < span_gaps or abs((idx + 1) - int(center_idx)) < span_gaps
        if gap_is_near_neck:
            for insert_idx in range(1, extra_per_gap + 1):
                t = float(insert_idx) / float(extra_per_gap + 1)
                refined.append((1.0 - t) * np.asarray(rings[idx], dtype=float) + t * np.asarray(rings[idx + 1], dtype=float))
    refined.append(np.asarray(rings[-1], dtype=float))
    return refined


def _interface_ring_arrays(state: sep.VolumetricPitoisState) -> list[np.ndarray]:
    vertex_rings = _liquid_interface_vertex_rings(state)
    if not vertex_rings:
        return []
    centers_raw = [
        np.mean(np.asarray([_vertex_xyz(vertex) for vertex in ring], dtype=float), axis=0)
        for ring in vertex_rings
        if ring
    ]
    if len(centers_raw) != len(vertex_rings):
        return []
    axis = _unit_vector(np.asarray(state.top_sphere_center, dtype=float) - np.asarray(state.bottom_sphere_center, dtype=float))
    reference = _vertex_xyz(vertex_rings[0][0]) - centers_raw[0]
    e1, e2 = _orthonormal_basis(axis, reference)
    ordered = [_ordered_ring_array(ring, axis=axis, e1=e1, e2=e2) for ring in vertex_rings]
    centers = np.asarray([np.mean(ring, axis=0) for ring in ordered], dtype=float)
    mid = 0.5 * (centers[0] + centers[-1])
    s_vals = np.asarray([(center - mid) @ axis for center in centers], dtype=float)
    order = np.argsort(s_vals)
    return [ordered[int(idx)] for idx in order]


def _ring_array_segments(rings: list[np.ndarray]) -> np.ndarray:
    segments = []
    for ring in rings:
        ring = np.asarray(ring, dtype=float)
        if len(ring) < 2:
            continue
        for i in range(len(ring)):
            segments.append([ring[i], ring[(i + 1) % len(ring)]])
    for lower, upper in zip(rings[:-1], rings[1:]):
        n_ring = min(len(lower), len(upper))
        for i in range(n_ring):
            segments.append([lower[i], upper[i]])
    return np.asarray(segments, dtype=float)


def _write_interface_triangle_msh(rings: list[np.ndarray], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_rings = [np.asarray(ring, dtype=float) for ring in rings if len(ring) >= 3]
    if len(valid_rings) < 2:
        points = np.empty((0, 3), dtype=float)
        triangles: list[tuple[int, int, int]] = []
    else:
        n_phi = min(len(ring) for ring in valid_rings)
        valid_rings = [ring[:n_phi] for ring in valid_rings]
        points = np.vstack(valid_rings)

        def idx(layer_idx: int, angular_idx: int) -> int:
            return layer_idx * n_phi + (angular_idx % n_phi)

        triangles = []
        for layer_idx in range(len(valid_rings) - 1):
            for angular_idx in range(n_phi):
                a = idx(layer_idx, angular_idx)
                b = idx(layer_idx, angular_idx + 1)
                c = idx(layer_idx + 1, angular_idx)
                d = idx(layer_idx + 1, angular_idx + 1)
                triangles.append((a, c, b))
                triangles.append((b, c, d))

    with path.open("w", encoding="utf-8") as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write(f"{len(points)}\n")
        for node_id, xyz in enumerate(points, start=1):
            f.write(f"{node_id} {xyz[0]:.17e} {xyz[1]:.17e} {xyz[2]:.17e}\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write(f"{len(triangles)}\n")
        for elem_id, tri in enumerate(triangles, start=1):
            a, b, c = (idx_ + 1 for idx_ in tri)
            f.write(f"{elem_id} 2 2 1 1 {a} {b} {c}\n")
        f.write("$EndElements\n")
    return path


def _write_surface_triangle_msh(state: sep.VolumetricPitoisState, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    export_vertices = list(state.volume_export_vertices)
    export_points = [_vertex_xyz(vertex) for vertex in export_vertices]
    face_map: dict[tuple[int, int, int], tuple[tuple[int, int, int], int] | None] = {}
    for tet in np.asarray(state.volume_export_tets, dtype=int):
        a, b, c, d = (int(idx) for idx in tet)
        for face, opposite in (
            ((a, b, c), d),
            ((a, d, b), c),
            ((b, d, c), a),
            ((c, d, a), b),
        ):
            key = tuple(sorted(face))
            if key in face_map:
                face_map[key] = None
            else:
                face_map[key] = (face, opposite)

    boundary = [payload for payload in face_map.values() if payload is not None]
    used_export_ids = sorted({idx for face, _ in boundary for idx in face})
    compact_id = {old_idx: new_idx for new_idx, old_idx in enumerate(used_export_ids)}
    points = [export_points[old_idx] for old_idx in used_export_ids]
    triangles = []
    for face, opposite in boundary:
        idx = tuple(compact_id[i] for i in face)
        a, b, c = (points[i] for i in idx)
        opposite_point = export_points[opposite]
        normal = np.cross(b - a, c - a)
        tri_center = (a + b + c) / 3.0
        if float(np.dot(normal, opposite_point - tri_center)) > 0.0:
            idx = (idx[0], idx[2], idx[1])
        triangles.append(idx)

    with path.open("w", encoding="utf-8") as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write(f"{len(points)}\n")
        for node_id, xyz in enumerate(points, start=1):
            f.write(f"{node_id} {xyz[0]:.17e} {xyz[1]:.17e} {xyz[2]:.17e}\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write(f"{len(triangles)}\n")
        for elem_id, tri in enumerate(triangles, start=1):
            a, b, c = (idx + 1 for idx in tri)
            f.write(f"{elem_id} 2 2 1 1 {a} {b} {c}\n")
        f.write("$EndElements\n")
    return path


def _ring_segments(state: sep.VolumetricPitoisState) -> np.ndarray:
    return _ring_array_segments(_interface_ring_arrays(state))


def _render_interface_only_snapshot(state: sep.VolumetricPitoisState, title: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rings = _interface_ring_arrays(state)
    segments = _ring_array_segments(rings)

    fig = plt.figure(figsize=(7.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    if segments.size:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        ax.add_collection3d(Line3DCollection(1.0e3 * segments, colors="#111111", linewidths=0.72))

    if rings:
        pts = np.asarray([point for ring in rings for point in ring], dtype=float)
        ax.scatter(1.0e3 * pts[:, 0], 1.0e3 * pts[:, 1], 1.0e3 * pts[:, 2], s=5.0, c="#111111", depthshade=False)
        for ring in (rings[0], rings[-1]):
            cl = np.asarray(ring, dtype=float)
            cl_segments = np.asarray([[cl[i], cl[(i + 1) % len(cl)]] for i in range(len(cl))], dtype=float)
            from mpl_toolkits.mplot3d.art3d import Line3DCollection

            ax.add_collection3d(Line3DCollection(1.0e3 * cl_segments, colors="#d00000", linewidths=2.2))
            ax.scatter(1.0e3 * cl[:, 0], 1.0e3 * cl[:, 1], 1.0e3 * cl[:, 2], s=18.0, c="#d00000", depthshade=False)

    limit_mm = 1.7
    ax.set_xlim(-limit_mm, limit_mm)
    ax.set_ylim(-limit_mm, limit_mm)
    ax.set_zlim(-limit_mm, limit_mm)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(**_separation_view_kwargs())
    ax.set_xlabel("x [mm]", labelpad=10)
    ax.set_ylabel("y [mm]", labelpad=10)
    ax.set_zlabel("z [mm]", labelpad=10)
    ax.set_title(title, pad=18)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_interface_snapshot(state: sep.VolumetricPitoisState, title: str, png_path: Path) -> Path:
    sep._render_mesh_snapshot(
        state,
        title,
        png_path,
        **_separation_view_kwargs(),
    )
    return _render_interface_only_snapshot(state, title, png_path)


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
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Max free speed [m/s]")
    axes[2].set_xlabel("Time [ms]")
    axes[2].grid(alpha=0.28, which="both")

    path4 = fig_dir / "equilibrium_residuals_vs_t.png"
    fig4.tight_layout()
    fig4.savefig(path4, dpi=180, bbox_inches="tight")
    plt.close(fig4)
    paths.append(path4)

    return paths


def _generate_one_mesh_level(mesh_spec: dict[str, int | float | str]) -> dict[str, int | float | str]:
    name = str(mesh_spec["name"])
    out_dir = OUT_ROOT / "mesh_sets" / name
    fig_dir = out_dir / "fig"
    result_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    config = initialshape_config(mesh_spec)
    state = _prepare_initialshape_state(config, mesh_spec)
    for v in state.HC.V:
        v.u = np.zeros(3, dtype=float)

    volume = _match_initial_hyperct_msh_volume(state)
    sep._update_duals_and_masses(state)
    sep._update_pressure_scalar(state)

    _save_interface_snapshot(
        state,
        f"{config.title}: {name} HyperCT mesh",
        fig_dir / "mesh_initial.png",
    )
    points, tets, _node_ids = sep._snapshot_msh_indexed_mesh(state)

    summary = {
        "name": name,
        "mesh_generator": "HyperCT",
        "msh_format": "Gmsh v2 tetra export only",
        "particle_radius_mm": float(PDF_PARTICLE_RADIUS_M * 1.0e3),
        "initial_D_um": float(PDF_INITIAL_SEPARATION_D_M * 1.0e6),
        "target_volume_ul": float(PDF_BRIDGE_VOLUME_M3 * 1.0e9),
        "sum_abs_tetra_volume_ul": float(volume * 1.0e9),
        "nodes": int(points.shape[0]),
        "tetrahedra": int(tets.shape[0]),
        "contact_line_nodes": int(config.refinement and len(state.outer_rings[0]) or len(state.outer_rings[0])),
        "contact_line_radial_rings": int(config.contact_line_radial_rings),
        "extra_cl_radial_rings": int(config.extra_cl_radial_rings),
        "cl_extra_axial_layers": int(config.cl_extra_axial_layers),
        "cl_radial_bias_ratio": float(config.contact_line_radial_bias_ratio),
        "interface_radial_distribution_ratio": float(USER_INTERFACE_RADIAL_DISTRIBUTION_RATIO),
        "interface_neck_extra_axial_layers": int(mesh_spec["interface_neck_extra_axial_layers"]),
        "outer_interface_rings": int(len(state.outer_rings)),
        "png": str(fig_dir / "mesh_initial.png"),
        "msh": str(fig_dir / "mesh_initial.msh"),
    }
    _write_json(result_dir / "mesh_summary.json", summary)
    return summary


def _generate_mesh_level_sets() -> list[dict[str, int | float | str]]:
    specs = _mesh_level_specs()
    summaries = []
    for name in USER_MESH_LEVEL_ORDER:
        if name not in specs:
            raise KeyError(f"Unknown mesh level {name!r}.")
        print(f"[mesh set] building {name}", flush=True)
        summary = _generate_one_mesh_level(specs[name])
        summaries.append(summary)
        print(
            "[mesh set] "
            f"{name}: nodes={summary['nodes']} tet={summary['tetrahedra']} "
            f"volume={summary['sum_abs_tetra_volume_ul']:.9f} uL",
            flush=True,
        )

    _write_json(OUT_ROOT / "mesh_sets" / "mesh_set_summary.json", summaries)
    return summaries


def _save_relaxation_snapshot(
    state: sep.VolumetricPitoisState,
    *,
    step: int,
    fig_dir: Path,
) -> Path:
    return _save_interface_snapshot(
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
    snapshot_volume = float(sep._snapshot_msh_volume_m3(state))
    target_volume = float(state.config.initial_bridge_volume_m3)
    dual_cell_sum = float(sep._mesh_volume_m3(state.HC))
    weight = np.array([0.0, 0.0, -float(state.config.rho_f) * snapshot_volume * float(state.config.gravity_mps2)], dtype=float)
    return {
        "config": {
            "refinement": int(state.config.refinement),
            "particle_radius_mm": float(state.config.particle_radius * 1.0e3),
            "target_cap_radius_mm": float(state.config.target_cap_radius * 1.0e3),
            "initial_gap_um": float(sep._gap(state) * 1.0e6),
            "target_volume_ul": float(target_volume * 1.0e9),
            "volume_ul": float(snapshot_volume * 1.0e9),
            "volume_rel_error": float((snapshot_volume - target_volume) / max(target_volume, 1.0e-30)),
            "dual_cell_sum_ul": float(dual_cell_sum * 1.0e9),
            "gamma_npm": float(state.config.gamma),
            "mu_pas": float(state.config.mu_f),
            "rho_f": float(state.config.rho_f),
            "gravity_mps2": float(state.config.gravity_mps2),
            "interface_radial_distribution_ratio": float(USER_INTERFACE_RADIAL_DISTRIBUTION_RATIO),
            "interface_neck_refinement_gaps": int(USER_INTERFACE_NECK_REFINEMENT_GAPS),
            "interface_neck_extra_rings_per_gap": int(USER_INTERFACE_NECK_EXTRA_RINGS_PER_GAP),
            "interface_neck_extra_axial_layers": int(USER_INTERFACE_NECK_EXTRA_AXIAL_LAYERS),
        },
        "pitois_eq3_initial_profile": dict(getattr(state, "pitois_eq3_initial_profile", {})),
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
    initial_metrics: dict | None = None,
) -> list[dict]:
    history: list[dict] = []
    run_wall_start = time.perf_counter()
    interactive_shown = False
    sim_time_s = 0.0
    working_dt = float(state.config.dt)
    history_every = max(1, int(USER_HISTORY_EVERY))
    pressure_balance_every = max(1, int(USER_PRESSURE_BALANCE_EVERY))
    log_every = max(1, int(USER_LOG_EVERY))
    snapshot_every = max(1, int(USER_MESH_SNAPSHOT_EVERY_ITERATIONS))

    for v in state.HC.V:
        v.u = np.zeros(3, dtype=float)

    if initial_metrics is None:
        sep._update_duals_and_masses(state)
        sep._update_pressure_scalar(state)
        _maybe_update_continuity_pressure_scalar(state, dt=working_dt)
        if USER_UNPIN_CONTACT_LINE:
            sep._update_moving_contact_line(state, dt=None)
            _set_target_msh_volume_metadata(state)
            sep._update_duals_and_masses(state)
            sep._update_pressure_scalar(state)
            _maybe_update_continuity_pressure_scalar(state, dt=working_dt)
        else:
            _set_target_msh_volume_metadata(state)
        _solve_total_force_balance_pressure(state)
        initial_metrics = _free_force_metrics(state)
    row0 = _history_row_from_metrics(state, step=0, time_s=sim_time_s, metrics=initial_metrics)
    history.append(row0)
    print(
        f"[equilibrium] step 0/{max_steps}: "
        f"step_wall=0.000000 s, elapsed=0.000000 s, "
        f"max_free_force={row0['max_free_force_norm_n']:.6e} N, "
        f"net_total={row0['net_total_force_norm_n']:.6e} N, "
        f"max_free_speed={row0['max_free_speed_mps']:.6e} m/s",
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

            sep._update_duals_and_masses(state)
            sep._update_pressure_scalar(state)
            _maybe_update_continuity_pressure_scalar(state, dt=dt_step)
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
                if USER_UNPIN_CONTACT_LINE:
                    sep._update_moving_contact_line(state, dt=None)
                    _set_target_msh_volume_metadata(state)
                else:
                    sep._axisymmetrize_layer_rings(state)
                    sep._regularize_layer_order(state)
                    _set_target_msh_volume_metadata(state)
                issue = _first_nonfinite_state_issue(state)
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
        history_due = step % history_every == 0 or step == max_steps
        pressure_balance_due = step % pressure_balance_every == 0 or step == max_steps
        volume_correction_due = step % max(1, int(USER_VOLUME_CORRECTION_EVERY)) == 0 or step == max_steps
        if volume_correction_due:
            _correct_current_hyperct_msh_volume(state)
        if history_due or pressure_balance_due:
            sep._update_duals_and_masses(state)
            sep._update_pressure_scalar(state)
            _maybe_update_continuity_pressure_scalar(state, dt=dt_step)
        if pressure_balance_due:
            _solve_total_force_balance_pressure(state)
        row = _history_row(state, step=step, time_s=sim_time_s) if history_due else None
        step_wall = time.perf_counter() - step_wall_start
        elapsed = time.perf_counter() - run_wall_start

        if row is not None:
            print(
                f"[equilibrium] step {step}/{max_steps}: "
                f"step_wall={step_wall:.6f} s, elapsed={elapsed:.6f} s, "
                f"dt={dt_step:.3e} s, "
                f"max_free_force={row['max_free_force_norm_n']:.6e} N, "
                f"net_total={row['net_total_force_norm_n']:.6e} N, "
                f"max_free_speed={row['max_free_speed_mps']:.6e} m/s",
                flush=True,
            )
        elif step == 1 or step % log_every == 0:
            print(
                f"[equilibrium] step {step}/{max_steps}: "
                f"step_wall={step_wall:.6f} s, elapsed={elapsed:.6f} s, "
                f"dt={dt_step:.3e} s",
                flush=True,
            )

        if step % snapshot_every == 0:
            _save_relaxation_snapshot(state, step=step, fig_dir=fig_dir)
        if USER_OPEN_INTERACTIVE_WINDOW and not interactive_shown and int(USER_INTERACTIVE_STEP) == step:
            _show_interactive_snapshot(state, step=step)
            interactive_shown = True

        if row is not None:
            history.append(row)
            if (
                float(row["max_free_force_norm_n"]) <= force_tol
                and float(row["net_total_force_norm_n"]) <= total_force_tol
                and float(row["max_free_speed_mps"]) <= speed_tol
            ):
                break

    for v in state.HC.V:
        v.u = np.zeros(3, dtype=float)
    _correct_current_hyperct_msh_volume(state)
    sep._update_duals_and_masses(state)
    sep._update_pressure_scalar(state)
    _maybe_update_continuity_pressure_scalar(state, dt=working_dt)
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
        f"max_free_speed={final_row['max_free_speed_mps']:.6e} m/s",
        flush=True,
    )
    if USER_OPEN_INTERACTIVE_WINDOW and not interactive_shown and int(USER_INTERACTIVE_STEP) == int(final_row["step"]):
        _show_interactive_snapshot(state, step=int(final_row["step"]))
    return history


def main() -> None:
    _install_initialshape_export_order()
    _install_initialshape_safe_move()
    if bool(USER_GENERATE_MESH_LEVEL_SETS):
        summaries = _generate_mesh_level_sets()
        print("=" * 72)
        print("  Case 2b HyperCT mesh level sets")
        print("=" * 72)
        for row in summaries:
            print(
                f"{row['name']:>13s}: "
                f"nodes={row['nodes']}, tet={row['tetrahedra']}, "
                f"volume={row['sum_abs_tetra_volume_ul']:.9f} uL"
            )
        print(f"Saved mesh set summary    = {OUT_ROOT / 'mesh_sets' / 'mesh_set_summary.json'}")
        return

    config = initialshape_config()
    state = _prepare_initialshape_state(config)

    out_dir = OUT_ROOT
    fig_dir = out_dir / "fig"
    result_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    for v in state.HC.V:
        v.u = np.zeros(3, dtype=float)
    initial_snapshot_volume = _match_initial_hyperct_msh_volume(state)
    print(
        "[initial volume] "
        f"target={1.0e9 * float(config.initial_bridge_volume_m3):.9f} uL, "
        f"current={1.0e9 * initial_snapshot_volume:.9f} uL",
        flush=True,
    )
    sep._update_duals_and_masses(state)
    sep._update_pressure_scalar(state)
    _maybe_update_continuity_pressure_scalar(state, dt=float(state.config.dt))
    if USER_UNPIN_CONTACT_LINE:
        sep._update_moving_contact_line(state, dt=None)
        _set_target_msh_volume_metadata(state)
        sep._update_duals_and_masses(state)
        sep._update_pressure_scalar(state)
        _maybe_update_continuity_pressure_scalar(state, dt=float(state.config.dt))
    _solve_total_force_balance_pressure(state)
    initial_metrics = _free_force_metrics(state)
    print(
        "[initial guess] "
        f"max_free_force={float(initial_metrics['max_free_force_norm']):.6e} N, "
        f"net_total={float(np.linalg.norm(initial_metrics['net_total'])):.6e} N",
        flush=True,
    )

    if USER_RENDER_INITIAL_GUESS:
        _save_interface_snapshot(
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
        initial_metrics=initial_metrics,
    )
    final_metrics = _free_force_metrics(state)

    if USER_RENDER_FINAL_STATE:
        _save_interface_snapshot(
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
    print("  Case 2b initial equilibrium shape")
    print("=" * 72)
    print(f"Refinement                = {config.refinement}")
    print(f"Contact-line nodes        = {len(state.outer_rings[0])}")
    print(f"CL radial rings           = {config.contact_line_radial_rings}")
    print(f"Extra CL outer rings      = {config.extra_cl_radial_rings}")
    print(f"CL radial bias ratio      = {config.contact_line_radial_bias_ratio}")
    print(f"Interface radial ratio    = {USER_INTERFACE_RADIAL_DISTRIBUTION_RATIO:.6g}")
    print(f"Neck refine gaps          = {USER_INTERFACE_NECK_REFINEMENT_GAPS}")
    print(f"Neck extra rings/gap      = {USER_INTERFACE_NECK_EXTRA_RINGS_PER_GAP}")
    print(f"Neck extra axial layers   = {USER_INTERFACE_NECK_EXTRA_AXIAL_LAYERS}")
    print(f"Sidewall axial stride     = {config.axial_ring_stride}")
    print(f"Sidewall axial layers     = {len(state.outer_rings)}")
    if total_cl_rings <= 1 and abs(float(config.contact_line_radial_bias_ratio) - 1.0) > 1.0e-12:
        print("CL radial bias active?    = no (need at least 2 total CL rings)")
    print(f"Contact line              = {'unpinned' if USER_UNPIN_CONTACT_LINE else 'pinned'}")
    print(f"Contact angle             = {config.contact_angle_deg:.3f} deg")
    print(f"Particle radius           = {config.particle_radius * 1.0e3:.3f} mm")
    print(f"Bridge contact radius     = {config.target_cap_radius * 1.0e3:.3f} mm")
    eq3_meta = dict(getattr(state, "pitois_eq3_initial_profile", {}))
    if eq3_meta:
        print("Initial shape equation    = Pitois Eq. [3]")
        print(f"Eq. [3] 2Gamma            = {float(eq3_meta['eq3_two_gamma_1_per_m']):.12e} 1/m")
        print(f"Eq. [3] neck radius       = {float(eq3_meta['eq3_neck_radius_mm']):.9f} mm")
        print(f"Eq. [3] CL slope dr/dz    = {float(eq3_meta['eq3_contact_slope_dr_dz']):.9f}")
    print(f"Surface tension           = {config.gamma:.4f} N/m")
    print(f"Viscosity                 = {config.mu_f:.3e} Pa s")
    print(f"Density                   = {config.rho_f:.3f} kg/m^3")
    print(f"Gravity acceleration      = {config.gravity_mps2:.3f} m/s^2")
    print(f"Relax dt                  = {config.dt:.3e} s")
    print(f"Total iterations          = {USER_TOTAL_ITERATIONS}")
    print(f"Acceleration workers      = {USER_ACCEL_WORKERS}")
    print(f"Mesh snapshot every       = {USER_MESH_SNAPSHOT_EVERY_ITERATIONS}")
    final_snapshot_volume = float(sep._snapshot_msh_volume_m3(state))
    target_snapshot_volume = float(config.initial_bridge_volume_m3)
    print(f"Target bridge volume      = {1.0e9 * target_snapshot_volume:.9f} uL")
    print(f"Final .msh volume         = {1.0e9 * final_snapshot_volume:.9f} uL")
    print(
        f"Final volume rel. error   = "
        f"{(final_snapshot_volume - target_snapshot_volume) / max(target_snapshot_volume, 1.0e-30):+.6e}"
    )
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
