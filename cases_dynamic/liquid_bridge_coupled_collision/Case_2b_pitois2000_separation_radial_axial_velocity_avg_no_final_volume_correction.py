"""Case 2b: Pitois-2000 Fig. 5 separation with radial/axial velocity averaging.

This rebuilds the deleted Case 2b source in a compact form:

1. Reuse the volumetric catenoid-like bridge mesh from the equilibrium
   benchmark.
2. Move only the top and bottom cap boundaries with prescribed velocities.
3. Compute volumetric motion forces from the Cauchy stress tensor plus
   explicit surface/contact-line tension terms.
4. Use a one-fixed-one-moving sphere motion, matching the Fig. 5 apparatus
   interpretation used here, and write separation-only history JSON and mesh
   PNG snapshots under
   ``out/Case_2b``.

Pressure note:
    This case uses a projection-style incompressible Navier-Stokes closure for
    the volumetric force path. The pressure force uses hydrostatic pressure and
    the discrete projection pressure that enforces continuity; the geometric
    Laplace pressure scalar is disabled because Heron curvature already supplies
    the surface-tension force.

Axisymmetric velocity note:
    This variant keeps the imported 3-D ``.msh`` topology, but before vertex
    advection it averages each structured ring velocity into cylindrical radial
    and axial components. That removes azimuthal velocity variation while still
    allowing radial contact-line sliding and axial bridge stretching.

Contact-angle note:
    The Cox-Voinov contact-line law uses a finite equilibrium contact angle
    matching the hard-coded initial-shape build; using ``theta = 0 deg`` with
    the imported initial geometry would drive artificial contact-line advance.

Gravity note:
    The Fig. 5 experiment is not gravity-free, so this case now includes a
    hydrostatic pressure contribution ``-rho g (z - z_ref)`` along the
    physical vertical ``z`` axis. This is still a surrogate model, but it is
    closer to the experiment than the old gravity-free version.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import warnings
from collections import Counter, defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ddgclib-mpl")
import matplotlib

matplotlib.use("Agg")

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning,
)

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from hyperct import Complex


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases_dynamic.liquid_bridge_equilibrium.Case_5_volumetric_stress_equilibrium_particle_particle_bridge_benchmark import (
    HEX_TETS,
    PRISM_TETS,
    _build_structured_volumetric_catenoid,
)
from ddgclib._curvatures_heron import hndA_i_interface
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.dynamic_integrators._integrators_dynamic import _move as _hyperct_move, _recompute_duals
from ddgclib.operators.stress import cauchy_stress, dual_area_vector, dual_volume, velocity_difference_tensor_pointwise


def _move(vertex, pos, HC, bV) -> None:
    old_key = tuple(vertex.x)
    if old_key not in HC.V.cache:
        HC.V.cache[old_key] = vertex
    _hyperct_move(vertex, pos, HC, bV)


OUT_ROOT = Path(__file__).resolve().parent / "out" / "Case_2b_radial_axial_velocity_avg"
INITIALSHAPE_OUT_ROOT = Path(__file__).resolve().parent / "out" / "Case_2b_initialshape"
HARDCODED_INITIAL_MSH_PATH = INITIALSHAPE_OUT_ROOT / "fig" / "mesh_iter0100.msh"


PITOIS_FIG5_FIRST_D_OVER_R = 0.014403481854782
PITOIS_FIG5_FIRST_FORCE_MN = 1.515311539463195
PITOIS_PARTICLE_RADIUS_M = 4.0e-3
PITOIS_BRIDGE_VOLUME_M3 = 1.10e-9


def _pitois_eq6_contact_radius_raw(*, R: float, D: float, V: float) -> float:
    b2 = -float(R) * float(D) + np.sqrt(
        (float(R) * float(D)) ** 2 + 2.0 * float(R) * float(V) / np.pi
    )
    if b2 <= 0.0:
        raise ValueError("Pitois Eq. [6] produced non-positive contact radius squared")
    return float(np.sqrt(b2))


PITOIS_FIG5_FIRST_EQ6_CONTACT_RADIUS_M = _pitois_eq6_contact_radius_raw(
    R=PITOIS_PARTICLE_RADIUS_M,
    D=PITOIS_FIG5_FIRST_D_OVER_R * PITOIS_PARTICLE_RADIUS_M,
    V=PITOIS_BRIDGE_VOLUME_M3,
)

# ---------------------------------------------------------------------------
# USER CONTROLS
# Edit these values at the top of the file when you want to change the
# separation run or open an interactive viewer without touching the code below.
# ---------------------------------------------------------------------------
USER_REFINEMENT = 1
USER_USE_LAST_INITIALSHAPE_MSH = True
# Exact number of radial rings used near the contact line.
USER_CONTACT_LINE_RADIAL_RINGS = 3
# Additional outer-band rings near the contact line. These resolve the wetted
# cap close to the CL; runtime remeshing can add more if the outer gap grows.
USER_EXTRA_CL_RADIAL_RINGS = 4
# Extra axial rings inserted next to the two contact lines at mesh creation.
# This resolves the free-surface side edge immediately next to each CL.
USER_CL_EXTRA_AXIAL_LAYERS = 4
# Extra axial rings inserted around the bridge neck. The ordinary separation
# run leaves this at zero; the hard-coded initial-shape .msh import below uses
# the same neck refinement that produced mesh_iter0100.msh.
USER_NECK_EXTRA_AXIAL_LAYERS = 0
USER_INITIAL_MSH_CL_EXTRA_AXIAL_LAYERS = 12
USER_INITIAL_MSH_NECK_EXTRA_AXIAL_LAYERS = 12
USER_INITIAL_MSH_CL_RADIAL_BIAS_RATIO = 1.1
# 1 keeps all sidewall axial rings. 2 keeps every other axial ring, etc.
USER_SIDEWALL_AXIAL_RING_STRIDE = 1
# Geometric spacing bias for the outer radial band near the contact line.
# 1.0 = equal spacing. Values above 1.0 make the CL-side cells slightly finer.
USER_CL_RADIAL_BIAS_RATIO = 1.4
USER_DT_S = 0.01
USER_TOTAL_STEPS = 200 #4000 #
USER_ENABLE_ADAPTIVE_DT = True
USER_ADAPTIVE_DT_MAX_S = USER_DT_S
USER_ADAPTIVE_DT_MIN_S = 1.0e-5
USER_ADAPTIVE_DT_CAPILLARY_SAFETY = 0.25
USER_ADAPTIVE_DT_MESH_DISPLACEMENT_FRAC = 0.15

USER_RECORD_EVERY_STEPS = 10
USER_MESH_SNAPSHOT_EVERY_STEPS = 10
# Preferred minimum axis-window width in mm. The renderer will automatically
# expand beyond this if the actual mesh/sphere geometry is larger.
USER_X_AXIS_MIN_MM = -1
USER_X_AXIS_MAX_MM = 1
USER_Z_AXIS_MIN_MM = -1.7
USER_Z_AXIS_MAX_MM = 1.7
USER_INITIAL_NECK_RADIUS_RATIO = 0.44
USER_ENABLE_INITIAL_RELAXATION = True
USER_INITIAL_RELAX_STEPS = 120
USER_INITIAL_RELAX_DT_S = 2.0e-4
USER_INITIAL_RELAX_TOL_SPEED = 1.0e-7
USER_SHOW_MESH_FACES = False
USER_SHOW_MESH_VERTICES = False
# False renders a clean structured surface wireframe. True exposes the real
# compute-triangle diagonals, which is useful for debugging but visually busy.
USER_SHOW_REAL_COMPUTE_TRIANGLES = False
# When real compute triangles are enabled, "full" shows all edges/vertices.
# Switch to "meridian_strip" for a Fig. 1-style section view.
USER_REAL_TRIANGLE_VIEW = "full"
USER_SHOW_CONTACT_RING_OVERLAY = False
USER_SHOW_SURFACE_OVERLAY = True
USER_SURFACE_OVERLAY_ALPHA = 0.5
USER_MESH_VERTEX_SIZE = 10.0
USER_MESH_ALPHA = 0.99
USER_CAP_EDGE_ALPHA = 0.95
USER_SHOW_CAP_TRIANGLE_EDGES = True
USER_FILL_PARTICLE_CAP_SURFACES = False
USER_INCLUDE_GRAVITY = True
USER_GRAVITY_MPS2 = 9.81
USER_INTEGRATION_SUBSTEPS = 1
USER_CONTACT_RADIUS_SAMPLES = 128
USER_ENABLE_CONTINUITY_PRESSURE = True
USER_ENABLE_RUNTIME_REMESH = True
USER_RUNTIME_REMESH_CHECK_EVERY_STEPS = 25
USER_RUNTIME_REMESH_CL_GAP_MAX_FRACTION = 0.050
USER_RUNTIME_REMESH_CL_GAP_MIN_FRACTION = 0.006
USER_RUNTIME_REMESH_SIDE_EDGE_MAX_FRACTION = 0.060
USER_RUNTIME_REMESH_RADIAL_OUTER_GAP_MAX = 0.040
USER_RUNTIME_REMESH_RADIAL_OUTER_GAP_MIN = 0.030
USER_RUNTIME_REMESH_MAX_CL_EXTRA_AXIAL_LAYERS = 32
USER_RUNTIME_REMESH_MIN_CL_EXTRA_AXIAL_LAYERS = 0
USER_RUNTIME_REMESH_MAX_EXTRA_CL_RADIAL_RINGS = 8
USER_RUNTIME_REMESH_MIN_EXTRA_CL_RADIAL_RINGS = 0
USER_ENABLE_MESH_QUALITY_GUARD = False
USER_MESH_QUALITY_MIN_FRACTION = 0.10
USER_MESH_EDGE_MIN_FRACTION = 0.10
USER_LAYER_SPACING_MIN_FRACTION = 0.10
USER_MESH_QUALITY_ACCEPT_REL_TOL = 0.10
USER_MESH_HARD_ACCEPT_REL_TOL = 0.05
# The exported tet split is still not a fully trustworthy conforming quality
# measure, so q_min should only hard-stop the run when it becomes catastrophically
# small in an absolute sense. Real collapse is guarded primarily by edge length
# and adjacent-layer spacing.
USER_MESH_QUALITY_HARD_FRACTION = 0.0
USER_MESH_QUALITY_ABS_FLOOR = 2.5e-4
USER_MESH_EDGE_HARD_FRACTION = 0.30
USER_LAYER_SPACING_HARD_FRACTION = 0.30
USER_MESH_QUALITY_MAX_RETRIES = 10
USER_ALLOW_CONTACT_LINE_GROWTH = True
USER_USE_COX_VOINOV_CONTACT_LINE_LAW = True
USER_ENABLE_DYNAMIC_CONTACT_ANGLE = True
USER_DYNAMIC_CONTACT_ANGLE_MAX_DEG = 25.0
USER_DYNAMIC_CONTACT_ANGLE_MIN_DEG = 0.0
# Absolute safety cap. The actual per-step cap is also limited by the
# imposed sphere-separation speed so the CL cannot outrun the experiment.
USER_CONTACT_LINE_MAX_SLIDE_UM = 5.0
USER_CONTACT_LINE_MAX_SLIDE_SPEED_FACTOR = 1.0
USER_CONTACT_LINE_CONTINUATION_WEIGHT = 0.02
USER_CONTACT_LINE_FIT_RINGS = 4
USER_CONTACT_LINE_COX_MACRO_LENGTH_M = 1.0e-3
USER_CONTACT_LINE_COX_SLIP_LENGTH_M = 2.0e-9
USER_ACCEL_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))
USER_ENFORCE_NO_SWIRL = True
USER_AVERAGE_VELOCITY_RADIAL_AXIAL = True
USER_ENABLE_NS_PRESSURE_PROJECTION = True
USER_NS_PRESSURE_PROJECTION_MAX_ITERS = 120
USER_NS_PRESSURE_PROJECTION_TOL = 1.0e-5
USER_NS_PRESSURE_PROJECTION_SOR = 1.35
USER_MATCH_INITIAL_FORCE_TO_FIG5 = True
USER_INITIAL_FORCE_CALIBRATION_MAX_ITERS = 1
USER_ENABLE_ARRAY_FORCE_BACKEND = True
# "numpy" is usually fastest for this moderate-size mesh on CPU. Set to
# "torch" to route the vectorized edge algebra through PyTorch tensors.
USER_ARRAY_FORCE_BACKEND = "numpy"
USER_REPORT_CAP_VISCOUS_STRESS = False
USER_UPDATE_PITOIS_COMPARE_EVERY_STEP = True
USER_VOLUME_CONSTRAINT_STOP_FRACTION = 5.0e-3
# If False, the code only checks volume drift with the guard below; it does
# not apply the post-step geometric volume projection.
USER_ENABLE_FINAL_VOLUME_CORRECTION = False
USER_HARDCODED_INITIAL_NECK_RADIUS_RATIO = 0.6424
USER_HARDCODED_INITIAL_CONTACT_ANGLE_DEG = 10.0

USER_OPEN_INTERACTIVE_WINDOW = True
USER_INTERACTIVE_STEP = USER_TOTAL_STEPS

USER_INTERACTIVE_ELEV_DEG =  25.0 #0#
USER_INTERACTIVE_AZIM_DEG =  45.0 #-87#


@dataclass(frozen=True)
class VolumetricPitoisConfig:
    name: str
    title: str
    refinement: int = USER_REFINEMENT
    use_last_initialshape_msh: bool = USER_USE_LAST_INITIALSHAPE_MSH
    contact_line_radial_rings: int = USER_CONTACT_LINE_RADIAL_RINGS
    extra_cl_radial_rings: int = USER_EXTRA_CL_RADIAL_RINGS
    cl_extra_axial_layers: int = USER_CL_EXTRA_AXIAL_LAYERS
    neck_extra_axial_layers: int = USER_NECK_EXTRA_AXIAL_LAYERS
    axial_ring_stride: int = USER_SIDEWALL_AXIAL_RING_STRIDE
    contact_line_radial_bias_ratio: float = USER_CL_RADIAL_BIAS_RATIO
    particle_radius: float = PITOIS_PARTICLE_RADIUS_M
    target_cap_radius: float = PITOIS_FIG5_FIRST_EQ6_CONTACT_RADIUS_M
    initial_d_over_r: float = PITOIS_FIG5_FIRST_D_OVER_R
    initial_bridge_volume_m3: float = PITOIS_BRIDGE_VOLUME_M3
    initial_neck_radius_ratio: float = USER_INITIAL_NECK_RADIUS_RATIO
    enable_initial_relaxation: bool = USER_ENABLE_INITIAL_RELAXATION
    initial_relax_steps: int = USER_INITIAL_RELAX_STEPS
    initial_relax_dt: float = USER_INITIAL_RELAX_DT_S
    initial_relax_tol_speed: float = USER_INITIAL_RELAX_TOL_SPEED
    gamma: float = 2.10e-2
    mu_f: float = 1.0e-1
    rho_f: float = 965.0
    dt: float = USER_DT_S
    enable_adaptive_dt: bool = USER_ENABLE_ADAPTIVE_DT
    adaptive_dt_max_s: float = USER_ADAPTIVE_DT_MAX_S
    adaptive_dt_min_s: float = USER_ADAPTIVE_DT_MIN_S
    adaptive_dt_capillary_safety: float = USER_ADAPTIVE_DT_CAPILLARY_SAFETY
    adaptive_dt_mesh_displacement_frac: float = USER_ADAPTIVE_DT_MESH_DISPLACEMENT_FRAC
    n_steps: int = USER_TOTAL_STEPS
    cap_speed: float = 5.0e-6
    max_acceleration: float = 5.0e-3
    integration_substeps: int = USER_INTEGRATION_SUBSTEPS
    record_every: int = USER_RECORD_EVERY_STEPS
    mesh_snapshot_every: int = USER_MESH_SNAPSHOT_EVERY_STEPS
    display_motion_scale: float = 4000.0
    use_axisymmetric_laplace_pressure: bool = False
    pressure_scale: float = 0.0
    include_gravity: bool = USER_INCLUDE_GRAVITY
    gravity_mps2: float = USER_GRAVITY_MPS2
    contact_radius_samples: int = USER_CONTACT_RADIUS_SAMPLES
    enable_continuity_pressure: bool = USER_ENABLE_CONTINUITY_PRESSURE
    enable_runtime_remesh: bool = USER_ENABLE_RUNTIME_REMESH
    runtime_remesh_check_every_steps: int = USER_RUNTIME_REMESH_CHECK_EVERY_STEPS
    runtime_remesh_cl_gap_max_fraction: float = USER_RUNTIME_REMESH_CL_GAP_MAX_FRACTION
    runtime_remesh_cl_gap_min_fraction: float = USER_RUNTIME_REMESH_CL_GAP_MIN_FRACTION
    runtime_remesh_side_edge_max_fraction: float = USER_RUNTIME_REMESH_SIDE_EDGE_MAX_FRACTION
    runtime_remesh_radial_outer_gap_max: float = USER_RUNTIME_REMESH_RADIAL_OUTER_GAP_MAX
    runtime_remesh_radial_outer_gap_min: float = USER_RUNTIME_REMESH_RADIAL_OUTER_GAP_MIN
    runtime_remesh_max_cl_extra_axial_layers: int = USER_RUNTIME_REMESH_MAX_CL_EXTRA_AXIAL_LAYERS
    runtime_remesh_min_cl_extra_axial_layers: int = USER_RUNTIME_REMESH_MIN_CL_EXTRA_AXIAL_LAYERS
    runtime_remesh_max_extra_cl_radial_rings: int = USER_RUNTIME_REMESH_MAX_EXTRA_CL_RADIAL_RINGS
    runtime_remesh_min_extra_cl_radial_rings: int = USER_RUNTIME_REMESH_MIN_EXTRA_CL_RADIAL_RINGS
    enable_mesh_quality_guard: bool = USER_ENABLE_MESH_QUALITY_GUARD
    mesh_quality_min_fraction: float = USER_MESH_QUALITY_MIN_FRACTION
    mesh_edge_min_fraction: float = USER_MESH_EDGE_MIN_FRACTION
    layer_spacing_min_fraction: float = USER_LAYER_SPACING_MIN_FRACTION
    mesh_quality_accept_rel_tol: float = USER_MESH_QUALITY_ACCEPT_REL_TOL
    mesh_hard_accept_rel_tol: float = USER_MESH_HARD_ACCEPT_REL_TOL
    mesh_quality_hard_fraction: float = USER_MESH_QUALITY_HARD_FRACTION
    mesh_quality_abs_floor: float = USER_MESH_QUALITY_ABS_FLOOR
    mesh_edge_hard_fraction: float = USER_MESH_EDGE_HARD_FRACTION
    layer_spacing_hard_fraction: float = USER_LAYER_SPACING_HARD_FRACTION
    mesh_quality_max_retries: int = USER_MESH_QUALITY_MAX_RETRIES
    allow_contact_line_growth: bool = USER_ALLOW_CONTACT_LINE_GROWTH
    use_cox_voinov_contact_line_law: bool = USER_USE_COX_VOINOV_CONTACT_LINE_LAW
    enable_dynamic_contact_angle: bool = USER_ENABLE_DYNAMIC_CONTACT_ANGLE
    dynamic_contact_angle_max_deg: float = USER_DYNAMIC_CONTACT_ANGLE_MAX_DEG
    dynamic_contact_angle_min_deg: float = USER_DYNAMIC_CONTACT_ANGLE_MIN_DEG
    contact_line_max_slide_um: float = USER_CONTACT_LINE_MAX_SLIDE_UM
    contact_line_max_slide_speed_factor: float = USER_CONTACT_LINE_MAX_SLIDE_SPEED_FACTOR
    contact_line_continuation_weight: float = USER_CONTACT_LINE_CONTINUATION_WEIGHT
    contact_line_fit_rings: int = USER_CONTACT_LINE_FIT_RINGS
    contact_line_cox_macro_length_m: float = USER_CONTACT_LINE_COX_MACRO_LENGTH_M
    contact_line_cox_slip_length_m: float = USER_CONTACT_LINE_COX_SLIP_LENGTH_M
    accel_workers: int = USER_ACCEL_WORKERS
    enforce_no_swirl: bool = USER_ENFORCE_NO_SWIRL
    average_velocity_radial_axial: bool = USER_AVERAGE_VELOCITY_RADIAL_AXIAL
    enable_ns_pressure_projection: bool = USER_ENABLE_NS_PRESSURE_PROJECTION
    ns_pressure_projection_max_iters: int = USER_NS_PRESSURE_PROJECTION_MAX_ITERS
    ns_pressure_projection_tol: float = USER_NS_PRESSURE_PROJECTION_TOL
    ns_pressure_projection_sor: float = USER_NS_PRESSURE_PROJECTION_SOR
    match_initial_force_to_fig5: bool = USER_MATCH_INITIAL_FORCE_TO_FIG5
    initial_force_calibration_max_iters: int = USER_INITIAL_FORCE_CALIBRATION_MAX_ITERS
    enable_array_force_backend: bool = USER_ENABLE_ARRAY_FORCE_BACKEND
    array_force_backend: str = USER_ARRAY_FORCE_BACKEND
    report_cap_viscous_stress: bool = USER_REPORT_CAP_VISCOUS_STRESS
    update_pitois_compare_every_step: bool = USER_UPDATE_PITOIS_COMPARE_EVERY_STEP
    volume_constraint_stop_fraction: float = USER_VOLUME_CONSTRAINT_STOP_FRACTION
    enable_final_volume_correction: bool = USER_ENABLE_FINAL_VOLUME_CORRECTION
    # Contact-line equilibrium angle used by the Cox-Voinov law. The imported
    # initial shape was built with this finite wetting angle; using 0 deg here
    # would make the same initial geometry advance instead of recede.
    contact_angle_deg: float = USER_HARDCODED_INITIAL_CONTACT_ANGLE_DEG

    @property
    def relative_speed(self) -> float:
        return self.cap_speed


@dataclass
class VolumetricPitoisState:
    config: VolumetricPitoisConfig
    HC: object
    bV_caps: set
    cap_bottom: list
    cap_top: list
    cap_bottom_interior: list
    cap_top_interior: list
    bottom_contact_ring: list
    top_contact_ring: list
    cap_bottom_center: object
    cap_top_center: object
    layer_centers: list
    layer_fractions: tuple[float, ...]
    outer_rings: list
    layer_rings: list
    surface_edges: list
    surface_boundary_indices: set
    mps: object
    radial_scale: float
    axial_scale: float
    initial_gap: float
    cap_ring_factors: tuple[float, ...]
    bottom_sphere_center: np.ndarray
    top_sphere_center: np.ndarray
    target_volume_m3: float
    target_snapshot_volume_m3: float
    pressure_scalar: float = 0.0
    force_match_pressure_offset_pa: float = 0.0
    initial_force_calibration_target_mn: float = PITOIS_FIG5_FIRST_FORCE_MN
    initial_force_calibration_fixed_axial_mn: float = 0.0
    pressure_projection_map: dict[int, float] = field(default_factory=dict)
    cauchy_stress_cache: dict[int, np.ndarray] = field(default_factory=dict)
    interface_surface_tension_cache: dict[int, np.ndarray] = field(default_factory=dict)
    contact_line_force_cache: dict[int, np.ndarray] = field(default_factory=dict)
    dual_area_vector_cache: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    pressure_projection_rows_cache: object = None
    array_force_geometry_cache: object = None
    elapsed_time_s: float = 0.0
    last_step_dt: float = USER_DT_S
    last_bottom_contact_line_speed: float = 0.0
    last_top_contact_line_speed: float = 0.0
    last_dt_limit_cl: float = USER_DT_S
    last_dt_limit_capillary: float = USER_DT_S
    last_dt_limit_mesh: float = USER_DT_S
    last_dt_limiter: str = "fixed"
    runtime_substep_count: int = 0
    runtime_remesh_count: int = 0
    last_remesh_reason: str = "none"
    surface_export_vertices: list = field(default_factory=list)
    surface_export_node_ids: dict[int, int] = field(default_factory=dict)
    surface_export_side_tris: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    surface_export_bottom_cap_tris: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    surface_export_top_cap_tris: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    volume_export_vertices: list = field(default_factory=list)
    volume_export_node_ids: dict[int, int] = field(default_factory=dict)
    volume_export_tets: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=int))
    topology_frozen: bool = False
    frozen_layer_ring_id_structure: tuple = field(default_factory=tuple)
    frozen_surface_vertex_id_order: tuple = field(default_factory=tuple)
    frozen_volume_vertex_id_order: tuple = field(default_factory=tuple)
    baseline_tet_quality_min: float = 0.0
    baseline_tet_edge_min_m: float = 0.0
    baseline_adjacent_layer_spacing_min_m: float = 0.0
    last_step_continuity_volume_error_m3: float = 0.0
    last_step_continuity_displacement_m3: float = 0.0
    last_ns_projection_iterations: int = 0
    last_ns_divergence_l2: float = 0.0
    last_ns_projected_divergence_l2: float = 0.0
    last_ns_pressure_l2: float = 0.0


def _refresh_cap_boundary_sets(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "topology_frozen", False)):
        _assert_fixed_topology(state)
        return
    state.bottom_contact_ring = list(state.layer_rings[0][-1]) if state.layer_rings and state.layer_rings[0] else []
    state.top_contact_ring = list(state.layer_rings[-1][-1]) if state.layer_rings and state.layer_rings[-1] else []
    bottom_contact_ids = {id(v) for v in state.bottom_contact_ring}
    top_contact_ids = {id(v) for v in state.top_contact_ring}
    state.cap_bottom_interior = [v for v in state.cap_bottom if id(v) not in bottom_contact_ids]
    state.cap_top_interior = [v for v in state.cap_top if id(v) not in top_contact_ids]
    state.bV_caps = set(state.cap_bottom_interior + state.cap_top_interior)


def _surface_export_vertex_order(state: VolumetricPitoisState) -> list:
    ordered: list = []
    seen: set[int] = set()

    def add(vertex) -> None:
        key = id(vertex)
        if key in seen:
            return
        seen.add(key)
        ordered.append(vertex)

    add(state.cap_bottom_center)
    for rings in state.layer_rings:
        for ring in rings:
            for vertex in ring:
                add(vertex)
    add(state.cap_top_center)
    return ordered


def _layer_ring_id_structure(state: VolumetricPitoisState) -> tuple:
    return tuple(
        tuple(tuple(id(vertex) for vertex in ring) for ring in rings)
        for rings in state.layer_rings
    )


def _surface_vertex_id_order(state: VolumetricPitoisState) -> tuple:
    return tuple(id(vertex) for vertex in _surface_export_vertex_order(state))


def _volume_export_vertex_order(state: VolumetricPitoisState) -> list:
    vertices = list(state.HC.V)
    seen = {id(vertex) for vertex in vertices}
    for vertex in list(state.layer_centers):
        if id(vertex) in seen:
            continue
        vertices.append(vertex)
        seen.add(id(vertex))
    for rings in list(state.layer_rings):
        for ring in list(rings):
            for vertex in list(ring):
                if id(vertex) in seen:
                    continue
                vertices.append(vertex)
                seen.add(id(vertex))
    return vertices


def _structured_layer_block_vertices(state: VolumetricPitoisState, layer_idx: int) -> list:
    vertices: list = []
    if 0 <= int(layer_idx) < len(state.layer_centers):
        vertices.append(state.layer_centers[int(layer_idx)])
    if 0 <= int(layer_idx) < len(state.layer_rings):
        for ring in state.layer_rings[int(layer_idx)]:
            vertices.extend(list(ring))
    return vertices


def _initialshape_msh_vertex_order(state: VolumetricPitoisState) -> list:
    """Match the node order written by Case_2b_initialshape mesh_iter0100.msh.

    The initial-shape export writes the liquid axial layers first and appends
    the two spherical cap-support layers after them. The .msh tet list uses
    only the liquid-layer block, so this order keeps imported tets valid.
    """

    if len(state.layer_rings) < 3 or len(state.layer_centers) != len(state.layer_rings):
        return _volume_export_vertex_order(state)

    ordered: list = []
    seen: set[int] = set()

    def add_block(layer_idx: int) -> None:
        for vertex in _structured_layer_block_vertices(state, layer_idx):
            key = id(vertex)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(vertex)

    for layer_idx in range(1, len(state.layer_rings) - 1):
        add_block(layer_idx)
    add_block(0)
    add_block(len(state.layer_rings) - 1)

    for vertex in _volume_export_vertex_order(state):
        key = id(vertex)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(vertex)

    return ordered


def _volume_tet_indices_for_vertex_order(
    state: VolumetricPitoisState,
    ordered_vertices: list,
) -> np.ndarray:
    node_ids = {id(vertex): idx for idx, vertex in enumerate(ordered_vertices)}
    tets: list[tuple[int, int, int, int]] = []

    def idx(vertex) -> int:
        try:
            return int(node_ids[id(vertex)])
        except KeyError as exc:
            raise RuntimeError(
                "Volumetric topology references a vertex missing from the export order"
            ) from exc

    layer_pairs = list(
        zip(
            state.layer_centers[:-1],
            state.layer_centers[1:],
            state.layer_rings[:-1],
            state.layer_rings[1:],
        )
    )
    for pair_idx, (
        lower_center,
        upper_center,
        lower_radial_rings,
        upper_radial_rings,
    ) in enumerate(layer_pairs):
        if len(layer_pairs) > 2 and pair_idx in (0, len(layer_pairs) - 1):
            # The end intervals fill the spherical solid-cap chord volume. They
            # are boundary/support geometry, not liquid volume cells.
            continue
        first_lower = lower_radial_rings[0]
        first_upper = upper_radial_rings[0]
        n_ring = len(first_lower)
        for i in range(n_ring):
            prism = (
                lower_center,
                first_lower[i],
                first_lower[(i + 1) % n_ring],
                upper_center,
                first_upper[i],
                first_upper[(i + 1) % n_ring],
            )
            prism_idx = [idx(vertex) for vertex in prism]
            for tet in PRISM_TETS:
                tets.append(tuple(prism_idx[j] for j in tet))

        for band in range(len(lower_radial_rings) - 1):
            inner_lower = lower_radial_rings[band]
            outer_lower = lower_radial_rings[band + 1]
            inner_upper = upper_radial_rings[band]
            outer_upper = upper_radial_rings[band + 1]
            for i in range(n_ring):
                i_next = (i + 1) % n_ring
                hexahedron = (
                    inner_lower[i],
                    inner_lower[i_next],
                    outer_lower[i_next],
                    outer_lower[i],
                    inner_upper[i],
                    inner_upper[i_next],
                    outer_upper[i_next],
                    outer_upper[i],
                )
                hexahedron_idx = [idx(vertex) for vertex in hexahedron]
                for tet in HEX_TETS:
                    tets.append(tuple(hexahedron_idx[j] for j in tet))

    return np.asarray(tets, dtype=int) if tets else np.empty((0, 4), dtype=int)


def _tet_connectivity_multiset(tets: np.ndarray) -> Counter:
    tet_idx = np.asarray(tets, dtype=int)
    if tet_idx.ndim != 2 or tet_idx.shape[1] != 4:
        return Counter()
    return Counter(tuple(sorted(int(idx) for idx in tet)) for tet in tet_idx)


def _msh_tet_topology_matches_state(
    state: VolumetricPitoisState,
    imported_tets: np.ndarray,
    volume_vertices: list,
) -> bool:
    tet_idx = np.asarray(imported_tets, dtype=int)
    if tet_idx.ndim != 2 or tet_idx.shape[1] != 4:
        return False
    if tet_idx.size and (
        int(np.min(tet_idx)) < 0 or int(np.max(tet_idx)) >= len(volume_vertices)
    ):
        return False
    expected_tets = _volume_tet_indices_for_vertex_order(state, volume_vertices)
    if expected_tets.shape != tet_idx.shape:
        return False
    return _tet_connectivity_multiset(expected_tets) == _tet_connectivity_multiset(tet_idx)


def _assert_fixed_topology(state: VolumetricPitoisState) -> None:
    if not bool(getattr(state, "topology_frozen", False)):
        return
    if _layer_ring_id_structure(state) != state.frozen_layer_ring_id_structure:
        raise RuntimeError("Fixed topology violated: layer ring order/connectivity changed during time stepping")
    if _surface_vertex_id_order(state) != state.frozen_surface_vertex_id_order:
        raise RuntimeError("Fixed topology violated: surface vertex ID order changed during time stepping")
    current_export_ids = tuple(id(vertex) for vertex in state.surface_export_vertices)
    if current_export_ids != state.frozen_surface_vertex_id_order:
        raise RuntimeError("Fixed topology violated: exported surface vertex IDs were rebuilt during time stepping")
    current_volume_export_ids = tuple(id(vertex) for vertex in state.volume_export_vertices)
    if current_volume_export_ids != state.frozen_volume_vertex_id_order:
        raise RuntimeError("Fixed topology violated: volumetric export vertex IDs were rebuilt during time stepping")


def _ensure_surface_export_node_ids(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "topology_frozen", False)):
        _assert_fixed_topology(state)
        return
    if not state.surface_export_vertices:
        state.surface_export_vertices = _surface_export_vertex_order(state)
        state.surface_export_node_ids = {
            id(vertex): idx for idx, vertex in enumerate(state.surface_export_vertices)
        }
        return

    for vertex in _surface_export_vertex_order(state):
        key = id(vertex)
        if key in state.surface_export_node_ids:
            continue
        state.surface_export_node_ids[key] = len(state.surface_export_vertices)
        state.surface_export_vertices.append(vertex)


def _surface_vertex_idx(state: VolumetricPitoisState, vertex) -> int:
    return int(state.surface_export_node_ids[id(vertex)])


def _ensure_volume_export_node_ids(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "topology_frozen", False)):
        _assert_fixed_topology(state)
        return
    if not state.volume_export_vertices:
        state.volume_export_vertices = _volume_export_vertex_order(state)
        state.volume_export_node_ids = {
            id(vertex): idx for idx, vertex in enumerate(state.volume_export_vertices)
        }
        return

    for vertex in _volume_export_vertex_order(state):
        key = id(vertex)
        if key in state.volume_export_node_ids:
            continue
        state.volume_export_node_ids[key] = len(state.volume_export_vertices)
        state.volume_export_vertices.append(vertex)


def _freeze_volume_msh_topology(state: VolumetricPitoisState) -> None:
    _ensure_volume_export_node_ids(state)
    imported_tets = getattr(state, "imported_initial_msh_tets", None)
    if imported_tets is not None:
        state.volume_export_tets = np.asarray(imported_tets, dtype=int).copy()
        return
    state.volume_export_tets = _volume_tet_indices_for_vertex_order(
        state,
        state.volume_export_vertices,
    )


def _freeze_surface_topology(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "topology_frozen", False)):
        _assert_fixed_topology(state)
        return
    _ensure_surface_export_node_ids(state)
    _freeze_volume_msh_topology(state)

    side_triangles: list[tuple[int, int, int]] = []
    ordered_rings = [list(ring) for ring in state.outer_rings]
    for lower_ring, upper_ring in zip(ordered_rings[:-1], ordered_rings[1:]):
        n_ring = min(len(lower_ring), len(upper_ring))
        if n_ring < 2:
            continue
        for i in range(n_ring):
            i_next = (i + 1) % n_ring
            a = _surface_vertex_idx(state, lower_ring[i])
            b = _surface_vertex_idx(state, lower_ring[i_next])
            c = _surface_vertex_idx(state, upper_ring[i_next])
            d = _surface_vertex_idx(state, upper_ring[i])
            pa = np.asarray(lower_ring[i].x_a[:3], dtype=float)
            pb = np.asarray(lower_ring[i_next].x_a[:3], dtype=float)
            pc = np.asarray(upper_ring[i_next].x_a[:3], dtype=float)
            pd = np.asarray(upper_ring[i].x_a[:3], dtype=float)
            if _quad_uses_ac_diagonal(pa, pb, pc, pd):
                side_triangles.append((a, b, c))
                side_triangles.append((a, c, d))
            else:
                side_triangles.append((a, b, d))
                side_triangles.append((b, c, d))

    def freeze_cap(rings: list, center_vertex) -> np.ndarray:
        triangles: list[tuple[int, int, int]] = []
        if not rings:
            return np.empty((0, 3), dtype=int)
        center_idx = _surface_vertex_idx(state, center_vertex)
        n_ring = len(rings[0])
        for i in range(n_ring):
            i_next = (i + 1) % n_ring
            triangles.append(
                (
                    center_idx,
                    _surface_vertex_idx(state, rings[0][i]),
                    _surface_vertex_idx(state, rings[0][i_next]),
                )
            )
        for band in range(len(rings) - 1):
            inner = rings[band]
            outer = rings[band + 1]
            for i in range(n_ring):
                i_next = (i + 1) % n_ring
                a = _surface_vertex_idx(state, inner[i])
                b = _surface_vertex_idx(state, inner[i_next])
                c = _surface_vertex_idx(state, outer[i_next])
                d = _surface_vertex_idx(state, outer[i])
                pa = np.asarray(inner[i].x_a[:3], dtype=float)
                pb = np.asarray(inner[i_next].x_a[:3], dtype=float)
                pc = np.asarray(outer[i_next].x_a[:3], dtype=float)
                pd = np.asarray(outer[i].x_a[:3], dtype=float)
                if _quad_uses_ac_diagonal(pa, pb, pc, pd):
                    triangles.append((a, b, c))
                    triangles.append((a, c, d))
                else:
                    triangles.append((a, b, d))
                    triangles.append((b, c, d))
        return np.asarray(triangles, dtype=int)

    state.surface_export_side_tris = (
        np.asarray(side_triangles, dtype=int) if side_triangles else np.empty((0, 3), dtype=int)
    )
    state.surface_export_bottom_cap_tris = freeze_cap(state.layer_rings[0], state.cap_bottom_center)
    state.surface_export_top_cap_tris = freeze_cap(state.layer_rings[-1], state.cap_top_center)
    state.frozen_layer_ring_id_structure = _layer_ring_id_structure(state)
    state.frozen_surface_vertex_id_order = tuple(id(vertex) for vertex in state.surface_export_vertices)
    state.frozen_volume_vertex_id_order = tuple(id(vertex) for vertex in state.volume_export_vertices)
    state.topology_frozen = True


def _surface_points_array(state: VolumetricPitoisState) -> np.ndarray:
    _ensure_surface_export_node_ids(state)
    return np.asarray(
        [np.asarray(vertex.x_a[:3], dtype=float) for vertex in state.surface_export_vertices],
        dtype=float,
    )


def _surface_triangles_xyz_from_indices(state: VolumetricPitoisState, tris: np.ndarray) -> np.ndarray:
    tri_idx = np.asarray(tris, dtype=int)
    if tri_idx.size == 0:
        return np.empty((0, 3, 3), dtype=float)
    points = _surface_points_array(state)
    return np.asarray([points[np.asarray(tri, dtype=int)] for tri in tri_idx], dtype=float)


def _canonicalize_ring_orders(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "topology_frozen", False)):
        _assert_fixed_topology(state)
        return
    if not state.layer_rings:
        return

    bottom_ring_center, _top_ring_center, axis = _contact_plane_centers_and_axis(
        SimpleNamespace(outer_rings=state.outer_rings)
    )
    reference = np.asarray(state.outer_rings[0][0].x_a[:3], dtype=float) - bottom_ring_center
    e1, e2 = _orthonormal_tangent_basis(axis, reference)

    ordered_layers: list[list[list[object]]] = []
    ordered_outer: list[list[object]] = []
    for rings in state.layer_rings:
        if not rings:
            ordered_layers.append([])
            ordered_outer.append([])
            continue
        outer_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in rings[-1]], axis=0)
        ordered_rings: list[list[object]] = []
        for ring in rings:
            coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
            rel = coords - outer_center[None, :]
            rel -= np.outer(np.dot(rel, axis), axis)
            angles = np.arctan2(rel @ e2, rel @ e1)
            order = np.argsort(angles)
            ordered_rings.append([ring[i] for i in order])
        ordered_layers.append(ordered_rings)
        ordered_outer.append(ordered_rings[-1])

    state.layer_rings = ordered_layers
    state.outer_rings = ordered_outer
    state.surface_export_vertices = []
    state.surface_export_node_ids = {}
    state.surface_export_side_tris = np.empty((0, 3), dtype=int)
    state.surface_export_bottom_cap_tris = np.empty((0, 3), dtype=int)
    state.surface_export_top_cap_tris = np.empty((0, 3), dtype=int)
    if not bool(getattr(state, "preserve_imported_initial_msh", False)):
        state.volume_export_vertices = []
        state.volume_export_node_ids = {}
        state.volume_export_tets = np.empty((0, 4), dtype=int)
    _refresh_cap_boundary_sets(state)


def _radial_ring_factors_from_count(n_rings: int) -> tuple[float, ...]:
    n_rings = int(n_rings)
    if n_rings < 1:
        raise ValueError("contact_line_radial_rings must be at least 1")
    return tuple(float(i + 1) / float(n_rings) for i in range(n_rings))


def _radial_ring_factors_with_outer_refinement(
    n_rings: int,
    extra_outer_rings: int,
    outer_bias_ratio: float = 1.0,
) -> tuple[float, ...]:
    base = list(_radial_ring_factors_from_count(n_rings))
    extra_outer_rings = max(0, int(extra_outer_rings))
    outer_bias_ratio = max(1.0, float(outer_bias_ratio))
    if extra_outer_rings == 0 and abs(outer_bias_ratio - 1.0) <= 1.0e-12:
        return tuple(base)

    outer_start = float(base[-2]) if len(base) >= 2 else 0.0
    total_outer_segments = max(1, extra_outer_rings + 1)
    outer_width = max(0.0, 1.0 - outer_start)
    if outer_width <= 1.0e-15:
        return tuple(base)

    if abs(outer_bias_ratio - 1.0) <= 1.0e-12:
        outer_gaps = [outer_width / float(total_outer_segments)] * total_outer_segments
    else:
        series = np.array(
            [outer_bias_ratio ** power for power in range(total_outer_segments - 1, -1, -1)],
            dtype=float,
        )
        outer_gaps = list(outer_width * series / float(np.sum(series)))

    factors = list(base[:-1])
    radius = outer_start
    for gap in outer_gaps:
        radius += float(gap)
        factors.append(min(radius, 1.0))

    factors[-1] = 1.0
    factors = sorted({round(float(f), 12) for f in factors})
    return tuple(float(f) for f in factors)


def separation_config() -> VolumetricPitoisConfig:
    config = VolumetricPitoisConfig(
        name="Case_2b_radial_axial_velocity_avg",
        title="Case 2b: radial/axial averaged volumetric separation",
    )
    if bool(config.use_last_initialshape_msh):
        return _initial_msh_matched_config(config)
    return config


def _initial_msh_matched_config(config: VolumetricPitoisConfig) -> VolumetricPitoisConfig:
    """Use the structured resolution that produced Case_2b_initialshape mesh_iter0100.msh."""

    return replace(
        config,
        cl_extra_axial_layers=int(USER_INITIAL_MSH_CL_EXTRA_AXIAL_LAYERS),
        neck_extra_axial_layers=int(USER_INITIAL_MSH_NECK_EXTRA_AXIAL_LAYERS),
        contact_line_radial_bias_ratio=float(USER_INITIAL_MSH_CL_RADIAL_BIAS_RATIO),
    )


def _pitois_eq6_contact_radius(*, R: float, D: float, V: float) -> float:
    return _pitois_eq6_contact_radius_raw(R=R, D=D, V=V)


def _hardcoded_initial_msh_build_config(config: VolumetricPitoisConfig) -> VolumetricPitoisConfig:
    return replace(
        _initial_msh_matched_config(config),
        use_last_initialshape_msh=False,
        initial_d_over_r=0.0,
        target_cap_radius=_pitois_eq6_contact_radius(
            R=float(config.particle_radius),
            D=0.0,
            V=float(config.initial_bridge_volume_m3),
        ),
        initial_neck_radius_ratio=float(USER_HARDCODED_INITIAL_NECK_RADIUS_RATIO),
        enable_initial_relaxation=False,
        contact_angle_deg=float(USER_HARDCODED_INITIAL_CONTACT_ANGLE_DEG),
    )


def _cap_radius(vertices: list) -> float:
    return max((float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float))) for v in vertices), default=1.0)


def _scale_mesh(HC, radial_scale: float, axial_scale: float) -> None:
    for v in list(HC.V):
        coord = np.asarray(v.x_a[:3], dtype=float)
        HC.V.move(
            v,
            (
                radial_scale * coord[0],
                radial_scale * coord[1],
                axial_scale * coord[2],
            ),
        )


def _move_vertices_batch(vertices, targets, HC, bV) -> None:
    """Move a vertex group without transient cache collisions.

    Some ring updates are permutations: a vertex can be moved onto another
    vertex's current coordinates before that second vertex is moved. Doing
    those updates one-by-one causes ``hyperct`` to evict the later vertex from
    the cache. A temporary staging move avoids that failure mode.
    """
    vertices = list(vertices)
    targets = [tuple(map(float, target)) for target in targets]
    if not vertices:
        return
    if len(vertices) != len(targets):
        raise ValueError("vertices and targets must have the same length")
    if len(vertices) == 1:
        _move(vertices[0], targets[0], HC, bV)
        return

    coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in HC.V], dtype=float)
    scale = max(float(np.max(np.abs(coords))), 1.0)
    base = 10.0 * (scale + 1.0)
    delta = 1.0e-3 * (scale + 1.0)
    staged = [
        (base + delta * float(i + 1), base + 2.0 * delta * float(i + 1), base + 3.0 * delta * float(i + 1))
        for i in range(len(vertices))
    ]

    def safe_move(vertex, target) -> None:
        old_key = tuple(vertex.x)
        if old_key not in HC.V.cache:
            HC.V.cache[old_key] = vertex
        _move(vertex, target, HC, bV)

    for v, tmp in zip(vertices, staged):
        safe_move(v, tmp)
    for v, target in zip(vertices, targets):
        safe_move(v, target)


def _clear_force_caches(state: VolumetricPitoisState) -> None:
    state.cauchy_stress_cache.clear()
    state.interface_surface_tension_cache.clear()
    state.contact_line_force_cache.clear()


def _clear_geometry_caches(state: VolumetricPitoisState) -> None:
    _clear_force_caches(state)
    state.dual_area_vector_cache.clear()
    state.pressure_projection_rows_cache = None
    state.array_force_geometry_cache = None


def _uses_exact_imported_msh(state: VolumetricPitoisState) -> bool:
    return bool(
        getattr(state, "preserve_imported_initial_msh", False)
        and getattr(state, "loaded_exact_initial_msh_path", None)
    )


def _dual_area_vector_cached(v_i, v_j, state: VolumetricPitoisState) -> np.ndarray:
    key = (id(v_i), id(v_j))
    cached = state.dual_area_vector_cache.get(key)
    if cached is not None:
        return cached
    area_vec = np.asarray(dual_area_vector(v_i, v_j, state.HC, dim=3), dtype=float)
    state.dual_area_vector_cache[key] = area_vec
    return area_vec


_TORCH_MODULE = None
_TORCH_IMPORT_ATTEMPTED = False


def _torch_module_or_none():
    global _TORCH_MODULE, _TORCH_IMPORT_ATTEMPTED
    if _TORCH_IMPORT_ATTEMPTED:
        return _TORCH_MODULE
    _TORCH_IMPORT_ATTEMPTED = True
    try:
        import torch  # type: ignore
    except Exception:
        _TORCH_MODULE = None
    else:
        _TORCH_MODULE = torch
    return _TORCH_MODULE


def _can_use_array_force_backend(state: VolumetricPitoisState) -> bool:
    if not bool(getattr(state.config, "enable_array_force_backend", False)):
        return False
    if not _uses_exact_imported_msh(state):
        return False
    return bool(getattr(state, "volume_export_vertices", None))


def _array_force_backend_name(state: VolumetricPitoisState) -> str:
    name = str(getattr(state.config, "array_force_backend", "numpy")).strip().lower()
    if name == "torch" and _torch_module_or_none() is None:
        return "numpy"
    return "torch" if name == "torch" else "numpy"


def _array_force_geometry(state: VolumetricPitoisState) -> dict[str, object]:
    vertices = list(state.volume_export_vertices or _volume_export_vertex_order(state))
    vertex_ids = tuple(id(v) for v in vertices)
    cached = getattr(state, "array_force_geometry_cache", None)
    if isinstance(cached, dict) and cached.get("vertex_ids") == vertex_ids:
        return cached

    index = {id(v): idx for idx, v in enumerate(vertices)}
    edge_i: list[int] = []
    edge_j: list[int] = []
    area_vecs: list[np.ndarray] = []
    for v in vertices:
        i = index.get(id(v))
        if i is None:
            continue
        for nb in getattr(v, "nn", []):
            j = index.get(id(nb))
            if j is None:
                continue
            try:
                area_vec = _dual_area_vector_cached(v, nb, state)
            except (KeyError, IndexError, ValueError, RuntimeError, ZeroDivisionError, StopIteration):
                continue
            area_vec = np.asarray(area_vec, dtype=float)
            if area_vec.shape != (3,) or not np.all(np.isfinite(area_vec)):
                continue
            if float(np.linalg.norm(area_vec)) <= 1.0e-30:
                continue
            edge_i.append(int(i))
            edge_j.append(int(j))
            area_vecs.append(area_vec)

    geom = {
        "vertices": vertices,
        "vertex_ids": vertex_ids,
        "index": index,
        "edge_i": np.asarray(edge_i, dtype=np.int64),
        "edge_j": np.asarray(edge_j, dtype=np.int64),
        "area_vec": np.asarray(area_vecs, dtype=float) if area_vecs else np.empty((0, 3), dtype=float),
        "torch_cache": {},
    }
    state.array_force_geometry_cache = geom
    return geom


def _array_state_vectors(
    state: VolumetricPitoisState,
    geom: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = geom["vertices"]
    points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
    velocities = np.asarray([np.asarray(v.u[:3], dtype=float) for v in vertices], dtype=float)
    volumes = np.asarray([_vertex_dual_volume_m3(v) for v in vertices], dtype=float)
    return points, velocities, np.maximum(volumes, 1.0e-18)


def _pressure_array_for_vertices(
    state: VolumetricPitoisState,
    geom: dict[str, object],
    points: np.ndarray,
    *,
    include_projection: bool,
) -> np.ndarray:
    vertices = geom["vertices"]
    pressure = np.zeros(len(vertices), dtype=float)
    if bool(state.config.include_gravity):
        z_ref = 0.5 * float(state.bottom_sphere_center[2] + state.top_sphere_center[2])
        pressure -= float(state.config.rho_f) * float(state.config.gravity_mps2) * (points[:, 2] - z_ref)
    if include_projection and getattr(state, "pressure_projection_map", None):
        pressure += np.asarray(
            [float(state.pressure_projection_map.get(id(v), 0.0)) for v in vertices],
            dtype=float,
        )
    return pressure


def _velocity_gradient_array_numpy(
    geom: dict[str, object],
    velocities: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    n_vertices = int(velocities.shape[0])
    du = np.zeros((n_vertices, 3, 3), dtype=float)
    edge_i = np.asarray(geom["edge_i"], dtype=np.int64)
    edge_j = np.asarray(geom["edge_j"], dtype=np.int64)
    area_vec = np.asarray(geom["area_vec"], dtype=float)
    if edge_i.size == 0:
        return du
    delta_u = velocities[edge_j] - velocities[edge_i]
    np.add.at(du, edge_i, delta_u[:, :, None] * area_vec[:, None, :])
    du *= (0.5 / np.maximum(volumes, 1.0e-18))[:, None, None]
    return du


def _cauchy_force_array_numpy(
    state: VolumetricPitoisState,
    geom: dict[str, object],
    pressure: np.ndarray,
    velocities: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    n_vertices = int(velocities.shape[0])
    force = np.zeros((n_vertices, 3), dtype=float)
    edge_i = np.asarray(geom["edge_i"], dtype=np.int64)
    edge_j = np.asarray(geom["edge_j"], dtype=np.int64)
    area_vec = np.asarray(geom["area_vec"], dtype=float)
    if edge_i.size == 0:
        return force

    du = _velocity_gradient_array_numpy(geom, velocities, volumes)
    sigma = float(state.config.mu_f) * (du + np.swapaxes(du, 1, 2))
    diag = np.arange(3)
    sigma[:, diag, diag] -= pressure[:, None]
    sigma_face = 0.5 * (sigma[edge_i] + sigma[edge_j])
    contrib = np.einsum("eij,ej->ei", sigma_face, area_vec, optimize=True)
    np.add.at(force, edge_i, contrib)
    return force


def _torch_geometry_tensors(geom: dict[str, object]):
    torch = _torch_module_or_none()
    if torch is None:
        return None
    cache = geom.get("torch_cache")
    if not isinstance(cache, dict):
        cache = {}
        geom["torch_cache"] = cache
    cached = cache.get("cpu_float64")
    if cached is not None:
        return cached
    tensors = (
        torch.as_tensor(np.asarray(geom["edge_i"], dtype=np.int64), dtype=torch.long),
        torch.as_tensor(np.asarray(geom["edge_j"], dtype=np.int64), dtype=torch.long),
        torch.as_tensor(np.asarray(geom["area_vec"], dtype=float), dtype=torch.float64),
    )
    cache["cpu_float64"] = tensors
    return tensors


def _velocity_gradient_array_torch(
    geom: dict[str, object],
    velocities: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    torch = _torch_module_or_none()
    tensors = _torch_geometry_tensors(geom)
    if torch is None or tensors is None:
        return _velocity_gradient_array_numpy(geom, velocities, volumes)
    edge_i, edge_j, area_vec = tensors
    n_vertices = int(velocities.shape[0])
    du = torch.zeros((n_vertices, 3, 3), dtype=torch.float64)
    if edge_i.numel() == 0:
        return du.numpy()
    vel = torch.as_tensor(velocities, dtype=torch.float64)
    vols = torch.as_tensor(np.maximum(volumes, 1.0e-18), dtype=torch.float64)
    delta_u = vel.index_select(0, edge_j) - vel.index_select(0, edge_i)
    contrib = delta_u[:, :, None] * area_vec[:, None, :]
    du.index_add_(0, edge_i, contrib)
    du *= (0.5 / vols).reshape((-1, 1, 1))
    return du.numpy()


def _cauchy_force_array_torch(
    state: VolumetricPitoisState,
    geom: dict[str, object],
    pressure: np.ndarray,
    velocities: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    torch = _torch_module_or_none()
    tensors = _torch_geometry_tensors(geom)
    if torch is None or tensors is None:
        return _cauchy_force_array_numpy(state, geom, pressure, velocities, volumes)
    edge_i, edge_j, area_vec = tensors
    n_vertices = int(velocities.shape[0])
    force = torch.zeros((n_vertices, 3), dtype=torch.float64)
    if edge_i.numel() == 0:
        return force.numpy()
    du = torch.as_tensor(
        _velocity_gradient_array_torch(geom, velocities, volumes),
        dtype=torch.float64,
    )
    sigma = float(state.config.mu_f) * (du + du.transpose(1, 2))
    p = torch.as_tensor(pressure, dtype=torch.float64)
    diag = torch.arange(3, dtype=torch.long)
    sigma[:, diag, diag] -= p.reshape((-1, 1))
    sigma_face = 0.5 * (sigma.index_select(0, edge_i) + sigma.index_select(0, edge_j))
    contrib = torch.einsum("eij,ej->ei", sigma_face, area_vec)
    force.index_add_(0, edge_i, contrib)
    return force.numpy()


def _remove_swirl_components_array(
    values: np.ndarray,
    *,
    points: np.ndarray,
    axis_origin: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    points = np.asarray(points, dtype=float)
    axis_origin = np.asarray(axis_origin, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or points.shape != values.shape
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(axis_origin))
        or not np.all(np.isfinite(axis))
    ):
        return values.copy()
    rel = points - np.asarray(axis_origin, dtype=float)[None, :]
    axis /= max(float(np.linalg.norm(axis)), 1.0e-30)
    if not np.all(np.isfinite(axis)):
        return values.copy()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        radial = rel - np.outer(rel @ axis, axis)
        azimuthal = np.cross(axis[None, :], radial)
    if not (np.all(np.isfinite(radial)) and np.all(np.isfinite(azimuthal))):
        return values.copy()
    norms = np.linalg.norm(azimuthal, axis=1)
    out = np.asarray(values, dtype=float).copy()
    active = norms > 1.0e-30
    if np.any(active):
        e_phi = azimuthal[active] / norms[active, None]
        out[active] -= np.sum(out[active] * e_phi, axis=1)[:, None] * e_phi
    return out


def _array_force_all(
    state: VolumetricPitoisState,
    *,
    pressure: np.ndarray,
    velocities: np.ndarray | None = None,
    include_contact_line: bool = True,
) -> tuple[np.ndarray, dict[str, object], np.ndarray, np.ndarray, np.ndarray] | None:
    if not _can_use_array_force_backend(state):
        return None
    geom = _array_force_geometry(state)
    points, current_velocities, volumes = _array_state_vectors(state, geom)
    if velocities is None:
        velocities = current_velocities
    backend = _array_force_backend_name(state)
    if backend == "torch":
        force = _cauchy_force_array_torch(state, geom, pressure, velocities, volumes)
    else:
        force = _cauchy_force_array_numpy(state, geom, pressure, velocities, volumes)

    index = geom["index"]
    for idx, v in enumerate(geom["vertices"]):
        if getattr(v, "is_interface", False):
            force[idx] += _interface_surface_tension_force(v, state=state)
    if include_contact_line:
        for v in list(state.bottom_contact_ring) + list(state.top_contact_ring):
            idx = index.get(id(v))
            if idx is not None:
                force[int(idx)] += _Fcl(v, state=state)

    if bool(getattr(state.config, "enforce_no_swirl", False)):
        axis_origin, axis = _swirl_axis_geometry(state)
        force = _remove_swirl_components_array(
            force,
            points=points,
            axis_origin=axis_origin,
            axis=axis,
        )
    return force, geom, points, velocities, volumes


def _clip_acceleration_array(accel: np.ndarray, state: VolumetricPitoisState) -> np.ndarray:
    accel = np.asarray(accel, dtype=float)
    norms = np.linalg.norm(accel, axis=1)
    limit = float(state.config.max_acceleration)
    active = norms > max(limit, 0.0)
    if np.any(active) and limit > 0.0:
        accel = accel.copy()
        accel[active] *= (limit / norms[active])[:, None]
    return accel


def _array_advance_symplectic_step(state: VolumetricPitoisState, *, dt: float) -> bool:
    if not _can_use_array_force_backend(state):
        return False
    geom = _array_force_geometry(state)
    points, velocities, volumes = _array_state_vectors(state, geom)
    pressure = _pressure_array_for_vertices(state, geom, points, include_projection=True)
    packed = _array_force_all(
        state,
        pressure=pressure,
        velocities=velocities,
        include_contact_line=True,
    )
    if packed is None:
        return False
    force, geom, points, velocities, volumes = packed
    mass = np.maximum(float(state.config.rho_f) * volumes, 1.0e-18)
    accel = _clip_acceleration_array(force / mass[:, None], state)
    if bool(getattr(state.config, "enforce_no_swirl", False)):
        axis_origin, axis = _swirl_axis_geometry(state)
        accel = _remove_swirl_components_array(
            accel,
            points=points,
            axis_origin=axis_origin,
            axis=axis,
        )

    vertices = geom["vertices"]
    movers = []
    new_velocities = []
    bV_ids = {id(v) for v in state.bV_caps}
    for idx, v in enumerate(vertices):
        if id(v) in bV_ids:
            continue
        u_new = np.asarray(v.u[:3], dtype=float) + float(dt) * accel[idx]
        movers.append(v)
        new_velocities.append(u_new)
    for v, u_new in zip(movers, new_velocities):
        v.u[:3] = np.asarray(u_new, dtype=float)
    _enforce_radial_axial_velocity_average_field(state)
    targets = []
    for v in movers:
        x_new = np.asarray(v.x_a[:3], dtype=float) + float(dt) * np.asarray(v.u[:3], dtype=float)
        targets.append(tuple(map(float, x_new)))
    _move_vertices_batch(movers, targets, state.HC, state.bV_caps)
    return True


def _update_duals_and_masses(state: VolumetricPitoisState) -> None:
    _clear_geometry_caches(state)
    _recompute_duals(state.HC)
    try:
        if not bool(getattr(state, "topology_frozen", False)):
            raise RuntimeError("lumped tet masses require frozen topology")
        points, tets, _node_ids = _snapshot_msh_indexed_mesh(state)
        tet_volumes = np.abs(_tet_signed_volumes(points, tets))
        if tet_volumes.size and points.shape[0] == len(state.volume_export_vertices):
            weights = np.repeat(tet_volumes / 4.0, 4)
            lumped = np.bincount(tets.reshape(-1), weights=weights, minlength=points.shape[0])
            for idx, v in enumerate(state.volume_export_vertices):
                v.m = max(float(lumped[idx]), 1.0e-12)
            return
    except Exception:
        pass

    for v in state.HC.V:
        v.m = max(float(dual_volume(v, state.HC, dim=3)), 1.0e-12)


def _axisymmetric_profile(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_vals = []
    radii = []
    for ring in state.outer_rings:
        if not ring:
            continue
        coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
        z_vals.append(float(np.mean(coords[:, 2])))
        radii.append(float(np.mean(np.linalg.norm(coords[:, :2], axis=1))))

    z = np.array(z_vals, dtype=float)
    r = np.array(radii, dtype=float)
    if z.size < 3:
        return z, r, np.zeros_like(z)

    order = np.argsort(z)
    z = z[order]
    r = r[order]

    dr_dz = np.gradient(r, z, edge_order=1)
    d2r_dz2 = np.gradient(dr_dz, z, edge_order=1)
    denom = np.maximum(1.0 + dr_dz**2, 1.0e-12)
    kappa_meridional = -d2r_dz2 / np.power(denom, 1.5)
    kappa_azimuthal = 1.0 / np.maximum(r * np.sqrt(denom), 1.0e-12)
    return z, r, kappa_meridional + kappa_azimuthal


def _update_pressure_scalar(state: VolumetricPitoisState) -> None:
    _clear_force_caches(state)
    state.pressure_projection_map.clear()
    state.pressure_scalar = 0.0


def _clip_acceleration(accel: np.ndarray, state: VolumetricPitoisState) -> np.ndarray:
    accel = np.asarray(accel, dtype=float)
    accel_norm = float(np.linalg.norm(accel))
    if accel_norm > state.config.max_acceleration:
        accel = accel * (state.config.max_acceleration / accel_norm)
    return accel


def _vertex_dual_volume_m3(v) -> float:
    return max(float(getattr(v, "m", 0.0)), 1.0e-18)


def _vertex_mass_kg(v, state: VolumetricPitoisState) -> float:
    return max(float(state.config.rho_f) * _vertex_dual_volume_m3(v), 1.0e-18)


def _boundary_vertex_lookup(state: VolumetricPitoisState) -> dict[tuple[float, float, float], object]:
    lookup: dict[tuple[float, float, float], object] = {}
    for ring in state.outer_rings:
        for v in ring:
            lookup[tuple(np.round(np.asarray(v.x_a[:3], dtype=float), 12))] = v
    for rings in (state.layer_rings[0], state.layer_rings[-1]):
        for ring in rings:
            for v in ring:
                lookup[tuple(np.round(np.asarray(v.x_a[:3], dtype=float), 12))] = v
    lookup[tuple(np.round(np.asarray(state.cap_bottom_center.x_a[:3], dtype=float), 12))] = state.cap_bottom_center
    lookup[tuple(np.round(np.asarray(state.cap_top_center.x_a[:3], dtype=float), 12))] = state.cap_top_center
    return lookup


def _boundary_dual_areas_and_normals(
    state: VolumetricPitoisState,
) -> tuple[list[object], dict[int, float], dict[int, np.ndarray]]:
    side_triangles = _actual_compute_side_surface_triangles(state)
    bottom_cap_triangles, top_cap_triangles = _actual_compute_cap_triangles(state)
    triangle_sets = [arr for arr in (side_triangles, bottom_cap_triangles, top_cap_triangles) if arr.size]
    if not triangle_sets:
        return [], {}, {}

    lookup = _boundary_vertex_lookup(state)
    all_boundary_vertices = list(lookup.values())
    if not all_boundary_vertices:
        return [], {}, {}
    interior_ref = np.mean(
        [np.asarray(v.x_a[:3], dtype=float) for v in all_boundary_vertices],
        axis=0,
    )

    normal_sum: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=float))
    area_map: dict[int, float] = defaultdict(float)
    vertex_objects: dict[int, object] = {}

    for tri_set in triangle_sets:
        for tri in np.asarray(tri_set, dtype=float):
            a, b, c = tri
            normal = np.cross(b - a, c - a)
            norm_n = float(np.linalg.norm(normal))
            if norm_n <= 1.0e-30:
                continue
            centroid = (a + b + c) / 3.0
            if float(np.dot(normal, centroid - interior_ref)) < 0.0:
                normal = -normal
            area = 0.5 * float(np.linalg.norm(normal))
            for point in tri:
                key = tuple(np.round(np.asarray(point, dtype=float), 12))
                v = lookup.get(key)
                if v is None:
                    continue
                vid = id(v)
                vertex_objects[vid] = v
                normal_sum[vid] = normal_sum[vid] + normal
                area_map[vid] += area / 3.0

    normal_map: dict[int, np.ndarray] = {}
    for vid, nvec in normal_sum.items():
        norm_n = float(np.linalg.norm(nvec))
        if norm_n <= 1.0e-30:
            continue
        normal_map[vid] = nvec / norm_n

    vertices = [vertex_objects[vid] for vid in vertex_objects.keys()]
    return vertices, dict(area_map), normal_map


def _boundary_volume_flux(state: VolumetricPitoisState) -> float:
    boundary_vertices, area_map, normal_map = _boundary_dual_areas_and_normals(state)
    flux = 0.0
    for v in boundary_vertices:
        area = float(area_map.get(id(v), 0.0))
        normal = normal_map.get(id(v))
        if area <= 0.0 or normal is None:
            continue
        flux += area * float(np.dot(np.asarray(v.u[:3], dtype=float), np.asarray(normal, dtype=float)))
    return float(flux)


def _contact_line_vertex_ids(state: VolumetricPitoisState) -> set[int]:
    ids = {id(v) for v in state.bottom_contact_ring}
    ids.update(id(v) for v in state.top_contact_ring)
    return ids


def _enforce_continuity_velocity_constraint(state: VolumetricPitoisState) -> None:
    if not bool(getattr(state.config, "enable_continuity_pressure", False)):
        return
    boundary_vertices, area_map, normal_map = _boundary_dual_areas_and_normals(state)
    if not boundary_vertices:
        return

    flux = 0.0
    denom = 0.0
    constrained_ids = _contact_line_vertex_ids(state)
    free_entries: list[tuple[object, float, np.ndarray]] = []
    for v in boundary_vertices:
        area = float(area_map.get(id(v), 0.0))
        normal = normal_map.get(id(v))
        if area <= 0.0 or normal is None:
            continue
        normal = np.asarray(normal, dtype=float)
        flux += area * float(np.dot(np.asarray(v.u[:3], dtype=float), normal))
        if v not in state.bV_caps and id(v) not in constrained_ids:
            free_entries.append((v, area, normal))
            denom += area * area

    if denom <= 1.0e-30:
        return
    lambda_u = -flux / denom
    for v, area, normal in free_entries:
        corrected = np.asarray(v.u[:3], dtype=float) + lambda_u * area * normal
        if bool(getattr(state.config, "enforce_no_swirl", False)):
            axis_origin, axis = _swirl_axis_geometry(state)
            corrected = _remove_swirl_component(
                corrected,
                point=np.asarray(v.x_a[:3], dtype=float),
                axis_origin=axis_origin,
                axis=axis,
            )
        v.u = np.asarray(corrected, dtype=float)


def _prescribed_volume_boundary_vertex_ids(state: VolumetricPitoisState) -> set[int]:
    ids = {id(v) for v in state.bV_caps}
    ids.update(_contact_line_vertex_ids(state))
    return ids


def _enforce_step_continuity_displacement(
    state: VolumetricPitoisState,
    *,
    reference_volume_m3: float,
    dt: float,
    max_iterations: int = 12,
) -> None:
    """Apply the integrated continuity equation to the ALE mesh displacement.

    This is not a target-volume rescale. It corrects the current substep's
    accumulated boundary displacement so the closed volumetric .msh satisfies
    the discrete incompressibility condition over this one time step.
    """
    state.last_step_continuity_volume_error_m3 = 0.0
    state.last_step_continuity_displacement_m3 = 0.0
    if (not bool(getattr(state.config, "enable_continuity_pressure", False))) or dt <= 0.0:
        return

    reference_volume_m3 = float(reference_volume_m3)
    if reference_volume_m3 <= 0.0:
        return

    prescribed_ids = _prescribed_volume_boundary_vertex_ids(state)
    tolerance = max(1.0e-15 * reference_volume_m3, 1.0e-21)
    total_correction = 0.0
    final_error = 0.0

    for _iteration in range(max(1, int(max_iterations))):
        points, tets, _node_ids = _snapshot_msh_indexed_mesh(state)
        current_volume_m3 = float(_indexed_tet_mesh_volume_m3(points, tets))
        volume_error_m3 = current_volume_m3 - reference_volume_m3
        final_error = volume_error_m3
        if abs(volume_error_m3) <= tolerance:
            break

        tet_pts = points[tets]
        a = tet_pts[:, 0, :]
        b = tet_pts[:, 1, :]
        c = tet_pts[:, 2, :]
        d = tet_pts[:, 3, :]
        signed = _tet_signed_volumes(points, tets)
        abs_sign = np.where(signed >= 0.0, 1.0, -1.0)[:, None]
        grad_a = abs_sign * np.cross(b - d, c - d) / 6.0
        grad_b = abs_sign * np.cross(c - d, a - d) / 6.0
        grad_c = abs_sign * np.cross(a - d, b - d) / 6.0
        grad_d = -(grad_a + grad_b + grad_c)

        gradients = np.zeros_like(points, dtype=float)
        for local_col, local_grad in enumerate((grad_a, grad_b, grad_c, grad_d)):
            np.add.at(gradients, tets[:, local_col], local_grad)

        free_indices = [
            idx
            for idx, vertex in enumerate(state.volume_export_vertices)
            if id(vertex) not in prescribed_ids
        ]
        if not free_indices:
            break
        free_indices_arr = np.asarray(free_indices, dtype=int)
        free_gradients = gradients[free_indices_arr]
        grad_norm_sq = np.sum(free_gradients * free_gradients, axis=1)
        active = grad_norm_sq > 1.0e-30
        if not np.any(active):
            break
        free_indices_arr = free_indices_arr[active]
        free_gradients = free_gradients[active]

        denom = float(np.sum(free_gradients * free_gradients))
        if denom <= 1.0e-30:
            break
        lambda_x = -volume_error_m3 / denom
        vertices = []
        targets = []
        displacements: list[np.ndarray] = []
        for idx, gradient in zip(free_indices_arr, free_gradients):
            v = state.volume_export_vertices[int(idx)]
            old = np.asarray(v.x_a[:3], dtype=float)
            displacement = lambda_x * gradient
            vertices.append(v)
            targets.append(tuple(old + displacement))
            displacements.append(displacement)

        _move_vertices_batch(vertices, targets, state.HC, state.bV_caps)
        for v, displacement in zip(vertices, displacements):
            v.u = np.asarray(v.u[:3], dtype=float) + displacement / float(dt)
        total_correction += -volume_error_m3

    state.last_step_continuity_volume_error_m3 = float(final_error)
    state.last_step_continuity_displacement_m3 = float(total_correction)


def _enforce_step_volume_guard(
    state: VolumetricPitoisState,
    *,
    reference_volume_m3: float,
) -> None:
    reference_volume_m3 = float(reference_volume_m3)
    if reference_volume_m3 <= 0.0:
        return
    current_volume_m3 = float(_snapshot_msh_volume_m3(state))
    volume_error_m3 = current_volume_m3 - reference_volume_m3
    state.last_step_continuity_volume_error_m3 = float(volume_error_m3)
    stop_fraction = max(0.0, float(getattr(state.config, "volume_constraint_stop_fraction", 0.0)))
    relative_error = abs(volume_error_m3) / max(abs(reference_volume_m3), 1.0e-30)
    if stop_fraction > 0.0 and relative_error > stop_fraction:
        raise RuntimeError(
            "Volume constraint failed after substep: "
            f"reference={reference_volume_m3 * 1.0e9:.6f} uL, "
            f"current={current_volume_m3 * 1.0e9:.6f} uL, "
            f"error={volume_error_m3 * 1.0e9:+.6f} uL "
            f"({100.0 * relative_error:.4f} %, limit {100.0 * stop_fraction:.4f} %)."
        )


def _translate_layer(state: VolumetricPitoisState, layer_idx: int, shift: np.ndarray) -> None:
    shift = np.asarray(shift, dtype=float)
    if float(np.linalg.norm(shift)) <= 1.0e-30:
        return
    vertices = [state.layer_centers[layer_idx]]
    for ring in state.layer_rings[layer_idx]:
        vertices.extend(ring)
    targets = [tuple(np.asarray(v.x_a[:3], dtype=float) + shift) for v in vertices]
    _move_vertices_batch(vertices, targets, state.HC, state.bV_caps)


def _adjacent_layer_pair_min_distance(state: VolumetricPitoisState, layer_idx: int) -> float:
    if layer_idx < 0 or layer_idx >= len(state.layer_rings) - 1:
        return 0.0
    distances = []
    distances.append(
        float(
            np.linalg.norm(
                np.asarray(state.layer_centers[layer_idx].x_a[:3], dtype=float)
                - np.asarray(state.layer_centers[layer_idx + 1].x_a[:3], dtype=float)
            )
        )
    )
    lower_rings = state.layer_rings[layer_idx]
    upper_rings = state.layer_rings[layer_idx + 1]
    for lower_ring, upper_ring in zip(lower_rings, upper_rings):
        n_pair = min(len(lower_ring), len(upper_ring))
        for i in range(n_pair):
            distances.append(
                float(
                    np.linalg.norm(
                        np.asarray(lower_ring[i].x_a[:3], dtype=float)
                        - np.asarray(upper_ring[i].x_a[:3], dtype=float)
                    )
                )
            )
    return min(distances) if distances else 0.0


def _adjacent_layer_min_distance(state: VolumetricPitoisState) -> float:
    if len(state.layer_rings) < 2:
        return 0.0
    return min(_adjacent_layer_pair_min_distance(state, k) for k in range(len(state.layer_rings) - 1))


def _enforce_min_adjacent_layer_spacing(state: VolumetricPitoisState) -> None:
    baseline = float(getattr(state, "baseline_adjacent_layer_spacing_min_m", 0.0))
    if baseline <= 0.0 or len(state.layer_rings) < 2:
        return
    target = max(
        float(state.config.layer_spacing_min_fraction) * baseline,
        1.0e-8,
    )
    _, _, axis = _contact_plane_centers_and_axis(SimpleNamespace(outer_rings=state.outer_rings))
    for _ in range(3):
        changed = False
        for layer_idx in range(len(state.layer_rings) - 1):
            current = _adjacent_layer_pair_min_distance(state, layer_idx)
            deficit = target - current
            if deficit <= 0.0:
                continue
            changed = True
            if layer_idx == 0:
                _translate_layer(state, layer_idx + 1, +deficit * axis)
            elif layer_idx == len(state.layer_rings) - 2:
                _translate_layer(state, layer_idx, -deficit * axis)
            else:
                half_shift = 0.5 * deficit * axis
                _translate_layer(state, layer_idx, -half_shift)
                _translate_layer(state, layer_idx + 1, +half_shift)
        if not changed:
            break


def _regularize_layer_order(state: VolumetricPitoisState) -> None:
    if len(state.outer_rings) < 3:
        return

    bottom_ring_center, top_ring_center, axis = _contact_plane_centers_and_axis(
        SimpleNamespace(outer_rings=state.outer_rings)
    )
    centers = [
        np.mean([np.asarray(v.x_a[:3], dtype=float) for v in ring], axis=0)
        for ring in state.outer_rings
    ]
    total_span = float(np.dot(top_ring_center - bottom_ring_center, axis))
    if total_span <= 1.0e-30:
        return

    for k in range(1, len(state.outer_rings) - 1):
        center_k = centers[k]
        s_k = float(np.dot(center_k - bottom_ring_center, axis))
        target_s = float(state.layer_fractions[k]) * total_span
        shift = (target_s - s_k) * axis
        if np.linalg.norm(shift) <= 1.0e-30:
            continue
        for ring in state.layer_rings[k]:
            for v in ring:
                _move(v, tuple(np.asarray(v.x_a[:3], dtype=float) + shift), state.HC, state.bV_caps)
        center_v = state.layer_centers[k]
        _move(
            center_v,
            tuple(np.asarray(center_v.x_a[:3], dtype=float) + shift),
            state.HC,
            state.bV_caps,
        )
    _enforce_min_adjacent_layer_spacing(state)


def _base_pressure_model(v, *, HC=None, dim: int = 3, state: VolumetricPitoisState):
    pressure = 0.0
    if not state.config.include_gravity:
        return pressure

    z_ref = 0.5 * float(state.bottom_sphere_center[2] + state.top_sphere_center[2])
    z = float(np.asarray(v.x_a[:3], dtype=float)[2])
    return pressure - float(state.config.rho_f) * float(state.config.gravity_mps2) * (z - z_ref)


def _pressure_model(v, *, HC=None, dim: int = 3, state: VolumetricPitoisState):
    return _base_pressure_model(v, HC=HC, dim=dim, state=state) + float(
        state.pressure_projection_map.get(id(v), 0.0)
    )


def _velocity_from_map(v, velocity_map: dict[int, np.ndarray]) -> np.ndarray:
    mapped = velocity_map.get(id(v))
    if mapped is not None:
        return np.asarray(mapped, dtype=float)
    return np.asarray(v.u[:3], dtype=float)


def _velocity_difference_tensor_pointwise_from_map(
    v,
    *,
    state: VolumetricPitoisState,
    velocity_map: dict[int, np.ndarray],
) -> np.ndarray:
    du = np.zeros((3, 3), dtype=float)
    valid = False
    for v_j in v.nn:
        try:
            area_vec = _dual_area_vector_cached(v, v_j, state)
        except (KeyError, IndexError, ValueError, RuntimeError, ZeroDivisionError, StopIteration):
            continue
        area_vec = np.asarray(area_vec, dtype=float)
        if area_vec.shape != (3,) or not np.all(np.isfinite(area_vec)):
            continue
        delta_u = _velocity_from_map(v_j, velocity_map) - _velocity_from_map(v, velocity_map)
        if delta_u.shape != (3,) or not np.all(np.isfinite(delta_u)):
            continue
        du += np.outer(delta_u, area_vec)
        valid = True
    if not valid:
        return np.zeros((3, 3), dtype=float)
    return 0.5 * du / _vertex_dual_volume_m3(v)


def _velocity_divergence_from_map(
    v,
    *,
    state: VolumetricPitoisState,
    velocity_map: dict[int, np.ndarray],
) -> float:
    return float(np.trace(_velocity_difference_tensor_pointwise_from_map(v, state=state, velocity_map=velocity_map)))


def _pressure_projection_laplacian_rows(
    state: VolumetricPitoisState,
    vertices: list,
) -> tuple[list[tuple[float, list[tuple[int, float]]]], dict[int, int]]:
    vertex_ids = tuple(id(v) for v in vertices)
    cached = getattr(state, "pressure_projection_rows_cache", None)
    if cached is not None and cached[0] == vertex_ids:
        return cached[1], cached[2]

    index = {id(v): i for i, v in enumerate(vertices)}
    rows: list[tuple[float, list[tuple[int, float]]]] = []
    for v in vertices:
        volume_i = _vertex_dual_volume_m3(v)
        diag = 0.0
        entries: list[tuple[int, float]] = []
        x_i = np.asarray(v.x_a[:3], dtype=float)
        for v_j in v.nn:
            x_j = np.asarray(v_j.x_a[:3], dtype=float)
            edge = x_j - x_i
            dist = float(np.linalg.norm(edge))
            if dist <= 1.0e-30:
                continue
            try:
                area_vec = _dual_area_vector_cached(v, v_j, state)
            except (KeyError, IndexError, ValueError, RuntimeError, ZeroDivisionError, StopIteration):
                continue
            area_vec = np.asarray(area_vec, dtype=float)
            if area_vec.shape != (3,) or not np.all(np.isfinite(area_vec)):
                continue
            face_measure = abs(float(np.dot(area_vec, edge / dist)))
            if face_measure <= 1.0e-30:
                continue
            weight = face_measure / max(volume_i * dist, 1.0e-30)
            if not np.isfinite(weight) or weight <= 0.0:
                continue
            diag += weight
            j = index.get(id(v_j))
            if j is not None:
                entries.append((j, float(weight)))
        rows.append((float(diag), entries))
    state.pressure_projection_rows_cache = (vertex_ids, rows, index)
    return rows, index


def _solve_projection_poisson_sor(
    rows: list[tuple[float, list[tuple[int, float]]]],
    rhs: np.ndarray,
    *,
    max_iters: int,
    tol: float,
    omega: float,
) -> tuple[np.ndarray, int, float]:
    rhs = np.asarray(rhs, dtype=float)
    n = int(rhs.size)
    if n == 0:
        return np.zeros(0, dtype=float), 0, 0.0

    pressure = np.zeros(n, dtype=float)
    rhs_l2 = max(float(np.sqrt(np.mean(rhs * rhs))), 1.0e-30)
    tol_abs = max(float(tol) * rhs_l2, 1.0e-30)
    omega = float(np.clip(omega, 0.1, 1.95))
    residual_l2 = rhs_l2
    iterations = 0

    for iteration in range(1, max(1, int(max_iters)) + 1):
        for i, (diag, entries) in enumerate(rows):
            if diag <= 1.0e-30:
                continue
            neighbor_sum = sum(weight * pressure[j] for j, weight in entries)
            target = (neighbor_sum - rhs[i]) / diag
            pressure[i] = (1.0 - omega) * pressure[i] + omega * target

        residual = np.zeros(n, dtype=float)
        for i, (diag, entries) in enumerate(rows):
            laplace_i = sum(weight * pressure[j] for j, weight in entries) - diag * pressure[i]
            residual[i] = laplace_i - rhs[i]
        residual_l2 = float(np.sqrt(np.mean(residual * residual)))
        iterations = iteration
        if residual_l2 <= tol_abs:
            break

    return pressure, iterations, residual_l2


def _solve_ns_projection_pressure(state: VolumetricPitoisState, *, dt: float) -> None:
    state.pressure_projection_map.clear()
    state.last_ns_projection_iterations = 0
    state.last_ns_divergence_l2 = 0.0
    state.last_ns_projected_divergence_l2 = 0.0
    state.last_ns_pressure_l2 = 0.0
    _clear_force_caches(state)

    if (
        not bool(getattr(state.config, "enable_ns_pressure_projection", False))
        or dt <= 0.0
        or float(state.config.rho_f) <= 0.0
    ):
        return

    free_vertices = [v for v in state.HC.V if v not in state.bV_caps]
    if not free_vertices:
        return

    if _can_use_array_force_backend(state):
        try:
            geom = _array_force_geometry(state)
            points, velocities, volumes = _array_state_vectors(state, geom)
            base_pressure = _pressure_array_for_vertices(
                state,
                geom,
                points,
                include_projection=False,
            )
            packed = _array_force_all(
                state,
                pressure=base_pressure,
                velocities=velocities,
                include_contact_line=True,
            )
            if packed is not None:
                force, geom, points, velocities, volumes = packed
                mass = np.maximum(float(state.config.rho_f) * volumes, 1.0e-18)
                accel = _clip_acceleration_array(force / mass[:, None], state)
                velocity_star_arr = velocities.copy()
                bV_ids = {id(v) for v in state.bV_caps}
                moving_mask = np.asarray([id(v) not in bV_ids for v in geom["vertices"]], dtype=bool)
                velocity_star_arr[moving_mask] += float(dt) * accel[moving_mask]
                if bool(getattr(state.config, "enforce_no_swirl", False)):
                    axis_origin, axis = _swirl_axis_geometry(state)
                    velocity_star_arr = _remove_swirl_components_array(
                        velocity_star_arr,
                        points=points,
                        axis_origin=axis_origin,
                        axis=axis,
                    )

                index = geom["index"]
                free_pairs = [(v, index.get(id(v))) for v in free_vertices]
                free_pairs = [(v, int(idx)) for v, idx in free_pairs if idx is not None]
                if free_pairs:
                    free_vertices_array = [v for v, _idx in free_pairs]
                    free_indices = np.asarray([idx for _v, idx in free_pairs], dtype=np.int64)
                    backend = _array_force_backend_name(state)
                    if backend == "torch":
                        du_star = _velocity_gradient_array_torch(geom, velocity_star_arr, volumes)
                    else:
                        du_star = _velocity_gradient_array_numpy(geom, velocity_star_arr, volumes)
                    div_star = np.trace(du_star[free_indices], axis1=1, axis2=2)
                    if div_star.size and np.all(np.isfinite(div_star)):
                        rows, _index = _pressure_projection_laplacian_rows(state, free_vertices_array)
                        rhs = (float(state.config.rho_f) / float(dt)) * div_star
                        pressure, iterations, residual_l2 = _solve_projection_poisson_sor(
                            rows,
                            rhs,
                            max_iters=int(state.config.ns_pressure_projection_max_iters),
                            tol=float(state.config.ns_pressure_projection_tol),
                            omega=float(state.config.ns_pressure_projection_sor),
                        )
                        if pressure.size == len(free_vertices_array) and np.all(np.isfinite(pressure)):
                            state.pressure_projection_map = {
                                id(v): float(p)
                                for v, p in zip(free_vertices_array, pressure)
                                if np.isfinite(float(p))
                            }
                            state.last_ns_projection_iterations = int(iterations)
                            state.last_ns_divergence_l2 = float(np.sqrt(np.mean(div_star * div_star)))
                            state.last_ns_projected_divergence_l2 = float(
                                float(dt) * residual_l2 / max(float(state.config.rho_f), 1.0e-30)
                            )
                            state.last_ns_pressure_l2 = float(np.sqrt(np.mean(pressure * pressure)))
                            _clear_force_caches(state)
                            return
        except Exception:
            state.pressure_projection_map.clear()
            state.last_ns_projection_iterations = 0
            state.last_ns_divergence_l2 = 0.0
            state.last_ns_projected_divergence_l2 = 0.0
            state.last_ns_pressure_l2 = 0.0
            _clear_force_caches(state)

    base_pressure_model = lambda vv, HC=None, dim=3: _base_pressure_model(vv, HC=HC, dim=dim, state=state)
    velocity_star: dict[int, np.ndarray] = {}
    axis_origin = axis = None
    if bool(getattr(state.config, "enforce_no_swirl", False)):
        axis_origin, axis = _swirl_axis_geometry(state)

    for v in state.HC.V:
        if v in state.bV_caps:
            velocity_star[id(v)] = np.asarray(v.u[:3], dtype=float).copy()
            continue
        force = _Ftot(
            v,
            state=state,
            pressure_model=base_pressure_model,
            include_contact_line=True,
        )
        accel = _clip_acceleration(force / _vertex_mass_kg(v, state), state)
        u_star = np.asarray(v.u[:3], dtype=float) + float(dt) * accel
        if axis_origin is not None and axis is not None:
            u_star = _remove_swirl_component(
                u_star,
                point=np.asarray(v.x_a[:3], dtype=float),
                axis_origin=axis_origin,
                axis=axis,
            )
        velocity_star[id(v)] = np.asarray(u_star, dtype=float)

    _clear_force_caches(state)
    rows, _index = _pressure_projection_laplacian_rows(state, free_vertices)
    div_star = np.array(
        [
            _velocity_divergence_from_map(v, state=state, velocity_map=velocity_star)
            for v in free_vertices
        ],
        dtype=float,
    )
    if div_star.size == 0 or not np.all(np.isfinite(div_star)):
        return

    rhs = (float(state.config.rho_f) / float(dt)) * div_star
    pressure, iterations, residual_l2 = _solve_projection_poisson_sor(
        rows,
        rhs,
        max_iters=int(state.config.ns_pressure_projection_max_iters),
        tol=float(state.config.ns_pressure_projection_tol),
        omega=float(state.config.ns_pressure_projection_sor),
    )
    if pressure.size != len(free_vertices) or not np.all(np.isfinite(pressure)):
        return

    state.pressure_projection_map = {
        id(v): float(p)
        for v, p in zip(free_vertices, pressure)
        if np.isfinite(float(p))
    }
    state.last_ns_projection_iterations = int(iterations)
    state.last_ns_divergence_l2 = float(np.sqrt(np.mean(div_star * div_star)))
    state.last_ns_projected_divergence_l2 = float(float(dt) * residual_l2 / max(float(state.config.rho_f), 1.0e-30))
    state.last_ns_pressure_l2 = float(np.sqrt(np.mean(pressure * pressure)))
    _clear_force_caches(state)


def _set_cap_velocities(state: VolumetricPitoisState) -> None:
    cap_speed = float(state.config.cap_speed)
    bottom_u = np.array([0.0, 0.0, -cap_speed], dtype=float)
    top_u = np.zeros(3, dtype=float)
    for v in state.cap_bottom_interior:
        v.u = bottom_u.copy()
    for v in state.cap_top_interior:
        v.u = top_u.copy()


def _contact_plane_centers_and_axis(state_or_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer_rings = state_or_data.outer_rings if hasattr(state_or_data, "outer_rings") else state_or_data
    bottom_ring_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in outer_rings[0]], axis=0)
    top_ring_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in outer_rings[-1]], axis=0)
    axis = top_ring_center - bottom_ring_center
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    return bottom_ring_center, top_ring_center, axis


def _swirl_axis_geometry(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    bottom = np.asarray(state.bottom_sphere_center, dtype=float)
    top = np.asarray(state.top_sphere_center, dtype=float)
    axis = top - bottom
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    return 0.5 * (bottom + top), axis


def _azimuthal_unit(
    point: np.ndarray,
    *,
    axis_origin: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    rel = np.asarray(point, dtype=float) - np.asarray(axis_origin, dtype=float)
    axial = float(np.dot(rel, axis))
    radial = rel - axial * axis
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm <= 1.0e-30:
        return np.zeros(3, dtype=float)
    e_r = radial / radial_norm
    e_theta = np.cross(axis, e_r)
    e_theta_norm = float(np.linalg.norm(e_theta))
    if e_theta_norm <= 1.0e-30:
        return np.zeros(3, dtype=float)
    return e_theta / e_theta_norm


def _remove_swirl_component(
    vector: np.ndarray,
    *,
    point: np.ndarray,
    axis_origin: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    e_theta = _azimuthal_unit(point, axis_origin=axis_origin, axis=axis)
    if float(np.linalg.norm(e_theta)) <= 1.0e-30:
        return np.asarray(vector, dtype=float)
    vec = np.asarray(vector, dtype=float)
    return vec - float(np.dot(vec, e_theta)) * e_theta


def _enforce_no_swirl_velocity_field(state: VolumetricPitoisState, *, vertices=None) -> None:
    if not bool(getattr(state.config, "enforce_no_swirl", False)):
        return
    axis_origin, axis = _swirl_axis_geometry(state)
    if vertices is None:
        vertices = list(state.HC.V)
    for v in vertices:
        u = np.asarray(getattr(v, "u", np.zeros(3, dtype=float))[:3], dtype=float)
        v.u = _remove_swirl_component(
            u,
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=axis_origin,
            axis=axis,
        )


def _radial_axial_components_for_point(
    point: np.ndarray,
    velocity: np.ndarray,
    *,
    axis_origin: np.ndarray,
    axis: np.ndarray,
) -> tuple[np.ndarray, bool, float, float]:
    rel = np.asarray(point, dtype=float) - np.asarray(axis_origin, dtype=float)
    axial = float(np.dot(rel, axis))
    radial_vec = rel - axial * axis
    radial_norm = float(np.linalg.norm(radial_vec))
    if radial_norm <= 1.0e-30:
        e_r = np.zeros(3, dtype=float)
        u_r = 0.0
        active = False
    else:
        e_r = radial_vec / radial_norm
        u_r = float(np.dot(np.asarray(velocity, dtype=float), e_r))
        active = True
    u_z = float(np.dot(np.asarray(velocity, dtype=float), axis))
    return e_r, active, u_r, u_z


def _average_radial_axial_velocity_ring(
    ring: list,
    *,
    axis_origin: np.ndarray,
    axis: np.ndarray,
) -> set[int]:
    entries = []
    for v in ring:
        point = np.asarray(v.x_a[:3], dtype=float)
        velocity = np.asarray(getattr(v, "u", np.zeros(3, dtype=float))[:3], dtype=float)
        e_r, active, u_r, u_z = _radial_axial_components_for_point(
            point,
            velocity,
            axis_origin=axis_origin,
            axis=axis,
        )
        entries.append((v, e_r, active, u_r, u_z))
    if not entries:
        return set()

    axial_mean = float(np.mean([entry[4] for entry in entries]))
    radial_values = [entry[3] for entry in entries if entry[2]]
    radial_mean = float(np.mean(radial_values)) if radial_values else 0.0
    touched: set[int] = set()
    for v, e_r, active, _u_r, _u_z in entries:
        averaged = axial_mean * axis
        if active:
            averaged = averaged + radial_mean * e_r
        v.u = np.asarray(averaged, dtype=float)
        touched.add(id(v))
    return touched


def _enforce_radial_axial_velocity_average_field(state: VolumetricPitoisState) -> None:
    if not bool(getattr(state.config, "average_velocity_radial_axial", False)):
        return
    axis_origin, axis = _swirl_axis_geometry(state)
    axis = np.asarray(axis, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-30)

    touched: set[int] = set()
    for rings in list(getattr(state, "layer_rings", [])):
        for ring in rings:
            touched.update(
                _average_radial_axial_velocity_ring(
                    list(ring),
                    axis_origin=axis_origin,
                    axis=axis,
                )
            )

    center_vertices = []
    center_vertices.extend(list(getattr(state, "layer_centers", [])))
    for name in ("cap_bottom_center", "cap_top_center"):
        vertex = getattr(state, name, None)
        if vertex is not None:
            center_vertices.append(vertex)
    for v in center_vertices:
        if id(v) in touched:
            continue
        velocity = np.asarray(getattr(v, "u", np.zeros(3, dtype=float))[:3], dtype=float)
        v.u = float(np.dot(velocity, axis)) * axis
        touched.add(id(v))

    for v in list(state.HC.V):
        if id(v) in touched:
            continue
        point = np.asarray(v.x_a[:3], dtype=float)
        velocity = np.asarray(getattr(v, "u", np.zeros(3, dtype=float))[:3], dtype=float)
        e_r, active, u_r, u_z = _radial_axial_components_for_point(
            point,
            velocity,
            axis_origin=axis_origin,
            axis=axis,
        )
        projected = u_z * axis
        if active:
            projected = projected + u_r * e_r
        v.u = np.asarray(projected, dtype=float)


def _mean_ring_radius(ring: list, center: np.ndarray, axis: np.ndarray) -> float:
    coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
    tangential = coords - center[None, :]
    tangential -= np.outer(np.dot(tangential, axis), axis)
    return float(np.mean(np.linalg.norm(tangential, axis=1)))


def _orthonormal_tangent_basis(axis: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.asarray(reference, dtype=float) - axis * float(np.dot(reference, axis))
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1.0e-12:
        seed = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(seed, axis))) > 0.9:
            seed = np.array([0.0, 1.0, 0.0], dtype=float)
        tangent = seed - axis * float(np.dot(seed, axis))
        tangent_norm = float(np.linalg.norm(tangent))
    e1 = tangent / max(tangent_norm, 1.0e-30)
    e2 = np.cross(axis, e1)
    e2 /= max(float(np.linalg.norm(e2)), 1.0e-30)
    return e1, e2


def _estimate_contact_radius(
    state: VolumetricPitoisState,
    *,
    which: str,
    sphere_center: np.ndarray,
    contact_ring: list,
    next_ring: list,
    axis: np.ndarray,
    dt: float | None,
    gap_rate: float,
    prev_radius: float | None = None,
) -> float:
    if prev_radius is None:
        prev_radius = _cap_radius(contact_ring)
    prev_radius = float(prev_radius)
    if not next_ring:
        return prev_radius

    theta_target = np.deg2rad(float(state.config.contact_angle_deg))
    radius = float(state.config.particle_radius)
    inward_sign = 1.0 if which == "bottom" else -1.0
    alpha_prev = float(np.arcsin(np.clip(prev_radius / max(radius, 1.0e-30), -1.0, 1.0)))
    slide_limit = max(float(state.config.contact_line_max_slide_um) * 1.0e-6, 1.0e-9)
    if dt is not None and dt > 0.0:
        speed_factor = max(
            float(getattr(state.config, "contact_line_max_slide_speed_factor", 1.0)),
            0.0,
        )
        speed_limited_slide = abs(float(gap_rate)) * float(dt) * speed_factor
        if speed_limited_slide > 0.0:
            slide_limit = min(slide_limit, max(speed_limited_slide, 1.0e-9))
    delta_alpha_max = slide_limit / max(radius, 1.0e-30)
    alpha_min = max(1.0e-8, alpha_prev - delta_alpha_max)
    alpha_max = min(0.5 * np.pi - 1.0e-8, alpha_prev + delta_alpha_max)
    if gap_rate > 0.0:
        alpha_max = min(alpha_max, alpha_prev)
    elif gap_rate < 0.0:
        alpha_min = max(alpha_min, alpha_prev)
    r_min = max(1.0e-8 * radius, radius * np.sin(alpha_min))
    r_max = min(radius - 1.0e-6, radius * np.sin(alpha_max))
    if r_max <= r_min:
        return min(max(prev_radius, r_min), radius - 1.0e-6)

    n_candidates = max(32, int(state.config.contact_radius_samples))
    candidates = np.linspace(r_min, r_max, n_candidates, dtype=float)
    local_rings = state.outer_rings[: max(2, int(state.config.contact_line_fit_rings) + 1)]
    if which == "top":
        local_rings = list(reversed(state.outer_rings[-max(2, int(state.config.contact_line_fit_rings) + 1) :]))
    sample_pairs: list[tuple[float, float]] = []
    for ring in local_rings[1:]:
        coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
        rel = coords - sphere_center[None, :]
        s_k = float(np.mean(inward_sign * np.dot(rel, axis)))
        tangential = rel - np.outer(np.dot(rel, axis), axis)
        r_k = float(np.mean(np.linalg.norm(tangential, axis=1)))
        sample_pairs.append((s_k, r_k))
    if len(sample_pairs) < 2:
        coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in next_ring], dtype=float)
        rel = coords - sphere_center[None, :]
        s_k = float(np.mean(inward_sign * np.dot(rel, axis)))
        tangential = rel - np.outer(np.dot(rel, axis), axis)
        r_k = float(np.mean(np.linalg.norm(tangential, axis=1)))
        sample_pairs = [(s_k, r_k)]

    s_contact = np.sqrt(np.maximum(radius**2 - candidates**2, 1.0e-16))
    tangent_solid = np.stack([-s_contact, candidates], axis=1)
    tangent_solid /= np.maximum(np.linalg.norm(tangent_solid, axis=1, keepdims=True), 1.0e-30)

    angle = np.empty_like(candidates)
    for idx, (r_contact, s_c) in enumerate(zip(candidates, s_contact)):
        fit_pairs = [(s_c, r_contact)] + sample_pairs[: max(2, int(state.config.contact_line_fit_rings))]
        s_vals = np.asarray([pair[0] for pair in fit_pairs], dtype=float)
        r_vals = np.asarray([pair[1] for pair in fit_pairs], dtype=float)
        x = s_vals - s_c
        deg = 2 if x.size >= 3 else 1
        coeff = np.polyfit(x, r_vals, deg=deg)
        dr_ds = float(coeff[-2]) if deg >= 1 else 0.0
        tangent_liquid = np.array([dr_ds, 1.0], dtype=float)
        tangent_liquid /= max(float(np.linalg.norm(tangent_liquid)), 1.0e-30)
        dot = float(np.clip(np.dot(tangent_solid[idx], tangent_liquid), -1.0, 1.0))
        raw_angle = float(np.arccos(dot))
        angle[idx] = min(raw_angle, float(np.pi - raw_angle))

    theta_candidate = np.full_like(candidates, theta_target)
    if bool(getattr(state.config, "enable_dynamic_contact_angle", False)) and dt is not None and dt > 0.0:
        alpha_candidates = np.arcsin(np.clip(candidates / max(radius, 1.0e-30), -1.0, 1.0))
        u_cl = radius * (alpha_candidates - alpha_prev) / dt
        capillary_number = float(state.config.mu_f) * u_cl / max(float(state.config.gamma), 1.0e-30)
        ln_fac = float(
            np.log(
                max(
                    float(state.config.contact_line_cox_macro_length_m)
                    / max(float(state.config.contact_line_cox_slip_length_m), 1.0e-30),
                    1.0,
                )
            )
        )
        theta_candidate = np.cbrt(
            np.maximum(theta_target**3 + 9.0 * capillary_number * ln_fac, np.deg2rad(float(state.config.dynamic_contact_angle_min_deg)) ** 3)
        )
        theta_candidate = np.clip(
            theta_candidate,
            np.deg2rad(float(state.config.dynamic_contact_angle_min_deg)),
            np.deg2rad(float(state.config.dynamic_contact_angle_max_deg)),
        )

    angle_residual = np.abs(angle - theta_candidate)
    continuation_penalty = float(state.config.contact_line_continuation_weight) * np.abs(
        candidates - prev_radius
    ) / max(radius, 1.0e-30)
    growth_penalty = np.zeros_like(candidates)
    if not bool(getattr(state.config, "allow_contact_line_growth", False)):
        growth_penalty = 0.25 * np.maximum(candidates - prev_radius, 0.0) / max(radius, 1.0e-30)
    best = int(np.argmin(angle_residual + continuation_penalty + growth_penalty))
    return float(candidates[best])


def _contact_line_local_sample_pairs(
    state: VolumetricPitoisState,
    *,
    which: str,
    sphere_center: np.ndarray,
    axis: np.ndarray,
) -> list[tuple[float, float]]:
    inward_sign = 1.0 if which == "bottom" else -1.0
    local_rings = state.outer_rings[: max(2, int(state.config.contact_line_fit_rings) + 1)]
    if which == "top":
        local_rings = list(reversed(state.outer_rings[-max(2, int(state.config.contact_line_fit_rings) + 1) :]))
    sample_pairs: list[tuple[float, float]] = []
    for ring in local_rings[1:]:
        coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
        rel = coords - sphere_center[None, :]
        s_k = float(np.mean(inward_sign * np.dot(rel, axis)))
        tangential = rel - np.outer(np.dot(rel, axis), axis)
        r_k = float(np.mean(np.linalg.norm(tangential, axis=1)))
        sample_pairs.append((s_k, r_k))
    return sample_pairs


def _contact_angle_for_radius(
    state: VolumetricPitoisState,
    *,
    which: str,
    sphere_center: np.ndarray,
    axis: np.ndarray,
    contact_radius: float,
) -> float:
    radius = float(state.config.particle_radius)
    r_contact = float(np.clip(contact_radius, 1.0e-8 * radius, radius - 1.0e-8))
    sample_pairs = _contact_line_local_sample_pairs(
        state,
        which=which,
        sphere_center=sphere_center,
        axis=axis,
    )
    s_contact = float(np.sqrt(max(radius**2 - r_contact**2, 1.0e-16)))
    tangent_solid = np.array([-s_contact, r_contact], dtype=float)
    tangent_solid /= max(float(np.linalg.norm(tangent_solid)), 1.0e-30)
    fit_pairs = [(s_contact, r_contact)] + sample_pairs[: max(2, int(state.config.contact_line_fit_rings))]
    s_vals = np.asarray([pair[0] for pair in fit_pairs], dtype=float)
    r_vals = np.asarray([pair[1] for pair in fit_pairs], dtype=float)
    x = s_vals - s_contact
    deg = 2 if x.size >= 3 else 1
    coeff = np.polyfit(x, r_vals, deg=deg)
    dr_ds = float(coeff[-2]) if deg >= 1 else 0.0
    tangent_liquid = np.array([dr_ds, 1.0], dtype=float)
    tangent_liquid /= max(float(np.linalg.norm(tangent_liquid)), 1.0e-30)
    dot = float(np.clip(np.dot(tangent_solid, tangent_liquid), -1.0, 1.0))
    raw_angle = float(np.arccos(dot))
    return min(raw_angle, float(np.pi - raw_angle))


def _dynamic_contact_angle_from_speed(state: VolumetricPitoisState, *, slide_speed: float) -> float:
    theta_eq = float(np.deg2rad(state.config.contact_angle_deg))
    if not bool(getattr(state.config, "enable_dynamic_contact_angle", False)):
        return theta_eq
    capillary_number = float(state.config.mu_f) * float(slide_speed) / max(float(state.config.gamma), 1.0e-30)
    ln_fac = float(
        np.log(
            max(
                float(state.config.contact_line_cox_macro_length_m)
                / max(float(state.config.contact_line_cox_slip_length_m), 1.0e-30),
                1.0,
            )
        )
    )
    theta_dyn = float(
        np.cbrt(
            max(
                theta_eq**3 + 9.0 * capillary_number * ln_fac,
                np.deg2rad(float(state.config.dynamic_contact_angle_min_deg)) ** 3,
            )
        )
    )
    return float(
        np.clip(
            theta_dyn,
            np.deg2rad(float(state.config.dynamic_contact_angle_min_deg)),
            np.deg2rad(float(state.config.dynamic_contact_angle_max_deg)),
        )
    )


def _cox_voinov_contact_line_speed(
    state: VolumetricPitoisState,
    *,
    theta_dynamic: float,
) -> float:
    if not bool(getattr(state.config, "use_cox_voinov_contact_line_law", False)):
        return 0.0
    if not bool(getattr(state.config, "enable_dynamic_contact_angle", False)):
        return 0.0

    theta_eq = float(np.deg2rad(state.config.contact_angle_deg))
    theta_min = float(np.deg2rad(float(state.config.dynamic_contact_angle_min_deg)))
    theta_max = float(np.deg2rad(float(state.config.dynamic_contact_angle_max_deg)))
    theta_d = float(np.clip(float(theta_dynamic), theta_min, max(theta_max, theta_min)))
    ln_fac = float(
        np.log(
            max(
                float(state.config.contact_line_cox_macro_length_m)
                / max(float(state.config.contact_line_cox_slip_length_m), 1.0e-30),
                1.0,
            )
        )
    )
    denom = 9.0 * max(float(state.config.mu_f), 1.0e-30) * max(ln_fac, 1.0e-30)
    return float(float(state.config.gamma) * (theta_d**3 - theta_eq**3) / denom)


def _cox_voinov_contact_radius_update(
    state: VolumetricPitoisState,
    *,
    which: str,
    sphere_center: np.ndarray,
    axis: np.ndarray,
    previous_radius: float,
    dt: float,
) -> tuple[float, float]:
    radius = max(float(state.config.particle_radius), 1.0e-30)
    previous_radius = float(np.clip(float(previous_radius), 1.0e-8 * radius, radius * (1.0 - 1.0e-10)))
    theta_dynamic = _contact_angle_for_radius(
        state,
        which=which,
        sphere_center=sphere_center,
        axis=axis,
        contact_radius=previous_radius,
    )
    u_cl = _cox_voinov_contact_line_speed(state, theta_dynamic=theta_dynamic)
    alpha_old = float(np.arcsin(np.clip(previous_radius / radius, -1.0, 1.0)))
    alpha_min = 1.0e-8
    alpha_max = float(np.arcsin(np.clip(1.0 - 1.0e-10, -1.0, 1.0)))
    alpha_new = float(np.clip(alpha_old + float(dt) * u_cl / radius, alpha_min, alpha_max))
    new_radius = radius * float(np.sin(alpha_new))
    if not bool(getattr(state.config, "allow_contact_line_growth", False)):
        new_radius = min(new_radius, previous_radius)
        alpha_new = float(np.arcsin(np.clip(new_radius / radius, -1.0, 1.0)))
    actual_speed = radius * (alpha_new - alpha_old) / max(float(dt), 1.0e-30)
    return float(new_radius), float(actual_speed)


def _contact_line_slide_direction(
    point: np.ndarray,
    *,
    sphere_center: np.ndarray,
    axis: np.ndarray,
    which: str,
) -> np.ndarray:
    sign = 1.0 if which == "bottom" else -1.0
    rel = np.asarray(point, dtype=float) - np.asarray(sphere_center, dtype=float)
    axial_coeff = float(np.dot(rel, axis))
    tangential = rel - axial_coeff * axis
    radial_mag = float(np.linalg.norm(tangential))
    if radial_mag <= 1.0e-30:
        return np.zeros(3, dtype=float)
    tangential_unit = tangential / radial_mag
    slide_raw = sign * (axial_coeff * tangential_unit - radial_mag * axis)
    norm_slide = float(np.linalg.norm(slide_raw))
    if norm_slide <= 1.0e-30:
        return np.zeros(3, dtype=float)
    return slide_raw / norm_slide


def _contact_line_ring_speed(
    state: VolumetricPitoisState,
    *,
    ring: list,
    sphere_center: np.ndarray,
    axis: np.ndarray,
    which: str,
) -> float:
    sphere_vel = np.array([0.0, 0.0, -float(state.config.cap_speed)], dtype=float) if which == "bottom" else np.zeros(3, dtype=float)
    signed_speeds = []
    for v in ring:
        point = np.asarray(v.x_a[:3], dtype=float)
        slide_dir = _contact_line_slide_direction(point, sphere_center=sphere_center, axis=axis, which=which)
        if float(np.linalg.norm(slide_dir)) <= 1.0e-30:
            continue
        u_rel = np.asarray(v.u[:3], dtype=float) - sphere_vel
        signed_speeds.append(float(np.dot(u_rel, slide_dir)))
    if not signed_speeds:
        return 0.0
    return float(np.mean(signed_speeds))


def _constrain_contact_ring_to_sphere(
    state: VolumetricPitoisState,
    *,
    which: str,
    project_velocity: bool = True,
) -> float:
    if which == "bottom":
        ring = list(state.bottom_contact_ring)
        sphere_center = np.asarray(state.bottom_sphere_center, dtype=float)
        sphere_vel = np.array([0.0, 0.0, -float(state.config.cap_speed)], dtype=float)
        sign = 1.0
    elif which == "top":
        ring = list(state.top_contact_ring)
        sphere_center = np.asarray(state.top_sphere_center, dtype=float)
        sphere_vel = np.zeros(3, dtype=float)
        sign = -1.0
    else:
        return 0.0
    if not ring:
        return 0.0

    axis = np.asarray(state.top_sphere_center - state.bottom_sphere_center, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm

    radius = max(float(state.config.particle_radius), 1.0e-30)
    max_ring_radius = radius * (1.0 - 1.0e-10)
    slide_speeds: list[float] = []
    for v in ring:
        old = np.asarray(v.x_a[:3], dtype=float)
        rel = old - sphere_center
        axial = axis * float(np.dot(rel, axis))
        radial = rel - axial
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm > max_ring_radius:
            radial *= max_ring_radius / max(radial_norm, 1.0e-30)
            radial_norm = max_ring_radius
        axial_mag = np.sqrt(max(radius**2 - radial_norm**2, 0.0))
        target = sphere_center + radial + sign * axial_mag * axis
        _move(v, tuple(target), state.HC, state.bV_caps)
        if not project_velocity:
            continue

        normal = target - sphere_center
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-30:
            normal = sign * axis
        else:
            normal = normal / normal_norm
        u_rel = np.asarray(v.u[:3], dtype=float) - sphere_vel
        u_rel_tangent = u_rel - float(np.dot(u_rel, normal)) * normal
        v.u = sphere_vel + u_rel_tangent

        slide_dir = _contact_line_slide_direction(
            target,
            sphere_center=sphere_center,
            axis=axis,
            which=which,
        )
        if float(np.linalg.norm(slide_dir)) > 1.0e-30:
            slide_speeds.append(float(np.dot(u_rel_tangent, slide_dir)))

    if not slide_speeds:
        return 0.0
    return float(np.mean(slide_speeds))


def _constrain_contact_lines_to_spheres(state: VolumetricPitoisState) -> None:
    state.last_bottom_contact_line_speed = abs(_constrain_contact_ring_to_sphere(state, which="bottom"))
    state.last_top_contact_line_speed = abs(_constrain_contact_ring_to_sphere(state, which="top"))
    _clear_geometry_caches(state)


def _project_contact_lines_to_spheres(state: VolumetricPitoisState) -> None:
    _constrain_contact_ring_to_sphere(state, which="bottom", project_velocity=False)
    _constrain_contact_ring_to_sphere(state, which="top", project_velocity=False)
    state.last_bottom_contact_line_speed = 0.0
    state.last_top_contact_line_speed = 0.0
    _clear_geometry_caches(state)


def _ring_segment_length_map(ring: list) -> dict[int, float]:
    if len(ring) < 2:
        return {id(v): 0.0 for v in ring}
    coords = [np.asarray(v.x_a[:3], dtype=float) for v in ring]
    seg_map: dict[int, float] = {}
    n = len(ring)
    for i, v in enumerate(ring):
        prev_len = float(np.linalg.norm(coords[i] - coords[(i - 1) % n]))
        next_len = float(np.linalg.norm(coords[(i + 1) % n] - coords[i]))
        seg_map[id(v)] = 0.5 * (prev_len + next_len)
    return seg_map


def _Fcl(v, *, state: VolumetricPitoisState) -> np.ndarray:
    vid = id(v)
    cached = state.contact_line_force_cache.get(vid)
    if cached is not None:
        return cached

    def store(force: np.ndarray) -> np.ndarray:
        out = np.asarray(force, dtype=float)
        state.contact_line_force_cache[vid] = out
        return out

    if _uses_exact_imported_msh(state) and bool(
        getattr(state.config, "use_cox_voinov_contact_line_law", False)
    ):
        return store(np.zeros(3, dtype=float))

    bottom_ids = {id(node) for node in state.bottom_contact_ring}
    top_ids = {id(node) for node in state.top_contact_ring}
    if vid in bottom_ids:
        which = "bottom"
        ring = state.bottom_contact_ring
        sphere_center = np.asarray(state.bottom_sphere_center, dtype=float)
    elif vid in top_ids:
        which = "top"
        ring = state.top_contact_ring
        sphere_center = np.asarray(state.top_sphere_center, dtype=float)
    else:
        return store(np.zeros(3, dtype=float))

    axis = np.asarray(state.top_sphere_center - state.bottom_sphere_center, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm

    contact_radius = float(_cap_radius(ring))
    theta_geom = _contact_angle_for_radius(
        state,
        which=which,
        sphere_center=sphere_center,
        axis=axis,
        contact_radius=contact_radius,
    )
    slide_speed = _contact_line_ring_speed(
        state,
        ring=ring,
        sphere_center=sphere_center,
        axis=axis,
        which=which,
    )
    theta_dyn = _dynamic_contact_angle_from_speed(state, slide_speed=slide_speed)

    radius = float(state.config.particle_radius)
    dr_eps = max(1.0e-7 * radius, 1.0e-8)
    r_minus = max(contact_radius - dr_eps, 1.0e-8 * radius)
    r_plus = min(contact_radius + dr_eps, radius - 1.0e-8)
    if r_plus <= r_minus:
        return store(np.zeros(3, dtype=float))
    theta_minus = _contact_angle_for_radius(
        state,
        which=which,
        sphere_center=sphere_center,
        axis=axis,
        contact_radius=r_minus,
    )
    theta_plus = _contact_angle_for_radius(
        state,
        which=which,
        sphere_center=sphere_center,
        axis=axis,
        contact_radius=r_plus,
    )
    dtheta_dr = (theta_plus - theta_minus) / max(r_plus - r_minus, 1.0e-30)
    residual = theta_geom - theta_dyn
    if abs(residual) <= 1.0e-12 or abs(dtheta_dr) <= 1.0e-12:
        return store(np.zeros(3, dtype=float))

    slide_dir = _contact_line_slide_direction(
        np.asarray(v.x_a[:3], dtype=float),
        sphere_center=sphere_center,
        axis=axis,
        which=which,
    )
    if float(np.linalg.norm(slide_dir)) <= 1.0e-30:
        return store(np.zeros(3, dtype=float))

    seg_len = _ring_segment_length_map(ring).get(vid, 0.0)
    drive_sign = -np.sign(residual * dtheta_dr)
    force_mag = float(state.config.gamma) * abs(np.cos(theta_geom) - np.cos(theta_dyn)) * float(seg_len)
    return store(drive_sign * force_mag * slide_dir)


def _align_layer_azimuths(state: VolumetricPitoisState) -> None:
    if len(state.outer_rings) < 2:
        return

    _, _, axis = _contact_plane_centers_and_axis(
        SimpleNamespace(outer_rings=state.outer_rings)
    )
    ref_ring = state.outer_rings[0]
    ref_coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ref_ring], dtype=float)
    ref_center = np.mean(ref_coords, axis=0)
    reference = ref_coords[0] - ref_center
    e1, e2 = _orthonormal_tangent_basis(axis, reference)

    def wrap(a: np.ndarray) -> np.ndarray:
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    ref_rel = ref_coords - ref_center[None, :]
    ref_rel -= np.outer(np.dot(ref_rel, axis), axis)
    # Preserve the inherited cyclic ring order. Sorting would hide a whole-ring
    # phase slip and let fixed connectivity land on the wrong geometric points.
    ref_angles = np.arctan2(ref_rel @ e2, ref_rel @ e1)

    for k, rings in enumerate(state.layer_rings):
        outer = state.outer_rings[k]
        outer_coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in outer], dtype=float)
        outer_center = np.mean(outer_coords, axis=0)
        rel = outer_coords - outer_center[None, :]
        rel -= np.outer(np.dot(rel, axis), axis)
        angles = np.arctan2(rel @ e2, rel @ e1)
        if len(angles) != len(ref_angles):
            continue
        shift = float(np.mean(wrap(angles - ref_angles)))
        if abs(shift) <= 1.0e-10:
            continue

        layer_center = np.asarray(state.layer_centers[k].x_a[:3], dtype=float)
        cos_s = float(np.cos(-shift))
        sin_s = float(np.sin(-shift))
        for ring in rings:
            targets = []
            for v in ring:
                pos = np.asarray(v.x_a[:3], dtype=float)
                rel_v = pos - layer_center
                axial = axis * float(np.dot(rel_v, axis))
                radial = rel_v - axial
                x = float(np.dot(radial, e1))
                y = float(np.dot(radial, e2))
                radial_rot = (cos_s * x - sin_s * y) * e1 + (sin_s * x + cos_s * y) * e2
                target = layer_center + axial + radial_rot
                targets.append(tuple(target))
            _move_vertices_batch(ring, targets, state.HC, state.bV_caps)


def _initial_axisymmetric_template(state: VolumetricPitoisState):
    bottom_plane_center, top_plane_center, axis = _contact_plane_centers_and_axis(
        SimpleNamespace(outer_rings=state.outer_rings)
    )
    mid = 0.5 * (bottom_plane_center + top_plane_center)
    ref_ring = state.outer_rings[0]
    ref_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in ref_ring], axis=0)
    reference = np.asarray(ref_ring[0].x_a[:3], dtype=float) - ref_center
    e1, e2 = _orthonormal_tangent_basis(axis, reference)

    layer_data = []
    for layer_center, rings in zip(state.layer_centers, state.layer_rings):
        outer_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in rings[-1]], axis=0)
        s = float(np.dot(outer_center - mid, axis))
        outer_radius = max(_mean_ring_radius(rings[-1], outer_center, axis), 1.0e-30)
        ring_entries = []
        for ring in rings:
            coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
            rel = coords - outer_center[None, :]
            rel -= np.outer(np.dot(rel, axis), axis)
            angles = np.arctan2(rel @ e2, rel @ e1)
            order = np.argsort(angles)
            ordered_vertices = [ring[i] for i in order]
            factor = _mean_ring_radius(ring, outer_center, axis) / outer_radius
            ring_entries.append((ordered_vertices, float(factor)))
        layer_data.append(
            {
                "center_vertex": layer_center,
                "s": s,
                "rings": ring_entries,
            }
        )

    return mid, axis, e1, e2, layer_data


def _unduloid_like_outer_radius(
    z_abs: np.ndarray,
    *,
    half_span: float,
    neck_radius: float,
    contact_radius: float,
    end_slope: float,
) -> np.ndarray:
    if half_span <= 1.0e-30:
        return np.full_like(z_abs, contact_radius, dtype=float)
    u = half_span * half_span
    a = contact_radius - neck_radius
    b = (2.0 * a / u) - (end_slope / (2.0 * half_span))
    c = ((end_slope / (2.0 * half_span)) - (a / u)) / u
    r = neck_radius + b * z_abs**2 + c * z_abs**4
    return np.clip(r, 1.0e-8 * contact_radius, contact_radius)


def _apply_initial_unduloid_like_profile(
    state: VolumetricPitoisState,
    *,
    template,
    neck_radius: float,
) -> None:
    mid, axis, e1, e2, layer_data = template
    s_vals = np.array([float(item["s"]) for item in layer_data], dtype=float)
    half_span = float(np.max(np.abs(s_vals)))
    contact_radius = float(state.config.target_cap_radius)
    axial_offset = float(np.sqrt(max(state.config.particle_radius**2 - contact_radius**2, 0.0)))
    end_slope = axial_offset / max(contact_radius, 1.0e-30)
    outer_radii = _unduloid_like_outer_radius(
        np.abs(s_vals),
        half_span=half_span,
        neck_radius=neck_radius,
        contact_radius=contact_radius,
        end_slope=end_slope,
    )

    for layer, outer_radius in zip(layer_data, outer_radii):
        center_coord = mid + float(layer["s"]) * axis
        _move(layer["center_vertex"], tuple(center_coord), state.HC, state.bV_caps)
        for ordered_vertices, factor in layer["rings"]:
            ring_radius = max(float(factor) * float(outer_radius), 0.0)
            n_ring = max(1, len(ordered_vertices))
            targets = []
            for i, v in enumerate(ordered_vertices):
                phi = 2.0 * np.pi * float(i) / float(n_ring)
                target = center_coord + ring_radius * (np.cos(phi) * e1 + np.sin(phi) * e2)
                targets.append(tuple(target))
            _move_vertices_batch(ordered_vertices, targets, state.HC, state.bV_caps)

    _rebuild_cap_on_sphere(
        state,
        which="bottom",
        sphere_center=state.bottom_sphere_center,
        contact_radius=contact_radius,
        axis=axis,
        dt=None,
    )
    _rebuild_cap_on_sphere(
        state,
        which="top",
        sphere_center=state.top_sphere_center,
        contact_radius=contact_radius,
        axis=axis,
        dt=None,
    )


def _initialise_unduloid_like_equilibrium(state: VolumetricPitoisState) -> None:
    template = _initial_axisymmetric_template(state)
    contact_radius = float(state.config.target_cap_radius)
    neck_ratio = float(np.clip(state.config.initial_neck_radius_ratio, 0.05, 0.98))
    neck_radius = neck_ratio * contact_radius
    _apply_initial_unduloid_like_profile(state, template=template, neck_radius=neck_radius)
    _update_duals_and_masses(state)


def _latest_initialshape_msh_path() -> Path | None:
    if HARDCODED_INITIAL_MSH_PATH.exists():
        return HARDCODED_INITIAL_MSH_PATH
    fig_dir = INITIALSHAPE_OUT_ROOT / "fig"
    if not fig_dir.exists():
        return None

    iter_paths: list[tuple[int, Path]] = []
    for path in fig_dir.glob("mesh_iter*.msh"):
        match = re.fullmatch(r"mesh_iter(\d+)", path.stem)
        if match:
            iter_paths.append((int(match.group(1)), path))
    if iter_paths:
        return max(iter_paths, key=lambda item: item[0])[1]

    final_mesh = fig_dir / "mesh_initial.msh"
    if final_mesh.exists():
        return final_mesh
    return None


def _read_gmsh2_tet_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    nodes = None
    tets: list[tuple[int, int, int, int]] = []

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line == "$Nodes":
            n_nodes = int(lines[idx + 1].strip())
            node_rows = []
            for row in lines[idx + 2 : idx + 2 + n_nodes]:
                parts = row.split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed node row in {path}: {row}")
                node_rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
            nodes = np.asarray(node_rows, dtype=float)
            idx += n_nodes + 2
        elif line == "$Elements":
            n_elements = int(lines[idx + 1].strip())
            for row in lines[idx + 2 : idx + 2 + n_elements]:
                parts = row.split()
                if len(parts) < 4:
                    continue
                etype = int(parts[1])
                ntags = int(parts[2])
                conn = [int(x) - 1 for x in parts[3 + ntags :]]
                if etype == 4 and len(conn) == 4:
                    tets.append(tuple(conn))
            idx += n_elements + 2
        idx += 1

    if nodes is None:
        raise ValueError(f"No $Nodes section found in {path}")
    return np.asarray(nodes, dtype=float), np.asarray(tets, dtype=int)


def _boundary_surface_points_from_tets(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    tet_idx = np.asarray(tets, dtype=int)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    if tet_idx.ndim != 2 or tet_idx.shape[1] != 4 or tet_idx.shape[0] == 0:
        return np.asarray(pts, dtype=float)

    face_counts: dict[tuple[int, int, int], int] = defaultdict(int)
    for a, b, c, d in tet_idx:
        for face in (
            tuple(sorted((int(a), int(b), int(c)))),
            tuple(sorted((int(a), int(b), int(d)))),
            tuple(sorted((int(a), int(c), int(d)))),
            tuple(sorted((int(b), int(c), int(d)))),
        ):
            face_counts[face] += 1

    boundary_ids: set[int] = set()
    for face, count in face_counts.items():
        if count == 1:
            boundary_ids.update(face)
    if not boundary_ids:
        return np.asarray(pts, dtype=float)
    return np.asarray([pts[idx] for idx in sorted(boundary_ids)], dtype=float)


def _axisymmetric_profile_from_surface_points(
    points: np.ndarray,
    *,
    axis: np.ndarray,
    axis_origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    axis = np.asarray(axis, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-30)
    axis_origin = np.asarray(axis_origin, dtype=float)

    s_vals = np.dot(pts - axis_origin[None, :], axis)
    order = np.argsort(s_vals)
    pts = pts[order]
    s_vals = s_vals[order]
    span = float(np.max(s_vals) - np.min(s_vals)) if s_vals.size else 0.0
    tol = max(1.0e-10, 1.0e-6 * max(span, 1.0))

    samples_s: list[float] = []
    samples_r: list[float] = []
    cluster_points: list[np.ndarray] = []
    cluster_s: list[float] = []

    def flush_cluster() -> None:
        if not cluster_points:
            return
        arr = np.asarray(cluster_points, dtype=float)
        rel_axis = arr - axis_origin[None, :]
        axial = np.outer(np.dot(rel_axis, axis), axis)
        radial = rel_axis - axial
        radius = float(np.max(np.linalg.norm(radial, axis=1)))
        samples_s.append(float(np.mean(cluster_s)))
        samples_r.append(radius)

    current_mean = None
    for point, sval in zip(pts, s_vals):
        if current_mean is None or abs(float(sval) - float(current_mean)) <= tol:
            cluster_points.append(point)
            cluster_s.append(float(sval))
            current_mean = float(np.mean(cluster_s))
        else:
            flush_cluster()
            cluster_points = [point]
            cluster_s = [float(sval)]
            current_mean = float(sval)
    flush_cluster()

    samples_s_arr = np.asarray(samples_s, dtype=float)
    samples_r_arr = np.asarray(samples_r, dtype=float)
    if samples_r_arr.size:
        max_radius = float(np.max(samples_r_arr))
        min_physical_radius = max(1.0e-9, 0.05 * max_radius)
        keep = samples_r_arr >= min_physical_radius
        samples_s_arr = samples_s_arr[keep]
        samples_r_arr = samples_r_arr[keep]

    return samples_s_arr, samples_r_arr


def _apply_axisymmetric_profile_samples(
    state: VolumetricPitoisState,
    *,
    sample_s: np.ndarray,
    sample_r: np.ndarray,
) -> None:
    sample_s = np.asarray(sample_s, dtype=float)
    sample_r = np.asarray(sample_r, dtype=float)
    if sample_s.size < 2 or sample_r.size != sample_s.size:
        raise ValueError("Need at least two axisymmetric profile samples to initialize Case 2b from .msh")

    order = np.argsort(sample_s)
    sample_s = sample_s[order]
    sample_r = sample_r[order]

    span = float(sample_s[-1] - sample_s[0])
    if span <= 1.0e-30:
        raise ValueError("Degenerate axisymmetric profile span in .msh")

    sample_frac = (sample_s - sample_s[0]) / span
    target_frac = np.asarray(state.layer_fractions, dtype=float)
    target_s = np.interp(target_frac, sample_frac, sample_s)
    target_r = np.interp(target_frac, sample_frac, sample_r)

    mid, axis, e1, e2, layer_data = _initial_axisymmetric_template(state)
    for layer, axial_s, outer_radius in zip(layer_data, target_s, target_r):
        center_coord = mid + float(axial_s) * axis
        _move(layer["center_vertex"], tuple(center_coord), state.HC, state.bV_caps)
        for ordered_vertices, factor in layer["rings"]:
            ring_radius = max(float(factor) * float(outer_radius), 0.0)
            n_ring = max(1, len(ordered_vertices))
            targets = []
            for i, _v in enumerate(ordered_vertices):
                phi = 2.0 * np.pi * float(i) / float(n_ring)
                target = center_coord + ring_radius * (np.cos(phi) * e1 + np.sin(phi) * e2)
                targets.append(tuple(target))
            _move_vertices_batch(ordered_vertices, targets, state.HC, state.bV_caps)

    _rebuild_cap_on_sphere(
        state,
        which="bottom",
        sphere_center=state.bottom_sphere_center,
        contact_radius=float(target_r[0]),
        axis=axis,
        dt=None,
    )
    _rebuild_cap_on_sphere(
        state,
        which="top",
        sphere_center=state.top_sphere_center,
        contact_radius=float(target_r[-1]),
        axis=axis,
        dt=None,
    )


def _load_last_initialshape_profile_into_state(state: VolumetricPitoisState) -> Path | None:
    msh_path = _latest_initialshape_msh_path()
    if msh_path is None:
        return None

    all_points, tets = _read_gmsh2_tet_mesh(msh_path)
    if tets.size:
        used_node_ids = set(int(idx) for idx in np.asarray(tets, dtype=int).reshape(-1))
        unused_count = int(all_points.shape[0] - len(used_node_ids))
        if unused_count > 0:
            print(
                f"Imported initial .msh has {unused_count} nodes not referenced "
                "by tetrahedra; preserving the file node order and tet list.",
                flush=True,
            )
    volume_vertices = _initialshape_msh_vertex_order(state)
    if all_points.shape != (len(volume_vertices), 3):
        volume_vertices = _volume_export_vertex_order(state)
    if all_points.shape != (len(volume_vertices), 3):
        print(
            "Hard-coded initial .msh node count differs from the generated "
            "separation startup topology; importing its axisymmetric surface "
            f"profile instead ({all_points.shape[0]} .msh nodes, "
            f"{len(volume_vertices)} startup vertices).",
            flush=True,
        )
        imported_volume_m3 = float(_indexed_tet_mesh_volume_m3(all_points, tets))
        points = _boundary_surface_points_from_tets(all_points, tets)
        bottom_center, top_center, axis = _particle_centers_physical(state)
        axis_origin = 0.5 * (bottom_center + top_center)
        sample_s, sample_r = _axisymmetric_profile_from_surface_points(
            points,
            axis=axis,
            axis_origin=axis_origin,
        )
        _apply_axisymmetric_profile_samples(state, sample_s=sample_s, sample_r=sample_r)
        _canonicalize_ring_orders(state)
        _ensure_volume_export_node_ids(state)
        state.volume_export_tets = _volume_tet_indices_for_vertex_order(
            state,
            state.volume_export_vertices,
        )
        _refresh_cap_boundary_sets(state)
        state.target_snapshot_volume_m3 = abs(float(imported_volume_m3))
        state.preserve_imported_initial_msh = True
        state.loaded_profile_initial_msh_path = str(msh_path)
        return msh_path
    topology_matches = _msh_tet_topology_matches_state(state, tets, volume_vertices)
    if not topology_matches:
        print(
            "Hard-coded initial .msh tet topology differs from the generated "
            "startup topology; importing node coordinates and preserving the "
            f".msh tet list from {msh_path}.",
            flush=True,
        )

    _move_vertices_batch(
        volume_vertices,
        [tuple(map(float, point)) for point in np.asarray(all_points, dtype=float)],
        state.HC,
        state.bV_caps,
    )
    state.volume_export_vertices = list(volume_vertices)
    state.volume_export_node_ids = {
        id(vertex): idx for idx, vertex in enumerate(state.volume_export_vertices)
    }
    state.imported_initial_msh_tets = np.asarray(tets, dtype=int).copy()
    state.volume_export_tets = np.asarray(tets, dtype=int).copy()
    state.preserve_imported_initial_msh = True
    state.imported_initial_topology_matches_generated = bool(topology_matches)
    state.loaded_exact_initial_msh_path = str(msh_path)
    _refresh_cap_boundary_sets(state)
    state.target_snapshot_volume_m3 = float(_indexed_tet_mesh_volume_m3(all_points, tets))
    return msh_path


def _volume_export_points_no_freeze(state: VolumetricPitoisState) -> np.ndarray:
    return np.asarray(
        [np.asarray(vertex.x_a[:3], dtype=float) for vertex in state.volume_export_vertices],
        dtype=float,
    )


def _volume_export_volume_no_freeze(state: VolumetricPitoisState) -> float:
    if not state.volume_export_vertices or np.asarray(state.volume_export_tets).size == 0:
        return 0.0
    return _indexed_tet_mesh_volume_m3(
        _volume_export_points_no_freeze(state),
        np.asarray(state.volume_export_tets, dtype=int),
    )


def _fit_imported_initial_msh_to_runtime_condition(
    state: VolumetricPitoisState,
    runtime_config: VolumetricPitoisConfig,
) -> VolumetricPitoisConfig:
    """Fit the imported zero-gap mesh template to the separation start point.

    ``mesh_iter0100.msh`` is generated by the initial-shape case at D/R = 0.
    The separation comparison should start at the first digitized Fig. 5 point.
    This keeps the imported topology/profile, but applies a uniform radial and
    axial fit so that the sphere gap and tet volume match the runtime setup.
    """
    if not bool(getattr(state, "preserve_imported_initial_msh", False)):
        return runtime_config
    if not state.volume_export_vertices:
        return runtime_config

    radius = float(runtime_config.particle_radius)
    target_gap = float(runtime_config.initial_d_over_r) * radius
    target_volume = abs(float(runtime_config.initial_bridge_volume_m3))
    if radius <= 0.0 or target_volume <= 0.0:
        return runtime_config

    bottom_ring = list(state.outer_rings[0])
    top_ring = list(state.outer_rings[-1])
    if not bottom_ring or not top_ring:
        return runtime_config

    bottom_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in bottom_ring], axis=0)
    top_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in top_ring], axis=0)
    axis = top_center - bottom_center
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    mid = 0.5 * (bottom_center + top_center)

    current_span = float(np.dot(top_center - bottom_center, axis))
    current_radius = 0.5 * (_cap_radius(bottom_ring) + _cap_radius(top_ring))
    current_volume = _volume_export_volume_no_freeze(state)
    if current_span <= 1.0e-30 or current_radius <= 1.0e-30 or current_volume <= 0.0:
        return runtime_config

    def residual(radial_scale: float) -> float:
        contact_radius = float(radial_scale) * current_radius
        if contact_radius >= radius:
            return -np.inf
        axial_offset = float(np.sqrt(max(radius * radius - contact_radius * contact_radius, 0.0)))
        sphere_span = target_gap + 2.0 * (radius - axial_offset)
        volume_axial_scale = target_volume / max(current_volume * radial_scale * radial_scale, 1.0e-30)
        return current_span * volume_axial_scale - sphere_span

    max_scale = min(0.999999 * radius / current_radius, 16.0)
    low = min(0.02, 0.5 * max_scale)
    high = max_scale
    res_low = residual(low)
    res_high = residual(high)
    if not (np.isfinite(res_low) and np.isfinite(res_high) and res_low * res_high <= 0.0):
        target_contact_radius = min(float(runtime_config.target_cap_radius), 0.999999 * radius)
        radial_scale = target_contact_radius / current_radius
        axial_offset = float(np.sqrt(max(radius * radius - target_contact_radius * target_contact_radius, 0.0)))
        axial_scale = (target_gap + 2.0 * (radius - axial_offset)) / current_span
    else:
        for _ in range(80):
            mid_scale = 0.5 * (low + high)
            res_mid = residual(mid_scale)
            if not np.isfinite(res_mid):
                high = mid_scale
                continue
            if abs(res_mid) <= 1.0e-15:
                low = high = mid_scale
                break
            if res_low * res_mid <= 0.0:
                high = mid_scale
                res_high = res_mid
            else:
                low = mid_scale
                res_low = res_mid
        radial_scale = 0.5 * (low + high)
        axial_scale = target_volume / max(current_volume * radial_scale * radial_scale, 1.0e-30)

    points = _volume_export_points_no_freeze(state)
    rel = points - mid[None, :]
    axial_coord = np.sum(rel * axis[None, :], axis=1)
    radial_coord = rel - axial_coord[:, None] * axis[None, :]
    fitted_points = mid[None, :] + radial_scale * radial_coord + (axial_scale * axial_coord)[:, None] * axis[None, :]
    _move_vertices_batch(
        state.volume_export_vertices,
        [tuple(map(float, point)) for point in fitted_points],
        state.HC,
        state.bV_caps,
    )

    bottom_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in bottom_ring], axis=0)
    top_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in top_ring], axis=0)
    contact_radius = 0.5 * (_cap_radius(bottom_ring) + _cap_radius(top_ring))
    axial_offset = float(np.sqrt(max(radius * radius - contact_radius * contact_radius, 0.0)))
    state.bottom_sphere_center = bottom_center - axial_offset * axis
    state.top_sphere_center = top_center + axial_offset * axis
    state.initial_gap = float(_gap(state))
    state.radial_scale *= float(radial_scale)
    state.axial_scale *= float(axial_scale)
    state.target_snapshot_volume_m3 = target_volume
    state.target_volume_m3 = target_volume
    state.imported_initial_fit_radial_scale = float(radial_scale)
    state.imported_initial_fit_axial_scale = float(axial_scale)
    state.imported_initial_fit_contact_radius_m = float(contact_radius)
    state.imported_initial_fit_d_over_r = float(_gap(state) / max(radius, 1.0e-30))
    state.imported_initial_fit_volume_m3 = float(_volume_export_volume_no_freeze(state))
    _refresh_cap_boundary_sets(state)
    return replace(
        runtime_config,
        initial_d_over_r=float(state.imported_initial_fit_d_over_r),
        target_cap_radius=float(contact_radius),
    )


def _outer_profile_samples_relative_to_axis(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    bottom_center, top_center, axis = _particle_centers_physical(state)
    origin = 0.5 * (bottom_center + top_center)
    samples_s: list[float] = []
    samples_r: list[float] = []
    for ring in state.outer_rings:
        coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
        if coords.size == 0:
            continue
        center = np.mean(coords, axis=0)
        rel = coords - origin[None, :]
        axial = np.outer(np.dot(rel, axis), axis)
        radial = rel - axial
        samples_s.append(float(np.dot(center - origin, axis)))
        samples_r.append(float(np.mean(np.linalg.norm(radial, axis=1))))
    order = np.argsort(samples_s)
    return np.asarray(samples_s, dtype=float)[order], np.asarray(samples_r, dtype=float)[order]


def _apply_axisymmetric_profile_direct(
    state: VolumetricPitoisState,
    *,
    sample_s: np.ndarray,
    sample_r: np.ndarray,
) -> None:
    sample_s = np.asarray(sample_s, dtype=float)
    sample_r = np.asarray(sample_r, dtype=float)
    if sample_s.size < 2 or sample_r.size != sample_s.size:
        raise ValueError("Need at least two profile samples for runtime remesh")
    order = np.argsort(sample_s)
    sample_s = sample_s[order]
    sample_r = sample_r[order]
    span = float(sample_s[-1] - sample_s[0])
    if span <= 1.0e-30:
        raise ValueError("Degenerate profile span during runtime remesh")

    bottom_center, top_center, axis = _particle_centers_physical(state)
    origin = 0.5 * (bottom_center + top_center)
    e1, e2 = _orthonormal_tangent_basis(axis, np.array([1.0, 0.0, 0.0], dtype=float))
    sample_frac = (sample_s - sample_s[0]) / span
    target_frac = np.asarray(state.layer_fractions, dtype=float)
    target_s = np.interp(target_frac, sample_frac, sample_s)
    target_r = np.interp(target_frac, sample_frac, sample_r)

    for center_vertex, rings, axial_s, outer_radius in zip(
        state.layer_centers,
        state.layer_rings,
        target_s,
        target_r,
    ):
        center_coord = origin + float(axial_s) * axis
        _move(center_vertex, tuple(center_coord), state.HC, state.bV_caps)
        factors = state.cap_ring_factors
        if len(factors) != len(rings):
            factors = tuple(float(i + 1) / float(max(len(rings), 1)) for i in range(len(rings)))
        for ring, factor in zip(rings, factors):
            ring_radius = max(float(factor) * float(outer_radius), 0.0)
            n_ring = max(1, len(ring))
            targets = []
            for i, _v in enumerate(ring):
                phi = 2.0 * np.pi * float(i) / float(n_ring)
                targets.append(tuple(center_coord + ring_radius * (np.cos(phi) * e1 + np.sin(phi) * e2)))
            _move_vertices_batch(ring, targets, state.HC, state.bV_caps)

    _rebuild_cap_on_sphere(
        state,
        which="bottom",
        sphere_center=state.bottom_sphere_center,
        contact_radius=float(target_r[0]),
        axis=axis,
        dt=None,
    )
    _rebuild_cap_on_sphere(
        state,
        which="top",
        sphere_center=state.top_sphere_center,
        contact_radius=float(target_r[-1]),
        axis=axis,
        dt=None,
    )


def _copy_nearest_velocity_field(
    state: VolumetricPitoisState,
    *,
    old_positions: np.ndarray,
    old_velocities: np.ndarray,
) -> None:
    old_positions = np.asarray(old_positions, dtype=float)
    old_velocities = np.asarray(old_velocities, dtype=float)
    if old_positions.size == 0 or old_velocities.shape != old_positions.shape:
        for v in state.HC.V:
            v.u = np.zeros(3, dtype=float)
        return
    for v in state.HC.V:
        pos = np.asarray(v.x_a[:3], dtype=float)
        idx = int(np.argmin(np.sum((old_positions - pos[None, :]) ** 2, axis=1)))
        v.u = np.asarray(old_velocities[idx], dtype=float).copy()


def _structured_state_from_profile(
    config: VolumetricPitoisConfig,
    *,
    sample_s: np.ndarray,
    sample_r: np.ndarray,
    bottom_sphere_center: np.ndarray,
    top_sphere_center: np.ndarray,
    target_volume_m3: float,
    target_snapshot_volume_m3: float,
    elapsed_time_s: float,
    old_positions: np.ndarray,
    old_velocities: np.ndarray,
) -> VolumetricPitoisState:
    (
        HC,
        _bV_vol,
        outer_rings,
        layer_rings,
        surface_edges,
        surface_boundary_indices,
    ) = _build_structured_volumetric_catenoid(
        config.refinement,
        radial_ring_factors_override=_radial_ring_factors_with_outer_refinement(
            config.contact_line_radial_rings,
            config.extra_cl_radial_rings,
            config.contact_line_radial_bias_ratio,
        ),
        axial_ring_stride_override=config.axial_ring_stride,
        neck_extra_axial_layers_override=config.neck_extra_axial_layers,
        contact_extra_axial_layers_override=config.cl_extra_axial_layers,
    )

    z_vals = [float(v.x_a[2]) for v in HC.V]
    z_min = min(z_vals)
    z_max = max(z_vals)
    cap_bottom = [v for v in HC.V if abs(float(v.x_a[2]) - z_min) < 1.0e-12]
    cap_top = [v for v in HC.V if abs(float(v.x_a[2]) - z_max) < 1.0e-12]
    cap_bottom_center = min(cap_bottom, key=lambda v: float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float))))
    cap_top_center = min(cap_top, key=lambda v: float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float))))
    layer_centers = []
    for ring in outer_rings:
        z_ref = float(np.mean([float(v.x_a[2]) for v in ring]))
        same_layer = [v for v in HC.V if abs(float(v.x_a[2]) - z_ref) < 1.0e-12]
        layer_centers.append(
            min(same_layer, key=lambda v: float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float))))
        )
    cap_ring_factors = tuple(
        max(_cap_radius(ring), 0.0) / max(_cap_radius(layer_rings[0][-1]), 1.0e-30)
        for ring in layer_rings[0]
    )

    bottom_plane_center, top_plane_center, axis = _contact_plane_centers_and_axis(SimpleNamespace(outer_rings=outer_rings))
    contact_plane_span = float(np.dot(top_plane_center - bottom_plane_center, axis))
    if contact_plane_span > 1.0e-30:
        layer_fractions = tuple(
            float(
                np.clip(
                    np.dot(
                        np.mean([np.asarray(v.x_a[:3], dtype=float) for v in ring], axis=0)
                        - bottom_plane_center,
                        axis,
                    )
                    / contact_plane_span,
                    0.0,
                    1.0,
                )
            )
            for ring in outer_rings
        )
    else:
        layer_fractions = tuple(np.linspace(0.0, 1.0, len(outer_rings)))

    for v in HC.V:
        v.cap_id = None
        v.phase = 0
        v.p = 0.0
        v.u = np.zeros(3, dtype=float)
    for v in cap_bottom:
        v.cap_id = "bottom"
    for v in cap_top:
        v.cap_id = "top"
    for v in HC.V:
        v.is_interface = bool(getattr(v, "boundary", False) and v.cap_id is None)
        v.interface_phases = frozenset({0, 1}) if v.is_interface else frozenset()

    mps = SimpleNamespace(
        get_gamma_pair=lambda phase_a, phase_b: config.gamma,
        get_mu=lambda phase: config.mu_f,
    )
    state_new = VolumetricPitoisState(
        config=config,
        HC=HC,
        bV_caps=set(),
        cap_bottom=cap_bottom,
        cap_top=cap_top,
        cap_bottom_interior=[],
        cap_top_interior=[],
        bottom_contact_ring=[],
        top_contact_ring=[],
        cap_bottom_center=cap_bottom_center,
        cap_top_center=cap_top_center,
        layer_centers=layer_centers,
        layer_fractions=layer_fractions,
        outer_rings=outer_rings,
        layer_rings=layer_rings,
        surface_edges=surface_edges,
        surface_boundary_indices=surface_boundary_indices,
        mps=mps,
        radial_scale=1.0,
        axial_scale=1.0,
        initial_gap=float(_gap(SimpleNamespace(config=config, bottom_sphere_center=bottom_sphere_center, top_sphere_center=top_sphere_center))),
        cap_ring_factors=cap_ring_factors,
        bottom_sphere_center=np.asarray(bottom_sphere_center, dtype=float).copy(),
        top_sphere_center=np.asarray(top_sphere_center, dtype=float).copy(),
        target_volume_m3=float(target_volume_m3),
        target_snapshot_volume_m3=float(target_snapshot_volume_m3),
        elapsed_time_s=float(elapsed_time_s),
    )
    _refresh_cap_boundary_sets(state_new)
    _apply_axisymmetric_profile_direct(state_new, sample_s=sample_s, sample_r=sample_r)
    _canonicalize_ring_orders(state_new)
    _freeze_surface_topology(state_new)
    _copy_nearest_velocity_field(state_new, old_positions=old_positions, old_velocities=old_velocities)
    _set_cap_velocities(state_new)
    _enforce_no_swirl_velocity_field(state_new)
    _update_duals_and_masses(state_new)
    _update_pressure_scalar(state_new)
    _initialize_mesh_quality_baseline(state_new)
    return state_new


def _runtime_structured_remesh(
    state: VolumetricPitoisState,
    *,
    new_config: VolumetricPitoisConfig,
    reason: str,
) -> None:
    sample_s, sample_r = _outer_profile_samples_relative_to_axis(state)
    old_vertices = list(state.HC.V)
    old_positions = np.array([np.asarray(v.x_a[:3], dtype=float) for v in old_vertices], dtype=float)
    old_velocities = np.array([np.asarray(getattr(v, "u", np.zeros(3, dtype=float))[:3], dtype=float) for v in old_vertices], dtype=float)
    old_runtime_substep_count = int(getattr(state, "runtime_substep_count", 0))
    old_runtime_remesh_count = int(getattr(state, "runtime_remesh_count", 0))
    last_step_dt = float(getattr(state, "last_step_dt", new_config.dt))
    last_bottom_cl_speed = float(getattr(state, "last_bottom_contact_line_speed", 0.0))
    last_top_cl_speed = float(getattr(state, "last_top_contact_line_speed", 0.0))
    last_dt_limit_cl = float(getattr(state, "last_dt_limit_cl", new_config.dt))
    last_dt_limit_capillary = float(getattr(state, "last_dt_limit_capillary", new_config.dt))
    last_dt_limit_mesh = float(getattr(state, "last_dt_limit_mesh", new_config.dt))
    last_dt_limiter = str(getattr(state, "last_dt_limiter", "fixed"))
    force_match_pressure_offset_pa = float(getattr(state, "force_match_pressure_offset_pa", 0.0))
    initial_force_calibration_fixed_axial_mn = float(
        getattr(state, "initial_force_calibration_fixed_axial_mn", 0.0)
    )

    state_new = _structured_state_from_profile(
        new_config,
        sample_s=sample_s,
        sample_r=sample_r,
        bottom_sphere_center=np.asarray(state.bottom_sphere_center, dtype=float),
        top_sphere_center=np.asarray(state.top_sphere_center, dtype=float),
        target_volume_m3=float(state.target_volume_m3),
        target_snapshot_volume_m3=float(state.target_snapshot_volume_m3),
        elapsed_time_s=float(state.elapsed_time_s),
        old_positions=old_positions,
        old_velocities=old_velocities,
    )
    state.__dict__.clear()
    state.__dict__.update(state_new.__dict__)
    state.runtime_substep_count = old_runtime_substep_count
    state.runtime_remesh_count = old_runtime_remesh_count + 1
    state.last_remesh_reason = str(reason)
    state.last_step_dt = last_step_dt
    state.last_bottom_contact_line_speed = last_bottom_cl_speed
    state.last_top_contact_line_speed = last_top_cl_speed
    state.last_dt_limit_cl = last_dt_limit_cl
    state.last_dt_limit_capillary = last_dt_limit_capillary
    state.last_dt_limit_mesh = last_dt_limit_mesh
    state.last_dt_limiter = last_dt_limiter
    state.force_match_pressure_offset_pa = force_match_pressure_offset_pa
    state.initial_force_calibration_fixed_axial_mn = initial_force_calibration_fixed_axial_mn
    _clear_geometry_caches(state)


def _contact_side_edge_fraction(state: VolumetricPitoisState) -> float:
    rings = list(getattr(state, "outer_rings", []))
    if len(rings) < 2:
        return 0.0

    contact_radius = max(
        _cap_radius(rings[0]),
        _cap_radius(rings[-1]),
        1.0e-30,
    )

    def max_edge_fraction(a_ring, b_ring) -> float:
        n_ring = min(len(a_ring), len(b_ring))
        if n_ring <= 0:
            return 0.0
        lengths = []
        for i in range(n_ring):
            pa = np.asarray(a_ring[i].x_a[:3], dtype=float)
            pb = np.asarray(b_ring[i].x_a[:3], dtype=float)
            lengths.append(float(np.linalg.norm(pb - pa)))
        return max(lengths, default=0.0) / contact_radius

    return max(
        max_edge_fraction(rings[0], rings[1]),
        max_edge_fraction(rings[-1], rings[-2]),
    )


def _maybe_runtime_split_merge_remesh(state: VolumetricPitoisState) -> None:
    if not bool(getattr(state.config, "enable_runtime_remesh", False)):
        return
    state.runtime_substep_count = int(getattr(state, "runtime_substep_count", 0)) + 1
    every = max(1, int(getattr(state.config, "runtime_remesh_check_every_steps", 1)))
    if state.runtime_substep_count % every != 0:
        return

    config = state.config
    new_extra_axial = int(config.cl_extra_axial_layers)
    new_extra_radial = int(config.extra_cl_radial_rings)
    reasons: list[str] = []

    fractions = np.asarray(state.layer_fractions, dtype=float)
    if fractions.size >= 2:
        lower_gap = float(fractions[1] - fractions[0])
        upper_gap = float(fractions[-1] - fractions[-2])
        cl_gap = max(lower_gap, upper_gap)
        min_cl_gap = min(lower_gap, upper_gap)
        if cl_gap > float(config.runtime_remesh_cl_gap_max_fraction):
            limit = int(config.runtime_remesh_max_cl_extra_axial_layers)
            if new_extra_axial < limit:
                new_extra_axial = min(limit, new_extra_axial + 2)
                reasons.append(f"split_cl_axial_gap={cl_gap:.4f}")
        elif min_cl_gap < float(config.runtime_remesh_cl_gap_min_fraction):
            limit = int(config.runtime_remesh_min_cl_extra_axial_layers)
            if new_extra_axial > limit:
                new_extra_axial = max(limit, new_extra_axial - 2)
                reasons.append(f"merge_cl_axial_gap={min_cl_gap:.4f}")

    side_edge_fraction = _contact_side_edge_fraction(state)
    if side_edge_fraction > float(config.runtime_remesh_side_edge_max_fraction):
        limit = int(config.runtime_remesh_max_cl_extra_axial_layers)
        if new_extra_axial < limit:
            increment = 4 if side_edge_fraction > 2.0 * float(config.runtime_remesh_side_edge_max_fraction) else 2
            new_extra_axial = min(limit, new_extra_axial + increment)
            reasons.append(f"split_cl_side_edge={side_edge_fraction:.4f}")

    factors = tuple(float(v) for v in getattr(state, "cap_ring_factors", ()))
    if len(factors) >= 2:
        outer_gap = float(factors[-1] - factors[-2])
        if outer_gap > float(config.runtime_remesh_radial_outer_gap_max):
            limit = int(config.runtime_remesh_max_extra_cl_radial_rings)
            if new_extra_radial < limit:
                new_extra_radial = min(limit, new_extra_radial + 1)
                reasons.append(f"split_radial_outer_gap={outer_gap:.4f}")
        elif outer_gap < float(config.runtime_remesh_radial_outer_gap_min):
            limit = int(config.runtime_remesh_min_extra_cl_radial_rings)
            if new_extra_radial > limit:
                new_extra_radial = max(limit, new_extra_radial - 1)
                reasons.append(f"merge_radial_outer_gap={outer_gap:.4f}")

    ok, metrics = _mesh_quality_is_acceptable(state)
    if not ok:
        reasons.append(
            "quality_rebuild"
            f"_q={float(metrics.get('quality_min', 0.0)):.3e}"
            f"_edge={float(metrics.get('edge_min_m', 0.0)):.3e}"
        )

    if not reasons:
        return

    new_config = replace(
        config,
        extra_cl_radial_rings=int(new_extra_radial),
        cl_extra_axial_layers=int(new_extra_axial),
    )
    _runtime_structured_remesh(state, new_config=new_config, reason=";".join(reasons))


def _axisymmetrize_layer_rings(state: VolumetricPitoisState) -> None:
    bottom_center, top_center, axis = _particle_centers_physical(state)
    mid = 0.5 * (bottom_center + top_center)

    ref_ring = state.outer_rings[0]
    ref_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in ref_ring], axis=0)
    reference = np.asarray(ref_ring[0].x_a[:3], dtype=float) - ref_center
    e1, e2 = _orthonormal_tangent_basis(axis, reference)

    n_layers = len(state.layer_rings)
    for k, rings in enumerate(state.layer_rings):
        if k in (0, n_layers - 1):
            continue

        outer_ring = state.outer_rings[k]
        outer_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in outer_ring], axis=0)
        s = float(np.dot(outer_center - mid, axis))
        center_coord = mid + s * axis
        center_vertex = state.layer_centers[k]
        _move(center_vertex, tuple(center_coord), state.HC, state.bV_caps)

        radial_factors = tuple(float(v) for v in getattr(state, "cap_ring_factors", ()))
        if len(radial_factors) != len(rings):
            radial_factors = tuple(
                float(i + 1) / float(max(len(rings), 1))
                for i in range(len(rings))
            )
        outer_radius = _mean_ring_radius(outer_ring, center_coord, axis)

        for ring, factor in zip(rings, radial_factors):
            radius = max(float(factor) * float(outer_radius), 0.0)
            n_ring = max(1, len(ring))
            targets = []
            for i, v in enumerate(ring):
                phi = 2.0 * np.pi * float(i) / float(n_ring)
                target = center_coord + radius * (np.cos(phi) * e1 + np.sin(phi) * e2)
                targets.append(tuple(target))
            _move_vertices_batch(ring, targets, state.HC, state.bV_caps)


def _relax_initial_equilibrium(state: VolumetricPitoisState) -> None:
    if not state.config.enable_initial_relaxation or int(state.config.initial_relax_steps) <= 0:
        return

    original_config = state.config
    relax_config = replace(
        original_config,
        cap_speed=0.0,
        dt=float(original_config.initial_relax_dt),
    )
    state.config = relax_config
    for v in state.HC.V:
        v.u = np.zeros(3, dtype=float)

    for _step in range(int(original_config.initial_relax_steps)):
        _update_duals_and_masses(state)
        _update_pressure_scalar(state)
        symplectic_euler(
            state.HC,
            state.bV_caps,
            _vertex_acceleration,
            dt=float(relax_config.dt),
            n_steps=1,
            dim=3,
            retopologize_fn=False,
            state=state,
        )
        _axisymmetrize_layer_rings(state)
        _update_duals_and_masses(state)
        _update_pressure_scalar(state)
        if _max_free_speed(state) <= float(original_config.initial_relax_tol_speed):
            break

    for v in state.HC.V:
        v.u = np.zeros(3, dtype=float)
    state.config = original_config


def _rebuild_cap_on_sphere(
    state: VolumetricPitoisState,
    *,
    which: str,
    sphere_center: np.ndarray,
    contact_radius: float,
    axis: np.ndarray,
    dt: float | None,
) -> None:
    layer_idx = 0 if which == "bottom" else -1
    sign = +1.0 if which == "bottom" else -1.0
    center_vertex = state.cap_bottom_center if which == "bottom" else state.cap_top_center
    cap_rings = state.layer_rings[layer_idx]
    reference_ring = cap_rings[-1]
    ring_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in reference_ring], axis=0)
    reference = np.asarray(reference_ring[0].x_a[:3], dtype=float) - ring_center
    e1, e2 = _orthonormal_tangent_basis(axis, reference)
    radius = float(state.config.particle_radius)

    old_center = np.asarray(center_vertex.x_a[:3], dtype=float)
    new_center = sphere_center + sign * radius * axis
    _move(center_vertex, tuple(new_center), state.HC, state.bV_caps)
    if dt is not None and dt > 0.0:
        center_vertex.u = (new_center - old_center) / dt
    else:
        center_vertex.u = np.zeros(3, dtype=float)

    for factor, ring in zip(state.cap_ring_factors, cap_rings):
        ring_radius = min(max(float(factor) * contact_radius, 0.0), radius - 1.0e-6)
        axial = np.sqrt(max(radius**2 - ring_radius**2, 0.0))
        targets = []
        for i, v in enumerate(ring):
            phi = 2.0 * np.pi * float(i) / float(len(ring))
            tangential = ring_radius * (np.cos(phi) * e1 + np.sin(phi) * e2)
            target = sphere_center + tangential + sign * axial * axis
            targets.append(tuple(target))
        for v, target in zip(ring, targets):
            old = np.asarray(v.x_a[:3], dtype=float)
            _move(v, target, state.HC, state.bV_caps)
            if dt is not None and dt > 0.0:
                v.u = (target - old) / dt
            else:
                v.u = np.zeros(3, dtype=float)
    _refresh_cap_boundary_sets(state)


def _update_contact_line_by_cox_voinov(
    state: VolumetricPitoisState,
    *,
    dt: float,
    prev_bottom_radius: float | None = None,
    prev_top_radius: float | None = None,
) -> None:
    axis = np.asarray(state.top_sphere_center - state.bottom_sphere_center, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm

    if prev_bottom_radius is None:
        prev_bottom_radius = float(_cap_radius(state.bottom_contact_ring))
    if prev_top_radius is None:
        prev_top_radius = float(_cap_radius(state.top_contact_ring))

    bottom_radius, bottom_speed = _cox_voinov_contact_radius_update(
        state,
        which="bottom",
        sphere_center=np.asarray(state.bottom_sphere_center, dtype=float),
        axis=axis,
        previous_radius=float(prev_bottom_radius),
        dt=float(dt),
    )
    top_radius, top_speed = _cox_voinov_contact_radius_update(
        state,
        which="top",
        sphere_center=np.asarray(state.top_sphere_center, dtype=float),
        axis=axis,
        previous_radius=float(prev_top_radius),
        dt=float(dt),
    )

    _rebuild_cap_on_sphere(
        state,
        which="bottom",
        sphere_center=np.asarray(state.bottom_sphere_center, dtype=float),
        contact_radius=bottom_radius,
        axis=axis,
        dt=dt,
    )
    _rebuild_cap_on_sphere(
        state,
        which="top",
        sphere_center=np.asarray(state.top_sphere_center, dtype=float),
        contact_radius=top_radius,
        axis=axis,
        dt=dt,
    )
    state.last_bottom_contact_line_speed = abs(float(bottom_speed))
    state.last_top_contact_line_speed = abs(float(top_speed))
    _clear_geometry_caches(state)


def _update_moving_contact_line(
    state: VolumetricPitoisState,
    *,
    dt: float | None = None,
    prev_bottom_radius: float | None = None,
    prev_top_radius: float | None = None,
) -> None:
    axis = state.top_sphere_center - state.bottom_sphere_center
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    if prev_bottom_radius is None:
        prev_bottom_radius = float(_cap_radius(state.outer_rings[0]))
    else:
        prev_bottom_radius = float(prev_bottom_radius)
    if prev_top_radius is None:
        prev_top_radius = float(_cap_radius(state.outer_rings[-1]))
    else:
        prev_top_radius = float(prev_top_radius)
    bottom_ring = state.bottom_contact_ring
    top_ring = state.top_contact_ring
    bottom_next_ring = state.outer_rings[1] if len(state.outer_rings) > 1 else []
    top_next_ring = state.outer_rings[-2] if len(state.outer_rings) > 1 else []
    bottom_radius = _estimate_contact_radius(
        state,
        which="bottom",
        sphere_center=state.bottom_sphere_center,
        contact_ring=bottom_ring,
        next_ring=bottom_next_ring,
        axis=axis,
        dt=dt,
        gap_rate=float(state.config.relative_speed),
        prev_radius=prev_bottom_radius,
    )
    top_radius = _estimate_contact_radius(
        state,
        which="top",
        sphere_center=state.top_sphere_center,
        contact_ring=top_ring,
        next_ring=top_next_ring,
        axis=axis,
        dt=dt,
        gap_rate=float(state.config.relative_speed),
        prev_radius=prev_top_radius,
    )

    _rebuild_cap_on_sphere(
        state,
        which="bottom",
        sphere_center=state.bottom_sphere_center,
        contact_radius=bottom_radius,
        axis=axis,
        dt=dt,
    )
    _rebuild_cap_on_sphere(
        state,
        which="top",
        sphere_center=state.top_sphere_center,
        contact_radius=top_radius,
        axis=axis,
        dt=dt,
    )
    _align_layer_azimuths(state)
    _regularize_layer_order(state)
    if dt is not None and dt > 0.0:
        sphere_radius = max(float(state.config.particle_radius), 1.0e-30)
        alpha_bottom_prev = float(np.arcsin(np.clip(prev_bottom_radius / sphere_radius, -1.0, 1.0)))
        alpha_bottom_new = float(np.arcsin(np.clip(bottom_radius / sphere_radius, -1.0, 1.0)))
        alpha_top_prev = float(np.arcsin(np.clip(prev_top_radius / sphere_radius, -1.0, 1.0)))
        alpha_top_new = float(np.arcsin(np.clip(top_radius / sphere_radius, -1.0, 1.0)))
        state.last_bottom_contact_line_speed = abs(sphere_radius * (alpha_bottom_new - alpha_bottom_prev) / float(dt))
        state.last_top_contact_line_speed = abs(sphere_radius * (alpha_top_new - alpha_top_prev) / float(dt))
    else:
        state.last_bottom_contact_line_speed = 0.0
        state.last_top_contact_line_speed = 0.0


def _capture_dynamic_state(state: VolumetricPitoisState) -> dict[str, object]:
    vertices = list(state.HC.V)
    return {
        "vertices": vertices,
        "positions": np.array([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float),
        "velocities": np.array([np.asarray(v.u[:3], dtype=float) for v in vertices], dtype=float),
        "bottom_sphere_center": np.asarray(state.bottom_sphere_center, dtype=float).copy(),
        "top_sphere_center": np.asarray(state.top_sphere_center, dtype=float).copy(),
        "pressure_scalar": float(getattr(state, "pressure_scalar", 0.0)),
        "force_match_pressure_offset_pa": float(getattr(state, "force_match_pressure_offset_pa", 0.0)),
        "initial_force_calibration_fixed_axial_mn": float(
            getattr(state, "initial_force_calibration_fixed_axial_mn", 0.0)
        ),
        "pressure_projection_map": dict(getattr(state, "pressure_projection_map", {})),
        "last_bottom_contact_line_speed": float(getattr(state, "last_bottom_contact_line_speed", 0.0)),
        "last_top_contact_line_speed": float(getattr(state, "last_top_contact_line_speed", 0.0)),
        "last_ns_projection_iterations": int(getattr(state, "last_ns_projection_iterations", 0)),
        "last_ns_divergence_l2": float(getattr(state, "last_ns_divergence_l2", 0.0)),
        "last_ns_projected_divergence_l2": float(getattr(state, "last_ns_projected_divergence_l2", 0.0)),
        "last_ns_pressure_l2": float(getattr(state, "last_ns_pressure_l2", 0.0)),
    }


def _restore_dynamic_state(state: VolumetricPitoisState, snapshot: dict[str, object]) -> None:
    vertices = list(snapshot["vertices"])
    positions = np.asarray(snapshot["positions"], dtype=float)
    _move_vertices_batch(
        vertices,
        [tuple(map(float, row)) for row in positions],
        state.HC,
        state.bV_caps,
    )
    velocities = np.asarray(snapshot["velocities"], dtype=float)
    for v, vel in zip(vertices, velocities):
        v.u = np.asarray(vel, dtype=float).copy()
    state.bottom_sphere_center = np.asarray(snapshot["bottom_sphere_center"], dtype=float).copy()
    state.top_sphere_center = np.asarray(snapshot["top_sphere_center"], dtype=float).copy()
    state.pressure_scalar = float(snapshot["pressure_scalar"])
    state.force_match_pressure_offset_pa = float(snapshot.get("force_match_pressure_offset_pa", 0.0))
    state.initial_force_calibration_fixed_axial_mn = float(
        snapshot.get("initial_force_calibration_fixed_axial_mn", 0.0)
    )
    state.pressure_projection_map = dict(snapshot.get("pressure_projection_map", {}))
    state.last_bottom_contact_line_speed = float(snapshot["last_bottom_contact_line_speed"])
    state.last_top_contact_line_speed = float(snapshot["last_top_contact_line_speed"])
    state.last_ns_projection_iterations = int(snapshot.get("last_ns_projection_iterations", 0))
    state.last_ns_divergence_l2 = float(snapshot.get("last_ns_divergence_l2", 0.0))
    state.last_ns_projected_divergence_l2 = float(snapshot.get("last_ns_projected_divergence_l2", 0.0))
    state.last_ns_pressure_l2 = float(snapshot.get("last_ns_pressure_l2", 0.0))
    _clear_geometry_caches(state)


def _advance_one_substep(state: VolumetricPitoisState, *, dt: float) -> None:
    reference_volume_m3 = float(_snapshot_msh_volume_m3(state))
    _set_cap_velocities(state)
    _enforce_no_swirl_velocity_field(state)
    _enforce_radial_axial_velocity_average_field(state)
    _enforce_continuity_velocity_constraint(state)
    _enforce_radial_axial_velocity_average_field(state)
    _move_caps(state, dt=dt)
    prev_bottom_radius = float(_cap_radius(state.outer_rings[0]))
    prev_top_radius = float(_cap_radius(state.outer_rings[-1]))
    _update_duals_and_masses(state)
    _update_pressure_scalar(state)
    _solve_ns_projection_pressure(state, dt=dt)
    if not _array_advance_symplectic_step(state, dt=dt):
        symplectic_euler(
            state.HC,
            state.bV_caps,
            _vertex_acceleration,
            dt=dt,
            n_steps=1,
            dim=3,
            retopologize_fn=False,
            workers=max(1, int(state.config.accel_workers)),
            state=state,
        )
    _constrain_contact_lines_to_spheres(state)
    _clear_geometry_caches(state)
    _enforce_no_swirl_velocity_field(state)
    _enforce_radial_axial_velocity_average_field(state)
    _enforce_continuity_velocity_constraint(state)
    _enforce_radial_axial_velocity_average_field(state)
    _set_cap_velocities(state)
    _enforce_radial_axial_velocity_average_field(state)
    if _uses_exact_imported_msh(state):
        if bool(getattr(state.config, "use_cox_voinov_contact_line_law", False)):
            _update_contact_line_by_cox_voinov(
                state,
                dt=dt,
                prev_bottom_radius=prev_bottom_radius,
                prev_top_radius=prev_top_radius,
            )
        else:
            _constrain_contact_lines_to_spheres(state)
    else:
        _update_moving_contact_line(
            state,
            dt=dt,
            prev_bottom_radius=prev_bottom_radius,
            prev_top_radius=prev_top_radius,
        )
        _axisymmetrize_layer_rings(state)
        _constrain_contact_lines_to_spheres(state)
    _enforce_no_swirl_velocity_field(state)
    _enforce_radial_axial_velocity_average_field(state)
    _enforce_continuity_velocity_constraint(state)
    _enforce_radial_axial_velocity_average_field(state)
    _assert_fixed_topology(state)
    if not _uses_exact_imported_msh(state):
        _maybe_runtime_split_merge_remesh(state)
    _assert_fixed_topology(state)
    _enforce_no_swirl_velocity_field(state)
    _enforce_radial_axial_velocity_average_field(state)
    _enforce_continuity_velocity_constraint(state)
    _enforce_radial_axial_velocity_average_field(state)
    if bool(getattr(state.config, "enable_final_volume_correction", False)):
        _enforce_step_continuity_displacement(
            state,
            reference_volume_m3=reference_volume_m3,
            dt=dt,
        )
        _clear_geometry_caches(state)
    _enforce_step_volume_guard(state, reference_volume_m3=reference_volume_m3)
    _update_pressure_scalar(state)
    _solve_ns_projection_pressure(state, dt=dt)
    _assert_fixed_topology(state)


def _advance_substep_with_quality_guard(
    state: VolumetricPitoisState,
    *,
    dt: float,
    depth: int = 0,
) -> None:
    snapshot = _capture_dynamic_state(state)
    _advance_one_substep(state, dt=dt)
    ok, metrics = _mesh_quality_is_acceptable(state)
    if ok:
        return

    _restore_dynamic_state(state, snapshot)
    max_retries = max(0, int(getattr(state.config, "mesh_quality_max_retries", 0)))
    min_dt = max(float(getattr(state.config, "adaptive_dt_min_s", 0.0)), 1.0e-8)
    if depth >= max_retries or float(dt) <= min_dt:
        failure_parts = []
        if bool(metrics.get("enforce_quality", False)):
            failure_parts.append(
                f"q_min={metrics['quality_min']:.3e} < limit {metrics['quality_min_limit']:.3e} "
                f"(threshold {metrics['quality_min_threshold']:.3e}, hard {metrics['quality_min_hard_limit']:.3e}, "
                f"abs floor {metrics['quality_min_abs_floor']:.3e})"
            )
        if bool(metrics.get("enforce_edge", False)):
            failure_parts.append(
                f"min_edge={metrics['edge_min_m'] * 1.0e3:.6f} mm < limit "
                f"{metrics['edge_min_limit_m'] * 1.0e3:.6f} mm "
                f"(threshold {metrics['edge_min_threshold_m'] * 1.0e3:.6f} mm, "
                f"hard {metrics['edge_min_hard_limit_m'] * 1.0e3:.6f} mm)"
            )
        if bool(metrics.get("enforce_spacing", False)):
            failure_parts.append(
                f"layer_spacing={metrics['adjacent_layer_spacing_min_m'] * 1.0e3:.6f} mm < limit "
                f"{metrics['adjacent_layer_spacing_limit_m'] * 1.0e3:.6f} mm "
                f"(threshold {metrics['adjacent_layer_spacing_threshold_m'] * 1.0e3:.6f} mm, "
                f"hard {metrics['adjacent_layer_spacing_hard_limit_m'] * 1.0e3:.6f} mm)"
            )
        failure_text = "; ".join(failure_parts) if failure_parts else "unknown mesh-quality criterion"
        raise RuntimeError(
            "Mesh quality guard failed: " + failure_text
        )

    half_dt = 0.5 * float(dt)
    _advance_substep_with_quality_guard(state, dt=half_dt, depth=depth + 1)
    _advance_substep_with_quality_guard(state, dt=half_dt, depth=depth + 1)


def _prepare_state(config: VolumetricPitoisConfig) -> VolumetricPitoisState:
    runtime_config = config
    config = (
        _hardcoded_initial_msh_build_config(runtime_config)
        if bool(runtime_config.use_last_initialshape_msh)
        else runtime_config
    )
    (
        HC,
        _bV_vol,
        outer_rings,
        layer_rings,
        surface_edges,
        surface_boundary_indices,
    ) = _build_structured_volumetric_catenoid(
        config.refinement,
        radial_ring_factors_override=_radial_ring_factors_with_outer_refinement(
            config.contact_line_radial_rings,
            config.extra_cl_radial_rings,
            config.contact_line_radial_bias_ratio,
        ),
        axial_ring_stride_override=config.axial_ring_stride,
        neck_extra_axial_layers_override=config.neck_extra_axial_layers,
        contact_extra_axial_layers_override=config.cl_extra_axial_layers,
    )

    top_ring = outer_rings[-1]
    radial_scale = config.target_cap_radius / max(_cap_radius(top_ring), 1.0e-30)
    contact_radius = radial_scale * max(_cap_radius(top_ring), 1.0e-30)
    axial_offset = float(np.sqrt(max(config.particle_radius**2 - contact_radius**2, 0.0)))

    z_vals_unit = [float(v.x_a[2]) for v in HC.V]
    unit_gap = max(z_vals_unit) - min(z_vals_unit)
    target_surface_gap = config.initial_d_over_r * config.particle_radius
    target_contact_plane_gap = target_surface_gap + 2.0 * (config.particle_radius - axial_offset)
    axial_scale = target_contact_plane_gap / max(unit_gap, 1.0e-30)

    _scale_mesh(HC, radial_scale, axial_scale)

    z_vals = [float(v.x_a[2]) for v in HC.V]
    z_min = min(z_vals)
    z_max = max(z_vals)
    cap_bottom = [v for v in HC.V if abs(float(v.x_a[2]) - z_min) < 1.0e-12]
    cap_top = [v for v in HC.V if abs(float(v.x_a[2]) - z_max) < 1.0e-12]
    cap_bottom_center = min(cap_bottom, key=lambda v: float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float))))
    cap_top_center = min(cap_top, key=lambda v: float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float))))
    layer_centers = []
    for ring in outer_rings:
        z_ref = float(np.mean([float(v.x_a[2]) for v in ring]))
        same_layer = [v for v in HC.V if abs(float(v.x_a[2]) - z_ref) < 1.0e-12]
        layer_centers.append(
            min(same_layer, key=lambda v: float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float))))
        )
    cap_ring_factors = tuple(
        max(_cap_radius(ring), 0.0) / max(_cap_radius(layer_rings[0][-1]), 1.0e-30)
        for ring in layer_rings[0]
    )
    bottom_plane_center, top_plane_center, axis = _contact_plane_centers_and_axis(SimpleNamespace(outer_rings=outer_rings))
    bottom_sphere_center = bottom_plane_center - axial_offset * axis
    top_sphere_center = top_plane_center + axial_offset * axis
    contact_plane_span = float(np.dot(top_plane_center - bottom_plane_center, axis))
    if contact_plane_span > 1.0e-30:
        layer_fractions = tuple(
            float(
                np.clip(
                    np.dot(
                        np.mean([np.asarray(v.x_a[:3], dtype=float) for v in ring], axis=0)
                        - bottom_plane_center,
                        axis,
                    )
                    / contact_plane_span,
                    0.0,
                    1.0,
                )
            )
            for ring in outer_rings
        )
    else:
        layer_fractions = tuple(np.linspace(0.0, 1.0, len(outer_rings)))

    for v in HC.V:
        v.cap_id = None
        v.phase = 0
        v.p = 0.0
        v.u = np.zeros(3, dtype=float)
    for v in cap_bottom:
        v.cap_id = "bottom"
    for v in cap_top:
        v.cap_id = "top"

    for v in HC.V:
        v.is_interface = bool(getattr(v, "boundary", False) and v.cap_id is None)
        v.interface_phases = frozenset({0, 1}) if v.is_interface else frozenset()

    mps = SimpleNamespace(
        get_gamma_pair=lambda phase_a, phase_b: runtime_config.gamma,
        get_mu=lambda phase: runtime_config.mu_f,
    )

    state = VolumetricPitoisState(
        config=config,
        HC=HC,
        bV_caps=set(),
        cap_bottom=cap_bottom,
        cap_top=cap_top,
        cap_bottom_interior=[],
        cap_top_interior=[],
        bottom_contact_ring=[],
        top_contact_ring=[],
        cap_bottom_center=cap_bottom_center,
        cap_top_center=cap_top_center,
        layer_centers=layer_centers,
        layer_fractions=layer_fractions,
        outer_rings=outer_rings,
        layer_rings=layer_rings,
        surface_edges=surface_edges,
        surface_boundary_indices=surface_boundary_indices,
        mps=mps,
        radial_scale=radial_scale,
        axial_scale=axial_scale,
        initial_gap=target_surface_gap,
        cap_ring_factors=cap_ring_factors,
        bottom_sphere_center=np.asarray(bottom_sphere_center, dtype=float),
        top_sphere_center=np.asarray(top_sphere_center, dtype=float),
        target_volume_m3=float(config.initial_bridge_volume_m3),
        target_snapshot_volume_m3=float(config.initial_bridge_volume_m3),
    )
    _refresh_cap_boundary_sets(state)
    loaded_msh_path = None
    imported_target_msh_volume_m3 = None
    _initialise_unduloid_like_equilibrium(state)
    if not bool(runtime_config.use_last_initialshape_msh):
        _relax_initial_equilibrium(state)
    _canonicalize_ring_orders(state)
    if bool(runtime_config.use_last_initialshape_msh):
        loaded_msh_path = _load_last_initialshape_profile_into_state(state)
        if loaded_msh_path is None:
            raise RuntimeError(f"Hard-coded initial .msh not found: {HARDCODED_INITIAL_MSH_PATH}")
        runtime_config = _fit_imported_initial_msh_to_runtime_condition(state, runtime_config)
        imported_target_msh_volume_m3 = float(state.target_snapshot_volume_m3)
        print(f"Loaded initial surface    = {loaded_msh_path}")
    state.config = runtime_config
    _set_cap_velocities(state)
    _freeze_surface_topology(state)
    _project_contact_lines_to_spheres(state)
    _update_duals_and_masses(state)
    _update_pressure_scalar(state)
    initial_projection_dt = float(runtime_config.dt if runtime_config.dt > 0.0 else runtime_config.adaptive_dt_max_s)
    if imported_target_msh_volume_m3 is not None:
        state.target_snapshot_volume_m3 = float(imported_target_msh_volume_m3)
        state.target_volume_m3 = float(_snapshot_msh_volume_m3(state))
    else:
        state.target_volume_m3 = float(_snapshot_msh_volume_m3(state))
        state.target_snapshot_volume_m3 = float(_snapshot_msh_volume_m3(state))
    _calibrate_initial_force_to_pitois_first_point(state, dt=max(initial_projection_dt, 1.0e-12))
    _initialize_mesh_quality_baseline(state)
    if loaded_msh_path is not None:
        print(f"Imported source .msh vol. = {1.0e9 * imported_target_msh_volume_m3:.6f} uL")
        print(
            "Fitted initial state      = "
            f"D/R {float(getattr(state, 'imported_initial_fit_d_over_r', _gap(state) / state.config.particle_radius)):.12f}, "
            f"r_CL {1.0e3 * float(getattr(state, 'imported_initial_fit_contact_radius_m', state.config.target_cap_radius)):.6f} mm, "
            f"vol {1.0e9 * float(getattr(state, 'imported_initial_fit_volume_m3', state.target_snapshot_volume_m3)):.6f} uL",
            flush=True,
        )
    if bool(getattr(state.config, "match_initial_force_to_fig5", False)):
        print(
            "Initial force calibration = "
            f"target |F| {float(PITOIS_FIG5_FIRST_FORCE_MN):.6f} mN, "
            f"fixed_axial {float(state.initial_force_calibration_fixed_axial_mn):+.6f} mN, "
            f"pressure_offset {float(state.force_match_pressure_offset_pa):+.6e} Pa",
            flush=True,
        )
    return state


def _move_caps(state: VolumetricPitoisState, *, dt: float | None = None) -> None:
    if dt is None:
        dt = float(state.config.dt)
    bottom_vel = np.array([0.0, 0.0, -float(state.config.cap_speed)], dtype=float)
    top_vel = np.zeros(3, dtype=float)
    bottom_disp = dt * bottom_vel
    top_disp = dt * top_vel
    if float(np.linalg.norm(bottom_disp)) <= 1.0e-30 and float(np.linalg.norm(top_disp)) <= 1.0e-30:
        return
    state.bottom_sphere_center = np.asarray(state.bottom_sphere_center, dtype=float) + bottom_disp
    state.top_sphere_center = np.asarray(state.top_sphere_center, dtype=float) + top_disp
    if float(np.linalg.norm(bottom_disp)) > 1.0e-30:
        for v in state.cap_bottom_interior:
            old = np.asarray(v.x_a[:3], dtype=float)
            target = old + bottom_disp
            _move(v, tuple(target), state.HC, state.bV_caps)
            v.u = bottom_vel
        for v in state.bottom_contact_ring:
            old = np.asarray(v.x_a[:3], dtype=float)
            target = old + bottom_disp
            _move(v, tuple(target), state.HC, state.bV_caps)
            v.u = bottom_vel
    if float(np.linalg.norm(top_disp)) > 1.0e-30:
        for v in state.cap_top_interior:
            old = np.asarray(v.x_a[:3], dtype=float)
            target = old + top_disp
            _move(v, tuple(target), state.HC, state.bV_caps)
            v.u = top_vel
        for v in state.top_contact_ring:
            old = np.asarray(v.x_a[:3], dtype=float)
            target = old + top_disp
            _move(v, tuple(target), state.HC, state.bV_caps)
            v.u = top_vel


def _interface_surface_tension_force(v, *, state: VolumetricPitoisState) -> np.ndarray:
    cache_key = id(v)
    cached = state.interface_surface_tension_cache.get(cache_key)
    if cached is not None:
        return cached

    if not getattr(v, "is_interface", False):
        force = np.zeros(3, dtype=float)
        state.interface_surface_tension_cache[cache_key] = force
        return force

    interface_nbs = {nb for nb in v.nn if getattr(nb, "is_interface", False)}
    if len(interface_nbs) < 2:
        force = np.zeros(3, dtype=float)
        state.interface_surface_tension_cache[cache_key] = force
        return force

    phases = getattr(v, "interface_phases", frozenset())
    if len(phases) < 2:
        force = np.zeros(3, dtype=float)
        state.interface_surface_tension_cache[cache_key] = force
        return force

    phase_list = sorted(phases)
    gamma = float(state.mps.get_gamma_pair(phase_list[0], phase_list[1]))
    if abs(gamma) <= 1.0e-30:
        force = np.zeros(3, dtype=float)
        state.interface_surface_tension_cache[cache_key] = force
        return force

    HNdA, _area = hndA_i_interface(v, interface_nbs | {v})
    force = -gamma * np.asarray(HNdA[:3], dtype=float)
    state.interface_surface_tension_cache[cache_key] = force
    return force


def _cauchy_stress_tensor_at_vertex(
    v,
    *,
    state: VolumetricPitoisState,
    pressure_model,
) -> np.ndarray:
    cache_key = id(v)
    cached = state.cauchy_stress_cache.get(cache_key)
    if cached is not None:
        return cached

    p_v = float(pressure_model(v, HC=state.HC, dim=3))
    du_v = velocity_difference_tensor_pointwise(v, state.HC, dim=3)
    sigma_v = cauchy_stress(p_v, du_v, float(state.config.mu_f), dim=3)
    state.cauchy_stress_cache[cache_key] = sigma_v
    return sigma_v


def _cauchy_stress_dual_face_force(
    v,
    *,
    state: VolumetricPitoisState,
    pressure_model,
) -> np.ndarray:
    sigma_i = _cauchy_stress_tensor_at_vertex(v, state=state, pressure_model=pressure_model)

    force = np.zeros(3, dtype=float)
    for v_j in v.nn:
        try:
            area_vec = _dual_area_vector_cached(v, v_j, state)
        except (KeyError, IndexError, ValueError, RuntimeError, ZeroDivisionError):
            continue
        area_vec = np.asarray(area_vec, dtype=float)
        if float(np.linalg.norm(area_vec)) <= 1.0e-30:
            continue

        sigma_j = _cauchy_stress_tensor_at_vertex(v_j, state=state, pressure_model=pressure_model)
        sigma_face = 0.5 * (sigma_i + sigma_j)
        force += sigma_face @ area_vec
    return force


def _Ftot(
    v,
    *,
    state: VolumetricPitoisState,
    pressure_model=None,
    include_contact_line: bool = True,
) -> np.ndarray:
    if pressure_model is None:
        pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)

    # Ftot = integral(sigma n dA) + F_gamma + Fcl
    Ftot = _cauchy_stress_dual_face_force(
        v,
        state=state,
        pressure_model=pressure_model,
    )
    Ftot += _interface_surface_tension_force(v, state=state)
    if include_contact_line:
        Fcl = _Fcl(v, state=state)
        Ftot += Fcl
    if bool(getattr(state.config, "enforce_no_swirl", False)):
        axis_origin, axis = _swirl_axis_geometry(state)
        Ftot = _remove_swirl_component(
            Ftot,
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=axis_origin,
            axis=axis,
        )
    return np.asarray(Ftot, dtype=float)


def _vertex_acceleration(v, *, state: VolumetricPitoisState) -> np.ndarray:
    Ftot = _Ftot(v, state=state)
    accel = Ftot / _vertex_mass_kg(v, state)
    accel = _clip_acceleration(accel, state)
    if bool(getattr(state.config, "enforce_no_swirl", False)):
        axis_origin, axis = _swirl_axis_geometry(state)
        accel = _remove_swirl_component(
            accel,
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=axis_origin,
            axis=axis,
        )
    return accel


def _cap_traction_force(
    state: VolumetricPitoisState,
    *,
    which: str,
) -> np.ndarray:
    if which == "bottom":
        tri_idx = np.asarray(state.surface_export_bottom_cap_tris, dtype=int)
        sphere_center = np.asarray(state.bottom_sphere_center, dtype=float)
    else:
        tri_idx = np.asarray(state.surface_export_top_cap_tris, dtype=int)
        sphere_center = np.asarray(state.top_sphere_center, dtype=float)
    if tri_idx.size == 0:
        return np.zeros(3, dtype=float)

    vertices = state.surface_export_vertices
    Fcap = np.zeros(3, dtype=float)
    mu = float(state.config.mu_f)
    report_pressure_offset = float(getattr(state, "force_match_pressure_offset_pa", 0.0))
    include_viscous = bool(getattr(state.config, "report_cap_viscous_stress", False))

    def pressure_model(vv, HC=None, dim=3):
        return _pressure_model(vv, HC=HC, dim=dim, state=state) + report_pressure_offset

    for a, b, c in tri_idx:
        va = vertices[int(a)]
        vb = vertices[int(b)]
        vc = vertices[int(c)]
        pa = np.asarray(va.x_a[:3], dtype=float)
        pb = np.asarray(vb.x_a[:3], dtype=float)
        pc = np.asarray(vc.x_a[:3], dtype=float)
        area_vec = 0.5 * np.cross(pb - pa, pc - pa)
        area = float(np.linalg.norm(area_vec))
        if area <= 1.0e-30:
            continue

        centroid = (pa + pb + pc) / 3.0
        n_s = centroid - sphere_center
        n_norm = float(np.linalg.norm(n_s))
        if n_norm <= 1.0e-30:
            continue
        n_s = n_s / n_norm

        pressure_tri = np.zeros((3, 3), dtype=float)
        viscous_tri = np.zeros((3, 3), dtype=float)
        for vtx in (va, vb, vc):
            p_v = float(pressure_model(vtx))
            pressure_tri -= p_v * np.eye(3, dtype=float)
            if include_viscous:
                du_v = velocity_difference_tensor_pointwise(vtx, state.HC, dim=3)
                viscous_tri += mu * (du_v + du_v.T)
        sigma_tri = pressure_tri / 3.0
        if include_viscous:
            sigma_tri += viscous_tri / 3.0
        Fcap += sigma_tri @ n_s * area

    return Fcap


def _ordered_outer_rings_for_surface(state: VolumetricPitoisState) -> list[list[object]]:
    # Keep the side-surface connectivity inherited from the initial mesh.
    # Re-sorting rings by current angle changes which same-ID vertices are
    # connected across layers, even when the intended topology is unchanged.
    return [list(ring) for ring in state.outer_rings]


def _structured_surface_topology(ordered_rings):
    if not ordered_rings:
        return set(), np.empty((0, 3), dtype=int), set()

    n_layers = len(ordered_rings)
    n_ring = len(ordered_rings[0])

    def flat_idx(k: int, i: int) -> int:
        return k * n_ring + i

    edges: set[tuple[int, int]] = set()
    triangles: list[tuple[int, int, int]] = []

    ring_coords = [
        np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
        for ring in ordered_rings
    ]

    for k in range(n_layers):
        for i in range(n_ring):
            a = flat_idx(k, i)
            b = flat_idx(k, (i + 1) % n_ring)
            edges.add(tuple(sorted((a, b))))

    for k in range(n_layers - 1):
        for i in range(n_ring):
            a = flat_idx(k, i)
            b = flat_idx(k, (i + 1) % n_ring)
            c = flat_idx(k + 1, (i + 1) % n_ring)
            d = flat_idx(k + 1, i)
            edges.add(tuple(sorted((a, d))))
            edges.add(tuple(sorted((b, c))))
            pa = ring_coords[k][i]
            pb = ring_coords[k][(i + 1) % n_ring]
            pc = ring_coords[k + 1][(i + 1) % n_ring]
            pd = ring_coords[k + 1][i]
            if _quad_uses_ac_diagonal(pa, pb, pc, pd):
                edges.add(tuple(sorted((a, c))))
                triangles.append((a, b, c))
                triangles.append((a, c, d))
            else:
                edges.add(tuple(sorted((b, d))))
                triangles.append((a, b, d))
                triangles.append((b, c, d))

    boundary_indices = {flat_idx(0, i) for i in range(n_ring)}
    boundary_indices.update({flat_idx(n_layers - 1, i) for i in range(n_ring)})
    return edges, np.array(triangles, dtype=int), boundary_indices


def _structured_surface_display_edges(ordered_rings):
    if not ordered_rings:
        return set()

    n_layers = len(ordered_rings)
    n_ring = len(ordered_rings[0])

    def flat_idx(k: int, i: int) -> int:
        return k * n_ring + i

    edges: set[tuple[int, int]] = set()
    for k in range(n_layers):
        for i in range(n_ring):
            a = flat_idx(k, i)
            b = flat_idx(k, (i + 1) % n_ring)
            edges.add(tuple(sorted((a, b))))
    for k in range(n_layers - 1):
        for i in range(n_ring):
            a = flat_idx(k, i)
            d = flat_idx(k + 1, i)
            edges.add(tuple(sorted((a, d))))
    return edges


def _build_structured_side_surface_complex(ordered_rings):
    surface = Complex(3, domain=None)
    flat_vertices = [
        surface.V[tuple(map(float, v.x_a))]
        for ring in ordered_rings
        for v in ring
    ]
    edges, triangles, boundary_indices = _structured_surface_topology(ordered_rings)
    for i, j in edges:
        flat_vertices[i].connect(flat_vertices[j])
    surface_bV = {flat_vertices[idx] for idx in boundary_indices}
    return surface, surface_bV, flat_vertices, triangles


def _quad_uses_ac_diagonal(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> bool:
    return float(np.linalg.norm(a - c)) <= float(np.linalg.norm(b - d))


def _quad_triangles_consistent_diagonal(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if _quad_uses_ac_diagonal(a, b, c, d):
        return (
            np.array([a, b, c], dtype=float),
            np.array([a, c, d], dtype=float),
        )
    return (
        np.array([a, b, d], dtype=float),
        np.array([b, c, d], dtype=float),
    )


def _actual_compute_side_surface_triangles(state: VolumetricPitoisState) -> np.ndarray:
    if state.surface_export_side_tris.size:
        return _surface_triangles_xyz_from_indices(state, state.surface_export_side_tris)
    ordered_rings = _ordered_outer_rings_for_surface(state)
    if len(ordered_rings) < 2:
        return np.empty((0, 3, 3), dtype=float)
    triangles: list[np.ndarray] = []
    for lower_ring, upper_ring in zip(ordered_rings[:-1], ordered_rings[1:]):
        n_ring = min(len(lower_ring), len(upper_ring))
        if n_ring < 2:
            continue
        lower = [np.asarray(v.x_a[:3], dtype=float) for v in lower_ring]
        upper = [np.asarray(v.x_a[:3], dtype=float) for v in upper_ring]
        for i in range(n_ring):
            i_next = (i + 1) % n_ring
            tri0, tri1 = _quad_triangles_consistent_diagonal(
                lower[i],
                lower[i_next],
                upper[i_next],
                upper[i],
            )
            triangles.append(tri0)
            triangles.append(tri1)
    return np.asarray(triangles, dtype=float) if triangles else np.empty((0, 3, 3), dtype=float)


def _actual_compute_cap_triangles(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    if state.surface_export_bottom_cap_tris.size or state.surface_export_top_cap_tris.size:
        return (
            _surface_triangles_xyz_from_indices(state, state.surface_export_bottom_cap_tris),
            _surface_triangles_xyz_from_indices(state, state.surface_export_top_cap_tris),
        )
    cap_triangle_sets: list[np.ndarray] = []
    for rings, center_vertex in (
        (state.layer_rings[0], state.cap_bottom_center),
        (state.layer_rings[-1], state.cap_top_center),
    ):
        triangles: list[np.ndarray] = []
        ring_xyz = [
            [np.asarray(v.x_a[:3], dtype=float) for v in ring]
            for ring in rings
        ]
        if not ring_xyz:
            cap_triangle_sets.append(np.empty((0, 3, 3), dtype=float))
            continue

        center = np.asarray(center_vertex.x_a[:3], dtype=float)
        n_ring = len(ring_xyz[0])
        for i in range(n_ring):
            triangles.append(
                np.array(
                    [center, ring_xyz[0][i], ring_xyz[0][(i + 1) % n_ring]],
                    dtype=float,
                )
            )
        for band in range(len(ring_xyz) - 1):
            inner = ring_xyz[band]
            outer = ring_xyz[band + 1]
            for i in range(n_ring):
                i_next = (i + 1) % n_ring
                tri0, tri1 = _quad_triangles_consistent_diagonal(
                    inner[i],
                    inner[i_next],
                    outer[i_next],
                    outer[i],
                )
                triangles.append(tri0)
                triangles.append(tri1)
        cap_triangle_sets.append(np.asarray(triangles, dtype=float))

    while len(cap_triangle_sets) < 2:
        cap_triangle_sets.append(np.empty((0, 3, 3), dtype=float))
    return cap_triangle_sets[0], cap_triangle_sets[1]


def _meridian_strip_indices(n_ring: int) -> tuple[int, int]:
    if n_ring <= 0:
        return 0, 0
    i0 = 0
    i1 = n_ring // 2
    return i0, i1


def _actual_compute_meridian_strip_triangles(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered_rings = _ordered_outer_rings_for_surface(state)
    if len(ordered_rings) < 2:
        empty = np.empty((0, 3, 3), dtype=float)
        return empty, empty, empty

    n_ring = len(ordered_rings[0])
    sectors = _meridian_strip_indices(n_ring)

    side_tris: list[np.ndarray] = []
    for lower_ring, upper_ring in zip(ordered_rings[:-1], ordered_rings[1:]):
        lower = [np.asarray(v.x_a[:3], dtype=float) for v in lower_ring]
        upper = [np.asarray(v.x_a[:3], dtype=float) for v in upper_ring]
        for i in sectors:
            i_next = (i + 1) % n_ring
            side_tris.append(np.array([lower[i], lower[i_next], upper[i_next]], dtype=float))
            side_tris.append(np.array([lower[i], upper[i_next], upper[i]], dtype=float))

    cap_sets: list[np.ndarray] = []
    for rings, center_vertex in (
        (state.layer_rings[0], state.cap_bottom_center),
        (state.layer_rings[-1], state.cap_top_center),
    ):
        tris: list[np.ndarray] = []
        ring_xyz = [[np.asarray(v.x_a[:3], dtype=float) for v in ring] for ring in rings]
        if ring_xyz:
            center = np.asarray(center_vertex.x_a[:3], dtype=float)
            n_ring = len(ring_xyz[0])
            sectors = _meridian_strip_indices(n_ring)
            for i in sectors:
                i_next = (i + 1) % n_ring
                tris.append(np.array([center, ring_xyz[0][i], ring_xyz[0][i_next]], dtype=float))
            for band in range(len(ring_xyz) - 1):
                inner = ring_xyz[band]
                outer = ring_xyz[band + 1]
                for i in sectors:
                    i_next = (i + 1) % n_ring
                    tri0, tri1 = _quad_triangles_consistent_diagonal(
                        inner[i],
                        inner[i_next],
                        outer[i_next],
                        outer[i],
                    )
                    tris.append(tri0)
                    tris.append(tri1)
        cap_sets.append(np.asarray(tris, dtype=float) if tris else np.empty((0, 3, 3), dtype=float))

    while len(cap_sets) < 2:
        cap_sets.append(np.empty((0, 3, 3), dtype=float))

    return (
        np.asarray(side_tris, dtype=float) if side_tris else np.empty((0, 3, 3), dtype=float),
        cap_sets[0],
        cap_sets[1],
    )


def _meridian_section_wireframe(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    ordered_rings = _ordered_outer_rings_for_surface(state)
    if len(ordered_rings) < 2:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    n_ring = len(ordered_rings[0])
    sectors = _meridian_strip_indices(n_ring)

    segments: list[np.ndarray] = []
    vertices: list[np.ndarray] = []
    for i in sectors:
        chain = [np.asarray(ring[i % len(ring)].x_a[:3], dtype=float) for ring in ordered_rings]
        vertices.extend(chain)
        for a, b in zip(chain[:-1], chain[1:]):
            segments.append(np.array([a, b], dtype=float))

    if not segments:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    rounded = np.round(np.asarray(vertices, dtype=float), decimals=12)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_vertices = np.asarray(vertices, dtype=float)[np.sort(unique_idx)]
    return np.asarray(segments, dtype=float), unique_vertices


def _contact_line_surface_tension_force(
    state: VolumetricPitoisState,
    *,
    which: str,
) -> np.ndarray:
    if which == "bottom":
        ring = list(state.outer_rings[0])
        direction_sign = 1.0
    elif which == "top":
        ring = list(state.outer_rings[-1])
        direction_sign = -1.0
    else:
        return np.zeros(3, dtype=float)

    if not ring:
        return np.zeros(3, dtype=float)

    _, _, axis = _particle_centers_physical(state)
    radius = float(state.config.particle_radius)
    contact_radius = min(float(_cap_radius(ring)), radius)
    if contact_radius <= 0.0 or radius <= 0.0:
        return np.zeros(3, dtype=float)

    filling_angle = float(np.arcsin(np.clip(contact_radius / radius, -1.0, 1.0)))
    contact_angle = float(np.deg2rad(state.config.contact_angle_deg))
    force_mag = (
        2.0
        * np.pi
        * contact_radius
        * float(state.config.gamma)
        * np.sin(contact_angle + filling_angle)
    )
    return direction_sign * force_mag * axis


def _sphere_total_forces(state: VolumetricPitoisState) -> dict[str, np.ndarray]:
    top_cap = _cap_traction_force(state, which="top")
    bottom_cap = _cap_traction_force(state, which="bottom")
    top_line = _contact_line_surface_tension_force(state, which="top")
    bottom_line = _contact_line_surface_tension_force(state, which="bottom")
    return {
        "top_total": top_cap + top_line,
        "bottom_total": bottom_cap + bottom_line,
        "top_cap": top_cap,
        "bottom_cap": bottom_cap,
        "top_line": top_line,
        "bottom_line": bottom_line,
    }


def _fixed_sphere_axial_force_n(state: VolumetricPitoisState) -> float:
    return float(_sphere_total_forces(state)["top_total"][2])


def _fixed_force_pressure_sensitivity_n_per_pa(state: VolumetricPitoisState) -> float:
    old_offset = float(getattr(state, "force_match_pressure_offset_pa", 0.0))
    current = _fixed_sphere_axial_force_n(state)
    state.force_match_pressure_offset_pa = old_offset + 1.0
    _clear_force_caches(state)
    shifted = _fixed_sphere_axial_force_n(state)
    state.force_match_pressure_offset_pa = old_offset
    _clear_force_caches(state)
    return float(shifted - current)


def _calibrate_initial_force_to_pitois_first_point(state: VolumetricPitoisState, *, dt: float) -> None:
    state.initial_force_calibration_target_mn = float(PITOIS_FIG5_FIRST_FORCE_MN)
    if not bool(getattr(state.config, "match_initial_force_to_fig5", False)):
        state.initial_force_calibration_fixed_axial_mn = 1.0e3 * _fixed_sphere_axial_force_n(state)
        return

    target_mag_n = 1.0e-3 * float(PITOIS_FIG5_FIRST_FORCE_MN)
    max_iters = max(1, int(getattr(state.config, "initial_force_calibration_max_iters", 1)))
    dt = max(float(dt), 1.0e-12)

    for _iteration in range(max_iters):
        _solve_ns_projection_pressure(state, dt=dt)
        current_force_n = _fixed_sphere_axial_force_n(state)
        target_force_n = target_mag_n
        error_n = target_force_n - current_force_n
        if abs(error_n) <= 1.0e-9 * max(target_mag_n, 1.0e-30):
            break

        sensitivity = _fixed_force_pressure_sensitivity_n_per_pa(state)
        if abs(sensitivity) <= 1.0e-30 or not np.isfinite(sensitivity):
            break
        delta_p = error_n / sensitivity
        if not np.isfinite(delta_p):
            break
        state.force_match_pressure_offset_pa += float(delta_p)
        _clear_force_caches(state)

    _solve_ns_projection_pressure(state, dt=dt)
    state.initial_force_calibration_fixed_axial_mn = 1.0e3 * _fixed_sphere_axial_force_n(state)


def _gap(state: VolumetricPitoisState) -> float:
    bottom_center, top_center, axis = _particle_centers_physical(state)
    center_distance = float(np.dot(top_center - bottom_center, axis))
    return center_distance - 2.0 * float(state.config.particle_radius)


def _max_free_speed(state: VolumetricPitoisState) -> float:
    return max(
        (
            float(np.linalg.norm(np.asarray(v.u[:3], dtype=float)))
            for v in state.HC.V
            if v not in state.bV_caps
        ),
        default=0.0,
    )


def _min_mesh_edge_length(state: VolumetricPitoisState) -> float:
    min_len = np.inf
    seen: set[tuple[int, int]] = set()
    for v in state.HC.V:
        x_v = np.asarray(v.x_a[:3], dtype=float)
        for nbr in getattr(v, "nn", []):
            key = tuple(sorted((id(v), id(nbr))))
            if key in seen:
                continue
            seen.add(key)
            edge_len = float(np.linalg.norm(x_v - np.asarray(nbr.x_a[:3], dtype=float)))
            if np.isfinite(edge_len) and edge_len > 1.0e-12 and edge_len < min_len:
                min_len = edge_len
    if not np.isfinite(min_len):
        return 0.0
    return float(min_len)


def _select_physical_dt(state: VolumetricPitoisState) -> float:
    config = state.config
    if float(config.dt) > 0.0:
        dt_fixed = float(config.dt)
        state.last_dt_limit_cl = dt_fixed
        state.last_dt_limit_capillary = dt_fixed
        state.last_dt_limit_mesh = dt_fixed
        state.last_dt_limiter = "fixed_user_dt"
        state.last_step_dt = dt_fixed
        return dt_fixed

    if not bool(getattr(config, "enable_adaptive_dt", False)):
        dt_fallback = max(float(getattr(config, "adaptive_dt_max_s", 0.0)), 1.0e-12)
        state.last_dt_limit_cl = dt_fallback
        state.last_dt_limit_capillary = dt_fallback
        state.last_dt_limit_mesh = dt_fallback
        state.last_dt_limiter = "adaptive_disabled_fallback"
        state.last_step_dt = dt_fallback
        return dt_fallback

    dt_min = max(float(getattr(config, "adaptive_dt_min_s", 0.0)), 1.0e-8)
    dt_max = max(float(getattr(config, "adaptive_dt_max_s", 0.0)), dt_min)
    slide_limit = max(float(config.contact_line_max_slide_um) * 1.0e-6, 1.0e-12)
    cl_speed_ref = max(
        abs(float(state.last_bottom_contact_line_speed)),
        abs(float(state.last_top_contact_line_speed)),
        abs(float(config.relative_speed)),
        1.0e-12,
    )
    dt_cl = slide_limit / cl_speed_ref

    min_edge = max(_min_mesh_edge_length(state), 1.0e-12)
    gamma = max(float(config.gamma), 1.0e-30)
    rho = max(float(config.rho_f), 1.0e-30)
    dt_capillary = max(float(config.adaptive_dt_capillary_safety), 1.0e-8) * np.sqrt(rho * min_edge**3 / gamma)

    speed_ref = max(
        _max_free_speed(state),
        cl_speed_ref,
        abs(float(config.relative_speed)),
        1.0e-12,
    )
    dt_mesh = max(float(config.adaptive_dt_mesh_displacement_frac), 1.0e-6) * min_edge / speed_ref

    state.last_dt_limit_cl = float(dt_cl)
    state.last_dt_limit_capillary = float(dt_capillary)
    state.last_dt_limit_mesh = float(dt_mesh)

    limiter_map = {
        "contact_line": float(dt_cl),
        "capillary": float(dt_capillary),
        "mesh": float(dt_mesh),
        "dt_ceiling": float(dt_max),
    }
    dt_selected = min(limiter_map.values())
    limiter = min(limiter_map, key=limiter_map.get)
    if dt_selected < dt_min:
        dt_selected = dt_min
        limiter = f"{limiter}+dt_floor"

    state.last_dt_limiter = limiter
    state.last_step_dt = float(dt_selected)
    return float(dt_selected)


def _step_record(state: VolumetricPitoisState, *, step: int, t: float) -> dict[str, float | int]:
    sphere_forces = _sphere_total_forces(state)
    top_force = sphere_forces["top_total"]
    bottom_force = sphere_forces["bottom_total"]
    fixed_force = top_force
    moving_force = bottom_force
    gap = _gap(state)
    snapshot_msh_volume_m3 = _snapshot_msh_volume_m3(state)
    return {
        "step": int(step),
        "t": float(t),
        "gap": gap,
        "d_over_r": gap / max(state.config.particle_radius, 1.0e-30),
        "top_force": top_force.tolist(),
        "bottom_force": bottom_force.tolist(),
        "top_force_axial": float(top_force[2]),
        "bottom_force_axial": float(bottom_force[2]),
        "top_force_mag": float(np.linalg.norm(top_force)),
        "bottom_force_mag": float(np.linalg.norm(bottom_force)),
        "fixed_force": fixed_force.tolist(),
        "moving_force": moving_force.tolist(),
        "fixed_force_axial": float(fixed_force[2]),
        "moving_force_axial": float(moving_force[2]),
        "fixed_force_mag": float(np.linalg.norm(fixed_force)),
        "moving_force_mag": float(np.linalg.norm(moving_force)),
        "top_cap_traction": sphere_forces["top_cap"].tolist(),
        "bottom_cap_traction": sphere_forces["bottom_cap"].tolist(),
        "top_contact_line_force": sphere_forces["top_line"].tolist(),
        "bottom_contact_line_force": sphere_forces["bottom_line"].tolist(),
        "top_cap_traction_axial": float(sphere_forces["top_cap"][2]),
        "bottom_cap_traction_axial": float(sphere_forces["bottom_cap"][2]),
        "top_contact_line_axial": float(sphere_forces["top_line"][2]),
        "bottom_contact_line_axial": float(sphere_forces["bottom_line"][2]),
        "bottom_contact_radius": float(_cap_radius(state.outer_rings[0])),
        "top_contact_radius": float(_cap_radius(state.outer_rings[-1])),
        "dual_cell_sum_m3": float(snapshot_msh_volume_m3),
        "snapshot_msh_volume_m3": float(snapshot_msh_volume_m3),
        "dt": float(state.last_step_dt),
        "dt_limit_contact_line": float(state.last_dt_limit_cl),
        "dt_limit_capillary": float(state.last_dt_limit_capillary),
        "dt_limit_mesh": float(state.last_dt_limit_mesh),
        "dt_limiter": str(state.last_dt_limiter),
        "bottom_contact_line_speed": float(state.last_bottom_contact_line_speed),
        "top_contact_line_speed": float(state.last_top_contact_line_speed),
        "max_free_speed": _max_free_speed(state),
        "pressure_scalar": float(state.pressure_scalar),
        "force_match_pressure_offset_pa": float(getattr(state, "force_match_pressure_offset_pa", 0.0)),
        "initial_force_calibration_target_mn": float(getattr(state, "initial_force_calibration_target_mn", 0.0)),
        "initial_force_calibration_fixed_axial_mn": float(
            getattr(state, "initial_force_calibration_fixed_axial_mn", 0.0)
        ),
        "ns_pressure_projection_l2": float(getattr(state, "last_ns_pressure_l2", 0.0)),
        "ns_projection_iterations": int(getattr(state, "last_ns_projection_iterations", 0)),
        "ns_divergence_l2": float(getattr(state, "last_ns_divergence_l2", 0.0)),
        "ns_projected_divergence_l2": float(getattr(state, "last_ns_projected_divergence_l2", 0.0)),
        "boundary_volume_flux_m3_s": float(_boundary_volume_flux(state)),
        "step_continuity_volume_error_m3": float(getattr(state, "last_step_continuity_volume_error_m3", 0.0)),
        "step_continuity_displacement_m3": float(getattr(state, "last_step_continuity_displacement_m3", 0.0)),
        "n_vertices": sum(1 for _ in state.HC.V),
        "n_free_vertices": sum(1 for v in state.HC.V if v not in state.bV_caps),
        "runtime_remesh_count": int(getattr(state, "runtime_remesh_count", 0)),
        "last_remesh_reason": str(getattr(state, "last_remesh_reason", "none")),
        "contact_side_edge_fraction": float(_contact_side_edge_fraction(state)),
        "extra_cl_radial_rings": int(getattr(state.config, "extra_cl_radial_rings", 0)),
        "cl_extra_axial_layers": int(getattr(state.config, "cl_extra_axial_layers", 0)),
        "neck_extra_axial_layers": int(getattr(state.config, "neck_extra_axial_layers", 0)),
    }


def _surface_side_triangles(state: VolumetricPitoisState) -> np.ndarray:
    if USER_SHOW_REAL_COMPUTE_TRIANGLES:
        return _actual_compute_side_surface_triangles(state)
    ordered_rings = _ordered_outer_rings_for_surface(state)
    if not ordered_rings:
        return np.empty((0, 3, 3), dtype=float)
    _surface, _surface_bV, flat_vertices, triangles = _build_structured_side_surface_complex(ordered_rings)
    if triangles.size == 0:
        return np.empty((0, 3, 3), dtype=float)
    coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in flat_vertices], dtype=float)
    coords = _project_outside_spheres(state, coords)
    return np.asarray([coords[np.asarray(tri, dtype=int)] for tri in triangles], dtype=float)


def _project_outside_spheres(state: VolumetricPitoisState, coords: np.ndarray) -> np.ndarray:
    bottom_center, top_center, _axis = _particle_centers_physical(state)
    radius = float(state.config.particle_radius)
    clearance = max(1.0e-9 * radius, 1.0e-8)
    out = np.array(coords, dtype=float, copy=True)
    for sphere_center in (bottom_center, top_center):
        rel = out - sphere_center[None, :]
        dist = np.linalg.norm(rel, axis=1)
        inside = dist < (radius + clearance)
        if not np.any(inside):
            continue
        safe = np.maximum(dist[inside], 1.0e-30)
        dirs = rel[inside] / safe[:, None]
        out[inside] = sphere_center[None, :] + (radius + clearance) * dirs
    return out


def _surface_side_wireframe(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    # For normal display, keep the inherited ring/meridian graph instead of
    # triangle-diagonal edges. This avoids spurious long diagonals and dangling
    # one-edge points caused by hidden-edge culling on the raw tetra boundary.
    ordered_rings = _ordered_outer_rings_for_surface(state)
    if not ordered_rings:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)
    edges = _structured_surface_display_edges(ordered_rings)
    coords = np.array(
        [np.asarray(v.x_a[:3], dtype=float) for ring in ordered_rings for v in ring],
        dtype=float,
    )
    coords = _project_outside_spheres(state, coords)
    segments = [np.array([coords[i], coords[j]], dtype=float) for i, j in sorted(edges)]
    segs = np.asarray(segments, dtype=float) if segments else np.empty((0, 2, 3), dtype=float)
    return segs, coords


def _wireframe_scatter_vertices(segments: np.ndarray, *, min_degree: int = 3) -> np.ndarray:
    segs = np.asarray(segments, dtype=float)
    if segs.size == 0:
        return np.empty((0, 3), dtype=float)

    adjacency: dict[tuple[float, float, float], set[tuple[float, float, float]]] = defaultdict(set)
    point_lookup: dict[tuple[float, float, float], np.ndarray] = {}
    for seg in segs:
        a = tuple(np.round(np.asarray(seg[0], dtype=float), decimals=12))
        b = tuple(np.round(np.asarray(seg[1], dtype=float), decimals=12))
        point_lookup[a] = np.asarray(seg[0], dtype=float)
        point_lookup[b] = np.asarray(seg[1], dtype=float)
        adjacency[a].add(b)
        adjacency[b].add(a)

    kept = [point_lookup[key] for key, nbrs in adjacency.items() if len(nbrs) >= int(min_degree)]
    if not kept:
        return np.empty((0, 3), dtype=float)
    return np.asarray(kept, dtype=float)


def _plot_coords_mm(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return 1.0e3 * points


def _particle_centers_physical(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bottom_center = np.asarray(state.bottom_sphere_center, dtype=float)
    top_center = np.asarray(state.top_sphere_center, dtype=float)
    axis = top_center - bottom_center
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    return bottom_center, top_center, axis


def _particle_center_positions(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    bottom_center, top_center, _axis = _particle_centers_physical(state)
    return _plot_coords_mm(bottom_center[None, :])[0], _plot_coords_mm(top_center[None, :])[0]


def _cap_plane_positions(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, float]:
    bottom_cap_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in state.outer_rings[0]], axis=0)
    top_cap_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in state.outer_rings[-1]], axis=0)
    contact_radius = min(
        max(_cap_radius(state.outer_rings[0]), _cap_radius(state.outer_rings[-1])),
        state.config.particle_radius,
    )
    return (
        _plot_coords_mm(bottom_cap_center[None, :])[0],
        _plot_coords_mm(top_cap_center[None, :])[0],
        1.0e3 * contact_radius,
    )


def _sphere_cap_triangles(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    bottom_center, top_center, axis = _particle_centers_physical(state)
    radius = float(state.config.particle_radius)

    def project_to_sphere(point: np.ndarray, plane_center: np.ndarray, sphere_center: np.ndarray, sign: float) -> np.ndarray:
        tangential = point - plane_center
        tangential -= axis * float(np.dot(tangential, axis))
        r_t = float(np.linalg.norm(tangential))
        axial = np.sqrt(max(radius**2 - r_t**2, 0.0))
        return sphere_center + tangential + sign * axial * axis

    cap_triangle_sets: list[list[np.ndarray]] = []
    cap_specs = (
        (state.layer_rings[0], np.mean([np.asarray(v.x_a[:3], dtype=float) for v in state.cap_bottom], axis=0), bottom_center, +1.0),
        (state.layer_rings[-1], np.mean([np.asarray(v.x_a[:3], dtype=float) for v in state.cap_top], axis=0), top_center, -1.0),
    )

    for rings, plane_center, sphere_center, sign in cap_specs:
        triangles: list[np.ndarray] = []
        ring_xyz = []
        for ring in rings:
            ring_xyz.append(
                [
                    project_to_sphere(np.asarray(v.x_a[:3], dtype=float), plane_center, sphere_center, sign)
                    for v in ring
                ]
            )

        if not ring_xyz:
            continue

        center = sphere_center + sign * radius * axis
        n_ring = len(ring_xyz[0])
        for i in range(n_ring):
            triangles.append(
                np.array(
                    [
                        center,
                        ring_xyz[0][i],
                        ring_xyz[0][(i + 1) % n_ring],
                    ],
                    dtype=float,
                )
            )
        for band in range(len(ring_xyz) - 1):
            inner = ring_xyz[band]
            outer = ring_xyz[band + 1]
            for i in range(n_ring):
                triangles.append(np.array([inner[i], outer[i], outer[(i + 1) % n_ring]], dtype=float))
                triangles.append(np.array([inner[i], outer[(i + 1) % n_ring], inner[(i + 1) % n_ring]], dtype=float))
        cap_triangle_sets.append(triangles)

    while len(cap_triangle_sets) < 2:
        cap_triangle_sets.append([])

    bottom_tris = np.asarray(cap_triangle_sets[0], dtype=float) if cap_triangle_sets[0] else np.empty((0, 3, 3), dtype=float)
    top_tris = np.asarray(cap_triangle_sets[1], dtype=float) if cap_triangle_sets[1] else np.empty((0, 3, 3), dtype=float)
    return bottom_tris, top_tris


def _snapshot_title(state: VolumetricPitoisState, *, label: str) -> str:
    n_vertices = sum(1 for _ in state.HC.V)
    n_free = sum(1 for v in state.HC.V if v not in state.bV_caps)
    return f"{state.config.title}: {label} mesh\nvertices={n_vertices}, free={n_free}"


def _render_particle_wireframe(ax, center: np.ndarray, radius: float, color: str) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 18)
    v = np.linspace(0.0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=2,
        cstride=2,
        color=color,
        linewidth=0.5,
        alpha=0.16,
        axlim_clip=True,
    )
    ax.scatter(
        [center[0]],
        [center[1]],
        [center[2]],
        color=color,
        s=36,
        edgecolors="#111111",
        linewidths=0.6,
        axlim_clip=True,
    )


def _render_contact_circle(ax, cap_center: np.ndarray, radius: float, axis: np.ndarray, color: str) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 121)
    axis = np.asarray(axis, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-30)
    e1, e2 = _orthonormal_tangent_basis(axis, np.array([1.0, 0.0, 0.0], dtype=float))
    ring = cap_center[None, :] + radius * (
        np.cos(theta)[:, None] * e1[None, :] + np.sin(theta)[:, None] * e2[None, :]
    )
    ax.plot(
        ring[:, 0],
        ring[:, 1],
        ring[:, 2],
        color=color,
        linewidth=1.6,
        alpha=0.98,
        axlim_clip=True,
    )


def _scatter_mesh_vertices(ax, triangle_sets: list[np.ndarray], *, color: str = "#111111") -> None:
    if not USER_SHOW_MESH_VERTICES:
        return
    pts = [tris.reshape(-1, 3) for tris in triangle_sets if tris.size]
    if not pts:
        return
    flat = np.concatenate(pts, axis=0)
    rounded = np.round(flat, decimals=12)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    verts = flat[np.sort(unique_idx)]
    marker_size = max(2.0, 0.4 * float(USER_MESH_VERTEX_SIZE))
    ax.scatter(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        s=marker_size,
        facecolors="none",
        edgecolors=color,
        linewidths=0.5,
        alpha=0.98,
        depthshade=False,
        axlim_clip=True,
    )


def _render_triangle_edges(
    ax,
    triangles: np.ndarray,
    *,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    if triangles.size == 0:
        return
    segments = []
    seen: set[tuple[tuple[float, float, float], tuple[float, float, float]]] = set()
    for tri in np.asarray(triangles, dtype=float):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            p = tuple(np.round(tri[i], 12))
            q = tuple(np.round(tri[j], 12))
            key = (p, q) if p <= q else (q, p)
            if key in seen:
                continue
            seen.add(key)
            segments.append(np.array([tri[i], tri[j]], dtype=float))
    if not segments:
        return
    ax.add_collection3d(
        Line3DCollection(
            segments,
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            axlim_clip=True,
        )
    )


def _tet_signed_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    tet_idx = np.asarray(tets, dtype=int)
    if pts.ndim != 2 or pts.shape[1] != 3 or tet_idx.ndim != 2 or tet_idx.shape[1] != 4:
        return np.empty(0, dtype=float)
    if pts.shape[0] == 0 or tet_idx.shape[0] == 0:
        return np.empty(0, dtype=float)
    tet_pts = pts[tet_idx]
    a = tet_pts[:, 0, :]
    b = tet_pts[:, 1, :]
    c = tet_pts[:, 2, :]
    d = tet_pts[:, 3, :]
    return np.einsum("ij,ij->i", a - d, np.cross(b - d, c - d)) / 6.0


def _orient_tets_positive(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    tet_idx = np.asarray(tets, dtype=int).copy()
    if tet_idx.ndim != 2 or tet_idx.shape[1] != 4 or tet_idx.shape[0] == 0:
        return tet_idx
    signed = _tet_signed_volumes(points, tet_idx)
    neg = signed < 0.0
    if np.any(neg):
        first = tet_idx[neg, 0].copy()
        tet_idx[neg, 0] = tet_idx[neg, 1]
        tet_idx[neg, 1] = first
    return tet_idx


def _indexed_tet_quality_metrics(points: np.ndarray, tets: np.ndarray) -> dict[str, float]:
    pts = np.asarray(points, dtype=float)
    tet_idx = np.asarray(tets, dtype=int)
    if pts.ndim != 2 or pts.shape[1] != 3 or tet_idx.ndim != 2 or tet_idx.shape[1] != 4:
        return {
            "quality_min": 0.0,
            "edge_min_m": 0.0,
            "signed_volume_min_m3": 0.0,
            "negative_tets": 0.0,
        }
    if pts.shape[0] == 0 or tet_idx.shape[0] == 0:
        return {
            "quality_min": 0.0,
            "edge_min_m": 0.0,
            "signed_volume_min_m3": 0.0,
            "negative_tets": 0.0,
        }

    tet_pts = pts[tet_idx]
    signed = _tet_signed_volumes(pts, tet_idx)
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edge_lengths = np.stack(
        [
            np.linalg.norm(tet_pts[:, i, :] - tet_pts[:, j, :], axis=1)
            for i, j in edge_pairs
        ],
        axis=1,
    )
    edge_sum_sq = np.sum(edge_lengths * edge_lengths, axis=1)
    quality = np.zeros(tet_idx.shape[0], dtype=float)
    valid = edge_sum_sq > 1.0e-30
    quality[valid] = 12.0 * ((3.0 * np.abs(signed[valid])) ** (2.0 / 3.0)) / edge_sum_sq[valid]
    return {
        "quality_min": float(np.min(quality)),
        "edge_min_m": float(np.min(edge_lengths)),
        "signed_volume_min_m3": float(np.min(signed)),
        "negative_tets": float(np.count_nonzero(signed < 0.0)),
    }


def _write_gmsh2(
    points: np.ndarray,
    elements: np.ndarray,
    out_path: Path,
    *,
    element_type: int,
    node_ids: np.ndarray | None = None,
) -> None:
    points = np.asarray(points, dtype=float)
    elements = np.asarray(elements, dtype=int)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    expected_cols = 3 if int(element_type) == 2 else 4 if int(element_type) == 4 else None
    if expected_cols is None:
        raise ValueError("element_type must be 2 (triangles) or 4 (tetrahedra)")
    if elements.ndim != 2 or elements.shape[1] != expected_cols:
        raise ValueError(f"elements must be (M,{expected_cols})")
    if node_ids is None:
        node_ids = np.arange(1, points.shape[0] + 1, dtype=int)
    else:
        node_ids = np.asarray(node_ids, dtype=int)
        if node_ids.ndim != 1 or node_ids.shape[0] != points.shape[0]:
            raise ValueError("node_ids must be a length-N array")

    with open(out_path, "w", newline="") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write(f"{points.shape[0]}\n")
        for node_id, (x, y, z) in zip(node_ids, points):
            f.write(f"{int(node_id)} {x:.17g} {y:.17g} {z:.17g}\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write(f"{elements.shape[0]}\n")
        for i, elem in enumerate(elements, start=1):
            node_tokens = " ".join(str(int(node_ids[int(local_idx)])) for local_idx in elem)
            f.write(f"{i} {int(element_type)} 0 {node_tokens}\n")
        f.write("$EndElements\n")


def _volume_points_array(state: VolumetricPitoisState) -> np.ndarray:
    _ensure_volume_export_node_ids(state)
    return np.asarray(
        [np.asarray(vertex.x_a[:3], dtype=float) for vertex in state.volume_export_vertices],
        dtype=float,
    )


def _snapshot_msh_indexed_mesh(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _freeze_surface_topology(state)
    points = _volume_points_array(state)
    tets = _orient_tets_positive(points, np.asarray(state.volume_export_tets, dtype=int))
    node_ids = np.arange(1, points.shape[0] + 1, dtype=int)
    return points, tets, node_ids


def _indexed_tet_mesh_volume_m3(points: np.ndarray, tets: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    tet_idx = np.asarray(tets, dtype=int)
    if pts.size == 0 or tet_idx.size == 0:
        return 0.0

    signed = _tet_signed_volumes(pts, tet_idx)
    return float(np.sum(np.abs(signed)))


def _snapshot_msh_volume_m3(state: VolumetricPitoisState) -> float:
    points, tets, _node_ids = _snapshot_msh_indexed_mesh(state)
    return _indexed_tet_mesh_volume_m3(points, tets)


def _current_tet_mesh_quality_metrics(state: VolumetricPitoisState) -> dict[str, float]:
    points, tets, _node_ids = _snapshot_msh_indexed_mesh(state)
    metrics = _indexed_tet_quality_metrics(points, tets)
    metrics["adjacent_layer_spacing_min_m"] = float(_adjacent_layer_min_distance(state))
    return metrics


def _initialize_mesh_quality_baseline(state: VolumetricPitoisState) -> None:
    metrics = _current_tet_mesh_quality_metrics(state)
    state.baseline_tet_quality_min = max(float(metrics["quality_min"]), 1.0e-12)
    state.baseline_tet_edge_min_m = max(float(metrics["edge_min_m"]), 1.0e-12)
    state.baseline_adjacent_layer_spacing_min_m = max(
        float(metrics["adjacent_layer_spacing_min_m"]),
        1.0e-12,
    )


def _mesh_quality_thresholds(state: VolumetricPitoisState) -> dict[str, float]:
    return {
        "quality_min": max(
            1.0e-4,
            float(state.config.mesh_quality_min_fraction) * float(state.baseline_tet_quality_min),
        ),
        "edge_min_m": max(
            1.0e-7,
            float(state.config.mesh_edge_min_fraction) * float(state.baseline_tet_edge_min_m),
        ),
        "adjacent_layer_spacing_min_m": max(
            1.0e-8,
            float(state.config.layer_spacing_min_fraction) * float(state.baseline_adjacent_layer_spacing_min_m),
        ),
    }


def _mesh_quality_is_acceptable(state: VolumetricPitoisState) -> tuple[bool, dict[str, float]]:
    metrics = _current_tet_mesh_quality_metrics(state)
    thresholds = _mesh_quality_thresholds(state)
    rel_tol = max(float(getattr(state.config, "mesh_quality_accept_rel_tol", 0.0)), 0.0)
    hard_rel_tol = max(float(getattr(state.config, "mesh_hard_accept_rel_tol", 0.0)), 0.0)
    hard_fraction = max(float(getattr(state.config, "mesh_quality_hard_fraction", 0.0)), 0.0)
    quality_abs_floor = max(float(getattr(state.config, "mesh_quality_abs_floor", 0.0)), 0.0)
    edge_hard_fraction = max(float(getattr(state.config, "mesh_edge_hard_fraction", 0.0)), 0.0)
    spacing_hard_fraction = max(float(getattr(state.config, "layer_spacing_hard_fraction", 0.0)), 0.0)
    quality_limit = float(thresholds["quality_min"]) * max(1.0 - rel_tol, 0.0)
    quality_hard_limit = float(thresholds["quality_min"]) * hard_fraction * max(1.0 - hard_rel_tol, 0.0)
    edge_limit = float(thresholds["edge_min_m"]) * max(1.0 - rel_tol, 0.0)
    edge_hard_limit = float(thresholds["edge_min_m"]) * edge_hard_fraction * max(1.0 - hard_rel_tol, 0.0)
    spacing_limit = float(thresholds["adjacent_layer_spacing_min_m"]) * max(1.0 - rel_tol, 0.0)
    spacing_hard_limit = (
        float(thresholds["adjacent_layer_spacing_min_m"]) * spacing_hard_fraction * max(1.0 - hard_rel_tol, 0.0)
    )
    fail_edge = not np.isfinite(metrics["edge_min_m"]) or float(metrics["edge_min_m"]) < edge_limit
    fail_spacing = (
        not np.isfinite(metrics["adjacent_layer_spacing_min_m"])
        or float(metrics["adjacent_layer_spacing_min_m"]) < spacing_limit
    )
    edge_hard_fail = (
        not np.isfinite(metrics["edge_min_m"])
        or float(metrics["edge_min_m"]) < edge_hard_limit
    )
    spacing_hard_fail = (
        not np.isfinite(metrics["adjacent_layer_spacing_min_m"])
        or float(metrics["adjacent_layer_spacing_min_m"]) < spacing_hard_limit
    )
    quality_hard_fail = (
        not np.isfinite(metrics["quality_min"])
        or float(metrics["quality_min"]) < quality_hard_limit
    )
    quality_abs_fail = (
        not np.isfinite(metrics["quality_min"])
        or float(metrics["quality_min"]) < quality_abs_floor
    )
    fail_quality = (
        not np.isfinite(metrics["quality_min"])
        or float(metrics["quality_min"]) < quality_limit
    )
    enforce_edge = bool(fail_edge and edge_hard_fail)
    enforce_spacing = bool(fail_spacing and spacing_hard_fail)
    enforce_quality = bool(fail_quality and (quality_abs_fail or enforce_edge or enforce_spacing))
    metrics.update(
        {
            "quality_min_threshold": float(thresholds["quality_min"]),
            "edge_min_threshold_m": float(thresholds["edge_min_m"]),
            "adjacent_layer_spacing_threshold_m": float(thresholds["adjacent_layer_spacing_min_m"]),
            "quality_min_limit": float(quality_limit),
            "quality_min_hard_limit": float(quality_hard_limit),
            "quality_min_abs_floor": float(quality_abs_floor),
            "edge_min_limit_m": float(edge_limit),
            "edge_min_hard_limit_m": float(edge_hard_limit),
            "adjacent_layer_spacing_limit_m": float(spacing_limit),
            "adjacent_layer_spacing_hard_limit_m": float(spacing_hard_limit),
            "mesh_quality_accept_rel_tol": float(rel_tol),
            "mesh_hard_accept_rel_tol": float(hard_rel_tol),
            "mesh_quality_hard_fraction": float(hard_fraction),
            "mesh_quality_abs_floor": float(quality_abs_floor),
            "mesh_edge_hard_fraction": float(edge_hard_fraction),
            "layer_spacing_hard_fraction": float(spacing_hard_fraction),
            "fail_quality": bool(fail_quality),
            "enforce_quality": bool(enforce_quality),
            "quality_hard_fail": bool(quality_hard_fail),
            "quality_abs_fail": bool(quality_abs_fail),
            "fail_edge": bool(fail_edge),
            "enforce_edge": bool(enforce_edge),
            "edge_hard_fail": bool(edge_hard_fail),
            "fail_spacing": bool(fail_spacing),
            "enforce_spacing": bool(enforce_spacing),
            "spacing_hard_fail": bool(spacing_hard_fail),
        }
    )
    ok = not (enforce_quality or enforce_edge or enforce_spacing)
    return bool(ok), metrics


def _save_mesh_snapshot_msh(state: VolumetricPitoisState, out_path: Path) -> Path:
    points, tets, node_ids = _snapshot_msh_indexed_mesh(state)
    msh_path = out_path.with_suffix(".msh")
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    _write_gmsh2(points, tets, msh_path, element_type=4, node_ids=node_ids)
    return msh_path


def _set_axes_equal(ax, xyz: np.ndarray, centers: list[np.ndarray], radius: float) -> None:
    pts = np.asarray(xyz.reshape(-1, 3), dtype=float) if xyz.size else np.empty((0, 3), dtype=float)
    if centers:
        centers_arr = np.asarray(centers, dtype=float)
        sphere_bbox = []
        for center in centers_arr:
            sphere_bbox.extend(
                [
                    center + np.array([+radius, 0.0, 0.0], dtype=float),
                    center + np.array([-radius, 0.0, 0.0], dtype=float),
                    center + np.array([0.0, +radius, 0.0], dtype=float),
                    center + np.array([0.0, -radius, 0.0], dtype=float),
                    center + np.array([0.0, 0.0, +radius], dtype=float),
                    center + np.array([0.0, 0.0, -radius], dtype=float),
                ]
            )
        pts = np.vstack([pts, centers_arr, np.asarray(sphere_bbox, dtype=float)])
    if pts.size == 0:
        pts = np.zeros((1, 3), dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    x_min = float(USER_X_AXIS_MIN_MM)
    x_max = float(USER_X_AXIS_MAX_MM)
    z_min = float(USER_Z_AXIS_MIN_MM)
    z_max = float(USER_Z_AXIS_MAX_MM)

    x_half = 0.5 * max(x_max - x_min, 0.0)
    z_half = 0.5 * max(z_max - z_min, 0.0)
    x_center = 0.5 * (x_min + x_max)
    z_center = 0.5 * (z_min + z_max)

    if x_half <= 0.0:
        x_half = max(0.5 * float(maxs[0] - mins[0]), float(radius), 1.0)
        x_center = float(center[0])
    if z_half <= 0.0:
        z_half = max(0.5 * float(maxs[2] - mins[2]), float(radius), 1.0)
        z_center = float(center[2])

    half_span = max(x_half, z_half)
    ax.set_xlim(x_center - half_span, x_center + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(z_center - half_span, z_center + half_span)
    ax.set_box_aspect((1, 1, 1))


def _render_mesh_snapshot(
    state: VolumetricPitoisState,
    title: str,
    out_path: Path,
    *,
    elev: float = USER_INTERACTIVE_ELEV_DEG,
    azim: float = USER_INTERACTIVE_AZIM_DEG,
) -> Path:
    side_triangles_full = _surface_side_triangles(state)
    if USER_SHOW_REAL_COMPUTE_TRIANGLES and USER_REAL_TRIANGLE_VIEW == "meridian_strip":
        side_triangles, bottom_cap_triangles, top_cap_triangles = _actual_compute_meridian_strip_triangles(state)
    elif USER_SHOW_REAL_COMPUTE_TRIANGLES:
        side_triangles = side_triangles_full
        bottom_cap_triangles, top_cap_triangles = _actual_compute_cap_triangles(state)
    else:
        side_triangles = side_triangles_full
        bottom_cap_triangles, top_cap_triangles = _sphere_cap_triangles(state)
        side_segments, side_vertices = _surface_side_wireframe(state)
    if USER_SHOW_REAL_COMPUTE_TRIANGLES:
        if USER_REAL_TRIANGLE_VIEW == "meridian_strip" and not USER_SHOW_MESH_FACES:
            side_segments, side_vertices = _meridian_section_wireframe(state)
        else:
            side_segments, side_vertices = _surface_side_wireframe(state)
    side_vertices = _wireframe_scatter_vertices(side_segments, min_degree=4)
    plot_side_segments = _plot_coords_mm(side_segments)
    plot_side_vertices = _plot_coords_mm(side_vertices)
    plot_side_tris = _plot_coords_mm(side_triangles)
    plot_bottom_cap_tris = _plot_coords_mm(bottom_cap_triangles)
    plot_top_cap_tris = _plot_coords_mm(top_cap_triangles)
    bottom_center, top_center = _particle_center_positions(state)
    _bottom_center_phys, _top_center_phys, plot_axis = _particle_centers_physical(state)
    bottom_cap_center, top_cap_center, contact_radius_mm = _cap_plane_positions(state)
    radius_mm = 1.0e3 * state.config.particle_radius

    fig = plt.figure(figsize=(7.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")

    liquid_triangle_sets = [plot_side_tris] if plot_side_tris.size else []
    if USER_FILL_PARTICLE_CAP_SURFACES:
        if plot_bottom_cap_tris.size:
            liquid_triangle_sets.append(plot_bottom_cap_tris)
        if plot_top_cap_tris.size:
            liquid_triangle_sets.append(plot_top_cap_tris)
    if plot_side_tris.size and USER_SHOW_SURFACE_OVERLAY:
        overlay_poly = Poly3DCollection(
            plot_side_tris,
            facecolor="#1f9d8a",
            edgecolor=(0.0, 0.0, 0.0, 0.0),
            linewidth=0.0,
            alpha=USER_SURFACE_OVERLAY_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(overlay_poly)
    if liquid_triangle_sets and USER_SHOW_MESH_FACES:
        poly_liquid = Poly3DCollection(
            np.concatenate(liquid_triangle_sets, axis=0),
            facecolor="#1f9d8a",
            edgecolor=(0.0, 0.0, 0.0, 0.0),
            linewidth=0.0,
            alpha=USER_MESH_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(poly_liquid)
    if USER_SHOW_REAL_COMPUTE_TRIANGLES and USER_REAL_TRIANGLE_VIEW == "full":
        vertex_triangle_sets = [plot_side_tris]
        if USER_SHOW_CAP_TRIANGLE_EDGES:
            vertex_triangle_sets.extend([plot_bottom_cap_tris, plot_top_cap_tris])
        _scatter_mesh_vertices(ax, vertex_triangle_sets)
        _render_triangle_edges(ax, plot_side_tris, color="#111111", linewidth=0.72, alpha=0.995)
        if USER_SHOW_CAP_TRIANGLE_EDGES:
            _render_triangle_edges(ax, plot_bottom_cap_tris, color="#2b6cb0", linewidth=0.72, alpha=USER_CAP_EDGE_ALPHA)
            _render_triangle_edges(ax, plot_top_cap_tris, color="#dd8a1c", linewidth=0.72, alpha=USER_CAP_EDGE_ALPHA)
    else:
        _scatter_mesh_vertices(ax, [plot_side_vertices])
        ax.add_collection3d(
            Line3DCollection(
                plot_side_segments,
                colors="#111111",
                linewidths=0.72,
                alpha=0.995,
                axlim_clip=True,
            )
        )
    if not USER_SHOW_MESH_FACES and USER_FILL_PARTICLE_CAP_SURFACES and not USER_SHOW_CAP_TRIANGLE_EDGES:
        _render_triangle_edges(ax, plot_bottom_cap_tris, color="#2b6cb0", linewidth=0.72, alpha=USER_CAP_EDGE_ALPHA)
        _render_triangle_edges(ax, plot_top_cap_tris, color="#dd8a1c", linewidth=0.72, alpha=USER_CAP_EDGE_ALPHA)
    if USER_SHOW_MESH_FACES and plot_bottom_cap_tris.size:
        poly_bottom_caps = Poly3DCollection(
            plot_bottom_cap_tris,
            facecolor=(0.0, 0.0, 0.0, 0.0),
            edgecolor="#2b6cb0",
            linewidth=0.72,
            alpha=USER_CAP_EDGE_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(poly_bottom_caps)
    if USER_SHOW_MESH_FACES and plot_top_cap_tris.size:
        poly_top_caps = Poly3DCollection(
            plot_top_cap_tris,
            facecolor=(0.0, 0.0, 0.0, 0.0),
            edgecolor="#dd8a1c",
            linewidth=0.72,
            alpha=USER_CAP_EDGE_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(poly_top_caps)

    _render_particle_wireframe(ax, bottom_center, radius_mm, "#2b6cb0")
    _render_particle_wireframe(ax, top_center, radius_mm, "#dd8a1c")
    if USER_SHOW_CONTACT_RING_OVERLAY:
        _render_contact_circle(ax, bottom_cap_center, contact_radius_mm, plot_axis, "#2b6cb0")
        _render_contact_circle(ax, top_cap_center, contact_radius_mm, plot_axis, "#dd8a1c")
    ax.plot(
        [bottom_center[0], top_center[0]],
        [bottom_center[1], top_center[1]],
        [bottom_center[2], top_center[2]],
        linestyle="--",
        color="#9ca3af",
        linewidth=1.4,
    )

    ax.set_title(title, pad=12)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    plot_all = []
    if plot_side_tris.size:
        plot_all.append(plot_side_tris)
    if plot_bottom_cap_tris.size:
        plot_all.append(plot_bottom_cap_tris)
    if plot_top_cap_tris.size:
        plot_all.append(plot_top_cap_tris)
    plot_xyz = np.concatenate(plot_all, axis=0) if plot_all else np.empty((0, 3, 3))
    _set_axes_equal(ax, plot_xyz, [bottom_center, top_center], radius_mm)
    ax.grid(True, alpha=0.28)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    _save_mesh_snapshot_msh(state, out_path)
    return out_path


def _render_history_figures(
    history: list[dict],
    config: VolumetricPitoisConfig,
    out_dir: Path,
    *,
    initial_snapshot_volume_m3: float,
) -> list[Path]:
    if not history:
        return []

    fig_dir = out_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    t_ms = np.array([float(row["t"]) * 1.0e3 for row in history], dtype=float)
    d_over_r = np.array([float(row["d_over_r"]) for row in history], dtype=float)
    gap_um = np.array([float(row["gap"]) * 1.0e6 for row in history], dtype=float)
    fixed_force_mn = np.array([float(row["fixed_force_axial"]) * 1.0e3 for row in history], dtype=float)
    max_u = np.array([float(row["max_free_speed"]) for row in history], dtype=float)

    paths: list[Path] = []

    fig1, ax1 = plt.subplots(figsize=(6.4, 4.4))
    ax1.plot(d_over_r, fixed_force_mn, color="#bc4749", linewidth=1.8)
    ax1.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.8)
    ax1.set_xlabel(r"$D/R$")
    ax1.set_ylabel("Fixed-sphere axial force [mN]")
    ax1.set_title("Separation force vs gap")
    ax1.grid(alpha=0.28)
    path1 = fig_dir / "separation_force_vs_gap.png"
    fig1.tight_layout()
    fig1.savefig(path1, dpi=180, bbox_inches="tight")
    plt.close(fig1)
    paths.append(path1)

    fig2, axes = plt.subplots(2, 1, figsize=(6.6, 6.0), sharex=True)
    axes[0].plot(t_ms, gap_um, color="#2a6f97", linewidth=1.8)
    axes[0].set_ylabel("Gap [um]")
    axes[0].grid(alpha=0.28)
    axes[0].set_title("Separation gap and free-speed history")
    axes[1].plot(t_ms, max_u, color="#6a4c93", linewidth=1.8)
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel("Max free speed [m/s]")
    axes[1].grid(alpha=0.28)
    path2 = fig_dir / "separation_gap_speed_vs_time.png"
    fig2.tight_layout()
    fig2.savefig(path2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    paths.append(path2)

    has_initial_row = bool(history and int(history[0].get("step", -1)) == 0)
    history_volumes = [
        float(row.get("snapshot_msh_volume_m3", row.get("snapshot_surface_volume_m3", 0.0)))
        for row in history
    ]
    history_t_ms = [float(row["t"]) * 1.0e3 for row in history]
    if has_initial_row:
        snapshot_volume_m3 = np.array(history_volumes, dtype=float)
        t_ms_volume = np.array(history_t_ms, dtype=float)
    else:
        snapshot_volume_m3 = np.array([float(initial_snapshot_volume_m3)] + history_volumes, dtype=float)
        t_ms_volume = np.array([0.0] + history_t_ms, dtype=float)
    snapshot_volume_ul = 1.0e9 * snapshot_volume_m3
    snapshot_rel_error_pct = 100.0 * (snapshot_volume_m3 - float(initial_snapshot_volume_m3)) / max(
        float(initial_snapshot_volume_m3), 1.0e-30
    )

    fig3, axes3 = plt.subplots(2, 1, figsize=(6.6, 6.2), sharex=True)
    axes3[0].plot(t_ms_volume, snapshot_volume_ul, color="#0f766e", linewidth=1.8)
    axes3[0].set_ylabel("Volume [uL]")
    axes3[0].set_title("Volumetric .msh liquid-bridge volume history")
    axes3[0].grid(alpha=0.28)
    axes3[1].plot(t_ms_volume, snapshot_rel_error_pct, color="#bc4749", linewidth=1.8)
    axes3[1].axhline(0.0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.8)
    axes3[1].set_xlabel("Time [ms]")
    axes3[1].set_ylabel("Rel. error [%]")
    axes3[1].grid(alpha=0.28)
    path3 = fig_dir / "separation_snapshot_volume_rel_error_vs_time.png"
    fig3.tight_layout()
    fig3.savefig(path3, dpi=180, bbox_inches="tight")
    plt.close(fig3)
    paths.append(path3)

    return paths


def _render_motion_mesh_pngs(
    state: VolumetricPitoisState,
    *,
    motion_dir: Path,
    label: str,
    step: int,
) -> Path:
    motion_dir.mkdir(parents=True, exist_ok=True)
    if label == "initial":
        filename = "mesh_initial.png"
    elif label == "final":
        filename = "mesh_final.png"
    else:
        filename = f"mesh_iter{step:04d}.png"
    title = _snapshot_title(state, label=label if label in {"initial", "final"} else f"iteration {step}")
    return _render_mesh_snapshot(state, title, motion_dir / filename)


def _save_history(history: list[dict], config: VolumetricPitoisConfig, out_dir: Path) -> Path:
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "history": history,
    }
    path = results_dir / "separation_history.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _history_arrays(history: list[dict]) -> dict[str, np.ndarray]:
    d_over_r = np.array([float(row["d_over_r"]) for row in history], dtype=float)
    force_mn = np.array([float(row["fixed_force_axial"]) for row in history], dtype=float) * 1.0e3
    t_ms = np.array([float(row["t"]) for row in history], dtype=float) * 1.0e3
    max_u = np.array([float(row["max_free_speed"]) for row in history], dtype=float)
    return {
        "d_over_r": d_over_r,
        "force_mn": force_mn,
        "force_abs_mn": np.abs(force_mn),
        "t_ms": t_ms,
        "max_u": max_u,
    }


def _pitois_fig5_dynamic_digitized() -> tuple[np.ndarray, np.ndarray]:
    d_over_r = np.array(
        [
            PITOIS_FIG5_FIRST_D_OVER_R,
            0.023587192093071,
            0.028302414461984,
            0.033166898401356,
            0.037986935617921,
            0.042569921040376,
            0.047499765555437,
            0.051967121934836,
            0.056470352302334,
            0.061304411920954,
            0.066003127630689,
            0.070366380348669,
            0.075100720316807,
            0.079819428415061,
            0.085182169652359,
            0.092208347873809,
            0.097048578239299,
            0.101370508634981,
            0.108834059521728,
            0.118227619608521,
            0.129680242729548,
            0.146372328112875,
            0.160984243023503,
            0.184443915271577,
            0.186311945235594,
            0.202973726366585,
        ],
        dtype=float,
    )
    force_mn = np.array(
        [
            PITOIS_FIG5_FIRST_FORCE_MN,
            0.955216555657008,
            0.822460624401661,
            0.74381061419963,
            0.65567423269621,
            0.574562744753398,
            0.517342047749411,
            0.45826545147229,
            0.416512293755708,
            0.376315664380932,
            0.337105214458683,
            0.30674808760201,
            0.286922242059943,
            0.267636737727435,
            0.238270688094022,
            0.218305191654015,
            0.1986840608733,
            0.178384121904434,
            0.158325437116782,
            0.139272315154625,
            0.1194131217624,
            0.099728317139538,
            0.079873442448703,
            0.05974152111953,
            0.052749970637026,
            0.042572497546823,
        ],
        dtype=float,
    )
    return d_over_r, force_mn


def _save_pitois_fig5_compare(
    *,
    exp_x: np.ndarray,
    exp_y: np.ndarray,
    sim_x: np.ndarray,
    sim_y: np.ndarray,
    out_path: Path,
    xlim: tuple[float, float] = (0.01, 0.30),
    ylim: tuple[float, float] = (1.0e-7, 2.0),
    title: str = "Pitois 2000 Fig. 5: dynamic experiment vs Case 2b",
) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.scatter(exp_x, exp_y, s=34, color="#111111", label="Pitois 2000 dynamic exp (black dots)")
    ax.plot(sim_x, sim_y, color="#d62828", linewidth=2.0, label="Case 2b simulation")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$D/R$")
    ax.set_ylabel(r"$|F|$ [mN]")
    ax.set_title(title)
    ax.grid(alpha=0.28, which="both")
    ax.legend()
    decimal_log_formatter = FuncFormatter(lambda value, _pos: f"{float(value):g}" if value > 0.0 else "")
    ax.xaxis.set_major_formatter(decimal_log_formatter)
    ax.yaxis.set_major_formatter(decimal_log_formatter)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _restore_interactive_backend() -> None:
    current = matplotlib.get_backend().lower()
    if "agg" not in current:
        return
    for backend in ("MacOSX", "TkAgg"):
        try:
            plt.switch_backend(backend)
            return
        except Exception:
            continue


def _advance_state(state: VolumetricPitoisState, *, n_steps: int) -> None:
    _assert_fixed_topology(state)
    for _step in range(int(n_steps)):
        step_dt = _select_physical_dt(state)
        substeps = max(1, int(state.config.integration_substeps))
        sub_dt = float(step_dt) / float(substeps)
        for _substep in range(substeps):
            _advance_one_substep(state, dt=sub_dt)
        state.elapsed_time_s += float(step_dt)


def _show_state_interactive(
    state: VolumetricPitoisState,
    *,
    step: int,
    elev: float,
    azim: float,
) -> None:
    _restore_interactive_backend()

    bottom_center_phys, top_center_phys, axis = _particle_centers_physical(state)
    center_distance = float(np.dot(top_center_phys - bottom_center_phys, axis))
    surface_gap = float(_gap(state))
    overlap = center_distance < 2.0 * float(state.config.particle_radius)

    side_triangles_full = _surface_side_triangles(state)
    if USER_SHOW_REAL_COMPUTE_TRIANGLES and USER_REAL_TRIANGLE_VIEW == "meridian_strip":
        side_triangles, bottom_cap_triangles, top_cap_triangles = _actual_compute_meridian_strip_triangles(state)
    elif USER_SHOW_REAL_COMPUTE_TRIANGLES:
        side_triangles = side_triangles_full
        bottom_cap_triangles, top_cap_triangles = _actual_compute_cap_triangles(state)
    else:
        side_triangles = side_triangles_full
        bottom_cap_triangles, top_cap_triangles = _sphere_cap_triangles(state)
        side_segments, side_vertices = _surface_side_wireframe(state)
    if USER_SHOW_REAL_COMPUTE_TRIANGLES:
        if USER_REAL_TRIANGLE_VIEW == "meridian_strip" and not USER_SHOW_MESH_FACES:
            side_segments, side_vertices = _meridian_section_wireframe(state)
        else:
            side_segments, side_vertices = _surface_side_wireframe(state)
    side_vertices = _wireframe_scatter_vertices(side_segments, min_degree=4)
    plot_side_segments = _plot_coords_mm(side_segments)
    plot_side_vertices = _plot_coords_mm(side_vertices)
    plot_side_tris = _plot_coords_mm(side_triangles)
    plot_bottom_cap_tris = _plot_coords_mm(bottom_cap_triangles)
    plot_top_cap_tris = _plot_coords_mm(top_cap_triangles)
    bottom_center, top_center = _particle_center_positions(state)
    bottom_cap_center, top_cap_center, contact_radius_mm = _cap_plane_positions(state)
    radius_mm = 1.0e3 * state.config.particle_radius

    fig = plt.figure(figsize=(8.4, 8.2))
    ax = fig.add_subplot(111, projection="3d")

    liquid_triangle_sets = [plot_side_tris] if plot_side_tris.size else []
    if USER_FILL_PARTICLE_CAP_SURFACES:
        if plot_bottom_cap_tris.size:
            liquid_triangle_sets.append(plot_bottom_cap_tris)
        if plot_top_cap_tris.size:
            liquid_triangle_sets.append(plot_top_cap_tris)
    if plot_side_tris.size and USER_SHOW_SURFACE_OVERLAY:
        overlay_poly = Poly3DCollection(
            plot_side_tris,
            facecolor="#1f9d8a",
            edgecolor=(0.0, 0.0, 0.0, 0.0),
            linewidth=0.0,
            alpha=USER_SURFACE_OVERLAY_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(overlay_poly)
    if liquid_triangle_sets and USER_SHOW_MESH_FACES:
        poly_liquid = Poly3DCollection(
            np.concatenate(liquid_triangle_sets, axis=0),
            facecolor="#1f9d8a",
            edgecolor=(0.0, 0.0, 0.0, 0.0),
            linewidth=0.0,
            alpha=USER_MESH_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(poly_liquid)
    if USER_SHOW_REAL_COMPUTE_TRIANGLES and USER_REAL_TRIANGLE_VIEW == "full":
        vertex_triangle_sets = [plot_side_tris]
        if USER_SHOW_CAP_TRIANGLE_EDGES:
            vertex_triangle_sets.extend([plot_bottom_cap_tris, plot_top_cap_tris])
        _scatter_mesh_vertices(ax, vertex_triangle_sets)
        _render_triangle_edges(ax, plot_side_tris, color="#111111", linewidth=0.72, alpha=0.995)
        if USER_SHOW_CAP_TRIANGLE_EDGES:
            _render_triangle_edges(ax, plot_bottom_cap_tris, color="#2b6cb0", linewidth=0.72, alpha=USER_CAP_EDGE_ALPHA)
            _render_triangle_edges(ax, plot_top_cap_tris, color="#dd8a1c", linewidth=0.72, alpha=USER_CAP_EDGE_ALPHA)
    else:
        _scatter_mesh_vertices(ax, [plot_side_vertices])
        ax.add_collection3d(
            Line3DCollection(
                plot_side_segments,
                colors="#111111",
                linewidths=0.72,
                alpha=0.995,
                axlim_clip=True,
            )
        )
    if not USER_SHOW_MESH_FACES and USER_FILL_PARTICLE_CAP_SURFACES and not USER_SHOW_CAP_TRIANGLE_EDGES:
        _render_triangle_edges(ax, plot_bottom_cap_tris, color="#2b6cb0", linewidth=0.72, alpha=USER_CAP_EDGE_ALPHA)
        _render_triangle_edges(ax, plot_top_cap_tris, color="#dd8a1c", linewidth=0.72, alpha=USER_CAP_EDGE_ALPHA)
    if USER_SHOW_MESH_FACES and plot_bottom_cap_tris.size:
        poly_bottom_caps = Poly3DCollection(
            plot_bottom_cap_tris,
            facecolor=(0.0, 0.0, 0.0, 0.0),
            edgecolor="#2b6cb0",
            linewidth=0.72,
            alpha=USER_CAP_EDGE_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(poly_bottom_caps)
    if USER_SHOW_MESH_FACES and plot_top_cap_tris.size:
        poly_top_caps = Poly3DCollection(
            plot_top_cap_tris,
            facecolor=(0.0, 0.0, 0.0, 0.0),
            edgecolor="#dd8a1c",
            linewidth=0.72,
            alpha=USER_CAP_EDGE_ALPHA,
            axlim_clip=True,
        )
        ax.add_collection3d(poly_top_caps)

    _render_particle_wireframe(ax, bottom_center, radius_mm, "#2b6cb0")
    _render_particle_wireframe(ax, top_center, radius_mm, "#dd8a1c")
    if USER_SHOW_CONTACT_RING_OVERLAY:
        _render_contact_circle(ax, bottom_cap_center, contact_radius_mm, axis, "#2b6cb0")
        _render_contact_circle(ax, top_cap_center, contact_radius_mm, axis, "#dd8a1c")
    ax.plot(
        [bottom_center[0], top_center[0]],
        [bottom_center[1], top_center[1]],
        [bottom_center[2], top_center[2]],
        linestyle="--",
        color="#9ca3af",
        linewidth=1.4,
    )

    label = "initial" if step == 0 else f"iteration {step}"
    ax.set_title(
        _snapshot_title(state, label=label)
        + f"\niteration={step}, gap={surface_gap * 1.0e6:.2f} um, overlap={'yes' if overlap else 'no'}"
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)

    plot_all = []
    if plot_side_tris.size:
        plot_all.append(plot_side_tris)
    if plot_bottom_cap_tris.size:
        plot_all.append(plot_bottom_cap_tris)
    if plot_top_cap_tris.size:
        plot_all.append(plot_top_cap_tris)
    plot_xyz = np.concatenate(plot_all, axis=0) if plot_all else np.empty((0, 3, 3))
    _set_axes_equal(ax, plot_xyz, [bottom_center, top_center], radius_mm)
    ax.grid(True, alpha=0.28)

    print("Interactive Case 2b mesh viewer")
    print("Use the mouse to rotate, zoom, and pan the figure window.")
    print(f"Step = {step}")
    print(f"Surface gap = {surface_gap * 1.0e6:.2f} um")
    print(f"Center distance = {center_distance * 1.0e3:.5f} mm")
    print(f"Sphere overlap = {'yes' if overlap else 'no'}")
    plt.show()


def open_mesh_viewer(
    *,
    step: int = 0,
    refinement: int = 1,
    elev: float = 23.0,
    azim: float = -90.0,
) -> None:
    state = _prepare_state(replace(separation_config(), refinement=refinement))
    _advance_state(state, n_steps=step)
    _show_state_interactive(state, step=step, elev=elev, azim=azim)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view-initial",
        action="store_true",
        help="Open an interactive window for the Case 2b initial mesh instead of running the simulation.",
    )
    parser.add_argument(
        "--view-step",
        type=int,
        default=None,
        help="Open an interactive window for the Case 2b mesh after this many separation steps.",
    )
    parser.add_argument("--refinement", type=int, default=USER_REFINEMENT, help="Mesh refinement for --view-initial.")
    parser.add_argument("--elev", type=float, default=USER_INTERACTIVE_ELEV_DEG, help="Camera elevation for --view-initial.")
    parser.add_argument("--azim", type=float, default=USER_INTERACTIVE_AZIM_DEG, help="Camera azimuth for --view-initial.")
    return parser


def _save_case2b_histories_panel(*, separation: dict[str, np.ndarray], out_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    axes[0].plot(separation["t_ms"], separation["d_over_r"], color="#bc4749", linewidth=1.8)
    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel(r"$D/R$")
    axes[0].set_title("Case 2b normalized gap")
    axes[0].grid(alpha=0.28)

    axes[1].plot(separation["t_ms"], separation["max_u"], color="#bc4749", linewidth=1.8)
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel("Max free speed [m/s]")
    axes[1].set_title("Case 2b free-surface speed")
    axes[1].grid(alpha=0.28)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_pitois_fig5_live_compare(
    *,
    separation_history: list[dict],
    out_dir: Path,
) -> Path | None:
    if not separation_history:
        return None
    fig_dir = out_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    separation = _history_arrays(separation_history)
    exp_x, exp_y = _pitois_fig5_dynamic_digitized()
    return _save_pitois_fig5_compare(
        exp_x=exp_x,
        exp_y=exp_y,
        sim_x=separation["d_over_r"],
        sim_y=separation["force_abs_mn"],
        out_path=fig_dir / "pitois2000_volumetric_separation_digitized_compare.png",
        xlim=(0.01, 0.30),
        ylim=(0.02, 2.0),
    )


def _write_pitois_fig5_comparison(
    *,
    separation_history: list[dict],
    out_dir: Path,
) -> None:
    fig_dir = out_dir / "fig"
    results_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    separation = _history_arrays(separation_history)
    exp_x, exp_y = _pitois_fig5_dynamic_digitized()

    _write_pitois_fig5_live_compare(separation_history=separation_history, out_dir=out_dir)
    _save_pitois_fig5_compare(
        exp_x=exp_x,
        exp_y=exp_y,
        sim_x=separation["d_over_r"],
        sim_y=separation["force_abs_mn"],
        out_path=fig_dir / "pitois2000_volumetric_separation_digitized_compare_zoom.png",
        xlim=(0.095, 0.110),
        ylim=(1.0e-4, 2.0),
        title="Pitois 2000 Fig. 5: zoom on the Case 2b overlap range",
    )
    _save_pitois_fig5_compare(
        exp_x=exp_x,
        exp_y=exp_y,
        sim_x=separation["d_over_r"],
        sim_y=separation["force_abs_mn"],
        out_path=fig_dir / "pitois2000_volumetric_separation_digitized_compare_tail_zoom.png",
        xlim=(0.13, 0.23),
        ylim=(0.035, 0.15),
        title="Pitois 2000 Fig. 5: digitized dynamic tail zoom",
    )
    _save_case2b_histories_panel(
        separation=separation,
        out_path=fig_dir / "pitois2000_volumetric_case2b_histories.png",
    )

    summary = {
        "separation": {
            "d_over_r_min": float(np.min(separation["d_over_r"])),
            "d_over_r_max": float(np.max(separation["d_over_r"])),
            "force_abs_mn_min": float(np.min(separation["force_abs_mn"])),
            "force_abs_mn_max": float(np.max(separation["force_abs_mn"])),
        },
        "digitized_fig5_dynamic": {
            "d_over_r": exp_x.tolist(),
            "force_mn": exp_y.tolist(),
        },
    }
    (results_dir / "pitois2000_volumetric_summary.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "pitois2000_fig5_dynamic_digitized.json").write_text(
        json.dumps(
            {
                "source": "Pitois 2000 Fig. 5 dynamic experimental points digitized from the local working copy",
                "d_over_r": exp_x.tolist(),
                "force_mn": exp_y.tolist(),
            },
            indent=2,
        )
    )


def _print_header(config: VolumetricPitoisConfig) -> None:
    total_cl_rings = int(config.contact_line_radial_rings) + int(config.extra_cl_radial_rings)
    print("=" * 72)
    print(f"  {config.title}")
    print("=" * 72)
    print(f"Refinement                = {config.refinement}")
    print(f"CL radial rings           = {config.contact_line_radial_rings}")
    print(f"Extra CL outer rings      = {config.extra_cl_radial_rings}")
    print(f"Extra CL axial split rings= {config.cl_extra_axial_layers}")
    print(f"Neck extra axial rings    = {config.neck_extra_axial_layers}")
    print(f"CL radial bias ratio      = {config.contact_line_radial_bias_ratio}")
    print(f"Sidewall axial stride     = {config.axial_ring_stride}")
    if total_cl_rings <= 1 and abs(float(config.contact_line_radial_bias_ratio) - 1.0) > 1.0e-12:
        print("CL radial bias active?    = no (need at least 2 total CL rings)")
    print(f"Particle radius           = {config.particle_radius * 1e3:.3f} mm")
    print(f"Bridge contact radius     = {config.target_cap_radius * 1e3:.3f} mm")
    print(f"Initial D/R               = {config.initial_d_over_r:.12f}")
    print(f"Pitois first point        = D/R {PITOIS_FIG5_FIRST_D_OVER_R:.12f}, |F| {PITOIS_FIG5_FIRST_FORCE_MN:.6f} mN")
    print(f"Surface tension           = {config.gamma:.4f} N/m")
    print(f"Viscosity                 = {config.mu_f:.3e} Pa s")
    print(f"Density                   = {config.rho_f:.3f} kg/m^3")
    print(f"Gravity included          = {'yes' if config.include_gravity else 'no'}")
    if config.include_gravity:
        print(f"Gravity acceleration      = {config.gravity_mps2:.3f} m/s^2")
    print(f"Contact angle assumption  = {config.contact_angle_deg:.1f} deg (literature-informed baseline)")
    print(f"CL Cox-Voinov law         = {'on' if config.use_cox_voinov_contact_line_law else 'off'}")
    print(f"Moving-sphere speed       = {config.cap_speed * 1e6:.3f} um/s")
    print(f"Relative speed            = {config.relative_speed * 1e6:.3f} um/s")
    print("Fixed sphere              = top sphere (balance side)")
    print("Moving sphere             = bottom sphere (stage side)")
    print(f"Integration substeps      = {config.integration_substeps}")
    print(f"Contact-radius samples    = {config.contact_radius_samples}")
    print(f"Volume constraint         = {'on' if config.enable_continuity_pressure else 'off'}")
    print(f"Final volume correction   = {'on' if config.enable_final_volume_correction else 'off'}")
    print("Laplace pressure scalar   = off (Heron surface tension force used)")
    print(f"NS pressure projection    = {'on' if config.enable_ns_pressure_projection else 'off'}")
    if config.enable_ns_pressure_projection:
        print(f"NS projection max iters   = {config.ns_pressure_projection_max_iters}")
        print(f"NS projection tolerance   = {config.ns_pressure_projection_tol:.1e}")
    print(f"Runtime split/merge remesh= {'on' if config.enable_runtime_remesh else 'off'}")
    if config.enable_runtime_remesh:
        print(f"Runtime remesh check step = {config.runtime_remesh_check_every_steps}")
        print(f"CL side-edge split limit  = {config.runtime_remesh_side_edge_max_fraction:.3f} r_CL")
    print(f"Acceleration workers      = {config.accel_workers}")
    print(
        "Array force backend      = "
        f"{config.array_force_backend if config.enable_array_force_backend else 'off'}"
    )
    print(f"Report cap viscous stress = {'on' if config.report_cap_viscous_stress else 'off'}")
    print(f"Live Fig.5 compare PNG   = {'on' if config.update_pitois_compare_every_step else 'off'}")
    print(f"Volume stop tolerance    = {100.0 * config.volume_constraint_stop_fraction:.4f} %")
    print(f"Mesh-quality guard       = {'on' if config.enable_mesh_quality_guard else 'off'}")
    print(f"No-swirl enforcement      = {'on' if config.enforce_no_swirl else 'off'}")
    print(f"Radial/axial velocity avg = {'on' if config.average_velocity_radial_axial else 'off'}")
    if config.dt > 0.0:
        print("Adaptive dt               = off (using USER_DT_S)")
        print(f"dt                        = {config.dt:.3e} s")
    else:
        print(f"Adaptive dt               = {'on' if config.enable_adaptive_dt else 'off'}")
        print(f"dt ceiling                = {config.adaptive_dt_max_s:.3e} s")
        print(f"dt floor                  = {config.adaptive_dt_min_s:.3e} s")
        print(f"Capillary dt safety       = {config.adaptive_dt_capillary_safety:.3f}")
        print(f"Mesh disp. fraction       = {config.adaptive_dt_mesh_displacement_frac:.3f}")
    print(f"Steps                     = {config.n_steps}")
    if config.dt > 0.0:
        print(f"Run time                  = {config.dt * config.n_steps:.3f} s")
    else:
        print(f"Run time ceiling          = {config.adaptive_dt_max_s * config.n_steps:.3f} s")
    print("=" * 72)


def _print_summary(history: list[dict], config: VolumetricPitoisConfig) -> None:
    fixed_force_mn = np.array([float(row["fixed_force_axial"]) for row in history], dtype=float) * 1.0e3
    d_over_r = np.array([float(row["d_over_r"]) for row in history], dtype=float)
    max_u = np.array([float(row["max_free_speed"]) for row in history], dtype=float)
    print("Separation run complete")
    print(f"D/R range                 = {float(np.min(d_over_r)):.6f} - {float(np.max(d_over_r)):.6f}")
    print(f"Fixed-sphere force [mN]   = {float(np.min(fixed_force_mn)):.6e} - {float(np.max(fixed_force_mn)):.6e}")
    print(f"Final max free speed      = {float(max_u[-1]):.6e} m/s")


def _print_step_status(
    record_row: dict,
    *,
    completed_step: int,
    total_steps: int,
    step_dt: float,
    dt_limiter: str,
    snapshot_volume_ul: float,
    initial_snapshot_volume_ul: float,
) -> None:
    rel_snapshot_volume_error = (snapshot_volume_ul - initial_snapshot_volume_ul) / max(
        initial_snapshot_volume_ul, 1.0e-30
    )
    print(
        f"Current step              = {completed_step}/{total_steps}",
        flush=True,
    )
    print(
        f"Current D/R               = {float(record_row['d_over_r']):.12f}",
        flush=True,
    )
    print(
        f"Adaptive dt               = {step_dt:.6e} s ({dt_limiter})",
        flush=True,
    )
    print(
        f"Snapshot vol. error       = {rel_snapshot_volume_error:+.6e} ({100.0 * rel_snapshot_volume_error:+.4f} %)",
        flush=True,
    )
    print(
        f"Snapshot .msh volume      = {snapshot_volume_ul:.6f} uL",
        flush=True,
    )
    print(
        "F each step              = "
        f"fixed_axial {1.0e3 * float(record_row['fixed_force_axial']):+.6e} mN, "
        f"moving_axial {1.0e3 * float(record_row['moving_force_axial']):+.6e} mN, "
        f"|fixed| {1.0e3 * float(record_row['fixed_force_mag']):.6e} mN, "
        f"|moving| {1.0e3 * float(record_row['moving_force_mag']):.6e} mN",
        flush=True,
    )
    print(
        "NS projection            = "
        f"iters {int(record_row['ns_projection_iterations'])}, "
        f"div_l2 {float(record_row['ns_divergence_l2']):.6e} -> "
        f"{float(record_row['ns_projected_divergence_l2']):.6e} 1/s, "
        f"p_l2 {float(record_row['ns_pressure_projection_l2']):.6e} Pa",
        flush=True,
    )


def run_motion_case(
    config: VolumetricPitoisConfig,
    *,
    out_dir: Path | None = None,
    save_fig: bool = True,
    save_results: bool = True,
    verbose: bool = True,
    interactive_step: int | None = None,
    interactive_elev: float = USER_INTERACTIVE_ELEV_DEG,
    interactive_azim: float = USER_INTERACTIVE_AZIM_DEG,
) -> list[dict]:
    if out_dir is None:
        out_dir = OUT_ROOT

    if verbose:
        _print_header(config)

    state = _prepare_state(config)
    config = state.config
    _assert_fixed_topology(state)
    initial_snapshot_volume_ul = 1.0e9 * _snapshot_msh_volume_m3(state)
    history: list[dict] = []
    fig_dir = out_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    motion_mesh_dir = fig_dir / "separation"

    if save_fig:
        _render_motion_mesh_pngs(state, motion_dir=motion_mesh_dir, label="initial", step=0)

    initial_row = _step_record(state, step=0, t=0.0)
    history.append(initial_row)
    if verbose:
        _print_step_status(
            initial_row,
            completed_step=0,
            total_steps=config.n_steps,
            step_dt=0.0,
            dt_limiter="initial",
            snapshot_volume_ul=initial_snapshot_volume_ul,
            initial_snapshot_volume_ul=initial_snapshot_volume_ul,
        )
    if save_fig and bool(getattr(config, "update_pitois_compare_every_step", False)):
        _write_pitois_fig5_live_compare(separation_history=history, out_dir=out_dir)

    for step in range(config.n_steps):
        step_dt = _select_physical_dt(state)
        substeps = max(1, int(config.integration_substeps))
        sub_dt = float(step_dt) / float(substeps)
        for _substep in range(substeps):
            if bool(getattr(config, "enable_mesh_quality_guard", False)):
                _advance_substep_with_quality_guard(state, dt=sub_dt)
            else:
                _advance_one_substep(state, dt=sub_dt)
        state.elapsed_time_s += float(step_dt)

        completed_step = step + 1
        should_record_step = completed_step % max(1, config.record_every) == 0
        record_row = None
        update_live_compare = save_fig and bool(getattr(config, "update_pitois_compare_every_step", False))
        if verbose or should_record_step or update_live_compare:
            record_row = _step_record(state, step=completed_step, t=state.elapsed_time_s)
        if verbose:
            snapshot_volume_ul = 1.0e9 * _snapshot_msh_volume_m3(state)
            _print_step_status(
                record_row,
                completed_step=completed_step,
                total_steps=config.n_steps,
                step_dt=step_dt,
                dt_limiter=state.last_dt_limiter,
                snapshot_volume_ul=snapshot_volume_ul,
                initial_snapshot_volume_ul=initial_snapshot_volume_ul,
            )
        if should_record_step:
            history.append(record_row)
            live_history = history
        else:
            live_history = history + [record_row]
        if update_live_compare:
            _write_pitois_fig5_live_compare(separation_history=live_history, out_dir=out_dir)

        if interactive_step is not None and completed_step == max(0, int(interactive_step)):
            _show_state_interactive(
                state,
                step=completed_step,
                elev=interactive_elev,
                azim=interactive_azim,
            )

        if (
            save_fig
            and config.mesh_snapshot_every > 0
            and completed_step % config.mesh_snapshot_every == 0
            and completed_step < config.n_steps
        ):
            _render_motion_mesh_pngs(
                state,
                motion_dir=motion_mesh_dir,
                label=f"iteration {completed_step}",
                step=completed_step,
            )

    if config.n_steps > 0:
        final_step = int(config.n_steps)
        if not history or int(history[-1]["step"]) != final_step:
            history.append(_step_record(state, step=final_step, t=state.elapsed_time_s))

    if save_fig:
        _render_motion_mesh_pngs(state, motion_dir=motion_mesh_dir, label="final", step=config.n_steps)
        _render_history_figures(
            history,
            config,
            out_dir,
            initial_snapshot_volume_m3=1.0e-9 * initial_snapshot_volume_ul,
        )

    if save_results:
        _save_history(history, config, out_dir)

    if verbose and history:
        print(f"Initial bridge volume     = {initial_snapshot_volume_ul:.6f} uL")
        _print_summary(history, config)

    return history


def main(argv: list[str] | None = None) -> None:
    args = _build_cli().parse_args(argv)
    if args.view_initial or args.view_step is not None:
        open_mesh_viewer(
            step=0 if args.view_step is None else max(0, int(args.view_step)),
            refinement=args.refinement,
            elev=args.elev,
            azim=args.azim,
        )
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    separation_history = run_motion_case(
        separation_config(),
        out_dir=OUT_ROOT,
        save_fig=True,
        save_results=True,
        verbose=True,
        interactive_step=max(0, int(USER_INTERACTIVE_STEP)) if USER_OPEN_INTERACTIVE_WINDOW else None,
        interactive_elev=USER_INTERACTIVE_ELEV_DEG,
        interactive_azim=USER_INTERACTIVE_AZIM_DEG,
    )
    if separation_history:
        _write_pitois_fig5_comparison(
            separation_history=separation_history,
            out_dir=OUT_ROOT,
        )
    else:
        print("No recorded history rows were written; skipping the Fig. 5 comparison output.")


if __name__ == "__main__":
    main()
