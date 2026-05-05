"""Computation-only solver for Case 2b Gmsh Pitois-2000 separation.

This module owns state/configuration, mesh input, physics forces, pressure,
contact-line motion, volume handling, and time stepping.  Plotting, .msh/CSV/JSON
output, Fig. 5 digitized experiment data, and CLI code live in
``Case_2b_axisym_pitois2000_separation_Gmsh v2.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases_dynamic.liquid_bridge_equilibrium.Case_5_volumetric_stress_equilibrium_particle_particle_bridge_benchmark import (
    HEX_TETS,
    PRISM_TETS,
)
from ddgclib.dynamic_integrators._integrators_dynamic import _move, _recompute_duals
from ddgclib.operators.surface_tension import dual_area_heron, surface_tension_force
from ddgclib.operators.stress import dual_area_vector, dual_volume

class _DisplayVertex:
    def __init__(self, coords):
        self.x_a = np.asarray(coords, dtype=float)
        self.nn: set[object] = set()
        self.boundary = False

    def connect(self, other) -> None:
        self.nn.add(other)
        other.nn.add(self)


class _DisplayVertexStore(dict):
    def __missing__(self, key):
        vertex = _DisplayVertex(key)
        self[key] = vertex
        return vertex

    def __iter__(self):
        return iter(self.values())

    def move(self, vertex, pos):
        old_key = None
        for key, value in list(self.items()):
            if value is vertex:
                old_key = key
                break
        if old_key is not None:
            del self[old_key]
        vertex.x_a = np.asarray(pos, dtype=float)
        self[tuple(float(x) for x in vertex.x_a[:3])] = vertex


class Complex:
    def __init__(self, _dim: int, domain=None):
        self.domain = domain
        self.V = _DisplayVertexStore()


HARDCODED_INITIAL_MSH_PATH = (
    Path(__file__).resolve().parent
    / "out"
    / "Case_2b_axisym_initialshape_Gmsh"
    / "fig"
    / "mesh_iter0012.msh"
)

# ---------------------------------------------------------------------------
# USER CONTROLS
# Edit these values at the top of the file when you want to change the
# separation run or open an interactive viewer without touching the code below.
# ---------------------------------------------------------------------------
USER_REFINEMENT = 1
# Exact number of radial rings used near the contact line.
USER_CONTACT_LINE_RADIAL_RINGS = 2
# Additional outer-band rings near the contact line. Keep this consistent with
# the initialshape case so imported profiles retain the refined CL topology.
USER_EXTRA_CL_RADIAL_RINGS = 4
# 1 keeps all sidewall axial rings.  The neck has high curvature, so keep the
# full axial stack for the physical axisymmetric case.
USER_SIDEWALL_AXIAL_RING_STRIDE = 1
USER_NECK_EXTRA_AXIAL_LAYERS = 12
USER_CL_EXTRA_AXIAL_LAYERS = 6
# Radial gap ratio. Larger r is finer; adjacent gap ratio is 1.1.
USER_CL_RADIAL_BIAS_RATIO = 1.1
# Max separation iterations. Keep at 1 for the requested smoke check; increase
# here when you want a longer Fig. 5 sweep.
USER_TOTAL_STEPS = 2000
# Time-step rule:
#   USER_DT_S > 0: fixed user step.
#   USER_DT_S < 0: physical step, dt = min(dt_CL, dt_capillary, dt_mesh).
# This keeps the automatic step computed from numerical/physical limits only.
USER_DT_S = 0.0005
USER_ENABLE_ADAPTIVE_DT = USER_DT_S < 0.0
# Negative means no artificial ceiling; the selected value is the right_dt
# physical minimum. Positive USER_DT_S remains the fixed step.
USER_ADAPTIVE_DT_MAX_S = -1.0 if USER_DT_S < 0.0 else USER_DT_S
# Do not let a user floor override the physical capillary/mesh/contact-line
# limit. The selector only keeps a tiny numerical guard against exactly zero.
USER_ADAPTIVE_DT_MIN_S = 0.0
USER_ADAPTIVE_DT_CAPILLARY_SAFETY = 0.25
USER_ADAPTIVE_DT_MESH_DISPLACEMENT_FRAC = 0.15

USER_RECORD_EVERY_STEPS = 1
USER_MESH_SNAPSHOT_EVERY_STEPS = 10
USER_INCLUDE_GRAVITY = True
USER_GRAVITY_MPS2 = 9.81
USER_INTEGRATION_SUBSTEPS = 1
USER_CONTACT_RADIUS_SAMPLES = 128
USER_GEOMETRIC_VOLUME_CORRECTION_MAX_ITERS = 12
USER_GEOMETRIC_VOLUME_CORRECTION_REL_TOL = 1.0e-6
USER_MAX_ACCELERATION = 5.0e-3
USER_ABORT_ON_NONFINITE_STATE = True
USER_ENABLE_GMSH_GEOMETRIC_VOLUME_CORRECTION = False
USER_GMSH_GEOMETRIC_VOLUME_CORRECTION_TRIGGER_REL = 5.0e-3
USER_MAX_VOLUME_REL_ERROR_FOR_ABORT = 0.25
USER_ALLOW_CONTACT_LINE_GROWTH = True
USER_ENABLE_DYNAMIC_CONTACT_ANGLE = True
USER_DYNAMIC_CONTACT_ANGLE_MAX_DEG = 25.0
USER_DYNAMIC_CONTACT_ANGLE_MIN_DEG = 0.0
USER_CONTACT_LINE_MAX_SLIDE_UM = 5.0
USER_CONTACT_LINE_CONTINUATION_WEIGHT = 0.02
USER_CONTACT_LINE_FIT_RINGS = 4
USER_CONTACT_LINE_COX_MACRO_LENGTH_M = 1.0e-3
USER_CONTACT_LINE_COX_SLIP_LENGTH_M = 2.0e-9
# The run loops honor this value now. Keep the default serial because this
# operator is Python/object-graph heavy; 8 ThreadPool workers tested slower on
# the current axisymmetric force-cache path. Raise manually after benchmarking.
USER_ACCEL_WORKERS = 1
USER_ENFORCE_NO_SWIRL = True
USER_ENFORCE_FULL_AXISYMMETRY = True


def _pitois_eq6_wetted_radius(
    *,
    particle_radius_m: float,
    bridge_volume_m3: float,
    initial_d_over_r: float,
) -> float:
    """Compute the wetted/contact radius b from Pitois et al. Eq. [6].

    Source: Pitois, Moucheront, Chateau, J. Colloid Interface Sci. 231
    (2000), Eq. [6].  The paper gives the bridge volume and sphere radius,
    not a separate fixed contact radius.  Eq. [6] gives
    the cylindrical/flat-profile bridge volume

        V = (pi R / 2) * (H(b)^2 - D^2),  H(b) = D + b^2 / R,

    so the contact/wetted radius b is computed from R, V, and the starting
    separation D = (D/R)R.
    """
    radius = float(particle_radius_m)
    volume = float(bridge_volume_m3)
    gap = float(initial_d_over_r) * radius
    if radius <= 0.0 or volume <= 0.0:
        raise ValueError("Pitois Eq. [6] needs positive particle radius and bridge volume.")
    height_at_contact = math.sqrt(gap * gap + 2.0 * volume / (math.pi * radius))
    return math.sqrt(max(radius * (height_at_contact - gap), 0.0))


@dataclass(frozen=True)
class VolumetricPitoisConfig:
    name: str
    title: str
    refinement: int = USER_REFINEMENT
    contact_line_radial_rings: int = USER_CONTACT_LINE_RADIAL_RINGS
    extra_cl_radial_rings: int = USER_EXTRA_CL_RADIAL_RINGS
    axial_ring_stride: int = USER_SIDEWALL_AXIAL_RING_STRIDE
    neck_extra_axial_layers: int = USER_NECK_EXTRA_AXIAL_LAYERS
    cl_extra_axial_layers: int = USER_CL_EXTRA_AXIAL_LAYERS
    contact_line_radial_bias_ratio: float = USER_CL_RADIAL_BIAS_RATIO
    particle_radius: float = 4.0e-3
    # Starting normalized surface gap used by the Case_2b separation setup.
    initial_d_over_r: float = 0.014484537138212631
    # Bridge volume: V = 1.1 mm^3 = 1.10e-9 m^3.
    initial_bridge_volume_m3: float = 1.10e-9
    # Pitois gives R, V, and D/R, not a contact radius directly.
    # Therefore the default target_cap_radius is not a hard-coded 1.54 mm value;
    # it is computed as the wetted radius b from Pitois Eq. [6].
    use_pitois_eq6_contact_radius: bool = True
    target_cap_radius: float = 0.0
    gamma: float = 2.10e-2
    # Pitois Fig. 5 liquid 2 is the 100 Pa s PDMS oil.
    mu_f: float = 1.0e2
    rho_f: float = 965.0
    dt: float = USER_DT_S
    enable_adaptive_dt: bool = USER_ENABLE_ADAPTIVE_DT
    adaptive_dt_max_s: float = USER_ADAPTIVE_DT_MAX_S
    adaptive_dt_min_s: float = USER_ADAPTIVE_DT_MIN_S
    adaptive_dt_capillary_safety: float = USER_ADAPTIVE_DT_CAPILLARY_SAFETY
    adaptive_dt_mesh_displacement_frac: float = USER_ADAPTIVE_DT_MESH_DISPLACEMENT_FRAC
    n_steps: int = USER_TOTAL_STEPS
    cap_speed: float = 5.0e-6
    max_acceleration: float = USER_MAX_ACCELERATION
    integration_substeps: int = USER_INTEGRATION_SUBSTEPS
    record_every: int = USER_RECORD_EVERY_STEPS
    mesh_snapshot_every: int = USER_MESH_SNAPSHOT_EVERY_STEPS
    display_motion_scale: float = 4000.0
    # Pressure is reserved for the internal flow/continuity field; liquid-gas
    # capillarity is handled only through the Heron surface-tension force.
    internal_pressure_pa: float = 0.0
    continuity_pressure_pa: float = 0.0
    pressure_scale: float = 1.0
    include_gravity: bool = USER_INCLUDE_GRAVITY
    gravity_mps2: float = USER_GRAVITY_MPS2
    contact_radius_samples: int = USER_CONTACT_RADIUS_SAMPLES
    enable_gmsh_geometric_volume_correction: bool = USER_ENABLE_GMSH_GEOMETRIC_VOLUME_CORRECTION
    gmsh_geometric_volume_correction_trigger_rel: float = USER_GMSH_GEOMETRIC_VOLUME_CORRECTION_TRIGGER_REL
    max_volume_rel_error_for_abort: float = USER_MAX_VOLUME_REL_ERROR_FOR_ABORT
    allow_contact_line_growth: bool = USER_ALLOW_CONTACT_LINE_GROWTH
    enable_dynamic_contact_angle: bool = USER_ENABLE_DYNAMIC_CONTACT_ANGLE
    dynamic_contact_angle_max_deg: float = USER_DYNAMIC_CONTACT_ANGLE_MAX_DEG
    dynamic_contact_angle_min_deg: float = USER_DYNAMIC_CONTACT_ANGLE_MIN_DEG
    contact_line_max_slide_um: float = USER_CONTACT_LINE_MAX_SLIDE_UM
    contact_line_continuation_weight: float = USER_CONTACT_LINE_CONTINUATION_WEIGHT
    contact_line_fit_rings: int = USER_CONTACT_LINE_FIT_RINGS
    contact_line_cox_macro_length_m: float = USER_CONTACT_LINE_COX_MACRO_LENGTH_M
    contact_line_cox_slip_length_m: float = USER_CONTACT_LINE_COX_SLIP_LENGTH_M
    accel_workers: int = USER_ACCEL_WORKERS
    enforce_no_swirl: bool = USER_ENFORCE_NO_SWIRL
    # Literature-informed baseline for silicone oil on clean oxide-like solids
    # in air; this is stored explicitly for the Fig. 5 separation setup.
    contact_angle_deg: float = 0.0

    @property
    def relative_speed(self) -> float:
        return self.cap_speed

    def __post_init__(self) -> None:
        if bool(self.use_pitois_eq6_contact_radius):
            object.__setattr__(
                self,
                "target_cap_radius",
                _pitois_eq6_wetted_radius(
                    particle_radius_m=self.particle_radius,
                    bridge_volume_m3=self.initial_bridge_volume_m3,
                    initial_d_over_r=self.initial_d_over_r,
                ),
            )
        if float(self.target_cap_radius) <= 0.0:
            raise ValueError("target_cap_radius must be positive.")


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
    simulated_force_n: float = 0.0
    top_sphere_Fp_n: float = 0.0
    top_sphere_Fs_n: float = 0.0
    top_sphere_Fv_n: float = 0.0
    top_sphere_Fcl_n: float = 0.0
    top_sphere_Ftot_n: float = 0.0
    continuity_pressure_scalar: float = 0.0
    continuity_pressure_map: dict[int, float] = field(default_factory=dict)
    elapsed_time_s: float = 0.0
    last_step_dt: float = USER_DT_S
    last_bottom_contact_line_speed: float = 0.0
    last_top_contact_line_speed: float = 0.0
    last_dt_limit_cl: float = USER_DT_S
    last_dt_limit_capillary: float = USER_DT_S
    last_dt_limit_mesh: float = USER_DT_S
    last_dt_limiter: str = "fixed"
    last_contact_line_volume_delta_m3: float = 0.0
    contact_line_slide_velocity_map: dict[int, float] = field(default_factory=dict)
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
    return list(state.HC.V)


def _assert_fixed_topology(state: VolumetricPitoisState) -> None:
    if not bool(getattr(state, "topology_frozen", False)):
        return
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        current_volume_export_ids = tuple(id(vertex) for vertex in state.volume_export_vertices)
        if current_volume_export_ids != state.frozen_volume_vertex_id_order:
            raise RuntimeError("Fixed topology violated: Gmsh volume vertex IDs changed during time stepping")
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


def _side_surface_tri_indices_from_rings(state: VolumetricPitoisState) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
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
                triangles.append((a, b, c))
                triangles.append((a, c, d))
            else:
                triangles.append((a, b, d))
                triangles.append((b, c, d))
    return np.asarray(triangles, dtype=int) if triangles else np.empty((0, 3), dtype=int)


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


def _volume_vertex_idx(state: VolumetricPitoisState, vertex) -> int:
    return int(state.volume_export_node_ids[id(vertex)])


def _freeze_volume_msh_topology(state: VolumetricPitoisState) -> None:
    _ensure_volume_export_node_ids(state)

    tets: list[tuple[int, int, int, int]] = []
    for lower_center, upper_center, lower_radial_rings, upper_radial_rings in zip(
        state.layer_centers[:-1],
        state.layer_centers[1:],
        state.layer_rings[:-1],
        state.layer_rings[1:],
    ):
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
            prism_idx = [_volume_vertex_idx(state, vertex) for vertex in prism]
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
                hexahedron_idx = [_volume_vertex_idx(state, vertex) for vertex in hexahedron]
                for tet in HEX_TETS:
                    tets.append(tuple(hexahedron_idx[j] for j in tet))

    if tets:
        points = _volume_points_array(state)
        oriented: list[tuple[int, int, int, int]] = []
        for tet in tets:
            a, b, c, d = tet
            pa = points[int(a)]
            pb = points[int(b)]
            pc = points[int(c)]
            pd = points[int(d)]
            vol = float(np.dot(pa - pd, np.cross(pb - pd, pc - pd))) / 6.0
            oriented.append((b, a, c, d) if vol < 0.0 else tet)
        tets = oriented

    state.volume_export_tets = (
        np.asarray(tets, dtype=int) if tets else np.empty((0, 4), dtype=int)
    )


def _freeze_surface_topology(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "topology_frozen", False)):
        _assert_fixed_topology(state)
        return
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        if not state.surface_export_side_tris.size:
            state.surface_export_side_tris = _side_surface_tri_indices_from_rings(state)
        state.frozen_layer_ring_id_structure = _layer_ring_id_structure(state)
        state.frozen_surface_vertex_id_order = tuple(id(vertex) for vertex in state.surface_export_vertices)
        state.frozen_volume_vertex_id_order = tuple(id(vertex) for vertex in state.volume_export_vertices)
        state.topology_frozen = True
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


def separation_config() -> VolumetricPitoisConfig:
    return VolumetricPitoisConfig(
        name="Case_2b_axisym_pitois2000_separation_Gmsh",
        title="Case 2b axisym: volumetric pre-bridged separation",
    )


def _cap_radius(vertices: list) -> float:
    return max((float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float))) for v in vertices), default=1.0)


def _move_vertices_batch(vertices, targets, HC, bV) -> None:
    """Move a vertex group without transient cache collisions.

    Some ring updates are permutations: a vertex can be moved onto another
    vertex's current coordinates before that second vertex is moved. Doing
    those updates one-by-one causes the vertex cache to evict the later vertex from
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

    for v, tmp in zip(vertices, staged):
        _move(v, tmp, HC, bV)
    for v, target in zip(vertices, targets):
        _move(v, target, HC, bV)


def _base_update_duals_and_masses(state: VolumetricPitoisState) -> None:
    if hasattr(state, "_dual_area_vector_cache"):
        delattr(state, "_dual_area_vector_cache")
    _recompute_duals(state.HC)
    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if vertices and tets.size:
        points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
        lumped = np.zeros(len(vertices), dtype=float)
        for a, b, c, d in tets:
            ia = int(a)
            ib = int(b)
            ic = int(c)
            idd = int(d)
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]
            pd = points[idd]
            volume = abs(float(np.dot(pa - pd, np.cross(pb - pd, pc - pd))) / 6.0)
            share = 0.25 * volume
            lumped[ia] += share
            lumped[ib] += share
            lumped[ic] += share
            lumped[idd] += share
        for vertex, mass in zip(vertices, lumped):
            vertex.m = max(float(mass), 1.0e-12)
        return
    for v in state.HC.V:
        v.m = max(float(dual_volume(v, state.HC, dim=3)), 1.0e-12)


def _resolve_pressure_local(v, pressure_model=None, HC=None, dim: int = 3) -> float:
    if pressure_model is not None:
        return float(pressure_model(v, HC=HC, dim=dim))
    p_val = getattr(v, "p", 0.0)
    if np.ndim(p_val) == 0:
        return float(p_val)
    return float(np.asarray(p_val).ravel()[0])


def _cached_dual_area_vector(v, v_j, *, state: VolumetricPitoisState, dim: int) -> np.ndarray:
    cache = getattr(state, "_dual_area_vector_cache", None)
    if cache is None:
        cache = {}
        state._dual_area_vector_cache = cache
    key = (id(v), id(v_j), int(dim))
    cached = cache.get(key)
    if cached is not None:
        return cached
    A_ij = dual_area_vector(v, v_j, state.HC, dim)
    cache[key] = A_ij
    return A_ij


def _gmsh_surface_heron_force_map(state: VolumetricPitoisState) -> dict[int, np.ndarray]:
    cached = getattr(state, "_gmsh_surface_heron_force_cache", None)
    if cached is not None:
        return cached

    side_tris = np.asarray(getattr(state, "surface_export_side_tris", np.empty((0, 3), dtype=int)), dtype=int)
    surface_vertices = list(getattr(state, "surface_export_vertices", []))
    if side_tris.size == 0 or not surface_vertices:
        state._gmsh_surface_heron_force_cache = {}
        return state._gmsh_surface_heron_force_cache

    used_indices = sorted({int(idx) for tri in side_tris for idx in tri})
    surface_clone = Complex(3, domain=None)
    clone_by_idx = {
        idx: surface_clone.V[tuple(float(x) for x in np.asarray(surface_vertices[idx].x_a[:3], dtype=float))]
        for idx in used_indices
    }
    for tri in side_tris:
        a, b, c = [int(idx) for idx in tri]
        clone_by_idx[a].connect(clone_by_idx[b])
        clone_by_idx[b].connect(clone_by_idx[c])
        clone_by_idx[c].connect(clone_by_idx[a])

    contact_ids = {id(v) for v in state.bottom_contact_ring + state.top_contact_ring}
    for idx, surface_v in clone_by_idx.items():
        source_v = surface_vertices[idx]
        surface_v.boundary = id(source_v) in contact_ids
        surface_v.u = np.zeros(3, dtype=float)
        surface_v.p = 0.0
        surface_v.m = max(float(dual_area_heron(surface_v)), 1.0e-12)
        surface_v.phase = 0
        surface_v.is_interface = not surface_v.boundary
        surface_v.interface_phases = frozenset({0, 1}) if surface_v.is_interface else frozenset()

    force_map = {
        id(surface_vertices[idx]): surface_tension_force(surface_v, gamma=state.config.gamma, dim=3)
        for idx, surface_v in clone_by_idx.items()
    }
    state._gmsh_surface_heron_force_cache = force_map
    return force_map


def _force_Fs_Heron(v, *, dim: int, state: VolumetricPitoisState) -> np.ndarray:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        force = _gmsh_surface_heron_force_map(state).get(id(v))
        if force is None:
            return np.zeros(dim)
        return np.asarray(force[:dim], dtype=float)
    if getattr(v, "is_interface", False) and state.mps is not None:
        phases = getattr(v, "interface_phases", frozenset())
        if len(phases) >= 2:
            phase_list = sorted(phases)
            gamma = state.mps.get_gamma_pair(phase_list[0], phase_list[1])
            if gamma != 0.0:
                return surface_tension_force(v, gamma=gamma, dim=dim)
    return np.zeros(dim)


def _force_Fp_Fv_Fs_components(
    v,
    *,
    dim: int,
    state: VolumetricPitoisState,
    pressure_model=None,
) -> SimpleNamespace:
    Fp = np.zeros(dim)
    Fv = np.zeros(dim)
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        Fv = _gmsh_viscous_force_map(state).get(id(v), np.zeros(3, dtype=float))[:dim]
    elif state.HC is not None and hasattr(v, "vd"):
        p_i = _resolve_pressure_local(v, pressure_model, state.HC, dim)
        u_i = v.u[:dim]
        x_i = v.x_a[:dim]
        mu = state.mps.get_mu(v.phase) if state.mps is not None else 0.0
        for v_j in v.nn:
            try:
                A_ij = _cached_dual_area_vector(v, v_j, state=state, dim=dim)
            except (KeyError, IndexError, ValueError, RuntimeError, ZeroDivisionError):
                continue
            p_j = _resolve_pressure_local(v_j, pressure_model, state.HC, dim)
            Fp -= 0.5 * (p_i + p_j) * A_ij
            delta_u = v_j.u[:dim] - u_i
            d_ij = v_j.x_a[:dim] - x_i
            d_norm = np.linalg.norm(d_ij)
            if d_norm < 1.0e-30:
                continue
            d_hat = d_ij / d_norm
            Fv += (mu / d_norm) * delta_u * np.dot(d_hat, A_ij)
    Fp += _pressure_boundary_traction(v, state=state, pressure_model=pressure_model)[:dim]
    Fs = _force_Fs_Heron(v, dim=dim, state=state)
    return SimpleNamespace(
        Fp=np.asarray(Fp, dtype=float),
        Fv=np.asarray(Fv, dtype=float),
        Fs=np.asarray(Fs, dtype=float),
    )


def _base_update_pressure_scalar(state: VolumetricPitoisState) -> None:
    # Pressure split:
    #   p_cont: zero-mean continuity pressure correction.
    # Liquid-gas capillarity itself still enters through Fs_Heron.
    state.pressure_scalar = (
        float(getattr(state.config, "internal_pressure_pa", 0.0))
        + float(getattr(state, "continuity_pressure_scalar", 0.0))
    )


def _clip_acceleration(accel: np.ndarray, state: VolumetricPitoisState) -> np.ndarray:
    accel = np.asarray(accel, dtype=float)
    if not np.all(np.isfinite(accel)):
        return np.zeros_like(accel, dtype=float)
    accel_norm = float(np.linalg.norm(accel))
    if accel_norm > state.config.max_acceleration:
        accel = accel * (state.config.max_acceleration / accel_norm)
    return accel


def _state_finite_summary(state: VolumetricPitoisState) -> tuple[bool, str]:
    vertices = list(getattr(state, "volume_export_vertices", [])) or list(getattr(state.HC, "V", []))
    if not vertices:
        return True, "no vertices"

    coords = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
    velocities = np.asarray([np.asarray(getattr(v, "u", np.zeros(3, dtype=float))[:3], dtype=float) for v in vertices], dtype=float)
    if not np.all(np.isfinite(coords)):
        bad = np.argwhere(~np.isfinite(coords))[0]
        return False, f"non-finite coordinate at vertex {int(bad[0])}, component {int(bad[1])}"
    if not np.all(np.isfinite(velocities)):
        bad = np.argwhere(~np.isfinite(velocities))[0]
        return False, f"non-finite velocity at vertex {int(bad[0])}, component {int(bad[1])}"

    max_coord = float(np.max(np.abs(coords)))
    max_vel = float(np.max(np.linalg.norm(velocities, axis=1)))
    volume = float(_snapshot_msh_volume_m3(state))
    if not np.isfinite(volume):
        return False, "non-finite .msh volume"
    target_volume = float(getattr(state, "target_snapshot_volume_m3", volume))
    if np.isfinite(target_volume) and target_volume > 0.0:
        rel_volume_error = abs(volume - target_volume) / max(target_volume, 1.0e-30)
        if rel_volume_error > float(getattr(state.config, "max_volume_rel_error_for_abort", 0.25)):
            return False, (
                f".msh volume drift: current = {volume:.6e} m^3, "
                f"target = {target_volume:.6e} m^3, rel = {rel_volume_error:.6e}"
            )
    if not np.isfinite(max_coord) or max_coord > 1.0:
        return False, f"coordinate blow-up: max |x| = {max_coord:.6e} m"
    if not np.isfinite(max_vel) or max_vel > 1.0:
        return False, f"velocity blow-up: max |u| = {max_vel:.6e} m/s"
    return True, f"max |x| = {max_coord:.6e} m, max |u| = {max_vel:.6e} m/s, volume = {volume:.6e} m^3"


def _assert_state_finite(state: VolumetricPitoisState, *, where: str) -> None:
    ok, detail = _state_finite_summary(state)
    if ok:
        return
    message = f"Non-finite/unstable state after {where}: {detail}"
    if bool(USER_ABORT_ON_NONFINITE_STATE):
        raise RuntimeError(message)
    print(f"WARNING: {message}", flush=True)


def _boundary_vertex_lookup(state: VolumetricPitoisState) -> dict[tuple[float, float, float], object]:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        return {
            tuple(np.round(np.asarray(v.x_a[:3], dtype=float), 12)): v
            for v in state.surface_export_vertices
        }
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


def _boundary_area_normal_maps_cached(
    state: VolumetricPitoisState,
) -> tuple[list[object], dict[int, float], dict[int, np.ndarray]]:
    cached = getattr(state, "_boundary_area_normal_cache", None)
    if cached is not None:
        return cached
    cached = _boundary_dual_areas_and_normals(state)
    state._boundary_area_normal_cache = cached
    return cached


def _pressure_boundary_traction(
    v,
    *,
    state: VolumetricPitoisState,
    pressure_model=None,
) -> np.ndarray:
    if pressure_model is None:
        pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
    _vertices, area_map, normal_map = _boundary_area_normal_maps_cached(state)
    area = float(area_map.get(id(v), 0.0))
    normal = normal_map.get(id(v))
    if area <= 0.0 or normal is None:
        return np.zeros(3, dtype=float)
    pressure = float(pressure_model(v, HC=state.HC, dim=3))
    return pressure * area * np.asarray(normal, dtype=float)


def _gmsh_viscous_force_map(state: VolumetricPitoisState) -> dict[int, np.ndarray]:
    cached = getattr(state, "_gmsh_viscous_force_cache", None)
    if cached is not None:
        return cached

    vertices = list(getattr(state, "volume_export_vertices", [])) or list(state.HC.V)
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    force_map = {id(vertex): np.zeros(3, dtype=float) for vertex in vertices}
    if not vertices or tets.size == 0:
        state._gmsh_viscous_force_cache = force_map
        return force_map

    mu = state.mps.get_mu(0) if state.mps is not None else float(state.config.mu_f)
    if abs(float(mu)) <= 1.0e-30:
        state._gmsh_viscous_force_cache = force_map
        return force_map

    coords = np.asarray([np.asarray(vertex.x_a[:3], dtype=float) for vertex in vertices], dtype=float)
    velocities = np.asarray([np.asarray(vertex.u[:3], dtype=float) for vertex in vertices], dtype=float)
    for tet in tets:
        ids = [int(i) for i in tet]
        if any(i < 0 or i >= len(vertices) for i in ids):
            continue
        tetra = coords[ids]
        basis = np.ones((4, 4), dtype=float)
        basis[:, 1:] = tetra
        try:
            inv_basis = np.linalg.inv(basis)
        except np.linalg.LinAlgError:
            continue
        volume = abs(float(np.linalg.det(basis))) / 6.0
        if volume <= 1.0e-30:
            continue

        grad_phi = inv_basis[1:, :].T
        grad_u = velocities[ids].T @ grad_phi
        tau = float(mu) * (grad_u + grad_u.T)
        for local_idx, vertex_idx in enumerate(ids):
            force_map[id(vertices[vertex_idx])] += -volume * (tau @ grad_phi[local_idx])

    state._gmsh_viscous_force_cache = force_map
    return force_map


def _update_continuity_pressure_scalar(state: VolumetricPitoisState, *, dt: float) -> None:
    base_pressure = float(getattr(state.config, "continuity_pressure_pa", 0.0))
    state.continuity_pressure_scalar = base_pressure
    state.continuity_pressure_map = {}
    _update_pressure_scalar(state)
    if dt <= 0.0:
        return

    vertices = list(getattr(state, "volume_export_vertices", [])) or list(state.HC.V)
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if not vertices or tets.size == 0:
        return

    try:
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import lsmr
    except Exception:
        return

    n_vertices = len(vertices)
    coords = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
    id_to_idx = {id(vertex): idx for idx, vertex in enumerate(vertices)}
    masses = np.asarray([max(float(getattr(v, "m", 0.0)), 1.0e-12) for v in vertices], dtype=float)
    velocities = np.asarray([np.asarray(v.u[:3], dtype=float) for v in vertices], dtype=float)

    zero_pressure = lambda vv, HC=None, dim=3: 0.0
    predictor_force = np.zeros((n_vertices, 3), dtype=float)
    for vertex in vertices:
        if vertex in state.bV_caps:
            continue
        predictor_force[id_to_idx[id(vertex)]] = _Ftot(
            vertex,
            state=state,
            pressure_model=zero_pressure,
            include_contact_line=True,
        )
    predictor_velocity = velocities + float(dt) * predictor_force / masses[:, None]

    stiffness = lil_matrix((n_vertices, n_vertices), dtype=float)
    rhs = np.zeros(n_vertices, dtype=float)
    for tet in tets:
        ids = [int(i) for i in tet]
        if any(i < 0 or i >= n_vertices for i in ids):
            continue
        tetra = coords[ids]
        basis = np.ones((4, 4), dtype=float)
        basis[:, 1:] = tetra
        try:
            inv_basis = np.linalg.inv(basis)
        except np.linalg.LinAlgError:
            continue
        volume = abs(float(np.linalg.det(basis))) / 6.0
        if volume <= 1.0e-30:
            continue
        grad_phi = inv_basis[1:, :].T
        u_tet = np.mean(predictor_velocity[ids], axis=0)
        for a in range(4):
            ia = ids[a]
            rhs[ia] += (volume / float(dt)) * float(np.dot(grad_phi[a], u_tet))
            for b in range(4):
                stiffness[ia, ids[b]] += volume * float(np.dot(grad_phi[a], grad_phi[b]))

    matrix_csr = stiffness.tocsr()
    try:
        solved_pressure = np.asarray(
            lsmr(
                matrix_csr,
                rhs,
                atol=1.0e-12,
                btol=1.0e-12,
                maxiter=max(1000, 4 * n_vertices),
            )[0],
            dtype=float,
        )
    except Exception:
        return
    if solved_pressure.size < n_vertices or not np.all(np.isfinite(solved_pressure[:n_vertices])):
        return
    solved_pressure = solved_pressure[:n_vertices]
    solved_pressure = solved_pressure - float(np.mean(solved_pressure))

    solved_pressure = solved_pressure + base_pressure
    state.continuity_pressure_map = {id(vertex): float(solved_pressure[idx]) for idx, vertex in enumerate(vertices)}
    state.continuity_pressure_scalar = float(np.mean(solved_pressure)) if solved_pressure.size else base_pressure
    _update_pressure_scalar(state)


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


def _pressure_model(v, *, HC=None, dim: int = 3, state: VolumetricPitoisState):
    pressure_map = getattr(state, "continuity_pressure_map", None)
    if pressure_map and id(v) in pressure_map:
        pressure = (
            float(getattr(state.config, "internal_pressure_pa", 0.0))
            + float(pressure_map[id(v)])
        )
    else:
        pressure = float(state.pressure_scalar)
    if not state.config.include_gravity:
        return pressure

    z_ref = 0.5 * float(state.bottom_sphere_center[2] + state.top_sphere_center[2])
    z = float(np.asarray(v.x_a[:3], dtype=float)[2])
    return pressure - float(state.config.rho_f) * float(state.config.gravity_mps2) * (z - z_ref)


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
) -> float:
    prev_radius = _cap_radius(contact_ring)
    if not next_ring:
        return prev_radius

    theta_target = np.deg2rad(float(state.config.contact_angle_deg))
    radius = float(state.config.particle_radius)
    inward_sign = 1.0 if which == "bottom" else -1.0
    alpha_prev = float(np.arcsin(np.clip(prev_radius / max(radius, 1.0e-30), -1.0, 1.0)))
    slide_limit = max(float(state.config.contact_line_max_slide_um) * 1.0e-6, 1.0e-9)
    delta_alpha_max = slide_limit / max(radius, 1.0e-30)
    alpha_min = max(1.0e-8, alpha_prev - delta_alpha_max)
    alpha_max = min(0.5 * np.pi - 1.0e-8, alpha_prev + delta_alpha_max)
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
        angle[idx] = float(np.arccos(dot))

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


def _force_Fcl_Cox(v, *, state: VolumetricPitoisState) -> np.ndarray:
    bottom_center, top_center, axis = _particle_centers_physical(state)
    specs = (
        (
            state.bottom_contact_ring,
            np.asarray(bottom_center, dtype=float),
            np.array([0.0, 0.0, -float(state.config.cap_speed)], dtype=float),
            "bottom",
        ),
        (
            state.top_contact_ring,
            np.asarray(top_center, dtype=float),
            np.zeros(3, dtype=float),
            "top",
        ),
    )
    for ring, sphere_center, sphere_velocity, which in specs:
        if not ring or v not in ring:
            continue
        slide_dir = _contact_line_slide_direction(
            np.asarray(v.x_a[:3], dtype=float),
            sphere_center=sphere_center,
            axis=axis,
            which=which,
        )
        if float(np.linalg.norm(slide_dir)) <= 1.0e-30:
            return np.zeros(3, dtype=float)
        slide_velocity_map = getattr(state, "contact_line_slide_velocity_map", {})
        u_cl = slide_velocity_map.get(id(v))
        if u_cl is None:
            u_cl = float(np.dot(np.asarray(v.u[:3], dtype=float) - sphere_velocity, slide_dir))
        theta_dyn = _dynamic_contact_angle_from_speed(state, slide_speed=u_cl)
        line_length = float(_ring_segment_length_map(ring).get(id(v), 0.0))
        if line_length <= 0.0:
            return np.zeros(3, dtype=float)
        line_direction = axis if which == "top" else -axis
        return float(state.config.gamma) * line_length * float(np.cos(theta_dyn)) * line_direction
    return np.zeros(3, dtype=float)


def _align_layer_azimuths(state: VolumetricPitoisState) -> None:
    if len(state.outer_rings) < 2:
        return

    bottom_ring_center, _top_ring_center, axis = _contact_plane_centers_and_axis(
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


def _gmsh_element_to_tets(element_type: int, row: list[int]) -> list[list[int]]:
    if int(element_type) == 4 and len(row) >= 4:
        return [row[:4]]
    if int(element_type) == 5 and len(row) >= 8:
        a, b, c, d, e, f, g, h = row[:8]
        return [[a, b, d, e], [b, c, d, g], [b, d, e, g], [b, e, f, g], [d, e, g, h]]
    if int(element_type) == 6 and len(row) >= 6:
        a, b, c, d, e, f = row[:6]
        return [[a, b, c, d], [b, c, d, e], [c, d, e, f]]
    if int(element_type) == 7 and len(row) >= 5:
        a, b, c, d, apex = row[:5]
        return [[a, b, c, apex], [a, c, d, apex]]
    return []


def _read_gmsh2_points_triangles_tets(path: Path, lines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = lines.index("$Nodes") + 1
    n_nodes = int(lines[idx].split()[0])
    idx += 1
    node_tags: list[int] = []
    points: list[tuple[float, float, float]] = []
    for _node in range(n_nodes):
        parts = lines[idx].split()
        idx += 1
        if len(parts) < 4:
            raise ValueError(f"Bad Gmsh 2 node row while reading {path}: {parts}")
        node_tags.append(int(parts[0]))
        points.append((float(parts[1]), float(parts[2]), float(parts[3])))

    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
    idx = lines.index("$Elements") + 1
    n_elements = int(lines[idx].split()[0])
    idx += 1
    triangles: list[list[int]] = []
    tets: list[list[int]] = []
    for _element in range(n_elements):
        parts = [int(x) for x in lines[idx].split()]
        idx += 1
        if len(parts) < 4:
            continue
        element_type = int(parts[1])
        n_tags = int(parts[2])
        node_part = parts[3 + n_tags :]
        try:
            row = [tag_to_idx[tag] for tag in node_part]
        except KeyError as exc:
            raise ValueError(f"Element references missing node tag {exc} while reading {path}") from exc
        if element_type == 2 and len(row) >= 3:
            triangles.append(row[:3])
        elif element_type == 3 and len(row) >= 4:
            a, b, c, d = row[:4]
            triangles.append([a, b, c])
            triangles.append([a, c, d])
        else:
            tets.extend(_gmsh_element_to_tets(element_type, row))

    return (
        np.asarray(points, dtype=float),
        np.asarray(triangles, dtype=int),
        np.asarray(tets, dtype=int),
        np.asarray(node_tags, dtype=int),
    )


def _read_gmsh4_points_triangles_tets(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()

    mesh_format = lines[lines.index("$MeshFormat") + 1].split()
    version = float(mesh_format[0]) if mesh_format else 4.1
    if version < 4.0:
        return _read_gmsh2_points_triangles_tets(path, lines)

    idx = lines.index("$Nodes") + 1
    n_blocks, n_nodes, _min_tag, _max_tag = map(int, lines[idx].split()[:4])
    idx += 1
    node_tags: list[int] = []
    points: list[tuple[float, float, float]] = []
    for _block in range(n_blocks):
        _entity_dim, _entity_tag, _parametric, n_block_nodes = map(int, lines[idx].split()[:4])
        idx += 1
        block_tags = [int(lines[idx + j].split()[0]) for j in range(n_block_nodes)]
        idx += n_block_nodes
        for tag in block_tags:
            xyz = lines[idx].split()
            idx += 1
            node_tags.append(int(tag))
            points.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
    if len(points) != n_nodes:
        raise ValueError(f"Node count mismatch while reading {path}")

    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
    idx = lines.index("$Elements") + 1
    n_blocks, _n_elements, _min_elem, _max_elem = map(int, lines[idx].split()[:4])
    idx += 1
    triangles: list[list[int]] = []
    tets: list[list[int]] = []
    for _block in range(n_blocks):
        _entity_dim, _entity_tag, element_type, n_block_elements = map(int, lines[idx].split()[:4])
        idx += 1
        for _element in range(n_block_elements):
            parts = [int(x) for x in lines[idx].split()]
            idx += 1
            row = [tag_to_idx[tag] for tag in parts[1:]]
            if int(element_type) == 2 and len(row) >= 3:
                triangles.append(row[:3])
            elif int(element_type) == 3 and len(row) >= 4:
                a, b, c, d = row[:4]
                triangles.append([a, b, c])
                triangles.append([a, c, d])
            else:
                tets.extend(_gmsh_element_to_tets(element_type, row))

    return (
        np.asarray(points, dtype=float),
        np.asarray(triangles, dtype=int) if triangles else np.empty((0, 3), dtype=int),
        np.asarray(tets, dtype=int) if tets else np.empty((0, 4), dtype=int),
        np.asarray(node_tags, dtype=int),
    )


def _axisymmetric_profile_from_gmsh_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    z_abs = np.abs(pts[:, 2])
    radii = np.linalg.norm(pts[:, :2], axis=1)

    by_z: dict[float, float] = {}
    for z_val, radius in zip(z_abs, radii):
        key = round(float(z_val), 15)
        by_z[key] = max(by_z.get(key, 0.0), float(radius))

    half_z_all = np.asarray(sorted(by_z), dtype=float)
    half_r_all = np.asarray([by_z[float(z)] for z in half_z_all], dtype=float)
    if half_z_all.size < 2:
        raise ValueError(f"Could not read an axisymmetric profile from {HARDCODED_INITIAL_MSH_PATH}")

    keep_z: list[float] = []
    keep_r: list[float] = []
    current_max = -float("inf")
    tol = max(1.0e-12, 1.0e-9 * float(np.max(half_r_all)))
    for z_val, radius in zip(half_z_all, half_r_all):
        if radius + tol < current_max:
            continue
        keep_z.append(float(z_val))
        keep_r.append(float(radius))
        current_max = max(current_max, float(radius))

    half_z = np.asarray(keep_z, dtype=float)
    half_r = np.asarray(keep_r, dtype=float)
    sample_s = np.concatenate((-half_z[:0:-1], half_z))
    sample_r = np.concatenate((half_r[:0:-1], half_r))
    return sample_s, sample_r


def _ordered_ring_from_mask(vertices: list, points: np.ndarray, mask: np.ndarray) -> list:
    ids = np.flatnonzero(mask)
    if ids.size == 0:
        return []
    angles = np.arctan2(points[ids, 1], points[ids, 0])
    order = np.argsort(angles)
    return [vertices[int(ids[i])] for i in order]


def _gmsh_free_surface_rings(vertices: list, points: np.ndarray) -> list[list[object]]:
    sample_s, sample_r = _axisymmetric_profile_from_gmsh_points(points)
    z_vals = points[:, 2]
    r_vals = np.linalg.norm(points[:, :2], axis=1)
    z_span = max(float(np.max(z_vals) - np.min(z_vals)), 1.0e-12)
    r_max = max(float(np.max(r_vals)), 1.0e-12)
    z_tol = max(5.0e-12, 1.0e-8 * z_span)
    r_tol = max(5.0e-12, 1.0e-6 * r_max)
    rings: list[list[object]] = []
    for z_target, r_target in zip(sample_s, sample_r):
        mask = (np.abs(z_vals - float(z_target)) <= z_tol) & (np.abs(r_vals - float(r_target)) <= r_tol)
        ring = _ordered_ring_from_mask(vertices, points, mask)
        if len(ring) >= 3:
            rings.append(ring)
    if len(rings) < 2:
        raise RuntimeError(f"Could not extract liquid-air rings directly from {HARDCODED_INITIAL_MSH_PATH}")
    return rings


def _split_gmsh_boundary_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    bottom_sphere_center: np.ndarray,
    top_sphere_center: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri_idx = np.asarray(triangles, dtype=int)
    if tri_idx.size == 0:
        empty = np.empty((0, 3), dtype=int)
        return empty, empty, empty

    bottom: list[list[int]] = []
    top: list[list[int]] = []
    side: list[list[int]] = []
    tol = max(2.0e-8, 2.0e-5 * float(radius))
    for tri in tri_idx:
        xyz = points[np.asarray(tri, dtype=int)]
        centroid = np.mean(xyz, axis=0)
        bottom_vertex = (
            (xyz[:, 2] <= tol)
            & (np.abs(np.linalg.norm(xyz - bottom_sphere_center[None, :], axis=1) - float(radius)) <= tol)
        )
        top_vertex = (
            (xyz[:, 2] >= -tol)
            & (np.abs(np.linalg.norm(xyz - top_sphere_center[None, :], axis=1) - float(radius)) <= tol)
        )
        if centroid[2] < 0.0 and bool(np.all(bottom_vertex)):
            bottom.append([int(v) for v in tri])
        elif centroid[2] > 0.0 and bool(np.all(top_vertex)):
            top.append([int(v) for v in tri])
        else:
            side.append([int(v) for v in tri])

    return (
        np.asarray(side, dtype=int) if side else np.empty((0, 3), dtype=int),
        np.asarray(bottom, dtype=int) if bottom else np.empty((0, 3), dtype=int),
        np.asarray(top, dtype=int) if top else np.empty((0, 3), dtype=int),
    )


def _build_gmsh_axisym_ring_specs(
    vertices: list,
    points: np.ndarray,
    *,
    bottom_sphere_center: np.ndarray,
    top_sphere_center: np.ndarray,
) -> list[SimpleNamespace]:
    bottom = np.asarray(bottom_sphere_center, dtype=float)
    top = np.asarray(top_sphere_center, dtype=float)
    axis = top - bottom
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    mid = 0.5 * (bottom + top)
    e1, e2 = _orthonormal_tangent_basis(axis, np.array([1.0, 0.0, 0.0], dtype=float))

    pts = np.asarray(points, dtype=float)
    groups: dict[tuple[float, float], list[tuple[float, object]]] = defaultdict(list)
    for vertex, point in zip(vertices, pts):
        rel = np.asarray(point, dtype=float) - mid
        s = float(np.dot(rel, axis))
        radial_vec = rel - s * axis
        r = float(np.linalg.norm(radial_vec))
        if r <= 1.0e-30:
            angle = 0.0
        else:
            angle = float(np.arctan2(float(np.dot(radial_vec, e2)), float(np.dot(radial_vec, e1))))
        groups[(round(s, 12), round(r, 12))].append((angle, vertex))

    specs: list[SimpleNamespace] = []
    for (s0, r0), entries in groups.items():
        entries.sort(key=lambda item: item[0])
        specs.append(
            SimpleNamespace(
                s0=float(s0),
                r0=float(r0),
                vertices=[vertex for _angle, vertex in entries],
                angles=tuple(float(angle) for angle, _vertex in entries),
            )
        )
    specs.sort(key=lambda spec: (float(spec.s0), float(spec.r0), len(spec.vertices)))
    return specs


def _prepare_gmsh_state(config: VolumetricPitoisConfig) -> VolumetricPitoisState:
    points, triangles, tets, _node_tags = _read_gmsh4_points_triangles_tets(HARDCODED_INITIAL_MSH_PATH)
    HC = Complex(3, domain=None)
    vertices = [HC.V[tuple(float(x) for x in point)] for point in points]

    for tet in np.asarray(tets, dtype=int):
        tet_vertices = [vertices[int(i)] for i in tet]
        for a in range(4):
            for b in range(a + 1, 4):
                tet_vertices[a].connect(tet_vertices[b])
    for tri in np.asarray(triangles, dtype=int):
        tri_vertices = [vertices[int(i)] for i in tri]
        for a in range(3):
            tri_vertices[a].connect(tri_vertices[(a + 1) % 3])
    HC.gmsh_volume_vertices = list(vertices)
    HC.gmsh_volume_tets = np.asarray(tets, dtype=int)

    outer_rings = _gmsh_free_surface_rings(vertices, points)
    layer_rings = [[ring] for ring in outer_rings]
    layer_centers = [min(ring, key=lambda v: float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float)))) for ring in outer_rings]
    z_bottom = float(np.mean([float(v.x_a[2]) for v in outer_rings[0]]))
    z_top = float(np.mean([float(v.x_a[2]) for v in outer_rings[-1]]))
    contact_radius = 0.5 * (_cap_radius(outer_rings[0]) + _cap_radius(outer_rings[-1]))
    axial_offset = float(np.sqrt(max(float(config.particle_radius) ** 2 - contact_radius**2, 0.0)))
    bottom_sphere_center = np.array([0.0, 0.0, z_bottom - axial_offset], dtype=float)
    top_sphere_center = np.array([0.0, 0.0, z_top + axial_offset], dtype=float)

    dist_bottom = np.abs(np.linalg.norm(points - bottom_sphere_center[None, :], axis=1) - float(config.particle_radius))
    dist_top = np.abs(np.linalg.norm(points - top_sphere_center[None, :], axis=1) - float(config.particle_radius))
    sphere_tol = max(2.0e-8, 2.0e-5 * float(config.particle_radius))
    cap_bottom = [vertices[int(i)] for i in np.flatnonzero((points[:, 2] <= sphere_tol) & (dist_bottom <= sphere_tol))]
    cap_top = [vertices[int(i)] for i in np.flatnonzero((points[:, 2] >= -sphere_tol) & (dist_top <= sphere_tol))]
    cap_bottom_center = min(cap_bottom, key=lambda v: float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float)))) if cap_bottom else outer_rings[0][0]
    cap_top_center = min(cap_top, key=lambda v: float(np.linalg.norm(np.asarray(v.x_a[:2], dtype=float)))) if cap_top else outer_rings[-1][0]

    side_tris, bottom_cap_tris, top_cap_tris = _split_gmsh_boundary_triangles(
        points,
        triangles,
        bottom_sphere_center=bottom_sphere_center,
        top_sphere_center=top_sphere_center,
        radius=float(config.particle_radius),
    )

    interface_ids = {id(v) for ring in outer_rings for v in ring}
    cap_ids = {id(v) for v in cap_bottom + cap_top}
    for v in vertices:
        v.cap_id = "bottom" if v in cap_bottom else ("top" if v in cap_top else None)
        v.phase = 0
        v.p = 0.0
        v.u = np.zeros(3, dtype=float)
        v.boundary = bool(id(v) in interface_ids or id(v) in cap_ids)
        v.is_interface = bool(id(v) in interface_ids)
        v.interface_phases = frozenset({0, 1}) if v.is_interface else frozenset()

    layer_z = np.asarray([np.mean([float(v.x_a[2]) for v in ring]) for ring in outer_rings], dtype=float)
    z_span = max(float(layer_z[-1] - layer_z[0]), 1.0e-30)
    layer_fractions = tuple(float((z - layer_z[0]) / z_span) for z in layer_z)
    mesh_volume_m3 = _indexed_tet_mesh_volume_m3(points, tets)
    mps = SimpleNamespace(
        get_gamma_pair=lambda phase_a, phase_b: config.gamma,
        get_mu=lambda phase: config.mu_f,
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
        surface_edges=[],
        surface_boundary_indices=set(),
        mps=mps,
        radial_scale=1.0,
        axial_scale=1.0,
        initial_gap=float(np.linalg.norm(top_sphere_center - bottom_sphere_center) - 2.0 * float(config.particle_radius)),
        cap_ring_factors=(1.0,),
        bottom_sphere_center=bottom_sphere_center,
        top_sphere_center=top_sphere_center,
        target_volume_m3=float(mesh_volume_m3),
        target_snapshot_volume_m3=float(mesh_volume_m3),
    )
    state.gmsh_compute_mesh = True
    state.gmsh_axisym_reference = np.array([1.0, 0.0, 0.0], dtype=float)
    state.gmsh_axisym_ring_specs = _build_gmsh_axisym_ring_specs(
        vertices,
        points,
        bottom_sphere_center=bottom_sphere_center,
        top_sphere_center=top_sphere_center,
    )
    state.loaded_initial_msh_path = str(HARDCODED_INITIAL_MSH_PATH)
    state.volume_export_vertices = list(vertices)
    state.volume_export_node_ids = {id(vertex): idx for idx, vertex in enumerate(vertices)}
    state.volume_export_tets = np.asarray(tets, dtype=int)
    state.surface_export_vertices = list(vertices)
    state.surface_export_node_ids = {id(vertex): idx for idx, vertex in enumerate(vertices)}
    state.surface_export_side_tris = np.asarray(side_tris, dtype=int)
    if not state.surface_export_side_tris.size:
        state.surface_export_side_tris = _side_surface_tri_indices_from_rings(state)
    state.surface_export_bottom_cap_tris = np.asarray(bottom_cap_tris, dtype=int)
    state.surface_export_top_cap_tris = np.asarray(top_cap_tris, dtype=int)
    _refresh_cap_boundary_sets(state)
    state.frozen_volume_vertex_id_order = tuple(id(vertex) for vertex in state.volume_export_vertices)
    state.frozen_surface_vertex_id_order = tuple(id(vertex) for vertex in state.surface_export_vertices)
    state.frozen_layer_ring_id_structure = _layer_ring_id_structure(state)
    state.topology_frozen = True
    _set_cap_velocities(state)
    _update_duals_and_masses(state)
    _update_pressure_scalar(state)
    return state


def _snapshot_projectable_vertex_positions(state: VolumetricPitoisState) -> tuple[list, np.ndarray]:
    projectable_vertices = [
        v
        for v in state.HC.V
        if (v not in state.bV_caps) and (getattr(v, "cap_id", None) is None)
    ]
    if not projectable_vertices:
        return [], np.empty((0, 3), dtype=float)
    positions = np.array([np.asarray(v.x_a[:3], dtype=float) for v in projectable_vertices], dtype=float)
    return projectable_vertices, positions


def _axisym_water_volume_scaled_positions(
    state: VolumetricPitoisState,
    positions: np.ndarray,
    *,
    meridional_scale: float,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return np.empty((0, 3), dtype=float)
    if (not np.all(np.isfinite(positions))) or float(np.max(np.abs(positions))) > 1.0:
        raise ValueError("rejecting non-physical trial coordinates during axisymmetric volume scaling")

    axis = state.top_sphere_center - state.bottom_sphere_center
    axis_norm = float(np.linalg.norm(axis))
    if (not np.isfinite(axis_norm)) or axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    mid = 0.5 * (state.bottom_sphere_center + state.top_sphere_center)
    if (not np.all(np.isfinite(mid))) or float(np.max(np.abs(mid))) > 1.0:
        raise ValueError("rejecting non-physical sphere centers during axisymmetric volume scaling")

    rel = positions - mid[None, :]
    axial_mag = np.sum(rel * axis[None, :], axis=1)
    axial = axial_mag[:, None] * axis[None, :]
    radial = rel - axial
    lam = max(float(meridional_scale), 1.0e-6)
    # In the axisymmetric bridge, shrinking the meridian should compress axial
    # extent more strongly than radial extent to reduce the water-filled tet sum
    # without moving the sphere-caps directly.
    return mid[None, :] + (lam * lam) * axial + lam * radial


def _axisym_force_snapshot_volume_to_target(
    state: VolumetricPitoisState,
    *,
    rel_tol: float | None = None,
    max_iters: int | None = None,
) -> float:
    target = float(state.target_snapshot_volume_m3)
    if not np.isfinite(target) or target <= 0.0:
        return float(_snapshot_msh_volume_m3(state))

    if rel_tol is None:
        rel_tol = float(USER_GEOMETRIC_VOLUME_CORRECTION_REL_TOL)
    if max_iters is None:
        max_iters = max(16, int(USER_GEOMETRIC_VOLUME_CORRECTION_MAX_ITERS))

    current = float(_snapshot_msh_volume_m3(state))
    if not np.isfinite(current) or current <= 0.0:
        return current
    if abs(current - target) / max(target, 1.0e-30) <= float(rel_tol):
        return current

    projectable_vertices, base_positions = _snapshot_projectable_vertex_positions(state)
    if not projectable_vertices:
        return current

    def _restore_base() -> None:
        _apply_free_vertex_positions(state, projectable_vertices, base_positions)

    def _volume_at(scale: float) -> float:
        trial_positions = _axisym_water_volume_scaled_positions(
            state,
            base_positions,
            meridional_scale=float(scale),
        )
        _apply_free_vertex_positions(state, projectable_vertices, trial_positions)
        trial_volume = _snapshot_msh_volume_m3(state)
        _restore_base()
        return float(trial_volume)

    low = 1.0
    high = 1.0
    if current < target:
        high = 1.05
        vol_high = _volume_at(high)
        while vol_high < target and high < 8.0:
            high *= 1.10
            vol_high = _volume_at(high)
        if vol_high < target:
            return current
    else:
        low = 0.95
        vol_low = _volume_at(low)
        while vol_low > target and low > 0.02:
            low *= 0.90
            vol_low = _volume_at(low)
        if vol_low > target:
            return current
        high = 1.0

    final_scale = 1.0
    for _ in range(max(1, int(max_iters))):
        mid = 0.5 * (low + high)
        vol_mid = _volume_at(mid)
        final_scale = mid
        if abs(vol_mid - target) / max(target, 1.0e-30) <= float(rel_tol):
            break
        if vol_mid < target:
            low = mid
        else:
            high = mid

    final_positions = _axisym_water_volume_scaled_positions(
        state,
        base_positions,
        meridional_scale=float(final_scale),
    )
    _apply_free_vertex_positions(state, projectable_vertices, final_positions)
    return float(_snapshot_msh_volume_m3(state))


def _apply_free_vertex_positions(
    state: VolumetricPitoisState,
    free_vertices: list,
    positions: np.ndarray,
) -> None:
    if not free_vertices:
        return
    _move_vertices_batch(
        free_vertices,
        [tuple(map(float, row)) for row in np.asarray(positions, dtype=float)],
        state.HC,
        state.bV_caps,
    )


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

    new_center = sphere_center + sign * radius * axis
    _move(center_vertex, tuple(new_center), state.HC, state.bV_caps)

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
            _move(v, target, state.HC, state.bV_caps)
    _refresh_cap_boundary_sets(state)


def _base_update_moving_contact_line(state: VolumetricPitoisState, *, dt: float | None = None) -> None:
    axis = state.top_sphere_center - state.bottom_sphere_center
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    prev_bottom_radius = float(_cap_radius(state.outer_rings[0]))
    prev_top_radius = float(_cap_radius(state.outer_rings[-1]))
    bottom_ring = state.bottom_contact_ring
    top_ring = state.top_contact_ring
    bottom_coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in bottom_ring], dtype=float)
    top_coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in top_ring], dtype=float)
    bottom_rel = bottom_coords - state.bottom_sphere_center[None, :]
    top_rel = top_coords - state.top_sphere_center[None, :]
    bottom_tangential = bottom_rel - np.outer(np.dot(bottom_rel, axis), axis)
    top_tangential = top_rel - np.outer(np.dot(top_rel, axis), axis)
    bottom_radius = float(np.mean(np.linalg.norm(bottom_tangential, axis=1)))
    top_radius = float(np.mean(np.linalg.norm(top_tangential, axis=1)))

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


def _base_move_caps(state: VolumetricPitoisState, *, dt: float | None = None) -> None:
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


_FORCE_COMPONENT_NAMES = ("Fp", "Fs", "Fv", "Fcl")


def _force_terms_total(force_terms: SimpleNamespace) -> np.ndarray:
    return np.asarray(force_terms.Fp + force_terms.Fs + force_terms.Fv + force_terms.Fcl, dtype=float)


def _project_force_terms_no_swirl(
    force_terms: SimpleNamespace,
    *,
    point: np.ndarray,
    axis_origin: np.ndarray,
    axis: np.ndarray,
) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            name: _remove_swirl_component(
                np.asarray(getattr(force_terms, name), dtype=float),
                point=point,
                axis_origin=axis_origin,
                axis=axis,
            )
            for name in _FORCE_COMPONENT_NAMES
        }
    )


def _base_force_terms(
    v,
    *,
    state: VolumetricPitoisState,
    pressure_model=None,
    include_contact_line: bool = True,
) -> SimpleNamespace:
    if pressure_model is None:
        pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)

    force_terms = _force_Fp_Fv_Fs_components(
        v,
        dim=3,
        state=state,
        pressure_model=pressure_model,
    )
    Fp = force_terms.Fp
    Fs = force_terms.Fs
    Fv = force_terms.Fv
    Fcl = _force_Fcl_Cox(v, state=state) if include_contact_line else np.zeros(3, dtype=float)

    force_terms = SimpleNamespace(Fp=Fp, Fs=Fs, Fv=Fv, Fcl=Fcl)
    if bool(getattr(state.config, "enforce_no_swirl", False)):
        axis_origin, axis = _swirl_axis_geometry(state)
        force_terms = _project_force_terms_no_swirl(
            force_terms,
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=axis_origin,
            axis=axis,
        )
    return force_terms


def _axisym_clear_caches(state: VolumetricPitoisState) -> None:
    for attr in (
        "_axisym_force_terms_cache",
        "_gmsh_axisym_force_terms_cache",
        "_gmsh_surface_heron_force_cache",
        "_gmsh_viscous_force_cache",
        "_boundary_area_normal_cache",
        "_side_area_normal_cache",
    ):
        if hasattr(state, attr):
            delattr(state, attr)


def _axisym_force_workers(state: VolumetricPitoisState) -> int:
    if not bool(USER_ENFORCE_FULL_AXISYMMETRY):
        return 1
    return max(1, int(getattr(state.config, "accel_workers", 1)))


def _axisym_basis(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bottom_center, top_center, axis = _particle_centers_physical(state)
    mid = 0.5 * (bottom_center + top_center)
    ref_ring = state.outer_rings[0] if state.outer_rings else []
    if ref_ring:
        ref_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in ref_ring], axis=0)
        reference = np.asarray(ref_ring[0].x_a[:3], dtype=float) - ref_center
    else:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    e1, e2 = _orthonormal_tangent_basis(axis, reference)
    return mid, axis, e1, e2


def _axisym_layer_center_coord(
    state: VolumetricPitoisState,
    *,
    layer_idx: int,
    mid: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    outer_ring = state.outer_rings[layer_idx]
    if outer_ring:
        outer_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in outer_ring], axis=0)
        axial_s = float(np.dot(outer_center - mid, axis))
    else:
        axial_s = float(np.dot(np.asarray(state.layer_centers[layer_idx].x_a[:3], dtype=float) - mid, axis))
    return mid + axial_s * axis


def _axisym_ring_radius(ring: list, *, center_coord: np.ndarray, axis: np.ndarray) -> float:
    if not ring:
        return 0.0
    coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
    rel = coords - center_coord[None, :]
    rel -= np.outer(np.dot(rel, axis), axis)
    return float(np.mean(np.linalg.norm(rel, axis=1)))


def _axisym_ring_radial_units(
    ring: list,
    *,
    center_coord: np.ndarray,
    axis: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> list[np.ndarray]:
    units: list[np.ndarray] = []
    n_ring = max(1, len(ring))
    for i, v in enumerate(ring):
        rel = np.asarray(v.x_a[:3], dtype=float) - center_coord
        radial = rel - axis * float(np.dot(rel, axis))
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm <= 1.0e-30:
            phi = 2.0 * np.pi * float(i) / float(n_ring)
            units.append(np.cos(phi) * e1 + np.sin(phi) * e2)
        else:
            units.append(radial / radial_norm)
    return units


def _axisymmetrize_full_geometry(state: VolumetricPitoisState) -> None:
    if not bool(USER_ENFORCE_FULL_AXISYMMETRY):
        return
    mid, axis, e1, e2 = _axisym_basis(state)
    n_layers = len(state.layer_rings)
    if n_layers == 0:
        return

    bottom_radius = float(_cap_radius(state.bottom_contact_ring)) if state.bottom_contact_ring else float(state.config.target_cap_radius)
    top_radius = float(_cap_radius(state.top_contact_ring)) if state.top_contact_ring else float(state.config.target_cap_radius)
    _rebuild_cap_on_sphere(
        state,
        which="bottom",
        sphere_center=state.bottom_sphere_center,
        contact_radius=bottom_radius,
        axis=axis,
        dt=None,
    )
    _rebuild_cap_on_sphere(
        state,
        which="top",
        sphere_center=state.top_sphere_center,
        contact_radius=top_radius,
        axis=axis,
        dt=None,
    )

    for k in range(1, max(1, n_layers - 1)):
        if k >= n_layers - 1:
            break
        rings = state.layer_rings[k]
        center_coord = _axisym_layer_center_coord(state, layer_idx=k, mid=mid, axis=axis)
        _move(state.layer_centers[k], tuple(center_coord), state.HC, state.bV_caps)
        for ring in rings:
            radius = _axisym_ring_radius(ring, center_coord=center_coord, axis=axis)
            n_ring = max(1, len(ring))
            targets = []
            for i, _v in enumerate(ring):
                phi = 2.0 * np.pi * float(i) / float(n_ring)
                target = center_coord + radius * (np.cos(phi) * e1 + np.sin(phi) * e2)
                targets.append(tuple(target))
            _move_vertices_batch(ring, targets, state.HC, state.bV_caps)
    _refresh_cap_boundary_sets(state)


def _axisymmetrize_full_velocity_field(state: VolumetricPitoisState) -> None:
    if not bool(USER_ENFORCE_FULL_AXISYMMETRY):
        return
    mid, axis, e1, e2 = _axisym_basis(state)
    for k, rings in enumerate(state.layer_rings):
        center_coord = _axisym_layer_center_coord(state, layer_idx=k, mid=mid, axis=axis)
        for ring in rings:
            if not ring:
                continue
            radial_units = _axisym_ring_radial_units(
                ring,
                center_coord=center_coord,
                axis=axis,
                e1=e1,
                e2=e2,
            )
            ur = float(
                np.mean(
                    [
                        float(np.dot(np.asarray(v.u[:3], dtype=float), e_r))
                        for v, e_r in zip(ring, radial_units)
                    ]
                )
            )
            uz = float(np.mean([float(np.dot(np.asarray(v.u[:3], dtype=float), axis)) for v in ring]))
            for v, e_r in zip(ring, radial_units):
                v.u = ur * e_r + uz * axis
        center_v = state.layer_centers[k]
        center_v.u = float(np.dot(np.asarray(center_v.u[:3], dtype=float), axis)) * axis


def _axisym_force_cache_key(
    state: VolumetricPitoisState,
    *,
    include_contact_line: bool,
) -> tuple:
    return (
        bool(include_contact_line),
        round(float(getattr(state, "pressure_scalar", 0.0)), 18),
    )


def _gmsh_axisym_basis(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bottom_center, top_center, axis = _particle_centers_physical(state)
    mid = 0.5 * (bottom_center + top_center)
    reference = np.asarray(getattr(state, "gmsh_axisym_reference", np.array([1.0, 0.0, 0.0])), dtype=float)
    e1, e2 = _orthonormal_tangent_basis(axis, reference)
    return mid, axis, e1, e2


def _gmsh_axisym_spec_radial_units(spec, e1: np.ndarray, e2: np.ndarray) -> list[np.ndarray]:
    return [
        float(np.cos(angle)) * e1 + float(np.sin(angle)) * e2
        for angle in getattr(spec, "angles", ())
    ]


def _gmsh_axisym_group_axial_radius(
    vertices: list,
    *,
    center_coord: np.ndarray,
    axis: np.ndarray,
) -> tuple[float, float]:
    if not vertices:
        return 0.0, 0.0
    coords = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
    rel = coords - center_coord[None, :]
    axial = np.dot(rel, axis)
    radial = rel - np.outer(axial, axis)
    return float(np.mean(axial)), float(np.mean(np.linalg.norm(radial, axis=1)))


def _gmsh_axisym_uniform_cap_id(vertices: list) -> str | None:
    cap_ids = {getattr(v, "cap_id", None) for v in vertices}
    if cap_ids == {"bottom"}:
        return "bottom"
    if cap_ids == {"top"}:
        return "top"
    return None


def _gmsh_axisymmetrize_geometry(state: VolumetricPitoisState) -> None:
    if (not bool(USER_ENFORCE_FULL_AXISYMMETRY)) or (not bool(getattr(state, "gmsh_compute_mesh", False))):
        return
    specs = list(getattr(state, "gmsh_axisym_ring_specs", []))
    if not specs:
        return
    mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    radius = float(state.config.particle_radius)
    for spec in specs:
        vertices = list(getattr(spec, "vertices", []))
        if not vertices:
            continue
        _s_mean, r_mean = _gmsh_axisym_group_axial_radius(vertices, center_coord=mid, axis=axis)
        cap_id = _gmsh_axisym_uniform_cap_id(vertices)
        if cap_id is None:
            coords = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
            rel = coords - mid[None, :]
            axial = np.dot(rel, axis)
            center_coord = mid + float(np.mean(axial)) * axis
        else:
            sphere_center = (
                np.asarray(state.bottom_sphere_center, dtype=float)
                if cap_id == "bottom"
                else np.asarray(state.top_sphere_center, dtype=float)
            )
            sign = 1.0 if cap_id == "bottom" else -1.0
            cap_r = float(np.clip(r_mean, 0.0, radius))
            axial_offset = float(np.sqrt(max(radius**2 - cap_r**2, 0.0)))
            center_coord = sphere_center + sign * axial_offset * axis
            r_mean = cap_r

        angles = tuple(getattr(spec, "angles", ()))
        if len(vertices) == 1 or r_mean <= 1.0e-30:
            targets = [tuple(center_coord) for _v in vertices]
        else:
            targets = [
                tuple(center_coord + r_mean * (float(np.cos(angle)) * e1 + float(np.sin(angle)) * e2))
                for angle in angles
            ]
        _move_vertices_batch(vertices, targets, state.HC, state.bV_caps)
    _axisym_clear_caches(state)


def _gmsh_axisymmetrize_velocity_field(state: VolumetricPitoisState) -> None:
    if (not bool(USER_ENFORCE_FULL_AXISYMMETRY)) or (not bool(getattr(state, "gmsh_compute_mesh", False))):
        return
    specs = list(getattr(state, "gmsh_axisym_ring_specs", []))
    if not specs:
        return
    _mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    for spec in specs:
        vertices = list(getattr(spec, "vertices", []))
        if not vertices:
            continue
        units = _gmsh_axisym_spec_radial_units(spec, e1, e2)
        if len(vertices) == 1 or float(getattr(spec, "r0", 0.0)) <= 1.0e-30:
            uz = float(np.mean([float(np.dot(np.asarray(v.u[:3], dtype=float), axis)) for v in vertices]))
            for v in vertices:
                v.u = uz * axis
            continue
        ur = float(
            np.mean(
                [
                    float(np.dot(np.asarray(v.u[:3], dtype=float), e_r))
                    for v, e_r in zip(vertices, units)
                ]
            )
        )
        uz = float(np.mean([float(np.dot(np.asarray(v.u[:3], dtype=float), axis)) for v in vertices]))
        for v, e_r in zip(vertices, units):
            v.u = ur * e_r + uz * axis


def _gmsh_axisymmetrize_state(state: VolumetricPitoisState) -> None:
    if (not bool(USER_ENFORCE_FULL_AXISYMMETRY)) or (not bool(getattr(state, "gmsh_compute_mesh", False))):
        return
    _gmsh_axisymmetrize_geometry(state)
    _gmsh_axisymmetrize_velocity_field(state)
    _project_gmsh_contact_lines_to_spheres(state)
    _set_cap_velocities(state)
    _axisym_clear_caches(state)


def _gmsh_axisym_force_terms_map(
    state: VolumetricPitoisState,
    *,
    pressure_model,
    include_contact_line: bool,
) -> dict[int, SimpleNamespace]:
    cache = getattr(state, "_gmsh_axisym_force_terms_cache", {})
    key = _axisym_force_cache_key(
        state,
        include_contact_line=include_contact_line,
    )
    if key in cache:
        return cache[key]

    free_vertices = [v for v in state.HC.V if v not in state.bV_caps]
    raw_map = {
        id(v): _base_force_terms(
            v,
            state=state,
            pressure_model=pressure_model,
            include_contact_line=include_contact_line,
        )
        for v in free_vertices
    }
    mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    averaged: dict[int, SimpleNamespace] = {}
    covered: set[int] = set()
    for spec in getattr(state, "gmsh_axisym_ring_specs", []):
        vertices = list(getattr(spec, "vertices", []))
        units = _gmsh_axisym_spec_radial_units(spec, e1, e2)
        pairs = [(v, e_r) for v, e_r in zip(vertices, units) if id(v) in raw_map]
        if not pairs:
            continue
        _s_mean, r_mean = _gmsh_axisym_group_axial_radius(
            [v for v, _e_r in pairs],
            center_coord=mid,
            axis=axis,
        )
        if len(pairs) == 1 or r_mean <= 1.0e-30:
            group_terms = SimpleNamespace(
                **{
                    name: float(np.mean([float(np.dot(getattr(raw_map[id(v)], name), axis)) for v, _e_r in pairs]))
                    * axis
                    for name in _FORCE_COMPONENT_NAMES
                }
            )
            for v, _e_r in pairs:
                averaged[id(v)] = group_terms
                covered.add(id(v))
            continue
        for v, e_r in pairs:
            averaged[id(v)] = SimpleNamespace(
                **{
                    name: (
                        float(np.mean([float(np.dot(getattr(raw_map[id(member)], name), member_e_r)) for member, member_e_r in pairs]))
                        * e_r
                        + float(np.mean([float(np.dot(getattr(raw_map[id(member)], name), axis)) for member, _member_e_r in pairs]))
                        * axis
                    )
                    for name in _FORCE_COMPONENT_NAMES
                }
            )
            covered.add(id(v))

    for v in free_vertices:
        vid = id(v)
        if vid in covered:
            continue
        averaged[vid] = _project_force_terms_no_swirl(
            raw_map[vid],
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=mid,
            axis=axis,
        )

    cache[key] = averaged
    state._gmsh_axisym_force_terms_cache = cache
    return averaged


def _axisym_force_terms_map(
    state: VolumetricPitoisState,
    *,
    pressure_model,
    include_contact_line: bool,
) -> dict[int, SimpleNamespace]:
    cache = getattr(state, "_axisym_force_terms_cache", {})
    key = _axisym_force_cache_key(
        state,
        include_contact_line=include_contact_line,
    )
    if key in cache:
        return cache[key]

    mid, axis, e1, e2 = _axisym_basis(state)
    free_vertices = [v for v in state.HC.V if v not in state.bV_caps]
    workers = _axisym_force_workers(state)

    def raw_force_terms(v) -> tuple[int, SimpleNamespace]:
        return id(v), _base_force_terms(
            v,
            state=state,
            pressure_model=pressure_model,
            include_contact_line=include_contact_line,
        )

    if workers > 1 and len(free_vertices) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            raw_items = list(pool.map(raw_force_terms, free_vertices))
        raw_map = dict(raw_items)
    else:
        raw_map = dict(raw_force_terms(v) for v in free_vertices)

    averaged: dict[int, SimpleNamespace] = {}
    covered: set[int] = set()
    for k, rings in enumerate(state.layer_rings):
        center_coord = _axisym_layer_center_coord(state, layer_idx=k, mid=mid, axis=axis)
        for ring in rings:
            members = [v for v in ring if id(v) in raw_map]
            if not members:
                continue
            radial_units = _axisym_ring_radial_units(
                members,
                center_coord=center_coord,
                axis=axis,
                e1=e1,
                e2=e2,
            )
            for v, e_r in zip(members, radial_units):
                averaged[id(v)] = SimpleNamespace(
                    **{
                        name: (
                            float(
                                np.mean(
                                    [
                                        float(np.dot(getattr(raw_map[id(member)], name), member_e_r))
                                        for member, member_e_r in zip(members, radial_units)
                                    ]
                                )
                            )
                            * e_r
                            + float(np.mean([float(np.dot(getattr(raw_map[id(member)], name), axis)) for member in members]))
                            * axis
                        )
                        for name in _FORCE_COMPONENT_NAMES
                    }
                )
                covered.add(id(v))

        center_v = state.layer_centers[k]
        vid = id(center_v)
        if vid in raw_map:
            averaged[vid] = SimpleNamespace(
                **{
                    name: float(np.dot(getattr(raw_map[vid], name), axis)) * axis
                    for name in _FORCE_COMPONENT_NAMES
                }
            )
            covered.add(vid)

    for v in free_vertices:
        vid = id(v)
        if vid in covered:
            continue
        averaged[vid] = _project_force_terms_no_swirl(
            raw_map[vid],
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=mid,
            axis=axis,
        )

    cache[key] = averaged
    state._axisym_force_terms_cache = cache
    return averaged


def _update_duals_and_masses(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        if hasattr(state, "_dual_area_vector_cache"):
            delattr(state, "_dual_area_vector_cache")
        vertices = list(state.volume_export_vertices)
        points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
        tets = np.asarray(state.volume_export_tets, dtype=int)
        lumped = np.zeros(len(vertices), dtype=float)
        for a, b, c, d in tets:
            pa = points[int(a)]
            pb = points[int(b)]
            pc = points[int(c)]
            pd = points[int(d)]
            volume = abs(float(np.dot(pa - pd, np.cross(pb - pd, pc - pd))) / 6.0)
            share = 0.25 * volume
            lumped[int(a)] += share
            lumped[int(b)] += share
            lumped[int(c)] += share
            lumped[int(d)] += share
        for vertex, mass in zip(vertices, lumped):
            vertex.m = max(float(mass), 1.0e-12)
        _axisym_clear_caches(state)
        return
    _base_update_duals_and_masses(state)
    _axisym_clear_caches(state)


def _update_pressure_scalar(state: VolumetricPitoisState) -> None:
    _base_update_pressure_scalar(state)
    _axisym_clear_caches(state)


def _prepare_state(config: VolumetricPitoisConfig) -> VolumetricPitoisState:
    state = _prepare_gmsh_state(config)
    print(f"Loaded initial .msh      = {HARDCODED_INITIAL_MSH_PATH}")
    _axisym_clear_caches(state)
    return state


def _project_gmsh_contact_lines_to_spheres(state: VolumetricPitoisState) -> None:
    radius = float(state.config.particle_radius)
    specs = (
        (
            state.bottom_contact_ring,
            np.asarray(state.bottom_sphere_center, dtype=float),
            np.array([0.0, 0.0, -float(state.config.cap_speed)], dtype=float),
        ),
        (
            state.top_contact_ring,
            np.asarray(state.top_sphere_center, dtype=float),
            np.zeros(3, dtype=float),
        ),
    )
    for ring, sphere_center, sphere_velocity in specs:
        for v in ring:
            old = np.asarray(v.x_a[:3], dtype=float)
            rel = old - sphere_center
            rel_norm = float(np.linalg.norm(rel))
            if rel_norm <= 1.0e-30:
                radial = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                radial = rel / rel_norm
            target = sphere_center + radius * radial
            _move(v, tuple(target), state.HC, state.bV_caps)
            normal = (target - sphere_center) / max(radius, 1.0e-30)
            rel_u = np.asarray(v.u[:3], dtype=float) - sphere_velocity
            v.u = sphere_velocity + rel_u - float(np.dot(rel_u, normal)) * normal


def _move_caps(state: VolumetricPitoisState, *, dt: float | None = None) -> None:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        if dt is None:
            dt = float(state.config.dt)
        bottom_vel = np.array([0.0, 0.0, -float(state.config.cap_speed)], dtype=float)
        top_vel = np.zeros(3, dtype=float)
        bottom_disp = float(dt) * bottom_vel
        top_disp = float(dt) * top_vel
        if float(np.linalg.norm(bottom_disp)) <= 1.0e-30 and float(np.linalg.norm(top_disp)) <= 1.0e-30:
            return
        state.bottom_sphere_center = np.asarray(state.bottom_sphere_center, dtype=float) + bottom_disp
        state.top_sphere_center = np.asarray(state.top_sphere_center, dtype=float) + top_disp
        for v in state.cap_bottom_interior:
            target = np.asarray(v.x_a[:3], dtype=float) + bottom_disp
            _move(v, tuple(target), state.HC, state.bV_caps)
            v.u = bottom_vel
        for v in state.cap_top_interior:
            target = np.asarray(v.x_a[:3], dtype=float) + top_disp
            _move(v, tuple(target), state.HC, state.bV_caps)
            v.u = top_vel
        _axisym_clear_caches(state)
        return
    _base_move_caps(state, dt=dt)
    _axisymmetrize_full_velocity_field(state)
    _axisym_clear_caches(state)


def _update_moving_contact_line(state: VolumetricPitoisState, *, dt: float | None = None) -> None:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        if dt is None:
            dt = float(state.config.dt)
        before_volume = float(_snapshot_msh_volume_m3(state))
        axis = np.asarray(state.top_sphere_center, dtype=float) - np.asarray(state.bottom_sphere_center, dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1.0e-30:
            axis = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            axis = axis / axis_norm

        prev_bottom_radius = (
            _mean_ring_radius(state.bottom_contact_ring, np.asarray(state.bottom_sphere_center, dtype=float), axis)
            if state.bottom_contact_ring
            else 0.0
        )
        prev_top_radius = (
            _mean_ring_radius(state.top_contact_ring, np.asarray(state.top_sphere_center, dtype=float), axis)
            if state.top_contact_ring
            else 0.0
        )

        bottom_next_ring = state.outer_rings[1] if len(state.outer_rings) > 1 else []
        top_next_ring = state.outer_rings[-2] if len(state.outer_rings) > 1 else []
        bottom_radius = _estimate_contact_radius(
            state,
            which="bottom",
            sphere_center=np.asarray(state.bottom_sphere_center, dtype=float),
            contact_ring=state.bottom_contact_ring,
            next_ring=bottom_next_ring,
            axis=axis,
            dt=dt,
            gap_rate=float(state.config.relative_speed),
        )
        top_radius = _estimate_contact_radius(
            state,
            which="top",
            sphere_center=np.asarray(state.top_sphere_center, dtype=float),
            contact_ring=state.top_contact_ring,
            next_ring=top_next_ring,
            axis=axis,
            dt=dt,
            gap_rate=float(state.config.relative_speed),
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
        _align_layer_azimuths(state)
        _regularize_layer_order(state)
        after_volume = float(_snapshot_msh_volume_m3(state))
        sphere_radius = max(float(state.config.particle_radius), 1.0e-30)
        bottom_slide_velocity = 0.0
        top_slide_velocity = 0.0
        if float(dt) > 0.0:
            alpha_bottom_prev = float(np.arcsin(np.clip(prev_bottom_radius / sphere_radius, -1.0, 1.0)))
            alpha_bottom_new = float(np.arcsin(np.clip(bottom_radius / sphere_radius, -1.0, 1.0)))
            alpha_top_prev = float(np.arcsin(np.clip(prev_top_radius / sphere_radius, -1.0, 1.0)))
            alpha_top_new = float(np.arcsin(np.clip(top_radius / sphere_radius, -1.0, 1.0)))
            bottom_slide_velocity = sphere_radius * (alpha_bottom_new - alpha_bottom_prev) / float(dt)
            top_slide_velocity = sphere_radius * (alpha_top_new - alpha_top_prev) / float(dt)
            state.last_bottom_contact_line_speed = abs(bottom_slide_velocity)
            state.last_top_contact_line_speed = abs(top_slide_velocity)
        else:
            state.last_bottom_contact_line_speed = 0.0
            state.last_top_contact_line_speed = 0.0
        state.contact_line_slide_velocity_map = {
            **{id(v): float(bottom_slide_velocity) for v in state.bottom_contact_ring},
            **{id(v): float(top_slide_velocity) for v in state.top_contact_ring},
        }
        state.last_contact_line_volume_delta_m3 = after_volume - before_volume
        _axisym_clear_caches(state)
        return
    before_volume = float(_snapshot_msh_volume_m3(state))
    _base_update_moving_contact_line(state, dt=dt)
    _axisymmetrize_full_geometry(state)
    _axisymmetrize_full_velocity_field(state)
    after_volume = float(_snapshot_msh_volume_m3(state))
    state.last_contact_line_volume_delta_m3 = after_volume - before_volume
    _axisym_clear_caches(state)


def _project_volume_to_target(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        if bool(getattr(state.config, "enable_gmsh_geometric_volume_correction", False)):
            target = float(getattr(state, "target_snapshot_volume_m3", 0.0))
            current = float(_snapshot_msh_volume_m3(state))
            trigger = max(float(getattr(state.config, "gmsh_geometric_volume_correction_trigger_rel", 0.0)), 0.0)
            if (
                np.isfinite(target)
                and target > 0.0
                and np.isfinite(current)
                and abs(current - target) / max(target, 1.0e-30) > trigger
            ):
                _axisym_force_snapshot_volume_to_target(
                    state,
                    rel_tol=float(USER_GEOMETRIC_VOLUME_CORRECTION_REL_TOL),
                    max_iters=max(1, int(USER_GEOMETRIC_VOLUME_CORRECTION_MAX_ITERS)),
                )
                _gmsh_axisymmetrize_state(state)
        _axisym_clear_caches(state)
        return
    if bool(USER_ENFORCE_FULL_AXISYMMETRY):
        # Axisymmetric runs conserve volume through continuity pressure in Ftot,
        # not by post-step geometric rescaling of the liquid bridge.
        _axisymmetrize_full_geometry(state)
        _axisymmetrize_full_velocity_field(state)
        _axisym_clear_caches(state)
        return
    _axisym_clear_caches(state)


def _Ftot_terms(
    v,
    *,
    state: VolumetricPitoisState,
    pressure_model=None,
    include_contact_line: bool = True,
) -> SimpleNamespace:
    if pressure_model is None:
        pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        if (v in state.bV_caps) or (not bool(USER_ENFORCE_FULL_AXISYMMETRY)):
            return _base_force_terms(
                v,
                state=state,
                pressure_model=pressure_model,
                include_contact_line=include_contact_line,
            )
        force_terms_map = _gmsh_axisym_force_terms_map(
            state,
            pressure_model=pressure_model,
            include_contact_line=include_contact_line,
        )
        if id(v) in force_terms_map:
            return force_terms_map[id(v)]
        return _base_force_terms(
            v,
            state=state,
            pressure_model=pressure_model,
            include_contact_line=include_contact_line,
        )
    if (v in state.bV_caps) or (not bool(USER_ENFORCE_FULL_AXISYMMETRY)):
        return _base_force_terms(
            v,
            state=state,
            pressure_model=pressure_model,
            include_contact_line=include_contact_line,
        )
    force_terms_map = _axisym_force_terms_map(
        state,
        pressure_model=pressure_model,
        include_contact_line=include_contact_line,
    )
    if id(v) in force_terms_map:
        return force_terms_map[id(v)]
    zeros = np.zeros(3, dtype=float)
    return SimpleNamespace(Fp=zeros, Fs=zeros, Fv=zeros, Fcl=zeros)


def _Ftot(
    v,
    *,
    state: VolumetricPitoisState,
    pressure_model=None,
    include_contact_line: bool = True,
) -> np.ndarray:
    return _force_terms_total(
        _Ftot_terms(
            v,
            state=state,
            pressure_model=pressure_model,
            include_contact_line=include_contact_line,
        )
    )


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


def _contact_refined_surface_display_wireframe(
    state: VolumetricPitoisState,
    ordered_rings,
) -> tuple[np.ndarray, np.ndarray]:
    if not ordered_rings:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    coords = np.asarray(
        [[np.asarray(v.x_a[:3], dtype=float) for v in ring] for ring in ordered_rings],
        dtype=float,
    )
    edges = _structured_surface_display_edges(ordered_rings)
    flat = coords.reshape((-1, 3))
    segments = [np.asarray([flat[i], flat[j]], dtype=float) for i, j in sorted(edges)]
    return np.asarray(segments, dtype=float), flat


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


def _solver_top_sphere_force_components_n(state: VolumetricPitoisState) -> SimpleNamespace:
    _bottom_center, _top_center, axis = _particle_centers_physical(state)
    top_sphere_Fp_n = 0.0
    top_sphere_Fs_n = 0.0
    top_sphere_Fv_n = 0.0
    top_sphere_Fcl_n = 0.0
    for v in state.cap_top:
        force_terms = _Ftot_terms(v, state=state)
        Fp = force_terms.Fp
        Fs = force_terms.Fs
        Fv = force_terms.Fv
        Fcl = force_terms.Fcl
        Ftot = Fp + Fs + Fv + Fcl
        top_sphere_Fp_n += float(np.dot(Fp, axis))
        top_sphere_Fs_n += float(np.dot(Fs, axis))
        top_sphere_Fv_n += float(np.dot(Fv, axis))
        top_sphere_Fcl_n += float(np.dot(Fcl, axis))
    top_sphere_Ftot_n = top_sphere_Fp_n + top_sphere_Fs_n + top_sphere_Fv_n + top_sphere_Fcl_n
    return SimpleNamespace(
        Fp=float(top_sphere_Fp_n),
        Fs=float(top_sphere_Fs_n),
        Fv=float(top_sphere_Fv_n),
        Fcl=float(top_sphere_Fcl_n),
        Ftot=float(top_sphere_Ftot_n),
    )


def _update_top_sphere_force_n(state: VolumetricPitoisState) -> None:
    _update_pressure_scalar(state)
    force_components = _solver_top_sphere_force_components_n(state)
    state.top_sphere_Fp_n = float(force_components.Fp)
    state.top_sphere_Fs_n = float(force_components.Fs)
    state.top_sphere_Fv_n = float(force_components.Fv)
    state.top_sphere_Fcl_n = float(force_components.Fcl)
    state.top_sphere_Ftot_n = float(force_components.Ftot)
    state.simulated_force_n = float(force_components.Ftot)


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
    if float(config.dt) > 0.0 and not bool(getattr(config, "enable_adaptive_dt", False)):
        dt_fixed = float(config.dt)
        state.last_dt_limit_cl = dt_fixed
        state.last_dt_limit_capillary = dt_fixed
        state.last_dt_limit_mesh = dt_fixed
        state.last_dt_limiter = "fixed_user_dt"
        state.last_step_dt = dt_fixed
        return dt_fixed

    if not bool(getattr(config, "enable_adaptive_dt", False)):
        dt_fallback = max(
            float(config.dt),
            float(getattr(config, "adaptive_dt_max_s", 0.0)),
            1.0e-12,
        )
        state.last_dt_limit_cl = dt_fallback
        state.last_dt_limit_capillary = dt_fallback
        state.last_dt_limit_mesh = dt_fallback
        state.last_dt_limiter = "adaptive_disabled_fallback"
        state.last_step_dt = dt_fallback
        return dt_fallback

    dt_min = max(float(getattr(config, "adaptive_dt_min_s", 0.0)), 1.0e-12)
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
    }
    configured_ceiling = (
        float(config.dt)
        if float(config.dt) > 0.0
        else float(getattr(config, "adaptive_dt_max_s", 0.0))
    )
    if configured_ceiling > 0.0:
        limiter_map["dt_ceiling"] = max(float(configured_ceiling), dt_min)
    dt_selected = min(limiter_map.values())
    limiter = min(limiter_map, key=limiter_map.get)
    if dt_selected < dt_min:
        dt_selected = dt_min
        limiter = f"{limiter}+dt_floor"

    state.last_dt_limiter = limiter
    state.last_step_dt = float(dt_selected)
    return float(dt_selected)


def _step_record(state: VolumetricPitoisState, *, step: int, t: float) -> dict[str, float | int]:
    top_sphere_force_n = float(getattr(state, "simulated_force_n", 0.0))
    bottom_sphere_force_n = -top_sphere_force_n
    top_sphere_Fp_n = float(getattr(state, "top_sphere_Fp_n", 0.0))
    top_sphere_Fs_n = float(getattr(state, "top_sphere_Fs_n", 0.0))
    top_sphere_Fv_n = float(getattr(state, "top_sphere_Fv_n", 0.0))
    top_sphere_Fcl_n = float(getattr(state, "top_sphere_Fcl_n", 0.0))
    top_sphere_Ftot_n = float(getattr(state, "top_sphere_Ftot_n", top_sphere_force_n))
    if state.top_contact_ring:
        top_cl_avg_u = np.mean(
            [np.asarray(v.u[:3], dtype=float) for v in state.top_contact_ring],
            axis=0,
        )
    else:
        top_cl_avg_u = np.zeros(3, dtype=float)
    gap = _gap(state)
    snapshot_msh_volume_m3 = _snapshot_msh_volume_m3(state)
    return {
        "step": int(step),
        "t": float(t),
        "gap": gap,
        "d_over_r": gap / max(state.config.particle_radius, 1.0e-30),
        "top_sphere_Fp_n": top_sphere_Fp_n,
        "top_sphere_Fs_n": top_sphere_Fs_n,
        "top_sphere_Fv_n": top_sphere_Fv_n,
        "top_sphere_Fcl_n": top_sphere_Fcl_n,
        "top_sphere_Ftot_n": top_sphere_Ftot_n,
        "top_sphere_Fp_mN": 1.0e3 * top_sphere_Fp_n,
        "top_sphere_Fs_mN": 1.0e3 * top_sphere_Fs_n,
        "top_sphere_Fv_mN": 1.0e3 * top_sphere_Fv_n,
        "top_sphere_Fcl_mN": 1.0e3 * top_sphere_Fcl_n,
        "top_sphere_Ftot_mN": 1.0e3 * top_sphere_Ftot_n,
        "top_cl_avg_ux_mps": float(top_cl_avg_u[0]),
        "top_cl_avg_uy_mps": float(top_cl_avg_u[1]),
        "top_cl_avg_uz_mps": float(top_cl_avg_u[2]),
        "top_sphere_force_n": float(top_sphere_force_n),
        "bottom_sphere_force_n": float(bottom_sphere_force_n),
        "fixed_force_axial": float(top_sphere_force_n),
        "moving_force_axial": float(bottom_sphere_force_n),
        "fixed_force_mag": abs(float(top_sphere_force_n)),
        "moving_force_mag": abs(float(bottom_sphere_force_n)),
        "snapshot_msh_volume_m3": float(snapshot_msh_volume_m3),
        "dt": float(state.last_step_dt),
        "dt_limiter": str(state.last_dt_limiter),
        "max_free_speed": _max_free_speed(state),
    }

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



def _volume_points_array(state: VolumetricPitoisState) -> np.ndarray:
    _ensure_volume_export_node_ids(state)
    return np.asarray(
        [np.asarray(vertex.x_a[:3], dtype=float) for vertex in state.volume_export_vertices],
        dtype=float,
    )


def _snapshot_msh_indexed_mesh(
    state: VolumetricPitoisState,
    *,
    freeze_topology: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if freeze_topology:
        _freeze_surface_topology(state)
    points = _volume_points_array(state)
    tets = np.asarray(state.volume_export_tets, dtype=int)
    node_ids = np.arange(1, points.shape[0] + 1, dtype=int)
    return points, tets, node_ids


def _indexed_tet_mesh_volume_m3(points: np.ndarray, tets: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    tet_idx = np.asarray(tets, dtype=int)
    if pts.size == 0 or tet_idx.size == 0:
        return 0.0

    volume = 0.0
    for a, b, c, d in tet_idx:
        pa = pts[int(a)]
        pb = pts[int(b)]
        pc = pts[int(c)]
        pd = pts[int(d)]
        # Physical volumetric .msh volume is the sum of positive tetra volumes.
        # Never rely on cancellation from mixed element orientation.
        volume += abs(float(np.dot(pa - pd, np.cross(pb - pd, pc - pd))) / 6.0)
    return float(volume)


def _snapshot_msh_volume_m3(
    state: VolumetricPitoisState,
    *,
    freeze_topology: bool = True,
    side_triangles: np.ndarray | None = None,
    bottom_cap_triangles: np.ndarray | None = None,
    top_cap_triangles: np.ndarray | None = None,
) -> float:
    del side_triangles, bottom_cap_triangles, top_cap_triangles
    points, tets, _node_ids = _snapshot_msh_indexed_mesh(state, freeze_topology=freeze_topology)
    return _indexed_tet_mesh_volume_m3(points, tets)



def _advance_free_vertices_from_Ftot(state: VolumetricPitoisState, *, dt: float) -> None:
    free_vertices = [v for v in state.HC.V if v not in state.bV_caps]
    updates = {}
    for v in free_vertices:
        force_terms = _Ftot_terms(v, state=state)
        Fp = force_terms.Fp
        Fs = force_terms.Fs
        Fv = force_terms.Fv
        Fcl = force_terms.Fcl
        Ftot = Fp + Fs + Fv + Fcl
        mass = max(float(getattr(v, "m", 0.0)), 1.0e-12)
        a = _clip_acceleration(np.asarray(Ftot, dtype=float) / mass, state)
        u_old = np.asarray(v.u[:3], dtype=float)
        x_old = np.asarray(v.x_a[:3], dtype=float)

        # Motion update:
        #     a    = Ftot / m
        #     u    = u + a * dt
        #     dxyz = u * dt
        #     x    = x + dxyz
        u_new = u_old + a * float(dt)
        dxyz = u_new * float(dt)
        x_new = x_old + dxyz
        updates[v] = (x_new, u_new)

    for v, (x_new, u_new) in updates.items():
        v.u[:3] = u_new
        _move(v, x_new, state.HC, state.bV_caps)


def advance_one_step(state: VolumetricPitoisState, *, step_index: int | None = None) -> float:
    step_dt = _select_physical_dt(state)
    substeps = max(1, int(state.config.integration_substeps))
    sub_dt = float(step_dt) / float(substeps)
    for _substep in range(substeps):
        _set_cap_velocities(state)
        _enforce_no_swirl_velocity_field(state)
        _move_caps(state, dt=sub_dt)
        _gmsh_axisymmetrize_state(state)
        _update_duals_and_masses(state)
        _update_pressure_scalar(state)
        _update_continuity_pressure_scalar(state, dt=sub_dt)
        _update_top_sphere_force_n(state)
        _advance_free_vertices_from_Ftot(state, dt=sub_dt)
        _gmsh_axisymmetrize_state(state)
        _enforce_no_swirl_velocity_field(state)
        _set_cap_velocities(state)
        _update_moving_contact_line(state, dt=sub_dt)
        _gmsh_axisymmetrize_state(state)
        _enforce_no_swirl_velocity_field(state)
        _assert_fixed_topology(state)
        _project_volume_to_target(state)
        _gmsh_axisymmetrize_state(state)
        _assert_fixed_topology(state)
        _update_duals_and_masses(state)
        _update_pressure_scalar(state)
        _update_continuity_pressure_scalar(state, dt=sub_dt)
        _update_top_sphere_force_n(state)
        _assert_fixed_topology(state)
    label = step_index + 1 if step_index is not None else "advance"
    _assert_state_finite(state, where=f"step {label}")
    state.elapsed_time_s += float(step_dt)
    return float(step_dt)


def _advance_state(state: VolumetricPitoisState, *, n_steps: int) -> None:
    _assert_fixed_topology(state)
    for step in range(int(n_steps)):
        advance_one_step(state, step_index=step)


__all__ = [
    "USER_REFINEMENT",
    "USER_TOTAL_STEPS",
    "VolumetricPitoisConfig",
    "VolumetricPitoisState",
    "_actual_compute_cap_triangles",
    "_actual_compute_meridian_strip_triangles",
    "_actual_compute_side_surface_triangles",
    "_advance_state",
    "_assert_fixed_topology",
    "_build_structured_side_surface_complex",
    "_cap_radius",
    "_contact_refined_surface_display_wireframe",
    "_gap",
    "_meridian_section_wireframe",
    "_ordered_outer_rings_for_surface",
    "_orthonormal_tangent_basis",
    "_prepare_state",
    "_snapshot_msh_indexed_mesh",
    "_snapshot_msh_volume_m3",
    "_step_record",
    "_update_top_sphere_force_n",
    "advance_one_step",
    "separation_config",
]
