"""Case 2b axisym: Gmsh-mesh Pitois-2000 Fig. 5 separation case.

This case keeps the Pitois separation solver path from
``Case_2b_axisym_pitois2000_separation.py`` but initializes the compute mesh
directly from the hard-coded Gmsh ``mesh_iter0012.msh`` file.

1. Load the Gmsh volume/surface mesh as the compute mesh.
2. Move only the top and bottom cap interiors with prescribed velocities.
3. Leave the contact-line rings free so they can slide on the spheres.
4. Compute axial bridge force through the volumetric stress path and write the
   same separation history, CSV, MSH, and PNG outputs as the base case.

Important limitation:
    This case now uses a sparse tetrahedral velocity-pressure projection for
    incompressibility. The capillary pressure driving the force balance is
    still an axisymmetric Laplace-pressure surrogate built from the evolving
    outer bridge profile.

Contact-angle note:
    The Pitois 2000 Fig. 5 setup in this script uses a baseline modeling
    assumption of ``theta = 0 deg``. This is not a directly reported Pitois
    measurement in the current workflow. It is a literature-informed
    approximation motivated by silicone oils spreading essentially completely
    in air on clean solid substrates, including oxide-like high-energy solids;
    see Svitova et al., Langmuir 2002, DOI: 10.1021/la020006x.

Gravity note:
    The Fig. 5 experiment is not gravity-free, so this case now includes a
    hydrostatic pressure contribution ``-rho g (z - z_ref)`` along the
    physical vertical ``z`` axis. This is still a surrogate model, but it is
    closer to the experiment than the old gravity-free version.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field, replace
import json
import math
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import warnings
from collections import defaultdict

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

try:
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve
except Exception:  # pragma: no cover - only used when scipy is unavailable.
    coo_matrix = None
    diags = None
    spsolve = None


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases_dynamic.liquid_bridge_equilibrium.Case_5_volumetric_stress_equilibrium_particle_particle_bridge_benchmark import (
    HEX_TETS,
    PRISM_TETS,
)
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.dynamic_integrators._integrators_dynamic import _move, _recompute_duals
from ddgclib.operators.multiphase_stress import multiphase_stress_force
from ddgclib.operators.surface_tension import dual_area_heron, surface_tension_force
from ddgclib.operators.stress import cauchy_stress, dual_area_vector, dual_volume, velocity_difference_tensor_pointwise


class _DisplayVertex:
    def __init__(self, coords):
        self.x_a = np.asarray(coords, dtype=float)
        self.nn: set[object] = set()
        self.boundary = False

    def connect(self, other) -> None:
        self.nn.add(other)
        other.nn.add(self)


class _DisplayVertexStore(dict):
    def __init__(self):
        super().__init__()
        self._key_by_vertex_id: dict[int, tuple[float, float, float]] = {}

    def __setitem__(self, key, vertex):
        old_vertex = self.get(key)
        if old_vertex is not None and old_vertex is not vertex:
            self._key_by_vertex_id.pop(id(old_vertex), None)
        super().__setitem__(key, vertex)
        self._key_by_vertex_id[id(vertex)] = key

    def __delitem__(self, key):
        old_vertex = self.get(key)
        if old_vertex is not None:
            self._key_by_vertex_id.pop(id(old_vertex), None)
        super().__delitem__(key)

    def clear(self):
        self._key_by_vertex_id.clear()
        super().clear()

    def __missing__(self, key):
        vertex = _DisplayVertex(key)
        self[key] = vertex
        return vertex

    def __iter__(self):
        return iter(self.values())

    def move(self, vertex, pos):
        old_key = self._key_by_vertex_id.get(id(vertex))
        if old_key is not None and self.get(old_key) is vertex:
            del self[old_key]
        else:
            for key, value in list(self.items()):
                if value is vertex:
                    del self[key]
                    break
        vertex.x_a = np.asarray(pos, dtype=float)
        self[tuple(float(x) for x in vertex.x_a[:3])] = vertex


class Complex:
    def __init__(self, _dim: int, domain=None):
        self.domain = domain
        self.V = _DisplayVertexStore()


OUT_ROOT = Path(__file__).resolve().with_suffix("")
FIG5_COMPARE_PNG_NAME = "1_pitois2000_volumetric_separation_digitized_compare.png"
HARDCODED_INITIAL_MSH_PATH = (
    Path(__file__).resolve().parent
    / "out"
    / "Case_2b_axisym_initialshape_Gmsh"
    / "fig"
    / "mesh_iter0012.msh"
)


def _required_initial_msh_path() -> Path:
    path = HARDCODED_INITIAL_MSH_PATH
    if path.name != "mesh_iter0012.msh":
        raise RuntimeError(f"Initial mesh must be mesh_iter0012.msh, got {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Required initial Gmsh mesh not found: {path}")
    return path

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
USER_TOTAL_STEPS = 15000
# Time-step rule:
#   USER_DT_S > 0: fixed user step.
#   USER_DT_S < 0: physical step, dt = min(dt_CL, dt_capillary, dt_mesh).
# This keeps the automatic step computed from numerical/physical limits only.
USER_DT_S = 0.01
USER_TARGET_SWEEP_TIME_S = abs(USER_DT_S) * max(1, USER_TOTAL_STEPS)
USER_ENABLE_ADAPTIVE_DT = USER_DT_S < 0.0
# Negative means no artificial ceiling; the selected value is the right_dt
# physical minimum. Positive USER_DT_S remains the fixed step.
USER_ADAPTIVE_DT_MAX_S = -1.0 if USER_DT_S < 0.0 else USER_DT_S
# Do not let a user floor override the physical capillary/mesh/contact-line
# limit. The selector only keeps a tiny numerical guard against exactly zero.
USER_ADAPTIVE_DT_MIN_S = 0.0
USER_ADAPTIVE_DT_CAPILLARY_SAFETY = 0.25
USER_ADAPTIVE_DT_MESH_DISPLACEMENT_FRAC = 0.15

USER_RECORD_EVERY_STEPS = 100
USER_MESH_SNAPSHOT_EVERY_STEPS = 100
# The imported t=0 geometry is calibrated to the Fig. 5 setup.  Do not use the
# first near-start row as independent dynamic validation in the comparison plot.
USER_FIG5_COMPARE_SKIP_INITIAL_STEPS = 0
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
USER_INITIAL_RELAX_DAMPING = 8.0
USER_INITIAL_RELAX_TOL_SPEED = 1.0e-7
USER_SHOW_MESH_FACES = False
USER_SHOW_MESH_VERTICES = True
USER_SHOW_REAL_COMPUTE_TRIANGLES = True
USER_SHOW_SIDE_TRIANGLE_DIAGONALS = True
# "full" shows all real compute-mesh edges/vertices in PNGs and the
# interactive window. Switch to "meridian_strip" only when you explicitly
# want a Fig. 1-style section view instead of the full 3D graph.
USER_REAL_TRIANGLE_VIEW = "full"
USER_SHOW_CONTACT_RING_OVERLAY = False
USER_SHOW_SURFACE_OVERLAY = False
USER_SURFACE_OVERLAY_ALPHA = 0.58
USER_MESH_VERTEX_SIZE = 5.0
USER_WIREFRAME_MERIDIANS = 24
USER_MESH_ALPHA = 0.99
USER_CAP_EDGE_ALPHA = 0.0
USER_FILL_PARTICLE_CAP_SURFACES = False
USER_INCLUDE_GRAVITY = True
USER_GRAVITY_MPS2 = 9.81
USER_INTEGRATION_SUBSTEPS = 5
USER_CONTACT_RADIUS_SAMPLES = 128
# Gmsh path: use a discrete incompressible velocity projection on the loaded
# tetrahedral mesh instead of the old scalar Fp,proj volume-force controller.
USER_ENABLE_INCOMPRESSIBLE_PROJECTION = True
USER_INCOMPRESSIBLE_PROJECTION_REGULARIZATION = 1.0e-12
USER_ENABLE_VOLUME_PROJECTION = False
USER_ENABLE_PRESSURE_TRIAL_REFINEMENT = True
USER_VOLUME_PROJECTION_MAX_ITERS = 12
USER_VOLUME_PROJECTION_REL_TOL = 1.0e-6
USER_MAX_ACCELERATION = 5.0e-3
USER_ABORT_ON_NONFINITE_STATE = True
USER_ENABLE_GMSH_GEOMETRIC_VOLUME_CORRECTION = True
USER_AXISYM_SURFACE_VOLUME_CORRECTION_TRIGGER_REL = 5.0e-5
USER_AXISYM_SURFACE_VOLUME_CORRECTION_MAX_RADIAL_STEP_UM = 1.0
USER_AXISYM_SURFACE_VOLUME_CORRECTION_MAX_REL_RADIUS_STEP = 2.0e-3
USER_AXISYM_SURFACE_VOLUME_CORRECTION_MIN_TET_VOLUME_FRACTION = 0.25
USER_GMSH_GEOMETRIC_VOLUME_CORRECTION_TRIGGER_REL = USER_AXISYM_SURFACE_VOLUME_CORRECTION_TRIGGER_REL
USER_MAX_VOLUME_REL_ERROR_FOR_ABORT = 0.25
USER_ENABLE_CONTACT_LINE_VOLUME_SLIDE = True
USER_CONTACT_LINE_VOLUME_SLIDE_TRIGGER_REL = 5.0e-5
USER_CONTACT_LINE_VOLUME_SLIDE_MIN_RADIUS_FRACTION = 0.15
USER_CONTACT_LINE_VOLUME_SLIDE_MAX_RADIUS_FRACTION = 1.05
USER_ENABLE_POSITION_VOLUME_CONSTRAINT = True
USER_POSITION_VOLUME_CONSTRAINT_TRIGGER_REL = 1.0e-5
USER_POSITION_VOLUME_CONSTRAINT_MAX_ITERS = 12
# Eq. 3 pressure is evaluated from a local quadratic neck fit.  The single
# waist ring is excluded because its discrete cusp otherwise dominates d2r/dz2.
USER_PRESSURE_NECK_FIT_SIDE_RINGS = 2
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

USER_OPEN_INTERACTIVE_WINDOW = True
USER_INTERACTIVE_STEP = USER_TOTAL_STEPS

USER_INTERACTIVE_ELEV_DEG =  25.0 #0#
USER_INTERACTIVE_AZIM_DEG =  45.0 #-87#


def _pitois_eq6_wetted_radius(
    *,
    particle_radius_m: float,
    bridge_volume_m3: float,
    initial_d_over_r: float,
) -> float:
    """Compute the wetted/contact radius b from Pitois et al. Eq. [6].

    Source: Pitois, Moucheront, Chateau, J. Colloid Interface Sci. 231
    (2000), Fig. 5 and Eq. [6].  Fig. 5 gives the experiment as a bridge of
    liquid 2 with volume V = 1.1 mm^3 between ruby spheres of radius R = 4 mm.
    The paper does not give a separate fixed contact radius.  Eq. [6] gives
    the cylindrical/flat-profile bridge volume

        V = (pi R / 2) * (H(b)^2 - D^2),  H(b) = D + b^2 / R,

    so the contact/wetted radius b is computed from R, V, and the starting
    separation D = (D/R)R.  With this Case_2b starting D/R this gives
    b = 1.207424315572 mm; using the first digitized Fig. 5 point gives
    b = 1.207887812431 mm.
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
    # Starting normalized surface gap. This is the existing Case_2b start value,
    # close to the first digitized Fig. 5 point D/R = 0.014403481854782.
    initial_d_over_r: float = 0.014484537138212631
    # Pitois Fig. 5 bridge volume: V = 1.1 mm^3 = 1.10e-9 m^3.
    initial_bridge_volume_m3: float = 1.10e-9
    # Pitois Fig. 5 gives R, V, and D/R, not a contact radius directly.
    # Therefore the default target_cap_radius is not a hard-coded 1.54 mm value;
    # it is computed as the wetted radius b from Pitois Eq. [6].
    use_pitois_eq6_contact_radius: bool = True
    target_cap_radius: float = 0.0
    initial_neck_radius_ratio: float = USER_INITIAL_NECK_RADIUS_RATIO
    enable_initial_relaxation: bool = USER_ENABLE_INITIAL_RELAXATION
    initial_relax_steps: int = USER_INITIAL_RELAX_STEPS
    initial_relax_dt: float = USER_INITIAL_RELAX_DT_S
    initial_relax_damping: float = USER_INITIAL_RELAX_DAMPING
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
    damping: float = 2.0e-1
    max_acceleration: float = USER_MAX_ACCELERATION
    integration_substeps: int = USER_INTEGRATION_SUBSTEPS
    record_every: int = USER_RECORD_EVERY_STEPS
    mesh_snapshot_every: int = USER_MESH_SNAPSHOT_EVERY_STEPS
    display_motion_scale: float = 4000.0
    use_axisymmetric_laplace_pressure: bool = True
    pressure_scale: float = 1.0
    include_gravity: bool = USER_INCLUDE_GRAVITY
    gravity_mps2: float = USER_GRAVITY_MPS2
    contact_radius_samples: int = USER_CONTACT_RADIUS_SAMPLES
    enable_incompressible_projection: bool = USER_ENABLE_INCOMPRESSIBLE_PROJECTION
    incompressible_projection_regularization: float = USER_INCOMPRESSIBLE_PROJECTION_REGULARIZATION
    enable_volume_projection: bool = USER_ENABLE_VOLUME_PROJECTION
    volume_projection_max_iters: int = USER_VOLUME_PROJECTION_MAX_ITERS
    volume_projection_rel_tol: float = USER_VOLUME_PROJECTION_REL_TOL
    enable_gmsh_geometric_volume_correction: bool = USER_ENABLE_GMSH_GEOMETRIC_VOLUME_CORRECTION
    gmsh_geometric_volume_correction_trigger_rel: float = USER_GMSH_GEOMETRIC_VOLUME_CORRECTION_TRIGGER_REL
    max_volume_rel_error_for_abort: float = USER_MAX_VOLUME_REL_ERROR_FOR_ABORT
    pressure_neck_fit_side_rings: int = USER_PRESSURE_NECK_FIT_SIDE_RINGS
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
    pressure_projection_scalar: float = 0.0
    pressure_projection_area_map: dict[int, float] = field(default_factory=dict)
    pressure_projection_normal_map: dict[int, np.ndarray] = field(default_factory=dict)
    incompressible_projection_pressure: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    incompressible_divergence_before_l2: float = 0.0
    incompressible_divergence_after_l2: float = 0.0
    elapsed_time_s: float = 0.0
    last_step_dt: float = USER_DT_S
    last_bottom_contact_line_speed: float = 0.0
    last_top_contact_line_speed: float = 0.0
    last_dt_limit_cl: float = USER_DT_S
    last_dt_limit_capillary: float = USER_DT_S
    last_dt_limit_mesh: float = USER_DT_S
    last_dt_limiter: str = "fixed"
    last_contact_line_volume_delta_m3: float = 0.0
    last_gmsh_cl_volume_slide_status: str = "not_run"
    last_gmsh_cl_volume_slide_scale: float = 1.0
    last_gmsh_cl_volume_slide_rel_before: float = 0.0
    last_gmsh_cl_volume_slide_rel_after: float = 0.0
    last_position_volume_constraint_status: str = "not_run"
    last_position_volume_constraint_rel_before: float = 0.0
    last_position_volume_constraint_rel_after: float = 0.0
    last_axisym_surface_volume_correction_status: str = "not_run"
    last_axisym_surface_volume_correction_alpha_m: float = 0.0
    last_axisym_surface_volume_correction_rel_before: float = 0.0
    last_axisym_surface_volume_correction_rel_after: float = 0.0
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


def _canonicalize_ring_orders(state: VolumetricPitoisState) -> None:
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
    n_rings = int(n_rings)
    base = list(_radial_ring_factors_from_count(n_rings))
    extra_outer_rings = max(0, int(extra_outer_rings))
    outer_bias_ratio = max(1.0, float(outer_bias_ratio))
    if extra_outer_rings == 0 and abs(outer_bias_ratio - 1.0) <= 1.0e-12:
        return tuple(base)

    # If no extra rings are requested, bias can only move existing rings; it
    # cannot add visible node lines.  Keep this case available but do not use it
    # for CL refinement production meshes.
    if extra_outer_rings == 0 and abs(outer_bias_ratio - 1.0) > 1.0e-12:
        total_rings = max(1, n_rings + extra_outer_rings)
        gap_weights = np.array(
            [outer_bias_ratio ** power for power in range(total_rings - 1, -1, -1)],
            dtype=float,
        )
        factors = np.cumsum(gap_weights / float(np.sum(gap_weights)))
        factors[-1] = 1.0
        return tuple(float(round(factor, 12)) for factor in factors)

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
    pressure_scalar: float = 0.0


def _sidewall_layer_fractions_with_contact_refinement(
    n_layers: int,
    contact_bias_ratio: float,
) -> tuple[float, ...]:
    """Symmetric sidewall fractions with refinement at both CLs and the neck."""
    n_layers = int(n_layers)
    if n_layers <= 2:
        return tuple(np.linspace(0.0, 1.0, max(n_layers, 1), dtype=float))

    ratio = max(1.0, float(contact_bias_ratio))
    if abs(ratio - 1.0) <= 1.0e-12:
        return tuple(float(v) for v in np.linspace(0.0, 1.0, n_layers, dtype=float))

    def half_gap_weights(n_gaps: int) -> np.ndarray:
        # Small gaps at the contact line and the neck; largest gap halfway
        # between them.  Neighboring gaps still follow the user-requested ratio.
        center = 0.5 * float(max(n_gaps - 1, 0))
        exponents = [int(round(center - abs(float(i) - center))) for i in range(n_gaps)]
        return np.array([ratio ** power for power in exponents], dtype=float)

    if n_layers % 2 == 1:
        mid = n_layers // 2
        weights = half_gap_weights(mid)
        half_gaps = 0.5 * weights / float(np.sum(weights))
        fractions = np.empty(n_layers, dtype=float)
        fractions[0] = 0.0
        fractions[mid] = 0.5
        lower = np.cumsum(half_gaps)
        fractions[1 : mid + 1] = lower
        fractions[mid + 1 : -1] = 1.0 - lower[-2::-1]
    else:
        half = n_layers // 2
        weights = half_gap_weights(half)
        half_gaps = 0.5 * weights / float(np.sum(weights))
        lower = np.cumsum(half_gaps)
        fractions = np.concatenate([[0.0], lower, 1.0 - lower[-2::-1]])

    fractions[0] = 0.0
    fractions[-1] = 1.0
    return tuple(float(np.clip(v, 0.0, 1.0)) for v in fractions)


def _redistribute_sidewall_layers_axially(
    HC,
    outer_rings: list,
    layer_rings: list,
    layer_fractions: tuple[float, ...],
) -> None:
    if len(outer_rings) != len(layer_rings) or len(outer_rings) != len(layer_fractions):
        return
    if len(outer_rings) <= 2:
        return

    bottom_center, top_center, axis = _contact_plane_centers_and_axis(SimpleNamespace(outer_rings=outer_rings))
    total_span = float(np.dot(top_center - bottom_center, axis))
    if not np.isfinite(total_span) or total_span <= 1.0e-30:
        return

    groups: list[list[object]] = []
    centers: list[np.ndarray] = []
    for rings in layer_rings:
        z_ref = float(np.mean([float(v.x_a[2]) for v in rings[-1]]))
        same_layer = [v for v in HC.V if abs(float(v.x_a[2]) - z_ref) < 1.0e-12]
        seen: set[int] = set()
        group: list[object] = []
        for vertex in same_layer:
            key = id(vertex)
            if key in seen:
                continue
            seen.add(key)
            group.append(vertex)
        groups.append(group)
        centers.append(np.mean([np.asarray(v.x_a[:3], dtype=float) for v in rings[-1]], axis=0))

    for group, center, fraction in zip(groups, centers, layer_fractions):
        target_center = bottom_center + float(fraction) * total_span * axis
        delta = target_center - center
        if float(np.linalg.norm(delta)) <= 1.0e-18:
            continue
        for vertex in group:
            target = np.asarray(vertex.x_a[:3], dtype=float) + delta
            _move(vertex, tuple(target), HC, set())


def separation_config() -> VolumetricPitoisConfig:
    return VolumetricPitoisConfig(
        name="Case_2b_axisym_pitois2000_separation_Gmsh",
        title="Case 2b axisym: volumetric pre-bridged separation",
    )


def _mesh_volume_m3(HC) -> float:
    if hasattr(HC, "gmsh_volume_vertices") and hasattr(HC, "gmsh_volume_tets"):
        vertices = list(getattr(HC, "gmsh_volume_vertices"))
        points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
        return float(_indexed_tet_mesh_volume_m3(points, np.asarray(getattr(HC, "gmsh_volume_tets"), dtype=int)))
    return float(sum(float(dual_volume(v, HC, dim=3)) for v in HC.V))


def _cap_z_span(vertices: list) -> tuple[float, float]:
    z_vals = [float(v.x_a[2]) for v in vertices]
    return min(z_vals), max(z_vals)


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
    those updates one-by-one causes the vertex cache to evict the later vertex from
    the cache. A temporary staging move avoids that failure mode.
    """
    vertices = list(vertices)
    targets = [tuple(map(float, target)) for target in targets]
    if not vertices:
        return
    if len(vertices) != len(targets):
        raise ValueError("vertices and targets must have the same length")
    if isinstance(getattr(HC, "V", None), _DisplayVertexStore) and len(vertices) >= 128:
        all_vertices = list(HC.V.values())
        for v, target in zip(vertices, targets):
            v.x_a = np.asarray(target, dtype=float)
        HC.V.clear()
        for v in all_vertices:
            HC.V[tuple(float(x) for x in np.asarray(v.x_a[:3], dtype=float))] = v
        return
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


def _update_duals_and_masses(state: VolumetricPitoisState) -> None:
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


def _surface_tension_force_heron(v, *, dim: int, state: VolumetricPitoisState) -> np.ndarray:
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


def _multiphase_stress_force_cached(
    v,
    *,
    dim: int,
    state: VolumetricPitoisState,
    pressure_model=None,
) -> np.ndarray:
    F = np.zeros(dim)
    if state.HC is not None and hasattr(v, "vd"):
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
            F -= 0.5 * (p_i + p_j) * A_ij
            delta_u = v_j.u[:dim] - u_i
            d_ij = v_j.x_a[:dim] - x_i
            d_norm = np.linalg.norm(d_ij)
            if d_norm < 1.0e-30:
                continue
            d_hat = d_ij / d_norm
            F += (mu / d_norm) * delta_u * np.dot(d_hat, A_ij)
    F += _surface_tension_force_heron(v, dim=dim, state=state)
    return F


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


def _axisymmetric_neck_curvature_from_eq3_fit(
    z: np.ndarray,
    r: np.ndarray,
    *,
    side_rings: int,
) -> float | None:
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    valid = np.isfinite(z) & np.isfinite(r) & (r > 0.0)
    z = z[valid]
    r = r[valid]
    if z.size < 5:
        return None

    order = np.argsort(z)
    z = z[order]
    r = r[order]
    waist_idx = int(np.argmin(r))
    side = max(1, int(side_rings))
    lo = max(0, waist_idx - side)
    hi = min(len(z), waist_idx + side + 1)
    fit_indices = [idx for idx in range(lo, hi) if idx != waist_idx]

    if len(fit_indices) < 3:
        lo = max(0, waist_idx - side)
        hi = min(len(z), waist_idx + side + 1)
        fit_indices = list(range(lo, hi))
    if len(fit_indices) < 3:
        return None

    z_ref = float(z[waist_idx])
    zz = z[np.asarray(fit_indices, dtype=int)] - z_ref
    rr = r[np.asarray(fit_indices, dtype=int)]
    if np.unique(np.round(zz, 15)).size < 3:
        return None

    try:
        a, b, c = np.polyfit(zz, rr, 2)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None

    r_fit = float(c)
    if (not np.isfinite(r_fit)) or r_fit <= 1.0e-12:
        return None
    dr_dz = float(b)
    d2r_dz2 = float(2.0 * a)
    denom = max(1.0 + dr_dz * dr_dz, 1.0e-12)
    kappa_meridional = -d2r_dz2 / float(np.power(denom, 1.5))
    kappa_azimuthal = 1.0 / max(r_fit * float(np.sqrt(denom)), 1.0e-12)
    kappa = kappa_meridional + kappa_azimuthal
    if not np.isfinite(kappa):
        return None
    return float(kappa)


def _update_pressure_scalar(state: VolumetricPitoisState) -> None:
    if not state.config.use_axisymmetric_laplace_pressure:
        state.pressure_scalar = 0.0
        return

    z, r, kappa = _axisymmetric_profile(state)
    if z.size < 3:
        state.pressure_scalar = 0.0
        return

    kappa_fit = _axisymmetric_neck_curvature_from_eq3_fit(
        z,
        r,
        side_rings=int(getattr(state.config, "pressure_neck_fit_side_rings", USER_PRESSURE_NECK_FIT_SIDE_RINGS)),
    )
    if kappa_fit is not None:
        state.pressure_scalar = float(state.config.pressure_scale * state.config.gamma * kappa_fit)
        return

    center = len(kappa) // 2
    lo = max(0, center - 2)
    hi = min(len(kappa), center + 3)
    kappa_sample = kappa[lo:hi]
    if kappa_sample.size == 0 or not np.all(np.isfinite(kappa_sample)):
        state.pressure_scalar = 0.0
        return
    state.pressure_scalar = float(state.config.pressure_scale * state.config.gamma * np.mean(kappa_sample))


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


def _volume_gradient_areas_and_normals(
    state: VolumetricPitoisState,
) -> tuple[list[object], dict[int, float], dict[int, np.ndarray]]:
    _freeze_surface_topology(state)
    points = _volume_points_array(state)
    tets = np.asarray(state.volume_export_tets, dtype=int)
    vertices = list(state.volume_export_vertices)
    if points.size == 0 or tets.size == 0 or not vertices:
        return [], {}, {}

    grad_by_idx = np.zeros_like(points, dtype=float)
    for a, b, c, d in tets:
        ia = int(a)
        ib = int(b)
        ic = int(c)
        idd = int(d)
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        pd = points[idd]
        signed_six_volume = float(np.dot(pa - pd, np.cross(pb - pd, pc - pd)))
        if abs(signed_six_volume) <= 1.0e-30:
            continue
        orient = 1.0 if signed_six_volume >= 0.0 else -1.0
        ga = orient * np.cross(pb - pd, pc - pd) / 6.0
        gb = orient * np.cross(pc - pd, pa - pd) / 6.0
        gc = orient * np.cross(pa - pd, pb - pd) / 6.0
        gd = -(ga + gb + gc)
        grad_by_idx[ia] += ga
        grad_by_idx[ib] += gb
        grad_by_idx[ic] += gc
        grad_by_idx[idd] += gd

    active_vertices: list[object] = []
    area_map: dict[int, float] = {}
    normal_map: dict[int, np.ndarray] = {}
    for idx, vertex in enumerate(vertices):
        grad = np.asarray(grad_by_idx[idx], dtype=float)
        area = float(np.linalg.norm(grad))
        if area <= 1.0e-30:
            continue
        vid = id(vertex)
        active_vertices.append(vertex)
        area_map[vid] = area
        normal_map[vid] = grad / area
    return active_vertices, area_map, normal_map


def _Fp_proj(v, *, state: VolumetricPitoisState) -> np.ndarray:
    p_proj = float(getattr(state, "pressure_projection_scalar", 0.0))
    if abs(p_proj) <= 1.0e-30:
        return np.zeros(3, dtype=float)
    area = float(state.pressure_projection_area_map.get(id(v), 0.0))
    normal = state.pressure_projection_normal_map.get(id(v))
    if area <= 0.0 or normal is None:
        return np.zeros(3, dtype=float)
    return p_proj * area * np.asarray(normal, dtype=float)


def _update_pressure_projection_scalar(state: VolumetricPitoisState, *, dt: float) -> None:
    state.pressure_projection_scalar = 0.0
    state.pressure_projection_area_map = {}
    state.pressure_projection_normal_map = {}
    if (not state.config.enable_volume_projection) or dt <= 0.0:
        return

    # Fp,proj is a physical pressure-traction correction.  It acts on the
    # liquid boundary only: liquid-gas sidewall plus wetted sphere caps.  The old
    # tetra-volume gradient map touched interior nodes, which is not a physical
    # pressure traction.
    boundary_vertices, area_map, normal_map = _boundary_dual_areas_and_normals(state)
    if not boundary_vertices:
        return

    current_volume = float(_snapshot_msh_volume_m3(state))
    target_volume = float(getattr(state, "target_snapshot_volume_m3", current_volume))
    predicted_contact_line_delta = 0.0
    if bool(USER_ENFORCE_FULL_AXISYMMETRY):
        predicted_contact_line_delta = float(getattr(state, "last_contact_line_volume_delta_m3", 0.0))
    desired_dVdt = (target_volume - current_volume - predicted_contact_line_delta) / float(dt)
    pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
    dVdt_pred = 0.0
    C = 0.0
    for v in boundary_vertices:
        vid = id(v)
        area = float(area_map.get(vid, 0.0))
        normal = normal_map.get(vid)
        if area <= 0.0 or normal is None:
            continue

        if v in state.bV_caps:
            u_pred = np.asarray(v.u[:3], dtype=float)
        else:
            Ftot = _Ftot(
                v,
                state=state,
                pressure_model=pressure_model,
                include_projected_pressure=False,
                include_contact_line=True,
                include_damping=True,
            )
            accel = Ftot / max(float(getattr(v, "m", 0.0)), 1.0e-12)
            accel = _clip_acceleration(accel, state)
            u_pred = np.asarray(v.u[:3], dtype=float) + float(dt) * accel
            C += (area * area) / max(float(getattr(v, "m", 0.0)), 1.0e-12)

        dVdt_pred += area * float(np.dot(u_pred, normal))

    if C <= 1.0e-30:
        return

    state.pressure_projection_scalar = float((desired_dVdt - dVdt_pred) / (float(dt) * C))
    state.pressure_projection_area_map = area_map
    state.pressure_projection_normal_map = normal_map


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


def _resplit_contact_axial_layers(state: VolumetricPitoisState) -> None:
    """Keep extra liquid-gas sidewall layers biased toward both contact lines."""
    extra_total = max(0, int(getattr(state.config, "cl_extra_axial_layers", 0)))
    if extra_total <= 0 or len(state.layer_rings) < 4:
        return

    lower_extra = (extra_total + 1) // 2
    upper_extra = extra_total // 2
    ratio = max(1.0, float(state.config.contact_line_radial_bias_ratio))

    def split(indices: list[int], *, cl_at_start: bool) -> None:
        if len(indices) < 3:
            return
        cl_to_anchor = list(indices) if cl_at_start else list(reversed(indices))
        gaps = len(cl_to_anchor) - 1
        weights = ratio ** np.arange(gaps, dtype=float)
        fractions = np.concatenate([[0.0], np.cumsum(weights / float(np.sum(weights)))])
        cl_idx = cl_to_anchor[0]
        anchor_idx = cl_to_anchor[-1]
        endpoint_layers = [
            [state.layer_centers[cl_idx]],
            *state.layer_rings[cl_idx],
        ]
        anchor_layers = [
            [state.layer_centers[anchor_idx]],
            *state.layer_rings[anchor_idx],
        ]
        for idx, frac in zip(cl_to_anchor[1:-1], fractions[1:-1]):
            target_groups = [
                [state.layer_centers[idx]],
                *state.layer_rings[idx],
            ]
            for target_group, cl_group, anchor_group in zip(target_groups, endpoint_layers, anchor_layers):
                targets = []
                for v_cl, v_anchor in zip(cl_group, anchor_group):
                    x_cl = np.asarray(v_cl.x_a[:3], dtype=float)
                    x_anchor = np.asarray(v_anchor.x_a[:3], dtype=float)
                    targets.append(tuple((1.0 - float(frac)) * x_cl + float(frac) * x_anchor))
                _move_vertices_batch(target_group, targets, state.HC, state.bV_caps)

    if lower_extra > 0 and lower_extra + 1 < len(state.layer_rings):
        split(list(range(0, lower_extra + 2)), cl_at_start=True)
    if upper_extra > 0 and len(state.layer_rings) - upper_extra - 2 >= 0:
        split(
            list(range(len(state.layer_rings) - upper_extra - 2, len(state.layer_rings))),
            cl_at_start=False,
        )


def _pressure_model(v, *, HC=None, dim: int = 3, state: VolumetricPitoisState):
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


def _ring_center_and_radius(ring: list, center: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, float]:
    ring_coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
    ring_center = np.mean(ring_coords, axis=0)
    tangential = ring_coords - center[None, :]
    tangential -= np.outer(np.dot(tangential, axis), axis)
    radii = np.linalg.norm(tangential, axis=1)
    return ring_center, float(np.mean(radii))


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
    inward_sign = 1.0 if which == "bottom" else -1.0
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
    return float(np.arccos(dot))


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
    # Surface tension in this case is supplied only by the validated Heron
    # operator path, via _surface_tension_force_heron().
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


def _apply_axisymmetric_outer_radii_profile(
    state: VolumetricPitoisState,
    *,
    template,
    outer_radii: np.ndarray,
) -> None:
    mid, axis, e1, e2, layer_data = template
    contact_radius = float(state.config.target_cap_radius)

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


def _apply_initial_unduloid_like_profile(
    state: VolumetricPitoisState,
    *,
    template,
    neck_radius: float,
) -> None:
    s_vals = np.array([float(item["s"]) for item in template[4]], dtype=float)
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
    _apply_axisymmetric_outer_radii_profile(
        state,
        template=template,
        outer_radii=outer_radii,
    )


def _apply_initial_power_pinch_profile(
    state: VolumetricPitoisState,
    *,
    template,
    exponent: float,
    neck_radius: float,
) -> None:
    s_vals = np.array([float(item["s"]) for item in template[4]], dtype=float)
    half_span = float(np.max(np.abs(s_vals)))
    contact_radius = float(state.config.target_cap_radius)
    if half_span <= 1.0e-30:
        outer_radii = np.full_like(s_vals, contact_radius, dtype=float)
    else:
        u = np.clip(np.abs(s_vals) / half_span, 0.0, 1.0)
        outer_radii = neck_radius + (contact_radius - neck_radius) * np.power(u, float(exponent))
        outer_radii = np.clip(outer_radii, neck_radius, contact_radius)
    _apply_axisymmetric_outer_radii_profile(
        state,
        template=template,
        outer_radii=outer_radii,
    )


def _initialise_unduloid_like_equilibrium(state: VolumetricPitoisState) -> None:
    template = _initial_axisymmetric_template(state)
    contact_radius = float(state.config.target_cap_radius)
    target_volume = float(state.config.initial_bridge_volume_m3)
    _freeze_volume_msh_topology(state)

    def _apply_ratio_and_volume(neck_ratio: float) -> float:
        ratio = float(np.clip(neck_ratio, 0.05, 0.98))
        _apply_initial_unduloid_like_profile(
            state,
            template=template,
            neck_radius=ratio * contact_radius,
        )
        _update_duals_and_masses(state)
        return float(
            _indexed_tet_mesh_volume_m3(
                _volume_points_array(state),
                np.asarray(state.volume_export_tets, dtype=int),
            )
        )

    neck_ratio_seed = float(np.clip(state.config.initial_neck_radius_ratio, 0.05, 0.98))
    ratio_low = 0.05
    ratio_high = 0.98
    volume_low = _apply_ratio_and_volume(ratio_low)
    volume_high = _apply_ratio_and_volume(ratio_high)

    solved = False
    if min(volume_low, volume_high) <= target_volume <= max(volume_low, volume_high):
        low = ratio_low
        high = ratio_high
        increasing = volume_high >= volume_low
        best_ratio = neck_ratio_seed
        best_err = float("inf")
        for _ in range(32):
            mid = 0.5 * (low + high)
            vol_mid = _apply_ratio_and_volume(mid)
            err = abs(vol_mid - target_volume)
            if err < best_err:
                best_ratio = mid
                best_err = err
            if err / max(target_volume, 1.0e-30) <= 1.0e-8:
                best_ratio = mid
                break
            if increasing:
                if vol_mid < target_volume:
                    low = mid
                else:
                    high = mid
            else:
                if vol_mid > target_volume:
                    low = mid
                else:
                    high = mid
        _apply_ratio_and_volume(best_ratio)
        solved = True

    if not solved:
        # The quartic family is too fat for the requested bridge volume under the
        # positive-tetra volume definition. Fall back to a sharper analytic
        # power-law pinch profile and solve its exponent to hit the target.
        min_neck_radius = max(1.0e-3 * contact_radius, 1.0e-8)

        def _apply_power_and_volume(exponent: float) -> float:
            _apply_initial_power_pinch_profile(
                state,
                template=template,
                exponent=float(exponent),
                neck_radius=min_neck_radius,
            )
            _update_duals_and_masses(state)
            return float(
                _indexed_tet_mesh_volume_m3(
                    _volume_points_array(state),
                    np.asarray(state.volume_export_tets, dtype=int),
                )
            )

        exp_low = 0.5
        exp_high = 32.0
        vol_exp_low = _apply_power_and_volume(exp_low)
        vol_exp_high = _apply_power_and_volume(exp_high)
        if min(vol_exp_low, vol_exp_high) <= target_volume <= max(vol_exp_low, vol_exp_high):
            low = exp_low
            high = exp_high
            decreasing = vol_exp_high <= vol_exp_low
            best_exp = exp_high
            best_err = float("inf")
            for _ in range(40):
                mid = 0.5 * (low + high)
                vol_mid = _apply_power_and_volume(mid)
                err = abs(vol_mid - target_volume)
                if err < best_err:
                    best_exp = mid
                    best_err = err
                if err / max(target_volume, 1.0e-30) <= 1.0e-8:
                    best_exp = mid
                    break
                if decreasing:
                    if vol_mid > target_volume:
                        low = mid
                    else:
                        high = mid
                else:
                    if vol_mid < target_volume:
                        low = mid
                    else:
                        high = mid
            _apply_power_and_volume(best_exp)
        else:
            # As a last fallback, keep the sharpest available analytic profile.
            _apply_power_and_volume(exp_high)
    _update_duals_and_masses(state)


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
    initial_msh_path = _required_initial_msh_path()
    points, triangles, tets, _node_tags = _read_gmsh4_points_triangles_tets(initial_msh_path)
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
    state.loaded_initial_msh_path = str(initial_msh_path)
    state.volume_export_vertices = list(vertices)
    state.volume_export_node_ids = {id(vertex): idx for idx, vertex in enumerate(vertices)}
    state.volume_export_tets = np.asarray(tets, dtype=int)
    state.surface_export_vertices = list(vertices)
    state.surface_export_node_ids = {id(vertex): idx for idx, vertex in enumerate(vertices)}
    state.surface_export_side_tris = np.asarray(side_tris, dtype=int)
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


def _isotonic_increasing_fit(values: np.ndarray) -> np.ndarray:
    vals = [float(v) for v in np.asarray(values, dtype=float)]
    if not vals:
        return np.empty(0, dtype=float)

    blocks: list[list[float]] = []
    for value in vals:
        blocks.append([value, 1.0, 1.0])  # mean, weight, count
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            mean0, weight0, count0 = blocks[-2]
            mean1, weight1, count1 = blocks[-1]
            weight = weight0 + weight1
            mean = (mean0 * weight0 + mean1 * weight1) / weight
            blocks[-2] = [mean, weight, count0 + count1]
            blocks.pop()

    fitted: list[float] = []
    for mean, _weight, count in blocks:
        fitted.extend([float(mean)] * int(count))
    return np.asarray(fitted, dtype=float)


def _single_neck_projected_radii(radii: np.ndarray) -> np.ndarray:
    rs = np.asarray(radii, dtype=float)
    if rs.size <= 2:
        return rs.copy()

    neck_idx = int(np.argmin(rs))
    left = rs[: neck_idx + 1]
    right = rs[neck_idx:]

    left_fit = _isotonic_increasing_fit(left[::-1])[::-1]
    right_fit = _isotonic_increasing_fit(right)
    neck_radius = min(float(left_fit[-1]), float(right_fit[0]))
    left_fit[-1] = neck_radius
    right_fit[0] = neck_radius

    return np.concatenate([left_fit[:-1], right_fit])


def _scale_free_vertices_radially(
    state: VolumetricPitoisState,
    *,
    radial_scale: float,
) -> None:
    if abs(radial_scale - 1.0) <= 1.0e-12:
        return

    axis = state.top_sphere_center - state.bottom_sphere_center
    axis_norm = float(np.linalg.norm(axis))
    if (not np.isfinite(axis_norm)) or axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    mid = 0.5 * (state.bottom_sphere_center + state.top_sphere_center)

    for v in state.HC.V:
        if v in state.bV_caps:
            continue
        pos = np.asarray(v.x_a[:3], dtype=float)
        rel = pos - mid
        axial = axis * float(np.dot(rel, axis))
        radial = rel - axial
        target = mid + axial + radial_scale * radial
        _move(v, tuple(target), state.HC, state.bV_caps)


def _free_vertex_positions(state: VolumetricPitoisState) -> tuple[list, np.ndarray]:
    free_vertices = [v for v in state.HC.V if v not in state.bV_caps]
    if not free_vertices:
        return [], np.empty((0, 3), dtype=float)
    positions = np.array([np.asarray(v.x_a[:3], dtype=float) for v in free_vertices], dtype=float)
    return free_vertices, positions


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


def _radially_scaled_positions(
    state: VolumetricPitoisState,
    positions: np.ndarray,
    *,
    radial_scale: float,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return np.empty((0, 3), dtype=float)
    if (not np.all(np.isfinite(positions))) or float(np.max(np.abs(positions))) > 1.0:
        raise ValueError("rejecting non-physical trial coordinates during radial volume scaling")

    axis = state.top_sphere_center - state.bottom_sphere_center
    axis_norm = float(np.linalg.norm(axis))
    if (not np.isfinite(axis_norm)) or axis_norm <= 1.0e-30:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm
    mid = 0.5 * (state.bottom_sphere_center + state.top_sphere_center)
    if (not np.all(np.isfinite(mid))) or float(np.max(np.abs(mid))) > 1.0:
        raise ValueError("rejecting non-physical sphere centers during radial volume scaling")

    rel = positions - mid[None, :]
    axial_mag = np.sum(rel * axis[None, :], axis=1)
    axial = axial_mag[:, None] * axis[None, :]
    radial = rel - axial
    return mid[None, :] + axial + float(radial_scale) * radial


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
        rel_tol = float(state.config.volume_projection_rel_tol)
    if max_iters is None:
        max_iters = max(16, int(state.config.volume_projection_max_iters))

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


def _project_volume_to_target(state: VolumetricPitoisState) -> None:
    if not state.config.enable_volume_projection:
        return

    target = float(state.target_snapshot_volume_m3)
    if not np.isfinite(target) or target <= 0.0:
        return

    current = _snapshot_msh_volume_m3(state)
    if not np.isfinite(current) or current <= 0.0:
        return
    rel_err = abs(current - target) / max(target, 1.0e-30)
    if rel_err <= float(state.config.volume_projection_rel_tol):
        return

    projectable_vertices, base_positions = _snapshot_projectable_vertex_positions(state)
    if not projectable_vertices:
        return

    def _volume_at(scale: float) -> float:
        trial_positions = _radially_scaled_positions(state, base_positions, radial_scale=scale)
        _apply_free_vertex_positions(state, projectable_vertices, trial_positions)
        trial_volume = _snapshot_msh_volume_m3(state)
        _apply_free_vertex_positions(state, projectable_vertices, base_positions)
        return float(trial_volume)

    low = 1.0
    high = 1.0
    if current < target:
        high = 1.10
        vol_high = _volume_at(high)
        while vol_high < target and high < 8.0:
            high *= 1.25
            vol_high = _volume_at(high)
        if vol_high < target:
            return
    else:
        low = 0.80
        vol_low = _volume_at(low)
        while vol_low > target and low > 0.05:
            low *= 0.75
            vol_low = _volume_at(low)
        if vol_low > target:
            return
        high = 1.0

    n_bisect = max(1, int(state.config.volume_projection_max_iters))
    for _ in range(n_bisect):
        mid = 0.5 * (low + high)
        vol_mid = _volume_at(mid)
        if abs(vol_mid - target) / max(target, 1.0e-30) <= float(state.config.volume_projection_rel_tol):
            low = mid
            high = mid
            break
        if vol_mid < target:
            low = mid
        else:
            high = mid

    final_scale = 0.5 * (low + high)
    final_positions = _radially_scaled_positions(state, base_positions, radial_scale=final_scale)
    _apply_free_vertex_positions(state, projectable_vertices, final_positions)


def _force_snapshot_volume_to_target(
    state: VolumetricPitoisState,
    *,
    target_m3: float,
    rel_tol: float | None = None,
    max_iters: int | None = None,
) -> None:
    target = float(target_m3)
    if not np.isfinite(target) or target <= 0.0:
        return

    current = _snapshot_msh_volume_m3(state)
    if not np.isfinite(current) or current <= 0.0:
        return

    tol = float(state.config.volume_projection_rel_tol if rel_tol is None else rel_tol)
    if abs(current - target) / max(target, 1.0e-30) <= tol:
        return

    projectable_vertices, base_positions = _snapshot_projectable_vertex_positions(state)
    if not projectable_vertices:
        return

    def _volume_at(scale: float) -> float:
        trial_positions = _radially_scaled_positions(state, base_positions, radial_scale=scale)
        _apply_free_vertex_positions(state, projectable_vertices, trial_positions)
        trial_volume = _snapshot_msh_volume_m3(state)
        _apply_free_vertex_positions(state, projectable_vertices, base_positions)
        return float(trial_volume)

    low = 1.0
    high = 1.0
    if current < target:
        high = 1.10
        vol_high = _volume_at(high)
        while vol_high < target and high < 8.0:
            high *= 1.25
            vol_high = _volume_at(high)
        if vol_high < target:
            return
    else:
        low = 0.80
        vol_low = _volume_at(low)
        while vol_low > target and low > 0.05:
            low *= 0.75
            vol_low = _volume_at(low)
        if vol_low > target:
            return
        high = 1.0

    n_bisect = max(1, int(state.config.volume_projection_max_iters if max_iters is None else max_iters))
    for _ in range(n_bisect):
        mid = 0.5 * (low + high)
        vol_mid = _volume_at(mid)
        if abs(vol_mid - target) / max(target, 1.0e-30) <= tol:
            low = mid
            high = mid
            break
        if vol_mid < target:
            low = mid
        else:
            high = mid

    final_scale = 0.5 * (low + high)
    final_positions = _radially_scaled_positions(state, base_positions, radial_scale=final_scale)
    _apply_free_vertex_positions(state, projectable_vertices, final_positions)


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

        for ring in rings:
            coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
            rel = coords - center_coord[None, :]
            rel -= np.outer(np.dot(rel, axis), axis)
            radius = float(np.mean(np.linalg.norm(rel, axis=1)))
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
        damping=float(original_config.initial_relax_damping),
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


def _update_moving_contact_line(state: VolumetricPitoisState, *, dt: float | None = None) -> None:
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


def _Ftot(
    v,
    *,
    state: VolumetricPitoisState,
    pressure_model=None,
    include_projected_pressure: bool = True,
    include_contact_line: bool = True,
    include_damping: bool = True,
) -> np.ndarray:
    if pressure_model is None:
        pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)

    # Ftot = F_sigma,p,mu + Fp,proj + Fcl - c u
    Ftot = _multiphase_stress_force_cached(
        v,
        dim=3,
        state=state,
        pressure_model=pressure_model,
    )
    if include_projected_pressure:
        Fp_proj = _Fp_proj(v, state=state)
        Ftot += Fp_proj
    if include_contact_line:
        Fcl = _Fcl(v, state=state)
        Ftot += Fcl
    if include_damping and state.config.damping > 0.0:
        Ftot -= state.config.damping * np.asarray(v.u[:3], dtype=float)
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
    accel = Ftot / max(float(getattr(v, "m", 0.0)), 1.0e-12)
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


def _axisym_clear_caches(state: VolumetricPitoisState) -> None:
    for attr in (
        "_axisym_force_cache",
        "_axisym_accel_cache",
        "_gmsh_axisym_force_cache",
        "_gmsh_axisym_accel_cache",
        "_gmsh_surface_heron_force_cache",
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
    include_projected_pressure: bool,
    include_contact_line: bool,
    include_damping: bool,
) -> tuple:
    return (
        bool(include_projected_pressure),
        bool(include_contact_line),
        bool(include_damping),
        round(float(getattr(state, "pressure_scalar", 0.0)), 18),
        round(float(getattr(state, "pressure_projection_scalar", 0.0)), 18),
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


def _gmsh_axisym_force_map(
    state: VolumetricPitoisState,
    *,
    pressure_model,
    include_projected_pressure: bool,
    include_contact_line: bool,
    include_damping: bool,
) -> dict[int, np.ndarray]:
    if bool(getattr(state.config, "enable_incompressible_projection", False)):
        include_projected_pressure = False
    cache = getattr(state, "_gmsh_axisym_force_cache", {})
    key = _axisym_force_cache_key(
        state,
        include_projected_pressure=include_projected_pressure,
        include_contact_line=include_contact_line,
        include_damping=include_damping,
    )
    if key in cache:
        return cache[key]

    free_vertices = [v for v in state.HC.V if v not in state.bV_caps]
    raw_map = {
        id(v): _BASE_Ftot(
            v,
            state=state,
            pressure_model=pressure_model,
            include_projected_pressure=include_projected_pressure,
            include_contact_line=include_contact_line,
            include_damping=include_damping,
        )
        for v in free_vertices
    }
    mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    averaged: dict[int, np.ndarray] = {}
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
            force_z = float(np.mean([float(np.dot(raw_map[id(v)], axis)) for v, _e_r in pairs]))
            for v, _e_r in pairs:
                averaged[id(v)] = force_z * axis
                covered.add(id(v))
            continue
        force_r = float(np.mean([float(np.dot(raw_map[id(v)], e_r)) for v, e_r in pairs]))
        force_z = float(np.mean([float(np.dot(raw_map[id(v)], axis)) for v, _e_r in pairs]))
        for v, e_r in pairs:
            averaged[id(v)] = force_r * e_r + force_z * axis
            covered.add(id(v))

    for v in free_vertices:
        vid = id(v)
        if vid in covered:
            continue
        averaged[vid] = _remove_swirl_component(
            raw_map[vid],
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=mid,
            axis=axis,
        )

    cache[key] = averaged
    state._gmsh_axisym_force_cache = cache
    return averaged


def _gmsh_axisym_accel_map(state: VolumetricPitoisState) -> dict[int, np.ndarray]:
    cache = getattr(state, "_gmsh_axisym_accel_cache", None)
    if cache is not None:
        return cache

    pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
    force_map_nonproj = _gmsh_axisym_force_map(
        state,
        pressure_model=pressure_model,
        include_projected_pressure=False,
        include_contact_line=True,
        include_damping=True,
    )
    force_map_total = _gmsh_axisym_force_map(
        state,
        pressure_model=pressure_model,
        include_projected_pressure=True,
        include_contact_line=True,
        include_damping=True,
    )
    mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    cache = {}
    covered: set[int] = set()
    for spec in getattr(state, "gmsh_axisym_ring_specs", []):
        vertices = list(getattr(spec, "vertices", []))
        units = _gmsh_axisym_spec_radial_units(spec, e1, e2)
        pairs = [
            (v, e_r)
            for v, e_r in zip(vertices, units)
            if (v not in state.bV_caps) and (id(v) in force_map_nonproj)
        ]
        if not pairs:
            continue
        mass_avg = float(np.mean([max(float(getattr(v, "m", 0.0)), 1.0e-12) for v, _e_r in pairs]))
        _s_mean, r_mean = _gmsh_axisym_group_axial_radius(
            [v for v, _e_r in pairs],
            center_coord=mid,
            axis=axis,
        )
        if len(pairs) == 1 or r_mean <= 1.0e-30:
            force_z_nonproj = float(np.mean([float(np.dot(force_map_nonproj[id(v)], axis)) for v, _e_r in pairs]))
            force_z_proj = float(
                np.mean(
                    [
                        float(np.dot(force_map_total[id(v)] - force_map_nonproj[id(v)], axis))
                        for v, _e_r in pairs
                    ]
                )
            )
            accel = _clip_acceleration((force_z_nonproj / mass_avg) * axis, state) + (force_z_proj / mass_avg) * axis
            accel = _clip_acceleration(accel, state)
            for v, _e_r in pairs:
                cache[id(v)] = accel
                covered.add(id(v))
            continue
        force_r_nonproj = float(np.mean([float(np.dot(force_map_nonproj[id(v)], e_r)) for v, e_r in pairs]))
        force_z_nonproj = float(np.mean([float(np.dot(force_map_nonproj[id(v)], axis)) for v, _e_r in pairs]))
        force_r_proj = float(
            np.mean(
                [
                    float(np.dot(force_map_total[id(v)] - force_map_nonproj[id(v)], e_r))
                    for v, e_r in pairs
                ]
            )
        )
        force_z_proj = float(
            np.mean(
                [
                    float(np.dot(force_map_total[id(v)] - force_map_nonproj[id(v)], axis))
                    for v, _e_r in pairs
                ]
            )
        )
        for v, e_r in pairs:
            accel_nonproj = (force_r_nonproj / mass_avg) * e_r + (force_z_nonproj / mass_avg) * axis
            accel_proj = (force_r_proj / mass_avg) * e_r + (force_z_proj / mass_avg) * axis
            cache[id(v)] = _clip_acceleration(_clip_acceleration(accel_nonproj, state) + accel_proj, state)
            covered.add(id(v))

    for v in [candidate for candidate in state.HC.V if (candidate not in state.bV_caps) and (id(candidate) in force_map_nonproj)]:
        vid = id(v)
        if vid in covered:
            continue
        mass = max(float(getattr(v, "m", 0.0)), 1.0e-12)
        accel_nonproj = force_map_nonproj[vid] / mass
        accel_proj = (force_map_total[vid] - force_map_nonproj[vid]) / mass
        accel = _clip_acceleration(accel_nonproj, state) + _remove_swirl_component(
            accel_proj,
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=mid,
            axis=axis,
        )
        cache[vid] = _clip_acceleration(accel, state)
    state._gmsh_axisym_accel_cache = cache
    return cache


_BASE_update_duals_and_masses = _update_duals_and_masses
_BASE_update_pressure_scalar = _update_pressure_scalar
_BASE_update_pressure_projection_scalar = _update_pressure_projection_scalar
_BASE_move_caps = _move_caps
_BASE_update_moving_contact_line = _update_moving_contact_line
_BASE_project_volume_to_target = _project_volume_to_target
_BASE_Ftot = _Ftot


def _refresh_layer_fractions_from_current_profile(state: VolumetricPitoisState) -> None:
    if len(state.outer_rings) < 2:
        return
    bottom_center, top_center, axis = _contact_plane_centers_and_axis(
        SimpleNamespace(outer_rings=state.outer_rings)
    )
    total_span = float(np.dot(top_center - bottom_center, axis))
    if total_span <= 1.0e-30:
        return
    fractions = []
    for ring in state.outer_rings:
        ring_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in ring], axis=0)
        fractions.append(float(np.dot(ring_center - bottom_center, axis)) / total_span)
    fractions[0] = 0.0
    fractions[-1] = 1.0
    state.layer_fractions = tuple(float(np.clip(f, 0.0, 1.0)) for f in fractions)


def _axisym_force_map(
    state: VolumetricPitoisState,
    *,
    pressure_model,
    include_projected_pressure: bool,
    include_contact_line: bool,
    include_damping: bool,
) -> dict[int, np.ndarray]:
    cache = getattr(state, "_axisym_force_cache", {})
    key = _axisym_force_cache_key(
        state,
        include_projected_pressure=include_projected_pressure,
        include_contact_line=include_contact_line,
        include_damping=include_damping,
    )
    if key in cache:
        return cache[key]

    mid, axis, e1, e2 = _axisym_basis(state)
    free_vertices = [v for v in state.HC.V if v not in state.bV_caps]
    workers = _axisym_force_workers(state)

    def raw_force(v) -> tuple[int, np.ndarray]:
        return id(v), _BASE_Ftot(
            v,
            state=state,
            pressure_model=pressure_model,
            include_projected_pressure=include_projected_pressure,
            include_contact_line=include_contact_line,
            include_damping=include_damping,
        )

    if workers > 1 and len(free_vertices) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            raw_items = list(pool.map(raw_force, free_vertices))
        raw_map = dict(raw_items)
    else:
        raw_map = dict(raw_force(v) for v in free_vertices)

    averaged: dict[int, np.ndarray] = {}
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
            force_r = float(
                np.mean([float(np.dot(raw_map[id(v)], e_r)) for v, e_r in zip(members, radial_units)])
            )
            force_z = float(np.mean([float(np.dot(raw_map[id(v)], axis)) for v in members]))
            for v, e_r in zip(members, radial_units):
                averaged[id(v)] = force_r * e_r + force_z * axis
                covered.add(id(v))

        center_v = state.layer_centers[k]
        vid = id(center_v)
        if vid in raw_map:
            force_z = float(np.dot(raw_map[vid], axis))
            averaged[vid] = force_z * axis
            covered.add(vid)

    for v in free_vertices:
        vid = id(v)
        if vid in covered:
            continue
        averaged[vid] = _remove_swirl_component(
            raw_map[vid],
            point=np.asarray(v.x_a[:3], dtype=float),
            axis_origin=mid,
            axis=axis,
        )

    cache[key] = averaged
    state._axisym_force_cache = cache
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
    _BASE_update_duals_and_masses(state)
    _axisym_clear_caches(state)


def _update_pressure_scalar(state: VolumetricPitoisState) -> None:
    _BASE_update_pressure_scalar(state)
    _axisym_clear_caches(state)


def _refine_axisym_pressure_projection_scalar_by_trial(
    state: VolumetricPitoisState,
    *,
    dt: float,
) -> None:
    if (not bool(USER_ENFORCE_FULL_AXISYMMETRY)) or (not state.config.enable_volume_projection):
        return
    if dt <= 0.0:
        return

    target_volume = float(getattr(state, "target_snapshot_volume_m3", 0.0))
    if (not np.isfinite(target_volume)) or target_volume <= 0.0:
        return

    p_initial = float(getattr(state, "pressure_projection_scalar", 0.0))
    vertices = list(state.HC.V)
    coords = [np.asarray(v.x_a[:3], dtype=float).copy() for v in vertices]
    velocities = [np.asarray(v.u[:3], dtype=float).copy() for v in vertices]
    saved_bV_caps = set(state.bV_caps)
    saved_cap_bottom_interior = list(state.cap_bottom_interior)
    saved_cap_top_interior = list(state.cap_top_interior)
    saved_bottom_speed = float(state.last_bottom_contact_line_speed)
    saved_top_speed = float(state.last_top_contact_line_speed)
    saved_cl_delta = float(getattr(state, "last_contact_line_volume_delta_m3", 0.0))

    def restore() -> None:
        state.bV_caps = set(saved_bV_caps)
        state.cap_bottom_interior = list(saved_cap_bottom_interior)
        state.cap_top_interior = list(saved_cap_top_interior)
        for vertex, coord, velocity in zip(vertices, coords, velocities):
            _move(vertex, tuple(coord), state.HC, state.bV_caps)
            vertex.u = velocity.copy()
        state.last_bottom_contact_line_speed = saved_bottom_speed
        state.last_top_contact_line_speed = saved_top_speed
        state.last_contact_line_volume_delta_m3 = saved_cl_delta
        _axisym_clear_caches(state)

    def trial_end_volume(p_proj: float) -> float:
        restore()
        state.pressure_projection_scalar = float(p_proj)
        _axisym_clear_caches(state)
        symplectic_euler(
            state.HC,
            state.bV_caps,
            _vertex_acceleration,
            dt=float(dt),
            n_steps=1,
            dim=3,
            retopologize_fn=False,
            state=state,
        )
        _enforce_no_swirl_velocity_field(state)
        _set_cap_velocities(state)
        _update_moving_contact_line(state, dt=dt)
        _enforce_no_swirl_velocity_field(state)
        return float(_snapshot_msh_volume_m3(state))

    p_a = 0.0
    p_b = p_initial if abs(p_initial) > 1.0e-14 else 1.0e-8
    try:
        v_a = trial_end_volume(p_a)
        v_b = trial_end_volume(p_b)
    finally:
        restore()

    if not (np.isfinite(v_a) and np.isfinite(v_b)):
        state.pressure_projection_scalar = p_initial
        _axisym_clear_caches(state)
        return

    denom = v_b - v_a
    if abs(denom) <= max(1.0e-18 * target_volume, 1.0e-30):
        state.pressure_projection_scalar = p_initial
        _axisym_clear_caches(state)
        return

    p_refined = p_a + (target_volume - v_a) * (p_b - p_a) / denom
    p_limit = 20.0 * max(abs(p_a), abs(p_b), 1.0e-8)
    state.pressure_projection_scalar = float(np.clip(p_refined, -p_limit, p_limit))
    _axisym_clear_caches(state)


def _update_pressure_projection_scalar(
    state: VolumetricPitoisState,
    *,
    dt: float,
    refine: bool = True,
) -> None:
    if bool(getattr(state.config, "enable_incompressible_projection", False)):
        state.pressure_projection_scalar = 0.0
        state.pressure_projection_area_map = {}
        state.pressure_projection_normal_map = {}
        _axisym_clear_caches(state)
        return
    _BASE_update_pressure_projection_scalar(state, dt=dt)
    if not np.isfinite(float(getattr(state, "pressure_projection_scalar", 0.0))):
        state.pressure_projection_scalar = 0.0
        state.pressure_projection_area_map = {}
        state.pressure_projection_normal_map = {}
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        _axisym_clear_caches(state)
        return
    if refine and bool(USER_ENABLE_PRESSURE_TRIAL_REFINEMENT):
        _refine_axisym_pressure_projection_scalar_by_trial(state, dt=dt)
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
    _BASE_move_caps(state, dt=dt)
    _axisymmetrize_full_velocity_field(state)
    _axisym_clear_caches(state)


def _update_moving_contact_line(state: VolumetricPitoisState, *, dt: float | None = None) -> None:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        if dt is None:
            dt = float(state.config.dt)
        before_volume = float(_snapshot_msh_volume_m3(state))
        prev_bottom_radius = float(_cap_radius(state.bottom_contact_ring)) if state.bottom_contact_ring else 0.0
        prev_top_radius = float(_cap_radius(state.top_contact_ring)) if state.top_contact_ring else 0.0
        _project_gmsh_contact_lines_to_spheres(state)
        after_volume = float(_snapshot_msh_volume_m3(state))
        bottom_radius = float(_cap_radius(state.bottom_contact_ring)) if state.bottom_contact_ring else 0.0
        top_radius = float(_cap_radius(state.top_contact_ring)) if state.top_contact_ring else 0.0
        sphere_radius = max(float(state.config.particle_radius), 1.0e-30)
        if float(dt) > 0.0:
            alpha_bottom_prev = float(np.arcsin(np.clip(prev_bottom_radius / sphere_radius, -1.0, 1.0)))
            alpha_bottom_new = float(np.arcsin(np.clip(bottom_radius / sphere_radius, -1.0, 1.0)))
            alpha_top_prev = float(np.arcsin(np.clip(prev_top_radius / sphere_radius, -1.0, 1.0)))
            alpha_top_new = float(np.arcsin(np.clip(top_radius / sphere_radius, -1.0, 1.0)))
            state.last_bottom_contact_line_speed = abs(sphere_radius * (alpha_bottom_new - alpha_bottom_prev) / float(dt))
            state.last_top_contact_line_speed = abs(sphere_radius * (alpha_top_new - alpha_top_prev) / float(dt))
        else:
            state.last_bottom_contact_line_speed = 0.0
            state.last_top_contact_line_speed = 0.0
        state.last_contact_line_volume_delta_m3 = after_volume - before_volume
        _axisym_clear_caches(state)
        return
    before_volume = float(_snapshot_msh_volume_m3(state))
    _BASE_update_moving_contact_line(state, dt=dt)
    _axisymmetrize_full_geometry(state)
    _axisymmetrize_full_velocity_field(state)
    after_volume = float(_snapshot_msh_volume_m3(state))
    state.last_contact_line_volume_delta_m3 = after_volume - before_volume
    _axisym_clear_caches(state)


def _axisym_surface_radius_volume_correction(
    state: VolumetricPitoisState,
) -> None:
    state.last_axisym_surface_volume_correction_status = "not_run"
    state.last_axisym_surface_volume_correction_alpha_m = 0.0
    state.last_axisym_surface_volume_correction_rel_before = 0.0
    state.last_axisym_surface_volume_correction_rel_after = 0.0

    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or (not bool(getattr(state.config, "enable_gmsh_geometric_volume_correction", False)))
        or (not bool(USER_ENABLE_GMSH_GEOMETRIC_VOLUME_CORRECTION))
    ):
        state.last_axisym_surface_volume_correction_status = "disabled"
        return

    target = float(getattr(state, "target_snapshot_volume_m3", 0.0))
    current = float(_snapshot_msh_volume_m3(state))
    if (not np.isfinite(target)) or target <= 0.0:
        state.last_axisym_surface_volume_correction_status = "bad_target"
        return
    if (not np.isfinite(current)) or current <= 0.0:
        state.last_axisym_surface_volume_correction_status = "bad_current"
        return

    trigger = max(
        float(getattr(state.config, "gmsh_geometric_volume_correction_trigger_rel", 0.0)),
        float(USER_AXISYM_SURFACE_VOLUME_CORRECTION_TRIGGER_REL),
        0.0,
    )
    rel_before = float((current - target) / max(target, 1.0e-30))
    state.last_axisym_surface_volume_correction_rel_before = rel_before
    state.last_axisym_surface_volume_correction_rel_after = rel_before
    if abs(rel_before) <= trigger:
        state.last_axisym_surface_volume_correction_status = "within_trigger"
        return

    outer_rings = [list(ring) for ring in state.outer_rings if ring]
    if len(outer_rings) < 4:
        state.last_axisym_surface_volume_correction_status = "insufficient_surface"
        return

    mid, axis, _e1, _e2 = _gmsh_axisym_basis(state)
    ring_infos: list[dict] = []
    s_values: list[float] = []
    for ring_idx, ring in enumerate(outer_rings):
        coords = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in ring], dtype=float)
        rel = coords - mid[None, :]
        axial = np.dot(rel, axis)
        radial = rel - np.outer(axial, axis)
        radii = np.linalg.norm(radial, axis=1)
        finite = np.isfinite(coords).all(axis=1) & np.isfinite(radii) & (radii > 1.0e-12)
        if not np.any(finite):
            continue
        s_mean = float(np.mean(axial[finite]))
        radius_mean = float(np.mean(radii[finite]))
        s_values.append(s_mean)
        ring_infos.append(
            {
                "ring_idx": ring_idx,
                "ring": ring,
                "coords": coords,
                "radii": radii,
                "radial": radial,
                "finite": finite,
                "s": s_mean,
                "radius": radius_mean,
            }
        )

    if len(ring_infos) < 4 or len(s_values) < 4:
        state.last_axisym_surface_volume_correction_status = "insufficient_profile"
        return
    s_min = min(s_values)
    s_max = max(s_values)
    s_span = max(s_max - s_min, 1.0e-30)

    profile: list[tuple[float, float, float]] = []
    movable_infos: list[dict] = []
    last_ring_idx = len(outer_rings) - 1
    for info in ring_infos:
        xi = float(np.clip((float(info["s"]) - s_min) / s_span, 0.0, 1.0))
        weight = float(np.sin(np.pi * xi) ** 2)
        profile.append((float(info["s"]), float(info["radius"]), weight))
        if (
            int(info["ring_idx"]) in (0, last_ring_idx)
            or weight <= 1.0e-12
        ):
            continue
        finite = np.asarray(info["finite"], dtype=bool)
        ring = list(info["ring"])
        coords = np.asarray(info["coords"], dtype=float)
        radii = np.asarray(info["radii"], dtype=float)
        radial = np.asarray(info["radial"], dtype=float)
        vertices = []
        base_coords = []
        unit_radials = []
        base_radii = []
        for vertex, coord, radius_value, radial_vec, ok in zip(ring, coords, radii, radial, finite):
            if (
                (not bool(ok))
                or vertex in state.bV_caps
                or getattr(vertex, "cap_id", None) is not None
            ):
                continue
            vertices.append(vertex)
            base_coords.append(np.asarray(coord, dtype=float))
            unit_radials.append(np.asarray(radial_vec, dtype=float) / max(float(radius_value), 1.0e-30))
            base_radii.append(float(radius_value))
        if vertices:
            movable_infos.append(
                {
                    "vertices": vertices,
                    "coords": np.asarray(base_coords, dtype=float),
                    "unit_radials": np.asarray(unit_radials, dtype=float),
                    "radii": np.asarray(base_radii, dtype=float),
                    "weight": weight,
                }
            )

    if not movable_infos:
        state.last_axisym_surface_volume_correction_status = "no_movable_surface"
        return

    profile.sort(key=lambda item: item[0])
    s_profile = np.asarray([item[0] for item in profile], dtype=float)
    integrand = np.asarray([item[1] * item[2] for item in profile], dtype=float)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    dV_dalpha = float(2.0 * np.pi * trapezoid(integrand, s_profile))
    if (not np.isfinite(dV_dalpha)) or abs(dV_dalpha) <= 1.0e-30:
        state.last_axisym_surface_volume_correction_status = "zero_derivative"
        return

    mean_radius = float(np.mean([float(info["radius"]) for info in ring_infos]))
    max_step_abs = max(float(USER_AXISYM_SURFACE_VOLUME_CORRECTION_MAX_RADIAL_STEP_UM), 0.0) * 1.0e-6
    max_step_rel = max(float(USER_AXISYM_SURFACE_VOLUME_CORRECTION_MAX_REL_RADIUS_STEP), 0.0) * max(
        mean_radius,
        1.0e-30,
    )
    step_limits = [value for value in (max_step_abs, max_step_rel) if value > 0.0]
    if not step_limits:
        state.last_axisym_surface_volume_correction_status = "zero_step_limit"
        return
    max_step = min(step_limits)
    alpha_linear = float((target - current) / dV_dalpha)
    alpha_trial = float(np.clip(alpha_linear, -max_step, max_step))
    if (not np.isfinite(alpha_trial)) or abs(alpha_trial) <= 1.0e-15:
        state.last_axisym_surface_volume_correction_status = "tiny_correction"
        return

    all_vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if (not all_vertices) or tets.size == 0:
        state.last_axisym_surface_volume_correction_status = "empty_mesh"
        return
    base_positions = _volume_points_array(state)
    base_signed_six = _indexed_tet_signed_six_volumes(base_positions, tets)
    base_abs_six = np.abs(base_signed_six)
    max_abs_six = float(np.max(base_abs_six)) if base_abs_six.size else 0.0
    active = base_abs_six > max(1.0e-12 * max_abs_six, 1.0e-30)
    if not np.any(active):
        state.last_axisym_surface_volume_correction_status = "degenerate_mesh"
        return
    base_sign = np.sign(base_signed_six)
    base_min_abs = float(np.min(base_abs_six[active]))
    min_volume_fraction = float(
        np.clip(USER_AXISYM_SURFACE_VOLUME_CORRECTION_MIN_TET_VOLUME_FRACTION, 0.0, 1.0)
    )

    def restore_base() -> None:
        _move_vertices_batch(
            all_vertices,
            [tuple(float(x) for x in row) for row in base_positions],
            state.HC,
            state.bV_caps,
        )
        _axisym_clear_caches(state)

    def apply_alpha(alpha: float) -> bool:
        vertices: list[object] = []
        targets: list[tuple[float, float, float]] = []
        for info in movable_infos:
            dr = float(alpha) * float(info["weight"])
            coords = np.asarray(info["coords"], dtype=float)
            unit_radials = np.asarray(info["unit_radials"], dtype=float)
            radii = np.asarray(info["radii"], dtype=float)
            if np.any(radii + dr <= np.maximum(1.0e-8 * max(mean_radius, 1.0e-30), 0.05 * radii)):
                return False
            for vertex, target in zip(info["vertices"], coords + dr * unit_radials):
                vertices.append(vertex)
                targets.append(tuple(float(x) for x in target))
        if not vertices:
            return False
        _move_vertices_batch(vertices, targets, state.HC, state.bV_caps)
        _gmsh_axisymmetrize_state(state)
        return True

    def candidate_is_valid() -> bool:
        points = _volume_points_array(state)
        signed_six = _indexed_tet_signed_six_volumes(points, tets)
        if signed_six.shape != base_signed_six.shape or not np.all(np.isfinite(signed_six)):
            return False
        if np.any(np.sign(signed_six[active]) != base_sign[active]):
            return False
        abs_six = np.abs(signed_six[active])
        if abs_six.size == 0 or not np.all(np.isfinite(abs_six)):
            return False
        if base_min_abs > 0.0 and float(np.min(abs_six)) < min_volume_fraction * base_min_abs:
            return False
        return True

    best_alpha = 0.0
    best_error = abs(current - target)
    for factor in (1.0, 0.5, 0.25, 0.125, 0.0625):
        alpha = alpha_trial * factor
        if abs(alpha) <= 1.0e-15:
            continue
        if not apply_alpha(alpha):
            restore_base()
            continue
        trial_volume = float(_snapshot_msh_volume_m3(state))
        valid = candidate_is_valid()
        trial_error = abs(trial_volume - target) if np.isfinite(trial_volume) else float("inf")
        restore_base()
        if valid and trial_error < best_error:
            best_alpha = float(alpha)
            best_error = float(trial_error)

    if best_alpha == 0.0:
        state.last_axisym_surface_volume_correction_status = "no_valid_improvement"
        return

    if not apply_alpha(best_alpha):
        restore_base()
        state.last_axisym_surface_volume_correction_status = "apply_failed"
        return
    final_volume = float(_snapshot_msh_volume_m3(state))
    final_error = abs(final_volume - target) if np.isfinite(final_volume) else float("inf")
    if (not candidate_is_valid()) or final_error > best_error * (1.0 + 1.0e-9):
        restore_base()
        state.last_axisym_surface_volume_correction_status = "final_rejected"
        return

    state.last_axisym_surface_volume_correction_status = "applied"
    state.last_axisym_surface_volume_correction_alpha_m = float(best_alpha)
    state.last_axisym_surface_volume_correction_rel_after = float(
        (final_volume - target) / max(target, 1.0e-30)
    )
    _axisym_clear_caches(state)


def _project_volume_to_target(state: VolumetricPitoisState) -> None:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        _axisym_surface_radius_volume_correction(state)
        _axisym_clear_caches(state)
        return
    if bool(USER_ENFORCE_FULL_AXISYMMETRY):
        # Axisymmetric non-Gmsh runs conserve volume through their force path,
        # not by post-step geometric rescaling of the liquid bridge.
        _axisymmetrize_full_geometry(state)
        _axisymmetrize_full_velocity_field(state)
        _axisym_clear_caches(state)
        return
    if not state.config.enable_volume_projection:
        return
    _BASE_project_volume_to_target(state)
    _axisymmetrize_full_geometry(state)
    _axisymmetrize_full_velocity_field(state)
    _axisym_clear_caches(state)


def _gmsh_contact_ring_radius_and_angles(
    ring: list,
    sphere_center: np.ndarray,
    axis: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> tuple[float, list[float]]:
    if not ring:
        return 0.0, []
    radii: list[float] = []
    angles: list[float] = []
    for vertex in ring:
        rel = np.asarray(vertex.x_a[:3], dtype=float) - sphere_center
        axial = float(np.dot(rel, axis))
        radial = rel - axial * axis
        radius = float(np.linalg.norm(radial))
        radii.append(radius)
        if radius <= 1.0e-30:
            angles.append(0.0)
        else:
            angles.append(float(np.arctan2(float(np.dot(radial, e2)), float(np.dot(radial, e1)))))
    return float(np.mean(radii)), angles


def _move_gmsh_contact_ring_radius(
    state: VolumetricPitoisState,
    ring: list,
    *,
    sphere_center: np.ndarray,
    radius_value: float,
    sign: float,
    axis: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    angles: list[float],
) -> None:
    sphere_radius = float(state.config.particle_radius)
    radius_value = float(np.clip(radius_value, 0.0, sphere_radius))
    axial = math.sqrt(max(sphere_radius * sphere_radius - radius_value * radius_value, 0.0))
    targets = []
    for angle in angles:
        e_r = float(np.cos(angle)) * e1 + float(np.sin(angle)) * e2
        targets.append(tuple(sphere_center + sign * axial * axis + radius_value * e_r))
    _move_vertices_batch(ring, targets, state.HC, state.bV_caps)


def _apply_gmsh_contact_line_volume_slide(
    state: VolumetricPitoisState,
    *,
    dt: float | None = None,
) -> None:
    state.last_gmsh_cl_volume_slide_status = "not_run"
    state.last_gmsh_cl_volume_slide_scale = 1.0
    state.last_gmsh_cl_volume_slide_rel_before = 0.0
    state.last_gmsh_cl_volume_slide_rel_after = 0.0
    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or (not bool(USER_ENABLE_CONTACT_LINE_VOLUME_SLIDE))
        or (not bool(getattr(state.config, "allow_contact_line_growth", True)))
    ):
        state.last_gmsh_cl_volume_slide_status = "disabled"
        return
    target = float(getattr(state, "target_snapshot_volume_m3", 0.0))
    current = float(_snapshot_msh_volume_m3(state))
    if (not np.isfinite(target)) or target <= 0.0 or (not np.isfinite(current)) or current <= 0.0:
        state.last_gmsh_cl_volume_slide_status = "bad_volume"
        return
    state.last_gmsh_cl_volume_slide_rel_before = float((current - target) / max(target, 1.0e-30))
    state.last_gmsh_cl_volume_slide_rel_after = state.last_gmsh_cl_volume_slide_rel_before
    trigger = max(float(USER_CONTACT_LINE_VOLUME_SLIDE_TRIGGER_REL), 0.0)
    if abs(current - target) / max(target, 1.0e-30) <= trigger:
        state.last_gmsh_cl_volume_slide_status = "within_trigger"
        return

    _mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    bottom_center = np.asarray(state.bottom_sphere_center, dtype=float)
    top_center = np.asarray(state.top_sphere_center, dtype=float)
    bottom_radius, bottom_angles = _gmsh_contact_ring_radius_and_angles(
        state.bottom_contact_ring,
        bottom_center,
        axis,
        e1,
        e2,
    )
    top_radius, top_angles = _gmsh_contact_ring_radius_and_angles(
        state.top_contact_ring,
        top_center,
        axis,
        e1,
        e2,
    )
    if bottom_radius <= 1.0e-30 or top_radius <= 1.0e-30:
        state.last_gmsh_cl_volume_slide_status = "empty_contact_ring"
        return

    vertices = list(state.volume_export_vertices)
    base_positions = _volume_points_array(state)
    base_bottom_radius = float(bottom_radius)
    base_top_radius = float(top_radius)
    sphere_radius = float(state.config.particle_radius)
    min_fraction = float(np.clip(USER_CONTACT_LINE_VOLUME_SLIDE_MIN_RADIUS_FRACTION, 1.0e-6, 1.0))
    max_fraction = float(max(USER_CONTACT_LINE_VOLUME_SLIDE_MAX_RADIUS_FRACTION, 1.0))

    def restore() -> None:
        _move_vertices_batch(
            vertices,
            [tuple(float(x) for x in row) for row in base_positions],
            state.HC,
            state.bV_caps,
        )
        _gmsh_axisymmetrize_state(state)

    def apply_scale(scale: float) -> float:
        _move_gmsh_contact_ring_radius(
            state,
            state.bottom_contact_ring,
            sphere_center=bottom_center,
            radius_value=base_bottom_radius * float(scale),
            sign=1.0,
            axis=axis,
            e1=e1,
            e2=e2,
            angles=bottom_angles,
        )
        _move_gmsh_contact_ring_radius(
            state,
            state.top_contact_ring,
            sphere_center=top_center,
            radius_value=base_top_radius * float(scale),
            sign=-1.0,
            axis=axis,
            e1=e1,
            e2=e2,
            angles=top_angles,
        )
        _gmsh_axisymmetrize_state(state)
        return float(_snapshot_msh_volume_m3(state))

    if current > target:
        low = min_fraction
        high = 1.0
        vol_low = apply_scale(low)
        restore()
        if (not np.isfinite(vol_low)) or vol_low > target:
            state.last_gmsh_cl_volume_slide_status = "cannot_shrink_enough"
            return
    else:
        low = 1.0
        high = min(max_fraction, sphere_radius / max(base_bottom_radius, base_top_radius, 1.0e-30))
        if high <= 1.0:
            state.last_gmsh_cl_volume_slide_status = "no_growth_margin"
            return
        vol_high = apply_scale(high)
        restore()
        if (not np.isfinite(vol_high)) or vol_high < target:
            state.last_gmsh_cl_volume_slide_status = "cannot_grow_enough"
            return

    best_scale = 1.0
    best_error = abs(current - target)
    for _iter in range(24):
        scale = 0.5 * (low + high)
        volume = apply_scale(scale)
        error = abs(volume - target) if np.isfinite(volume) else float("inf")
        if error < best_error:
            best_error = error
            best_scale = scale
        restore()
        if error / max(target, 1.0e-30) <= trigger:
            break
        if current > target:
            if volume < target:
                low = scale
            else:
                high = scale
        else:
            if volume < target:
                low = scale
            else:
                high = scale

    old_bottom = base_bottom_radius
    old_top = base_top_radius
    apply_scale(best_scale)
    new_bottom = base_bottom_radius * float(best_scale)
    new_top = base_top_radius * float(best_scale)
    final_volume = float(_snapshot_msh_volume_m3(state))
    state.last_gmsh_cl_volume_slide_status = "applied"
    state.last_gmsh_cl_volume_slide_scale = float(best_scale)
    state.last_gmsh_cl_volume_slide_rel_after = float((final_volume - target) / max(target, 1.0e-30))
    if dt is not None and float(dt) > 0.0:
        state.last_bottom_contact_line_speed = abs(new_bottom - old_bottom) / float(dt)
        state.last_top_contact_line_speed = abs(new_top - old_top) / float(dt)
    _axisym_clear_caches(state)


def _apply_gmsh_position_volume_constraint(state: VolumetricPitoisState) -> None:
    state.last_position_volume_constraint_status = "not_run"
    state.last_position_volume_constraint_rel_before = 0.0
    state.last_position_volume_constraint_rel_after = 0.0
    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or (not bool(getattr(state.config, "enable_incompressible_projection", False)))
        or (not bool(USER_ENABLE_POSITION_VOLUME_CONSTRAINT))
    ):
        state.last_position_volume_constraint_status = "disabled"
        return
    target = float(getattr(state, "target_snapshot_volume_m3", 0.0))
    if (not np.isfinite(target)) or target <= 0.0:
        state.last_position_volume_constraint_status = "bad_target"
        return
    current = float(_snapshot_msh_volume_m3(state))
    if (not np.isfinite(current)) or current <= 0.0:
        state.last_position_volume_constraint_status = "bad_current"
        return

    trigger = max(float(USER_POSITION_VOLUME_CONSTRAINT_TRIGGER_REL), 0.0)
    rel_before = float((current - target) / max(target, 1.0e-30))
    state.last_position_volume_constraint_rel_before = rel_before
    state.last_position_volume_constraint_rel_after = rel_before
    if abs(rel_before) <= trigger:
        state.last_position_volume_constraint_status = "within_trigger"
        return

    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if (not vertices) or tets.size == 0:
        state.last_position_volume_constraint_status = "empty_mesh"
        return
    fixed = set(state.bV_caps)
    max_disp = 0.02 * max(float(state.config.particle_radius), 1.0e-30)

    def tangent_projector(vertex) -> np.ndarray:
        if vertex in state.bottom_contact_ring:
            center = np.asarray(state.bottom_sphere_center, dtype=float)
        elif vertex in state.top_contact_ring:
            center = np.asarray(state.top_sphere_center, dtype=float)
        else:
            return np.eye(3, dtype=float)
        normal = np.asarray(vertex.x_a[:3], dtype=float) - center
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-30:
            return np.eye(3, dtype=float)
        normal = normal / normal_norm
        return np.eye(3, dtype=float) - np.outer(normal, normal)

    for _iter in range(max(1, int(USER_POSITION_VOLUME_CONSTRAINT_MAX_ITERS))):
        points = _volume_points_array(state)
        current = float(_indexed_tet_mesh_volume_m3(points, tets))
        if (not np.isfinite(current)) or current <= 0.0:
            state.last_position_volume_constraint_status = "bad_iter_volume"
            return
        rel_current = float((current - target) / max(target, 1.0e-30))
        state.last_position_volume_constraint_rel_after = rel_current
        if abs(rel_current) <= trigger:
            state.last_position_volume_constraint_status = "applied"
            return

        tet_points = points[tets]
        pa = tet_points[:, 0, :]
        pb = tet_points[:, 1, :]
        pc = tet_points[:, 2, :]
        pd = tet_points[:, 3, :]
        tet_volumes = np.abs(np.einsum("ij,ij->i", pa - pd, np.cross(pb - pd, pc - pd))) / 6.0
        lumped = np.zeros(len(vertices), dtype=float)
        for local in range(4):
            np.add.at(lumped, tets[:, local], 0.25 * tet_volumes)

        gradient = _indexed_tet_volume_gradient(points, tets)
        direction = np.zeros_like(points, dtype=float)
        for idx, vertex in enumerate(vertices):
            if vertex in fixed or lumped[idx] <= 1.0e-30:
                continue
            direction[idx] = (tangent_projector(vertex) @ gradient[idx]) / max(float(lumped[idx]), 1.0e-30)

        derivative = float(np.sum(gradient * direction))
        if (not np.isfinite(derivative)) or abs(derivative) <= 1.0e-30:
            state.last_position_volume_constraint_status = "zero_derivative"
            return
        scale = (target - current) / derivative
        displacement = scale * direction
        disp_norms = np.linalg.norm(displacement, axis=1)
        max_norm = float(np.max(disp_norms)) if disp_norms.size else 0.0
        if max_norm > max_disp:
            displacement *= max_disp / max(max_norm, 1.0e-30)

        movable_vertices: list[object] = []
        targets: list[tuple[float, float, float]] = []
        for idx, vertex in enumerate(vertices):
            if vertex in fixed:
                continue
            target_pos = points[idx] + displacement[idx]
            if np.all(np.isfinite(target_pos)):
                movable_vertices.append(vertex)
                targets.append(tuple(float(x) for x in target_pos))
        if not movable_vertices:
            state.last_position_volume_constraint_status = "no_movable_vertices"
            return

        before = abs(current - target)
        old_positions = points.copy()
        best_positions: np.ndarray | None = None
        best_after = before
        best_rel_after = rel_current
        for factor in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial_vertices: list[object] = []
            trial_targets: list[tuple[float, float, float]] = []
            for idx, vertex in enumerate(vertices):
                if vertex in fixed:
                    continue
                target_pos = points[idx] + float(factor) * displacement[idx]
                if np.all(np.isfinite(target_pos)):
                    trial_vertices.append(vertex)
                    trial_targets.append(tuple(float(x) for x in target_pos))
            if not trial_vertices:
                continue
            _move_vertices_batch(trial_vertices, trial_targets, state.HC, state.bV_caps)
            _gmsh_axisymmetrize_state(state)
            after_volume = float(_snapshot_msh_volume_m3(state))
            after = abs(after_volume - target)
            if after < best_after:
                best_after = float(after)
                best_rel_after = float((after_volume - target) / max(target, 1.0e-30))
                best_positions = _volume_points_array(state).copy()
            _move_vertices_batch(
                vertices,
                [tuple(float(x) for x in row) for row in old_positions],
                state.HC,
                state.bV_caps,
            )
            _gmsh_axisymmetrize_state(state)

        if best_positions is None:
            state.last_position_volume_constraint_rel_after = rel_current
            state.last_position_volume_constraint_status = "no_improvement"
            return

        _move_vertices_batch(
            vertices,
            [tuple(float(x) for x in row) for row in best_positions],
            state.HC,
            state.bV_caps,
        )
        _gmsh_axisymmetrize_state(state)
        state.last_position_volume_constraint_rel_after = best_rel_after

    state.last_position_volume_constraint_status = "max_iters"
    _axisym_clear_caches(state)


def _Ftot(
    v,
    *,
    state: VolumetricPitoisState,
    pressure_model=None,
    include_projected_pressure: bool = True,
    include_contact_line: bool = True,
    include_damping: bool = True,
) -> np.ndarray:
    if pressure_model is None:
        pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
    if bool(getattr(state.config, "enable_incompressible_projection", False)):
        include_projected_pressure = False
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        if (v in state.bV_caps) or (not bool(USER_ENFORCE_FULL_AXISYMMETRY)):
            return _BASE_Ftot(
                v,
                state=state,
                pressure_model=pressure_model,
                include_projected_pressure=include_projected_pressure,
                include_contact_line=include_contact_line,
                include_damping=include_damping,
            )
        force_map = _gmsh_axisym_force_map(
            state,
            pressure_model=pressure_model,
            include_projected_pressure=include_projected_pressure,
            include_contact_line=include_contact_line,
            include_damping=include_damping,
        )
        if id(v) in force_map:
            return np.asarray(force_map[id(v)], dtype=float)
        return _BASE_Ftot(
            v,
            state=state,
            pressure_model=pressure_model,
            include_projected_pressure=include_projected_pressure,
            include_contact_line=include_contact_line,
            include_damping=include_damping,
        )
    if (v in state.bV_caps) or (not bool(USER_ENFORCE_FULL_AXISYMMETRY)):
        return _BASE_Ftot(
            v,
            state=state,
            pressure_model=pressure_model,
            include_projected_pressure=include_projected_pressure,
            include_contact_line=include_contact_line,
            include_damping=include_damping,
        )
    force_map = _axisym_force_map(
        state,
        pressure_model=pressure_model,
        include_projected_pressure=include_projected_pressure,
        include_contact_line=include_contact_line,
        include_damping=include_damping,
    )
    return np.asarray(force_map.get(id(v), np.zeros(3, dtype=float)), dtype=float)


def _vertex_acceleration(v, *, state: VolumetricPitoisState) -> np.ndarray:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        if v in state.bV_caps:
            return np.zeros(3, dtype=float)
        if bool(USER_ENFORCE_FULL_AXISYMMETRY):
            return np.asarray(_gmsh_axisym_accel_map(state).get(id(v), np.zeros(3, dtype=float)), dtype=float)
        mass = max(float(getattr(v, "m", 0.0)), 1.0e-12)
        F_nonproj = _Ftot(v, state=state, include_projected_pressure=False)
        accel = F_nonproj / mass
        accel = _clip_acceleration(accel, state)
        if bool(getattr(state.config, "enable_volume_projection", False)):
            F_total = _Ftot(v, state=state, include_projected_pressure=True)
            accel += (F_total - F_nonproj) / mass
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
    if (v in state.bV_caps) or (not bool(USER_ENFORCE_FULL_AXISYMMETRY)):
        mass = max(float(getattr(v, "m", 0.0)), 1.0e-12)
        F_nonproj = _Ftot(v, state=state, include_projected_pressure=False)
        accel = F_nonproj / mass
        accel = _clip_acceleration(accel, state)
        if bool(getattr(state.config, "enable_volume_projection", False)):
            accel += _Fp_proj(v, state=state) / mass
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

    cache = getattr(state, "_axisym_accel_cache", None)
    if cache is None:
        cache = {}
        pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
        force_map_nonproj = _axisym_force_map(
            state,
            pressure_model=pressure_model,
            include_projected_pressure=False,
            include_contact_line=True,
            include_damping=True,
        )
        force_map_total = _axisym_force_map(
            state,
            pressure_model=pressure_model,
            include_projected_pressure=True,
            include_contact_line=True,
            include_damping=True,
        )
        mid, axis, e1, e2 = _axisym_basis(state)
        covered: set[int] = set()
        for k, rings in enumerate(state.layer_rings):
            center_coord = _axisym_layer_center_coord(state, layer_idx=k, mid=mid, axis=axis)
            for ring in rings:
                members = [node for node in ring if (node not in state.bV_caps) and (id(node) in force_map_nonproj)]
                if not members:
                    continue
                radial_units = _axisym_ring_radial_units(
                    members,
                    center_coord=center_coord,
                    axis=axis,
                    e1=e1,
                    e2=e2,
                )
                mass_avg = float(np.mean([max(float(getattr(node, "m", 0.0)), 1.0e-12) for node in members]))
                force_r_nonproj = float(
                    np.mean([float(np.dot(force_map_nonproj[id(node)], e_r)) for node, e_r in zip(members, radial_units)])
                )
                force_z_nonproj = float(np.mean([float(np.dot(force_map_nonproj[id(node)], axis)) for node in members]))
                force_r_proj = float(
                    np.mean(
                        [
                            float(np.dot(force_map_total[id(node)] - force_map_nonproj[id(node)], e_r))
                            for node, e_r in zip(members, radial_units)
                        ]
                    )
                )
                force_z_proj = float(
                    np.mean(
                        [
                            float(np.dot(force_map_total[id(node)] - force_map_nonproj[id(node)], axis))
                            for node in members
                        ]
                    )
                )
                for node, e_r in zip(members, radial_units):
                    accel_nonproj = (force_r_nonproj / mass_avg) * e_r + (force_z_nonproj / mass_avg) * axis
                    accel_proj = (force_r_proj / mass_avg) * e_r + (force_z_proj / mass_avg) * axis
                    cache[id(node)] = _clip_acceleration(_clip_acceleration(accel_nonproj, state) + accel_proj, state)
                    covered.add(id(node))

            center_v = state.layer_centers[k]
            vid = id(center_v)
            if (center_v not in state.bV_caps) and (vid in force_map_nonproj):
                mass = max(float(getattr(center_v, "m", 0.0)), 1.0e-12)
                accel_nonproj = float(np.dot(force_map_nonproj[vid], axis)) / mass
                accel_proj = float(np.dot(force_map_total[vid] - force_map_nonproj[vid], axis)) / mass
                cache[vid] = _clip_acceleration(accel_nonproj * axis, state) + accel_proj * axis
                covered.add(vid)

        for node in [candidate for candidate in state.HC.V if (candidate not in state.bV_caps) and (id(candidate) in force_map_nonproj)]:
            vid = id(node)
            if vid in covered:
                continue
            mass = max(float(getattr(node, "m", 0.0)), 1.0e-12)
            accel_nonproj = force_map_nonproj[vid] / mass
            accel_proj = (force_map_total[vid] - force_map_nonproj[vid]) / mass
            cache[vid] = _clip_acceleration(_clip_acceleration(accel_nonproj, state) + accel_proj, state)
        state._axisym_accel_cache = cache

    return np.asarray(cache.get(id(v), np.zeros(3, dtype=float)), dtype=float)


def _tet_volume_and_shape_grads(tet_points: np.ndarray) -> tuple[float, np.ndarray] | None:
    x = np.ones((4, 4), dtype=float)
    x[:, 1:] = np.asarray(tet_points, dtype=float)
    det = float(np.linalg.det(x))
    volume = abs(det) / 6.0
    if (not np.isfinite(volume)) or volume <= 1.0e-30:
        return None
    try:
        inv_x = np.linalg.inv(x)
    except np.linalg.LinAlgError:
        return None
    grads = np.asarray(inv_x[1:, :].T, dtype=float)
    if not np.all(np.isfinite(grads)):
        return None
    return volume, grads


def _gmsh_tet_divergence_l2(state: VolumetricPitoisState) -> float:
    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if (not vertices) or tets.size == 0:
        return 0.0

    _project_gmsh_contact_line_velocities_to_sphere_tangents(state)
    _set_cap_velocities(state)
    points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
    velocities = np.asarray([np.asarray(v.u[:3], dtype=float) for v in vertices], dtype=float)
    weighted = 0.0
    volume_sum = 0.0
    for tet in tets:
        idx = np.asarray(tet, dtype=int)
        result = _tet_volume_and_shape_grads(points[idx])
        if result is None:
            continue
        volume, grads = result
        div_u = float(np.sum(np.einsum("ij,ij->i", velocities[idx], grads)))
        if not np.isfinite(div_u):
            continue
        weighted += volume * div_u * div_u
        volume_sum += volume
    if volume_sum <= 1.0e-30:
        return 0.0
    return float(math.sqrt(max(weighted / volume_sum, 0.0)))


def _project_gmsh_contact_line_velocities_to_sphere_tangents(state: VolumetricPitoisState) -> None:
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
            normal = np.asarray(v.x_a[:3], dtype=float) - sphere_center
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1.0e-30:
                continue
            normal = normal / normal_norm
            rel_u = np.asarray(v.u[:3], dtype=float) - sphere_velocity
            v.u = sphere_velocity + rel_u - float(np.dot(rel_u, normal)) * normal


def _apply_gmsh_incompressible_projection(state: VolumetricPitoisState, *, dt: float) -> None:
    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or (not bool(getattr(state.config, "enable_incompressible_projection", False)))
        or float(dt) <= 0.0
    ):
        return
    if coo_matrix is None or diags is None or spsolve is None:
        raise RuntimeError("USER_ENABLE_INCOMPRESSIBLE_PROJECTION requires scipy.sparse.")

    vertices = list(getattr(state, "volume_export_vertices", []))
    tets = np.asarray(getattr(state, "volume_export_tets", np.empty((0, 4), dtype=int)), dtype=int)
    if (not vertices) or tets.size == 0:
        return

    _project_gmsh_contact_line_velocities_to_sphere_tangents(state)
    _set_cap_velocities(state)
    points = np.asarray([np.asarray(v.x_a[:3], dtype=float) for v in vertices], dtype=float)
    velocities = np.asarray([np.asarray(v.u[:3], dtype=float) for v in vertices], dtype=float)
    if (not np.all(np.isfinite(points))) or (not np.all(np.isfinite(velocities))):
        return

    n_vertices = len(vertices)
    n_dofs = 3 * n_vertices
    g_rows: list[int] = []
    g_cols: list[int] = []
    g_data: list[float] = []
    lumped = np.zeros(n_vertices, dtype=float)
    tet_cache: list[tuple[int, np.ndarray, float, np.ndarray]] = []

    for cell_idx, tet in enumerate(tets):
        idx = np.asarray(tet, dtype=int)
        result = _tet_volume_and_shape_grads(points[idx])
        if result is None:
            continue
        volume, grads = result
        row_scale = math.sqrt(max(volume, 1.0e-30))
        tet_cache.append((int(cell_idx), idx, volume, grads))
        lumped[idx] += 0.25 * volume

    if not tet_cache:
        return

    def divergence_l2_from_velocity_array(velocity_array: np.ndarray) -> float:
        weighted = 0.0
        volume_sum = 0.0
        for _cell_idx, idx, volume, grads in tet_cache:
            div_u = float(np.sum(np.einsum("ij,ij->i", velocity_array[idx], grads)))
            if not np.isfinite(div_u):
                continue
            weighted += float(volume) * div_u * div_u
            volume_sum += float(volume)
        if volume_sum <= 1.0e-30:
            return 0.0
        return float(math.sqrt(max(weighted / volume_sum, 0.0)))

    for cell_idx, idx, volume, grads in tet_cache:
        row_scale = math.sqrt(max(volume, 1.0e-30))
        for a_local, ia in enumerate(idx):
            base = 3 * int(ia)
            for component in range(3):
                g_rows.append(base + component)
                g_cols.append(int(cell_idx))
                g_data.append(row_scale * float(grads[a_local, component]))

    def tangent_projector(vertex) -> np.ndarray:
        if vertex in state.bottom_contact_ring:
            center = np.asarray(state.bottom_sphere_center, dtype=float)
        elif vertex in state.top_contact_ring:
            center = np.asarray(state.top_sphere_center, dtype=float)
        else:
            return np.eye(3, dtype=float)
        normal = np.asarray(vertex.x_a[:3], dtype=float) - center
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-30:
            return np.eye(3, dtype=float)
        normal = normal / normal_norm
        return np.eye(3, dtype=float) - np.outer(normal, normal)

    w_rows: list[int] = []
    w_cols: list[int] = []
    w_data: list[float] = []
    fixed = set(state.bV_caps)
    for idx, v in enumerate(vertices):
        if v in fixed or lumped[idx] <= 1.0e-30:
            continue
        inv_mass = 1.0 / max(float(lumped[idx]), 1.0e-30)
        projector = tangent_projector(v)
        base = 3 * idx
        for a in range(3):
            for b in range(3):
                value = inv_mass * float(projector[a, b])
                if abs(value) > 0.0:
                    w_rows.append(base + a)
                    w_cols.append(base + b)
                    w_data.append(value)

    state.incompressible_divergence_before_l2 = float(divergence_l2_from_velocity_array(velocities))
    g_matrix = coo_matrix((g_data, (g_rows, g_cols)), shape=(n_dofs, int(tets.shape[0]))).tocsr()
    mobility = coo_matrix((w_data, (w_rows, w_cols)), shape=(n_dofs, n_dofs)).tocsr()
    velocity_vec = velocities.reshape(n_dofs)
    rhs = np.asarray(g_matrix.T @ velocity_vec, dtype=float)
    matrix = (g_matrix.T @ mobility @ g_matrix).tocsr()
    diag_mean = float(np.mean(np.abs(matrix.diagonal()))) if n_vertices else 1.0
    diag_mean = max(diag_mean, 1.0e-30)
    damping = max(float(getattr(state.config, "incompressible_projection_regularization", 0.0)), 0.0)
    if damping > 0.0:
        matrix = matrix + diags(
            np.full(int(tets.shape[0]), damping * diag_mean, dtype=float),
            0,
            shape=(int(tets.shape[0]), int(tets.shape[0])),
        )
    pressure = np.asarray(spsolve(matrix, rhs), dtype=float)
    if pressure.size != int(tets.shape[0]) or not np.all(np.isfinite(pressure)):
        pressure = np.nan_to_num(pressure, nan=0.0, posinf=0.0, neginf=0.0)
    state.incompressible_projection_pressure = pressure

    correction = np.asarray(mobility @ (g_matrix @ pressure), dtype=float)
    alpha = 1.0
    target_volume = float(getattr(state, "target_snapshot_volume_m3", 0.0))
    mid, axis, e1, e2 = _gmsh_axisym_basis(state)
    node_id_map = dict(getattr(state, "volume_export_node_ids", {}))

    def apply_trial_axisymmetry(trial_points: np.ndarray) -> None:
        radius = float(state.config.particle_radius)
        for spec in getattr(state, "gmsh_axisym_ring_specs", []):
            spec_vertices = list(getattr(spec, "vertices", []))
            point_indices = [node_id_map.get(id(vertex)) for vertex in spec_vertices]
            if not point_indices or any(idx is None for idx in point_indices):
                continue
            point_indices = [int(idx) for idx in point_indices]
            coords = trial_points[point_indices]
            rel = coords - mid[None, :]
            axial = np.dot(rel, axis)
            radial = rel - np.outer(axial, axis)
            r_mean = float(np.mean(np.linalg.norm(radial, axis=1)))
            cap_id = _gmsh_axisym_uniform_cap_id(spec_vertices)
            if cap_id is None:
                center_coord = mid + float(np.mean(axial)) * axis
            else:
                sphere_center = (
                    np.asarray(state.bottom_sphere_center, dtype=float)
                    if cap_id == "bottom"
                    else np.asarray(state.top_sphere_center, dtype=float)
                )
                sign = 1.0 if cap_id == "bottom" else -1.0
                r_mean = float(np.clip(r_mean, 0.0, radius))
                center_coord = sphere_center + sign * math.sqrt(max(radius * radius - r_mean * r_mean, 0.0)) * axis
            angles = tuple(getattr(spec, "angles", ()))
            if len(point_indices) == 1 or r_mean <= 1.0e-30:
                for point_idx in point_indices:
                    trial_points[point_idx] = center_coord
            else:
                for point_idx, angle in zip(point_indices, angles):
                    e_r = float(np.cos(angle)) * e1 + float(np.sin(angle)) * e2
                    trial_points[point_idx] = center_coord + r_mean * e_r

    def apply_trial_contact_projection(trial_points: np.ndarray) -> None:
        radius = float(state.config.particle_radius)
        for ring, sphere_center in (
            (state.bottom_contact_ring, np.asarray(state.bottom_sphere_center, dtype=float)),
            (state.top_contact_ring, np.asarray(state.top_sphere_center, dtype=float)),
        ):
            for vertex in ring:
                point_idx = node_id_map.get(id(vertex))
                if point_idx is None:
                    continue
                point_idx = int(point_idx)
                rel = trial_points[point_idx] - sphere_center
                rel_norm = float(np.linalg.norm(rel))
                if rel_norm <= 1.0e-30:
                    continue
                trial_points[point_idx] = sphere_center + radius * (rel / rel_norm)

    def predicted_step_volume(alpha_value: float) -> float:
        trial_points = points.copy()
        trial_velocity = (velocity_vec - float(alpha_value) * correction).reshape((n_vertices, 3))
        for vertex_idx, vertex in enumerate(vertices):
            if vertex in fixed:
                continue
            trial_points[vertex_idx] = trial_points[vertex_idx] + float(dt) * trial_velocity[vertex_idx]
        # Match the coordinate constraints applied immediately after the real
        # move: axisymmetric ring projection and contact-line projection.
        for _projection_pass in range(2):
            apply_trial_axisymmetry(trial_points)
            apply_trial_contact_projection(trial_points)
        return _indexed_tet_mesh_volume_m3(trial_points, tets)

    if target_volume > 0.0:
        candidates: list[tuple[float, float]] = []

        def add_candidate(alpha_value: float) -> tuple[float, float] | None:
            if not np.isfinite(alpha_value):
                return None
            alpha_value = float(alpha_value)
            volume_value = predicted_step_volume(alpha_value)
            if not np.isfinite(volume_value):
                return None
            pair = (alpha_value, float(volume_value))
            candidates.append(pair)
            return pair

        volume_tol = max(1.0e-8 * target_volume, 1.0e-30)
        previous_alpha_pair = add_candidate(float(getattr(state, "incompressible_projection_alpha", 1.0)))
        vol1_pair = add_candidate(1.0)
        if previous_alpha_pair is not None and vol1_pair is not None:
            denom = vol1_pair[1] - previous_alpha_pair[1]
            if abs(denom) > max(1.0e-18 * target_volume, 1.0e-30):
                add_candidate(
                    previous_alpha_pair[0]
                    + (target_volume - previous_alpha_pair[1])
                    * (vol1_pair[0] - previous_alpha_pair[0])
                    / denom
                )
        vol0_pair = None

        prelim_candidates = [
            (candidate_alpha, candidate_volume)
            for candidate_alpha, candidate_volume in candidates
            if np.isfinite(candidate_alpha) and np.isfinite(candidate_volume)
        ]
        prelim_best = (
            min(prelim_candidates, key=lambda item: abs(item[1] - target_volume))
            if prelim_candidates
            else None
        )
        needs_bracket = prelim_best is None or abs(prelim_best[1] - target_volume) > volume_tol
        if needs_bracket:
            vol0_pair = add_candidate(0.0)
            if vol0_pair is not None and vol1_pair is not None:
                denom = vol1_pair[1] - vol0_pair[1]
                if abs(denom) > max(1.0e-18 * target_volume, 1.0e-30):
                    add_candidate((target_volume - vol0_pair[1]) / denom)
            for alpha_endpoint in (-2.0, 2.0, -5.0, 5.0, -10.0, 10.0, -20.0, 20.0):
                add_candidate(alpha_endpoint)

            bracket: tuple[tuple[float, float], tuple[float, float]] | None = None
            for span in (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0):
                sorted_candidates = sorted(candidates, key=lambda item: item[0])
                for left, right in zip(sorted_candidates[:-1], sorted_candidates[1:]):
                    f_left = left[1] - target_volume
                    f_right = right[1] - target_volume
                    if f_left == 0.0:
                        bracket = (left, left)
                        break
                    if f_left * f_right <= 0.0:
                        bracket = (left, right)
                        break
                if bracket is not None:
                    break
                add_candidate(-span)
                add_candidate(span)

            if bracket is not None:
                left, right = bracket
                if left[0] == right[0]:
                    add_candidate(left[0])
                else:
                    lo_alpha, lo_volume = left
                    hi_alpha, hi_volume = right
                    lo_f = lo_volume - target_volume
                    hi_f = hi_volume - target_volume
                    for _root_iter in range(32):
                        mid_alpha = 0.5 * (lo_alpha + hi_alpha)
                        mid_volume = predicted_step_volume(mid_alpha)
                        if not np.isfinite(mid_volume):
                            break
                        candidates.append((mid_alpha, float(mid_volume)))
                        mid_f = mid_volume - target_volume
                        if abs(mid_f) <= max(1.0e-10 * target_volume, 1.0e-30):
                            break
                        if lo_f * mid_f <= 0.0:
                            hi_alpha, hi_volume, hi_f = mid_alpha, float(mid_volume), float(mid_f)
                        else:
                            lo_alpha, lo_volume, lo_f = mid_alpha, float(mid_volume), float(mid_f)
        finite_candidates = [
            (candidate_alpha, candidate_volume)
            for candidate_alpha, candidate_volume in candidates
            if np.isfinite(candidate_alpha) and np.isfinite(candidate_volume)
        ]
        if finite_candidates:
            alpha = min(finite_candidates, key=lambda item: abs(item[1] - target_volume))[0]
    state.incompressible_projection_alpha = float(alpha)

    projected_velocity = velocity_vec - float(alpha) * correction
    if not np.all(np.isfinite(projected_velocity)):
        projected_velocity = np.nan_to_num(projected_velocity, nan=0.0, posinf=0.0, neginf=0.0)
    projected_velocity = projected_velocity.reshape((n_vertices, 3))
    for idx, v in enumerate(vertices):
        if v in fixed:
            continue
        v.u = projected_velocity[idx]

    _project_gmsh_contact_line_velocities_to_sphere_tangents(state)
    _set_cap_velocities(state)
    final_velocities = np.asarray([np.asarray(v.u[:3], dtype=float) for v in vertices], dtype=float)
    state.incompressible_divergence_after_l2 = float(divergence_l2_from_velocity_array(final_velocities))
    _axisym_clear_caches(state)


def _advance_gmsh_incompressible_substep(state: VolumetricPitoisState, *, dt: float) -> bool:
    if (
        (not bool(getattr(state, "gmsh_compute_mesh", False)))
        or (not bool(getattr(state.config, "enable_incompressible_projection", False)))
    ):
        return False

    _set_cap_velocities(state)
    _enforce_no_swirl_velocity_field(state)
    _move_caps(state, dt=dt)
    _gmsh_axisymmetrize_state(state)
    _update_duals_and_masses(state)
    _update_pressure_scalar(state)
    _update_pressure_projection_scalar(state, dt=dt)

    free_vertices = [v for v in state.HC.V if v not in state.bV_caps]
    accel_by_id: dict[int, np.ndarray] = {}
    for v in free_vertices:
        accel_by_id[id(v)] = _clip_acceleration(_vertex_acceleration(v, state=state), state)

    for v in free_vertices:
        u_new = np.asarray(v.u[:3], dtype=float) + float(dt) * accel_by_id.get(id(v), np.zeros(3, dtype=float))
        v.u = np.nan_to_num(u_new, nan=0.0, posinf=0.0, neginf=0.0)

    _enforce_no_swirl_velocity_field(state)
    _set_cap_velocities(state)
    _apply_gmsh_incompressible_projection(state, dt=dt)
    _set_cap_velocities(state)

    moving_vertices: list[object] = []
    targets: list[tuple[float, float, float]] = []
    for v in free_vertices:
        if v in state.bV_caps:
            continue
        target = np.asarray(v.x_a[:3], dtype=float) + float(dt) * np.asarray(v.u[:3], dtype=float)
        if not np.all(np.isfinite(target)):
            continue
        moving_vertices.append(v)
        targets.append(tuple(float(x) for x in target))
    _move_vertices_batch(moving_vertices, targets, state.HC, state.bV_caps)
    _axisym_clear_caches(state)

    _gmsh_axisymmetrize_state(state)
    _enforce_no_swirl_velocity_field(state)
    _set_cap_velocities(state)
    _update_moving_contact_line(state, dt=dt)
    _gmsh_axisymmetrize_state(state)
    _enforce_no_swirl_velocity_field(state)
    _assert_fixed_topology(state)
    _apply_gmsh_contact_line_volume_slide(state, dt=dt)
    _apply_gmsh_position_volume_constraint(state)
    _project_volume_to_target(state)
    _gmsh_axisymmetrize_state(state)
    _assert_fixed_topology(state)
    _update_duals_and_masses(state)
    _update_pressure_scalar(state)
    _update_pressure_projection_scalar(state, dt=dt, refine=False)
    _assert_fixed_topology(state)
    return True


def _cap_force(cap_vertices: list, *, state: VolumetricPitoisState) -> np.ndarray:
    if not cap_vertices:
        return np.zeros(3, dtype=float)
    return np.sum(
        [
            _Ftot(
                v,
                state=state,
                include_projected_pressure=True,
                include_contact_line=False,
                include_damping=False,
            )
            for v in cap_vertices
        ],
        axis=0,
    )


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

    pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)
    p_proj = float(getattr(state, "pressure_projection_scalar", 0.0))
    vertices = state.surface_export_vertices
    Fcap = np.zeros(3, dtype=float)
    mu = float(state.config.mu_f)
    pressure_model = lambda vv, HC=None, dim=3: _pressure_model(vv, HC=HC, dim=dim, state=state)

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

        sigma_tri = np.zeros((3, 3), dtype=float)
        for vtx in (va, vb, vc):
            p_v = float(pressure_model(vtx)) + p_proj
            if bool(getattr(state, "gmsh_compute_mesh", False)):
                du_v = np.zeros((3, 3), dtype=float)
            else:
                du_v = velocity_difference_tensor_pointwise(vtx, state.HC, dim=3)
            sigma_tri += cauchy_stress(p_v, du_v, mu, dim=3)
        sigma_tri /= 3.0
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


def _front_band_wireframe(
    state: VolumetricPitoisState,
    *,
    elev: float,
    azim: float,
    half_width: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    ordered_rings = _ordered_outer_rings_for_surface(state)
    if len(ordered_rings) < 2:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    n_ring = len(ordered_rings[0])
    if n_ring <= 0:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    bottom_center, top_center, axis = _particle_centers_physical(state)
    axis_mid = 0.5 * (bottom_center + top_center)
    view = _view_direction(elev, azim)
    view_tan = view - axis * float(np.dot(view, axis))
    view_tan_norm = float(np.linalg.norm(view_tan))
    if view_tan_norm <= 1.0e-30:
        front_idx = 0
    else:
        view_tan /= view_tan_norm
        rel = np.array([np.asarray(v.x_a[:3], dtype=float) for v in ordered_rings[0]], dtype=float) - axis_mid[None, :]
        axial = np.outer(np.dot(rel, axis), axis)
        radial = rel - axial
        front_idx = int(np.argmax(radial @ view_tan))

    sector_indices = [((front_idx + offset) % n_ring) for offset in range(-half_width, half_width + 1)]

    segments: list[np.ndarray] = []
    vertices: list[np.ndarray] = []
    for idx in sector_indices:
        chain = [np.asarray(ring[idx].x_a[:3], dtype=float) for ring in ordered_rings]
        vertices.extend(chain)
        for a, b in zip(chain[:-1], chain[1:]):
            segments.append(np.array([a, b], dtype=float))

    for ring in ordered_rings:
        for idx_a, idx_b in zip(sector_indices[:-1], sector_indices[1:]):
            pa = np.asarray(ring[idx_a].x_a[:3], dtype=float)
            pb = np.asarray(ring[idx_b].x_a[:3], dtype=float)
            segments.append(np.array([pa, pb], dtype=float))
            vertices.extend([pa, pb])

    if not segments:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    seg_array = _project_outside_spheres(state, np.asarray(segments, dtype=float).reshape(-1, 3)).reshape(-1, 2, 3)
    vert_array = _project_outside_spheres(state, np.asarray(vertices, dtype=float))
    rounded = np.round(vert_array, decimals=12)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_vertices = vert_array[np.sort(unique_idx)]
    return seg_array, unique_vertices


def _side_surface_with_contact_rings(state: VolumetricPitoisState):
    ordered_rings = _ordered_outer_rings_for_surface(state)
    surface, surface_bV, _flat_vertices, _triangles = _build_structured_side_surface_complex(ordered_rings)
    for v in surface.V:
        v.boundary = v in surface_bV

    if not surface_bV:
        return surface, [], []

    z_vals = [float(v.x_a[2]) for v in surface_bV]
    z_min = min(z_vals)
    z_max = max(z_vals)
    tol = max(1.0e-12, 1.0e-9 * max(abs(z_max - z_min), 1.0))
    bottom_ring = [v for v in surface_bV if abs(float(v.x_a[2]) - z_min) < tol]
    top_ring = [v for v in surface_bV if abs(float(v.x_a[2]) - z_max) < tol]
    return surface, bottom_ring, top_ring


def _ring_surface_tension_force(ring: list, *, gamma: float, state: VolumetricPitoisState | None = None) -> np.ndarray:
    if not ring:
        return np.zeros(3, dtype=float)
    if state is not None and bool(getattr(state, "gmsh_compute_mesh", False)):
        force_map = _gmsh_surface_heron_force_map(state)
        return np.sum(
            [np.asarray(force_map.get(id(v), np.zeros(3, dtype=float)), dtype=float) for v in ring],
            axis=0,
        )
    return np.sum(
        [surface_tension_force(v, gamma=gamma, dim=3) for v in ring],
        axis=0,
    )


def _gorge_pressure_cap_forces(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    bottom_center, top_center, axis = _particle_centers_physical(state)
    waist_ring = min(
        [ring for ring in state.outer_rings if ring],
        key=lambda ring: float(_cap_radius(ring)),
        default=[],
    )
    if not waist_ring:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    waist_radius = float(_cap_radius(waist_ring))
    waist_center = np.mean([np.asarray(v.x_a[:3], dtype=float) for v in waist_ring], axis=0)
    pressure = float(state.pressure_scalar)
    if state.config.include_gravity:
        z_ref = 0.5 * float(bottom_center[2] + top_center[2])
        pressure -= float(state.config.rho_f) * float(state.config.gravity_mps2) * (float(waist_center[2]) - z_ref)
    pressure += float(getattr(state, "pressure_projection_scalar", 0.0))

    pressure_force = pressure * np.pi * waist_radius * waist_radius * axis
    return pressure_force, -pressure_force


def _sphere_total_forces(state: VolumetricPitoisState) -> dict[str, np.ndarray]:
    if bool(getattr(state, "gmsh_compute_mesh", False)):
        _bottom_center, _top_center, axis = _particle_centers_physical(state)
        top_cap, bottom_cap = _gorge_pressure_cap_forces(state)
        top_line = -_ring_surface_tension_force(state.top_contact_ring, gamma=state.config.gamma, state=state)
        bottom_line = -_ring_surface_tension_force(state.bottom_contact_ring, gamma=state.config.gamma, state=state)
        top_line = float(np.dot(top_line, axis)) * axis
        bottom_line = float(np.dot(bottom_line, axis)) * axis
        return {
            "top_total": top_cap + top_line,
            "bottom_total": bottom_cap + bottom_line,
            "top_cap": top_cap,
            "bottom_cap": bottom_cap,
            "top_line": top_line,
            "bottom_line": bottom_line,
        }
    top_cap = _cap_traction_force(state, which="top")
    bottom_cap = _cap_traction_force(state, which="bottom")
    _surface, bottom_ring, top_ring = _side_surface_with_contact_rings(state)
    top_line = _ring_surface_tension_force(top_ring, gamma=state.config.gamma, state=state)
    bottom_line = _ring_surface_tension_force(bottom_ring, gamma=state.config.gamma, state=state)
    return {
        "top_total": top_cap + top_line,
        "bottom_total": bottom_cap + bottom_line,
        "top_cap": top_cap,
        "bottom_cap": bottom_cap,
        "top_line": top_line,
        "bottom_line": bottom_line,
    }


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
    sphere_forces = _sphere_total_forces(state)
    top_force = sphere_forces["top_total"]
    bottom_force = sphere_forces["bottom_total"]
    fixed_force = top_force
    moving_force = bottom_force
    gap = _gap(state)
    dual_cell_sum_m3 = _mesh_volume_m3(state.HC)
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
        "dual_cell_sum_m3": float(dual_cell_sum_m3),
        "snapshot_msh_volume_m3": float(snapshot_msh_volume_m3),
        "dt": float(state.last_step_dt),
        "dt_limit_contact_line": float(state.last_dt_limit_cl),
        "dt_limit_capillary": float(state.last_dt_limit_capillary),
        "dt_limit_mesh": float(state.last_dt_limit_mesh),
        "dt_limiter": str(state.last_dt_limiter),
        "bottom_contact_line_speed": float(state.last_bottom_contact_line_speed),
        "top_contact_line_speed": float(state.last_top_contact_line_speed),
        "gmsh_cl_volume_slide_status": str(getattr(state, "last_gmsh_cl_volume_slide_status", "not_run")),
        "gmsh_cl_volume_slide_scale": float(getattr(state, "last_gmsh_cl_volume_slide_scale", 1.0)),
        "gmsh_cl_volume_slide_rel_before": float(getattr(state, "last_gmsh_cl_volume_slide_rel_before", 0.0)),
        "gmsh_cl_volume_slide_rel_after": float(getattr(state, "last_gmsh_cl_volume_slide_rel_after", 0.0)),
        "position_volume_constraint_status": str(getattr(state, "last_position_volume_constraint_status", "not_run")),
        "position_volume_constraint_rel_before": float(getattr(state, "last_position_volume_constraint_rel_before", 0.0)),
        "position_volume_constraint_rel_after": float(getattr(state, "last_position_volume_constraint_rel_after", 0.0)),
        "axisym_surface_volume_correction_status": str(
            getattr(state, "last_axisym_surface_volume_correction_status", "not_run")
        ),
        "axisym_surface_volume_correction_alpha_m": float(
            getattr(state, "last_axisym_surface_volume_correction_alpha_m", 0.0)
        ),
        "axisym_surface_volume_correction_rel_before": float(
            getattr(state, "last_axisym_surface_volume_correction_rel_before", 0.0)
        ),
        "axisym_surface_volume_correction_rel_after": float(
            getattr(state, "last_axisym_surface_volume_correction_rel_after", 0.0)
        ),
        "max_free_speed": _max_free_speed(state),
        "pressure_scalar": float(state.pressure_scalar),
        "n_vertices": sum(1 for _ in state.HC.V),
        "n_free_vertices": sum(1 for v in state.HC.V if v not in state.bV_caps),
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
    segments, coords = _contact_refined_surface_display_wireframe(state, ordered_rings)
    coords = _project_outside_spheres(state, coords)
    if segments.size:
        segments = _project_outside_spheres(state, segments.reshape((-1, 3))).reshape((-1, 2, 3))
    return segments, coords


def _cap_wireframe_vertex_pairs(state: VolumetricPitoisState) -> list[tuple[object, object]]:
    edge_pairs: list[tuple[object, object]] = []
    cap_specs = (
        (state.layer_rings[0], state.cap_bottom_center),
        (state.layer_rings[-1], state.cap_top_center),
    )
    for rings, center_vertex in cap_specs:
        if not rings:
            continue
        n_ring = len(rings[0])
        for ring in rings:
            for i in range(n_ring):
                edge_pairs.append((ring[i], ring[(i + 1) % n_ring]))
        for inner, outer in zip(rings[:-1], rings[1:]):
            for i in range(n_ring):
                edge_pairs.append((inner[i], outer[i]))
        for i in range(n_ring):
            edge_pairs.append((center_vertex, rings[0][i]))
    return edge_pairs


def _wireframe_from_vertex_pairs(
    state: VolumetricPitoisState,
    edge_pairs: list[tuple[object, object]],
) -> tuple[np.ndarray, np.ndarray]:
    if not edge_pairs:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    unique_vertices: dict[int, object] = {}
    unique_edges: set[tuple[int, int]] = set()
    for va, vb in edge_pairs:
        ida = id(va)
        idb = id(vb)
        if ida == idb:
            continue
        unique_vertices.setdefault(ida, va)
        unique_vertices.setdefault(idb, vb)
        edge = (ida, idb) if ida < idb else (idb, ida)
        unique_edges.add(edge)

    ordered_ids = list(unique_vertices.keys())
    id_to_idx = {vid: idx for idx, vid in enumerate(ordered_ids)}
    coords = np.array(
        [np.asarray(unique_vertices[vid].x_a[:3], dtype=float) for vid in ordered_ids],
        dtype=float,
    )
    coords = _project_outside_spheres(state, coords)
    segments = [
        np.array([coords[id_to_idx[ida]], coords[id_to_idx[idb]]], dtype=float)
        for ida, idb in sorted(unique_edges)
    ]
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


def _display_wireframe(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray]:
    ordered_rings = _ordered_outer_rings_for_surface(state)
    if not ordered_rings:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    flat_vertices = [v for ring in ordered_rings for v in ring]
    edge_pairs = [(flat_vertices[i], flat_vertices[j]) for i, j in sorted(_structured_surface_display_edges(ordered_rings))]
    edge_pairs.extend(_cap_wireframe_vertex_pairs(state))
    return _wireframe_from_vertex_pairs(state, edge_pairs)


def _view_direction(elev: float, azim: float) -> np.ndarray:
    elev_rad = np.deg2rad(float(elev))
    azim_rad = np.deg2rad(float(azim))
    view = np.array(
        [
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(view))
    if norm <= 1.0e-30:
        return np.array([0.0, -1.0, 0.0], dtype=float)
    return view / norm


def _visible_surface_wireframe(
    state: VolumetricPitoisState,
    *,
    elev: float,
    azim: float,
) -> tuple[np.ndarray, np.ndarray]:
    segments, coords = _display_wireframe(state)
    if segments.size == 0 and coords.size == 0:
        return segments, coords
    visibility_margin = 5.0e-5

    bottom_center, top_center, axis = _particle_centers_physical(state)
    axis_mid = 0.5 * (bottom_center + top_center)
    view = _view_direction(elev, azim)
    view_tan = view - axis * float(np.dot(view, axis))
    view_tan_norm = float(np.linalg.norm(view_tan))
    if view_tan_norm <= 1.0e-30:
        return segments, coords
    view_tan /= view_tan_norm

    def front_metric(points: np.ndarray) -> np.ndarray:
        rel = np.asarray(points, dtype=float) - axis_mid[None, :]
        rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
        axial = np.outer(np.dot(rel, axis), axis)
        radial = rel - axial
        radial = np.nan_to_num(radial, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            metric = radial @ view_tan
        return np.nan_to_num(metric, nan=0.0, posinf=0.0, neginf=0.0)

    if segments.size:
        segments = np.asarray(segments, dtype=float)
        keep = np.all(np.isfinite(segments), axis=(1, 2))
        segments = segments[keep]
        mids = 0.5 * (segments[:, 0, :] + segments[:, 1, :])
        seg_metric = front_metric(mids)
        segments = segments[seg_metric >= visibility_margin]
        if segments.size:
            changed = True
            while changed and segments.size:
                changed = False
                adjacency: dict[tuple[float, float, float], set[tuple[float, float, float]]] = defaultdict(set)
                point_lookup: dict[tuple[float, float, float], np.ndarray] = {}
                for seg in np.asarray(segments, dtype=float):
                    a = tuple(np.round(np.asarray(seg[0], dtype=float), decimals=12))
                    b = tuple(np.round(np.asarray(seg[1], dtype=float), decimals=12))
                    point_lookup[a] = np.asarray(seg[0], dtype=float)
                    point_lookup[b] = np.asarray(seg[1], dtype=float)
                    adjacency[a].add(b)
                    adjacency[b].add(a)
                prune_nodes = {node for node, nbrs in adjacency.items() if len(nbrs) <= 1}
                if prune_nodes:
                    keep_mask = []
                    for seg in np.asarray(segments, dtype=float):
                        a = tuple(np.round(np.asarray(seg[0], dtype=float), decimals=12))
                        b = tuple(np.round(np.asarray(seg[1], dtype=float), decimals=12))
                        keep_mask.append((a not in prune_nodes) and (b not in prune_nodes))
                    keep_mask = np.asarray(keep_mask, dtype=bool)
                    if not np.all(keep_mask):
                        segments = np.asarray(segments, dtype=float)[keep_mask]
                        changed = True

    if segments.size:
        coords = segments.reshape(-1, 3)
        rounded = np.round(coords, decimals=12)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        coords = coords[np.sort(unique_idx)]
    elif coords.size:
        coords = np.asarray(coords, dtype=float)
        coords = coords[np.all(np.isfinite(coords), axis=1)]
        coord_metric = front_metric(coords)
        coords = coords[coord_metric >= visibility_margin]

    return segments, coords


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
    contact_radius = min(
        max(_cap_radius(state.cap_bottom), _cap_radius(state.cap_top)),
        radius,
    )

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
    marker_size = max(2.0, float(USER_MESH_VERTEX_SIZE))
    ax.scatter(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        s=marker_size,
        facecolors=color,
        edgecolors=color,
        linewidths=0.35,
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


def _write_gmsh2_mixed(
    points: np.ndarray,
    out_path: Path,
    *,
    triangles: np.ndarray | None = None,
    tetrahedra: np.ndarray | None = None,
    node_ids: np.ndarray | None = None,
) -> None:
    points = np.asarray(points, dtype=float)
    triangles = np.empty((0, 3), dtype=int) if triangles is None else np.asarray(triangles, dtype=int)
    tetrahedra = np.empty((0, 4), dtype=int) if tetrahedra is None else np.asarray(tetrahedra, dtype=int)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must be (M,3)")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("tetrahedra must be (K,4)")
    if node_ids is None:
        node_ids = np.arange(1, points.shape[0] + 1, dtype=int)
    else:
        node_ids = np.asarray(node_ids, dtype=int)
        if node_ids.ndim != 1 or node_ids.shape[0] != points.shape[0]:
            raise ValueError("node_ids must be a length-N array")

    element_blocks = ((2, triangles), (4, tetrahedra))
    n_elements = int(triangles.shape[0] + tetrahedra.shape[0])
    with open(out_path, "w", newline="") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write(f"{points.shape[0]}\n")
        for node_id, (x, y, z) in zip(node_ids, points):
            f.write(f"{int(node_id)} {x:.17g} {y:.17g} {z:.17g}\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write(f"{n_elements}\n")
        elem_id = 1
        for element_type, elements in element_blocks:
            for elem in elements:
                node_tokens = " ".join(str(int(node_ids[int(local_idx)])) for local_idx in elem)
                f.write(f"{elem_id} {int(element_type)} 0 {node_tokens}\n")
                elem_id += 1
        f.write("$EndElements\n")


def _surface_triangle_indices_from_snapshot_points(
    points: np.ndarray,
    *triangle_sets: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 3), dtype=int)
    point_index: dict[tuple[float, float, float], int] = {
        tuple(round(float(c), 12) for c in point): int(i)
        for i, point in enumerate(points)
    }
    scale = max(float(np.max(np.linalg.norm(points, axis=1))) if points.size else 0.0, 1.0e-12)
    nearest_tol = max(1.0e-12, 1.0e-7 * scale)
    triangles: list[tuple[int, int, int]] = []

    def resolve(point: np.ndarray) -> int | None:
        key = tuple(round(float(c), 12) for c in point)
        idx = point_index.get(key)
        if idx is not None:
            return int(idx)
        distances = np.linalg.norm(points - np.asarray(point, dtype=float)[None, :], axis=1)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= nearest_tol:
            return nearest
        return None

    for triangle_set in triangle_sets:
        arr = np.asarray(triangle_set, dtype=float)
        if arr.size == 0:
            continue
        arr = arr.reshape((-1, 3, 3))
        for tri in arr:
            idxs = [resolve(point) for point in tri]
            if any(idx is None for idx in idxs):
                continue
            a, b, c = (int(idxs[0]), int(idxs[1]), int(idxs[2]))
            if len({a, b, c}) == 3:
                triangles.append((a, b, c))
    if not triangles:
        return np.empty((0, 3), dtype=int)
    return np.asarray(triangles, dtype=int)


def _triangle_soup_to_indexed_mesh(*triangle_sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords: list[tuple[float, float, float]] = []
    coord_index: dict[tuple[float, float, float], int] = {}
    triangles: list[tuple[int, int, int]] = []

    def ensure_vertex(point: np.ndarray) -> int:
        key = tuple(round(float(c), 12) for c in point)
        idx = coord_index.get(key)
        if idx is None:
            idx = len(coords)
            coords.append(tuple(float(c) for c in point))
            coord_index[key] = idx
        return idx

    for tri_set in triangle_sets:
        arr = np.asarray(tri_set, dtype=float)
        if arr.size == 0:
            continue
        for tri in arr:
            a = ensure_vertex(tri[0])
            b = ensure_vertex(tri[1])
            c = ensure_vertex(tri[2])
            if len({a, b, c}) == 3:
                triangles.append((a, b, c))

    if not coords or not triangles:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)
    return np.asarray(coords, dtype=float), np.asarray(triangles, dtype=int)


def _snapshot_surface_indexed_mesh(state: VolumetricPitoisState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _freeze_surface_topology(state)
    points = _surface_points_array(state)
    tri_array = np.vstack(
        [
            arr
            for arr in (
                state.surface_export_side_tris,
                state.surface_export_bottom_cap_tris,
                state.surface_export_top_cap_tris,
            )
            if np.asarray(arr, dtype=int).size
        ]
    ) if (
        state.surface_export_side_tris.size
        or state.surface_export_bottom_cap_tris.size
        or state.surface_export_top_cap_tris.size
    ) else np.empty((0, 3), dtype=int)
    node_ids = np.arange(1, points.shape[0] + 1, dtype=int)
    return points, tri_array, node_ids


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
    tet_points = pts[tet_idx]
    pa = tet_points[:, 0, :]
    pb = tet_points[:, 1, :]
    pc = tet_points[:, 2, :]
    pd = tet_points[:, 3, :]
    # Physical volumetric .msh volume is the sum of positive tetra volumes.
    # Never rely on cancellation from mixed element orientation.
    triple = np.einsum("ij,ij->i", pa - pd, np.cross(pb - pd, pc - pd))
    return float(np.sum(np.abs(triple)) / 6.0)


def _indexed_tet_signed_six_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    tet_idx = np.asarray(tets, dtype=int)
    if pts.size == 0 or tet_idx.size == 0:
        return np.empty(0, dtype=float)
    tet_points = pts[tet_idx]
    pa = tet_points[:, 0, :]
    pb = tet_points[:, 1, :]
    pc = tet_points[:, 2, :]
    pd = tet_points[:, 3, :]
    return np.einsum("ij,ij->i", pa - pd, np.cross(pb - pd, pc - pd))


def _indexed_tet_volume_gradient(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    tet_idx = np.asarray(tets, dtype=int)
    gradient = np.zeros_like(pts, dtype=float)
    if pts.size == 0 or tet_idx.size == 0:
        return gradient
    tet_points = pts[tet_idx]
    pa = tet_points[:, 0, :]
    pb = tet_points[:, 1, :]
    pc = tet_points[:, 2, :]
    pd = tet_points[:, 3, :]
    triple = np.einsum("ij,ij->i", pa - pd, np.cross(pb - pd, pc - pd))
    signed_scale = (np.sign(triple) / 6.0)[:, None]
    grad_a = signed_scale * np.cross(pb - pd, pc - pd)
    grad_b = signed_scale * np.cross(pc - pd, pa - pd)
    grad_c = signed_scale * np.cross(pa - pd, pb - pd)
    grad_d = -(grad_a + grad_b + grad_c)
    np.add.at(gradient, tet_idx[:, 0], grad_a)
    np.add.at(gradient, tet_idx[:, 1], grad_b)
    np.add.at(gradient, tet_idx[:, 2], grad_c)
    np.add.at(gradient, tet_idx[:, 3], grad_d)
    return gradient


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


def _wireframe_from_triangle_soup(*triangle_sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points, tris = _triangle_soup_to_indexed_mesh(*triangle_sets)
    if points.size == 0 or tris.size == 0:
        return np.empty((0, 2, 3), dtype=float), np.empty((0, 3), dtype=float)

    edges: set[tuple[int, int]] = set()
    for a, b, c in np.asarray(tris, dtype=int):
        edges.add(tuple(sorted((int(a), int(b)))))
        edges.add(tuple(sorted((int(b), int(c)))))
        edges.add(tuple(sorted((int(c), int(a)))))

    segments = np.asarray(
        [np.array([points[i], points[j]], dtype=float) for i, j in sorted(edges)],
        dtype=float,
    )
    return segments, np.asarray(points, dtype=float)


def _save_mesh_snapshot_msh(
    state: VolumetricPitoisState,
    out_path: Path,
    *,
    side_triangles: np.ndarray,
    bottom_cap_triangles: np.ndarray,
    top_cap_triangles: np.ndarray,
) -> Path:
    points, tets, node_ids = _snapshot_msh_indexed_mesh(state)
    triangles = _surface_triangle_indices_from_snapshot_points(
        points,
        side_triangles,
        bottom_cap_triangles,
        top_cap_triangles,
    )
    msh_path = out_path.with_suffix(".msh")
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    _write_gmsh2_mixed(points, msh_path, triangles=triangles, tetrahedra=tets, node_ids=node_ids)
    return msh_path


def _save_mesh_snapshot_tet_volume_csv(
    state: VolumetricPitoisState,
    out_path: Path,
) -> Path:
    points, tets, node_ids = _snapshot_msh_indexed_mesh(state)
    csv_path = out_path.with_name(f"{out_path.stem}_tet_volumes.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tet_id",
                "node1_id",
                "node2_id",
                "node3_id",
                "node4_id",
                "x1_m",
                "y1_m",
                "z1_m",
                "x2_m",
                "y2_m",
                "z2_m",
                "x3_m",
                "y3_m",
                "z3_m",
                "x4_m",
                "y4_m",
                "z4_m",
                "signed_volume_m3",
                "tet_volume_m3",
                "tet_volume_mm3",
                "tet_volume_uL",
            ]
        )
        for tet_id, (a, b, c, d) in enumerate(np.asarray(tets, dtype=int), start=1):
            pa = np.asarray(points[int(a)], dtype=float)
            pb = np.asarray(points[int(b)], dtype=float)
            pc = np.asarray(points[int(c)], dtype=float)
            pd = np.asarray(points[int(d)], dtype=float)
            signed_volume = float(np.dot(pa - pd, np.cross(pb - pd, pc - pd)) / 6.0)
            tet_volume = abs(signed_volume)
            writer.writerow(
                [
                    tet_id,
                    int(node_ids[int(a)]),
                    int(node_ids[int(b)]),
                    int(node_ids[int(c)]),
                    int(node_ids[int(d)]),
                    f"{pa[0]:.17g}",
                    f"{pa[1]:.17g}",
                    f"{pa[2]:.17g}",
                    f"{pb[0]:.17g}",
                    f"{pb[1]:.17g}",
                    f"{pb[2]:.17g}",
                    f"{pc[0]:.17g}",
                    f"{pc[1]:.17g}",
                    f"{pc[2]:.17g}",
                    f"{pd[0]:.17g}",
                    f"{pd[1]:.17g}",
                    f"{pd[2]:.17g}",
                    f"{signed_volume:.17g}",
                    f"{tet_volume:.17g}",
                    f"{1.0e9 * tet_volume:.17g}",
                    f"{1.0e9 * tet_volume:.17g}",
                ]
            )
    return csv_path


def _set_axes_equal(ax, xyz: np.ndarray, centers: list[np.ndarray], radius: float) -> None:
    pts = np.asarray(xyz.reshape(-1, 3), dtype=float) if xyz.size else np.empty((0, 3), dtype=float)
    if pts.size:
        pts = pts[np.all(np.isfinite(pts), axis=1)]
    if centers:
        centers_arr = np.asarray(centers, dtype=float)
        centers_arr = centers_arr[np.all(np.isfinite(centers_arr), axis=1)]
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
        if sphere_bbox:
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
    if not np.isfinite(half_span) or half_span <= 0.0:
        half_span = 1.0
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
        side_segments, side_vertices = _front_band_wireframe(state, elev=elev, azim=azim)
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
    if USER_SHOW_REAL_COMPUTE_TRIANGLES and USER_SHOW_SIDE_TRIANGLE_DIAGONALS and plot_side_tris.size:
        _render_triangle_edges(
            ax,
            plot_side_tris,
            color="#0f172a",
            linewidth=0.42,
            alpha=0.55,
        )
    scatter_vertices_from = plot_side_tris if (
        USER_SHOW_REAL_COMPUTE_TRIANGLES and USER_SHOW_SIDE_TRIANGLE_DIAGONALS and plot_side_tris.size
    ) else plot_side_vertices
    _scatter_mesh_vertices(ax, [scatter_vertices_from])
    ax.add_collection3d(
        Line3DCollection(
            plot_side_segments,
            colors="#111111",
            linewidths=0.72,
            alpha=0.995,
            axlim_clip=True,
        )
    )
    if not USER_SHOW_MESH_FACES and USER_FILL_PARTICLE_CAP_SURFACES:
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

    if USER_FILL_PARTICLE_CAP_SURFACES:
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
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    _save_mesh_snapshot_msh(
        state,
        out_path,
        side_triangles=side_triangles_full,
        bottom_cap_triangles=_actual_compute_cap_triangles(state)[0] if USER_SHOW_REAL_COMPUTE_TRIANGLES else bottom_cap_triangles,
        top_cap_triangles=_actual_compute_cap_triangles(state)[1] if USER_SHOW_REAL_COMPUTE_TRIANGLES else top_cap_triangles,
    )
    _save_mesh_snapshot_tet_volume_csv(state, out_path)
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

    fig_dir = out_dir
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

    snapshot_volume_m3 = np.array(
        [float(initial_snapshot_volume_m3)]
        + [float(row.get("snapshot_msh_volume_m3", row.get("snapshot_surface_volume_m3", 0.0))) for row in history],
        dtype=float,
    )
    snapshot_volume_ul = 1.0e9 * snapshot_volume_m3
    t_ms_volume = np.array([0.0] + [float(row["t"]) * 1.0e3 for row in history], dtype=float)
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
    out_png = _render_mesh_snapshot(state, title, motion_dir / filename)
    if label == "initial":
        initial_source = Path(getattr(state, "loaded_initial_msh_path", _required_initial_msh_path()))
        if initial_source.resolve() != _required_initial_msh_path().resolve():
            raise RuntimeError(f"Initial output must come from mesh_iter0012.msh, got {initial_source}")
        iter0_png = motion_dir / "mesh_iter0000.png"
        shutil.copyfile(out_png, iter0_png)
        for suffix in (".msh",):
            source = out_png.with_suffix(suffix)
            if source.exists():
                shutil.copyfile(source, iter0_png.with_suffix(suffix))
        source_csv = out_png.with_name(f"{out_png.stem}_tet_volumes.csv")
        if source_csv.exists():
            shutil.copyfile(source_csv, iter0_png.with_name(f"{iter0_png.stem}_tet_volumes.csv"))
    return out_png


def _save_history(history: list[dict], config: VolumetricPitoisConfig, out_dir: Path) -> Path:
    results_dir = out_dir
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
            0.014403481854782,
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
            1.515311539463195,
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
    title: str = "Pitois 2000 Fig. 5: dynamic experiment vs Case_2b_axisym",
) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    y_low, y_high = map(float, ylim)
    visible_y = []
    for x_arr, y_arr in ((exp_x, exp_y), (sim_x, sim_y)):
        x_vals = np.asarray(x_arr, dtype=float)
        y_vals = np.asarray(y_arr, dtype=float)
        mask = (
            np.isfinite(x_vals)
            & np.isfinite(y_vals)
            & (x_vals >= float(xlim[0]))
            & (x_vals <= float(xlim[1]))
            & (y_vals > 0.0)
        )
        if np.any(mask):
            visible_y.append(y_vals[mask])
    if visible_y:
        visible = np.concatenate(visible_y)
        y_low = min(y_low, 0.85 * float(np.min(visible)))
        y_high = max(y_high, 1.15 * float(np.max(visible)))

    ax.scatter(exp_x, exp_y, s=34, color="#111111", label="Pitois 2000 dynamic exp (black dots)")
    ax.plot(sim_x, sim_y, color="#d62828", linewidth=2.0, label="Case_2b_axisym simulation")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(y_low, y_high)
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
            if _advance_gmsh_incompressible_substep(state, dt=sub_dt):
                continue
            _set_cap_velocities(state)
            _enforce_no_swirl_velocity_field(state)
            _move_caps(state, dt=sub_dt)
            _gmsh_axisymmetrize_state(state)
            _update_duals_and_masses(state)
            _update_pressure_scalar(state)
            _update_pressure_projection_scalar(state, dt=sub_dt)
            symplectic_euler(
                state.HC,
                state.bV_caps,
                _vertex_acceleration,
                dt=sub_dt,
                n_steps=1,
                dim=3,
                retopologize_fn=False,
                workers=max(1, int(state.config.accel_workers)),
                state=state,
            )
            _gmsh_axisymmetrize_state(state)
            _enforce_no_swirl_velocity_field(state)
            _set_cap_velocities(state)
            _update_moving_contact_line(state, dt=sub_dt)
            _gmsh_axisymmetrize_state(state)
            _enforce_no_swirl_velocity_field(state)
            _assert_fixed_topology(state)
            _apply_gmsh_contact_line_volume_slide(state, dt=sub_dt)
            _apply_gmsh_position_volume_constraint(state)
            _project_volume_to_target(state)
            _gmsh_axisymmetrize_state(state)
            _assert_fixed_topology(state)
            _update_duals_and_masses(state)
            _update_pressure_scalar(state)
            _update_pressure_projection_scalar(state, dt=sub_dt, refine=False)
            _assert_fixed_topology(state)
        _assert_state_finite(state, where=f"advance step {_step + 1}")
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
        side_segments, side_vertices = _front_band_wireframe(state, elev=elev, azim=azim)
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
    if USER_SHOW_REAL_COMPUTE_TRIANGLES and USER_SHOW_SIDE_TRIANGLE_DIAGONALS and plot_side_tris.size:
        _render_triangle_edges(
            ax,
            plot_side_tris,
            color="#0f172a",
            linewidth=0.42,
            alpha=0.55,
        )
    scatter_vertices_from = plot_side_tris if (
        USER_SHOW_REAL_COMPUTE_TRIANGLES and USER_SHOW_SIDE_TRIANGLE_DIAGONALS and plot_side_tris.size
    ) else plot_side_vertices
    _scatter_mesh_vertices(ax, [scatter_vertices_from])
    ax.add_collection3d(
        Line3DCollection(
            plot_side_segments,
            colors="#111111",
            linewidths=0.72,
            alpha=0.995,
            axlim_clip=True,
        )
    )
    if not USER_SHOW_MESH_FACES and USER_FILL_PARTICLE_CAP_SURFACES:
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

    if USER_FILL_PARTICLE_CAP_SURFACES:
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

    print("Interactive Case_2b_axisym mesh viewer")
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
        help="Open an interactive window for the Case_2b_axisym initial mesh instead of running the simulation.",
    )
    parser.add_argument(
        "--view-step",
        type=int,
        default=None,
        help="Open an interactive window for the Case_2b_axisym mesh after this many separation steps.",
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
    axes[0].set_title("Case_2b_axisym normalized gap")
    axes[0].grid(alpha=0.28)

    axes[1].plot(separation["t_ms"], separation["max_u"], color="#bc4749", linewidth=1.8)
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel("Max free speed [m/s]")
    axes[1].set_title("Case_2b_axisym free-surface speed")
    axes[1].grid(alpha=0.28)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_pitois_fig5_comparison(
    *,
    separation_history: list[dict],
    out_dir: Path,
) -> None:
    fig_dir = out_dir
    results_dir = out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    separation = _history_arrays(separation_history)
    exp_x, exp_y = _pitois_fig5_dynamic_digitized()
    compare_mask = np.array(
        [int(row.get("step", 0)) >= int(USER_FIG5_COMPARE_SKIP_INITIAL_STEPS) for row in separation_history],
        dtype=bool,
    )
    if not np.any(compare_mask):
        compare_mask = np.ones_like(separation["d_over_r"], dtype=bool)

    _save_pitois_fig5_compare(
        exp_x=exp_x,
        exp_y=exp_y,
        sim_x=separation["d_over_r"][compare_mask],
        sim_y=separation["force_abs_mn"][compare_mask],
        out_path=out_dir / FIG5_COMPARE_PNG_NAME,
        xlim=(0.01, 0.30),
        ylim=(0.02, 2.0),
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
            "fig5_compare_skip_initial_steps": int(USER_FIG5_COMPARE_SKIP_INITIAL_STEPS),
            "fig5_compare_first_plotted_step": int(
                separation_history[int(np.flatnonzero(compare_mask)[0])]["step"]
                if np.any(compare_mask)
                else separation_history[0]["step"]
            ),
            "fig5_compare_note": (
                "The red curve includes the imported t=0 geometry and every completed "
                "post-advance step written during the run."
            ),
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


def _refresh_pitois_fig5_compare_png(
    *,
    separation_history: list[dict],
    out_dir: Path,
) -> None:
    if not separation_history:
        return
    separation = _history_arrays(separation_history)
    exp_x, exp_y = _pitois_fig5_dynamic_digitized()
    compare_mask = np.array(
        [int(row.get("step", 0)) >= int(USER_FIG5_COMPARE_SKIP_INITIAL_STEPS) for row in separation_history],
        dtype=bool,
    )
    if not np.any(compare_mask):
        compare_mask = np.ones_like(separation["d_over_r"], dtype=bool)
    _save_pitois_fig5_compare(
        exp_x=exp_x,
        exp_y=exp_y,
        sim_x=separation["d_over_r"][compare_mask],
        sim_y=separation["force_abs_mn"][compare_mask],
        out_path=out_dir / FIG5_COMPARE_PNG_NAME,
        xlim=(0.01, 0.30),
        ylim=(0.02, 2.0),
    )


def _print_header(config: VolumetricPitoisConfig) -> None:
    total_cl_rings = int(config.contact_line_radial_rings) + int(config.extra_cl_radial_rings)
    print("=" * 72)
    print(f"  {config.title}")
    print("=" * 72)
    print(f"Refinement                = {config.refinement}")
    print(f"CL radial rings           = {config.contact_line_radial_rings}")
    print(f"Extra CL outer rings      = {config.extra_cl_radial_rings}")
    print(f"CL radial bias ratio      = {config.contact_line_radial_bias_ratio}")
    print(f"Sidewall axial stride     = {config.axial_ring_stride}")
    print(f"Neck extra axial layers   = {config.neck_extra_axial_layers}")
    print(f"CL extra axial layers     = {config.cl_extra_axial_layers}")
    if total_cl_rings <= 1 and abs(float(config.contact_line_radial_bias_ratio) - 1.0) > 1.0e-12:
        print("CL radial bias active?    = no (need at least 2 total CL rings)")
    print(f"Particle radius           = {config.particle_radius * 1e3:.3f} mm")
    print(f"Bridge contact radius     = {config.target_cap_radius * 1e3:.3f} mm")
    print(f"Surface tension           = {config.gamma:.4f} N/m")
    print(f"Viscosity                 = {config.mu_f:.3e} Pa s")
    print(f"Density                   = {config.rho_f:.3f} kg/m^3")
    print(f"Gravity included          = {'yes' if config.include_gravity else 'no'}")
    if config.include_gravity:
        print(f"Gravity acceleration      = {config.gravity_mps2:.3f} m/s^2")
    print(f"Contact angle assumption  = {config.contact_angle_deg:.1f} deg (literature-informed baseline)")
    print(f"Moving-sphere speed       = {config.cap_speed * 1e6:.3f} um/s")
    print(f"Relative speed            = {config.relative_speed * 1e6:.3f} um/s")
    print("Fixed sphere              = top sphere (balance side)")
    print("Moving sphere             = bottom sphere (stage side)")
    print(f"Integration substeps      = {config.integration_substeps}")
    print(f"Contact-radius samples    = {config.contact_radius_samples}")
    if bool(getattr(config, "enable_incompressible_projection", False)):
        projection_label = "velocity-pressure projection on Gmsh tets"
    elif bool(USER_ENFORCE_FULL_AXISYMMETRY):
        projection_label = (
            "Fp,proj force only (no geometric projector)"
            if config.enable_volume_projection
            else "off"
        )
    else:
        projection_label = "on" if config.enable_volume_projection else "off"
    print(f"Volume projection         = {projection_label}")
    if bool(getattr(config, "gmsh_compute_mesh", False)) or bool(
        getattr(config, "enable_incompressible_projection", False)
    ):
        print(
            "Gmsh surface vol. fix     = "
            f"{'on' if config.enable_gmsh_geometric_volume_correction else 'off'}"
            f" (trigger {config.gmsh_geometric_volume_correction_trigger_rel:.3e})"
        )
        print(
            "Gmsh CL volume slide      = "
            f"{'on' if USER_ENABLE_CONTACT_LINE_VOLUME_SLIDE else 'off'}"
            f" (trigger {USER_CONTACT_LINE_VOLUME_SLIDE_TRIGGER_REL:.3e})"
        )
        print(
            "Gmsh position volume fix  = "
            f"{'on' if USER_ENABLE_POSITION_VOLUME_CONSTRAINT else 'off'}"
            f" (trigger {USER_POSITION_VOLUME_CONSTRAINT_TRIGGER_REL:.3e})"
        )
        print(
            "Surface radial step cap   = "
            f"{USER_AXISYM_SURFACE_VOLUME_CORRECTION_MAX_RADIAL_STEP_UM:.3f} um"
        )
    print(
        "Fp,proj trial refinement  = "
        f"{'off (replaced by incompressible projection)' if config.enable_incompressible_projection else ('on' if USER_ENABLE_PRESSURE_TRIAL_REFINEMENT else 'off')}"
    )
    print(f"No-swirl enforcement      = {'on' if config.enforce_no_swirl else 'off'}")
    if config.dt > 0.0 and not config.enable_adaptive_dt:
        print("Adaptive dt               = off (using USER_DT_S)")
        print(f"dt                        = {config.dt:.3e} s")
    else:
        print(f"Adaptive dt               = {'on' if config.enable_adaptive_dt else 'off'}")
        dt_ceiling = float(config.dt) if float(config.dt) > 0.0 else float(config.adaptive_dt_max_s)
        if dt_ceiling > 0.0:
            print(f"dt ceiling                = {dt_ceiling:.3e} s")
        else:
            print("dt ceiling                = none (USER_DT_S < 0: right_dt)")
        print(f"dt floor                  = {config.adaptive_dt_min_s:.3e} s")
        print(f"Capillary dt safety       = {config.adaptive_dt_capillary_safety:.3f}")
        print(f"Mesh disp. fraction       = {config.adaptive_dt_mesh_displacement_frac:.3f}")
    print(f"Steps                     = {config.n_steps}")
    if config.dt > 0.0 and not config.enable_adaptive_dt:
        print(f"Run time                  = {config.dt * config.n_steps:.3f} s")
    else:
        dt_ceiling = float(config.dt) if float(config.dt) > 0.0 else float(config.adaptive_dt_max_s)
        if dt_ceiling > 0.0:
            print(f"Run time ceiling          = {dt_ceiling * config.n_steps:.3f} s")
        else:
            print("Run time ceiling          = adaptive/no fixed ceiling")
    print("=" * 72)


def _print_summary(history: list[dict], config: VolumetricPitoisConfig) -> None:
    fixed_force_mn = np.array([float(row["fixed_force_axial"]) for row in history], dtype=float) * 1.0e3
    d_over_r = np.array([float(row["d_over_r"]) for row in history], dtype=float)
    max_u = np.array([float(row["max_free_speed"]) for row in history], dtype=float)
    print("Separation run complete")
    print(f"D/R range                 = {float(np.min(d_over_r)):.6f} - {float(np.max(d_over_r)):.6f}")
    print(f"Fixed-sphere force [mN]   = {float(np.min(fixed_force_mn)):.6e} - {float(np.max(fixed_force_mn)):.6e}")
    print(f"Final max free speed      = {float(max_u[-1]):.6e} m/s")


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
    _assert_fixed_topology(state)
    target_snapshot_volume_ul = 1.0e9 * float(state.target_snapshot_volume_m3)
    history: list[dict] = [_step_record(state, step=0, t=state.elapsed_time_s)]
    fig_dir = out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    motion_mesh_dir = out_dir

    if save_fig:
        _render_motion_mesh_pngs(state, motion_dir=motion_mesh_dir, label="initial", step=0)
        _refresh_pitois_fig5_compare_png(
            separation_history=history,
            out_dir=out_dir,
        )

    for step in range(config.n_steps):
        step_dt = _select_physical_dt(state)
        substeps = max(1, int(config.integration_substeps))
        sub_dt = float(step_dt) / float(substeps)
        for _substep in range(substeps):
            if _advance_gmsh_incompressible_substep(state, dt=sub_dt):
                continue
            _set_cap_velocities(state)
            _enforce_no_swirl_velocity_field(state)
            _move_caps(state, dt=sub_dt)
            _gmsh_axisymmetrize_state(state)
            _update_duals_and_masses(state)
            _update_pressure_scalar(state)
            _update_pressure_projection_scalar(state, dt=sub_dt)
            symplectic_euler(
                state.HC,
                state.bV_caps,
                _vertex_acceleration,
                dt=sub_dt,
                n_steps=1,
                dim=3,
                retopologize_fn=False,
                workers=max(1, int(state.config.accel_workers)),
                state=state,
            )
            _gmsh_axisymmetrize_state(state)
            _enforce_no_swirl_velocity_field(state)
            _set_cap_velocities(state)
            _update_moving_contact_line(state, dt=sub_dt)
            _gmsh_axisymmetrize_state(state)
            _enforce_no_swirl_velocity_field(state)
            _assert_fixed_topology(state)
            _apply_gmsh_contact_line_volume_slide(state, dt=sub_dt)
            _apply_gmsh_position_volume_constraint(state)
            _project_volume_to_target(state)
            _gmsh_axisymmetrize_state(state)
            _assert_fixed_topology(state)
            _update_duals_and_masses(state)
            _update_pressure_scalar(state)
            _update_pressure_projection_scalar(state, dt=sub_dt, refine=False)
            _assert_fixed_topology(state)
        _assert_state_finite(state, where=f"step {step + 1}")
        state.elapsed_time_s += float(step_dt)

        completed_step = step + 1
        record_row = None
        if verbose or completed_step % max(1, config.record_every) == 0:
            record_row = _step_record(state, step=completed_step, t=state.elapsed_time_s)
        if verbose:
            snapshot_volume_ul = 1.0e9 * _snapshot_msh_volume_m3(state)
            rel_snapshot_volume_error = (snapshot_volume_ul - target_snapshot_volume_ul) / max(
                target_snapshot_volume_ul, 1.0e-30
            )
            print(
                f"Current step              = {completed_step}/{config.n_steps}",
                flush=True,
            )
            print(
                f"Adaptive dt               = {step_dt:.6e} s ({state.last_dt_limiter})",
                flush=True,
            )
            print(
                f"Water vol. error          = {rel_snapshot_volume_error:+.6e} ({100.0 * rel_snapshot_volume_error:+.4f} %)",
                flush=True,
            )
            print(
                f"Water-filled mesh volume  = {snapshot_volume_ul:.6f} uL",
                flush=True,
            )
            if record_row is not None:
                print(
                    "F each step              = "
                    f"fixed_axial {1.0e3 * float(record_row['fixed_force_axial']):+.6e} mN, "
                    f"moving_axial {1.0e3 * float(record_row['moving_force_axial']):+.6e} mN, "
                    f"|fixed| {1.0e3 * float(record_row['fixed_force_mag']):.6e} mN, "
                    f"|moving| {1.0e3 * float(record_row['moving_force_mag']):.6e} mN",
                    flush=True,
                )
                print(
                    "CL slide speed           = "
                    f"bottom {float(record_row['bottom_contact_line_speed']):.6e} m/s, "
                    f"top {float(record_row['top_contact_line_speed']):.6e} m/s",
                    flush=True,
                )
                print(
                    "CL volume slide          = "
                    f"{record_row['gmsh_cl_volume_slide_status']}, "
                    f"scale {float(record_row['gmsh_cl_volume_slide_scale']):.6e}, "
                    f"rel {float(record_row['gmsh_cl_volume_slide_rel_before']):+.6e}"
                    f" -> {float(record_row['gmsh_cl_volume_slide_rel_after']):+.6e}",
                    flush=True,
                )
                print(
                    "Position volume fix      = "
                    f"{record_row['position_volume_constraint_status']}, "
                    f"rel {float(record_row['position_volume_constraint_rel_before']):+.6e}"
                    f" -> {float(record_row['position_volume_constraint_rel_after']):+.6e}",
                    flush=True,
                )
                print(
                    "Surface volume fix       = "
                    f"{record_row['axisym_surface_volume_correction_status']}, "
                    f"alpha {1.0e6 * float(record_row['axisym_surface_volume_correction_alpha_m']):+.6e} um, "
                    f"rel {float(record_row['axisym_surface_volume_correction_rel_before']):+.6e}"
                    f" -> {float(record_row['axisym_surface_volume_correction_rel_after']):+.6e}",
                    flush=True,
                )
                if bool(getattr(state.config, "enable_incompressible_projection", False)):
                    print(
                        "Div(u) projection L2     = "
                        f"{float(state.incompressible_divergence_before_l2):.6e} -> "
                        f"{float(state.incompressible_divergence_after_l2):.6e} 1/s",
                        flush=True,
                    )
        current_record_row = record_row if record_row is not None else _step_record(
            state,
            step=completed_step,
            t=state.elapsed_time_s,
        )
        history.append(current_record_row)
        if save_fig:
            _refresh_pitois_fig5_compare_png(
                separation_history=history,
                out_dir=out_dir,
            )

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
            initial_snapshot_volume_m3=float(state.target_snapshot_volume_m3),
        )

    if save_results:
        _save_history(history, config, out_dir)

    if verbose and history:
        print(f"Target bridge volume      = {target_snapshot_volume_ul:.6f} uL")
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
