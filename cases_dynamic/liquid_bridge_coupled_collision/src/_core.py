"""Shared solver for the Stage-2 coupled liquid-bridge cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from hyperct import Complex

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddgclib.dem import Particle, ParticleSystem, ContactDetector, HertzContact, dem_step
from ddgclib.dynamic_integrators import symplectic_euler
from ddgclib.operators.surface_tension import dual_area_heron, surface_tension_force
from ddgclib.operators.volume import Volume

from ._params import CoupledCaseConfig


OUT_ROOT = Path(__file__).resolve().parents[1] / "out"
_VOLUME = Volume()


@dataclass
class CoupledBridgeState:
    """Runtime state for a coupled DEM-fluid bridge simulation."""

    config: CoupledCaseConfig
    ps: ParticleSystem
    detector: ContactDetector
    contact_model: HertzContact
    HC_film: Complex
    bV_film: set
    particle_vertices: dict[int, set]
    anchored_vertices: dict[int, set]
    target_volume: float = 0.0
    bridge_formed: bool = False
    bridge_step: int | None = None
    bridge_connections: int = 0


def _build_particle_system(
    config: CoupledCaseConfig,
    *,
    left_center_x: float,
    right_center_x: float,
) -> tuple[ParticleSystem, ContactDetector, HertzContact]:
    ps = ParticleSystem(dim=3, gravity=np.zeros(3, dtype=float))
    ps.add(
        Particle.sphere(
            x=[left_center_x, 0.0, 0.0],
            radius=config.particle_radius,
            rho_s=config.rho_s,
            dim=3,
            u=np.array([config.v_approach, 0.0, 0.0], dtype=float),
            wetted=True,
            wetting_angle=0.0,
            liquid_volume=1.0,
        )
    )
    ps.add(
        Particle.sphere(
            x=[right_center_x, 0.0, 0.0],
            radius=config.particle_radius,
            rho_s=config.rho_s,
            dim=3,
            u=np.array([-config.v_approach, 0.0, 0.0], dtype=float),
            wetted=True,
            wetting_angle=0.0,
            liquid_volume=1.0,
        )
    )

    detector = ContactDetector(ps)
    contact_model = HertzContact(
        E=config.contact_E,
        nu=config.contact_nu,
        gamma_n=config.contact_gamma_n,
    )
    return ps, detector, contact_model


def _unit(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-30:
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        return np.array(fallback, dtype=float)
    return np.array(vec, dtype=float) / norm


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = map(float, vec)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _rotation_matrix_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_u = _unit(src)
    dst_u = _unit(dst)
    dot_val = float(np.clip(np.dot(src_u, dst_u), -1.0, 1.0))

    if dot_val > 1.0 - 1.0e-12:
        return np.eye(3, dtype=float)

    if dot_val < -1.0 + 1.0e-12:
        trial = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(src_u, trial))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = _unit(np.cross(src_u, trial))
        K = _skew(axis)
        return np.eye(3, dtype=float) + 2.0 * (K @ K)

    v = np.cross(src_u, dst_u)
    s = float(np.linalg.norm(v))
    K = _skew(v)
    return np.eye(3, dtype=float) + K + (K @ K) * ((1.0 - dot_val) / max(s * s, 1.0e-30))


def _orthonormal_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis_u = _unit(axis)
    trial = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(axis_u, trial))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
    e1 = _unit(np.cross(axis_u, trial), fallback=np.array([0.0, 1.0, 0.0], dtype=float))
    e2 = _unit(np.cross(axis_u, e1), fallback=np.array([0.0, 0.0, 1.0], dtype=float))
    return e1, e2


def _safe_dual_area(v) -> float:
    try:
        area = float(dual_area_heron(v))
    except Exception:
        return 1.0e-18
    if not np.isfinite(area) or area <= 1.0e-18:
        return 1.0e-18
    return area


def _safe_surface_tension_force(v, gamma: float, dim: int = 3) -> np.ndarray:
    try:
        force = np.asarray(surface_tension_force(v, gamma=gamma, dim=dim), dtype=float)
    except Exception:
        return np.zeros(dim, dtype=float)
    if force.shape[0] < dim or not np.all(np.isfinite(force[:dim])):
        return np.zeros(dim, dtype=float)
    return force[:dim]


def _spherical_point(radius: float, theta: float, phi: float) -> np.ndarray:
    return np.array(
        [
            radius * math.cos(theta) * math.sin(phi),
            radius * math.sin(theta) * math.sin(phi),
            radius * math.cos(phi),
        ],
        dtype=float,
    )


def _build_local_cap(radius: float, refinement: int, phi_max: float) -> Complex:
    """Build a structured spherical-cap mesh with regular theta-phi rings."""

    domain = [
        (0.0, 2.0 * math.pi),
        (0.0, float(phi_max)),
    ]

    # Keep roughly the same point count as the previous triangulate/refine path
    # for refinement=2, while producing cleaner, row-like point patterns.
    n_theta = max(6, 2 ** (refinement + 1))
    n_phi = max(3, 2 ** refinement + 1)

    HC_local = Complex(3, domain)
    north_pole = HC_local.V[tuple(_spherical_point(radius, 0.0, 0.0))]

    theta_values = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    phi_values = np.linspace(0.0, float(phi_max), n_phi)

    rings: list[list] = []
    for phi in phi_values[1:]:
        ring = []
        for theta in theta_values:
            coord = _spherical_point(radius, float(theta), float(phi))
            ring.append(HC_local.V[tuple(coord)])
        rings.append(ring)

    if not rings:
        return HC_local

    first_ring = rings[0]
    for i in range(n_theta):
        j = (i + 1) % n_theta
        north_pole.connect(first_ring[i])
        first_ring[i].connect(first_ring[j])

    prev_ring = first_ring
    for ring in rings[1:]:
        for i in range(n_theta):
            j = (i + 1) % n_theta
            ring[i].connect(ring[j])

        for i in range(n_theta):
            j = (i + 1) % n_theta
            prev_ring[i].connect(ring[i])
            prev_ring[i].connect(ring[j])

        prev_ring = ring

    HC_local.V.merge_all(cdist=1.0e-8)
    return HC_local


def _move_vertex(HC, bV, v, x_new: np.ndarray) -> None:
    _ = bV
    HC.V.move(v, tuple(map(float, x_new)))


def _particle_gap_axis(p_self: Particle, p_other: Particle) -> np.ndarray:
    return _unit(p_other.x_a[:3] - p_self.x_a[:3], fallback=np.array([1.0, 0.0, 0.0], dtype=float))


def _copy_cap_to_global(
    HC_local: Complex,
    HC_global: Complex,
    *,
    center: np.ndarray,
    away_dir: np.ndarray,
    particle_id: int,
    particle_vertices: dict[int, set],
) -> None:
    rotation = _rotation_matrix_align(np.array([0.0, 0.0, 1.0], dtype=float), away_dir)
    local_to_global = {}

    for v_local in HC_local.V:
        global_pos = center + rotation @ np.asarray(v_local.x_a, dtype=float)
        v_global = HC_global.V[tuple(map(float, global_pos))]
        v_global.particle_id = particle_id
        v_global.reference_offset = np.asarray(global_pos - center, dtype=float)
        v_global.u = np.zeros(3, dtype=float)
        v_global.p = np.array([0.0], dtype=float)
        v_global.drag_feedback_force = np.zeros(3, dtype=float)
        local_to_global[v_local] = v_global
        particle_vertices[particle_id].add(v_global)

    for v_local in HC_local.V:
        for vn_local in v_local.nn:
            local_to_global[v_local].connect(local_to_global[vn_local])


def _refresh_vertex_masses(HC: Complex, rho_f: float, film_thickness: float) -> None:
    for v in HC.V:
        v.m = rho_f * film_thickness * _safe_dual_area(v)
        v.u = np.asarray(getattr(v, "u", np.zeros(3, dtype=float)), dtype=float)
        v.drag_feedback_force = np.asarray(
            getattr(v, "drag_feedback_force", np.zeros(3, dtype=float)),
            dtype=float,
        )


def _anchored_vertices_for_particle(
    vertices: Iterable,
    *,
    center: np.ndarray,
    gap_axis: np.ndarray,
    anchor_cutoff: float,
) -> set:
    anchored = set()
    for v in vertices:
        r_hat = _unit(np.asarray(v.x_a[:3], dtype=float) - center)
        alignment = float(np.dot(r_hat, gap_axis))
        if alignment < anchor_cutoff:
            anchored.add(v)
    return anchored


def _anchored_vertex_ids(state: CoupledBridgeState) -> set[int]:
    if not state.bridge_formed:
        return {id(v) for v in state.HC_film.V}
    return {
        id(v)
        for vertices in state.anchored_vertices.values()
        for v in vertices
    }


def _reset_boundary_vertices(state: CoupledBridgeState) -> None:
    if state.bridge_formed:
        state.bV_film = {
            v
            for vertices in state.anchored_vertices.values()
            for v in vertices
        }
    else:
        state.bV_film = set(state.HC_film.V)


def _mesh_volume_from_arrays(points: np.ndarray, triangles: np.ndarray) -> float:
    if len(points) == 0 or len(triangles) == 0:
        return 0.0
    return float(_VOLUME((points, triangles), complex_dtype="vf"))


def _mesh_volume(state: CoupledBridgeState) -> float:
    if not state.config.enable_volume_projection:
        return 0.0
    verts, triangles = _extract_triangles(state.HC_film)
    points = np.array([np.asarray(v.x_a[:3], dtype=float) for v in verts], dtype=float)
    return _mesh_volume_from_arrays(points, triangles)


def _project_volume_to_target(state: CoupledBridgeState, *, n_bisect: int = 18) -> None:
    if (not state.config.enable_volume_projection) or (not state.bridge_formed) or state.target_volume <= 0.0:
        return

    verts, triangles = _extract_triangles(state.HC_film)
    if len(verts) == 0 or len(triangles) == 0:
        return

    anchored_ids = _anchored_vertex_ids(state)
    free_indices = [i for i, v in enumerate(verts) if id(v) not in anchored_ids]
    if not free_indices:
        return

    points = np.array([np.asarray(v.x_a[:3], dtype=float) for v in verts], dtype=float)
    target = float(state.target_volume)
    current = _mesh_volume_from_arrays(points, triangles)
    if current <= 0.0:
        return

    if abs(current - target) / max(target, 1.0e-30) < 1.0e-4:
        return

    center = np.mean(points[free_indices], axis=0)
    free_points = points[free_indices].copy()

    def _volume_at(scale: float) -> float:
        trial = points.copy()
        trial[free_indices] = center + scale * (free_points - center)
        return _mesh_volume_from_arrays(trial, triangles)

    low = 1.0
    high = 1.0
    if current < target:
        high = 1.25
        vol_high = _volume_at(high)
        while vol_high < target and high < 8.0:
            high *= 1.5
            vol_high = _volume_at(high)
        if vol_high < target:
            return
    else:
        low = 0.8
        vol_low = _volume_at(low)
        while vol_low > target and low > 0.05:
            low *= 0.75
            vol_low = _volume_at(low)
        if vol_low > target:
            return
        high = 1.0

    for _ in range(n_bisect):
        mid = 0.5 * (low + high)
        vol_mid = _volume_at(mid)
        if vol_mid < target:
            low = mid
        else:
            high = mid

    scale = 0.5 * (low + high)
    new_points = center + scale * (free_points - center)
    for idx, point in zip(free_indices, new_points):
        _move_vertex(state.HC_film, state.bV_film, verts[idx], point)

    _refresh_vertex_masses(state.HC_film, state.config.rho_f, state.config.film_thickness)
    _reset_boundary_vertices(state)


def _setup_disconnected_caps_case(config: CoupledCaseConfig) -> CoupledBridgeState:
    R = config.particle_radius
    x_offset = R + 0.5 * config.initial_sep

    ps, detector, contact_model = _build_particle_system(
        config,
        left_center_x=-x_offset,
        right_center_x=x_offset,
    )
    p1, p2 = ps.particles

    HC_local = _build_local_cap(R + config.film_thickness, config.cap_refinement, config.cap_phi_max)
    HC_film = Complex(3, domain=None)
    particle_vertices = {p1.id: set(), p2.id: set()}

    common_axis = _particle_gap_axis(p1, p2)
    _copy_cap_to_global(
        HC_local,
        HC_film,
        center=p1.x_a[:3],
        away_dir=-common_axis,
        particle_id=p1.id,
        particle_vertices=particle_vertices,
    )
    _copy_cap_to_global(
        HC_local,
        HC_film,
        center=p2.x_a[:3],
        away_dir=common_axis,
        particle_id=p2.id,
        particle_vertices=particle_vertices,
    )

    anchored_vertices = {
        p1.id: _anchored_vertices_for_particle(
            particle_vertices[p1.id],
            center=p1.x_a[:3],
            gap_axis=_particle_gap_axis(p1, p2),
            anchor_cutoff=config.anchor_gap_alignment,
        ),
        p2.id: _anchored_vertices_for_particle(
            particle_vertices[p2.id],
            center=p2.x_a[:3],
            gap_axis=_particle_gap_axis(p2, p1),
            anchor_cutoff=config.anchor_gap_alignment,
        ),
    }

    bV_film = set(HC_film.V)
    _refresh_vertex_masses(HC_film, config.rho_f, config.film_thickness)

    state = CoupledBridgeState(
        config=config,
        ps=ps,
        detector=detector,
        contact_model=contact_model,
        HC_film=HC_film,
        bV_film=bV_film,
        particle_vertices=particle_vertices,
        anchored_vertices=anchored_vertices,
    )
    _sync_surface_to_particles(state)
    _reset_boundary_vertices(state)
    state.target_volume = _mesh_volume(state)
    return state


def _load_equilibrium_catenoid_template(refinement: int):
    from cases_dynamic.liquid_bridge_equilibrium.Case_1_equilibrium_particle_particle_bridge_benchmark import (
        _build_live_endres_catenoid,
    )

    HC_unit, bV_unit = _build_live_endres_catenoid(refinement)
    verts_unit = list(HC_unit.V)
    min_z = min(float(v.x_a[2]) for v in verts_unit)
    max_z = max(float(v.x_a[2]) for v in verts_unit)
    lower_ring = [v for v in verts_unit if abs(float(v.x_a[2]) - min_z) < 1.0e-10]
    upper_ring = [v for v in verts_unit if abs(float(v.x_a[2]) - max_z) < 1.0e-10]
    lower_ids = {id(v) for v in lower_ring}
    upper_ids = {id(v) for v in upper_ring}
    boundary_ids = {id(v) for v in bV_unit}
    ring_radius = float(
        np.mean(
            [
                np.linalg.norm(np.asarray(v.x_a[:2], dtype=float))
                for v in lower_ring
            ]
        )
    )
    half_length = 0.5 * (max_z - min_z)
    return HC_unit, lower_ids, upper_ids, boundary_ids, ring_radius, half_length


def _solve_catenoid_ring_radius(R: float, gap: float, k_ratio: float) -> float:
    if gap < 0.0:
        raise ValueError("The catenoid initializer requires a non-negative particle gap.")

    rbs = np.linspace(1.0e-6 * R, 0.999999 * R, 8192)
    gaps = 2.0 * (k_ratio * rbs + np.sqrt(np.maximum(R * R - rbs * rbs, 0.0)) - R)
    max_gap = float(np.max(gaps))
    if gap > max_gap + 1.0e-10 * max(R, 1.0):
        raise ValueError(
            f"Requested gap {gap:.6e} m exceeds the catenoid-compatible maximum "
            f"{max_gap:.6e} m for particle radius {R:.6e} m."
        )

    diff = np.abs(gaps - gap)
    best = float(np.min(diff))
    candidate_idx = np.where(diff <= max(best * 1.05, 1.0e-8 * R))[0]
    if candidate_idx.size == 0:
        idx = int(np.argmin(diff))
    else:
        idx = int(candidate_idx[-1])
    return float(rbs[idx])


def _setup_equilibrium_catenoid_case(config: CoupledCaseConfig) -> CoupledBridgeState:
    R = config.particle_radius
    (
        HC_unit,
        lower_ids,
        upper_ids,
        _boundary_ids,
        unit_ring_radius,
        unit_half_length,
    ) = _load_equilibrium_catenoid_template(config.equilibrium_catenoid_refinement)

    k_ratio = unit_half_length / max(unit_ring_radius, 1.0e-30)
    ring_radius = _solve_catenoid_ring_radius(R, config.initial_sep, k_ratio)
    scale = ring_radius / max(unit_ring_radius, 1.0e-30)
    half_length = scale * unit_half_length
    axial_offset = math.sqrt(max(R * R - ring_radius * ring_radius, 0.0))

    left_center_x = -half_length - axial_offset
    right_center_x = half_length + axial_offset
    ps, detector, contact_model = _build_particle_system(
        config,
        left_center_x=left_center_x,
        right_center_x=right_center_x,
    )
    p1, p2 = ps.particles

    HC_film = Complex(3, domain=None)
    particle_vertices = {p1.id: set(), p2.id: set()}
    anchored_vertices = {p1.id: set(), p2.id: set()}
    local_to_global = {}

    for v_unit in HC_unit.V:
        x_u, y_u, z_u = map(float, v_unit.x_a)
        # Rotate the equilibrium catenoid so the bridge axis is along x.
        global_pos = np.array([scale * z_u, scale * x_u, scale * y_u], dtype=float)
        v_global = HC_film.V[tuple(map(float, global_pos))]
        nearest_pid = p1.id if abs(global_pos[0] - p1.x_a[0]) <= abs(global_pos[0] - p2.x_a[0]) else p2.id
        v_global.particle_id = 0 if nearest_pid == p1.id else 1
        v_global.u = np.zeros(3, dtype=float)
        v_global.p = np.array([0.0], dtype=float)
        v_global.drag_feedback_force = np.zeros(3, dtype=float)
        local_to_global[v_unit] = v_global
        particle_vertices[nearest_pid].add(v_global)

        if id(v_unit) in lower_ids:
            v_global.reference_offset = np.asarray(global_pos - p1.x_a[:3], dtype=float)
            anchored_vertices[p1.id].add(v_global)
        elif id(v_unit) in upper_ids:
            v_global.reference_offset = np.asarray(global_pos - p2.x_a[:3], dtype=float)
            anchored_vertices[p2.id].add(v_global)

    for v_unit in HC_unit.V:
        for vn_unit in v_unit.nn:
            local_to_global[v_unit].connect(local_to_global[vn_unit])

    bV_film = set().union(*anchored_vertices.values())
    _refresh_vertex_masses(HC_film, config.rho_f, config.film_thickness)

    state = CoupledBridgeState(
        config=config,
        ps=ps,
        detector=detector,
        contact_model=contact_model,
        HC_film=HC_film,
        bV_film=bV_film,
        particle_vertices=particle_vertices,
        anchored_vertices=anchored_vertices,
        target_volume=0.0,
        bridge_formed=True,
        bridge_step=0,
        bridge_connections=0,
    )
    _sync_surface_to_particles(state)
    _reset_boundary_vertices(state)
    return state


def setup_case(config: CoupledCaseConfig) -> CoupledBridgeState:
    """Build particles, fluid film, and initial frozen sets."""

    if config.initial_geometry == "equilibrium_catenoid":
        return _setup_equilibrium_catenoid_case(config)
    return _setup_disconnected_caps_case(config)


def _sync_surface_to_particles(state: CoupledBridgeState) -> None:
    p_lookup = {p.id: p for p in state.ps.particles}

    for pid, vertices in state.particle_vertices.items():
        particle = p_lookup[pid]
        if state.bridge_formed:
            move_vertices = state.anchored_vertices[pid]
        else:
            move_vertices = vertices

        for v in move_vertices:
            reference_offset = np.asarray(getattr(v, "reference_offset", np.zeros(3, dtype=float)), dtype=float)
            new_pos = particle.x_a[:3] + reference_offset
            _move_vertex(state.HC_film, state.bV_film, v, new_pos)
            v.u[:3] = particle.u[:3]

    _reset_boundary_vertices(state)


def _ordered_rim_loop(vertices: Iterable, center: np.ndarray, axis: np.ndarray) -> list:
    axis_u = _unit(axis)
    e1, e2 = _orthonormal_frame(axis_u)

    scored = []
    for v in vertices:
        r_vec = np.asarray(v.x_a[:3], dtype=float) - center
        r_hat = _unit(r_vec, fallback=axis_u)
        scored.append((float(np.dot(r_hat, axis_u)), v))

    max_alignment = max(alignment for alignment, _v in scored)
    rim = [v for alignment, v in scored if alignment >= max_alignment - 1.0e-8]

    def _angle(v) -> float:
        r_vec = np.asarray(v.x_a[:3], dtype=float) - center
        return math.atan2(float(np.dot(r_vec, e2)), float(np.dot(r_vec, e1)))

    return sorted(rim, key=_angle)


def _best_cyclic_shift(loop_a: list, loop_b: list) -> list:
    if not loop_a or not loop_b:
        return []

    if len(loop_a) != len(loop_b):
        target = min(len(loop_a), len(loop_b))

        def _sample(loop: list, n_target: int) -> list:
            if len(loop) == n_target:
                return loop
            indices = np.linspace(0, len(loop) - 1, n_target, dtype=int)
            return [loop[idx] for idx in indices]

        loop_a = _sample(loop_a, target)
        loop_b = _sample(loop_b, target)

    if not loop_a:
        return []

    coords_a = np.array([v.x_a[:3] for v in loop_a], dtype=float)
    coords_b = np.array([v.x_a[:3] for v in loop_b], dtype=float)

    best_shift = 0
    best_cost = float("inf")
    n = len(loop_a)
    for shift in range(n):
        rolled = np.roll(coords_b, shift=shift, axis=0)
        cost = float(np.mean(np.linalg.norm(coords_a - rolled, axis=1)))
        if cost < best_cost:
            best_cost = cost
            best_shift = shift

    return [loop_b[(i + best_shift) % n] for i in range(n)]


def _extract_triangles(HC: Complex) -> tuple[list, np.ndarray]:
    """Extract triangular faces from the 1-ring graph connectivity."""

    verts = list(HC.V)
    index = {id(v): i for i, v in enumerate(verts)}
    triangles = set()

    for i, v in enumerate(verts):
        nbr_ids = sorted(index[id(nb)] for nb in v.nn if index[id(nb)] > i)
        for k, j in enumerate(nbr_ids):
            j_neighbors = {index[id(nb)] for nb in verts[j].nn}
            for ell in nbr_ids[k + 1 :]:
                if ell in j_neighbors:
                    triangles.add(tuple(sorted((i, j, ell))))

    return verts, np.array(sorted(triangles), dtype=int)


def _capture_mesh_snapshot(state: CoupledBridgeState, label: str, *, step: int | None = None) -> dict:
    """Freeze the current interface mesh into a renderable snapshot."""

    verts, triangles = _extract_triangles(state.HC_film)
    coords = np.array([np.asarray(v.x_a[:3], dtype=float) for v in verts], dtype=float)
    anchored_ids = _anchored_vertex_ids(state)
    boundary_mask = np.array([id(v) in anchored_ids for v in verts], dtype=bool)
    particle_ids = np.array([int(getattr(v, "particle_id", -1)) for v in verts], dtype=int)
    particle_centers = np.array([np.asarray(p.x_a[:3], dtype=float) for p in state.ps.particles], dtype=float)
    particle_radii = np.array([float(p.radius) for p in state.ps.particles], dtype=float)

    return {
        "label": label,
        "step": -1 if step is None else int(step),
        "coords": coords,
        "triangles": triangles,
        "boundary_mask": boundary_mask,
        "particle_ids": particle_ids,
        "particle_centers": particle_centers,
        "particle_radii": particle_radii,
        "bridge_formed": bool(state.bridge_formed),
        "n_vertices": int(coords.shape[0]),
        "n_free_vertices": int(np.count_nonzero(~boundary_mask)),
    }


def _particle_color(pid: int) -> str:
    if pid == 0:
        return "#2f6db2"
    if pid == 1:
        return "#d07a1f"
    return "#6c757d"


def _mesh_axis_box(points: np.ndarray, particle_centers: np.ndarray, particle_radii: np.ndarray) -> tuple[np.ndarray, float]:
    all_points = [points]
    for center, radius in zip(particle_centers, particle_radii):
        all_points.append(
            np.array(
                [
                    center - radius,
                    center + radius,
                ],
                dtype=float,
            )
        )

    merged = np.vstack(all_points)
    mins = np.min(merged, axis=0)
    maxs = np.max(merged, axis=0)
    spans = np.maximum(maxs - mins, 1.0e-12)
    radius = 0.55 * float(np.max(spans))
    center = 0.5 * (mins + maxs)

    return center, radius


def _set_axes_equal(
    ax,
    points: np.ndarray,
    particle_centers: np.ndarray,
    particle_radii: np.ndarray,
    *,
    axis_box: tuple[np.ndarray, float] | None = None,
) -> None:
    if axis_box is None:
        center, radius = _mesh_axis_box(points, particle_centers, particle_radii)
    else:
        center, radius = axis_box

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _render_particle_wireframes(ax, particle_centers: np.ndarray, particle_radii: np.ndarray) -> None:
    u = np.linspace(0.0, 2.0 * math.pi, 24)
    v = np.linspace(0.0, math.pi, 12)
    uu, vv = np.meshgrid(u, v)

    for idx, (center, radius) in enumerate(zip(particle_centers, particle_radii)):
        x = center[0] + radius * np.cos(uu) * np.sin(vv)
        y = center[1] + radius * np.sin(uu) * np.sin(vv)
        z = center[2] + radius * np.cos(vv)
        ax.plot_wireframe(
            x,
            y,
            z,
            rstride=2,
            cstride=2,
            color=_particle_color(idx),
            linewidth=0.35,
            alpha=0.22,
        )


def _camera_angles_for_particles(particle_centers: np.ndarray) -> tuple[float, float]:
    """Use a fixed transverse view for the current Stage-2 two-particle cases."""

    _ = particle_centers
    return 23.0, -90.0


def _shared_mesh_axis_box(snapshots: dict[str, dict]) -> tuple[np.ndarray, float] | None:
    bounds = []
    for snapshot in snapshots.values():
        coords = np.asarray(snapshot["coords"], dtype=float)
        if coords.size:
            bounds.append(coords)

        particle_centers = np.asarray(snapshot["particle_centers"], dtype=float)
        particle_radii = np.asarray(snapshot["particle_radii"], dtype=float)
        for center, radius in zip(particle_centers, particle_radii):
            bounds.append(
                np.array(
                    [
                        center - radius,
                        center + radius,
                    ],
                    dtype=float,
                )
            )

    if not bounds:
        return None

    merged = np.vstack(bounds)
    mins = np.min(merged, axis=0)
    maxs = np.max(merged, axis=0)
    spans = np.maximum(maxs - mins, 1.0e-12)
    radius = 0.55 * float(np.max(spans))
    center = 0.5 * (mins + maxs)
    return center, radius


def _render_mesh_snapshot(
    config: CoupledCaseConfig,
    snapshot: dict,
    out_path: Path,
    *,
    axis_box: tuple[np.ndarray, float] | None = None,
) -> Path:
    """Render one 3D mesh PNG for the current interface configuration."""

    coords = snapshot["coords"]
    triangles = snapshot["triangles"]
    boundary_mask = snapshot["boundary_mask"]
    particle_ids = snapshot["particle_ids"]
    particle_centers = snapshot["particle_centers"]
    particle_radii = snapshot["particle_radii"]

    fig = plt.figure(figsize=(7.0, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_proj_type("ortho")

    if len(triangles) > 0:
        particle_tris = []
        particle_facecolors = []
        bridge_tris = []
        bridge_facecolors = []

        for tri in triangles:
            tri_ids = {int(pid) for pid in particle_ids[np.asarray(tri, dtype=int)]}
            if len(tri_ids) == 1 and next(iter(tri_ids)) in (0, 1):
                particle_tris.append(tri)
                particle_facecolors.append(
                    mcolors.to_rgba(_particle_color(next(iter(tri_ids))), alpha=0.74)
                )
            else:
                bridge_tris.append(tri)
                bridge_facecolors.append(mcolors.to_rgba("#1f9d8a", alpha=0.88))

        if particle_tris:
            surf_particle = ax.plot_trisurf(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                triangles=np.asarray(particle_tris, dtype=int),
                linewidth=0.55,
                edgecolor="#1f2933",
                antialiased=True,
                shade=False,
                color="#d9dde3",
                alpha=0.9,
            )
            surf_particle.set_facecolors(particle_facecolors)

        if bridge_tris:
            surf_bridge = ax.plot_trisurf(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                triangles=np.asarray(bridge_tris, dtype=int),
                linewidth=0.48,
                edgecolor="#111111",
                antialiased=True,
                shade=False,
                color="#1f9d8a",
                alpha=0.9,
            )
            surf_bridge.set_facecolors(bridge_facecolors)

    free_mask = ~boundary_mask
    if np.any(free_mask):
        free_colors = [_particle_color(int(pid)) for pid in particle_ids[free_mask]]
        ax.scatter(
            coords[free_mask, 0],
            coords[free_mask, 1],
            coords[free_mask, 2],
            s=11,
            c=free_colors,
            edgecolors="none",
            alpha=0.95,
            depthshade=False,
        )
    if np.any(boundary_mask):
        ax.scatter(
            coords[boundary_mask, 0],
            coords[boundary_mask, 1],
            coords[boundary_mask, 2],
            s=16,
            c="#111111",
            edgecolors="none",
            alpha=0.95,
            depthshade=False,
        )

    _render_particle_wireframes(ax, particle_centers, particle_radii)
    ax.plot(
        particle_centers[:, 0],
        particle_centers[:, 1],
        particle_centers[:, 2],
        color="#5b6470",
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
    )
    ax.scatter(
        particle_centers[:, 0],
        particle_centers[:, 1],
        particle_centers[:, 2],
        s=26,
        c=[_particle_color(i) for i in range(len(particle_centers))],
        edgecolors="#111111",
        linewidths=0.4,
        depthshade=False,
    )

    _set_axes_equal(ax, coords, particle_centers, particle_radii, axis_box=axis_box)
    elev, azim = _camera_angles_for_particles(particle_centers)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        f"{config.title}: {snapshot['label']} mesh\n"
        f"vertices={snapshot['n_vertices']}, free={snapshot['n_free_vertices']}",
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def _render_mesh_pngs(out_dir: Path, config: CoupledCaseConfig, snapshots: dict[str, dict]) -> list[Path]:
    """Render mesh snapshots to PNG files in the case figure directory."""

    fig_dir = out_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ordered: list[tuple[str, str]] = [("initial", "mesh_initial.png")]
    iter_keys = sorted(
        [key for key in snapshots if key.startswith("iter_")],
        key=lambda key: int(key.split("_", 1)[1]),
    )
    for key in iter_keys:
        iter_step = int(key.split("_", 1)[1])
        ordered.append((key, f"mesh_iter{iter_step:04d}.png"))
    ordered.extend(
        [
            ("bridge", "mesh_bridge.png"),
            ("final", "mesh_final.png"),
        ]
    )
    paths: list[Path] = []
    axis_box = _shared_mesh_axis_box(snapshots)

    for stale_path in fig_dir.glob("mesh_iter*.png"):
        stale_path.unlink()

    for key, filename in ordered:
        snapshot = snapshots.get(key)
        if snapshot is None:
            continue
        out_path = fig_dir / filename
        paths.append(_render_mesh_snapshot(config, snapshot, out_path, axis_box=axis_box))

    return paths


def _rim_gap_metrics(state: CoupledBridgeState) -> tuple[float, float]:
    if state.bridge_formed:
        min_cross = _min_cross_patch_distance(state)
        return min_cross, min_cross

    p1, p2 = state.ps.particles
    loop_1 = _ordered_rim_loop(state.particle_vertices[p1.id], p1.x_a[:3], _particle_gap_axis(p1, p2))
    loop_2 = _ordered_rim_loop(state.particle_vertices[p2.id], p2.x_a[:3], _particle_gap_axis(p2, p1))
    loop_2 = _best_cyclic_shift(loop_1, loop_2)
    if not loop_2:
        return float("inf"), float("inf")

    if not loop_1:
        return float("inf"), float("inf")

    distances = np.array(
        [
            np.linalg.norm(np.asarray(v1.x_a[:3], dtype=float) - np.asarray(v2.x_a[:3], dtype=float))
            for v1, v2 in zip(loop_1, loop_2)
        ],
        dtype=float,
    )
    return float(np.min(distances)), float(np.mean(distances))


def _stitch_bridge(state: CoupledBridgeState) -> int:
    p1, p2 = state.ps.particles
    loop_1 = _ordered_rim_loop(state.particle_vertices[p1.id], p1.x_a[:3], _particle_gap_axis(p1, p2))
    loop_2 = _ordered_rim_loop(state.particle_vertices[p2.id], p2.x_a[:3], _particle_gap_axis(p2, p1))
    loop_2 = _best_cyclic_shift(loop_1, loop_2)

    n = len(loop_1)
    if n == 0:
        return 0

    mid_ring = []
    for idx in range(n):
        a_i = loop_1[idx]
        b_i = loop_2[idx]
        midpoint = 0.5 * (
            np.asarray(a_i.x_a[:3], dtype=float) + np.asarray(b_i.x_a[:3], dtype=float)
        )
        v_mid = state.HC_film.V[tuple(map(float, midpoint))]
        v_mid.particle_id = -1
        v_mid.u = 0.5 * (
            np.asarray(getattr(a_i, "u", np.zeros(3, dtype=float)), dtype=float)
            + np.asarray(getattr(b_i, "u", np.zeros(3, dtype=float)), dtype=float)
        )
        v_mid.p = np.array([0.0], dtype=float)
        v_mid.drag_feedback_force = np.zeros(3, dtype=float)
        mid_ring.append(v_mid)

    n_new = 0
    for idx in range(n):
        a_i = loop_1[idx]
        a_next = loop_1[(idx + 1) % n]
        b_i = loop_2[idx]
        b_next = loop_2[(idx + 1) % n]
        m_i = mid_ring[idx]
        m_next = mid_ring[(idx + 1) % n]

        for v_left, v_right in (
            (m_i, m_next),
            (a_i, m_i),
            (a_next, m_i),
            (b_i, m_i),
            (b_next, m_i),
        ):
            if v_right not in v_left.nn:
                v_left.connect(v_right)
                n_new += 1

    state.bridge_formed = True
    _reset_boundary_vertices(state)
    _refresh_vertex_masses(state.HC_film, state.config.rho_f, state.config.film_thickness)
    return n_new


def _clear_drag_feedback(state: CoupledBridgeState) -> None:
    for v in state.HC_film.V:
        v.drag_feedback_force = np.zeros(3, dtype=float)


def _refresh_drag_feedback(state: CoupledBridgeState) -> dict[int, np.ndarray]:
    _clear_drag_feedback(state)

    particle_drag = {p.id: np.zeros(3, dtype=float) for p in state.ps.particles}
    if not state.bridge_formed:
        return particle_drag

    p_lookup = {p.id: p for p in state.ps.particles}
    coeff_prefactor = state.config.mu_f / max(state.config.film_thickness, 1.0e-12)
    anchored_ids = _anchored_vertex_ids(state)

    for pid, vertices in state.particle_vertices.items():
        particle = p_lookup[pid]
        for v in vertices:
            if id(v) in anchored_ids:
                continue
            coeff = coeff_prefactor * _safe_dual_area(v)
            force_on_fluid = coeff * (particle.u[:3] - v.u[:3])
            v.drag_feedback_force += force_on_fluid
            particle_drag[pid] -= force_on_fluid

    return particle_drag


def _capillary_forces(state: CoupledBridgeState) -> dict[int, np.ndarray]:
    particle_capillary = {p.id: np.zeros(3, dtype=float) for p in state.ps.particles}
    if not state.bridge_formed:
        return particle_capillary

    for pid, vertices in state.particle_vertices.items():
        for v in vertices:
            particle_capillary[pid] -= _safe_surface_tension_force(v, gamma=state.config.gamma, dim=3)
    return particle_capillary


def _fluid_force_bundle(state: CoupledBridgeState) -> dict[str, dict[int, np.ndarray]]:
    drag = _refresh_drag_feedback(state)
    capillary = _capillary_forces(state)
    total = {
        pid: capillary[pid] + drag[pid]
        for pid in capillary.keys()
    }
    return {
        "drag": drag,
        "capillary": capillary,
        "total": total,
    }


def _coupled_surface_acceleration(
    v,
    *,
    gamma: float,
    damping: float,
    max_accel: float,
    dim: int = 3,
) -> np.ndarray:
    force = _safe_surface_tension_force(v, gamma=gamma, dim=dim)
    force += np.asarray(getattr(v, "drag_feedback_force", np.zeros(dim, dtype=float)), dtype=float)[:dim]
    if damping > 0.0:
        force -= damping * np.asarray(v.u[:dim], dtype=float)

    mass = max(float(getattr(v, "m", 0.0)), 1.0e-18)
    accel = force / mass
    accel_norm = float(np.linalg.norm(accel))
    if accel_norm > max_accel:
        accel *= max_accel / accel_norm
    return accel


def _particle_gap(ps: ParticleSystem) -> float:
    p1, p2 = ps.particles
    return float(np.linalg.norm(p2.x_a[:3] - p1.x_a[:3])) - p1.radius - p2.radius


def _min_cross_patch_distance(state: CoupledBridgeState) -> float:
    p1, p2 = state.ps.particles
    verts_1 = list(state.particle_vertices[p1.id])
    verts_2 = list(state.particle_vertices[p2.id])
    if not verts_1 or not verts_2:
        return float("inf")

    coords_1 = np.array([v.x_a[:3] for v in verts_1], dtype=float)
    coords_2 = np.array([v.x_a[:3] for v in verts_2], dtype=float)
    min_dist = float("inf")
    for coord in coords_1:
        dist = np.linalg.norm(coords_2 - coord[np.newaxis, :], axis=1)
        min_dist = min(min_dist, float(np.min(dist)))
    return min_dist


def _max_free_vertex_speed(state: CoupledBridgeState) -> float:
    anchored_ids = _anchored_vertex_ids(state)
    speeds = [
        float(np.linalg.norm(v.u[:3]))
        for v in state.HC_film.V
        if id(v) not in anchored_ids
    ]
    return max(speeds, default=0.0)


def _bridge_neck_radius(state: CoupledBridgeState) -> float:
    if not state.bridge_formed:
        return float("inf")

    p1, p2 = state.ps.particles
    axis = _unit(p2.x_a[:3] - p1.x_a[:3], fallback=np.array([1.0, 0.0, 0.0], dtype=float))
    origin = 0.5 * (p1.x_a[:3] + p2.x_a[:3])
    radial_distances = []

    for v in state.HC_film.V:
        rel = np.asarray(v.x_a[:3], dtype=float) - origin
        axial = float(np.dot(rel, axis))
        radial = rel - axial * axis
        radial_distances.append(float(np.linalg.norm(radial)))

    return min(radial_distances, default=float("inf"))


def _step_record(
    state: CoupledBridgeState,
    *,
    step: int,
    t: float,
    force_bundle: dict[str, dict[int, np.ndarray]],
    min_rim_gap: float,
    mean_rim_gap: float,
) -> dict[str, float | int | bool | list[float]]:
    p1, p2 = state.ps.particles
    axis = _particle_gap_axis(p1, p2)
    total_force = force_bundle["total"][p1.id]
    cap_force = force_bundle["capillary"][p1.id]
    drag_force = force_bundle["drag"][p1.id]
    anchored_ids = _anchored_vertex_ids(state)
    mesh_volume = _mesh_volume(state)

    return {
        "step": step,
        "t": t,
        "x1": p1.x_a[:3].tolist(),
        "x2": p2.x_a[:3].tolist(),
        "v1": p1.u[:3].tolist(),
        "v2": p2.u[:3].tolist(),
        "sep": _particle_gap(state.ps),
        "contact_overlap": max(0.0, -_particle_gap(state.ps)),
        "bridge_formed": bool(state.bridge_formed),
        "bridge_step": -1 if state.bridge_step is None else int(state.bridge_step),
        "bridge_connections": int(state.bridge_connections),
        "min_rim_gap": min_rim_gap,
        "mean_rim_gap": mean_rim_gap,
        "min_cross_patch_distance": _min_cross_patch_distance(state),
        "n_vertices": sum(1 for _ in state.HC_film.V),
        "n_free_vertices": sum(1 for v in state.HC_film.V if id(v) not in anchored_ids),
        "mesh_volume": mesh_volume,
        "volume_error": mesh_volume - state.target_volume,
        "max_free_vertex_speed": _max_free_vertex_speed(state),
        "neck_radius": _bridge_neck_radius(state),
        "capillary_force_1": cap_force.tolist(),
        "drag_force_1": drag_force.tolist(),
        "total_force_1": total_force.tolist(),
        "capillary_force_mag": float(np.linalg.norm(cap_force)),
        "drag_force_mag": float(np.linalg.norm(drag_force)),
        "total_force_mag": float(np.linalg.norm(total_force)),
        "capillary_force_axial": float(np.dot(cap_force, axis)),
        "drag_force_axial": float(np.dot(drag_force, axis)),
        "total_force_axial": float(np.dot(total_force, axis)),
    }


def _advance_prescribed_particles(state: CoupledBridgeState) -> None:
    for particle in state.ps.particles:
        particle.x_a[:3] += particle.u[:3] * state.config.dt


def coupled_step(state: CoupledBridgeState, step: int) -> dict[str, dict[int, np.ndarray]]:
    """Advance fluid and DEM inside one synchronized macro time step."""

    _sync_surface_to_particles(state)
    _reset_boundary_vertices(state)

    min_rim_gap, mean_rim_gap = _rim_gap_metrics(state)
    # Trigger bridge creation from the nearest facing rim points. This is more
    # robust to structured-vs-unstructured rim sampling than a mean-gap test.
    if (not state.bridge_formed) and min_rim_gap <= state.config.bridge_connection_distance:
        state.bridge_connections = _stitch_bridge(state)
        state.bridge_step = step
        _sync_surface_to_particles(state)
        _project_volume_to_target(state)
        min_rim_gap, mean_rim_gap = _rim_gap_metrics(state)

    dudt_fn = lambda v: _coupled_surface_acceleration(
        v,
        gamma=state.config.gamma,
        damping=state.config.film_damping,
        max_accel=state.config.max_surface_acceleration,
        dim=3,
    )

    for _ in range(max(1, state.config.fluid_substeps)):
        _reset_boundary_vertices(state)
        _refresh_drag_feedback(state)
        symplectic_euler(
            state.HC_film,
            state.bV_film,
            dudt_fn,
            dt=state.config.dt_fluid,
            n_steps=1,
            dim=3,
            retopologize_fn=False,
        )

    _project_volume_to_target(state)

    force_bundle = _fluid_force_bundle(state)

    def _external_forces(ps: ParticleSystem) -> None:
        for p in ps.particles:
            p.force[:3] += force_bundle["total"].get(p.id, np.zeros(3, dtype=float))

    if state.config.prescribed_particle_motion:
        _advance_prescribed_particles(state)
    else:
        dem_step(
            state.ps,
            state.detector,
            state.contact_model,
            dt=state.config.dt,
            dim=3,
            method="velocity_verlet",
            external_forces_fn=_external_forces,
        )

    _sync_surface_to_particles(state)
    _reset_boundary_vertices(state)
    return force_bundle


def _print_case_header(config: CoupledCaseConfig) -> None:
    print("=" * 68)
    print(f"  {config.title}")
    print("=" * 68)
    print(f"Particle radius           = {config.particle_radius * 1e3:.3f} mm")
    print(f"Initial gap               = {config.initial_sep * 1e6:.1f} um")
    print(f"Approach velocity         = {config.v_approach:.4f} m/s per particle")
    print(f"Surface tension           = {config.gamma:.4f} N/m")
    print(f"Fluid viscosity           = {config.mu_f:.2e} Pa s")
    print(f"Film thickness            = {config.film_thickness * 1e6:.1f} um")
    print(f"Cap refinement            = {config.cap_refinement}")
    print(f"Initial geometry          = {config.initial_geometry}")
    print(f"Bridge connection dist.   = {config.bridge_connection_distance * 1e6:.1f} um")
    print(f"dt                        = {config.dt:.2e} s")
    print(f"Fluid substeps            = {config.fluid_substeps}")
    print(f"Volume projection         = {config.enable_volume_projection}")
    print(f"Total macro steps         = {config.n_steps}")
    print("=" * 68)


def _print_summary(history: list[dict], config: CoupledCaseConfig) -> None:
    final = history[-1]
    bridge_rows = [row for row in history if row["bridge_formed"]]
    bridge_time_ms = None
    if bridge_rows:
        bridge_time_ms = 1.0e3 * float(bridge_rows[0]["t"])

    max_capillary = max(float(row["capillary_force_mag"]) for row in history)
    max_drag = max(float(row["drag_force_mag"]) for row in history)
    min_sep = min(float(row["sep"]) for row in history)

    print()
    print(f"Completed {config.name}")
    if bridge_time_ms is None:
        print("Bridge formed             = no")
    else:
        print(f"Bridge formed             = yes at {bridge_time_ms:.3f} ms")
    print(f"Minimum separation        = {min_sep * 1e6:.2f} um")
    print(f"Final separation          = {float(final['sep']) * 1e6:.2f} um")
    print(f"Final free vertices       = {int(final['n_free_vertices'])}")
    print(f"Maximum capillary force   = {max_capillary * 1e6:.4f} uN")
    print(f"Maximum drag force        = {max_drag * 1e6:.4f} uN")
    print(f"Final max free-vert speed = {float(final['max_free_vertex_speed']):.6f} m/s")


def _save_history(out_dir: Path, config: CoupledCaseConfig, history: list[dict]) -> None:
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "history": history,
    }
    (results_dir / "history.json").write_text(json.dumps(payload, indent=2))


def _render_figures(out_dir: Path, config: CoupledCaseConfig, history: list[dict]) -> list[Path]:
    fig_dir = out_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    times_ms = np.array([float(row["t"]) for row in history], dtype=float) * 1.0e3
    sep_um = np.array([float(row["sep"]) for row in history], dtype=float) * 1.0e6
    cap_uN = np.array([float(row["capillary_force_mag"]) for row in history], dtype=float) * 1.0e6
    drag_uN = np.array([float(row["drag_force_mag"]) for row in history], dtype=float) * 1.0e6
    total_uN = np.array([float(row["total_force_mag"]) for row in history], dtype=float) * 1.0e6
    rim_gap_um = np.array([float(row["mean_rim_gap"]) for row in history], dtype=float) * 1.0e6
    min_cross_um = np.array([float(row["min_cross_patch_distance"]) for row in history], dtype=float) * 1.0e6
    free_vertices = np.array([int(row["n_free_vertices"]) for row in history], dtype=float)

    bridge_rows = [row for row in history if row["bridge_formed"]]
    bridge_time_ms = None
    if bridge_rows:
        bridge_time_ms = 1.0e3 * float(bridge_rows[0]["t"])

    paths: list[Path] = []

    fig1, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(times_ms, sep_um, color="tab:blue", linewidth=1.8)
    if bridge_time_ms is not None:
        ax1.axvline(bridge_time_ms, color="tab:red", linestyle="--", alpha=0.7, label="bridge formed")
        ax1.legend(loc="best")
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Particle gap [um]")
    ax1.set_title(f"{config.title}: separation")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    path1 = fig_dir / "separation.png"
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)
    paths.append(path1)

    fig2, ax2 = plt.subplots(figsize=(8, 4.8))
    ax2.plot(times_ms, cap_uN, color="tab:green", linewidth=1.8, label="capillary")
    ax2.plot(times_ms, drag_uN, color="tab:orange", linewidth=1.6, label="drag")
    ax2.plot(times_ms, total_uN, color="tab:purple", linewidth=1.3, label="total")
    if bridge_time_ms is not None:
        ax2.axvline(bridge_time_ms, color="tab:red", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("Force magnitude [uN]")
    ax2.set_title(f"{config.title}: particle force from fluid bridge")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    fig2.tight_layout()
    path2 = fig_dir / "forces.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    paths.append(path2)

    fig3, ax3 = plt.subplots(figsize=(8, 4.8))
    ax3.plot(times_ms, rim_gap_um, color="tab:blue", linewidth=1.8, label="mean rim gap")
    ax3.plot(times_ms, min_cross_um, color="tab:brown", linewidth=1.6, label="min cross-patch distance")
    ax3.set_xlabel("Time [ms]")
    ax3.set_ylabel("Distance [um]")
    ax3.set_title(f"{config.title}: bridge geometry metrics")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best")
    ax3b = ax3.twinx()
    ax3b.plot(times_ms, free_vertices, color="tab:red", linestyle="--", linewidth=1.4, label="free vertices")
    ax3b.set_ylabel("Free vertices [-]")
    fig3.tight_layout()
    path3 = fig_dir / "bridge_metrics.png"
    fig3.savefig(path3, dpi=150)
    plt.close(fig3)
    paths.append(path3)

    fig4, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(times_ms, sep_um, color="tab:blue", linewidth=1.6)
    axes[0].set_ylabel("Gap [um]")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(config.title)

    axes[1].plot(times_ms, cap_uN, color="tab:green", linewidth=1.6, label="capillary")
    axes[1].plot(times_ms, drag_uN, color="tab:orange", linewidth=1.4, label="drag")
    axes[1].plot(times_ms, total_uN, color="tab:purple", linewidth=1.2, label="total")
    axes[1].set_ylabel("Force [uN]")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times_ms, rim_gap_um, color="tab:blue", linewidth=1.6, label="mean rim gap")
    axes[2].plot(times_ms, min_cross_um, color="tab:brown", linewidth=1.4, label="min cross-patch")
    axes[2].set_ylabel("Distance [um]")
    axes[2].set_xlabel("Time [ms]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    if bridge_time_ms is not None:
        for ax in axes:
            ax.axvline(bridge_time_ms, color="tab:red", linestyle="--", alpha=0.45)

    fig4.tight_layout()
    path4 = fig_dir / "summary.png"
    fig4.savefig(path4, dpi=150)
    plt.close(fig4)
    paths.append(path4)

    return paths


def run_case(
    config: CoupledCaseConfig,
    *,
    out_dir: Path | None = None,
    save_fig: bool = True,
    save_results: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Run one coupled DEM-fluid bridge case."""

    if out_dir is None:
        out_dir = OUT_ROOT / config.name

    if verbose:
        _print_case_header(config)

    state = setup_case(config)
    history: list[dict] = []
    mesh_snapshots = {
        "initial": _capture_mesh_snapshot(state, "initial", step=0),
    }

    zero_force_bundle = {
        "drag": {p.id: np.zeros(3, dtype=float) for p in state.ps.particles},
        "capillary": {p.id: np.zeros(3, dtype=float) for p in state.ps.particles},
        "total": {p.id: np.zeros(3, dtype=float) for p in state.ps.particles},
    }

    for step in range(config.n_steps):
        t = step * config.dt
        force_bundle = coupled_step(state, step)
        min_rim_gap, mean_rim_gap = _rim_gap_metrics(state)

        completed_step = step + 1

        if state.bridge_formed and "bridge" not in mesh_snapshots and state.bridge_step == step:
            mesh_snapshots["bridge"] = _capture_mesh_snapshot(
                state,
                f"bridge onset (step {completed_step})",
                step=completed_step,
            )

        if (
            config.mesh_snapshot_every > 0
            and completed_step % config.mesh_snapshot_every == 0
            and completed_step < config.n_steps
        ):
            mesh_snapshots[f"iter_{completed_step:04d}"] = _capture_mesh_snapshot(
                state,
                f"iteration {completed_step}",
                step=completed_step,
            )

        if step % config.record_every == 0:
            history.append(
                _step_record(
                    state,
                    step=step,
                    t=t,
                    force_bundle=force_bundle if state.bridge_formed else zero_force_bundle,
                    min_rim_gap=min_rim_gap,
                    mean_rim_gap=mean_rim_gap,
                )
            )

    t_final = config.total_time
    min_rim_gap, mean_rim_gap = _rim_gap_metrics(state)
    final_force_bundle = _fluid_force_bundle(state)
    history.append(
        _step_record(
            state,
            step=config.n_steps,
            t=t_final,
            force_bundle=final_force_bundle if state.bridge_formed else zero_force_bundle,
            min_rim_gap=min_rim_gap,
            mean_rim_gap=mean_rim_gap,
        )
    )
    mesh_snapshots["final"] = _capture_mesh_snapshot(state, "final", step=config.n_steps)

    if save_results:
        _save_history(out_dir, config, history)

    if save_fig:
        _render_figures(out_dir, config, history)
        _render_mesh_pngs(out_dir, config, mesh_snapshots)

    if verbose:
        _print_summary(history, config)

    return history
