"""Output/experiment runner for Case 2b Gmsh Pitois-2000 separation.

This file owns plotting, .msh/CSV/JSON output, Fig. 5 digitized experiment
data, CLI printing, and interactive viewing.  Computation is imported from
``solver_v6.py``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import warnings

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

from solver_v6 import *

OUT_ROOT = Path(__file__).resolve().parent / Path(__file__).stem

# Output and Fig. 5 comparison controls live here, not in the solver.
USER_FIG5_COMPARE_SKIP_INITIAL_STEPS = 1
USER_X_AXIS_MIN_MM = -1
USER_X_AXIS_MAX_MM = 1
USER_Z_AXIS_MIN_MM = -1.7
USER_Z_AXIS_MAX_MM = 1.7
USER_SHOW_MESH_FACES = False
USER_SHOW_MESH_VERTICES = True
USER_SHOW_REAL_COMPUTE_TRIANGLES = True
USER_SHOW_SIDE_TRIANGLE_DIAGONALS = True
USER_REAL_TRIANGLE_VIEW = "full"
USER_SHOW_CONTACT_RING_OVERLAY = False
USER_SHOW_SURFACE_OVERLAY = False
USER_SURFACE_OVERLAY_ALPHA = 0.58
USER_MESH_VERTEX_SIZE = 5.0
USER_MESH_ALPHA = 0.99
USER_CAP_EDGE_ALPHA = 0.0
USER_FILL_PARTICLE_CAP_SURFACES = False
USER_OPEN_INTERACTIVE_WINDOW = True
USER_INTERACTIVE_STEP = USER_TOTAL_STEPS
USER_INTERACTIVE_ELEV_DEG = 25.0
USER_INTERACTIVE_AZIM_DEG = 45.0

# Digitized first-point Fig. 5 increment:
#   dF = F_black - F_white.
# The black value is the first digitized dynamic point below; the white value is
# read from the first open-circle static point in the same local Fig. 5 image.
PITOIS_FIG5_FIRST_BLACK_MN = 1.515311539463195
PITOIS_FIG5_FIRST_WHITE_MN = 0.38
PITOIS_FIG5_FIRST_DF_BLACKWHITE_MN = PITOIS_FIG5_FIRST_BLACK_MN - PITOIS_FIG5_FIRST_WHITE_MN
PITOIS_FIG5_FIRST_DF_BLACKWHITE_SIGNED_MN = -PITOIS_FIG5_FIRST_DF_BLACKWHITE_MN

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


def _save_mesh_snapshot_msh(
    state: VolumetricPitoisState,
    out_path: Path,
    *,
    side_triangles: np.ndarray,
    bottom_cap_triangles: np.ndarray,
    top_cap_triangles: np.ndarray,
) -> Path:
    del side_triangles, bottom_cap_triangles, top_cap_triangles
    points, tets, node_ids = _snapshot_msh_indexed_mesh(state)
    msh_path = out_path.with_suffix(".msh")
    msh_path.parent.mkdir(parents=True, exist_ok=True)
    _write_gmsh2(points, tets, msh_path, element_type=4, node_ids=node_ids)
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

    out_dir.mkdir(parents=True, exist_ok=True)

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
    ax1.set_ylabel("Simulated axial force [mN]")
    ax1.set_title("Separation force vs gap")
    ax1.grid(alpha=0.28)
    path1 = out_dir / "separation_force_vs_gap.png"
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
    path2 = out_dir / "separation_gap_speed_vs_time.png"
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
    path3 = out_dir / "separation_snapshot_volume_rel_error_vs_time.png"
    fig3.tight_layout()
    fig3.savefig(path3, dpi=180, bbox_inches="tight")
    plt.close(fig3)
    paths.append(path3)

    return paths


def _render_motion_mesh_pngs(
    state: VolumetricPitoisState,
    *,
    out_dir: Path,
    label: str,
    step: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if label == "initial":
        filename = "mesh_initial.png"
    elif label == "final":
        filename = "mesh_final.png"
    else:
        filename = f"mesh_iter{step:04d}.png"
    title = _snapshot_title(state, label=label if label in {"initial", "final"} else f"iteration {step}")
    out_png = _render_mesh_snapshot(state, title, out_dir / filename)
    if label == "initial":
        iter0_png = out_dir / "mesh_iter0000.png"
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
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "history": history,
    }
    path = out_dir / "separation_history.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _save_output_csv(history: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "output.csv"
    fields = [
        "step",
        "t_s",
        "dt_s",
        "d_over_r",
        "gap_m",
        "water_vol_error_rel",
        "water_vol_error_percent",
        "water_filled_mesh_volume_uL",
        "p0_neck_fit_pa",
        "p0_neck_fit_delta_p_pa",
        "p0_neck_fit_H_1pm",
        "diagnostic_lv_H_avg_1pm",
        "diagnostic_lv_H_avg_vertex_count",
        "p0_neck_fit_radius_mm",
        "p0_neck_fit_rz",
        "p0_neck_fit_rzz_1pm",
        "p0_neck_fit_vertex_count",
        "p0_neck_fit_half_window_rings",
        "p0_neck_fit_degree",
        "p0_neck_ring_index",
        "Ftopspherecap_p0_mN",
        "top_cl_avg_ux_mps",
        "top_cl_avg_uy_mps",
        "top_cl_avg_uz_mps",
        "top_cl_avg_ucl_x_mps",
        "top_cl_avg_ucl_y_mps",
        "top_cl_avg_ucl_z_mps",
        "top_cl_avg_dxyz_x_m",
        "top_cl_avg_dxyz_y_m",
        "top_cl_avg_dxyz_z_m",
        "top_cl_radius_m",
        "top_cl_radius_mm",
        "top_cl_vertex_count",
        "top_cl_avg_Fp0_mN",
        "top_cl_avg_Fp_flow_mN",
        "top_cl_avg_Ftau_mN",
        "top_cl_avg_Fhydro_mN",
        "top_cl_avg_Fp_mN",
        "top_cl_avg_Fs_mN",
        "top_cl_avg_Fv_mN",
        "top_cl_avg_Fcl_mN",
        "top_cl_avg_Fcl_static_mN",
        "top_cl_avg_Fcl_dynamic_minus_static_mN",
        "top_cl_avg_Finterface_total_mN",
        "top_cl_avg_Ftot_mN",
        "top_sphere_Fp0_mN",
        "top_sphere_Fp_flow_mN",
        "top_sphere_Fp_flow_stokes_mN",
        "top_sphere_Fp_flow_lub_mN",
        "top_sphere_Fp_flow_raw_plus_lub_mN",
        "top_sphere_Ftau_mN",
        "top_sphere_Fhydro_mN",
        "top_sphere_Fp_mN",
        "top_sphere_Fs_mN",
        "top_sphere_Fv_mN",
        "top_sphere_Fcl_mN",
        "top_sphere_Fcl_static_mN",
        "top_sphere_Fcl_dynamic_minus_static_mN",
        "top_sphere_dF_blackwhite_mN",
        "experiment_first_dF_blackwhite_mN",
        "top_sphere_Finterface_total_mN",
        "top_sphere_Ftot_mN",
        "top_sphere_force_mN",
    ]
    cache = getattr(_save_output_csv, "_write_cache", {})
    cache_key = str(path.resolve())
    previous_rows = int(cache.get(cache_key, 0))
    append_rows = path.exists() and 0 < previous_rows <= len(history)
    row_start = previous_rows if append_rows else 0
    mode = "a" if append_rows else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not append_rows:
            writer.writeheader()
        for row in history[row_start:]:
            top_sphere_force_n = float(row.get("top_sphere_force_n", row.get("top_sphere_Ftot_n", 0.0)))
            writer.writerow(
                {
                    "step": int(row.get("step", 0)),
                    "t_s": float(row.get("t", 0.0)),
                    "dt_s": float(row.get("dt", 0.0)),
                    "d_over_r": float(row.get("d_over_r", 0.0)),
                    "gap_m": float(row.get("gap", 0.0)),
                    "water_vol_error_rel": float(row.get("water_vol_error_rel", 0.0)),
                    "water_vol_error_percent": float(row.get("water_vol_error_percent", 0.0)),
                    "water_filled_mesh_volume_uL": float(row.get("water_filled_mesh_volume_uL", 0.0)),
                    "p0_neck_fit_pa": float(row.get("p0_neck_fit_pa", 0.0)),
                    "p0_neck_fit_delta_p_pa": float(row.get("p0_neck_fit_delta_p_pa", 0.0)),
                    "p0_neck_fit_H_1pm": float(row.get("p0_neck_fit_H_1pm", 0.0)),
                    "diagnostic_lv_H_avg_1pm": float(row.get("diagnostic_lv_H_avg_1pm", 0.0)),
                    "diagnostic_lv_H_avg_vertex_count": int(row.get("diagnostic_lv_H_avg_vertex_count", 0)),
                    "p0_neck_fit_radius_mm": float(row.get("p0_neck_fit_radius_mm", 0.0)),
                    "p0_neck_fit_rz": float(row.get("p0_neck_fit_rz", 0.0)),
                    "p0_neck_fit_rzz_1pm": float(row.get("p0_neck_fit_rzz_1pm", 0.0)),
                    "p0_neck_fit_vertex_count": int(row.get("p0_neck_fit_vertex_count", 0)),
                    "p0_neck_fit_half_window_rings": int(row.get("p0_neck_fit_half_window_rings", 0)),
                    "p0_neck_fit_degree": int(row.get("p0_neck_fit_degree", 0)),
                    "p0_neck_ring_index": int(row.get("p0_neck_ring_index", -1)),
                    "Ftopspherecap_p0_mN": float(row.get("Ftopspherecap_p0_mN", 0.0)),
                    "top_cl_avg_ux_mps": float(row.get("top_cl_avg_ux_mps", 0.0)),
                    "top_cl_avg_uy_mps": float(row.get("top_cl_avg_uy_mps", 0.0)),
                    "top_cl_avg_uz_mps": float(row.get("top_cl_avg_uz_mps", 0.0)),
                    "top_cl_avg_ucl_x_mps": float(row.get("top_cl_avg_ucl_x_mps", row.get("top_cl_avg_ux_mps", 0.0))),
                    "top_cl_avg_ucl_y_mps": float(row.get("top_cl_avg_ucl_y_mps", row.get("top_cl_avg_uy_mps", 0.0))),
                    "top_cl_avg_ucl_z_mps": float(row.get("top_cl_avg_ucl_z_mps", row.get("top_cl_avg_uz_mps", 0.0))),
                    "top_cl_avg_dxyz_x_m": float(row.get("top_cl_avg_dxyz_x_m", 0.0)),
                    "top_cl_avg_dxyz_y_m": float(row.get("top_cl_avg_dxyz_y_m", 0.0)),
                    "top_cl_avg_dxyz_z_m": float(row.get("top_cl_avg_dxyz_z_m", 0.0)),
                    "top_cl_radius_m": float(row.get("top_cl_radius_m", 0.0)),
                    "top_cl_radius_mm": float(row.get("top_cl_radius_mm", 0.0)),
                    "top_cl_vertex_count": int(row.get("top_cl_vertex_count", 0)),
                    "top_cl_avg_Fp0_mN": float(row.get("top_cl_avg_Fp0_mN", 0.0)),
                    "top_cl_avg_Fp_flow_mN": float(row.get("top_cl_avg_Fp_flow_mN", 0.0)),
                    "top_cl_avg_Ftau_mN": float(row.get("top_cl_avg_Ftau_mN", 0.0)),
                    "top_cl_avg_Fhydro_mN": float(row.get("top_cl_avg_Fhydro_mN", 0.0)),
                    "top_cl_avg_Fp_mN": float(row.get("top_cl_avg_Fp_mN", 0.0)),
                    "top_cl_avg_Fs_mN": float(row.get("top_cl_avg_Fs_mN", 0.0)),
                    "top_cl_avg_Fv_mN": float(row.get("top_cl_avg_Fv_mN", 0.0)),
                    "top_cl_avg_Fcl_mN": float(row.get("top_cl_avg_Fcl_mN", 0.0)),
                    "top_cl_avg_Fcl_static_mN": float(row.get("top_cl_avg_Fcl_static_mN", 0.0)),
                    "top_cl_avg_Fcl_dynamic_minus_static_mN": float(row.get("top_cl_avg_Fcl_dynamic_minus_static_mN", 0.0)),
                    "top_cl_avg_Finterface_total_mN": float(row.get("top_cl_avg_Finterface_total_mN", 0.0)),
                    "top_cl_avg_Ftot_mN": float(row.get("top_cl_avg_Ftot_mN", 0.0)),
                    "top_sphere_Fp0_mN": float(row.get("top_sphere_Fp0_mN", 0.0)),
                    "top_sphere_Fp_flow_mN": float(row.get("top_sphere_Fp_flow_mN", 0.0)),
                    "top_sphere_Fp_flow_stokes_mN": float(row.get("top_sphere_Fp_flow_stokes_mN", 0.0)),
                    "top_sphere_Fp_flow_lub_mN": float(row.get("top_sphere_Fp_flow_lub_mN", row.get("top_sphere_Fp_flow_mN", 0.0))),
                    "top_sphere_Fp_flow_raw_plus_lub_mN": float(row.get("top_sphere_Fp_flow_raw_plus_lub_mN", row.get("top_sphere_Fp_flow_mN", 0.0))),
                    "top_sphere_Ftau_mN": float(row.get("top_sphere_Ftau_mN", 0.0)),
                    "top_sphere_Fhydro_mN": float(row.get("top_sphere_Fhydro_mN", 0.0)),
                    "top_sphere_Fp_mN": float(row.get("top_sphere_Fp_mN", 0.0)),
                    "top_sphere_Fs_mN": float(row.get("top_sphere_Fs_mN", 0.0)),
                    "top_sphere_Fv_mN": float(row.get("top_sphere_Fv_mN", 0.0)),
                    "top_sphere_Fcl_mN": float(row.get("top_sphere_Fcl_mN", 0.0)),
                    "top_sphere_Fcl_static_mN": float(row.get("top_sphere_Fcl_static_mN", 0.0)),
                    "top_sphere_Fcl_dynamic_minus_static_mN": float(row.get("top_sphere_Fcl_dynamic_minus_static_mN", 0.0)),
                    "top_sphere_dF_blackwhite_mN": float(row.get("top_sphere_dF_blackwhite_mN", 0.0)),
                    "experiment_first_dF_blackwhite_mN": float(PITOIS_FIG5_FIRST_DF_BLACKWHITE_SIGNED_MN),
                    "top_sphere_Finterface_total_mN": float(row.get("top_sphere_Finterface_total_mN", 0.0)),
                    "top_sphere_Ftot_mN": float(row.get("top_sphere_Ftot_mN", 1.0e3 * top_sphere_force_n)),
                    "top_sphere_force_mN": 1.0e3 * top_sphere_force_n,
                }
            )
    cache[cache_key] = len(history)
    _save_output_csv._write_cache = cache
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
    sim_label: str = "Case_2b_axisym simulation",
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
    ax.plot(sim_x, sim_y, color="#d62828", linewidth=2.0, label=sim_label)
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
    out_dir.mkdir(parents=True, exist_ok=True)

    separation = _history_arrays(separation_history)
    exp_x, exp_y = _pitois_fig5_dynamic_digitized()
    compare_mask = np.array(
        [int(row.get("step", 0)) > int(USER_FIG5_COMPARE_SKIP_INITIAL_STEPS) for row in separation_history],
        dtype=bool,
    )
    if not np.any(compare_mask):
        compare_mask = np.ones_like(separation["d_over_r"], dtype=bool)

    _save_pitois_fig5_compare(
        exp_x=exp_x,
        exp_y=exp_y,
        sim_x=separation["d_over_r"][compare_mask],
        sim_y=separation["force_abs_mn"][compare_mask],
        out_path=out_dir / "pitois2000_volumetric_separation_digitized_compare.png",
        sim_label="Case_2b simulation",
        xlim=(0.01, 0.30),
        ylim=(0.02, 2.0),
    )
    _save_case2b_histories_panel(
        separation=separation,
        out_path=out_dir / "pitois2000_volumetric_case2b_histories.png",
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
                "The imported t=0/near-start geometry is calibrated to the Fig. 5 setup; "
                "the main comparison omits those near-start rows so the red curve shows "
                "post-advance computed dynamics only."
            ),
        },
        "digitized_fig5_dynamic": {
            "d_over_r": exp_x.tolist(),
            "force_mn": exp_y.tolist(),
        },
    }
    (out_dir / "pitois2000_volumetric_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "pitois2000_fig5_dynamic_digitized.json").write_text(
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
    print(f"Internal pressure offset  = {config.internal_pressure_pa:.3e} Pa")
    print(f"Continuity pressure seed  = {config.continuity_pressure_pa:.3e} Pa")
    print("Static capillary pressure = neck meridian-fit Young-Laplace p0")
    print("Interface surface force   = Heron curvature operator")
    print("Contact-line force        = Cox-angle capillary line force")
    print(f"Contact angle assumption  = {config.contact_angle_deg:.1f} deg (literature-informed baseline)")
    print(f"Moving-sphere speed       = {config.cap_speed * 1e6:.3f} um/s")
    print(f"Relative speed            = {config.relative_speed * 1e6:.3f} um/s")
    print(f"Lubrication speed factor  = {config.lubrication_relative_speed_factor:.3f}")
    print("Fixed sphere              = top sphere (balance side)")
    print("Moving sphere             = bottom sphere (stage side)")
    print(f"Integration substeps      = {config.integration_substeps}")
    print("Flow velocity             = sparse Stokes/continuity solve")
    print("Pressure force            = static p0 + Stokes pressure in liquid stress")
    print("Fp,proj                   = removed")
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
    print(f"Simulated force [mN]     = {float(np.min(fixed_force_mn)):.6e} - {float(np.max(fixed_force_mn)):.6e}")
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
    _update_top_sphere_force_n(state)
    target_snapshot_volume_ul = 1.0e9 * float(state.target_snapshot_volume_m3)
    history: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_fig:
        _render_motion_mesh_pngs(state, out_dir=out_dir, label="initial", step=0)

    for step in range(config.n_steps):
        step_dt = advance_one_step(state, step_index=step)

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
                top_sphere_force_mn = 1.0e3 * float(record_row["top_sphere_force_n"])
                bottom_sphere_force_mn = 1.0e3 * float(record_row["bottom_sphere_force_n"])
                print(
                    "F each step              = "
                    f"top_sphere_force_n {top_sphere_force_mn:+.6e} mN, "
                    f"bottom_sphere_force_n {bottom_sphere_force_mn:+.6e} mN, "
                    f"|top| {abs(top_sphere_force_mn):.6e} mN, "
                    f"|bottom| {abs(bottom_sphere_force_mn):.6e} mN",
                    flush=True,
                )
                print(
                    "F top breakdown          = "
                    f"Fp0 {float(record_row.get('top_sphere_Fp0_mN', 0.0)):+.6e} mN, "
                    f"Fp_flow {float(record_row.get('top_sphere_Fp_flow_mN', 0.0)):+.6e} mN, "
                    f"Fp_flow_lub {float(record_row.get('top_sphere_Fp_flow_lub_mN', 0.0)):+.6e} mN, "
                    f"Fp_flow_stokes_diag {float(record_row.get('top_sphere_Fp_flow_stokes_mN', 0.0)):+.6e} mN, "
                    f"Ftau {float(record_row.get('top_sphere_Ftau_mN', 0.0)):+.6e} mN, "
                    f"FCL {float(record_row.get('top_sphere_Fcl_mN', 0.0)):+.6e} mN, "
                    f"dFCL {float(record_row.get('top_sphere_Fcl_dynamic_minus_static_mN', 0.0)):+.6e} mN, "
                    f"dF_blackwhite {float(record_row.get('top_sphere_dF_blackwhite_mN', 0.0)):+.6e} mN, "
                    f"Ftop {float(record_row.get('top_sphere_Ftot_mN', top_sphere_force_mn)):+.6e} mN",
                    flush=True,
                )
        history.append(record_row if record_row is not None else _step_record(state, step=completed_step, t=state.elapsed_time_s))
        if save_results:
            _save_output_csv(history, out_dir)
        if save_fig and history:
            _write_pitois_fig5_comparison(
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
                out_dir=out_dir,
                label=f"iteration {completed_step}",
                step=completed_step,
            )

    if config.n_steps > 0:
        final_step = int(config.n_steps)
        if not history or int(history[-1]["step"]) != final_step:
            history.append(_step_record(state, step=final_step, t=state.elapsed_time_s))

    if save_fig:
        _render_motion_mesh_pngs(state, out_dir=out_dir, label="final", step=config.n_steps)
        _render_history_figures(
            history,
            config,
            out_dir,
            initial_snapshot_volume_m3=float(state.target_snapshot_volume_m3),
        )

    if save_results:
        _save_history(history, config, out_dir)
        _save_output_csv(history, out_dir)

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
