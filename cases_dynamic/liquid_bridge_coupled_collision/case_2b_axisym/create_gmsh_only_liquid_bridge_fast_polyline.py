#!/usr/bin/env python3
"""
Gmsh-only fast liquid-bridge mesh.

This version uses Gmsh for the whole geometry/mesh workflow, but avoids the slow
B-spline surface-of-revolution problem by using a piecewise-linear meridian
profile. Gmsh then creates the CAD geometry and tetrahedral mesh.

Workflow:
1. Build meridian geometry inside Gmsh/OpenCASCADE using points + straight lines.
2. Create a planar meridian surface.
3. Save CAD geometry before meshing.
4. Revolve the meridian surface around the z-axis.
5. Mesh the 3D liquid volume with Gmsh.
6. Refine near the liquid-air/sphere contact lines with approximate growth ratio 1.1.

This does NOT use HyperCT.
This does NOT manually write mesh elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np


@dataclass(frozen=True)
class BridgeParams:
    # Geometry.
    sphere_radius_m: float = 4.0e-3
    gap_m: float = 0.08e-3
    bridge_volume_m3: float = 1.1e-9
    contact_radius_m: float = 1.05e-3

    # Polyline geometry resolution.
    # Keep these modest for speed. Increase later after geometry is validated.
    sphere_arc_nodes: int = 8
    free_surface_nodes: int = 17

    # Mesh sizes.
    cl_progression_ratio: float = 1.1
    lc_contact_line_m: float = 1.0e-4
    lc_bulk_m: float = 4.0e-4

    # Output.
    out_dir: str = "out_gmsh_only_fast"
    name: str = "liquid_bridge_gmsh_only_fast"


def _sphere_z_bottom(r: np.ndarray, p: BridgeParams) -> np.ndarray:
    R = p.sphere_radius_m
    zc = -(R + 0.5 * p.gap_m)
    return zc + np.sqrt(np.maximum(R * R - r * r, 0.0))


def _sphere_z_top(r: np.ndarray, p: BridgeParams) -> np.ndarray:
    R = p.sphere_radius_m
    zc = +(R + 0.5 * p.gap_m)
    return zc - np.sqrt(np.maximum(R * R - r * r, 0.0))


def _free_radius_shape(z: np.ndarray, p: BridgeParams, radial_scale: float = 1.0) -> np.ndarray:
    """
    Smooth Fig. 1-like liquid-air profile, later represented by straight segments.

    This gives geometry. It is not the exact Young-Laplace elliptic solution.
    """
    r_cl = p.contact_radius_m
    z_cl = float(_sphere_z_top(np.array([r_cl]), p)[0])
    u = np.abs(z) / max(abs(z_cl), 1.0e-30)

    neck_ratio = 0.48
    smooth = 3.0 * u**2 - 2.0 * u**3
    return radial_scale * r_cl * (neck_ratio + (1.0 - neck_ratio) * smooth)


def _polygon_area_centroid_r(poly_rz: np.ndarray) -> tuple[float, float]:
    r = poly_rz[:, 0]
    z = poly_rz[:, 1]
    r2 = np.roll(r, -1)
    z2 = np.roll(z, -1)

    cross = r * z2 - r2 * z
    area_signed = 0.5 * float(np.sum(cross))
    if abs(area_signed) < 1.0e-30:
        return 0.0, 0.0

    centroid_r = float(np.sum((r + r2) * cross) / (6.0 * area_signed))
    return abs(area_signed), abs(centroid_r)


def _volume_from_meridian_polygon(poly_rz: np.ndarray) -> float:
    area, centroid_r = _polygon_area_centroid_r(poly_rz)
    return 2.0 * math.pi * area * centroid_r


def _make_meridian_polygon(p: BridgeParams, radial_scale: float) -> np.ndarray:
    r_cl = p.contact_radius_m

    # Bottom sphere-wetted curve: axis -> contact line.
    r_bottom = np.linspace(0.0, r_cl, p.sphere_arc_nodes)
    z_bottom = _sphere_z_bottom(r_bottom, p)
    z_bottom[0] = -0.5 * p.gap_m

    z_bot_cl = float(z_bottom[-1])
    z_top_cl = float(_sphere_z_top(np.array([r_cl]), p)[0])

    # Free liquid-air curve: bottom contact line -> top contact line.
    z_free = np.linspace(z_bot_cl, z_top_cl, p.free_surface_nodes)
    r_free = _free_radius_shape(z_free, p, radial_scale=radial_scale)
    r_free[0] = r_cl
    r_free[-1] = r_cl

    # Top sphere-wetted curve: contact line -> axis.
    r_top = np.linspace(r_cl, 0.0, p.sphere_arc_nodes)
    z_top = _sphere_z_top(r_top, p)
    z_top[-1] = 0.5 * p.gap_m

    poly = []
    poly.extend(zip(r_bottom, z_bottom))
    poly.extend(zip(r_free[1:], z_free[1:]))
    poly.extend(zip(r_top[1:], z_top[1:]))

    return np.asarray(poly, dtype=float)


def _solve_radial_scale_for_volume(p: BridgeParams) -> float:
    target = float(p.bridge_volume_m3)
    lo, hi = 0.3, 1.8

    def volume(scale: float) -> float:
        return _volume_from_meridian_polygon(_make_meridian_polygon(p, scale))

    for _ in range(30):
        if volume(lo) <= target <= volume(hi):
            break
        if volume(lo) > target:
            lo *= 0.75
        if volume(hi) < target:
            hi *= 1.25

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if volume(mid) < target:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def _line_loop_from_polygon(gmsh, poly_rz: np.ndarray, p: BridgeParams) -> tuple[list[int], list[int]]:
    """
    Add all meridian points and line segments to Gmsh OCC.

    Returns:
        point_tags, line_tags
    """
    point_tags: list[int] = []

    n = len(poly_rz)
    for i, (r, z) in enumerate(poly_rz):
        # smaller lc at the two contact-line positions and near the liquid-air surface
        # use bulk at the axis.
        if abs(r) < 1.0e-14:
            lc = p.lc_bulk_m
        else:
            lc = p.lc_contact_line_m if i in (p.sphere_arc_nodes - 1, p.sphere_arc_nodes + p.free_surface_nodes - 2) else p.lc_bulk_m
        tag = gmsh.model.occ.addPoint(float(r), 0.0, float(z), float(lc))
        point_tags.append(tag)

    line_tags: list[int] = []
    for i in range(n):
        j = (i + 1) % n
        line_tags.append(gmsh.model.occ.addLine(point_tags[i], point_tags[j]))

    return point_tags, line_tags


def build_gmsh_geometry_and_mesh(p: BridgeParams) -> None:
    try:
        import gmsh
    except ImportError as exc:
        raise SystemExit("Install gmsh first: pip install gmsh") from exc

    out_dir = Path(p.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    brep_path = out_dir / f"{p.name}.brep"
    step_path = out_dir / f"{p.name}.step"
    msh_path = out_dir / f"{p.name}.msh"

    radial_scale = _solve_radial_scale_for_volume(p)
    poly_rz = _make_meridian_polygon(p, radial_scale)
    estimated_volume = _volume_from_meridian_polygon(poly_rz)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(p.name)

    point_tags, line_tags = _line_loop_from_polygon(gmsh, poly_rz, p)

    # Contact-line point indices in the meridian polygon.
    bottom_cl_idx = p.sphere_arc_nodes - 1
    top_cl_idx = p.sphere_arc_nodes + p.free_surface_nodes - 2
    contact_line_points = [point_tags[bottom_cl_idx], point_tags[top_cl_idx]]

    wire = gmsh.model.occ.addWire(line_tags)
    meridian_surface = gmsh.model.occ.addPlaneSurface([wire])
    gmsh.model.occ.synchronize()

    # Revolve geometry only. The meridian boundary is polyline, so the revolved
    # surfaces are piecewise conical/frustum surfaces, much easier than B-splines.
    out = gmsh.model.occ.revolve(
        [(2, meridian_surface)],
        0.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        2.0 * math.pi,
    )
    gmsh.model.occ.synchronize()

    volume_tags = [tag for dim, tag in out if dim == 3]
    if not volume_tags:
        raise RuntimeError("Gmsh OCC revolve did not create a volume.")

    gmsh.model.addPhysicalGroup(3, volume_tags, 1)
    gmsh.model.setPhysicalName(3, 1, "liquid_volume")

    # Save geometry before meshing.
    gmsh.write(str(brep_path))
    gmsh.write(str(step_path))

    # Contact-line refinement using Gmsh size fields.
    # This approximates growth ratio 1.1 away from the contact line.
    field_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field_dist, "PointsList", contact_line_points)
    gmsh.model.mesh.field.setNumber(field_dist, "Sampling", 30)

    field_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(field_thr, "InField", field_dist)
    gmsh.model.mesh.field.setNumber(field_thr, "SizeMin", p.lc_contact_line_m)
    gmsh.model.mesh.field.setNumber(field_thr, "SizeMax", p.lc_bulk_m)
    gmsh.model.mesh.field.setNumber(field_thr, "DistMin", 2.0 * p.lc_contact_line_m)

    n_growth = 12
    dist_max = p.lc_contact_line_m * (
        p.cl_progression_ratio**n_growth - 1.0
    ) / (p.cl_progression_ratio - 1.0)
    gmsh.model.mesh.field.setNumber(field_thr, "DistMax", max(dist_max, 8.0 * p.lc_contact_line_m))
    gmsh.model.mesh.field.setAsBackgroundMesh(field_thr)

    # Speed-focused meshing. No curvature refinement, no expensive optimization.
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Algorithm", 5)       # 2D Delaunay, usually fastest/robust here
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)     # 3D Delaunay tetrahedra
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Optimize", 0)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)

    gmsh.model.mesh.generate(3)
    gmsh.write(str(msh_path))

    print("=" * 72)
    print("Gmsh-only liquid-bridge mesh complete")
    print("=" * 72)
    print(f"Geometry BREP             = {brep_path}")
    print(f"Geometry STEP             = {step_path}")
    print(f"Mesh file                 = {msh_path}")
    print(f"Sphere radius             = {p.sphere_radius_m * 1e3:.6f} mm")
    print(f"Gap D                     = {p.gap_m * 1e6:.6f} um")
    print(f"Contact radius            = {p.contact_radius_m * 1e3:.6f} mm")
    print(f"Target volume             = {p.bridge_volume_m3 * 1e9:.6f} uL")
    print(f"Meridian volume estimate  = {estimated_volume * 1e9:.6f} uL")
    print(f"Radial profile scale      = {radial_scale:.12f}")
    print(f"CL growth ratio target    = {p.cl_progression_ratio:.6f}")
    print(f"CL mesh size              = {p.lc_contact_line_m:.3e} m")
    print(f"Bulk mesh size            = {p.lc_bulk_m:.3e} m")

    gmsh.finalize()


def main() -> None:
    build_gmsh_geometry_and_mesh(BridgeParams())


if __name__ == "__main__":
    main()
