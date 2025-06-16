import numpy as np
import meshio
import matplotlib.pyplot as plt
import subprocess
import os
import csv
from numba import njit, prange

def generate_sphere_geo(filename="sphere.geo", lc=0.6):
    geo_str = f"""
SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthMin = {lc};
Mesh.CharacteristicLengthMax = {lc};
Sphere(1) = {{0, 0, 0, 1}};
Mesh 3;
"""
    with open(filename, "w") as f:
        f.write(geo_str)
    print(f"Geo file '{filename}' created.")

def generate_cylinder_geo(filename="cylinder.geo", radius=1.0, height=2.0, lc=0.6):
    geo_str = f"""
SetFactory("OpenCASCADE");
Mesh.CharacteristicLengthMin = {lc};
Mesh.CharacteristicLengthMax = {lc};
Cylinder(1) = {{0, 0, -{height/2}, 0, 0, {height}, {radius}, 2*Pi}};
Mesh 3;
"""
    with open(filename, "w") as f:
        f.write(geo_str)
    print(f"Geo file '{filename}' created.")

def run_gmsh(geo_file, msh_file):
    print(f"Running gmsh to generate mesh '{msh_file}' ...")
    try:
        subprocess.run(["gmsh", geo_file, "-3", "-format", "msh2", "-o", msh_file], check=True)
        print("Mesh generation completed.")
    except subprocess.CalledProcessError as e:
        print(f"Gmsh failed: {e}")
        exit(1)

@njit(parallel=True, fastmath=True)
def DDG_curvature_numba(vertices, faces):
    n = vertices.shape[0]
    L = np.zeros((n, n))
    A = np.zeros(n)
    for f in prange(faces.shape[0]):
        tri = faces[f]
        idx = [tri[0], tri[1], tri[2]]
        pts = [vertices[idx[0]], vertices[idx[1]], vertices[idx[2]]]
        for i in range(3):
            i0, i1, i2 = idx[i], idx[(i+1)%3], idx[(i+2)%3]
            v0, v1, v2 = pts[i], pts[(i+1)%3], pts[(i+2)%3]
            u = v1 - v0
            v = v2 - v0
            cross = np.cross(u, v)
            norm_cross = np.linalg.norm(cross)
            cot_angle = np.dot(u, v) / norm_cross if norm_cross > 1e-12 else 0.0
            L[i0, i1] += cot_angle / 2
            L[i1, i0] += cot_angle / 2
        face_area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        for i in idx:
            A[i] += face_area / 3
    H = np.zeros(n)
    for i in prange(n):
        lap = np.zeros(3)
        for j in range(n):
            lap += L[i, j] * (vertices[j] - vertices[i])
        if A[i] > 1e-12:
            H[i] = 0.5 * np.linalg.norm(lap) / A[i]
        else:
            H[i] = 0.0
    return H

def surface_vertex_neighbors(surface_tris, num_vertices):
    neighbors = [[] for _ in range(num_vertices)]
    for tri in surface_tris:
        for i in range(3):
            a = tri[i]
            b = tri[(i+1)%3]
            c = tri[(i+2)%3]
            if b not in neighbors[a]: neighbors[a].append(b)
            if c not in neighbors[a]: neighbors[a].append(c)
    return neighbors

def neighbor_averaged_curvature(vertex_curvature, surface_neighbors):
    n = len(vertex_curvature)
    avg_curvature = np.zeros(n)
    for i in range(n):
        nb = surface_neighbors[i] + [i]
        avg_curvature[i] = np.mean([vertex_curvature[j] for j in nb])
    return avg_curvature

@njit(parallel=True, fastmath=True)
def tet_volumes(points, tets):
    total_vol_flat = 0.0
    for i in prange(tets.shape[0]):
        a = points[tets[i,0]]
        b = points[tets[i,1]]
        c = points[tets[i,2]]
        d = points[tets[i,3]]
        vol = abs(np.dot((a-d), np.cross(b-d, c-d))) / 6
        total_vol_flat += vol
    return total_vol_flat

def triangle_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b-a, c-a))

def tet_volume(a, b, c, d):
    return abs(np.dot((a-d), np.cross(b-d, c-d))) / 6

def apex_point(a, b, c, kA, kB, kC):
    bary = (a + b + c) / 3
    n = np.cross(b - a, c - a)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return bary
    n = n / n_norm
    epsilon_k = 1e-8
    RA = 1 / kA if abs(kA) > epsilon_k else 1e10
    RB = 1 / kB if abs(kB) > epsilon_k else 1e10
    RC = 1 / kC if abs(kC) > epsilon_k else 1e10
    rA = np.linalg.norm(a - bary)
    rB = np.linalg.norm(b - bary)
    rC = np.linalg.norm(c - bary)
    def sag(R, r):
        val = R**2 - r**2
        return R - np.sqrt(val) if val >= 0 and R > 0 else 0.0
    sA = sag(RA, rA)
    sB = sag(RB, rB)
    sC = sag(RC, rC)
    s_avg = (sA + sB + sC) / 3.0
    k_face_avg = (kA + kB + kC) / 3.0
    apex = bary - np.sign(k_face_avg) * s_avg * n if abs(k_face_avg) > epsilon_k else bary
    return apex

def signed_tet_volume(a, b, c, d):
    return np.dot(np.cross(b - a, c - a), d - a) / 6.0

def curved_neighbor_volume(points, surface_tris, avg_vertex_curvature):
    face_apex_map = {}
    for tri in surface_tris:
        a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
        kA, kB, kC = avg_vertex_curvature[tri[0]], avg_vertex_curvature[tri[1]], avg_vertex_curvature[tri[2]]
        key = tuple(sorted(tri))
        face_apex_map[key] = apex_point(a, b, c, kA, kB, kC)

    edge_to_faces = {}
    for tri in surface_tris:
        for i in range(3):
            v0, v1 = tri[i], tri[(i+1)%3]
            e = tuple(sorted((v0, v1)))
            if e not in edge_to_faces:
                edge_to_faces[e] = []
            edge_to_faces[e].append(tuple(sorted(tri)))

    wedge_vol_sum = 0.0
    counted = set()
    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:
            f1, f2 = faces
            if (f1, f2) in counted or (f2, f1) in counted:
                continue
            counted.add((f1, f2))
            apex1 = face_apex_map.get(f1)
            apex2 = face_apex_map.get(f2)
            if apex1 is None or apex2 is None:
                continue
            a_edge, b_edge = points[edge[0]], points[edge[1]]
            wedge_vol = abs(signed_tet_volume(a_edge, b_edge, apex1, apex2))
            wedge_vol_sum += wedge_vol / 2.0
    return wedge_vol_sum

def curvature_corrected_volume(a, b, c, d, kA, kB, kC):
    bary = (a + b + c) / 3
    rA = np.linalg.norm(a - bary)
    rB = np.linalg.norm(b - bary)
    rC = np.linalg.norm(c - bary)
    RA = 1 / kA if abs(kA) > 1e-8 else 1e10
    RB = 1 / kB if abs(kB) > 1e-8 else 1e10
    RC = 1 / kC if abs(kC) > 1e-8 else 1e10
    def sag(R, r):
        val = R**2 - r**2
        return R - np.sqrt(val) if val >= 0 else 0.0
    sA = sag(RA, rA)
    sB = sag(RB, rB)
    sC = sag(RC, rC)
    s_avg = (sA + sB + sC) / 3
    A_flat = triangle_area(a, b, c)
    V_flat = tet_volume(a, b, c, d)
    V_corr = V_flat + (1/3) * A_flat * s_avg
    return V_corr, V_flat

def sorted_tuple(a, b, c):
    return tuple(sorted((a, b, c)))

def classify_tets(points, cells, surface_tris):
    surface_faces = set(sorted_tuple(*tri) for tri in surface_tris)
    tets_inside = []
    tets_surface = []
    tet_surface_faces = []
    for tet in cells:
        faces = [
            sorted_tuple(tet[0], tet[1], tet[2]),
            sorted_tuple(tet[0], tet[1], tet[3]),
            sorted_tuple(tet[0], tet[2], tet[3]),
            sorted_tuple(tet[1], tet[2], tet[3])
        ]
        found = False
        which = -1
        for i, face in enumerate(faces):
            if face in surface_faces:
                found = True
                which = i
                break
        if found:
            tets_surface.append(tet)
            tet_surface_faces.append(which)
        else:
            tets_inside.append(tet)
    return np.array(tets_inside), np.array(tets_surface), np.array(tet_surface_faces)

def polygon_area_3d(pts, normal):
    # Project polygon to plane and compute area using the shoelace formula
    # normal: assumed unit normal of polygon plane
    # pts: Nx3 array of polygon vertex coordinates
    axes = np.abs(normal)
    drop_axis = np.argmin(axes)
    idx = [0, 1, 2]
    idx.remove(drop_axis)
    pts_2d = pts[:, idx]
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]
    area_2d = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area_2d

def dual_cell_sagitta_patch(points, surface_tris, vertex_curvature):
    from collections import defaultdict
    n = points.shape[0]
    vertex_tris = defaultdict(list)
    for t, tri in enumerate(surface_tris):
        for v in tri:
            vertex_tris[v].append(t)
    circumcenters = []
    for tri in surface_tris:
        a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
        ab = b - a
        ac = c - a
        ab_cross_ac = np.cross(ab, ac)
        norm_sq = np.dot(ab_cross_ac, ab_cross_ac)
        if norm_sq < 1e-16:
            circumcenters.append((a + b + c) / 3)
            continue
        d = 2 * norm_sq
        alpha = np.dot(b - a, b - a) * np.dot(c - a, c - a)
        beta = np.dot(a - b, a - b) * np.dot(c - b, c - b)
        gamma = np.dot(a - c, a - c) * np.dot(b - c, b - c)
        cc = (a * (beta + gamma) + b * (alpha + gamma) + c * (alpha + beta)) / (2 * (alpha + beta + gamma))
        circumcenters.append(cc)
    circumcenters = np.array(circumcenters)

    total_V_dual_sagitta = 0.0
    for vi in range(n):
        tris = vertex_tris[vi]
        if len(tris) < 3:
            continue
        cc_pts = circumcenters[tris]
        # Get average normal
        normals = []
        for tidx in tris:
            tri = surface_tris[tidx]
            a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
            nrm = np.cross(b - a, c - a)
            norm_len = np.linalg.norm(nrm)
            if norm_len > 1e-12:
                normals.append(nrm / norm_len)
        if len(normals) == 0:
            continue
        avg_normal = np.mean(normals, axis=0)
        avg_normal /= np.linalg.norm(avg_normal)
        v = points[vi]
        # Order circumcenters counterclockwise for polygon area
        ref = cc_pts[0] - v
        x_axis = ref / (np.linalg.norm(ref) + 1e-16)
        y_axis = np.cross(avg_normal, x_axis)
        angles = []
        for cc in cc_pts:
            rel = cc - v
            x = np.dot(rel, x_axis)
            y = np.dot(rel, y_axis)
            angle = np.arctan2(y, x)
            angles.append(angle)
        idx_sort = np.argsort(angles)
        cc_pts_sorted = cc_pts[idx_sort]
        dual_area = polygon_area_3d(cc_pts_sorted, avg_normal)
        H = vertex_curvature[vi]
        eps = 1e-10
        if abs(H) < eps:
            continue
        R = 1.0 / H
        radii = np.linalg.norm(cc_pts_sorted - v, axis=1)
        r = np.mean(radii)
        val = R**2 - r**2
        sagitta = R - np.sqrt(val) if val >= 0 and R > 0 else 0.0
        V_sag = (1.0 / 3.0) * dual_area * sagitta
        total_V_dual_sagitta += V_sag
    return total_V_dual_sagitta

def detect_by_integral(int_HdA, int_KdA, points, K, verbose=False):
    mesh_x, mesh_y, mesh_z = points[:,0], points[:,1], points[:,2]
    axes = np.array([np.ptp(mesh_x), np.ptp(mesh_y), np.ptp(mesh_z)])
    axes_sorted = np.sort(axes)
    PI = np.pi
    SPHERE_K = 4 * PI

    aspect1 = axes_sorted[-1] / (axes_sorted[0] + 1e-12)
    K_pos = np.sum(K > 0.01) / len(K)
    K_zero = np.sum(np.abs(K) < 1e-3) / len(K)
    if verbose:
        print(f"    int_KdA={int_KdA:.3f}, int_HdA={int_HdA:.3f}, axes={axes}, aspect1={aspect1:.3f}, K_pos={K_pos:.3f}, K_zero={K_zero:.3f}")

    if (abs(int_KdA - SPHERE_K) < 1.5) and (aspect1 < 1.12) and (K_pos > 0.9):
        return "Sphere"
    if (abs(int_KdA - SPHERE_K) < 1.5) and (aspect1 >= 1.12) and (K_pos > 0.8):
        return "Spheroid"
    if (abs(int_KdA - SPHERE_K) < 1.5) and (K_zero > 0.25):
        return "Cylinder"
    return "Other"

def process_shape(label, points, tets, surface_tris, theoretical):
    vertex_curvature = DDG_curvature_numba(points, surface_tris)
    surface_neighbors = surface_vertex_neighbors(surface_tris, len(vertex_curvature))
    avg_vertex_curvature = neighbor_averaged_curvature(vertex_curvature, surface_neighbors)

    n = len(vertex_curvature)
    angle_sum = np.zeros(n)
    area_sum = np.zeros(n)
    for tri in surface_tris:
        a, b, c = tri
        va, vb, vc = points[a], points[b], points[c]
        ab = vb - va
        ac = vc - va
        bc = vc - vb
        ba = va - vb
        cb = vb - vc
        ca = va - vc
        norm_ab = np.linalg.norm(ab)
        norm_ac = np.linalg.norm(ac)
        norm_bc = np.linalg.norm(bc)
        angle_a = np.arccos(np.clip(np.dot(ab, ac) / (norm_ab * norm_ac), -1, 1))
        angle_b = np.arccos(np.clip(np.dot(bc, ba) / (norm_bc * np.linalg.norm(ba)), -1, 1))
        angle_c = np.arccos(np.clip(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb)), -1, 1))
        angle_sum[a] += angle_a
        angle_sum[b] += angle_b
        angle_sum[c] += angle_c
        area = triangle_area(va, vb, vc)
        area_sum[a] += area / 3
        area_sum[b] += area / 3
        area_sum[c] += area / 3
    gaussian_curvature = np.zeros(n)
    for i in range(n):
        area = area_sum[i] if area_sum[i] > 1e-12 else 1.0
        gaussian_curvature[i] = (2 * np.pi - angle_sum[i]) / area

    H = vertex_curvature
    K = gaussian_curvature
    sqrt_term = np.sqrt(np.maximum(0, H**2 - K))
    k1 = H + sqrt_term
    k2 = H - sqrt_term

    def detect_geometry_from_curvature(H, K, k1, k2):
        k1 = np.nan_to_num(k1)
        k2 = np.nan_to_num(k2)
        k_all = np.hstack([k1, k2])
        k_mean = np.mean(np.abs(k_all))
        k_std = np.std(k_all)
        rel_std = k_std / (np.abs(k_mean) + 1e-12)
        frac_both_pos = np.mean((k1 > 0) & (k2 > 0))
        frac_one_zero = np.mean((np.abs(k1) < 1e-4) | (np.abs(k2) < 1e-4))
        if frac_both_pos > 0.85 and rel_std < 0.12:
            return "Sphere"
        if frac_both_pos > 0.85 and rel_std >= 0.12:
            return "Spheroid"
        if frac_one_zero > 0.7 and frac_both_pos < 0.7:
            return "Cylinder"
        if np.mean(K < -1e-4) > 0.3:
            return "Saddle"
        return "Mixed"
    geometry_detected = detect_geometry_from_curvature(H, K, k1, k2)

    int_HdA = np.sum(H * area_sum)
    int_KdA = np.sum(K * area_sum)
    geometry_integral_detected = detect_by_integral(int_HdA, int_KdA, points, K, verbose=False)

    tets_inside, tets_surface, tet_surface_faces = classify_tets(points, tets, surface_tris)
    total_vol_piecewise_linear = tet_volumes(points, tets)
    total_vol_vertex_paraboloid = 0.0
    total_vol_neighbor_averaged = 0.0

    for tet in tets_inside:
        a, b, c, d = points[tet[0]], points[tet[1]], points[tet[2]], points[tet[3]]
        total_vol_vertex_paraboloid += tet_volume(a, b, c, d)
        total_vol_neighbor_averaged += tet_volume(a, b, c, d)

    for tet, face_id in zip(tets_surface, tet_surface_faces):
        idxs = [tet[0], tet[1], tet[2], tet[3]]
        face_tuples = [
            [0, 1, 2, 3],
            [0, 1, 3, 2],
            [0, 2, 3, 1],
            [1, 2, 3, 0]
        ]
        ids = face_tuples[face_id]
        a, b, c, d = points[idxs[ids[0]]], points[idxs[ids[1]]], points[idxs[ids[2]]], points[idxs[ids[3]]]
        kA = vertex_curvature[idxs[ids[0]]]
        kB = vertex_curvature[idxs[ids[1]]]
        kC = vertex_curvature[idxs[ids[2]]]
        kA_ave = avg_vertex_curvature[idxs[ids[0]]]
        kB_ave = avg_vertex_curvature[idxs[ids[1]]]
        kC_ave = avg_vertex_curvature[idxs[ids[2]]]
        V_corr, _ = curvature_corrected_volume(a, b, c, d, kA, kB, kC)
        total_vol_vertex_paraboloid += V_corr
        V_corr_ave, _ = curvature_corrected_volume(a, b, c, d, kA_ave, kB_ave, kC_ave)
        total_vol_neighbor_averaged += V_corr_ave

    wedge_vol_sum = curved_neighbor_volume(points, surface_tris, avg_vertex_curvature)
    total_vol_curved_neighbor_patch = total_vol_neighbor_averaged + wedge_vol_sum

    # Dual-Cell Sagitta Patch
    total_vol_dual_cell_sagitta_patch = total_vol_piecewise_linear + dual_cell_sagitta_patch(points, surface_tris, vertex_curvature)
    relerr_dual_cell_sagitta_patch = (total_vol_dual_cell_sagitta_patch - theoretical) / theoretical * 100

    relerr_piecewise_linear = (total_vol_piecewise_linear - theoretical) / theoretical * 100
    relerr_vertex_paraboloid = (total_vol_vertex_paraboloid - theoretical) / theoretical * 100
    relerr_neighbor_averaged = (total_vol_neighbor_averaged - theoretical) / theoretical * 100
    relerr_curved_neighbor_patch = (total_vol_curved_neighbor_patch - theoretical) / theoretical * 100

    print(f"{label}: relerr_piecewise_linear={relerr_piecewise_linear:.5f}, relerr_vertex_paraboloid={relerr_vertex_paraboloid:.5f}, relerr_neighbor_averaged={relerr_neighbor_averaged:.5f}, relerr_curved_neighbor_patch={relerr_curved_neighbor_patch:.5f}, relerr_dual_cell_sagitta_patch={relerr_dual_cell_sagitta_patch:.5f}")
    return (len(tets), relerr_piecewise_linear, total_vol_piecewise_linear,
            relerr_vertex_paraboloid, total_vol_vertex_paraboloid,
            relerr_neighbor_averaged, total_vol_neighbor_averaged,
            relerr_curved_neighbor_patch, total_vol_curved_neighbor_patch,
            relerr_dual_cell_sagitta_patch, total_vol_dual_cell_sagitta_patch,
            geometry_detected, geometry_integral_detected, theoretical, np.sum(H * area_sum), np.sum(K * area_sum))

if __name__ == '__main__':
    lc_values = np.linspace(0.1, 0.8, 8)
    shapes = [
        dict(name="Cylinder", ax=1.0, ay=1.0, az=1.0, height=2.0, theoretical=np.pi*1.0**2*2.0),
        dict(name="Sphere", ax=1.0, ay=1.0, az=1.0, theoretical=4/3*np.pi*1.0**3),
        dict(name="Spheroid", ax=1.5, ay=1.0, az=0.8, theoretical=4/3*np.pi*1.5*1.0*0.8)
    ]
    shape_colors = {"Cylinder": "C0", "Sphere": "C1", "Spheroid": "C2"}
    all_results = []

    for shape in shapes:
        for lc in lc_values:
            geo_file = f"{shape['name'].lower()}_lc{lc:.2f}.geo"
            msh_file = f"{shape['name'].lower()}_lc{lc:.2f}.msh"
            if os.path.isfile(geo_file): os.remove(geo_file)
            if os.path.isfile(msh_file): os.remove(msh_file)
            if shape["name"].lower() == "cylinder":
                generate_cylinder_geo(geo_file, radius=shape["ax"], height=shape["height"], lc=lc)
            else:
                generate_sphere_geo(geo_file, lc)
            run_gmsh(geo_file, msh_file)
            mesh = meshio.read(msh_file)
            points = mesh.points.copy()
            if shape["name"].lower() == "spheroid":
                points[:,0] *= shape['ax']
                points[:,1] *= shape['ay']
                points[:,2] *= shape['az']
            tets = mesh.cells_dict["tetra"]
            surface_tris = mesh.cells_dict["triangle"]
            (count, err_piecewise_linear, vol_piecewise_linear,
             err_vertex_paraboloid, vol_vertex_paraboloid,
             err_neighbor_averaged, vol_neighbor_averaged,
             err_curved_neighbor_patch, vol_curved_neighbor_patch,
             err_dual_cell_sagitta_patch, vol_dual_cell_sagitta_patch,
             geometry_detected, geometry_integral_detected, theoretical_vol, int_HdA, int_KdA) = process_shape(
                shape["name"], points, tets, surface_tris, shape["theoretical"])
            all_results.append(dict(
                shape=shape["name"],
                geometry_detected=geometry_detected,
                geometry_integral_detected=geometry_integral_detected,
                TetCount=count,
                Theory_Volume=theoretical_vol,
                RelErr_PiecewiseLinear=err_piecewise_linear,
                Volume_PiecewiseLinear=vol_piecewise_linear,
                RelErr_2ndOrderVertexBasedParaboloidPatch=err_vertex_paraboloid,
                Volume_2ndOrderVertexBasedParaboloidPatch=vol_vertex_paraboloid,
                RelErr_2ndOrderNeighborAveragedParaboloidPatch=err_neighbor_averaged,
                Volume_2ndOrderNeighborAveragedParaboloidPatch=vol_neighbor_averaged,
                RelErr_2ndOrderCurvedNeighborPatch=err_curved_neighbor_patch,
                Volume_2ndOrderCurvedNeighborPatch=vol_curved_neighbor_patch,
                RelErr_DualCellSagittaPatch=err_dual_cell_sagitta_patch,    # <--- HERE
                Volume_DualCellSagittaPatch=vol_dual_cell_sagitta_patch,    # <--- HERE
                int_HdA=int_HdA,
                int_KdA=int_KdA
            ))


    # --- Plotting: each shape one color, methods in legend ---
    plt.figure(figsize=(11, 8))
    markers = {
        "Piecewise Linear": "s",
        "2nd-Order Vertex-Based Paraboloid Patch": "o",
        "2nd-Order Neighbor-Averaged Paraboloid Patch": "^",
        "2nd-Order Curved Neighbor Patch": "d",
        "Dual-Cell Sagitta Patch": "*"
    }
    for shape in shapes:
        color = shape_colors[shape["name"]]
        tets, relerr_piecewise_linear, relerr_vertex_paraboloid, relerr_neighbor_averaged, relerr_curved_neighbor_patch, relerr_dual_cell_sagitta_patch = [], [], [], [], [], []
        for row in all_results:
            if row["shape"] == shape["name"]:
                tets.append(row["TetCount"])
                relerr_piecewise_linear.append(abs(row["RelErr_PiecewiseLinear"]))
                relerr_vertex_paraboloid.append(abs(row["RelErr_2ndOrderVertexBasedParaboloidPatch"]))
                relerr_neighbor_averaged.append(abs(row["RelErr_2ndOrderNeighborAveragedParaboloidPatch"]))
                relerr_curved_neighbor_patch.append(abs(row["RelErr_2ndOrderCurvedNeighborPatch"]))
                relerr_dual_cell_sagitta_patch.append(abs(row["RelErr_DualCellSagittaPatch"]))
        plt.scatter(
            tets, relerr_piecewise_linear,
            marker=markers["Piecewise Linear"], edgecolors=color, facecolors='none', s=65, lw=2,
            label=f"{shape['name']} Piecewise Linear"
        )
        plt.scatter(
            tets, relerr_vertex_paraboloid,
            marker=markers["2nd-Order Vertex-Based Paraboloid Patch"], color=color, s=62,
            label=f"{shape['name']} 2nd-Order Vertex-Based Paraboloid Patch"
        )
        plt.scatter(
            tets, relerr_neighbor_averaged,
            marker=markers["2nd-Order Neighbor-Averaged Paraboloid Patch"], color=color, alpha=0.7, s=62,
            label=f"{shape['name']} 2nd-Order Neighbor-Averaged Paraboloid Patch"
        )
        plt.scatter(
            tets, relerr_curved_neighbor_patch,
            marker=markers["2nd-Order Curved Neighbor Patch"], color=color, alpha=0.7, s=62,
            label=f"{shape['name']} 2nd-Order Curved Neighbor Patch"
        )
        plt.scatter(
            tets, relerr_dual_cell_sagitta_patch,
            marker=markers["Dual-Cell Sagitta Patch"], color=color, alpha=0.7, s=90,
            label=f"{shape['name']} Dual-Cell Sagitta Patch"
        )
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               loc='center left', bbox_to_anchor=(1.04, 0.5),
               fontsize=9, title_fontsize=10, borderaxespad=0)
    plt.subplots_adjust(right=0.76)  # Adjust to make room for legend
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Tet Elements (log scale)')
    plt.ylabel('Relative Error (%) (log scale)')
    plt.title('Volume Approximation Error vs. Tet Count')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

    # --- Write to CSV ---
    with open("volume_shape_comparison.csv", "w", newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shape", "geometry_detected", "geometry_integral_detected", "TetCount", "Theory_Volume",
                "RelErr_PiecewiseLinear",        "Volume_PiecewiseLinear",
                "RelErr_2ndOrderVertexBasedParaboloidPatch",    "Volume_2ndOrderVertexBasedParaboloidPatch",
                "RelErr_2ndOrderNeighborAveragedParaboloidPatch", "Volume_2ndOrderNeighborAveragedParaboloidPatch",
                "RelErr_2ndOrderCurvedNeighborPatch", "Volume_2ndOrderCurvedNeighborPatch",
                "RelErr_DualCellSagittaPatch", "Volume_DualCellSagittaPatch",
                "int_HdA", "int_KdA"
            ]
        )
        writer.writeheader()
        for row in all_results:
            rounded_row = {}
            for k, v in row.items():
                if isinstance(v, float):
                    rounded_row[k] = f"{v:.5f}"
                else:
                    rounded_row[k] = v
            writer.writerow(rounded_row)
    print("Results saved to volume_shape_comparison.csv")

    # --- Cleanup geo and msh files ---
    for shape in shapes:
        for lc in lc_values:
            geo_file = f"{shape['name'].lower()}_lc{lc:.2f}.geo"
            msh_file = f"{shape['name'].lower()}_lc{lc:.2f}.msh"
            if os.path.isfile(geo_file):
                os.remove(geo_file)
            if os.path.isfile(msh_file):
                os.remove(msh_file)
