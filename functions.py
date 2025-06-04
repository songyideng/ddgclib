import numpy as np
import meshio
import matplotlib.pyplot as plt
import os
import subprocess

################################################################################
# ----------- MESH GENERATION & CURVATURE METHODS ------------------------------
################################################################################

def generate_gmsh_cylinder_mesh(filename="cylinder.msh", radius=1.0, height=2.0, lc=0.3):
    """
    Generates a tetrahedral mesh for a cylinder using GMSH.
    """
    gmsh_script = f"""
    SetFactory("OpenCASCADE");
    Cylinder(1) = {{0, 0, -{height/2}, 0, 0, {height}, {radius}, 2*Pi}};
    Mesh.CharacteristicLengthMin = {lc};
    Mesh.CharacteristicLengthMax = {lc};
    Mesh 3;
    Save("{filename}");
    """
    with open("cylinder.geo", "w") as f:
        f.write(gmsh_script)
    os.system(f"gmsh cylinder.geo -3 -format msh2 -o {filename}")

def write_gmsh_cylinder_geo(geo_filename="cylinder.geo", radius=1.0, height=2.0, lc=0.3):
    """
    Writes a .geo file for a cylinder.
    """
    gmsh_script = f"""
SetFactory("OpenCASCADE");
Cylinder(1) = {{0, 0, -{height/2}, 0, 0, {height}, {radius}, 2*Pi}};
Mesh.CharacteristicLengthMin = {lc};
Mesh.CharacteristicLengthMax = {lc};
Mesh 3;
"""
    with open(geo_filename, "w") as f:
        f.write(gmsh_script)

def generate_sphere_geo(filename="sphere.geo", lc=0.6):
    """
    Generates a .geo file for a unit sphere.
    """
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

def run_gmsh(geo_file="sphere.geo", msh_file="sphere.msh"):
    """
    Runs GMSH to generate mesh from a .geo file.
    """
    print(f"Running gmsh to generate mesh '{msh_file}' ...")
    try:
        subprocess.run(["gmsh", geo_file, "-3", "-format", "msh2", "-o", msh_file], check=True)
        print("Mesh generation completed.")
    except subprocess.CalledProcessError as e:
        print(f"Gmsh failed: {e}")
        exit(1)

################################################################################
# ----------- CURVATURE, NORMAL, AND MISC GEOMETRIC UTILS ----------------------
################################################################################

def ddg_vertex_mean_curvature(vertices, faces):
    """
    Discrete mean curvature at vertices (cotangent Laplacian method).
    """
    n = vertices.shape[0]
    L = np.zeros((n, n))
    A = np.zeros(n)
    for tri in faces:
        idx = [tri[0], tri[1], tri[2]]
        pts = [vertices[i] for i in idx]
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
    for i in range(n):
        lap = np.zeros(3)
        for j in range(n):
            lap += L[i, j] * (vertices[j] - vertices[i])
        if A[i] > 1e-12:
            H[i] = 0.5 * np.linalg.norm(lap) / A[i]
        else:
            H[i] = 0.0
    return H

def ddg_vertex_normals(vertices, faces):
    """
    Vertex normals (area-weighted average of face normals).
    """
    n = vertices.shape[0]
    normals = np.zeros((n, 3))
    for tri in faces:
        idx = [tri[0], tri[1], tri[2]]
        v0, v1, v2 = vertices[idx]
        face_normal = np.cross(v1 - v0, v2 - v0)
        face_area = 0.5 * np.linalg.norm(face_normal)
        if face_area > 1e-12:
            face_normal = face_normal / np.linalg.norm(face_normal)
        else:
            face_normal = np.zeros(3)
        for i in idx:
            normals[i] += face_normal * face_area
    norms = np.linalg.norm(normals, axis=1)
    normals[norms > 0] = normals[norms > 0] / norms[norms > 0][:, None]
    return normals

def ddg_vertex_gaussian_curvature(vertices, faces):
    """
    Discrete Gaussian curvature at vertices.
    """
    n = vertices.shape[0]
    K = np.full(n, 2 * np.pi)
    for i in range(n):
        vi = vertices[i]
        face_indices = [f for f, tri in enumerate(faces) if i in tri]
        angle_sum = 0.0
        for fidx in face_indices:
            tri = faces[fidx]
            idx = list(tri)
            i0 = idx.index(i)
            v1 = vertices[idx[(i0+1)%3]] - vi
            v2 = vertices[idx[(i0+2)%3]] - vi
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            dotp = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(dotp)
            angle_sum += angle
        K[i] -= angle_sum
    return K

def triangle_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))
def tet_volume(a, b, c, d):
    return abs(np.dot((a - d), np.cross(b - d, c - d))) / 6
def tet_height(a, b, c, d):
    n = np.cross(b-a, c-a)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-14:
        return 0.0
    n_unit = n / n_norm
    h = np.abs(np.dot(d-a, n_unit))
    return h
def sorted_tuple(a, b, c):
    return tuple(sorted((a, b, c)))

################################################################################
# ----------- VOLUME CORRECTION METHODS ----------------------------------------
################################################################################

def volume_paraboloid_patch_method(a, b, c, d, kA, kB, kC):
    """
    Vertex-Based 2nd-Order Paraboloid Patch Correction.
    """
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
    V_corr = V_flat + (1 / 3) * A_flat * s_avg
    return V_corr, V_flat

def volume_paraboloid_patch_normal_method(a, b, c, d, kA, kB, kC, nA, nB, nC):
    """
    2nd-Order Paraboloid Patch with Averaged Normal Correction.
    """
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
    n_avg = (nA + nB + nC) / 3
    if np.linalg.norm(n_avg) > 1e-8:
        n_avg = n_avg / np.linalg.norm(n_avg)
    else:
        n_avg = np.array([0,0,1])
    A_flat = triangle_area(a, b, c)
    V_flat = tet_volume(a, b, c, d)
    V_corr = V_flat + (1/3) * A_flat * s_avg
    return V_corr, V_flat

def volume_high_order_paraboloid_patch_method(a, b, c, d, kA, kB, kC, nA, nB, nC):
    """
    2nd-Order Paraboloid Patch via Tangent Plane Integration.
    """
    V_flat = tet_volume(a, b, c, d)
    bary = (a + b + c) / 3
    n_avg = (nA + nB + nC) / 3
    if np.linalg.norm(n_avg) > 1e-8:
        n_avg = n_avg / np.linalg.norm(n_avg)
    else:
        n_avg = np.array([0,0,1])
    H_avg = (kA + kB + kC) / 3
    u_dir = (a - bary)
    if np.linalg.norm(u_dir) > 1e-8:
        u_dir = u_dir / np.linalg.norm(u_dir)
    else:
        u_dir = np.array([1,0,0])
    v_dir = np.cross(n_avg, u_dir)
    v_dir = v_dir / np.linalg.norm(v_dir)
    def to_uv(pt):
        rel = pt - bary
        u = np.dot(rel, u_dir)
        v = np.dot(rel, v_dir)
        return np.array([u, v])
    uv_a, uv_b, uv_c = to_uv(a), to_uv(b), to_uv(c)
    tri = np.array([uv_a, uv_b, uv_c])
    def poly_integral(tri):
        (x1,y1), (x2,y2), (x3,y3) = tri
        area = 0.5 * abs((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))
        Iu2 = area * (x1**2 + x2**2 + x3**2 + x1*x2 + x1*x3 + x2*x3)/12
        Iv2 = area * (y1**2 + y2**2 + y3**2 + y1*y2 + y1*y3 + y2*y3)/12
        return Iu2, Iv2, area
    Iu2, Iv2, tri_area = poly_integral(tri)
    mean_patch_height = 0.5 * H_avg * (Iu2 + Iv2) / tri_area
    V_patch = tri_area * mean_patch_height / 3
    return V_flat + V_patch, V_flat

def volume_second_order_taylor_patch_method(a, b, c, d, kA, kB, kC, gA, gB, gC):
    """
    Second-Order Taylor Patch (Mean + Gaussian Curvature Correction).
    """
    H = (kA + kB + kC) / 3
    K = (gA + gB + gC) / 3
    bary = (a + b + c) / 3
    rA = np.linalg.norm(a - bary)
    rB = np.linalg.norm(b - bary)
    rC = np.linalg.norm(c - bary)
    avg_r2 = (rA**2 + rB**2 + rC**2) / 3
    avg_r4 = (rA**4 + rB**4 + rC**4) / 3
    A_flat = triangle_area(a, b, c)
    deltaV_H = (1/6) * H * A_flat * avg_r2
    deltaV_K = (1/24) * K * A_flat * avg_r4
    V_flat = tet_volume(a, b, c, d)
    V_corr = V_flat + deltaV_H + deltaV_K
    return V_corr, V_flat

def volume_dual_basis_method(a, b, c, d, HA, HB, HC):
    """
    2nd-Order Dual Basis Area Correction.
    """
    A_flat = triangle_area(a, b, c)
    bary = (a + b + c) / 3
    ws = [1/3, 1/3, 1/3]
    rs = [np.linalg.norm(a - bary), np.linalg.norm(b - bary), np.linalg.norm(c - bary)]
    Hs = [HA, HB, HC]
    delta_A = sum([w * A_flat * (H/6 * r) for w, H, r in zip(ws, Hs, rs)])
    A_curved = A_flat + delta_A
    h = tet_height(a, b, c, d)
    V_curved = (1/3) * h * A_curved
    V_flat = (1/3) * h * A_flat
    return V_curved, V_flat

def volume_fem_quadrature_method(a, b, c, d, kA, kB, kC, n_samples=8):
    """
    2nd-Order FEM Quadrature Correction.
    """
    bary = (a + b + c) / 3
    area = triangle_area(a, b, c)
    volume_corr = 0.0
    for i in range(n_samples + 1):
        for j in range(n_samples + 1 - i):
            u = i / n_samples
            v = j / n_samples
            w = 1 - u - v
            pt = w * a + u * b + v * c
            k = w * kA + u * kB + v * kC
            r2 = np.sum((pt - bary)**2)
            h = 0.5 * k * r2
            dA = area / (0.5 * n_samples * n_samples)
            volume_corr += h * dA
    V_flat = tet_volume(a, b, c, d)
    V_corr = V_flat + (1/3) * volume_corr
    return V_corr, V_flat

################################################################################
# ----------- TOTAL VOLUME WRAPPERS --------------------------------------------
################################################################################

def total_volume_high_order_paraboloid_patch(points, tets, vertex_curvature, vertex_normals):
    """
    2nd-Order Paraboloid Patch via Tangent Plane Integration (entire mesh).
    """
    total_vol_flat = 0.0
    total_vol_corr = 0.0
    for tet in tets:
        iA, iB, iC, iD = tet
        a, b, c, d = points[iA], points[iB], points[iC], points[iD]
        kA, kB, kC = vertex_curvature[iA], vertex_curvature[iB], vertex_curvature[iC]
        nA, nB, nC = vertex_normals[iA], vertex_normals[iB], vertex_normals[iC]
        V_corr, V_flat = volume_high_order_paraboloid_patch_method(a, b, c, d, kA, kB, kC, nA, nB, nC)
        total_vol_corr += V_corr
        total_vol_flat += V_flat
    return total_vol_corr, total_vol_flat

def total_volume_second_order_taylor_patch(points, tets, vertex_curvature, vertex_gauss):
    """
    Second-Order Taylor Patch (Mean + Gaussian Curvature Correction) (entire mesh).
    """
    total_vol_corr = 0.0
    total_vol_flat = 0.0
    for tet in tets:
        iA, iB, iC, iD = tet
        a, b, c, d = points[iA], points[iB], points[iC], points[iD]
        kA, kB, kC = vertex_curvature[iA], vertex_curvature[iB], vertex_curvature[iC]
        gA, gB, gC = vertex_gauss[iA], vertex_gauss[iB], vertex_gauss[iC]
        V_corr, V_flat = volume_second_order_taylor_patch_method(a, b, c, d, kA, kB, kC, gA, gB, gC)
        total_vol_corr += V_corr
        total_vol_flat += V_flat
    return total_vol_corr, total_vol_flat

def total_volume_dual_basis(points, tets, tets_inside, tets_surface, tet_surface_faces, vertex_curvature):
    """
    2nd-Order Dual Basis Area Correction (entire mesh).
    """
    total_vol_flat = 0.0
    total_vol_corr = 0.0
    for tet in tets_inside:
        a, b, c, d = points[tet[0]], points[tet[1]], points[tet[2]], points[tet[3]]
        vol = tet_volume(a, b, c, d)
        total_vol_flat += vol
        total_vol_corr += vol
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
        HA = vertex_curvature[idxs[ids[0]]]
        HB = vertex_curvature[idxs[ids[1]]]
        HC = vertex_curvature[idxs[ids[2]]]
        V_corr, V_flat = volume_dual_basis_method(a, b, c, d, HA, HB, HC)
        total_vol_flat += V_flat
        total_vol_corr += V_corr
    return total_vol_corr, total_vol_flat

def total_volume_fem_quadrature(points, tets, tets_inside, tets_surface, tet_surface_faces, vertex_curvature, n_samples=8):
    """
    2nd-Order FEM Quadrature Correction (entire mesh).
    """
    total_vol_flat = 0.0
    total_vol_corr = 0.0
    for tet in tets_inside:
        a, b, c, d = points[tet[0]], points[tet[1]], points[tet[2]], points[tet[3]]
        vol = tet_volume(a, b, c, d)
        total_vol_flat += vol
        total_vol_corr += vol
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
        V_corr, V_flat = volume_fem_quadrature_method(a, b, c, d, kA, kB, kC, n_samples=n_samples)
        total_vol_flat += V_flat
        total_vol_corr += V_corr
    return total_vol_corr, total_vol_flat

################################################################################
# ----------- TETRAHEDRA CLASSIFICATION ----------------------------------------
################################################################################

def classify_tets(points, cells, surface_tris):
    def sorted_tuple(a, b, c):
        return tuple(sorted((a, b, c)))
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

################################################################################
# ----------- ERROR STUDY: CYLINDER --------------------------------------------
################################################################################

def run_volume_error_study_cylinder(radius, height, lc_values):
    """
    Runs a full error study for the cylinder mesh.
    """
    tet_counts = []
    err_piecewise = []
    err_paraboloid = []
    err_paraboloid_normal = []
    err_high_order_patch = []
    err_second_order_patch = []
    err_dual_basis = []
    err_fem_quadrature = []

    for lc in lc_values:
        mshfile = f"cylinder_lc{lc:.2f}.msh"
        geofile = "cylinder.geo"
        if os.path.isfile(geofile): os.remove(geofile)
        if os.path.isfile(mshfile): os.remove(mshfile)
        print(f"\nGenerating cylinder tet mesh with gmsh (lc={lc:.2f}) ...")
        generate_gmsh_cylinder_mesh(filename=mshfile, radius=radius, height=height, lc=lc)
        mesh = meshio.read(mshfile)
        points = mesh.points
        tets = mesh.cells_dict["tetra"]
        surface_tris = mesh.cells_dict["triangle"]
        vertex_curvature = ddg_vertex_mean_curvature(points, surface_tris)
        vertex_normals = ddg_vertex_normals(points, surface_tris)
        vertex_gauss = ddg_vertex_gaussian_curvature(points, surface_tris)
        tets_inside, tets_surface, tet_surface_faces = classify_tets(points, tets, surface_tris)

        vol_piecewise = 0.0
        vol_paraboloid = 0.0
        vol_paraboloid_normal = 0.0
        vol_high_order_patch = 0.0
        vol_dual_basis = 0.0
        vol_fem_quadrature = 0.0
        vol_second_order_patch = 0.0

        # Interior tets
        for tet in tets_inside:
            a, b, c, d = points[tet[0]], points[tet[1]], points[tet[2]], points[tet[3]]
            vol = tet_volume(a, b, c, d)
            vol_piecewise += vol
            vol_paraboloid += vol
            vol_paraboloid_normal += vol
            vol_high_order_patch += vol
            vol_dual_basis += vol
            vol_fem_quadrature += vol
            vol_second_order_patch += vol

        # Surface tets: corrections
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
            gA = vertex_gauss[idxs[ids[0]]]
            gB = vertex_gauss[idxs[ids[1]]]
            gC = vertex_gauss[idxs[ids[2]]]
            nA = vertex_normals[idxs[ids[0]]]
            nB = vertex_normals[idxs[ids[1]]]
            nC = vertex_normals[idxs[ids[2]]]

            Vp_corr, Vp_flat = volume_paraboloid_patch_method(a, b, c, d, kA, kB, kC)
            vol_piecewise += Vp_flat
            vol_paraboloid += Vp_corr

            Vpn_corr, _ = volume_paraboloid_patch_normal_method(a, b, c, d, kA, kB, kC, nA, nB, nC)
            vol_paraboloid_normal += Vpn_corr

            Vho_corr, _ = volume_high_order_paraboloid_patch_method(a, b, c, d, kA, kB, kC, nA, nB, nC)
            vol_high_order_patch += Vho_corr

            Vdb_corr, _ = volume_dual_basis_method(a, b, c, d, kA, kB, kC)
            vol_dual_basis += Vdb_corr

            Vfq_corr, _ = volume_fem_quadrature_method(a, b, c, d, kA, kB, kC, n_samples=8)
            vol_fem_quadrature += Vfq_corr

            Vso_corr, _ = volume_second_order_taylor_patch_method(a, b, c, d, kA, kB, kC, gA, gB, gC)
            vol_second_order_patch += Vso_corr

        theoretical = np.pi * radius**2 * height
        tet_counts.append(len(tets))
        err_piecewise.append(abs(vol_piecewise - theoretical) / theoretical * 100)
        err_paraboloid.append(abs(vol_paraboloid - theoretical) / theoretical * 100)
        err_paraboloid_normal.append(abs(vol_paraboloid_normal - theoretical) / theoretical * 100)
        err_high_order_patch.append(abs(vol_high_order_patch - theoretical) / theoretical * 100)
        err_second_order_patch.append(abs(vol_second_order_patch - theoretical) / theoretical * 100)
        err_dual_basis.append(abs(vol_dual_basis - theoretical) / theoretical * 100)
        err_fem_quadrature.append(abs(vol_fem_quadrature - theoretical) / theoretical * 100)
        print(
            f"lc={lc:.2f}, #tets={len(tets)}, "
            f"err_piecewise={err_piecewise[-1]:.3e}, "
            f"err_paraboloid={err_paraboloid[-1]:.3e}, "
            f"err_paraboloid_normal={err_paraboloid_normal[-1]:.3e}, "
            f"err_high_order_patch={err_high_order_patch[-1]:.3e}, "
            f"err_second_order_patch={err_second_order_patch[-1]:.3e}, "
            f"err_dual_basis={err_dual_basis[-1]:.3e}, "
            f"err_fem_quadrature={err_fem_quadrature[-1]:.3e}"
        )

    return (
        tet_counts,
        err_piecewise,
        err_paraboloid,
        err_paraboloid_normal,
        err_high_order_patch,
        err_second_order_patch,
        err_dual_basis,
        err_fem_quadrature
    )

################################################################################
# ----------- ERROR STUDY: SPHERE/ELLIPSOID ------------------------------------
################################################################################

def run_volume_error_study_sphere(ax_, ay_, az_, lc_values):
    """
    Error study for ellipsoid (axes ax_, ay_, az_) for a range of mesh sizes lc_values.
    """
    tet_counts = []
    err_piecewise = []
    err_paraboloid = []
    err_paraboloid_normal = []
    err_high_order_patch = []
    err_second_order_patch = []
    err_dual_basis = []
    err_fem_quadrature = []

    for lc in lc_values:
        mshfile = f"sphere_lc{lc:.2f}.msh"
        geofile = f"sphere_lc{lc:.2f}.geo"
        if os.path.isfile(geofile): os.remove(geofile)
        if os.path.isfile(mshfile): os.remove(mshfile)
        generate_sphere_geo(geofile, lc=lc)
        run_gmsh(geofile, mshfile)
        mesh = meshio.read(mshfile)
        points = mesh.points
        points_ellipsoid = points.copy()
        points_ellipsoid[:, 0] *= ax_
        points_ellipsoid[:, 1] *= ay_
        points_ellipsoid[:, 2] *= az_
        tets = mesh.cells_dict["tetra"]
        surface_tris = mesh.cells_dict["triangle"]
        vertex_curvature = ddg_vertex_mean_curvature(points_ellipsoid, surface_tris)
        vertex_normals = ddg_vertex_normals(points_ellipsoid, surface_tris)
        vertex_gauss = ddg_vertex_gaussian_curvature(points_ellipsoid, surface_tris)
        tets_inside, tets_surface, tet_surface_faces = classify_tets(points_ellipsoid, tets, surface_tris)

        # Piecewise and Paraboloid Patch Correction
        vol_piecewise = 0.0
        vol_paraboloid = 0.0
        for tet in tets_inside:
            a, b, c, d = points_ellipsoid[tet[0]], points_ellipsoid[tet[1]], points_ellipsoid[tet[2]], points_ellipsoid[tet[3]]
            vol = tet_volume(a, b, c, d)
            vol_piecewise += vol
            vol_paraboloid += vol
        for tet, face_id in zip(tets_surface, tet_surface_faces):
            idxs = [tet[0], tet[1], tet[2], tet[3]]
            face_tuples = [
                [0, 1, 2, 3],
                [0, 1, 3, 2],
                [0, 2, 3, 1],
                [1, 2, 3, 0]
            ]
            ids = face_tuples[face_id]
            a, b, c, d = points_ellipsoid[idxs[ids[0]]], points_ellipsoid[idxs[ids[1]]], points_ellipsoid[idxs[ids[2]]], points_ellipsoid[idxs[ids[3]]]
            kA = vertex_curvature[idxs[ids[0]]]
            kB = vertex_curvature[idxs[ids[1]]]
            kC = vertex_curvature[idxs[ids[2]]]
            V_corr, V_flat = volume_paraboloid_patch_method(a, b, c, d, kA, kB, kC)
            vol_piecewise += V_flat
            vol_paraboloid += V_corr

        # Dual Basis Correction
        vol_dual_basis, _ = total_volume_dual_basis(
            points_ellipsoid, tets, tets_inside, tets_surface, tet_surface_faces, vertex_curvature
        )

        # FEM Quadrature Correction
        vol_fem_quadrature, _ = total_volume_fem_quadrature(
            points_ellipsoid, tets, tets_inside, tets_surface, tet_surface_faces, vertex_curvature, n_samples=8
        )

        # Paraboloid Patch + Normal Correction
        vol_paraboloid_normal = 0.0
        for tet, face_id in zip(tets_surface, tet_surface_faces):
            idxs = [tet[0], tet[1], tet[2], tet[3]]
            face_tuples = [
                [0, 1, 2, 3],
                [0, 1, 3, 2],
                [0, 2, 3, 1],
                [1, 2, 3, 0]
            ]
            ids = face_tuples[face_id]
            a, b, c, d = points_ellipsoid[idxs[ids[0]]], points_ellipsoid[idxs[ids[1]]], points_ellipsoid[idxs[ids[2]]], points_ellipsoid[idxs[ids[3]]]
            kA = vertex_curvature[idxs[ids[0]]]
            kB = vertex_curvature[idxs[ids[1]]]
            kC = vertex_curvature[idxs[ids[2]]]
            nA = vertex_normals[idxs[ids[0]]]
            nB = vertex_normals[idxs[ids[1]]]
            nC = vertex_normals[idxs[ids[2]]]
            V_corr, V_flat = volume_paraboloid_patch_normal_method(a, b, c, d, kA, kB, kC, nA, nB, nC)
            vol_paraboloid_normal += V_corr
        vol_paraboloid_normal += sum(tet_volume(points_ellipsoid[tet[0]], points_ellipsoid[tet[1]], points_ellipsoid[tet[2]], points_ellipsoid[tet[3]]) for tet in tets_inside)

        # High-Order Paraboloid Patch
        vol_high_order_patch, _ = total_volume_high_order_paraboloid_patch(
            points_ellipsoid, tets, vertex_curvature, vertex_normals
        )

        # Second-Order Taylor Patch (Mean+Gaussian)
        vol_second_order_patch, _ = total_volume_second_order_taylor_patch(
            points_ellipsoid, tets, vertex_curvature, vertex_gauss
        )

        theoretical = (4 / 3) * np.pi * ax_ * ay_ * az_
        tet_counts.append(len(tets))
        err_piecewise.append(abs(vol_piecewise - theoretical) / theoretical * 100)
        err_paraboloid.append(abs(vol_paraboloid - theoretical) / theoretical * 100)
        err_paraboloid_normal.append(abs(vol_paraboloid_normal - theoretical) / theoretical * 100)
        err_high_order_patch.append(abs(vol_high_order_patch - theoretical) / theoretical * 100)
        err_second_order_patch.append(abs(vol_second_order_patch - theoretical) / theoretical * 100)
        err_dual_basis.append(abs(vol_dual_basis - theoretical) / theoretical * 100)
        err_fem_quadrature.append(abs(vol_fem_quadrature - theoretical) / theoretical * 100)
        print(
            f"lc={lc:.2f}, #tets={len(tets)}, "
            f"err_piecewise={err_piecewise[-1]:.3e}, "
            f"err_paraboloid={err_paraboloid[-1]:.3e}, "
            f"err_paraboloid_normal={err_paraboloid_normal[-1]:.3e}, "
            f"err_high_order_patch={err_high_order_patch[-1]:.3e}, "
            f"err_second_order_patch={err_second_order_patch[-1]:.3e}, "
            f"err_dual_basis={err_dual_basis[-1]:.3e}, "
            f"err_fem_quadrature={err_fem_quadrature[-1]:.3e}"
        )

    return (
        tet_counts,
        err_piecewise,
        err_paraboloid,
        err_paraboloid_normal,
        err_high_order_patch,
        err_second_order_patch,
        err_dual_basis,
        err_fem_quadrature
    )

################################################################################
# ----------- UNIFIED PLOT FUNCTION --------------------------------------------
################################################################################

def plot_volume_error_vs_tets(
    tet_counts,
    err_piecewise,
    err_paraboloid,
    err_paraboloid_normal,
    err_high_order_patch,
    err_second_order_patch,
    err_dual_basis,
    err_fem_quadrature,
    geometry_name="Cylinder"
):
    plt.figure(figsize=(11,7))
    plt.scatter(tet_counts, err_piecewise, label='Piecewise Linear (Flat Tetrahedra) Volume', color='C1', marker='s')
    plt.scatter(tet_counts, err_paraboloid, label='Vertex-Based 2nd-Order Paraboloid Patch Correction', color='C0', marker='o')
    plt.scatter(tet_counts, err_paraboloid_normal, label='2nd-Order Paraboloid Patch with Averaged Normal Correction', color='C4', marker='*')
    plt.scatter(tet_counts, err_high_order_patch, label='2nd-Order Paraboloid Patch via Tangent Plane Integration', color='C5', marker='X')
    plt.scatter(tet_counts, err_second_order_patch, label='Second-Order Taylor Patch (Mean + Gaussian Curvature Correction)', color='C9', marker='P')
    plt.scatter(tet_counts, err_dual_basis, label='2nd-Order Dual Basis Area Correction', color='C2', marker='^')
    plt.scatter(tet_counts, err_fem_quadrature, label='2nd-Order FEM Quadrature Correction', color='C3', marker='d')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Tet Elements (log scale)')
    plt.ylabel('Relative Error (%) (log scale)')
    plt.title(f'Volume Approximation Error vs. Tet Count ({geometry_name})')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
