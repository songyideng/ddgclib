import numpy as np
import trimesh
import matplotlib.pyplot as plt

def plot_surface_mesh(surface_pts, surface_faces, title='Initial Surface Mesh'):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(surface_pts[:,0], surface_pts[:,1], surface_pts[:,2], triangles=surface_faces, color='cyan', alpha=0.5)
    ax.scatter(surface_pts[:,0], surface_pts[:,1], surface_pts[:,2], s=30, c='blue', label='Vertices')
    ax.set_title(title)
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()

def DDG_curvature(vertices, faces):
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

def GeoMesh(subdivisions=3, radius=1.0):
    sphere_mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    surface_pts = sphere_mesh.vertices
    surface_faces = sphere_mesh.faces
    n_surf = surface_pts.shape[0]
    sphere_center = np.array([[0.0, 0.0, 0.0]])
    all_points = np.vstack((surface_pts, sphere_center))
    center_idx = n_surf
    tets = []
    for face in surface_faces:
        tet = [face[0], face[1], face[2], center_idx]
        tets.append(tet)
    tets = np.array(tets)
    return surface_pts, surface_faces, all_points, tets

def triangle_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b-a, c-a))

def tet_volume(a, b, c, d):
    return abs(np.dot((a-d), np.cross(b-d, c-d))) / 6

def curvature_corrected_volume(a, b, c, d, kA, kB, kC):
    bary = (a + b + c) / 3
    rA = np.linalg.norm(a - bary)
    rB = np.linalg.norm(b - bary)
    rC = np.linalg.norm(c - bary)
    RA = 1 / kA if kA > 1e-8 else 1e10
    RB = 1 / kB if kB > 1e-8 else 1e10
    RC = 1 / kC if kC > 1e-8 else 1e10
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

