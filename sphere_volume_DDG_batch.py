import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

def plot_surface_mesh(surface_pts, surface_faces, title='Initial Surface Mesh'):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(surface_pts[:,0], surface_pts[:,1], surface_pts[:,2], triangles=surface_faces, color='cyan', alpha=0.5)
    ax.scatter(surface_pts[:,0], surface_pts[:,1], surface_pts[:,2], s=30, c='blue', label='Vertices')
    ax.set_title(title)
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()

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

# Main experiment parameters
radius = 1.0
subdivisions_range = range(0, 6)  # Only 0 to 5

# Show initial mesh before loop
surface_pts, surface_faces, all_points, tets = GeoMesh(subdivisions=0, radius=radius)
plot_surface_mesh(surface_pts, surface_faces, title='Initial Mesh (subdivisions=0)')

tet_counts = []
relerr_corr = []
relerr_flat = []

for subdivisions in subdivisions_range:
    surface_pts, surface_faces, all_points, tets = GeoMesh(subdivisions=subdivisions, radius=radius)
    vertex_curvature = DDG_curvature(surface_pts, surface_faces)
    total_corr = 0.0
    total_flat = 0.0
    for tet in tets:
        iA, iB, iC, iD = tet
        a, b, c, d = all_points[iA], all_points[iB], all_points[iC], all_points[iD]
        kA, kB, kC = vertex_curvature[iA], vertex_curvature[iB], vertex_curvature[iC]
        V_corr, V_flat = curvature_corrected_volume(a, b, c, d, kA, kB, kC)
        total_corr += V_corr
        total_flat += V_flat
    theoretical = (4/3)*np.pi*radius**3
    tet_counts.append(len(tets))
    relerr_corr.append(100 * abs(total_corr - theoretical) / theoretical)
    relerr_flat.append(100 * abs(total_flat - theoretical) / theoretical)
    # Print requested information
    print(f"subdivisions = {subdivisions}")
    print("Average curvature (should be close to 1.0):", vertex_curvature.mean())
    print("Std curvature:", vertex_curvature.std())
    print(f"Total computed volume with curvature correction: {total_corr:.6f}")
    print(f"Sum of all flat tet volumes (no correction, surface-based tets): {total_flat:.6f}")
    print(f"Theoretical sphere volume: {theoretical:.6f}")
    print(f"Number of tets: {len(tets)}")
    print(f"Error in volume: {abs(total_corr - theoretical) / theoretical:.6f} (should be small)")
    print("-"*50)

# Plotting
plt.figure(figsize=(8,6))
plt.scatter(tet_counts, relerr_corr, label='Curvature-Corrected Volume', color='C0', marker='o')
plt.scatter(tet_counts, relerr_flat, label='Flat Tet Volume', color='C1', marker='s')
plt.axhline(0, color='k', linestyle='--', label='Theoretical Sphere Volume (0% error)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Tet Elements (log scale)')
plt.ylabel('Relative Error (%) (log scale)')
plt.title('Volume Approximation Error vs. Tet Count')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
