from pathlib import Path
from collections import defaultdict
import os
import numpy as np
import meshio

USER_OPEN_INTERACTIVE_WINDOW = True

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ddgclib-mpl")
import matplotlib
if not USER_OPEN_INTERACTIVE_WINDOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

script_dir = Path(__file__).resolve().parent
msh_file = script_dir / "mesh_iter9900.msh"
png_file = script_dir / "mesh_iter9900.png"

print("Reading mesh:", msh_file)
mesh = meshio.read(msh_file)
points = mesh.points

# Read tetra cells
tets = []
for cell_block in mesh.cells:
    print(cell_block.type, cell_block.data.shape)

    if cell_block.type == "tetra":
        tets.append(cell_block.data[:, :4])
    elif cell_block.type == "tetra10":
        tets.append(cell_block.data[:, :4])

if not tets:
    raise ValueError("No tetra cells found.")

tets = np.vstack(tets)

# Compute tetra volumes
def signed_tet_volume(a, b, c, d):
    return np.dot(a - d, np.cross(b - d, c - d)) / 6.0

vol_tetra = np.array([
    signed_tet_volume(points[t[0]], points[t[1]], points[t[2]], points[t[3]])
    for t in tets
])

print("sum(abs(vol_tetra)) =", np.sum(np.abs(vol_tetra)))
print("sum(vol_tetra)      =", np.sum(vol_tetra))
print("min(vol_tetra)      =", np.min(vol_tetra))
print("max(vol_tetra)      =", np.max(vol_tetra))

# Extract boundary faces
face_count = defaultdict(int)

for t in tets:
    faces = [
        (t[0], t[1], t[2]),
        (t[0], t[1], t[3]),
        (t[0], t[2], t[3]),
        (t[1], t[2], t[3]),
    ]
    for f in faces:
        face_count[tuple(sorted(map(int, f)))] += 1

boundary_faces = np.array(
    [f for f, count in face_count.items() if count == 1],
    dtype=int
)

print("number of tetrahedra      =", len(tets))
print("number of boundary faces  =", len(boundary_faces))

if len(boundary_faces) == 0:
    raise ValueError("No boundary faces found.")

# Extract boundary edges only
boundary_edge_set = set()
for f in boundary_faces:
    a, b, c = map(int, f)
    boundary_edge_set.add(tuple(sorted((a, b))))
    boundary_edge_set.add(tuple(sorted((a, c))))
    boundary_edge_set.add(tuple(sorted((b, c))))

boundary_edges = np.array(sorted(boundary_edge_set), dtype=int)
boundary_vertices = np.unique(boundary_faces.ravel())

print("number of boundary edges  =", len(boundary_edges))
print("number of boundary verts  =", len(boundary_vertices))

# Plot only the surface mesh
# Coordinates shown in mm, while volume check above stays in SI units.
plot_points = 1.0e3 * points
triangles_xyz = plot_points[boundary_faces]
edge_segments = plot_points[boundary_edges]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

mesh_collection = Poly3DCollection(
    triangles_xyz,
    alpha=0.10,
    facecolor="#5dade2",
    edgecolor="none",
)
ax.add_collection3d(mesh_collection)

edge_collection = Line3DCollection(
    edge_segments,
    colors="#111111",
    linewidths=0.35,
    alpha=0.90,
)
ax.add_collection3d(edge_collection)

ax.scatter(
    plot_points[boundary_vertices, 0],
    plot_points[boundary_vertices, 1],
    plot_points[boundary_vertices, 2],
    s=3.0,
    c="#111111",
    alpha=0.90,
    depthshade=False,
)

# Equal axis scaling
surface_pts = plot_points[boundary_vertices]
mins = surface_pts.min(axis=0)
maxs = surface_pts.max(axis=0)
center = 0.5 * (mins + maxs)
radius = 0.5 * np.max(maxs - mins)

ax.set_xlim(center[0] - radius, center[0] + radius)
ax.set_ylim(center[1] - radius, center[1] + radius)
ax.set_zlim(center[2] - radius, center[2] + radius)

ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_zlabel("z [mm]")
ax.set_title(
    f"Case 2b axisym Gmsh: {msh_file.stem}: surface mesh only\n"
    f"surface vertices={len(boundary_vertices)}, "
    f"boundary faces={len(boundary_faces)}, "
    f"boundary edges={len(boundary_edges)}"
)

plt.tight_layout()
fig.savefig(png_file, dpi=180, bbox_inches="tight")
print("Wrote:", png_file)

if USER_OPEN_INTERACTIVE_WINDOW:
    print("Opening interactive Matplotlib window. Close the window to finish.")
    plt.show()
else:
    plt.close(fig)