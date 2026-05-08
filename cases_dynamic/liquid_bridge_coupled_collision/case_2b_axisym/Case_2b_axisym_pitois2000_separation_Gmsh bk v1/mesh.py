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
msh_file = script_dir / "mesh_iter1490.msh"
png_file = script_dir / "mesh_iter1490.png"

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
        face_count[tuple(sorted(f))] += 1

boundary_faces = np.array([
    f for f, count in face_count.items()
    if count == 1
], dtype=int)

print("number of tetrahedra      =", len(tets))
print("number of boundary faces  =", len(boundary_faces))

# Extract every tetrahedral edge, not only the liquid-air interface.
edge_set = set()
for t in tets:
    for a, b in (
        (t[0], t[1]),
        (t[0], t[2]),
        (t[0], t[3]),
        (t[1], t[2]),
        (t[1], t[3]),
        (t[2], t[3]),
    ):
        edge_set.add(tuple(sorted((int(a), int(b)))))
tet_edges = np.array(sorted(edge_set), dtype=int)
print("number of tetra edges     =", len(tet_edges))

# Plot full tetrahedral mesh with Matplotlib: all tet edges + all vertices.
# Coordinates are shown in mm, while the volume check above stays in SI units.
plot_points = 1.0e3 * points
triangles_xyz = plot_points[boundary_faces]
edge_segments = plot_points[tet_edges]

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
    linewidths=0.16,
    alpha=0.58,
)
ax.add_collection3d(edge_collection)

ax.scatter(
    plot_points[:, 0],
    plot_points[:, 1],
    plot_points[:, 2],
    s=2.0,
    c="#111111",
    alpha=0.80,
    depthshade=False,
)

# Equal axis scaling
mins = plot_points.min(axis=0)
maxs = plot_points.max(axis=0)
center = 0.5 * (mins + maxs)
radius = 0.5 * np.max(maxs - mins)

ax.set_xlim(center[0] - radius, center[0] + radius)
ax.set_ylim(center[1] - radius, center[1] + radius)
ax.set_zlim(center[2] - radius, center[2] + radius)

ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_zlabel("z [mm]")
ax.set_title(
    f"Case 2b axisym Gmsh: {msh_file.stem}: full tetrahedral mesh\n"
    f"vertices={len(points)}, tets={len(tets)}, tet edges={len(tet_edges)}"
)

plt.tight_layout()
fig.savefig(png_file, dpi=180, bbox_inches="tight")
print("Wrote:", png_file)

if USER_OPEN_INTERACTIVE_WINDOW:
    print("Opening interactive Matplotlib window. Close the window to finish.")
    plt.show()
else:
    plt.close(fig)
