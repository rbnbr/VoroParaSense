# Import necessary libraries
import sys
sys.path.extend(["../"])
from src.para_voro import *
from src.para_voro_plots import *

# Create a sample dataset with 10 samples
np.random.seed(0)
points = np.random.rand(10, 2)

# Build the Voronoi diagram
vor = Voronoi(points)

[point_idx_region_mapping, region_idx_point_idx_mapping, regions_with_infinity_edges, 
 region_idx_neighboring_regions_indices_mapping, 
 region_idx_ridge_indices_mapping] = identify_infinity_regions_and_neighbors(vor)

aabbox = np.array([
    [-0, 1.2],
    [-0.2, 1.2]
])

new_vertices_and_ridge_indices = [clip_ridges_of_voronoi_region(
    regions_with_infinity_edges[i], vor, aabbox, region_idx_point_idx_mapping, 
    region_idx_ridge_indices_mapping)[1]
    for i in range(len(regions_with_infinity_edges))]

# combine them
ridge_indices = []
new_vertices = []

for vertices_, ridge_indices_ in new_vertices_and_ridge_indices:
    for vertices, ridge_index in zip(vertices_, ridge_indices_):
        if ridge_index not in ridge_indices:
            ridge_indices.append(ridge_index)
            new_vertices.append(vertices)

plt.close()
fig = plt.figure()
ax = fig.gca()
ax.scatter(points[:, 0], points[:, 1], c='blue', label='Samples')

voronoi_plot_2d(vor, show_vertices=True, show_points=False, line_colors="green", ax=ax)

color = ["black", "yellow", "green", "orange", "purple", "gray"]

for i, [vertices, ridge_index] in enumerate(zip(new_vertices, ridge_indices)):
    ax.scatter(vertices[:, 0], vertices[:, 1], c=color[i], label=f"New ridge vertices r{ridge_index}", s=2**7)

ax.set_xlim(aabbox[0])
ax.set_ylim(aabbox[1])

plt.legend()
plt.title('Voronoi Diagram with Clipped Infinite Ridges')
plt.show()
