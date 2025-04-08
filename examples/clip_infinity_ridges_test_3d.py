# Import necessary libraries
import sys
sys.path.extend(["../"])
from src.para_voro import *
from src.para_voro_plots import *

# Create a sample dataset with 10 samples
np.random.seed(0)

d = 10.0
aabbox = np.array([
    [-d, d],
    [-d, d],
    [-d, d]
])

points = np.random.random((10, 3))

vor = Voronoi(points)

[point_idx_region_mapping, region_idx_point_idx_mapping, regions_with_infinity_edges, 
 region_idx_neighboring_regions_indices_mapping, 
 region_idx_ridge_indices_mapping] = identify_infinity_regions_and_neighbors(vor)

print(regions_with_infinity_edges)

infinity_ridge_indices = np.argwhere(([np.any(np.asarray(dv) < 0) for dv in vor.ridge_vertices]))

print(infinity_ridge_indices)

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

for i in range(len(ridge_indices)):
    print(ridge_indices[i], new_vertices[i])

print(len(ridge_indices), sorted(ridge_indices))
print(len(infinity_ridge_indices), sorted(infinity_ridge_indices))
