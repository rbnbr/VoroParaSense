import sys
sys.path.extend(["../"])
from src.para_voro import *
from src.para_voro_plots import *

samples = np.array([
    [-1.0], [1.0]
])

labels = np.array([0, 1])

aabbox = np.array([[-2.0, 2.0]])

bandwidths = np.array([0.1])

plot_data(samples, labels, aabbox)

ridge_vertices = generate_geometric_segment_boundaries_via_voronoi(samples=samples, labels=labels, clip_bbox=aabbox)
plot_ridges(ridge_vertices, samples, labels, aabbox)

sample_points, sensitivities = ridge_based_para_sense(ridge_vertices=ridge_vertices, bandwidths=bandwidths, clip_box=aabbox, verbose=True)
plot_sensitivities(sample_points, sensitivities)

