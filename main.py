# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.para_voro import *
from src.para_voro_plots import *


def gen_data(dim: int = 3, num_samples: int = 200):
    # Generate sample data for two intersecting clusters
    cluster1 = np.random.randn(num_samples, dim)
    cluster2 = np.random.randn(num_samples, dim)

    cluster1 = cluster1 + np.random.randint(2, size=dim) - 1
    cluster2 = cluster2 + np.random.randint(2, size=dim) - 1
    samples = np.vstack([cluster1, cluster2])

    # Add labels for the two clusters
    labels = np.array([0] * num_samples + [1] * num_samples)

    return samples, labels


def main():
    np.random.seed(0)
    dim = 2
    samples, labels = gen_data(dim=dim, num_samples=30)

    # samples = np.array([
    #     [0.0, 0.0],
    #     [0.5, 0.0],
    #     [0.0, 0.5],
    #     [0.5, 0.5]
    # ])
    # labels = np.array([0, 0, 1, 1])

    aabbox = np.array([
        [-3.0, 3.0],
        [-3.0, 3.0],
        [-3.0, 3.0],
        [-3.0, 3.0],
        [-3.0, 3.0],
        [-3.0, 3.0],
        [-3.0, 3.0],
        [-3.0, 3.0],
        [-3.0, 3.0],
        [-3.0, 3.0],
        [-3.0, 3.0],
        [-3.0, 3.0],
        [-3.0, 3.0],
        [-3.0, 3.0]
    ])[:dim]
    bandwidths = np.array([
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
    ])[:dim]

    print("plot data")
    if dim <= 3:
        ax = plot_data(samples, labels, clip_box=aabbox)

    print("compute ridge vertices")

    assert len(bandwidths) == len(aabbox) == dim

    cch = True

    ridge_vertices = generate_geometric_segment_boundaries_via_voronoi(samples=samples, labels=labels, clip_bbox=aabbox, clip_convex_hull=cch, verbose=True)
    # ridge_vertices = make_object_array([np.array([[0.25, 0.0], [0.25, 0.5]]),
    #                                     np.array([[0.0, 0.25], [0.5, 0.25]])], dtype=object)

    print("plot relevant ridges")
    if dim <= 3:
        plot_ridges(ridge_vertices, samples=samples, labels=labels, clip_box=aabbox, clip_convex_hull=cch)

    print("compute sample points and sensitivities")
    sample_points, sensitivities = ridge_based_para_sense(ridge_vertices=ridge_vertices, bandwidths=bandwidths, clip_box=aabbox, verbose=True)

    plot_sensitivities(sample_points, sensitivities, bandwidths=bandwidths)
    pass


if __name__ == "__main__":
    main()
