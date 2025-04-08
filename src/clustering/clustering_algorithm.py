import numpy as np


class ClusteringAlgorithm:
    """
    An abstract clustering algorithm class for clustering to a fixed amount of clusters.
    """
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def process(self, data: np.ndarray):
        """
        Clusters the input into self.n_cluster clusters.
        Returns a list of clusters, where each cluster is a list of indices of the data elements that belong to this
        cluster.
        Also returns a list of centroids with one centroid for each cluster.
        :param data:
        :return: clusters, centroids
        """
        raise NotImplementedError()
