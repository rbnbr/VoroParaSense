import numpy as np
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from src.clustering.clustering_algorithm import ClusteringAlgorithm


class KMeansClusteringAlgorithm(ClusteringAlgorithm):
    def __init__(self, n_clusters: int, metric, initial_centers=None, *, kmeans_kwargs: dict = None, kmeans_plus_plus_kwargs: dict = None):
        super().__init__(n_clusters=n_clusters)

        self.metric = metric
        self.initial_centers = initial_centers
        self._initial_centers_indices = None
        self.kmeans_instance = None
        self.kmeans_kwargs = dict() if kmeans_kwargs is None else kmeans_kwargs
        self.kmeans_plus_plus_kwargs = dict() if kmeans_plus_plus_kwargs is None else kmeans_plus_plus_kwargs

    def initialize(self, data: np.ndarray):
        """
        Initializes the initial centers via the kmeans++ algorithm.
        Overrides the properties self.initial_centers and self._initial_centers_indices.
        :param data:
        :return: Returns the initial centers.
        """
        # logger.info("initialize centers via kmeans++")
        self._initial_centers_indices = kmeans_plusplus_initializer(data, self.n_clusters, **self.kmeans_plus_plus_kwargs).initialize(return_index=True)
        self.initial_centers = data[self._initial_centers_indices]
        # logger.debug("done with kmeans++")
        return self.initial_centers

    def process(self, data: np.ndarray, random_center: bool = None):
        """
        Runs the kmean instance on the provided data, sets the self.kmeans_instance.
        If self.initial_centers is None, chooses random centroids as initial centers.
        The chosen initial centers indices will be set to self._initial_centers_indices.
        Overrides the properties self._initial_centers_indices and self.kmeans_instance.
        :param random_center: If True, chooses center by random, if None uses existing centers, else applies kmeans++.
        :param data:
        :return: clusters, centroids
        """
        if random_center is True:
            # logger.info("using random centers")
            initial_centers_indices = np.random.choice(np.arange(len(data)), self.n_clusters, replace=False)
            initial_centers = data[initial_centers_indices]
            self._initial_centers_indices = initial_centers_indices
        elif self.initial_centers is None or random_center is None:
            # logger.info("using kmeans++")
            initial_centers = self.initialize(data)
        else:
            # logger.info("using existing centers")
            initial_centers = self.initial_centers

        kmeans_instance = kmeans(data, initial_centers, metric=self.metric, **self.kmeans_kwargs)
        self.kmeans_instance = kmeans_instance

        # logger.info("start kmeans process")
        kmeans_instance.process()
        # logger.debug("done kmeans process")
        return kmeans_instance.get_clusters(), kmeans_instance.get_centers()
