from collections import namedtuple

import numpy as np
from sklearn.cluster import KMeans
import torch

def evaluate_distribution(
        target: torch.Tensor,
        reference: torch.Tensor,
        *,
        N_cluster: int = 10, # number of clusters considered in k-means
        combined: bool = True, # False: results for each cluster. True: combine results of all clusters
    ) -> dict:
    """ compare the target and predicted distribution.
        Distributions are given as torch tensors containing data points clouds.
    """
    distribution_evaluation = DistributionEvaluation(target, N_cluster)
    distribution_evaluation.add("reference", reference)
    distribution_evaluation.process(True)
    results_clusters = {
        "histograms": distribution_evaluation.histograms,
        "errors": distribution_evaluation.get_errors(),
        "distances": distribution_evaluation.get_distances(),
    }
    return results_clusters,

class DistributionEvaluation:
    """ Class for comparison of distributions using k-means.
    N_clusters cluster centers are fitted to the data_target.
    For data_target and other added datasets, the number of points in each cluster are computed and saved to
    self.histograms by calling self.process().
    Calling self.get_errors() computes the L2 distance to the target histogram for every added dataset and saved to self.errors.
    Furthermore, the average distance of points in a cluster to the cluster center can be computed for every cluster by
     calling self.get_distances(). This requires that self.process is called with do_distance_transform=True.
    """

    def __init__(self,
                 data_target: np.array,  # data from distribution of target
                 N_clusters: int  # number of clusters used in k-means
                 ):
        self.data = {}
        self.data["target"] = data_target
        self.N_clusters = N_clusters

    @property
    def N_clusters(self):
        return self._N_clusters

    @N_clusters.setter
    def N_clusters(self, value):
        self._N_clusters = value
        # initialize k_means function accordingly
        self.k_means = KMeans(n_clusters=value, random_state=0).fit(self.data["target"])

    def add(self,
            name: str,  # identifier for distribution
            data_points: np.array,  # data points from distribution
            ):
        """ add or replace dataset. """
        self.data[name] = data_points

    def process(self,
                do_distance_transform: bool = False  # if True: compute k-means transforms
                ):
        """ compute results after all distributions are given. """
        self.predictions, self.histograms, self.distance_transforms = {}, {}, {}
        for key, value in self.data.items():
            # compute nearest cluster center for every data point
            self.predictions[key] = self.k_means.predict(value)
            # count number of points in a cluster
            self.histograms[key] = np.bincount(self.predictions[key])
            if do_distance_transform:
                # compute distance from every data point to every cluster center
                self.distance_transforms[key] = self.k_means.transform(value)

    def get_errors(self):
        """ return distance between distograms of training distribution and all other distributions. """
        errors = {key: compute_l2_distance(self.histograms["target"], value)
                  for key, value in self.histograms.items()
                  if not key == "target"}
        return errors

    def get_distances(self, squared: bool = True, combined: bool = True, std: bool = True):
        """ get (combined) average (squared) distance to cluster centers (and standard deviation) for every cluster for all distributions. """
        distances = {key: compute_distances(clusters, self.distance_transforms[key], self.N_clusters, squared=squared, combined=combined, std=std)
                     for key, clusters in self.predictions.items()
                     if not key == "target"}
        return distances

    ## depreceated
    def get_mean_distance(self, squared: bool = True, combined: bool = True, std: bool = True):
        """ (DEPRECEATED) get average (squared) distance to cluster centers over all clusters for all distributions. """
        distances = self.get_distances(squared=squared, combined=combined, std=std)
#        distances = {key: np.nanmean(value) for key, value in distances.items()} # this requires nanmean, for clusters may be empty -> NaN distance
        return distances


Distance = namedtuple("Distance", "RMS stddev")

def compute_distances(clusters: np.array, distances: np.array, N_clusters: int, squared: bool = True, std: bool = True, combined: bool = False):
    """ return for each cluster the average distance of data points to cluster center.

    Parameters
    ----------
    clusters : iterable
        contains clusters each data point belongs to.
    distances : numpy.array
        contains distances from each data point to each cluster center.
    N_clusters : int
        number of clusters
    squared : bool, default=True
        if True, compute RMS
    std : bool, default=True
        if True, return standard deviation as well
    combined : bool, default=True
        if True, return average distance and standard deviation combined for all clusters
    """
    distances = distances.min(1)  # distance to nearest cluster center
    if squared:
        distances = distances * distances
    if combined:
        average_distance = np.nanmean(distances)**(1-0.5*squared)
    else:
        average_distance = [np.nanmean(distances[clusters == cluster])**(1-0.5*squared) for cluster in range(N_clusters)]
    if not std:
        return average_distance
    if combined:
        standard_deviation = np.nanmean(np.abs(distances**(1-0.5*squared) - average_distance))
    else:
        standard_deviation = [np.nanmean(np.abs(distances[clusters == cluster]**(1-0.5*squared) - avg_dist))
                              for cluster, avg_dist in enumerate(average_distance)]
    return Distance(average_distance, standard_deviation)


def compute_l1_distance(target, values, div=1):
    """ Estimate L2 distance between values and target. """
    result = np.sum([np.abs(v - r)  / r for r, v in zip(target, values)])
    return result / div


def compute_l2_distance(target, values, div=1):
    """ Estimate L2 distance between values and target. """
    result = np.sum([(v - r)**2 / r**2 for r, v in zip(target, values)])
    return result / div