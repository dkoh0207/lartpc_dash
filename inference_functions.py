import numpy as np
import pandas as pd
import hdbscan

from sklearn.cluster import DBSCAN, OPTICS, MeanShift
from scipy.spatial.distance import cdist
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors

from collections import defaultdict
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import hdbscan


def predict_v2(embedding, seediness, nClusters=10, seed_threshold=0.8):
    pred = -np.ones((embedding.shape[0], ))
    seed_map = np.ones(seediness[:, -2].shape)
    margins = seediness[:, -1]
    scores = []
    if nClusters is None:
        nClusters = len(pred)
    for i in range(nClusters):
        if sum(seed_map) == 0:
            break
        if np.max(seediness[:, -2] * seed_map) < seed_threshold:
            break
        centroid = embedding[:, 4:][np.argmax(seediness[:, -2] * seed_map)]
        margin = margins[[np.argmax(seed_map)]]
        prob = np.exp(-np.linalg.norm(embedding[:, 4:] -
                                      centroid, axis=1)**2 / (2 * np.exp(margin)))
        scores.append(prob.reshape(prob.shape[0], 1))
        index = prob > 0.5
        pred[index] = i + 1
        seed_map[index] = 0
    scores = np.concatenate(scores, axis=1)
    pred = np.argmax(scores, axis=1)
    return pred, scores


def find_cluster_means(features, labels):
    '''
    For a given image, compute the centroids \mu_c for each
    cluster label in the embedding space.

    INPUTS:
        features (torch.Tensor) - the pixel embeddings, shape=(N, d) where
        N is the number of pixels and d is the embedding space dimension.

        labels (torch.Tensor) - ground-truth group labels, shape=(N, )

    OUTPUT:
        cluster_means (torch.Tensor) - (n_c, d) tensor where n_c is the number of
        distinct instances. Each row is a (1,d) vector corresponding to
        the coordinates of the i-th centroid.
    '''
    group_ids = sorted(np.unique(labels).astype(int))
    cluster_means = []
    #print(group_ids)
    for c in group_ids:
        index = labels.astype(int) == c
        mu_c = features[index].mean(0)
        cluster_means.append(mu_c)
    cluster_means = np.vstack(cluster_means)
    return group_ids, cluster_means


def choose_eps(embedding, n_neighbors=20, S=1.0, curve='convex', direction='decreasing'):

    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(embedding)
    distances, indices = knn.kneighbors(embedding)
    dists = np.sort(np.mean(distances, axis=1))[::-1]
    x = np.arange(len(dists))
    y = np.sort(np.mean(distances, axis=1))[::-1]
    kneedle = KneeLocator(x, y, S=S, curve=curve, direction=direction)
    return y[kneedle.knee]


def cluster_remainder(embedding, semi_predictions):
    if sum(semi_predictions == -1) == 0 or sum(semi_predictions != -1) == 0:
        return semi_predictions
    group_ids, predicted_cmeans = find_cluster_means(
        embedding, semi_predictions)
    semi_predictions[semi_predictions == -1] = np.argmin(
        cdist(embedding[semi_predictions == -1], predicted_cmeans[1:]), axis=1)
    return semi_predictions


def compute_purity_and_efficiency(pred, truth):
    clusters_pred = sorted(np.unique(pred))
    clusters_truth = sorted(np.unique(truth))
    plist = []
    elist = []
    for cp in clusters_pred:
        mask_p = pred == cp
        pur = []
        eff = []
        intlist = []
        for ct in clusters_truth:
            mask_t = truth == ct
            intersection = np.logical_and(mask_p, mask_t)
            intlist.append(sum(intersection))
            p = float(sum(intersection)) / sum(mask_p)
            e = float(sum(intersection)) / sum(mask_t)
            pur.append(p)
            eff.append(e)
        index = np.argmax(np.array(intlist))
        purity = pur[index]
        efficiency = eff[index]
        plist.append(purity)
        elist.append(efficiency)
    return sum(plist) / len(plist), sum(elist) / len(elist)
