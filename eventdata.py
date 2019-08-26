import numpy as np
import pandas as pd
import os
import re

from inference_functions import *

class EventData:
    """
    Input data csv must be of the form:

    - First 4 columns correspond to real x, y, z coordinates + b,
    where b is the batch index.

    - Remaining columns correspond to features/labels.
    """

    def __init__(self, dirname):

        self._data = {}
        self._dirname = dirname
        fnames = [os.path.join(self._dirname, f) for f in os.listdir(dirname)
                  if 'csv' in f]
        for i, f in enumerate(fnames):
            name = os.listdir(dirname)[i]
            name = re.findall('(\w+).csv', name)[0]
            df = pd.read_csv(f)
            self._data[name] = df
        self._events = sorted(np.unique(
            self._data['input_data']['b'].astype(int)))
        self.set_batch(0)

    def set_batch(self, bidx=0):

        d = {}
        mask = self._data['input_data'].b == bidx
        for key, df in self._data.items():
            d[key] = df[mask]
        self._filtered_data = d
        self._classes = sorted(np.unique(
            self._filtered_data['segment_label']['segment_label'].astype(int)))
        return self._filtered_data

    def set_class(self, c=0):
        d = {}
        mask = self._filtered_data['segment_label']['segment_label'] == c
        for key, df in self._filtered_data.items():
            d[key] = df[mask]
        self._filtered_data = d
        return self._filtered_data

    def set_batch_and_class(self, bidx=0, c=0):
        _ = self.set_batch(bidx)
        _ = self.set_class(c)
        return self._filtered_data

    def set_energy_threshold(self, threshold=0.05):
        d = {}
        mask = self._filtered_data['input_data']['energy_deposition'] > threshold
        for key, df in self._filtered_data.items():
            d[key] = df[mask]
        self._filtered_data = d
        return self._filtered_data

    def compute_proximity(self):
        """
        NOTE: Proximity is only computed for fixed batch id and semantic class.
        """
        data = self._filtered_data['embedding'].iloc[:, 4:].to_numpy().copy()
        clabels = self._filtered_data['cluster_label']['cluster_label'].to_numpy().astype(int).copy()
        proximity = np.zeros((self._filtered_data['input_data'].shape[0], ))
        #print("clabels = ", clabels)
        group_ids, cmeans = find_cluster_means(data, clabels)
        for i, gid in enumerate(group_ids):
            mask = clabels == gid
            dists = data[mask] - cmeans[i]
            dists = np.log(1 + np.sqrt(np.sum(np.power(dists, 2), axis=1)))
            proximity[mask] = dists.copy()
        proximity = pd.DataFrame(proximity.T, columns=['Proximity'])
        return proximity
        # centroids = []
        # proximity = np.zeros(data['segment_label'].shape)
        # proximity[:, 0:4] = data['coords'][:, 0:4].copy()
        # group_ids, cmeans = find_cluster_means(data['embedding'], clabels)
        # for i, gid in enumerate(group_ids):
        #     mask = clabels == gid
        #     dists = data['embedding'][mask] - cmeans[i]
        #     dists = np.log(1 + np.sqrt(np.sum(np.power(dists, 2), axis=1)))
        #     cent = data['coords'][np.argmin(dists)]
        #     centroids.append(cent)
        #     proximity[mask, 4] = dists.copy()
        # return proximity, np.asarray(centroids)

    def __repr__(self):

        s = []
        for key, df in self._data.items():
            s.append('-' * 20 + str(key) + '-' * 20)
            s.append(str(df.head()))
        return '\n'.join(s)
