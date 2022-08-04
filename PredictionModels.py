from sklearn import cluster
from DataManipulation import AdjustedVectorization, GroupLabeledTextData
import numpy as np
import pandas as pd
import math
from collections import Counter


class ClusteringMethods:
    """ Clusters unlabeled data.
        :param df :type pd.DataFrame: vectorized dataframe with token columns
        :param cluster_model :type str: common clustering models (see model_dict)
        :param n_clusters :type int: number of clusters
        pred_labels_list :type list[str]: predicted labels in same order as rows of df  """
    def __init__(self, df, cluster_model='agglomerative', n_clusters=2):
        from sklearn.mixture import GaussianMixture
        from sklearn.cluster import Birch, SpectralClustering, AgglomerativeClustering
        from sklearn.cluster import AffinityPropagation, KMeans, DBSCAN
        import hdbscan
        self.n_clusters = n_clusters
        self.model_dict = {
            'birch': Birch(n_clusters=n_clusters, threshold=0.1),
            'gaussian': GaussianMixture(n_components=n_clusters),
            'spectral': SpectralClustering(n_components=n_clusters),
            'agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
            'affinity-propagation': AffinityPropagation(damping=0.05),
            'kmeans': KMeans(n_clusters=n_clusters),
            'dbscan': DBSCAN(eps=1.25, min_samples=3),  # TODO: make formula for eps
            'hdbscan': hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2)
        }
        if cluster_model == 'fuzzy-cmeans':
            self.pred_labels = self.__fuzzy_cmeans(df)
        try:
            self.clusterer = self.model_dict[cluster_model]
            self.pred_labels = self.clusterer.fit_predict(df)
        except KeyError:
            raise KeyError('Cluster model not recognized. Must be in {birch, gaussian, spectral, agglomerative, '
                           'affinity-propagation, kmeans, dbscan, hdbscan, fuzzy-cmeans}')



    def __fuzzy_init_mat(self, df: pd.DataFrame) -> np.array:  # TODO: include method to change cluster_model
        # from DataManipulation import DimensionReduction
        # reduced_df = DimensionReduction(df, method='pca-standard', dim=2).reduced_df
        pred_labels = self.model_dict['agglomerative'].fit_predict(df)
        init_df = pd.DataFrame(0, index=range(df.shape[0]), columns=range(self.n_clusters))
        for lab_idx, pred_lab in enumerate(pred_labels):
            init_df.at[lab_idx, pred_lab] = 1  # TODO: change .at method -- current method is slow
        return init_df.to_numpy().T

    def __fuzzy_cmeans(self, df: pd.DataFrame, margin=(.00, .95)) -> list[str]:
        import skfuzzy as fuzz
        data = df.to_numpy().T
        model = fuzz.cluster.cmeans(data, c=self.n_clusters, m=2, error=0.01,
                                    maxiter=10, init=self.__fuzzy_init_mat(df))
        self.clusterer = model
        partition_matrix = model[1].T
        pred_labels, outlier_num = [], 1
        for idx, row in enumerate(partition_matrix):
            if margin[0] < np.max(row) < margin[1]:
                pred_label = np.where(row == np.max(row))[0][0]
            else:
                pred_label = -outlier_num
                outlier_num += 1
            pred_labels.append(pred_label)
        return pred_labels








