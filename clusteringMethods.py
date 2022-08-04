from sklearn import cluster
from DataManipulation import AdjustedVectorization
import numpy as np
import pandas as pd
import math
from collections import Counter
from sklearn.metrics import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import normalized_mutual_info_score

def manageClusterSize(setting, min_cluster_size, min_samples, cluster_selection_method):
    '''Adjust how conservative the clustering can be based on the setting'''
    if(setting < 1 or setting > 5):
        return min_cluster_size, min_samples, cluster_selection_method
    min_cluster_size = 2
    if(setting == 1):
        min_samples = 1
        cluster_selection_method = 'leaf'
    if(setting == 2):
        min_samples = 2
        cluster_selection_method = 'leaf'
    if(setting == 3):
        min_samples = 5
        cluster_selection_method = 'eom'
    if(setting == 4):
        min_samples = 10
        cluster_selection_method = 'eom'
    if(setting == 5):
        min_samples = 20
        cluster_selection_method = 'eom'
    return min_cluster_size, min_samples, cluster_selection_method

def clusterData(reducedData, cluster_method = 'hdbscan', setting = 0):
    '''Helper function to cluster reduced data with chosen method and conservativeness setting'''
    cluster_method = ClusteringMethods(reducedData, cluster_model= cluster_method, n_clusters=2, setting = setting)
    pred_labels = cluster_method.pred_labels
    #Make the outliers different negative numbers
    negative = -1
    for x in range(len(pred_labels)):
        if(pred_labels[x] == -1):
            pred_labels[x] = negative
            negative = negative - 1
    cluster_method.pred_labels = pred_labels
    return cluster_method

def getPredicitonLabels(cluster):
    """Helper funciton to return predicted labels given a ClusteringMethods object"""
    return cluster.pred_labels

def getScores(labels, clusterPredictions):
    """Helper funciton to return evaluation metrics if the correct labels are given"""
    homogenity = homogeneity_score(labels, clusterPredictions)
    complete = completeness_score(labels, clusterPredictions)
    nmi = normalized_mutual_info_score(labels, clusterPredictions)
    return homogenity, complete, nmi

class ClusteringMethods:
    """ Clusters unlabeled data.
        Parameters
            df(pd.DataFrame): vectorized dataframe with token columns
            cluster_model(str): common clustering models (see model_dict)
            n_clusters(int): number of clusters
            setting(int): Integer where 0 is default and a range of 1 - 5 can adjust how conservative the clustering is
                          where 1 will give more smaller groups and 5 will give less larger groups
            min_cluster_size(int): HDBSCAN parameter giving the smallest number of points in a group
            min_samples(int): Affects the conservativeness of clustering with higher being more conservative
            cluster_selection_method(int): Choice of clustering technique from 'eom', excess of mass, or 'leaf', for leaf nodes
        Variables:
            self.pred_labels(list[int]): Predictions of the cluster group of text in order of df
            self.clusterer(variabel): Clustering model used to make predicition

        pred_labels_list :type list[str]: predicted labels in same order as rows of df  """
    def __init__(self, df, cluster_model='agglomerative', n_clusters=2, min_cluster_size = 2, min_samples = 2, cluster_selection_method = 'eom', setting = 0):
        from sklearn.mixture import GaussianMixture
        from sklearn.cluster import Birch, SpectralClustering, AgglomerativeClustering
        from sklearn.cluster import AffinityPropagation, KMeans, DBSCAN
        import hdbscan
        self.n_clusters = n_clusters
        self.min_cluster_size, self.min_samples, self.cluster_selection_method = manageClusterSize(setting, min_cluster_size, min_samples, cluster_selection_method)
        self.model_dict = {
            'birch': Birch(n_clusters=n_clusters, threshold=0.1),
            'gaussian': GaussianMixture(n_components=n_clusters),
            'spectral': SpectralClustering(n_components=n_clusters),
            'agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
            'affinity-propagation': AffinityPropagation(damping=0.05),
            'kmeans': KMeans(n_clusters=n_clusters),
            'dbscan': DBSCAN(eps=1.25, min_samples=3),
            'hdbscan': hdbscan.HDBSCAN(min_cluster_size = self.min_cluster_size, min_samples = self.min_samples, cluster_selection_method = self.cluster_selection_method, gen_min_span_tree=True)
        }
        if cluster_model == 'fuzzy-cmeans':
            self.pred_labels = self.__fuzzy_cmeans(df)
        try:
            self.clusterer = self.model_dict[cluster_model]
            self.pred_labels = self.clusterer.fit_predict(df)
        except KeyError:
            raise KeyError('Cluster model not recognized. Must be in {birch, gaussian, spectral, agglomerative, '
                           'affinity-propagation, kmeans, dbscan, hdbscan, fuzzy-cmeans}')

    def __fuzzy_init_mat(self, df: pd.DataFrame) -> np.array: 
        pred_labels = self.model_dict['agglomerative'].fit_predict(df)
        init_df = pd.DataFrame(0, index=range(df.shape[0]), columns=range(self.n_clusters))
        for lab_idx, pred_lab in enumerate(pred_labels):
            init_df.at[lab_idx, pred_lab] = 1  
        return init_df.to_numpy().T

    def __fuzzy_cmeans(self, df: pd.DataFrame, margin=(.00, .95)) -> list[str]:
        """Implement fuzzy cmeans clustering"""
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








