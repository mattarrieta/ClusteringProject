from DataManipulation import AdjustedVectorization
from clusteringMethods import ClusteringMethods
from numClusters import NumClusters
from collections import Counter
import numpy as np

class DetermineOutliers:
    """ Removes any 'obvious' outliers from a data set where 'obvious' is determined by the given outlier_method.
        :param data :type list[str]: list of text documents
        :keyword labels :type Optional list[str]: list of corresponding document names to data text documents
     """
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels if labels is not None else []

    def __get_outlier_method(self, outlier_method):
        if outlier_method == 'dbscan':
            return self.dbscan()
        elif outlier_method == 'hdbscan':
            return self.hdbscan()
        elif outlier_method == 'small-clusters':
            return self.small_clusters()
        else:
            raise 'KeyError: Invalid outlier_method. Must be in {dbscan, hdbscan, small-clusters}.'

    def dbscan(self):
        base_df = AdjustedVectorization(self.data).tfidf()
        pred_labels = ClusteringMethods(base_df, cluster_model='dbscan').pred_labels
        outlier_indices = [idx for idx, num in enumerate(pred_labels) if num == -1]
        return outlier_indices

    def hdbscan(self):
        base_df = AdjustedVectorization(self.data).tfidf()
        from DataManipulation import DimensionReduction
        reduced_df = DimensionReduction(base_df, reduction_model='pca-standard', dim=20).reduced_df

        pred_labels = ClusteringMethods(reduced_df, cluster_model='hdbscan').pred_labels
        outlier_indices = [idx for idx, num in enumerate(pred_labels) if num == -1]
        return outlier_indices

    def small_clusters(self, min_clust_size=1):
        """ Labels all data points that are sorted into a cluster smaller than min_clust_size as an outlier.  """
        vectorizer = AdjustedVectorization(self.data)  # vectorize data without outliers
        base_df = AdjustedVectorization(self.data).tfidf()
        df = vectorizer.tfidf_weighted(min_df=5, max_df=.50, max_features=base_df.shape[0] * 10)
        bounds = NumClusters(base_df).suggested_bounds
        num_clust = NumClusters(df).silhouette(bounds=bounds, print_progress=True, margin=.10)
        pred_labels = ClusteringMethods(df, cluster_model='agglomerative', n_clusters=num_clust).pred_labels

        counter, clust_sizes = Counter(pred_labels), range(1, min_clust_size + 1)
        outlier_indices = [idx for idx, pred_label in enumerate(pred_labels) if counter[pred_label] in clust_sizes]
        return outlier_indices

    def remove_outliers(self, outlier_method='hdbscan'):
        outlier_indices = self.__get_outlier_method(outlier_method)
        labels_copy, data_copy = self.labels.copy(), self.data.copy()
        for val in reversed(outlier_indices):
            del data_copy[val]
            if len(labels_copy) != 0:
                del labels_copy[val]
        return (labels_copy, data_copy), outlier_indices

    @classmethod
    def reinsert_outliers(cls, pred_labels, outlier_indices):
        outlier_nums = [num for num in range(-1, -len(outlier_indices)-1, -1)]
        for idx, num in zip(outlier_indices, outlier_nums):
            pred_labels = np.insert(pred_labels, idx, num)
        return pred_labels