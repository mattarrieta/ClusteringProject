import pandas as pd
import numpy as np

class GroupLabeledTextData:
    """ Group all documents with the same name together.
        :param labeled_data :type tuple[list, list]: zipped list of text documents and their document name
        :keyword subset_labels :type Optional tuple[str]: tuple of document names that are used to group documents.
            If not given, uses all labels given in labeled_data.
        separated_data :type list[(label, list[text])]: list of grouped documents and their shared document name"""
    def __init__(self, labeled_data, subset_labels=None):
        self.labels, self.data = labeled_data
        self.subset_labels = subset_labels if subset_labels is not None else tuple(set(self.labels))
        self.separated_data = self.__separate_data()

    def __separate_data(self):
        separated_data = [(label, []) for label in self.subset_labels]
        for label, text in zip(self.labels, self.data):
            if label in self.subset_labels:
                idx = self.subset_labels.index(label)
                separated_data[idx][1].append(text)
        return separated_data


class GroupLabeledDataFrame:
    """ Useful information about collection of row vectors with same index label.
        :param df: pd.DataFrame with token columns and labeled index
        :return clusters: pandas.DataFrame collection of all row vectors with same index label
        :return centroids: center of mass of cluster
        :return cluster_balls: list of (radius, centroid, cluster, cluster_name) for each label
                radius: min(largest distance from centroid, 1.5*IQR of distances from centroid)  """
    def __init__(self, df: pd.DataFrame):
        self.df, unique_labels = df, set(df.index)
        self.centroids = []
        self.clusters = [(self.df.loc[self.df.index == label], label) for label in unique_labels]

    def get_cluster_balls(self, cluster: pd.DataFrame, cluster_name: str) -> (float, pd.DataFrame, pd.DataFrame, str):
        iqr_vec = 1.5 * (cluster.quantile(0.75) - cluster.quantile(0.25))  # 1.5 * IQR
        max_dist = np.linalg.norm(iqr_vec)
        centroid = cluster.mean()
        self.centroids.append(centroid)
        distances = [np.linalg.norm(cluster.iloc[i, :] - centroid) for i, _ in enumerate(cluster.index.values)]
        radius = min(max_dist, max(distances))
        return radius, centroid, cluster, cluster_name
